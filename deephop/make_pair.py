# --*-- coding: utf-8 --*--

import multiprocessing
import os
from typing import List
import pandas as pd
from pandas import DataFrame
from rdkit import DataStructs, Chem
from rdkit.Chem import AllChem, rdMolAlign, Mol


from calc_SC_RDKit import calc_SC_RDKit_score
from data_loader import load_data_frame


def calc_2D_similarity(s1: str, s2: str) -> float:
    mol1 = Chem.MolFromSmiles(s1)
    mol2 = Chem.MolFromSmiles(s2)
    if mol1 is None or mol2 is None:
        return -1.0
    try:
        f1 = AllChem.GetMorganFingerprint(mol1, 3)
        f2 = AllChem.GetMorganFingerprint(mol2, 3)
        return DataStructs.TanimotoSimilarity(f1, f2)
    except Exception:
        return -1.0


def calc_3d_similarity(ref: str, gen: str) -> float:
    ref_mol = Chem.MolFromSmiles(ref)
    gen_mol = Chem.MolFromSmiles(gen)
    if ref_mol is None or gen_mol is None:
        return -1.0
    try:
        ref_mol = Chem.AddHs(ref_mol)
        Chem.AllChem.EmbedMolecule(ref_mol)
        Chem.AllChem.UFFOptimizeMolecule(ref_mol)

        gen_mol = Chem.AddHs(gen_mol)
        Chem.AllChem.EmbedMolecule(gen_mol)
        Chem.AllChem.UFFOptimizeMolecule(gen_mol)

        pyO3A = rdMolAlign.GetO3A(gen_mol, ref_mol).Align()
        return calc_SC_RDKit_score(gen_mol, ref_mol)
    except Exception:
        return -1.0


def prepare_conformer(smiles: str):
    try:
        ref_mol = Chem.MolFromSmiles(smiles)
        ref_mol = Chem.AddHs(ref_mol)
        Chem.AllChem.EmbedMolecule(ref_mol)
        Chem.AllChem.UFFOptimizeMolecule(ref_mol)
    except Exception:
        return None
    return ref_mol


def calc_3d_score(ref_mol: Mol, gen_mol: Mol):
    try:
        pyO3A = rdMolAlign.GetO3A(gen_mol, ref_mol).Align()
        return calc_SC_RDKit_score(gen_mol, ref_mol)
    except:
        return -1.0


def get_use_cpu_count():
    cpu_num = multiprocessing.cpu_count()
    use_cpu_num = max(int(cpu_num * 0.9), 1)
    return use_cpu_num


class MapFunc(object):
    def __init__(self, score_func, smiles_list):
        self.score_func = score_func
        self.smiles_list = smiles_list

    def __call__(self, args):
        ref = args[0]
        dest = args[1]
        return self.score_func(self.smiles_list[ref], self.smiles_list[dest])


def parallel_run(pair_list, func):
    pool = multiprocessing.Pool()
    results = pool.map(func, pair_list)

    pool.close()
    pool.join()

    return results


def make_pair(df, smilarity_2d_threshold: float = 0.6, smilarity_3d_threshold: float = 0.6,
              delta_pvalue: float = 1.0,
              split_ratio: List[float] = [0.8, 0, 1, 0.1]) -> List[DataFrame]:
    data = []
    smiles_list = list(df['canonical_smiles'])
    total = len(smiles_list)
    # loc = df.columns.get_loc("pchembl_value")
    p_value_list = list(df['pchembl_value'])
    step_1_pairs = []
    # first making pair，then splitting
    for j, ref in enumerate(smiles_list):
        for k in range(j + 1, total):
            # target = smiles_list[k]
            delta_p = p_value_list[k] - p_value_list[j]
            if abs(delta_p) < 1.0:
                # delta_p is too littile, skip this pair
                continue
            if delta_p > 0:
                step_1_pairs.append([j, k, delta_p])
            else:
                step_1_pairs.append([k, j, delta_p])

    func = MapFunc(calc_2D_similarity, smiles_list)
    score_2d_list = parallel_run(step_1_pairs, func)
    step_2_pairs = []
    for i, score_2d in enumerate(score_2d_list):
        if smilarity_2d_threshold > score_2d > 0:
            new_list = step_1_pairs[i]
            new_list.append(score_2d)
            step_2_pairs.append(new_list)

    prepare_conformer_list = parallel_run(smiles_list, prepare_conformer)
    func_3d = MapFunc(calc_3d_score, prepare_conformer_list)
    score_3d_list = parallel_run(step_2_pairs, func_3d)
    for i, score_3d in enumerate(score_3d_list):
        if score_3d > smilarity_3d_threshold:
            ref, dest, p_value, score_2d = step_2_pairs[i]
            data.append([smiles_list[ref], smiles_list[dest], abs(p_value), score_2d, score_3d])

    return [pd.DataFrame(data=data, columns=['ref_smiles', 'target_smiles', 'delta_p', 'score_2d', 'score_3d', ])]


def get_data():
    df = load_data_frame('/home/aht/paper_code/shaungjia/chembl_webresource_client/scaffold_hopping_320target.csv')
    core_columns = ["canonical_smiles", "pchembl_value", "target_chembl_id"]
    df = df[df['canonical_smiles'].notnull() & df['pchembl_value'].notnull() & df['target_chembl_id'].notnull()]
    df = df[(df['canonical_smiles'].str.len() > 0) & (df['target_chembl_id'].str.len() > 0)]
    df = df.drop_duplicates(core_columns, keep='first')
    return df


if __name__ == '__main__':
    df = get_data()
    df = df.drop_duplicates(["canonical_smiles", "target_chembl_id"], keep='first')
    groups = df.groupby(['target_chembl_id'])
    data_list = []
    df_dicts = {k: v for k, v in groups}
    targets = [target_chembl_id for target_chembl_id, _ in df_dicts.items()]
    targets.sort(key=lambda k: len(df_dicts[k]), reverse=False)
    for target_chembl_id in targets:
        sub_df = df_dicts[target_chembl_id]
        tmp_save_file = f'tmp/{target_chembl_id}.csv'
        if os.path.isfile(tmp_save_file):
            continue
        # 300 is threshold of the number of molecules
        if len(sub_df) > 300:
            print(f"process traget {target_chembl_id}, size: {len(sub_df)}")
            # pd_train, pd_val, pd_test = make_pair(sub_df)
            splitted_data = make_pair(sub_df)
            for data in splitted_data:
                data['target_chembl_id'] = target_chembl_id
            data_list.append(splitted_data)
            assert len(splitted_data) == 1
            splitted_data[0].to_csv(f'tmp/{target_chembl_id}.csv', index=False)

    for i in range(len(data_list[0])):
        out_csv = f'/home/aht/paper_code/shaungjia/chembl_webresource_client/prepared_data/{i}.csv'
        result = pd.concat([v[i] for v in data_list])
        result.to_csv(out_csv, index=False)
