# --*-- coding: utf-8 --*--

import multiprocessing

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
import os
from data_loader import load_data_frame

def calc_scaffold_similarity(s1: str, s2: str) -> float:
    mol1 = Chem.MolFromSmiles(s1)
    mol2 = Chem.MolFromSmiles(s2)
    if mol1 is None or mol2 is None:
        return -1.0
    try:
        scafold1 = MurckoScaffold.GetScaffoldForMol(mol1)
        scafold2 = MurckoScaffold.GetScaffoldForMol(mol2)
        f1 = AllChem.GetMorganFingerprint(scafold1, 3)
        f2 = AllChem.GetMorganFingerprint(scafold2, 3)
        return DataStructs.TanimotoSimilarity(f1, f2)
    except Exception:
        return -1.0


def process_one_protein(args):
    file, save_path = args
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    df = load_data_frame(file)
    df["delta_p"] = df["delta_p"].apply(lambda x: abs(x))
    df["score_scaffold"] = df[["ref_smiles", "target_smiles"]].apply(lambda x: calc_scaffold_similarity(x[0], x[1]),
                                                                     axis=1)
    df.to_csv(save_path, index=False)


if __name__ == '__main__':
    arg_list = []
    root_dir = "/data/u1/projects/mget/hopping_pairs"
    out_dir = "/data/u1/projects/mget/hopping_pairs_with_scaffold"
    for f in os.listdir(root_dir):
        arg_list.append((f"{root_dir}/{f}", f"{out_dir}/{f}"))

    pool = multiprocessing.Pool()
    results = pool.map(process_one_protein, arg_list)
    pool.close()
    pool.join()
