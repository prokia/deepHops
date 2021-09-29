# --*-- coding: utf-8 --*--

from __future__ import division, unicode_literals
import argparse
import numpy as np
import subprocess

from rdkit import Chem
import pandas as pd

import data_loader
import onmt.opts
from make_pair import calc_2D_similarity, calc_3d_similarity, parallel_run
from split_data import TASKS


def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
        return ''


def get_rank(row, base, max_rank):
    for i in range(1, max_rank + 1):
        if row['target'] == row['{}{}'.format(base, i)]:
            return i
    return 0


def main(opt):
    with open(opt.src, 'r') as f:
        targets = [''.join(line.strip().split(' ')) for line in f.readlines()]

    with open(opt.cond, 'r') as f:
        protein_chembl_names = [TASKS[int(line.strip())] for line in f.readlines()]

    predictions = [[] for i in range(opt.beam_size)]
    out_file_no_ext = get_out_file_prefix(opt.score_file, 'gen')
    gen_file = f"{out_file_no_ext}.csv"
    if os.path.isfile(gen_file):
        print(f"Gen file {gen_file} exist, load...")
        test_df = data_loader.load_data_frame(gen_file)
    else:
        test_df = prepare_gen_file(opt, predictions, protein_chembl_names, targets)
        test_df.to_csv(gen_file, index=False)

    proteins = list(set(protein_chembl_names))
    final_summary = []
    for protein_name in proteins:
        result_of_one_protein = pvalue_summary(opt, test_df[test_df.protein_name == protein_name], protein_name)
        final_summary.append(result_of_one_protein)
    tmp_dict = {k: [] for k in final_summary[0].keys()}
    for k in tmp_dict.keys():
        tmp_dict[k].extend([r[k] for r in final_summary])
    tmp_dict['protein'] = proteins
    out_file_no_ext = get_out_file_prefix(opt.score_file, 'final')
    pd.DataFrame(tmp_dict).to_csv(f"{out_file_no_ext}.csv", index=False)


def prepare_gen_file(opt, predictions, protein_chembl_names, targets):
    test_df = pd.DataFrame({'target': targets, 'protein_name': protein_chembl_names})
    total = len(test_df)
    with open(opt.predictions, 'r') as f:
        for i, line in enumerate(f.readlines()):
            predictions[i % opt.beam_size].append(''.join(line.strip().split(' ')))
    for i, preds in enumerate(predictions):
        test_df['prediction_{}'.format(i + 1)] = preds
        test_df['canonical_prediction_{}'.format(i + 1)] = test_df['prediction_{}'.format(i + 1)].apply(
            lambda x: canonicalize_smiles(x))
    test_df['rank'] = test_df.apply(lambda row: get_rank(row, 'canonical_prediction_', opt.beam_size), axis=1)
    correct = 0
    for i in range(1, opt.beam_size + 1):
        correct += (test_df['rank'] == i).sum()
        invalid_smiles = (test_df['canonical_prediction_{}'.format(i)] == '').sum()
        if opt.invalid_smiles:
            print('Top-{}: {:.1f}% || Invalid SMILES {:.2f}%'.format(i, correct / total * 100,
                                                                     invalid_smiles / total * 100))
        else:
            print('Top-{}: {:.1f}%'.format(i, correct / total * 100))
    return test_df


import os


class MapFunc(object):
    def __init__(self, score_func):
        self.score_func = score_func

    def __call__(self, args):
        ref = args[0]
        dest = args[1]
        return self.score_func(ref, dest)


def pvalue_summary(opt, test_df, protein_chembl_name):
    out_file_no_ext = get_out_file_prefix(opt.score_file, protein_chembl_name)
    src_list = []
    hopped_list = []
    rank_list = []
    df = data_loader.load_data_frame(f"{opt.pvalue_dir}/{protein_chembl_name}.csv")
    # df_smiles_pvalue = df[df.target_chembl_id == protein_chembl_name][['canonical_smiles', 'pchembl_value']]
    df_smiles_pvalue = df.drop_duplicates(['canonical_smiles'], keep='first', inplace=False)
    test_df = test_df.drop_duplicates(['target'], keep='first', inplace=False)
    for i, row in test_df.iterrows():
        for j in range(1, opt.generate_num):
            if pd.notna(row[f"canonical_prediction_{j}"]) and len(row[f"canonical_prediction_{j}"]) > 0:
                hopped_list.append(row[f"canonical_prediction_{j}"])
                src_list.append(row['target'])
                rank_list.append(j)
    pred_df = pred_score_mtdnn(hopped_list, opt, out_file_no_ext)

    hopping_file = f"{out_file_no_ext}_hopping.csv"
    if not os.path.isfile(hopping_file):
        df_result = pd.DataFrame({"hopped": hopped_list, "ref": src_list, "rank_tag": rank_list})
        df_result = pd.merge(df_result, df_smiles_pvalue, left_on='ref', right_on='canonical_smiles',
                             how='left', suffixes=('', ''))
        del df_result["canonical_smiles"]
        tmp_df = pred_df[['smiles', protein_chembl_name]]
        tmp_df = tmp_df.drop_duplicates(['smiles'], keep='first', inplace=False)
        del pred_df
        df_result = pd.merge(df_result, tmp_df, left_on='hopped', right_on='smiles',
                             how='left', suffixes=('', ''))
        del df_result["smiles"]
        df_result.to_csv(hopping_file, index=False)
    else:
        df_result = data_loader.load_data_frame(hopping_file)

    final_result_file = f"{out_file_no_ext}_2d_3d.csv"
    if not os.path.isfile(final_result_file):
        pair_list = [(row['ref'], row['hopped']) for _, row in df_result.iterrows()]
        score_2d_list = parallel_run(pair_list, MapFunc(calc_2D_similarity))
        score_3d_list = parallel_run(pair_list, MapFunc(calc_3d_similarity))
        df_result['score_2d'] = score_2d_list
        df_result['score_3d'] = score_3d_list
        df_result.to_csv(final_result_file, index=False)
    return calc_success_rate(opt, final_result_file, protein_chembl_name)


def pred_score(hopped_list, opt, out_file_no_ext):
    tmp_mid_score = f'{out_file_no_ext}_s.csv'
    if not os.path.isfile(tmp_mid_score):
        df_to_predict = pd.DataFrame({"smiles": hopped_list})
        mid_pred_file = f'{out_file_no_ext}_tmp_to_pred.csv'
        df_to_predict.to_csv(mid_pred_file, index=False)
        # invoke the scorer by subprocessing
        param = f"--gpu 0 --test_path {mid_pred_file} --preds_path {tmp_mid_score} " \
                f"--checkpoint_dir {opt.scorer_model_dir} " \
                f"--pred_columns_name [hopped_pvalue]"

        command = f"cd {os.path.dirname(opt.scorer_start_py)} && python {opt.scorer_start_py} {param}"

        p = subprocess.Popen(command, shell=True)
        p.wait()
        del df_to_predict
        with open(tmp_mid_score) as reader:
            _ = reader.readline()
            columns = ["smiles", "label"]
            columns.extend(TASKS)
            pred_df = pd.read_csv(reader, delimiter=',', header=None, names=columns)
    return pred_df


def pred_score_mtdnn(hopped_list, opt, out_file_no_ext):
    tmp_mid_score = f'{out_file_no_ext}_s.csv'
    if not os.path.isfile(tmp_mid_score):
        df_to_predict = pd.DataFrame({"smiles": hopped_list})
        mid_pred_file = f'{out_file_no_ext}_tmp_to_pred.csv'
        df_to_predict.to_csv(mid_pred_file, index=False)
        param = f"--test_path {mid_pred_file} --preds_path {tmp_mid_score} " \
                f"--model_save_dir {opt.scorer_model_dir} "

        if os.path.isdir('/home/aht/paper_code/shaungjia/code/score'):
            command = f"cd /home/aht/paper_code/shaungjia/code/score && /home/ubuntu/anaconda3/envs/deepchem/bin/python evaluate.py {param}"
        elif os.path.isdir('/data/u1/projects/score'):
            # another machine
            command = f"cd /data/u1/projects/score && /data/u1/anaconda3/envs/deepchem/bin/python evaluate.py {param}"
        else:
            assert False

        p = subprocess.Popen(command, shell=True)
        p.wait()
        del df_to_predict

    with open(tmp_mid_score) as reader:
        _ = reader.readline()
        columns = ["smiles", "label"]
        columns.extend(TASKS)
        pred_df = pd.read_csv(reader, delimiter=',', header=None, names=columns)
    return pred_df


def get_out_file_prefix(score_file, protein_chembl_name):
    out_file_no_ext = f"{score_file.split('.')[0]}_{protein_chembl_name}"
    return out_file_no_ext


def strict_filter(df, opt):
    targets = []
    data_dir = opt.train_data_dir
    for tag in ['train', 'val', 'test']:
        with open(f'{data_dir}/tgt-{tag}.txt', 'r') as f:
            targets.extend([''.join(line.strip().split(' ')) for line in f.readlines()])
        with open(f'{data_dir}/src-{tag}.txt', 'r') as f:
            targets.extend([''.join(line.strip().split(' ')) for line in f.readlines()])
    if opt.transfer_train_data_dir is not None:
        data_dir = opt.transfer_train_data_dir
        for tag in ['train', 'val', 'test']:
            with open(f'{data_dir}/tgt-{tag}.txt', 'r') as f:
                targets.extend([''.join(line.strip().split(' ')) for line in f.readlines()])
            with open(f'{data_dir}/src-{tag}.txt', 'r') as f:
                targets.extend([''.join(line.strip().split(' ')) for line in f.readlines()])
    target_set = set(targets)
    df = df[~df.hopped.isin(target_set)]
    return df


def calc_success_rate(opt, final_result_file, protein_chembl_name, filter_func=strict_filter):
    result_dict = dict()
    df = data_loader.load_data_frame(final_result_file)
    print(f"before remove repeat: {len(df)}")
    result_dict['total_gen'] = len(df)
    if filter_func is not None:
        df = filter_func(df, opt)
    print(f"after remove repeat: {len(df)}")
    total = len(df)
    result_dict['remove_repeat_with_train'] = total
    mol_dict = {s: [0] * opt.generate_num for s in df['ref']}
    total_mol = len(mol_dict)
    print(f"ref number: {total_mol}")
    result_dict['ref_number'] = total_mol

    for topN in range(1, 11):
        tmp = df[
            (df.score_2d < 0.6) & (df.score_3d > 0.6) & (df[protein_chembl_name] - df.pchembl_value >= 1) & (
                    df.rank_tag <= topN)]
        print(f"{len(tmp)}:{len(tmp) / total:4f}")
        for k, v in mol_dict.items():
            v[topN - 1] = len(tmp[tmp.ref == k])

    print("topN rate:")
    for topN in range(1, 11):
        c = 0
        for k, v in mol_dict.items():
            if v[topN - 1] > 0:
                c += 1
        print(f"{topN}:{c / total_mol:4f}")
        result_dict[f"top_{topN}_success_ref_num"] = c
    sub_df = df[df.rank_tag == 1]
    result_dict['promotion_of_top1'] = np.nanmean(sub_df[protein_chembl_name] - sub_df.pchembl_value)
    return result_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='pvalue_score_predictions.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser)

    parser.add_argument('-beam_size', type=int, default=5,
                        help='Beam size')
    parser.add_argument('-invalid_smiles', action="store_true",
                        help='Show % of invalid SMILES')
    parser.add_argument('-predictions', type=str, default="",
                        help="Path to file containing the predictions")
    parser.add_argument('-src', type=str, default="",
                        help="Path to file containing src molecule")
    parser.add_argument('-tgt', type=str, default="",
                        help="Path to file containing target molecule")
    parser.add_argument('-cond', type=str, default="",
                        help="Path to file containing protein of the src anf target molecule")
    parser.add_argument('-score_file', type=str, default="",
                        help="the mid results of evaluation")
    group = parser.add_argument_group('fulmz_ext')

    group.add_argument('-pvalue_dir', type=str, default='/home/aht/paper_code/shaungjia/score_train_data',
                       help="the directory which saved the pvalue of known molecules per targets")
    group.add_argument('-train_data_dir', type=str, default='/home/aht/paper_code/shaungjia/score_train_data',
                       help="train data directory")
    group.add_argument('-transfer_train_data_dir', type=str, help="the data directory of transfer learning target")
    group.add_argument('-scorer_model_dir', type=str,
                       default='/home/aht/paper_code/shaungjia/code/score/model/do_chemprop/multi_task/model_graph_only/all_mtr',
                       help="the path of model used by scorer")
    group.add_argument('-scorer_start_py', type=str, default='/home/aht/tw/cell_paper/reg_shuangjia/predict.py',
                       help="the entry python file of scorer")
    group.add_argument('-gpu', type=int, default=0, help="GPU rank")
    group.add_argument('-generate_num', type=int, default=10, help="How many are hopped from a molecule")
    opt = parser.parse_args()
    main(opt)
