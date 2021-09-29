# --*-- coding: utf-8 --*--

import argparse
import multiprocessing

from sklearn.utils import shuffle

import data_loader
import pandas as pd
import os
import numpy as np

from calc_scaffold_smilarity import process_one_protein, calc_scaffold_similarity
TASKS = ['CHEMBL1075104', 'CHEMBL1075126', 'CHEMBL1075167', 'CHEMBL1163101', 'CHEMBL1293228', 'CHEMBL1741200',
         'CHEMBL1824', 'CHEMBL1841', 'CHEMBL1868', 'CHEMBL1906', 'CHEMBL1907600', 'CHEMBL1907601', 'CHEMBL1907602',
         'CHEMBL1907605', 'CHEMBL1908389', 'CHEMBL1936', 'CHEMBL1955', 'CHEMBL1974', 'CHEMBL1991', 'CHEMBL2041',
         'CHEMBL2068', 'CHEMBL2094115', 'CHEMBL2094126', 'CHEMBL2094127', 'CHEMBL2094128', 'CHEMBL2095188',
         'CHEMBL2095191', 'CHEMBL2096618', 'CHEMBL2111367', 'CHEMBL2111389', 'CHEMBL2147', 'CHEMBL2148', 'CHEMBL2185',
         'CHEMBL2207', 'CHEMBL2208', 'CHEMBL2276', 'CHEMBL2292', 'CHEMBL2345', 'CHEMBL2363049', 'CHEMBL2431',
         'CHEMBL2468', 'CHEMBL2527', 'CHEMBL2534', 'CHEMBL258', 'CHEMBL2595', 'CHEMBL2598', 'CHEMBL2599', 'CHEMBL262',
         'CHEMBL2637', 'CHEMBL267', 'CHEMBL2695', 'CHEMBL2793', 'CHEMBL2801', 'CHEMBL2815', 'CHEMBL2828', 'CHEMBL2835',
         'CHEMBL2850', 'CHEMBL2938', 'CHEMBL2959', 'CHEMBL2973', 'CHEMBL299', 'CHEMBL2996', 'CHEMBL301', 'CHEMBL3032',
         'CHEMBL3038469', 'CHEMBL3045', 'CHEMBL3055', 'CHEMBL308', 'CHEMBL3116', 'CHEMBL3130', 'CHEMBL3137261',
         'CHEMBL3142', 'CHEMBL3145', 'CHEMBL3231', 'CHEMBL3234', 'CHEMBL3267', 'CHEMBL3286', 'CHEMBL331', 'CHEMBL3357',
         'CHEMBL3476', 'CHEMBL3529', 'CHEMBL3553', 'CHEMBL3582', 'CHEMBL3587', 'CHEMBL3589', 'CHEMBL3629', 'CHEMBL3650',
         'CHEMBL3717', 'CHEMBL3778', 'CHEMBL3788', 'CHEMBL3797', 'CHEMBL3820', 'CHEMBL3835', 'CHEMBL3836', 'CHEMBL3861',
         'CHEMBL3905', 'CHEMBL3906', 'CHEMBL3920', 'CHEMBL3983', 'CHEMBL4005', 'CHEMBL4036', 'CHEMBL4045', 'CHEMBL4101',
         'CHEMBL4128', 'CHEMBL4179', 'CHEMBL4202', 'CHEMBL4203', 'CHEMBL4204', 'CHEMBL4224', 'CHEMBL4225', 'CHEMBL4247',
         'CHEMBL4282', 'CHEMBL4394', 'CHEMBL4439', 'CHEMBL4482', 'CHEMBL4501', 'CHEMBL4523', 'CHEMBL4525', 'CHEMBL4576',
         'CHEMBL4578', 'CHEMBL4599', 'CHEMBL4600', 'CHEMBL4630', 'CHEMBL4708', 'CHEMBL4722', 'CHEMBL4766', 'CHEMBL4816',
         'CHEMBL4852', 'CHEMBL4895', 'CHEMBL4898', 'CHEMBL4900', 'CHEMBL4937', 'CHEMBL5145', 'CHEMBL5147', 'CHEMBL5149',
         'CHEMBL5251', 'CHEMBL5261', 'CHEMBL5314', 'CHEMBL5330', 'CHEMBL5407', 'CHEMBL5408', 'CHEMBL5443', 'CHEMBL5469',
         'CHEMBL5491', 'CHEMBL5508', 'CHEMBL5518', 'CHEMBL5543', 'CHEMBL5600', 'CHEMBL5608', 'CHEMBL5719', 'CHEMBL5888',
         'CHEMBL6166']

# the 40 proteins in trainning set
include_list = {'CHEMBL4816', 'CHEMBL3130', 'CHEMBL3529', 'CHEMBL4204', 'CHEMBL267', 'CHEMBL4578', 'CHEMBL2534',
                'CHEMBL2148', 'CHEMBL2276', 'CHEMBL2695', 'CHEMBL4005', 'CHEMBL2835', 'CHEMBL3778', 'CHEMBL2637',
                'CHEMBL1906', 'CHEMBL1868', 'CHEMBL4501', 'CHEMBL2431', 'CHEMBL3650', 'CHEMBL5543', 'CHEMBL2111367',
                'CHEMBL2095191', 'CHEMBL3905', 'CHEMBL4203', 'CHEMBL4722', 'CHEMBL4898', 'CHEMBL3861', 'CHEMBL299',
                'CHEMBL2850', 'CHEMBL2828', 'CHEMBL5145', 'CHEMBL3234', 'CHEMBL3582', 'CHEMBL3267', 'CHEMBL3231',
                'CHEMBL3788', 'CHEMBL3145', 'CHEMBL1974', 'CHEMBL258', 'CHEMBL4247'}

# the 6 unseen proteins
external_test_set = ['CHEMBL1907601', 'CHEMBL1907602', 'CHEMBL2094127', 'CHEMBL2147', 'CHEMBL4523', 'CHEMBL5407']

# the 69 useen proteins, including external_test_set
big_external_test_set = ['CHEMBL1907601', 'CHEMBL1907602', 'CHEMBL2094127', 'CHEMBL2147', 'CHEMBL4523', 'CHEMBL5407',
                         'CHEMBL1908389',
                         'CHEMBL5408',
                         'CHEMBL3032',
                         'CHEMBL5719',
                         'CHEMBL262',
                         'CHEMBL2595',
                         'CHEMBL2527',
                         'CHEMBL308',
                         'CHEMBL2068',
                         'CHEMBL4045',
                         'CHEMBL4282',
                         'CHEMBL3587',
                         'CHEMBL2185',
                         'CHEMBL2095188',
                         'CHEMBL5314',
                         'CHEMBL4101',
                         'CHEMBL1991',
                         'CHEMBL3142',
                         'CHEMBL5600',
                         'CHEMBL2345',
                         'CHEMBL3286',
                         'CHEMBL2599',
                         'CHEMBL5469',
                         'CHEMBL4937',
                         'CHEMBL1824',
                         'CHEMBL5147',
                         'CHEMBL5608',
                         'CHEMBL2094126',
                         'CHEMBL3038469',
                         'CHEMBL2959',
                         'CHEMBL3920',
                         'CHEMBL5491',
                         'CHEMBL5518',
                         'CHEMBL4895',
                         'CHEMBL3983',
                         'CHEMBL5888',
                         'CHEMBL1075104',
                         'CHEMBL2292',
                         'CHEMBL2208',
                         'CHEMBL1955',
                         'CHEMBL2938',
                         'CHEMBL3629',
                         'CHEMBL2815',
                         'CHEMBL4179',
                         'CHEMBL4482',
                         'CHEMBL4600',
                         'CHEMBL4630',
                         'CHEMBL301',
                         'CHEMBL1907600',
                         'CHEMBL3045',
                         'CHEMBL2973',
                         'CHEMBL2111389',
                         'CHEMBL4225',
                         'CHEMBL331',
                         'CHEMBL3717',
                         'CHEMBL2041',
                         'CHEMBL1907605',
                         'CHEMBL3055',
                         'CHEMBL4708',
                         'CHEMBL2094128',
                         'CHEMBL2996',
                         'CHEMBL5330',
                         'CHEMBL5251']


def select_limit_ref_num(df, ref_max_occured=5):
    """
    max occured
    Args:
        ref_max_occured:

    Returns:

    """
    uniq_refs = set(df.ref_smiles)
    df_per_ref_list = []
    for ref in uniq_refs:
        cur = df[df.ref_smiles == ref]
        # 随机抽取 ref_max_occured 个pair
        cur = shuffle(cur)
        # cur.sort_values(by=["score_2d", "score_3d"], ascending=[True, False])
        df_per_ref_list.append(cur.head(ref_max_occured))
    return pd.concat(df_per_ref_list)

def select_by_target_uniq_rate(df, target_uniq_rate):
    """
    Args:
        df_train:
        target_uniq_rate:

    Returns:

    """
    uniq_targets = list(set(df.target_smiles))
    df_per_target_list = []
    base_num = int(np.floor(1 / target_uniq_rate))
    prob_threshold = 1/target_uniq_rate - base_num
    print(f"base_num: {base_num}, prob_threshold: {prob_threshold}")
    prob_of_targets = np.random.uniform(low=0.0, high=1.0, size=len(uniq_targets))
    for target, prob in zip(uniq_targets, prob_of_targets.tolist()):
        cur = df[df.target_smiles == target]
        # random select ref_max_occured pairs
        cur = shuffle(cur)
        select_num = base_num
        if prob_threshold > 0 and prob <= prob_threshold:
            select_num += 1
        df_per_target_list.append(cur.head(select_num))
    return pd.concat(df_per_target_list)

def filter_scaffold_smilarity(file):
    df = data_loader.load_data_frame(file)
    df["delta_p"] = df["delta_p"].apply(lambda x: abs(x))
    df["score_scaffold"] = df[["ref_smiles", "target_smiles"]].apply(lambda x: calc_scaffold_similarity(x[0], x[1]), axis=1)
    df = df[df.score_scaffold <= 0.6]
    del df["score_scaffold"]
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='split_data.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-hopping_pairs_dir', type=str,
                        default='/home/aht/paper_code/shaungjia/hopping_pairs',
                        help='the hooping pairs directory generated by make_pairs.py')
    parser.add_argument('-out_dir', type=str,
                        default='/home/aht/paper_code/shaungjia/code/MolecularGET-master/data_hopping',
                        help='the output directory after data splitting')
    parser.add_argument('-protein_group', type=str, choices=['data40', 'test6', 'test69'], help='data40: forty proteins in training set, test6: six unseen proteins, test69: 69 unseen proteins, the R2 of scorer is greater than 0.66')
    parser.add_argument('-proteins', nargs='+', type=str, help='the pairs of proteins to split')
    parser.add_argument('-ratio_list', nargs='+', type=float, default=[0.8, 0.1, 0.1], help='the ratio of train val test')
    parser.add_argument('-ref_max_occured', type=int, default=0, help='the max number of times as source per molecule')
    parser.add_argument('-target_uniq_rate', type=float, default=0.0, help='target_uniq_rate is the uniq rate of target molecules per protein')

    opt = parser.parse_args()
    d = opt.hopping_pairs_dir

    print(f"total task: {len(TASKS)}")
    df_train_list = []
    df_val_list = []
    df_test_list = []
    ratio_list = opt.ratio_list
    filename_list = ['train', 'test', 'val']
    assert len(ratio_list) == len(filename_list)
    result = [[] for _ in range(len(filename_list))]
    if opt.protein_group is None:
        select_list = opt.proteins
    else:
        select_dict = {'data40': include_list, 'test6': external_test_set, 'test69': big_external_test_set}
        select_list = select_dict[opt.protein_group]

    for x in select_list:
        df = data_loader.load_data_frame(f"{d}/{x}.csv")
        df = shuffle(df)
        if opt.ref_max_occured > 0 or opt.target_uniq_rate > 0.0:
            df = df[df.score_scaffold < 0.6]
            uniq_refs = list(set(df.ref_smiles))
            uniq_refs = shuffle(uniq_refs)
            test_size = int(len(uniq_refs) * 0.1)
            print(f"uniq_refs: {len(uniq_refs)}, test size: {test_size}")
            smi_of_test = uniq_refs[:test_size]
            smi_of_train = uniq_refs[test_size:]
            df_train = df[df.ref_smiles.isin(smi_of_train) & df.target_smiles.isin(smi_of_train)]
            df_test = df[df.ref_smiles.isin(smi_of_test) & df.target_smiles.isin(smi_of_test)]
            if opt.ref_max_occured > 0:
                # 从训练集中分割出验证集合
                df = select_limit_ref_num(df_train, opt.ref_max_occured)
            else:
                df = select_by_target_uniq_rate(df_train, opt.target_uniq_rate)
            df1 = df.sample(frac=0.9)
            df2 = df[~df.index.isin(df1.index)]
            result[0].append(df1.reset_index(drop=True))
            result[1].append(df_test)
            result[2].append(df2.reset_index(drop=True))
        else:
            pos = 0
            for i, r in enumerate(ratio_list):
                cur_size = int(len(df) * r)
                if cur_size > 0:
                    tmp_size = min(2000000000, cur_size)
                    cur_df = df[pos:pos + tmp_size]
                    result[i].append(cur_df)
                pos += cur_size


    save_dir = opt.out_dir
    os.makedirs(save_dir, exist_ok=True)


    def write_line(out, line, add_space=True):
        if add_space:
            tmp = [' '] * (len(line) * 2 - 1)
            tmp[0::2] = [s for s in line]
            line = ''.join(tmp)
        out.write(line)
        out.write(os.linesep)


    for i, f in enumerate(filename_list):
        df = pd.concat(result[i], axis=0, ignore_index=True)
        df = shuffle(df)
        with open(f"{save_dir}/src-{f}.txt", 'w') as out:
            _ = [write_line(out, s) for s in df['ref_smiles']]
        with open(f"{save_dir}/tgt-{f}.txt", 'w') as out:
            _ = [write_line(out, s) for s in df['target_smiles']]
        with open(f"{save_dir}/cond-{f}.txt", 'w') as out:
            _ = [write_line(out, f"{TASKS.index(x)}", False) for x in df['target_chembl_id']]
