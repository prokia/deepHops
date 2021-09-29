# --*-- coding: utf-8 --*--

from argparse import ArgumentParser

import deepchem as dc
from numpy import mean
from rdkit import Chem
from tensorflow_core.python.keras import backend

from util import find_best_model_checkpoint
from train import build_model, TASKS
import pandas as pd
import io
import tensorflow as tf
import os, time


def eval_one_mol(smiles, model_save_dir_root, gpu):
    # smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    featurizer = dc.feat.CircularFingerprint(size=1024)
    loader = dc.data.CSVLoader(tasks=[], smiles_field='smiles', featurizer=featurizer)

    test_file = io.StringIO(f"smiles\n{smiles}\n")
    test_dataset = loader.create_dataset(test_file, shard_size=8096)
    device = f"/gpu:{gpu}"
    pred_list = []
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1  # 程序最多只能占用指定gpu50%的显存
    config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
    sess = tf.compat.v1.Session(config=config)
    backend.set_session(sess)

    for m in ['seed0', 'seed1', 'seed2']:
        model_save_dir = f"{model_save_dir_root}/{m}"

        checkpoint = find_best_model_checkpoint(model_save_dir)
        with tf.device(device):
            model = build_model(model_save_dir)
            model.restore(checkpoint)
            model.batch_size = 2
            # model.to(device)
            result = model.predict(test_dataset)
            if result.shape[-1] == 1:
                result = result.squeeze(-1)
            pred_list.append(result[0][55])
    return mean(pred_list)


def eval_test(model_save_dir_root, test_file, pred_path, gpu):
    featurizer = dc.feat.CircularFingerprint(size=1024)
    loader = dc.data.CSVLoader(tasks=[], smiles_field='smiles', featurizer=featurizer)
    test_dataset = loader.create_dataset(test_file, shard_size=8096)
    device = f"cuda:{gpu}"
    pred_list = []
    for m in ['seed0', 'seed1', 'seed2']:
        model_save_dir = f"{model_save_dir_root}/{m}"
        model = build_model(model_save_dir)
        checkpoint = find_best_model_checkpoint(model_save_dir)
        model.restore(checkpoint)
        # model.to(device)
        result = model.predict(test_dataset)
        if result.shape[-1] == 1:
            pred = pd.DataFrame(result.squeeze(-1))
        else:
            pred = pd.DataFrame(result)
        pred['smiles'] = test_dataset.ids.tolist()
        # dummmy column for next pipeline
        pred['label'] = 0
        pred_list.append(pred)
    pred = pd.concat(pred_list)
    score_colnames = [c for c in pred.columns]
    score_colnames.remove('smiles')
    score_colnames.remove('label')
    # average the values from 3 models as the score
    pred.groupby(['smiles'], sort=False)[score_colnames].apply(lambda x: mean(x))
    # pred = pred.reset_index()
    cols = ['smiles', 'label']
    cols.extend(score_colnames)
    pred.to_csv(pred_path, index=False, columns=cols)


def run_pipe_server(model_save_dir_root):
    read_path = "/tmp/pipe_scorer.in"
    write_path = "/tmp/pipe_scorer.out"

    if os.path.exists(read_path):
        os.remove(read_path)
    if os.path.exists(write_path):
        os.remove(write_path)

    os.mkfifo(write_path)
    os.mkfifo(read_path)

    wf = os.open(write_path, os.O_SYNC | os.O_CREAT | os.O_RDWR)
    rf = os.open(read_path, os.O_RDONLY)
    featurizer = dc.feat.CircularFingerprint(size=1024)
    loader = dc.data.CSVLoader(tasks=[], smiles_field='smiles', featurizer=featurizer)

    gpu = 1
    device = f"/gpu:{gpu}"

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1  # 程序最多只能占用指定gpu50%的显存
    config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
    sess = tf.compat.v1.Session(config=config)
    backend.set_session(sess)
    model_list = []
    for m in ['seed0', 'seed1', 'seed2']:
        model_save_dir = f"{model_save_dir_root}/{m}"

        checkpoint = find_best_model_checkpoint(model_save_dir)
        with tf.device(device):
            model = build_model(model_save_dir)
            model.restore(checkpoint)
            model.batch_size = 2
        model_list.append(model)

    while True:
        s = os.read(rf, 1024)
        # # cur_command.write(s)
        if len(s) == 0:
            time.sleep(1e-3)
            continue
        smiles = s.decode()
        if "exit" in smiles:
            os.close(rf)
            break
        else:
            print(f"SMILES: {s}")
            # smiles = s.decode()
            test_file = io.StringIO(f"smiles\n{smiles}\n")
            test_dataset = loader.create_dataset(test_file, shard_size=8096)
            pred_list = []
            for model in model_list:
                result = model.predict(test_dataset)
                if result.shape[-1] == 1:
                    result = result.squeeze(-1)
                pred_list.append(result[0][55])
            score = mean(pred_list)
        os.write(wf, f"{score:.6f}".encode())
        print(f"result: {score:.6f}")
    os.close(rf)
    os.close(wf)


if __name__ == '__main__':
    # v = eval_one_mol(Chem.MolFromSmiles('COc1cc(C=Nn2c(-c3ccccc3)nc3ccccc3c2=O)cc(OC)c1OCC(=O)Nc1ccccc1F'),
    #                  '/home/aht/paper_code/shaungjia/code/score/model/total_mtr', 1)
    # print(v)
    parser = ArgumentParser(conflict_handler='resolve', description='Configure')
    parser.add_argument('--test_path', type=str,
                        default='/home/aht/paper_code/shaungjia/code/score/model/do_chemprop/data/total_mtr',
                        help='the directory of test data')
    parser.add_argument('--preds_path', type=str,
                        default='/home/aht/paper_code/shaungjia/code/score/model/total_mtr',
                        help='output directory')
    parser.add_argument('--model_save_dir', type=str, help='the model saved directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU rank')
    parser.add_argument('--scorer_pipe_server', action='store_true', default=False, help='accept the request of clients by named pipe')

    args = parser.parse_args()
    if args.scorer_pipe_server:
        run_pipe_server(args.model_save_dir)
    else:
        eval_test(args.model_save_dir, args.test_path, args.preds_path, args.gpu)
