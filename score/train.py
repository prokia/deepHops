# --*-- coding: utf-8 --*--

import shutil
from argparse import ArgumentParser
from functools import partial

import deepchem as dc
import numpy as np
from deepchem.models.optimizers import LearningRateSchedule
from tensorboardX import SummaryWriter
import os
from sklearn.metrics import jaccard_score, r2_score, mean_squared_error
import pandas as pd

from data_loader import load_data_frame
import tensorflow as tf

# from evaluate import evaluate_predictions
from mtdnn import Mtdnn

BATCH_SIZE = 128

r2_score_of_latest = []

MONITOR_VAL_PER_EPOCH = 5


def save_loss(writer, end_of_epoch, val_dataset, model, n_iter, batch_loss) -> bool:
    print(f"n_iter: {n_iter}")
    # lr = model.optimizer.learning_rate._create_tensor(n_iter).numpy()
    # writer.add_scalar('lr', lr, n_iter)
    writer.add_scalar('loss', batch_loss.numpy(), n_iter)
    is_end, epoch = end_of_epoch(n_iter)
    if is_end:
        result = model.predict(val_dataset)
        val_loss = loss_np(result, val_dataset.y)
        writer.add_scalar('val_loss', val_loss, n_iter)
        if epoch > 0 and epoch % MONITOR_VAL_PER_EPOCH == 0:
            rmse = np.sqrt(val_loss)
            writer.add_scalar('val_rmse', rmse, n_iter)

            global r2_score_of_latest
            r2_score_of_latest.append(rmse)
            if len(r2_score_of_latest) >= 5:
                should_early_stop = all([r2_score_of_latest[-i] >= r2_score_of_latest[-5] for i in range(1, 5)])
                return should_early_stop
    return False


def is_end_of_epoch(train_len, n_iter):
    u = train_len // BATCH_SIZE
    return n_iter > 0 and n_iter % u == 0, n_iter // u


def eval(val_dataset, metrics, model):
    result = model.predict(val_dataset)
    loss = loss_np(result, val_dataset.y)
    r2_score(result, val_dataset.y)
    # 9910 * 152 * 1
    return loss

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
TOTAL_TASK = len(TASKS)


def prepre_data(data_file_dir):
    select_cols = ['pchembl_value', 'canonical_smiles']
    total = dict()

    for task in TASKS:
        f = f'{data_file_dir}/{task}.csv'
        df = load_data_frame(f)
        # task = file.split('.')[0]
        df = df[select_cols]
        for i, v in df.iterrows():
            if v['canonical_smiles'] in total.keys():
                total[v['canonical_smiles']][task] = v['pchembl_value']
            else:
                total[v['canonical_smiles']] = {task: v['pchembl_value']}
        # df[task] = df['pchembl_value']
        # df_list.append(df)
    fr_dict = {'smiles': list(total.keys())}
    for t in TASKS:
        fr_dict[t] = [total[s][t] if t in total[s].keys() else None for s in fr_dict['smiles']]
    df_total = pd.DataFrame(fr_dict)
    df_total.to_csv('CHEMBL1075104.csv', index=False)


def loss(outputs, labels, weights):
    out, label = outputs[0], labels[0]
    out = tf.squeeze(out, -1)
    one = tf.ones_like(label)
    zero = tf.zeros_like(label)
    task_weight = tf.where(tf.less_equal(label, 1e-4), x=zero, y=one)
    squ = tf.square(out - label)
    return tf.reduce_sum(squ * task_weight) / tf.reduce_sum(
        tf.cast(tf.greater_equal(label, 1e-4), tf.float32))


def loss_np(outputs, labels):
    out = np.squeeze(outputs, -1)
    label = labels
    task_weight = np.greater_equal(label, 1e-4).astype(np.float64)
    squ = np.square(out - label)
    return np.sum(squ * task_weight) / np.sum(task_weight)


def train(data_file_dir, save_dir, data_mode, tasks=None):
    global r2_score_of_latest
    r2_score_of_latest = []
    featurizer = dc.feat.CircularFingerprint(size=1024)
    # featurizer = dc.feat.ConvMolFeaturizer()
    # dc.feat.M
    data_save_path = os.path.join(save_dir, 'data_cache')
    if os.path.exists(data_save_path):
        loaded, all_dataset, _ = dc.utils.save.load_dataset_from_disk(
            data_save_path)
        if loaded:
            train_dataset, val_dataset, _ = all_dataset
        else:
            raise RuntimeError("Bad data cache")
    else:
        if tasks is None:
            loader = dc.data.CSVLoader(tasks=TASKS, smiles_field='smiles', featurizer=featurizer)
        else:
            loader = dc.data.CSVLoader(tasks=tasks, smiles_field='smiles', featurizer=featurizer)
        os.makedirs(data_save_path)
        if data_mode == 'splitted':
            train_dataset, val_dataset, test_dataset = [
                loader.create_dataset(os.path.join(data_file_dir, f), shard_size=8096) for f in
                ['train.csv', 'val.csv', 'test.csv']]
            dc.utils.save.save_dataset_to_disk(data_save_path, train_dataset, val_dataset, test_dataset, None)
        else:
            dataset = loader.create_dataset(data_file_dir, shard_size=8096)
            splitter = dc.splits.RandomSplitter()
            # train_dataset, val_dataset, test_dataset = splitter.train_valid_test_split(dataset)
            train_dataset, val_dataset = splitter.train_test_split(dataset, frac_train=0.9)

            dc.utils.save.save_dataset_to_disk(data_save_path, train_dataset, val_dataset, val_dataset,
                                               dc.trans.NormalizationTransformer(transform_y=True, dataset=val_dataset,
                                                                                 move_mean=False))
            shutil.copytree(os.path.join(data_save_path, 'test_dir'), os.path.join(data_save_path, 'valid_dir'))

    ite_per_epoch = len(train_dataset) // BATCH_SIZE

    model = build_model(save_dir, ite_per_epoch)

    # Use R2 classification metric
    metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, task_averager=np.mean)
    writer = SummaryWriter(log_dir=os.path.join(save_dir))
    # writer.add_graph(model.model_instance)
    end_of_epoch = partial(is_end_of_epoch, len(train_dataset))
    callback_after_batch = [partial(save_loss, writer, end_of_epoch, val_dataset)]
    print(f"开始训练 train_dataset : {len(train_dataset)}, val_dataset: {len(val_dataset)}")
    model.fit(train_dataset, nb_epoch=1000, callbacks=callback_after_batch, restore=False, loss=loss,
              checkpoint_interval=ite_per_epoch * MONITOR_VAL_PER_EPOCH)
    writer.close()
    pass


class Lrs(LearningRateSchedule):
    """A learning rate that decreases exponentially with the number of training steps."""

    def __init__(self, initial_rate, decay_rate, decay_steps, staircase=True, min_lr=1e-6):
        """Create an exponentially decaying learning rate.
        The learning rate starts as initial_rate.  Every decay_steps training steps, it is multiplied by decay_rate.
        Parameters
        ----------
        initial_rate: float
          the initial learning rate
        decay_rate: float
          the base of the exponential
        decay_steps: int
          the number of training steps over which the rate decreases by decay_rate
        staircase: bool
          if True, the learning rate decreases by discrete jumps every decay_steps.
          if False, the learning rate decreases smoothly every step
        """
        self.initial_rate = initial_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.staircase = staircase
        self.min_lr = min_lr

    def _create_tensor(self, global_step):
        return tf.maximum(tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.initial_rate,
            decay_rate=self.decay_rate,
            decay_steps=self.decay_steps,
            staircase=self.staircase)(global_step),
                          tf.constant(self.min_lr))


def build_model(save_dir):
    # decay_lr = Lrs(5e-5, 0.7, ite_per_epoch * 2, True)
    model = Mtdnn(TOTAL_TASK, 1024, batch_size=BATCH_SIZE, layer_sizes=[1024, 391],
                  learning_rate=5e-5,
                  dropouts=0.5,
                  optimizer_type='adam',
                  weight_decay_penalty_type="l2",
                  weight_decay_penalty=0.002,
                  model_dir=save_dir,
                  tensorboard=False,
                  tensorboard_log_frequency=1, verbosity='high')

    # model = Mpnn(TOTAL_TASK, batch_size=BATCH_SIZE,
    #              learning_rate=decay_lr,
    #              optimizer_type='adam',
    #              model_dir=save_dir,
    #              tensorboard=False,
    #              mode='regression',
    #              dropout=0.35,
    #              graph_conv_layers=[1600, 1600, 1600, 1600, 1600],
    #              dense_layer_size=1600,
    #              )
    return model


def get_score(model_save_dir, checkpoint):
    save_path = os.path.join(model_save_dir, 'eval.csv')
    result_df = eval_test(model_save_dir, checkpoint)
    result_df.to_csv(save_path, index=False)


def eval_test(model_save_dir, checkpoint, tasks):
    model = build_model(model_save_dir)
    model.restore(checkpoint)
    data_save_path = os.path.join(model_save_dir, 'data_cache')
    loaded, all_dataset, _ = dc.utils.save.load_dataset_from_disk(
        data_save_path)
    if loaded:
        _, _, test_dataset = all_dataset
    result = model.predict(test_dataset)
    y = test_dataset.y
    if result.shape[-1] == 1:
        pred = pd.DataFrame(result.squeeze(-1))
    else:
        pred = pd.DataFrame(result)
    label = pd.DataFrame(y)

    result_df = summary(label, pred, tasks)
    return result_df


def summary(label, pred, tasks):
    count_dict = {"target": [], "r2": [], "total_val": [], "mse": [], "rmse": []}
    # get R2 per targets
    for i, task in enumerate(tasks):
        label_list = label[i].to_list()
        pred_list = pred[i].to_list()
        true_val = [v for v in label_list if v > 1e-4]
        pred_val = [pred_list[i] for i, v in enumerate(label_list) if v > 1e-4]
        r2 = r2_score(true_val, pred_val)

        # count_dict[TASKS[i]] = r2
        count_dict["target"].append(task)
        count_dict["r2"].append(r2)
        count_dict["total_val"].append(len(true_val))
        mse = mean_squared_error(true_val, pred_val)
        count_dict["mse"].append(mse)
        count_dict["rmse"].append(np.sqrt(mse))
    return pd.DataFrame(count_dict)


if __name__ == '__main__':
    parser = ArgumentParser(conflict_handler='resolve', description='Configure')
    # data_file_dir, save_dir
    parser.add_argument('--data_file_dir', type=str,
                        default='/home/aht/paper_code/shaungjia/code/score/model/do_chemprop/data/total_mtr',
                        help='train data directory')
    parser.add_argument('--result_dir', type=str,
                        default='/home/aht/paper_code/shaungjia/code/score/model/total_mtr',
                        help='the directory as output(models, tensorboard event file etc.)')
    parser.add_argument('--data_mode', type=str, default='should_split', choices=['splitted', 'should_split'],
                        help='splitted: data has been splitted, should_split: the data is raw, splitting is required')
    parser.add_argument('--per_task', action='store_true', default=False, help='Does one target trains one model')
    args = parser.parse_args()

    if args.per_task:
        for ds in os.listdir(args.data_file_dir):
            for seed in range(0, 3):
                data_file_dir = os.path.join(args.data_file_dir, f'{ds}/seed{seed}')
                result_dir = os.path.join(args.result_dir, f'{ds}/seed{seed}')
                train(data_file_dir, result_dir, args.data_mode, tasks=[ds[0:-2]])
    else:
        for seed in range(0, 3):
            data_file_dir = os.path.join(args.data_file_dir, f'seed{seed}')
            result_dir = os.path.join(args.result_dir, f'seed{seed}')
            train(data_file_dir, result_dir, args.data_mode)
