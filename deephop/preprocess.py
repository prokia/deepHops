#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Pre-process Data / features files and build vocabulary
"""

import argparse
import glob
import logging
import multiprocessing
# import torch.multiprocessing as multiprocessing
import sys
import gc
import os
import codecs
from functools import partial

import torch
from onmt.utils.logging import init_logger, logger
import onmt.myutils as myutils
import onmt.inputters as inputters
import onmt.opts as opts
import pickle

def check_existing_pt_files(opt):
    """ Checking if there are existing .pt files to avoid tampering """
    # We will use glob.glob() to find sharded {train|valid}.[0-9]*.pt
    # when training, so check to avoid tampering with existing pt files
    # or mixing them up.
    for t in ['train', 'valid', 'vocab']:
        pattern = opt.save_data + '.' + t + '*.pt'
        if glob.glob(pattern):
            sys.stderr.write("Please backup existing pt file: %s, "
                             "to avoid tampering, but now support continue run!\n" % pattern)

            # sys.exit(1)


def parse_args():
    """ Parsing arguments """
    parser = argparse.ArgumentParser(
        description='preprocess.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.add_md_help_argument(parser)
    opts.preprocess_opts(parser)
    parser.add_argument('-parrel_run', action='store_true', default=False,
                       help='生成靶标')
    parser.add_argument('-with_3d_confomer', action='store_true', default=False,
                       help='原子特征是否在最后3个维度加上坐标')
    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    check_existing_pt_files(opt)

    return opt


def parrel_func(example, have_3d):
    # myutils.str2graph(dataset.examples[i].src)
    graph = None
    try:
        smile = example.src
        graph = myutils.str2graph(smile, have_3d)
    except Exception as e:
        print(e)
        # raise e
    setattr(example, 'graph', graph)
    return example


multiprocessing.log_to_stderr(level=logging.DEBUG)


def build_save_in_shards_using_shards_size(src_corpus, tgt_corpus, fields,
                                           corpus_type, opt, condition_corpus=None):
    """
    Divide src_corpus and tgt_corpus into smaller multiples
    src_copus and tgt corpus files, then build shards, each
    shard will have opt.shard_size samples except last shard.

    The reason we do this is to avoid taking up too much memory due
    to sucking in a huge corpus file.
    """

    with codecs.open(src_corpus, "r", encoding="utf-8") as fsrc:
        with codecs.open(tgt_corpus, "r", encoding="utf-8") as ftgt:
            logger.info("Reading source and target files: %s %s."
                        % (src_corpus, tgt_corpus))
            src_data = fsrc.readlines()
            tgt_data = ftgt.readlines()
            if condition_corpus:
                fcond = codecs.open(condition_corpus, "r", encoding="utf-8")
                cond_data = fcond.readlines()

            need_split = True
            num_shards = int(len(src_data) / opt.shard_size)
            for x in range(num_shards):
                shard_file = src_corpus + f".{x}.txt"
                # 存在一个文件,就认为sharding成功了,不再重做sharding
                if os.path.isfile(shard_file):
                    logger.info(f"文件 {shard_file} 已经存在,不在重复sharding")
                    need_split = False
                    break
            if need_split:
                for x in range(num_shards):
                    logger.info("Splitting shard %d." % x)
                    f = codecs.open(src_corpus + ".{0}.txt".format(x), "w",
                                    encoding="utf-8")
                    f.writelines(
                        src_data[x * opt.shard_size: (x + 1) * opt.shard_size])
                    f.close()
                    f = codecs.open(tgt_corpus + ".{0}.txt".format(x), "w",
                                    encoding="utf-8")
                    f.writelines(
                        tgt_data[x * opt.shard_size: (x + 1) * opt.shard_size])
                    f.close()
                    if condition_corpus:
                        f = codecs.open(condition_corpus + ".{0}.txt".format(x), "w",
                                        encoding="utf-8")
                        f.writelines(
                            cond_data[x * opt.shard_size: (x + 1) * opt.shard_size])
                        f.close()
                num_written = num_shards * opt.shard_size
                if len(src_data) > num_written:
                    logger.info("Splitting shard %d." % num_shards)
                    f = codecs.open(src_corpus + ".{0}.txt".format(num_shards),
                                    'w', encoding="utf-8")
                    f.writelines(
                        src_data[num_shards * opt.shard_size:])
                    f.close()
                    f = codecs.open(tgt_corpus + ".{0}.txt".format(num_shards),
                                    'w', encoding="utf-8")
                    f.writelines(
                        tgt_data[num_shards * opt.shard_size:])
                    f.close()
                    if condition_corpus:
                        f = codecs.open(condition_corpus + ".{0}.txt".format(num_shards),
                                        'w', encoding="utf-8")
                        f.writelines(
                            cond_data[num_shards * opt.shard_size:])
                        f.close()



    src_list = sorted(glob.glob(src_corpus + '.*.txt'))
    tgt_list = sorted(glob.glob(tgt_corpus + '.*.txt'))
    cond_list = sorted(glob.glob(condition_corpus + '.*.txt'))

    ret_list = parrel_run_by_example(cond_list, corpus_type, fields, opt, src_list, tgt_list)
    # ret_list = parrel_run(condition_corpus, corpus_type, fields, opt, src_list, tgt_list)

    return ret_list


def run_one(param):
    index, src, opt, fields, tgt_list, condition_corpus, corpus_type = param
    dataset = inputters.build_dataset(
        fields, opt.data_type,
        src_path=src,
        tgt_path=tgt_list[index],
        src_dir=opt.src_dir,
        src_seq_length=opt.src_seq_length,
        tgt_seq_length=opt.tgt_seq_length,
        src_seq_length_trunc=opt.src_seq_length_trunc,
        tgt_seq_length_trunc=opt.tgt_seq_length_trunc,
        dynamic_dict=opt.dynamic_dict,
        sample_rate=opt.sample_rate,
        window_size=opt.window_size,
        window_stride=opt.window_stride,
        window=opt.window,
        image_channel_size=opt.image_channel_size
    )

    pt_file = "{:s}.{:s}.{:d}.pt".format(
        opt.save_data, corpus_type, index)

    # We save fields in vocab.pt seperately, so make it empty.
    dataset.fields = []
    if condition_corpus:
        # 加载条件
        with open(condition_corpus) as f:
            target_condition = [int(s.rstrip()) for s in f.readlines()]

    tmp_example = []
    _ = [parrel_func(e, opt.with_3d_confomer) for e in dataset.examples]
    for cond, result in zip(target_condition, dataset.examples):
        if getattr(result, 'graph') is not None:
            if condition_corpus:
                setattr(result, 'condition_target', cond)
            tmp_example.append(result)

    dataset.examples = tmp_example

    with open(pt_file, 'wb') as f:
        pickle.dump(dataset, f)

    os.remove(src)
    os.remove(tgt_list[index])
    return pt_file

def parrel_run_by_example(cond_list, corpus_type, fields, opt, src_list, tgt_list):
    ret_list = []
    assert len(cond_list) == len(src_list) and len(cond_list) == len(tgt_list)
    for index, src in enumerate(src_list):
        logger.info("Building shard %d." % index)
        pt_file = "{:s}.{:s}.{:d}.pt".format(
            opt.save_data, corpus_type, index)
        if os.path.isfile(pt_file):
            # 文件已经生成,不在重复prepare
            logger.info(f"文件 {pt_file} 已经生成,不在重复prepare")
            continue

        dataset = inputters.build_dataset(
            fields, opt.data_type,
            src_path=src,
            tgt_path=tgt_list[index],
            src_dir=opt.src_dir,
            src_seq_length=opt.src_seq_length,
            tgt_seq_length=opt.tgt_seq_length,
            src_seq_length_trunc=opt.src_seq_length_trunc,
            tgt_seq_length_trunc=opt.tgt_seq_length_trunc,
            dynamic_dict=opt.dynamic_dict,
            sample_rate=opt.sample_rate,
            window_size=opt.window_size,
            window_stride=opt.window_stride,
            window=opt.window,
            image_channel_size=opt.image_channel_size
        )

        # We save fields in vocab.pt seperately, so make it empty.
        dataset.fields = []
        if len(cond_list) > 0:
            # 加载条件
            with open(cond_list[index]) as f:
                target_condition = [int(s.rstrip()) for s in f.readlines()]

        logger.info(" * saving %sth %s data shard to %s."
                    % (index, corpus_type, pt_file))
        if opt.parrel_run:
            pool = multiprocessing.Pool()
            dataset.examples = pool.map(partial(parrel_func, have_3d=opt.with_3d_confomer), dataset)
            pool.close()
            pool.join()
        else:
            _ = [parrel_func(e, opt.with_3d_confomer) for e in dataset.examples]
        tmp_example = []

        for cond, result in zip(target_condition, dataset.examples):
            if getattr(result, 'graph') is not None:
                if len(cond_list) > 0:
                    setattr(result, 'condition_target', cond)
                tmp_example.append(result)

        dataset.examples = tmp_example

        with open(pt_file, 'wb') as f:
            pickle.dump(dataset, f)

        ret_list.append(pt_file)
        os.remove(src)
        os.remove(tgt_list[index])
        del dataset.examples
        gc.collect()
        del dataset
        if opt.parrel_run:
            del pool
        gc.collect()
    return ret_list


def build_save_dataset(corpus_type, fields, opt):
    """ Building and saving the dataset """
    assert corpus_type in ['train', 'valid']

    if corpus_type == 'train':
        src_corpus = opt.train_src
        tgt_corpus = opt.train_tgt
        condition_corpus = opt.train_cond
    else:
        src_corpus = opt.valid_src
        tgt_corpus = opt.valid_tgt
        condition_corpus = opt.valid_cond

    if (opt.shard_size > 0):
        return build_save_in_shards_using_shards_size(src_corpus,
                                                      tgt_corpus,
                                                      fields,
                                                      corpus_type,
                                                      opt, condition_corpus)

    # For data_type == 'img' or 'audio', currently we don't do
    # preprocess sharding. We only build a monolithic dataset.
    # But since the interfaces are uniform, it would be not hard
    # to do this should users need this feature.
    dataset = inputters.build_dataset(
        fields, opt.data_type,
        src_path=src_corpus,
        tgt_path=tgt_corpus,
        src_dir=opt.src_dir,
        src_seq_length=opt.src_seq_length,
        tgt_seq_length=opt.tgt_seq_length,
        src_seq_length_trunc=opt.src_seq_length_trunc,
        tgt_seq_length_trunc=opt.tgt_seq_length_trunc,
        dynamic_dict=opt.dynamic_dict,
        sample_rate=opt.sample_rate,
        window_size=opt.window_size,
        window_stride=opt.window_stride,
        window=opt.window,
        image_channel_size=opt.image_channel_size)

    # We save fields in vocab.pt seperately, so make it empty.
    dataset.fields = []

    pt_file = "{:s}.{:s}.pt".format(opt.save_data, corpus_type)
    logger.info(" * saving %s dataset to %s." % (corpus_type, pt_file))

    # torch.save(dataset, pt_file)
    with open(pt_file, 'wb') as f:
        pickle.dump(dataset, f)
    return [pt_file]


def build_save_vocab(train_dataset, fields, opt):
    """ Building and saving the vocab """

    fields = inputters.build_vocab(train_dataset, fields, opt.data_type,
                                   opt.share_vocab,
                                   opt.src_vocab,
                                   opt.src_vocab_size,
                                   opt.src_words_min_frequency,
                                   opt.tgt_vocab,
                                   opt.tgt_vocab_size,
                                   opt.tgt_words_min_frequency)
    # Can't save fields, so remove/reconstruct at training time.
    vocab_file = opt.save_data + '.vocab.pt'
    # torch.save(inputters.save_fields_to_vocab(fields), vocab_file)
    with open(vocab_file, 'wb') as f:
        pickle.dump(inputters.save_fields_to_vocab(fields), f)


def main():
    opt = parse_args()

    if (opt.max_shard_size > 0):
        raise AssertionError("-max_shard_size is deprecated, please use \
                             -shard_size (number of examples) instead.")

    init_logger(opt.log_file)
    logger.info("Extracting features...")

    # 下面的代码是尝试解决多进程prepare失败的问题,但是没有效果
    torch.multiprocessing.set_sharing_strategy('file_system')
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (65535, rlimit[1]))
    # END

    src_nfeats = inputters.get_num_features(
        opt.data_type, opt.train_src, 'src')
    tgt_nfeats = inputters.get_num_features(
        opt.data_type, opt.train_tgt, 'tgt')
    logger.info(" * number of source features: %d." % src_nfeats)
    logger.info(" * number of target features: %d." % tgt_nfeats)

    logger.info("Building `Fields` object...")
    fields = inputters.get_fields(opt.data_type, src_nfeats, tgt_nfeats)
    myutils.add_more_field(fields)
    logger.info("Building & saving training data...")
    train_dataset_files = build_save_dataset('train', fields, opt)

    logger.info("Building & saving validation data...")
    build_save_dataset('valid', fields, opt)

    logger.info("Building & saving vocabulary...")
    build_save_vocab(train_dataset_files, fields, opt)


if __name__ == "__main__":
    main()
