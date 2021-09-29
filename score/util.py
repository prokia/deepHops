# --*-- coding: utf-8 --*--

import os

def find_best_model_checkpoint(model_save_dir):
    all_ckpt = set()
    for f in os.listdir(model_save_dir):
        if not os.path.isfile(os.path.join(model_save_dir, f)) or not f.startswith('ckpt-'):
            continue
        all_ckpt.add(f.split('.')[0])
    ckpt_list = list(all_ckpt)
    ckpt_list.sort()
    return f"{model_save_dir}/{ckpt_list[0]}"