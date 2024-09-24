import math
import json
import os
import re
from utils.utils import list_files

CACHE_ROOT = "/mnt/bn/robotics-data-hl/zhb/cache"

def deep_update(d1, d2):
    for k, v in d2.items():
        if isinstance(v, dict):
            if v.get('__override__', False):
                d1[k] = v
                d1[k].pop("__override__", None)
            elif k in d1 and isinstance(d1[k], dict):
                deep_update(d1[k], v)
            else:
                d1[k] = v
        else:
            d1[k] = v

    return d1


def load_config(config_file):
    print(config_file)
    _config = json.load(open(config_file))
    config = {}
    if _config.get('parent', None):
        deep_update(config, load_config(_config['parent']))
    deep_update(config, _config)
    return config


def get_single_gpu_bsz(exp_config):
    if isinstance(exp_config['batch_size'], int):
        if isinstance(exp_config['train_dataset'], list):
            return exp_config['batch_size'] * len(exp_config['train_dataset'])
        else:
            assert isinstance(exp_config['train_dataset'], dict)
            return exp_config['batch_size']
    else:
        assert isinstance(exp_config['batch_size'], list)
        return sum(exp_config['batch_size'])

def get_exp_name(exp, mode='pretrain'):
    if mode == 'pretrain':
        return exp
    else:
        return f"{exp}_{mode}"

def get_cached_exp_info(exp, mode='pretrain'):
    if mode in {"pretrain", 'finetune'}:
        exp = get_exp_name(exp, mode)
        cached_exp_info = os.path.join(CACHE_ROOT, f"log_setting.{exp}.json")
        if not os.path.exists(cached_exp_info):
            return None
        with open(cached_exp_info, 'r') as f:
            cached_exp_info = json.load(f)
        return cached_exp_info

    else:
        ft_exp_name = get_exp_name(exp, mode='finetune')
        eval_cache_root = os.path.join(CACHE_ROOT, "eval")
        cache_list = os.listdir(eval_cache_root)
        cache_list = [f for f in cache_list if f.startswith(ft_exp_name)]
        step_list = [int(re.search(r"step_\d+", f).group()[5:]) for f in cache_list]
        cached_exp_info = {}
        for s, c in zip(step_list, cache_list):
            with open(os.path.join(eval_cache_root, c)) as f:
                cached_exp_info[s] = json.load(f)
        return cached_exp_info


def get_resume_path(exp, mode="pretrain"):
    assert mode in {"pretrain", "finetune"}, "Eval trials cannot be resumed."

    cached_exp_info = get_cached_exp_info(exp, mode)
    if cached_exp_info is None:
        return None

    ckpt_dir = cached_exp_info['ckpt_root']
    if isinstance(ckpt_dir, str):
        ckpt_dir = [ckpt_dir]
    ckpt_list = list_files(ckpt_dir)
    # FIXME: hack here to exclude the converted ckpt using deepspeed.
    ckpt_list = [c for c in ckpt_list if not c.endswith(".fp32.pt")]

    if len(ckpt_list) == 0:
        return None

    # resume from the last checkpoint
    ckpt_epochs = [re.search(r'epoch=\d+', ckpt).group()[6:].rjust(3, '0')
                   for ckpt in ckpt_list if ckpt.endswith('.ckpt')]
    ckpt_steps = [re.search(r'step=\d+', ckpt).group()[5:].rjust(8, '0')
                  for ckpt in ckpt_list if ckpt.endswith('.ckpt')]
    ckpt_ids = [int(e + s) for e, s in zip(ckpt_epochs, ckpt_steps)]

    ckpt_id_to_path = dict(zip(ckpt_ids, ckpt_list))
    last_id = max(ckpt_ids)
    resume_path = ckpt_id_to_path[last_id]
    return resume_path

def generate_calvin_ft_configs(pt_configs):
    raise NotImplementedError