import copy
import json
import os
import re
import sys
import datetime
import time
import warnings
import math

import numpy as np
from tqdm import tqdm
from typing import List, Optional

import requests
import argparse

from decision_transformer.utils.utils import list_files
from decision_transformer.utils.config_utils import load_config, deep_update, \
    generate_calvin_ft_configs, CACHE_ROOT, get_exp_name, get_resume_path, get_cached_exp_info,\
    get_single_gpu_bsz
from decision_transformer.utils.eval_utils import sort_ckpt

# Usage
"""
python3 scripts/trial_manager.py configs/data_ablation_exps_vqgan_512bsz/ -g 350 -t 6169063 -m pretrain -r
"""

# TODO: share this path with all config files.
MAX_GPU_NUM = 352
MAX_GPU_NUM_EVAL = 48

ARNOLD_TOKEN = "b8882c17c787f0358f87bae43dee3802f125abaa"
TASK_ID = "5858555"
CPU_PER_GPU = 15
MEM_PER_GPU = 200 * 1024
DEFAULT_ROLES = {
    "cpu_only": {"role": "worker", "num": 1, "cpu": 4, "mem": 32 * 1024, "gpu": 0, "ports": 1},
    "gpu": {"role": "worker", "ports": 1},
}
GROUP = {"ailab_nlp": 402, "ailab_nlp_v100": 457, "robot_training": 538, 'VideoProj': 87}
CLUSTER = {"lq": 17, "hl": 20}
REGION = "cn"

ARNOLD_VOLUMES_LQ = [
    {"name": "robotics-data-hl",    "access_mode": "RW", "roles": ["worker"]},
    {"name": "robotics-lq2024",     "access_mode": "RW", "roles": ["worker"]},
    {"name": "robotics-data-wht",   "access_mode": "RO", "roles": ["worker"]},
    {"name": "robotics-real-data",  "access_mode": "RO", "roles": ["worker"]},
    {"name": "robotics",            "access_mode": "RO", "roles": ["worker"]},
    {"name": "robotics-lq2024-data","access_mode": "RO", "roles": ["worker"]}
]

def get_date():
    now = str(datetime.datetime.now())
    date = now.split()[0]
    return date

def get_time():
    now = str(datetime.datetime.now())
    t = now.split('.')[0].replace(' ', '-').replace(':', '.')
    return t

def list_all_files(dirs, verbose=False):
    sub_dirs = list_files(dirs)
    all_files = []
    all_dirs = []

    if verbose:
        _iter = tqdm(sub_dirs)
    else:
        _iter = sub_dirs

    for d in _iter:
        if os.path.isdir(d):
            all_dirs.append(d)
        else:
            all_files.append(d)

    if all_dirs:
        all_files.extend(list_all_files(all_dirs))
    return all_files


def _backup(f_path):
    backup_dir = os.path.join(CACHE_ROOT, 'trial_info_backup')
    os.makedirs(backup_dir, exist_ok=True)
    backup_path = os.path.join(backup_dir, f"{os.path.basename(f_path)}_{get_date()}")
    if os.path.exists(f_path):
        print(f"Trial info backup to: {backup_path}")
        os.system(f"sudo cp {f_path} {backup_path}")


def _init_trial_info(trial_info_file):
    if os.path.exists(trial_info_file):
        with open(trial_info_file, "r") as f:
            trial_info = json.load(f)
    else:
        confirm = input(f"No trial info is loaded successfully for {trial_info_file}. "
                        f"Continue (c) or Quit (q): \n")
        if confirm == 'c':
            trial_info = {}
        else:
            exit(0)
    return trial_info


class Manager:
    def __init__(
            self,
            exp_dir,
            resume=False,
            token=ARNOLD_TOKEN,
            pretrain_info_load_path="trial_status.json",
            finetune_info_load_path="trial_status_ft.json",
            eval_info_load_path="trial_status_eval.json",
            max_gpu_num=MAX_GPU_NUM,
    ):
        """
        exp_dir:
            the dir that includes configs of all experiments to run

        Trial info keys:
            0) Trial ID
            1) Launched time stamp
            2) Running time
            3) Status
            4) Entrance command
            5) Number of GPUs
            6) Training Configs
            7) Exp names
            8) Model size
            9) Training dataset list
            10) Data ratio used for training
            11) Maximum steps of training
        """
        self.token = token
        self.resume = resume
        self.max_gpu_num = max_gpu_num

        self.pretrain_trial_info_file = os.path.join(CACHE_ROOT, pretrain_info_load_path)
        self.finetune_trial_info_file = os.path.join(CACHE_ROOT, finetune_info_load_path)
        self.eval_trial_info_file = os.path.join(CACHE_ROOT, eval_info_load_path)

        _backup(self.pretrain_trial_info_file)
        _backup(self.finetune_trial_info_file)
        _backup(self.eval_trial_info_file)

        self.pretrain_trial_info = _init_trial_info(self.pretrain_trial_info_file)
        self.finetune_trial_info = _init_trial_info(self.finetune_trial_info_file)
        self.eval_trial_info = _init_trial_info(self.eval_trial_info_file)

        exp_file_list = list_all_files(exp_dir)
        exp_file_list = [p for p in exp_file_list if re.search(
            r'(.*/__.*__/.*|.*/default.*|.*debug.*|.*finetune.*|.*eval.*)', p) is None]
        self.exps = {}
        for exp_config in exp_file_list:
            exp_name = os.path.basename(exp_config)
            if exp_name in self.exps:
                print(f"-- Duplicate experiment item: {exp_name}. \n\tExisting: {self.exps[exp_name]}. "
                      f"\n\tNew (ignored): {exp_config}. ")
                continue
            self.exps[os.path.basename(exp_config)] = exp_config

        self.headers = {"Authorization": f"Token {self.token}"}
        if REGION == "i18n":
            self.api_host = "https://arnold-i18n-api.byted.org/api/v2"
        elif REGION == "cn":
            self.api_host = "https://arnold-api.byted.org/api/v2"
        elif REGION == "us-ttp":
            self.api_host = "https://arnold-api.tiktokd.org/api/v2"

    def _get_trial_info_file_vars(self, mode='pretrain'):
        if mode == 'pretrain':
            return self.pretrain_trial_info, self.pretrain_trial_info_file
        elif mode == 'finetune':
            return self.finetune_trial_info, self.finetune_trial_info_file
        elif mode == 'eval':
            return self.eval_trial_info, self.eval_trial_info_file
        else:
            raise ValueError

    def _save_trial_status(self, mode='pretrain'):
        trial_info, trial_info_file = self._get_trial_info_file_vars(mode=mode)
        print(f"Trial status cached to: {trial_info_file}")
        with open(trial_info_file, "w") as f:
            json.dump(trial_info, f, indent=2)

    def _get_gpu_used_eval(self):
        gpu_num = 0
        trial_info, _ = self._get_trial_info_file_vars('eval')
        for exp_name, trials in trial_info.items():
            for trial in trials:
                if trial['status'] not in {'failed', 'finished', 'stopped', 'killed', 'succeeded'}:
                    gpu_num += trial['num_gpus']
        return gpu_num

    def _get_gpu_used(self):
        gpu_num = 0
        for mode in {'finetune', 'pretrain'}:
            trial_info, _ = self._get_trial_info_file_vars(mode)
            for exp_name, trial in trial_info.items():
                if trial['status'] not in {'failed', 'finished', 'stopped', 'killed', 'succeeded'}:
                     gpu_num += trial['num_gpus']
        return gpu_num

    def _update_single_trial_status(self, trial_info):
        # trial_id is None when the trial is a dummy trial.
        trial_id = trial_info.get('trial_id', None)

        if trial_info['status'] == 'finished':
            return

        if 'start_idx' in trial_info:
            trial_info['ckpt_idx'] = trial_info.pop('start_idx')

        if trial_id is not None:
            # update other trial infos
            trial_info['status'] = self.read_trial(trial_id)['runs'][0]['status']

            if 'last_time_stamp' in trial_info:
                new_time_stemp = time.time()
                time_elapsed = new_time_stemp - trial_info['last_time_stamp']
                trial_info['last_time_stamp'] = new_time_stemp
                trial_info['running_time'] += time_elapsed
            else:
                trial_info['last_time_stamp'] = time.time()

    def _update_trial_status(self):
        for mode in {"pretrain", "finetune", "eval"}:
            print(f"Updating trial status: {mode}.")
            trial_infos, _ = self._get_trial_info_file_vars(mode)

            for exp_name in trial_infos:
                trial_info = trial_infos[exp_name]
                if isinstance(trial_info, dict):
                    self._update_single_trial_status(trial_info)
                else:
                    assert isinstance(trial_info, list)
                    for t in trial_info:
                        self._update_single_trial_status(t)

            self._save_trial_status(mode)

    def _is_exp_finished(self, exp, mode='pretrain'):
        trial_infos, _ = self._get_trial_info_file_vars(mode)
        if exp not in trial_infos:
            return False
        trial_info = trial_infos[exp]
        if isinstance(trial_info, dict):
            return trial_info['status'].lower() in {"succeeded", "finished"}
        elif isinstance(trial_info, (list, tuple)):
            _is_finished = []
            for t in trial_info:
                _is_finished.append(t['status'].lower() in {"succeeded", "finished"})
            return bool(np.array(_is_finished).all())
        else:
            raise ValueError

    def _get_exp_status(self, exp, mode='pretrain'):
        trial_infos, _ = self._get_trial_info_file_vars(mode)
        if exp not in trial_infos:
            status = "Waiting"
            return status
        trial_info = trial_infos[exp]
        if isinstance(trial_info, dict):
            return trial_info['status']
        elif isinstance(trial_info, (list, tuple)):
            status = []
            for t in trial_info:
                status.append(t['status'])
            return str(status)
        else:
            raise ValueError

    def step(self, mode):
        """
                in each step, the manager will check the current status of all exps,
                    * if a trial is finished, then proceed to the next task.
                    * if running, then ignore
                    * if failed / others, then re-run the trial
                """
        assert mode in {'pretrain', 'finetune', 'eval'}
        self._update_trial_status()

        trial_infos, _ = self._get_trial_info_file_vars(mode)

        print("Checking Experiment Status.")
        for exp in self.exps:
            print(f"--- Mode: {mode}")

            if exp not in trial_infos:
                launch_func = self.launch_exp_eval if mode == 'eval' else self.launch_exp
                trial_info = launch_func(exp, mode)
                if trial_info:
                    trial_infos[exp] = trial_info

            elif mode in {"pretrain", "finetune"}:
                status = trial_infos[exp]['status']
                if status in {'failed', 'stopped', 'killed', 'killing'}:
                    # re-launch the trial
                    trial_id = trial_infos[exp]['trial_id']
                    self.stop_trial(trial_id)
                    trial_info = self.launch_exp(exp, mode=mode)
                    if trial_info:
                        trial_infos[exp] = trial_info

            else:
                prev_trials = trial_infos[exp]
                unfinished_ckpt_idx_list = []
                for trial in prev_trials:
                    status = trial['status']
                    if status in {'failed', 'stopped', 'killed', 'killing'}:
                        unfinished_ckpt_idx_list.append(trial['ckpt_idx'])
                        # stop the unfinished trials
                        trial_id = trial['trial_id']
                        self.stop_trial(trial_id)

                trial_info = self.launch_exp_eval(exp, mode=mode, ckpt_idx_list=unfinished_ckpt_idx_list)
                restart_idx_list = [t['ckpt_idx'] for t in trial_info]
                for t in prev_trials:
                    if t['ckpt_idx'] not in restart_idx_list:
                        trial_info.append(t)
                if trial_info:
                    trial_infos[exp] = trial_info

            print(f"{exp}: {self._get_exp_status(exp, mode=mode)}")

        self._update_trial_status()

        is_finished = True
        for exp in self.exps:
            exp_finished = self._is_exp_finished(exp, mode=mode)
            if not exp_finished:
                is_finished = False

        trial_status = {}
        for exp in self.exps:
            trial_status[exp] = self._get_exp_status(exp, mode=mode)

        print(f"A100 GPU Occupied: {self._get_gpu_used()}. V100 GPU Occupied: {self._get_gpu_used_eval()}.")
        print(f"Step done. {'Still running...' if not is_finished else 'Finished.'}")
        return is_finished, trial_status

    def _stop_exp(self, exp, mode='pretrain'):
        trial_infos, _ = self._get_trial_info_file_vars(mode)
        if exp not in trial_infos:
            return
        trial_info = trial_infos[exp]
        if isinstance(trial_info, dict):
            trial_id = trial_infos[exp]['trial_id']
            print(f"Stopping trial {trial_id} for experiment {exp}.")
            self.stop_trial(trial_id)
            return
        elif isinstance(trial_info, (list, tuple)):
            for t in trial_info:
                trial_id = t[exp]['trial_id']
                print(f"Stopping trial {trial_id} for experiment {exp}.")
                self.stop_trial(trial_id)
            return
        else:
            raise ValueError

    def stop(self, mode):
        print("=" * 20 + " Stopping trials. " + "=" * 20)
        assert mode in {'pretrain', 'finetune', 'eval'}
        trial_infos, _ = self._get_trial_info_file_vars(mode)
        self._update_trial_status()

        for exp in self.exps:
            self._stop_exp(exp, mode=mode)

        self._update_trial_status()
        trial_status = {}
        for exp in self.exps:
            trial_status[exp] = self._get_exp_status(exp, mode=mode)
            print(f"{exp}: {trial_status[exp]}.")
        return True, trial_status

    def _get_ft_model_load_path(self, exp):
        # return the latest ckpt of the pretrained models
        return get_resume_path(exp, mode='pretrain')

    def _get_mode3_ft_dataset_configs(self):
        raise NotImplementedError

    def _generate_ft_configs(self, pt_configs, ft_dataset='calvin'):
        ft_configs = copy.deepcopy(pt_configs)
        if ft_dataset == 'calvin':
            calvin_configs = generate_calvin_ft_configs(pt_configs)
            deep_update(ft_configs, calvin_configs)
        else:
            raise NotImplementedError

        ft_config_path = self._get_exp_cfg_path(
            pt_configs['exp_name'], ft_eval_on=ft_dataset, mode='finetune')
        with open(ft_config_path, 'w') as f:
            json.dump(ft_configs, f, indent=2)
        print(f"Finetune config generated: {ft_config_path}")

        return ft_configs

    def _get_trial_params(self, exp_config, mode='pretrain'):
        import math

        if mode in {"pretrain", "finetune"}:
            # for me, default role settings are enough.
            single_gpu_bsz = get_single_gpu_bsz(exp_config)

            if isinstance(single_gpu_bsz, list):
                single_gpu_bsz = sum(single_gpu_bsz)

            if mode == 'pretrain':
                total_bsz = exp_config.get("total_batch_size", 1024)
            else:
                total_bsz = 512

            role = DEFAULT_ROLES['gpu']

            n_gpu = math.ceil(total_bsz / single_gpu_bsz)
            if n_gpu <= 8:
                role['num'] = 1
                role['gpu'] = int(n_gpu)
            else:
                n_gpu_adjusted = int(math.ceil(n_gpu / 8)) * 8
                if n_gpu != n_gpu_adjusted:
                    warnings.warn(
                        f"Batch size results in a number of gpu that cannot be divided by 8. "
                        f"Batch size has been adjusted to: {n_gpu_adjusted * single_gpu_bsz}"
                    )
                role['num'] = int(n_gpu_adjusted / 8)
                role['gpu'] = 8

            role['cpu'] = role['gpu'] * CPU_PER_GPU
            role['mem'] = role['gpu'] * MEM_PER_GPU
            role['gpuv'] = "A100-SXM-80GB"

            return role

        elif mode == 'eval':
            # for evaluation, we use fixed params here
            role = DEFAULT_ROLES['gpu']
            role['num'] = 1
            role['gpu'] = 1
            role['gpuv'] = "Tesla-V100-SXM2-32GB"
            role['cpu'] = 16
            role['mem'] = 32 * 1024
            return role

        else:
            raise ValueError

    def _parse_data_list(self, dataset_config):
        data_names = []

        if dataset_config['type'] == 'ConcatDataset':
            for d in dataset_config['datasets']:
                data_names.extend(self._parse_data_list(d))

        elif dataset_config['type'] == 'GRDataset':
            p = dataset_config['data_dir']
            matched = re.search(r'anns/.+', p)
            if matched is not None:
                data_names.append(matched.group().split('/')[1])
            else:
                data_names.append('UNKNOWN')

        else:
            data_names.append(dataset_config['type'].lower().replace(
                'dataset', '').strip().strip('_-'))

        return data_names

    def _parse_exp_info(self, exp_config):
        model_size = exp_config['llm']['n_embd']

        train_dataset_list = []
        if isinstance(exp_config['train_dataset'], list):
            for data_cfg in exp_config['train_dataset']:
                train_dataset_list.extend(self._parse_data_list(data_cfg))
        else:
            train_dataset_list.extend(self._parse_data_list(exp_config['train_dataset']))

        data_size = exp_config.get('data_size', 1.)
        training_steps = exp_config['trainer']['max_steps']
        return {
            "model_size": model_size,
            "train_data": train_dataset_list,
            "data_size": data_size,
            "train_steps": training_steps
        }

    def _get_exp_cfg_path(self, exp, ft_eval_on='calvin', mode='pretrain'):
        if mode == 'pretrain':
            # use the config file in the trial docker
            return self.exps[exp]
        elif mode in {'finetune', 'eval'}:
            # use the generated config file based on the pretrain file
            os.makedirs(os.path.join(CACHE_ROOT, "auto_configs"), exist_ok=True)
            cfg_path = os.path.join(CACHE_ROOT, "auto_configs",
                                    get_exp_name(exp, mode=mode, ft_eval_on=ft_eval_on))
            return os.path.abspath(cfg_path)
        else:
            raise ValueError

    def _is_pretrain_finished(self, exp):
        return self._is_exp_finished(exp, mode='pretrain')

    def _is_finetune_finished(self, exp):
        return self._is_exp_finished(exp, mode='finetune')

    def _is_eval_finished(self, exp):
        return self._is_exp_finished(exp, mode='eval')

    def remove_exp(self, exp, mode='pretrain'):
        # if exp not in self.exps:
        #     return
        # remove all exp data
        cached_exp_info = get_cached_exp_info(exp, mode)
        remove_list = []
        if cached_exp_info is not None:
            if mode in {"pretrain", "finetune"}:
                def remove_dir(exp_info, dir_key):
                    if dir_key not in exp_info:
                        return
                    dir_list = exp_info[dir_key]
                    if isinstance(dir_list, str):
                        dir_list = [dir_list]
                    for d in dir_list:
                        if os.path.exists(d):
                            remove_list.append(d)

                remove_dir(cached_exp_info, "log_root")
                remove_dir(cached_exp_info, "ckpt_root")
                remove_dir(cached_exp_info, "tensorboard_dir")
                remove_dir(cached_exp_info, "cfgdir")

            elif mode == 'eval':
                remove_keys = ['eval_sr_path', 'eval_result_path']
                for s in cached_exp_info:
                    for k in remove_keys:
                        remove_list.append(cached_exp_info[s][k])

            else:
                raise ValueError

        return remove_list

    def launch_exp_eval(self, exp, mode='eval', ckpt_idx_list=None, eval_on='calvin'):
        if exp not in self.exps:
            return {}

        if eval_on != 'calvin':
            raise NotImplementedError

        if mode == 'eval' and not self._is_finetune_finished(exp):
            # eval trials only start after finetuning finished
            print(f"Finetune exp of {exp} is not finished. "
                  f"Evaluation will be launched after it ends.")
            return {}

        trial_infos, _ = self._get_trial_info_file_vars(mode)
        exp_config = load_config(self.exps[exp])
        exp_config['exp_name'] = exp
        model_size = exp_config['llm']['n_embd']

        ckpt_dir = get_cached_exp_info(exp, mode='finetune')
        if ckpt_dir is None: raise FileNotFoundError(f"No ckpt is found at: {ckpt_dir}")
        ckpt_dir = ckpt_dir['ckpt_root']
        ckpt_files, ckpt_steps = sort_ckpt(ckpt_dir)
        ckpt_num = len(ckpt_files)

        command_list = []
        trial_param_list = []
        # evaluate the last 10 ckpts for each experiment
        if ckpt_idx_list is None:
            ckpt_idx_list = range(max(0, ckpt_num - 8), ckpt_num, 2)

        _ckpt_idx_valid = []
        for i in ckpt_idx_list:
            _ckpt_idx = i
            num_ckpt = 2
            config_path = self._get_exp_cfg_path(exp, mode='pretrain')

            if self._get_gpu_used_eval() < MAX_GPU_NUM_EVAL:
                commands = f"run/run_evaluate_calvin.sh {_ckpt_idx} {num_ckpt} " \
                           f"--ckpt_dir {' '.join(ckpt_dir) if isinstance(ckpt_dir, (list, tuple)) else ckpt_dir} " \
                           f"--config_path {config_path} --is_pt_config"
                trial_params = self._get_trial_params(exp_config, mode='eval')
                command_list.append(commands)
                trial_param_list.append(trial_params)
                _ckpt_idx_valid.append(i)

            else:
                break

        trial_info_list = []
        for _ckpt_idx, commands, trial_params in zip(_ckpt_idx_valid, command_list, trial_param_list):
            trial_info = self._parse_exp_info(exp_config)
            print(f"Launch experiment for {exp}. \n\tArnold Task ID: {TASK_ID}. \n\tCommand: {commands}")
            trial_info.update(self.create_trial_and_run(
                args=commands,
                task_id=TASK_ID,
                mode=mode,
                exp_name=exp,
                group_name="ailab_nlp_v100",
                cluster_name="lq",
                trial_param=trial_params,
                comment=f"EVAL GR13 Model:{model_size} Config:{exp}"[:90]
            ))
            trial_info['ckpt_idx'] = _ckpt_idx
            trial_info_list.append(trial_info)

        return trial_info_list

    def launch_exp(self, exp, mode='pretrain'):
        """
        6) Training Configs
        7) Exp names
        8) Model size
        9) Training dataset list
        10) Data ratio used for training
        11) Maximum steps of training
        """
        assert mode in {"pretrain", "finetune"}

        if exp not in self.exps:
            return {}

        if mode == 'finetune' and not self._is_pretrain_finished(exp):
            # finetune exp only starts after pretraining finished
            return {}

        trial_infos, _ = self._get_trial_info_file_vars(mode)
        exp_config = load_config(self.exps[exp])
        if mode == 'pretrain' and exp_config['no_pretrain']:
            return {
                "mode": mode,
                "status": "finished",
            }

        model_size = exp_config['llm']['n_embd']
        exp_config['exp_name'] = exp
        if mode == 'finetune':
            # generate finetune configs from pretrain configs
            exp_config = self._generate_ft_configs(exp_config)

        trial_params = self._get_trial_params(exp_config, mode=mode)

        # number of times that the trial has been relaunched
        num_retry = trial_infos.get(exp, {}).get('num_retry', 0)
        trial_info = self._parse_exp_info(exp_config)

        if self._get_gpu_used() + trial_params['gpu'] * trial_params['num'] > self.max_gpu_num:
            return {}

        else:
            cfg_path = self._get_exp_cfg_path(exp, mode=mode)
            assert os.path.exists(cfg_path)
            commands = f"run/run.sh {cfg_path}"

            if self.resume:
                resume_path = get_resume_path(exp, mode=mode)
                if resume_path is not None:
                    commands = ' '.join([commands, '--resume', resume_path])

            print(f"Launch experiment for {exp}. \n\tArnold Task ID: {TASK_ID}. \n\tCommand: {commands}")
            trial_info.update(self.create_trial_and_run(
                args=commands,
                task_id=TASK_ID,
                mode=mode,
                exp_name=exp,
                group_name="robot_training",
                cluster_name="lq",
                trial_param=trial_params,
                comment=f"{mode.upper()} GR13 Model:{model_size} Config:{exp}"[:90]
            ))
            trial_info['num_retry'] = num_retry + 1
            return trial_info

    def create_trial_and_run(
        self,
        args,
        task_id,
        mode,
        exp_name,
        group_name,
        cluster_name,
        trial_param,
        keep_mins=1,
        comment="",
        restart_times=0,
    ):
        assert group_name in GROUP
        assert isinstance(trial_param, dict)
        gpuv = trial_param['gpuv']
        roles = [trial_param]
        group_ids = [GROUP[group_name]]
        cluster_id = CLUSTER[cluster_name]
        url = self.api_host + "/task/{task_id}/trial/"
        url = url.format(task_id=task_id)

        data = {
            "args": args,
            "keep_mins": keep_mins,
            "group_ids": group_ids,
            "cluster_id": cluster_id,
            "roles": roles,
            "comment": comment,
            "restart_times": restart_times,
            "preemptible": False,
            "mask_hosts": ["10.136.97.198"],
            "envs": {
                "ARNOLD_BYTENAS_VOLUMES": ARNOLD_VOLUMES_LQ
            }
        }

        while True:
            try:
                resp = requests.post(url, json=data, headers=self.headers).json()
                trial_id = resp[0]["id"]
                break
            except:
                print(f"{get_time()}: failed to get response with request.post. Experiment name: {exp_name}.")

        """
        0) Trial ID
        1) Launched time stamp
        2) Running time
        3) Status
        4) Entrance command
        5) Number of GPUs
        """

        trial_info = {
            "trial_id": trial_id,
            "mode": mode,
            "task_id": TASK_ID,
            "launched_time": get_time(),
            "launched_date": get_date(),
            "running_time": 0,
            "status": "waiting",
            "command": args,
            "num_gpus": roles[0]['num'] * roles[0]['gpu'],
            "gpu_type": gpuv,
        }
        return trial_info

    def read_trial(self, trial_id):
        for i in range(5):
            try:
                url = self.api_host + "/trial/{trial_id}/"
                url = url.format(trial_id=trial_id)
                resp = requests.get(url, headers=self.headers).json()
                return resp
            except:
                pass

        raise RuntimeError

    def stop_trial(self, trial_id):
        url = self.api_host + f"/trial/{trial_id}/stop/"
        url = url.format(trial_id=trial_id)
        resp = requests.post(url, headers=self.headers)
        return resp.status_code

    def delete_trial(self, trial_id):
        url = self.api_host + "/trial/{trial_id}/"
        url = url.format(trial_id=trial_id)
        resp = requests.delete(url, headers=self.headers)
        return resp.status_code



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('expdir', type=str, nargs='+')
    # translate expdir to abs path or not.
    parser.add_argument('-s', '--stop', action='store_true')
    parser.add_argument('-t', '--taskid', type=str)                 # ARNOLD TASK ID
    parser.add_argument('-r', '--resume', action='store_true')      # WHETHER TO RESUME
    parser.add_argument('-g', '--gpu-num', type=int, default=0)     # MAX GPU NUM
    # -m pretrain finetune eval
    # -m pretrain: only pretrain trials
    # -m pretrain finetune: first pretrain, then finetune. finetune priority is higher.
    parser.add_argument('-m', '--mode', type=str, nargs='+', default=['pretrain'])
    parser.add_argument('--reset', type=str, nargs='+', default=[])  # REMOVE EXPERIMENTS
    args = parser.parse_args()

    modes = args.mode
    if args.taskid is not None:
        assert re.match(r'\d{7}$', args.taskid) is not None, f"Unrecognized Task ID: {args.taskid}."
        TASK_ID = args.taskid

    exp_dir = args.expdir
    gpu_num = MAX_GPU_NUM if args.gpu_num == 0 else args.gpu_num
    assert gpu_num > 0

    trial_manager = Manager(
        exp_dir,
        resume=args.resume,
        max_gpu_num=gpu_num,
    )

    count = 0
    interval = 1200

    # defines the priority of all modes
    sorted_modes = ['eval', 'finetune', 'pretrain']

    reset_list = args.reset

    if args.stop:
        for mode in modes:
            trial_manager.stop(mode)

    elif len(reset_list) > 0:
        if reset_list[0] == 'all':
            # reset all specified exps under args.expdir
            reset_list = trial_manager.exps

        all_remove_list = []
        for exp in reset_list:
            for mode in modes:
                remove_list = trial_manager.remove_exp(exp, mode)
                all_remove_list.extend(remove_list)

        print(f"Experiments to be removed for mode {modes}: ")
        for exp in reset_list:
            print('-- ' + exp)

        print("Directories & Files to be removed: ")
        for d in all_remove_list:
            print('-- ' + d)

        confirm = input("Confirm? ('y' to confirm, otherwise no action will be taken.)")
        if confirm != 'y':
            exit()
        else:
            for d in all_remove_list:
                if os.path.exists(d):
                    print(f"Removing: {d}")
                    os.system(f"sudo rm -r {d}")

            for mode in modes:
                # reset trial status and cache to the disk
                trial_infos, trial_info_file = trial_manager._get_trial_info_file_vars(mode=mode)
                for exp in reset_list:
                    trial_infos.pop(exp, None)
                trial_manager._save_trial_status(mode=mode)

    else:
        while True:
            count += 1

            all_dones = []
            for mode in sorted_modes:
                if mode not in modes:
                    continue
                print("=" * 15 + f" {mode.upper()}: Step {count} (~{interval // 60} minutes each step) " + "=" * 15)
                _done, trial_status = trial_manager.step(mode)
                all_dones.append(_done)

            if np.all(all_dones):
                break

            if count >= 2:
                time.sleep(interval)


