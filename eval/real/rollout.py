import os
import torch
import torch.nn.functional as F
from eval.real.utils import *
from utils.config_utils import load_config
from utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
from eval.calvin.model_wrapper import CustomModel, CustomModelFlamingoVideo, CustomModelLLaVA, CustomModelQwen, CustomModelLLaVaVideo
import argparse
import torch.nn as nn

CACHE_ROOT = "/mnt/bn/robotics-data-lxh-lq/RoboVLM/eval/logs"
os.system(f"sudo mkdir -p {CACHE_ROOT}")
os.system(f"sudo chmod 777 {CACHE_ROOT}")

class Rollout(object):
    def __init__(self, config_path, ckpt_dir, ckpt_path, ckpt_idx, device_id, debug_model, no_cache):
        self.model = self.init_model(config_path, ckpt_dir, ckpt_path, ckpt_idx, device_id, debug_model, no_cache)
        self.device = self.model.device

        
    def init_model(config_path, ckpt_dir, ckpt_path, ckpt_idx, device_id, debug_model, no_cache):
        assert config_path != None
        configs = load_config(config_path)
        
        # Get checkpoint path
        from utils.eval_utils import sort_ckpt
        if isinstance(ckpt_dir, list):
            ckpt_dir = ckpt_dir[0]
        
        if ckpt_path is None:
            ckpt_files, _ = sort_ckpt(ckpt_dir)
            if ckpt_idx >= len(ckpt_files):
                exit(0)
            ckpt_path = ckpt_files[ckpt_idx]
            ckpt_dir = os.path.dirname(ckpt_path)
        else:
            import copy
            ckpt_path = ckpt_path or copy.copy(ckpt_dir)
            ckpt_dir = os.path.dirname(ckpt_path)
        
        # Handle DeepSpeed ckpt
        if os.path.isdir(ckpt_path) and not debug_model:
            target_ckpt_path = ckpt_path.replace(".ckpt", ".pt")
            print(f"converting {ckpt_path} to {target_ckpt_path}")
            convert_zero_checkpoint_to_fp32_state_dict(ckpt_path, target_ckpt_path)
            ckpt_path = target_ckpt_path

        device = torch.device('cuda', device_id)

        from utils.config_utils import get_exp_name
        eval_exp_name = get_exp_name(f"{os.path.basename(config_path)}", mode='eval')
        if no_cache:
            eval_log_dir = ckpt_dir
        else:
            eval_log_dir = os.path.join(CACHE_ROOT, eval_exp_name)
        os.system(f"sudo mkdir {eval_log_dir}")
        os.system(f"sudo chmod 777 -R {eval_log_dir}")
        
        if configs['model'] == 'flamingo':
            model_fn = CustomModel
        elif configs['model'] == 'llava':
            history_type = configs['act_head'].get('history_type', 'post')
            if history_type != 'video':
                model_fn = CustomModelLLaVA
            else:
                model_fn = CustomModelLLaVaVideo
        elif 'qwen' in configs['model']:
            model_fn = CustomModelQwen
        elif configs['model'] == 'flamingo_video':
            model_fn = CustomModelFlamingoVideo
        else:
            raise ValueError(f"Unsupported model type {configs['model']}")
        
        return model_fn(
            ckpt_path=ckpt_path,
            configs=configs,
            device=device,
            save_dir=eval_log_dir,
            debug=debug_model
        )

    def step(self, data):
        """Rollout a trajectory described by the text."""
        robot_state, static_rgb, hand_rgb, text, reset = data['robot_state'], data['static_rgb'], data['hand_rgb'], data['text'], data['reset']
        # TODO the robot_state is not used because now we just use the abs delta action
        # if we change to the relativa action, we need to use the robot_state to calculate the abs action 
        obs = {
            'rgb_obs': {
                'rgb_static': static_rgb,
                'rgb_gripper': hand_rgb,
            }
        }
        if reset:
            self.model.reset()
        
        action = self.model.step(obs, text)
        return action
