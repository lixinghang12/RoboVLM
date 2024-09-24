"""Evaluate the default decision transformer."""
import argparse
from genericpath import isdir
import json
import logging
import os
from pathlib import Path
import sys
import time
import yaml
import re
from moviepy.editor import ImageSequenceClip
import copy
from utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

# This is for using the locally installed repo clone when using slurm
from calvin_agent.models.calvin_base_model import CalvinBaseModel
sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pyrender

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    create_tsne,
    get_default_model_and_env,
    get_env_state_for_initial_condition,
    get_log_dir,
    join_vis_lang,
    # print_and_save,
)
from calvin_agent.utils.utils import get_all_checkpoints, get_checkpoints_for_epochs, get_last_checkpoint
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm
from utils.config_utils import load_config, deep_update

from calvin_env.envs.play_table_env import get_env

from eval.calvin.model_wrapper import CustomModel, CustomModelFlamingoVideo, CustomModelLLaVA, CustomModelQwen, CustomModelLLaVaVideo
from eval.calvin.eval_utils import print_and_save

logger = logging.getLogger(__name__)

EP_LEN = 360
NUM_SEQUENCES = 1000
CACHE_ROOT = "/mnt/bn/robotics-data-lxh-lq/RoboVLM/eval/logs"
os.system(f"sudo mkdir -p {CACHE_ROOT}")
os.system(f"sudo chmod 777 {CACHE_ROOT}")

def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)

    # insert your own env wrapper
    # env = Wrapper(env)
    return env

def evaluate_policy(model, env, eval_sr_path, eval_result_path, eval_log_dir=None, debug=False, raw_calvin=False):
    """Run this function to evaluate a model on the CALVIN challenge."""
    conf_dir = Path("/mnt/bn/robotics/resources/calvin/calvin_models") / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)

    eval_sequences = get_sequences(NUM_SEQUENCES)

    results = []

    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    sequence_i = 0
    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, debug, eval_log_dir, sequence_i, raw_calvin)
        results.append(result)
        if not debug:
            success_list = count_success(results)
            with open(eval_sr_path, 'a') as f:
                line =f"{sequence_i}/{NUM_SEQUENCES}: "
                for sr in success_list:
                    line += f"{sr:.3f} | "
                sequence_i += 1
                line += "\n"
                f.write(line)
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(success_list)]) + "|"
            )
        else:
            sequence_i += 1

    print_and_save(results, eval_sequences, eval_result_path, None)

    return results


def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations, debug, eval_log_dir, sequence_i, raw_calvin):
    """Evaluates a sequence of language instructions."""
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask_i, subtask in enumerate(eval_sequence):
        success = rollout(env, model, task_checker, subtask, val_annotations, debug, eval_log_dir, subtask_i, sequence_i, raw_calvin)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(env, model, task_oracle, subtask, val_annotations, debug, eval_log_dir, subtask_i, sequence_i, raw_calvin=False):
    """Run the actual rollout on one subtask (which is one natural language instruction)."""
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
        img_list = []
    obs = env.get_obs()
    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()

    for _ in range(EP_LEN):
        action = model.step(obs, lang_annotation)
        obs, _, _, current_info = env.step(action)
        if debug:
            img_copy = copy.deepcopy(obs['rgb_obs']['rgb_static'])
            img_list.append(img_copy)

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if debug:
                print(colored("success", "green"), end=" ")
                clip = ImageSequenceClip(img_list, fps=30)
                clip.write_gif(os.path.join(eval_log_dir, f'{sequence_i}-{subtask_i}-{subtask}-succ.gif'), fps=30)
            return True

    if debug:
        print(colored("fail", "red"), end=" ")
        clip = ImageSequenceClip(img_list, fps=30)
        clip.write_gif(os.path.join(eval_log_dir, f'{sequence_i}-{subtask_i}-{subtask}-fail.gif'), fps=30)
    return False


def main():
    seed_everything(0, workers=True)  # type:ignore
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument("--dataset_path", default='/mnt/bn/robotics/manipulation_data/calvin_data/task_ABCD_D',
                        type=str, help="Path to the dataset root directory.")
    parser.add_argument("--debug", action="store_true", help="Print debug info and visualize environment.")

    # yaml_path takes the highest priority, then the log_dir, finally the config_path
    parser.add_argument('--config_path', type=str, default=None, help='path to the config file')
    parser.add_argument('--is_pt_config', action="store_true",
                        help='whether the specified config path is a pretrain config file.')

    parser.add_argument('--ckpt_dir', type=str, nargs='+', default="", help="checkpoint directory of the training")
    parser.add_argument('--ckpt_path', type=str, default=None, help="checkpoint directory of the training")
    parser.add_argument('--ckpt_idx', type=int, default=-1,help="which ckpt is going to be evaluated")
    parser.add_argument('--device_id', default=0, type=int, help="CUDA device")
    parser.add_argument('--no_cache', action="store_true")
    parser.add_argument('--raw_calvin', action="store_true")
    parser.add_argument('--debug_model', action="store_true")
    
    args = parser.parse_args()
    env = make_env(args.dataset_path)
    # import pdb; pdb.set_trace()
    config_path = args.config_path
    ckpt_dir = args.ckpt_dir
    ckpt_idx = args.ckpt_idx
    device_id = args.device_id

    # Loading configs
    assert config_path != None
    # Load config file
    configs = load_config(config_path)
    if args.is_pt_config:
        exp_name = os.path.basename(config_path)
        configs['exp_name'] = exp_name
        # generate ft configs from pt configs
        from utils.config_utils import generate_calvin_ft_configs, deep_update
        ft_configs = generate_calvin_ft_configs(configs)
        deep_update(configs, ft_configs)
    
    # Get checkpoint path
    from utils.eval_utils import sort_ckpt
    # print(ckpt_dir)
    # import pdb; pdb.set_trace()
    if isinstance(ckpt_dir, list):
        ckpt_dir = ckpt_dir[0]
    if args.ckpt_path is None:
        ckpt_files, ckpt_steps = sort_ckpt(ckpt_dir)
        if ckpt_idx >= len(ckpt_files):
            exit(0)
        ckpt_path = ckpt_files[ckpt_idx]
        ckpt_step = ckpt_steps[ckpt_idx]
        ckpt_dir = os.path.dirname(ckpt_path)
    else:
        import copy
        ckpt_path = args.ckpt_path or copy.copy(ckpt_dir)
        ckpt_dir = os.path.dirname(ckpt_path)
        ckpt_step = 0
    

    # Handle DeepSpeed ckpt
    if os.path.isdir(ckpt_path) and not args.debug_model:
        target_ckpt_path = ckpt_path.replace(".ckpt", ".pt")
        # from utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
        print(f"converting {ckpt_path} to {target_ckpt_path}")
        convert_zero_checkpoint_to_fp32_state_dict(ckpt_path, target_ckpt_path)
        ckpt_path = target_ckpt_path

    device = torch.device('cuda', device_id)

    from utils.config_utils import get_exp_name
    eval_exp_name = get_exp_name(f"{os.path.basename(config_path)}", mode='eval')
    if args.no_cache:
        eval_log_dir = ckpt_dir
    else:
        eval_log_dir = os.path.join(CACHE_ROOT, eval_exp_name)
    os.system(f"sudo mkdir {eval_log_dir}")
    os.system(f"sudo chmod 777 -R {eval_log_dir}")
    
    if configs['model'] == 'flamingo':
        model_fn = CustomModel
    elif configs['model'] == 'llava':
        history_type = configs['act_head'].get('history_type', 'post')
        if history_type == 'post':
            model_fn = CustomModelLLaVA
        else:
            model_fn = CustomModelLLaVaVideo
    elif 'qwen' in configs['model']:
        model_fn = CustomModelQwen
    elif configs['model'] == 'flamingo_video':
        model_fn = CustomModelFlamingoVideo
    else:
        raise ValueError(f"Unsupported model type {configs['model']}")
    
    model = model_fn(
        ckpt_path=ckpt_path,
        configs=configs,
        device=device,
        save_dir=eval_log_dir,
        raw_calvin=args.raw_calvin,
        debug=args.debug_model)

    # Success rate and result files
    ckpt_step = ckpt_path.split('/')[-1].split('.')[0]
    sr_path = os.path.join(eval_log_dir, f"success_rate_calvin_{ckpt_step}.txt")
    result_path = os.path.join(eval_log_dir, f"results_calvin_{ckpt_step}.json")
    cache_file = os.path.join(eval_log_dir, f"meta_info_step_{ckpt_step}.json")

    if not args.no_cache:
        if os.path.exists(cache_file):
            os.system(f"sudo rm {cache_file}")
        with open(cache_file, 'w') as f:
            _info = {
                "eval_sr_path": sr_path,
                "eval_result_path": result_path,
                "eval_log_dir": eval_log_dir,
                "raw_config_path": configs.get('raw_config_path', None)
            }
            json.dump(_info, f, indent=2)

    evaluate_policy(
        model,
        env,
        eval_sr_path=sr_path,
        eval_result_path=result_path,
        eval_log_dir=ckpt_dir,
        debug=args.debug)

    if args.no_cache:
        os.system("sudo rm -r ./temp/")


if __name__ == "__main__":
    # path = "/mnt/bn/robotics/manipulation_data/calvin_data/task_ABCD_D"
    # print(1)
    # make_env("/mnt/bn/robotics/manipulation_data/calvin_data/task_ABCD_D")
    # print(2)
    main()