from ctypes import wstring_at
from distutils.command.config import config
from email.policy import strict
import json
import os.path
from lightning.pytorch.trainer import Trainer
from copy import deepcopy
import torch
import torchvision.transforms as T
from PIL import Image
from typing import Literal
import copy
import numpy as np
from eval.calvin.eval_utils import euler2rotm, rotm2euler
from calvin_agent.models.calvin_base_model import CalvinBaseModel
from scripts.main import init_trainer_config
from train.flamingo_trainer import FlamingoTrainer
from train.flamingo_video_trainer import FlamingoVideoTrainer
from train.llava_trainer import LLaVATrainer
from train.qwen_trainer import QwenTrainer
from utils.model_utils import build_tokenizer
from data.datamodule.gr_datamodule import GRDataModule
from data.calvin_dataset_video import DiskCalvinVideoDataset
import functools
from data.data_utils import preprocess_image, get_prompt_builder, tcp_to_world_frame
from queue import Queue
from model.policy_head.action_tokenizer import ActionTokenizer

fwd_decay_ratio = 1

class CustomModel(CalvinBaseModel):
    # model option
    def __init__(self,
                 ckpt_path,
                 configs,
                 device,
                 save_dir=None,
                 raw_calvin=False,
                 debug=False,
                 action_ensemble=False):
        
        self.model = FlamingoTrainer(configs=configs)
        self.init_config(ckpt_path, configs, device, save_dir, raw_calvin, debug)
        # self.model.model.lm_head.window_size = 1

    def init_config(self, ckpt_path, configs, device, save_dir=None, raw_calvin=False, debug=False):
        ### load and convert checkpoint
        self.debug = debug
        if not self.debug:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            if 'state_dict' in ckpt:
                new_state_dict = ckpt['state_dict']
            elif 'model_state_dict' in ckpt:
                new_state_dict = ckpt['model_state_dict']
            else:
                raise KeyError("no checkpoint dict in the loaded pretrain parameters")

            new_state_dict = self.convert_old_state_dict(new_state_dict)
            msg = self.model.load_state_dict(new_state_dict, strict=False)
            print(f"CKPT Loaded \n {msg}")

            ckpt_dir = os.path.dirname(ckpt_path)
            ckpt_name = os.path.basename(ckpt_path)
            save_dir = ckpt_dir if save_dir is None else save_dir
            load_info_path = os.path.join(save_dir, f"{ckpt_name}_loading_msg.json")
            if os.path.exists(load_info_path):
                os.system(f"sudo rm {load_info_path}")
            with open(load_info_path, 'w') as f:
                _info = {
                    "missing_keys": msg.missing_keys,
                    "unexpected_keys": msg.unexpected_keys
                }
                json.dump(_info, f, indent=2)
                print(f"Model loading msg is updated to: {load_info_path}")
        
        self.configs = configs
        
        dtype = torch.float32
        if self.configs['trainer']['precision'] == 'bf16':
            dtype = torch.bfloat16
        elif self.configs['trainer']['precision'] == 'fp16':
            dtype = torch.float16
        self.dtype = dtype
        self.act_head_configs = self.configs['act_head']
        self.raw_calvin = raw_calvin
        self.tcp_rel = self.configs.get('tcp_rel', False)
        
        print(f"raw action: {self.raw_calvin}")

        self.device = device
        self.policy = self.model
        self.policy = self.policy.to(self.dtype)
        # self.policy = self.policy.float()
        self.policy.to(self.device)
        self.policy.eval()

        if not hasattr(self.policy.model, 'lm_head'):
            self.policy.model.lm_head = self.policy.model.act_head
        # self.clip_preprocess = self.policy.model.clip_preprocess

        self.tokenizer = build_tokenizer(self.configs['tokenizer'])

        self.window_size = configs['window_size']
        self.fwd_pred_next_n = configs['fwd_pred_next_n']
        self.seq_len = self.configs['seq_len']
        self.use_hand_rgb = self.configs['use_hand_rgb']
        import clip
        if self.configs["model"] == 'qwen':
            clip_preprocess = self.model.model.qwen_model.transformer.visual.image_transform
        else:
            if self.configs['image_size'] == 336:
                _, clip_preprocess = clip.load('ViT-L/14@336px', device='cpu')
            else:
                _, clip_preprocess = clip.load('ViT-B/32', device='cpu')
        self.datamodule = GRDataModule(
            configs['train_dataset'],
            configs['val_dataset'],
            configs['batch_size'],
            num_workers=1,
            tokenizer=configs['tokenizer'],
            fwd_pred_next_n=configs['fwd_pred_next_n'],
            window_size=configs['window_size'],
            # image_fn=functools.partial(preprocess_image, image_processor=clip_preprocess)
            image_size=configs['image_size'],
            image_fn=functools.partial(preprocess_image, image_processor=clip_preprocess),
            discrete_action=configs['act_head']['action_space'] == 'discrete',
            use_mu_law=configs.get('use_mu_law', False),
            mu_val=configs.get('mu_val', 255),
            n_bin=configs['act_head'].get('n_bin', 256),
            min_action=configs['act_head'].get('min_action', -1),
            max_action=configs['act_head'].get('max_action', 1),
            discrete_action_history=configs.get('discrete_action_history', False),
            act_step=configs.get('fwd_pred_next_n', 1),
        )
        print(f"--------------Evaluating DataMoudle--------------")

        self.datamodule.initialize('train')
        self.dataset = self.datamodule.train_datasets()
        if isinstance(self.dataset, (list, tuple)):
            self.dataset = self.dataset[0]
        if hasattr(self.dataset, 'datasets'):
            self.dataset = self.dataset.datasets[0]
        self.text_preprocess = self.dataset.text_fn
        val_dataset_cfg = self.configs['val_dataset']
        self.c_scaler = self._get_c_scaler(val_dataset_cfg)

        self.image_preprocess = self.dataset.image_fn
        self.action_space = self.configs['act_head'].get('action_space', 'continuous')
        if self.action_space == "discrete":
            self.action_tokenizer = ActionTokenizer(
                self.tokenizer, bins=self.act_head_configs['n_bin'], \
                min_action=self.act_head_configs['min_action'], max_action=self.act_head_configs['max_action'])
            
        # input_size = (self.configs['image_size'], self.configs['image_size'])
        # image_mean = self.configs['image_mean']
        # image_std = self.configs['image_std']
        # self.image_preprocess = T.Compose([
        #     T.Resize(input_size, interpolation=Image.BICUBIC),
        #     T.Normalize(image_mean, image_std)])

        print(f"Evaluating checkpoint {ckpt_path}")

        self.rgb_list = []
        self.hand_rgb_list = []
        self.action_hist_list = []
        self.rollout_step_counter = 0

    def ensemble_action(self, action):
        if action.ndim >= 3:
            action = action.squeeze()

        if action.ndim == 1:
            action = action.unsqueeze(0)
        
        self.action_hist_list.append(action)
        
        act_cache = []
        # self.fwd_pred_next_n = 1
        max_len = self.fwd_pred_next_n
        # if self.tcp_rel:
        #     max_len = 1
        while len(self.action_hist_list) > max_len:
            self.action_hist_list.pop(0)
        
        idx = 0
        for act in self.action_hist_list[::-1]:
            # print(act.shape)
            act_cache.append(act[idx])
            idx += 1
        
        act_cache = torch.stack(act_cache, dim=0)
        
        weights = torch.tensor([fwd_decay_ratio ** i for i in range(len(act_cache))])
        weights = weights / weights.sum()
        
        weighted_act = (act_cache * weights.unsqueeze(1)).sum(dim=0)
        
        return weighted_act

    @staticmethod
    def convert_old_state_dict(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_k = k.replace('module.', '')
            else:
                new_k = k

            if not new_k.startswith('model.'):
                new_k = 'model.' + new_k
            
            new_state_dict[new_k] = state_dict[k]
        return new_state_dict

    def _get_c_scaler(self, val_dataset_cfg):
        calvin_configs = self._get_default_calvin_config()
        default_c_act_scaler = calvin_configs['c_act_scaler']
        if isinstance(val_dataset_cfg, (list, tuple)):
            for d_cfg in val_dataset_cfg:
                c_scaler = self._get_c_scaler(d_cfg)
                if c_scaler is not None:
                    return c_scaler
        elif val_dataset_cfg['type'] == 'ConcatDataset':
            for cfg in self.configs['val_dataset']['datasets']:
                if 'calvin' in cfg['data_dir'].lower():
                    calvin_configs = cfg
                    c_scaler = calvin_configs.get('c_act_scaler', default_c_act_scaler)
                    return c_scaler
        elif val_dataset_cfg['type'] == 'GRDataset':
            if 'calvin' in val_dataset_cfg['data_dir'].lower():
                c_scaler = val_dataset_cfg.get('c_act_scaler', default_c_act_scaler)
                return c_scaler

        import warnings
        warnings.warn(f"No calvin dataset is found in the val config. Use default c scaler: {default_c_act_scaler}")
        return default_c_act_scaler

    def _get_default_calvin_config(self):
        return {
            "type": "GRDataset",
            "data_dir": "/mnt/bn/robotics-real-data/gr2_labels_1110/CALVIN/task_ABCD_D/val",
            "c_act_scaler": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        }

    def _preprocess_input(self, input_dict):
        return self.dataset.collater((input_dict,))
    
    def preprocess(self, obs, lang, mode='continuous'):
        # preprocess image
        image = obs["rgb_obs"]['rgb_static']
        image = Image.fromarray(image)
        image_x = self.image_preprocess([image]).unsqueeze(0)
        
        # image = Image.fromarray(image)
        # image_x = self.image_preprocess(T.ToTensor()(image.convert('RGB'))).unsqueeze(0).unsqueeze(0)
        
        gripper = obs["rgb_obs"]['rgb_gripper']
        gripper = Image.fromarray(gripper)
        gripper_x = self.image_preprocess([gripper]).unsqueeze(0)
        
        # gripper = Image.fromarray(gripper)
        # gripper_x = self.image_preprocess(T.ToTensor()(gripper.convert('RGB'))).unsqueeze(0).unsqueeze(0)
        # expand image dimension
        # image_x = image_x.unsqueeze(1).unsqueeze(1)
        # gripper_x = gripper_x.unsqueeze(1).unsqueeze(1)
        # expand text dimension
        if mode == 'discrete':
            if 'llava' in self.policy.configs:
                model_name = self.policy.configs['llava']
            elif 'qwen' in self.policy.configs:
                model_name = 'qwen'
            else:
                model_name = self.policy.configs['llm']['pretrained_model_name_or_path']
            
            prompt_builder = get_prompt_builder(model_name, self.tokenizer.bos_token, self.tokenizer.eos_token)
            
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}?"\
                if self.fwd_pred_next_n == 1 else f"What {self.fwd_pred_next_n} step actions should the robot take to {lang}?"},
                {"from": "gpt", "value": ""},
            ]

            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])

            input_ids = torch.tensor(list(self.tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids))
            text_x = input_ids[:-1].unsqueeze(0)
            mask = torch.full((1, text_x.shape[-1]), True, dtype=torch.bool)
        else:
            text_x, mask = self.text_preprocess([lang])

        return image_x.to(self.device).to(self.dtype), gripper_x.to(self.device).to(self.dtype), text_x.to(self.device), mask.to(self.device)
    
    def step(self, obs, goal):
        """Step function."""
        input_dict = dict()
        image_x, gripper_x, text_x, mask = self.preprocess(obs, goal, self.action_space)
        
        input_dict["rgb"] = image_x
        input_dict["hand_rgb"] = gripper_x
        input_dict["text"] = text_x
        input_dict['text_mask'] = mask
        
        with torch.no_grad():
            action = self.policy.inference_step(input_dict)['action']
        
        if self.action_space != 'discrete':
            if action[0].ndim == action[1].ndim + 1:
                action = (action[0], action[1].unsqueeze(2))
            action = torch.cat([action[0], (torch.nn.functional.sigmoid(action[1]) > 0.5).float()], dim=-1)
       
        # action = action[0, 0, 0] # batch, seq_len, chunck_idx

        if isinstance(action, tuple):
            action = torch.cat([action[0], action[1]], dim=-1)
        
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
        
        if action.ndim == 2:
            action = action.unsqueeze(1)
        
        if action.ndim == 3:
            action = action.unsqueeze(1)

        action = action.detach().cpu()

        if self.tcp_rel:
            robot_obs = torch.from_numpy(obs['robot_obs']).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1, 1, self.fwd_pred_next_n, 1)
            action = tcp_to_world_frame(action, robot_obs)

        action = self.ensemble_action(action)

        # if self.tcp_rel:
        #     robot_obs = torch.from_numpy(obs['robot_obs']).unsqueeze(0).unsqueeze(0)
        #     # print(action.shape)
        #     # print(robot_obs.shape)
        #     action = action.unsqueeze(0).unsqueeze(0)
        #     action = tcp_to_world_frame(action, robot_obs)

        if isinstance(action, torch.Tensor):
            action = action.squeeze()
            if action.ndim == 2:
                action = action[0]
            # action = action.numpy()

        if self.configs.get('use_mu_law', False):
            from data.data_utils import inverse_mu_law_companding
            action = inverse_mu_law_companding(action, self.configs.get('mu_val', 255), maintain_last=True)
        
        if self.configs.get('norm_action', False):
            from data.data_utils import unnoramalize_action
            if isinstance(action, tuple):
                action = (unnoramalize_action(action[0], self.configs['norm_min'], self.configs['norm_max']), action[1])
            else:
                action = unnoramalize_action(action, self.configs['norm_min'], self.configs['norm_max'])

        if self.action_space == 'discrete':
                # action[-1] = 1 if action[-1] > 0 else -1
                pass
        else:
            if self.raw_calvin:
                action[-1] = (action[-1] - 0.5) * 2
            else:
                state = obs['robot_obs'] # (15,)
                xyz_state = state[:3]
                rpy_state = state[3:6]
                rotm_state = euler2rotm(rpy_state)
                rel_action = action.numpy()
                _c_rel_action = rel_action[:6] / self.c_scaler
                xyz_action = _c_rel_action[:3]
                rpy_action = _c_rel_action[3:6]
                # xyz_action = rel_action[:3] / 50 # scale down by 50
                # rpy_action = rel_action[3:6] / 33 # scale down by 20
                gripper_action = rel_action[6]
                rotm_action = euler2rotm(rpy_action)
                xyz_next_state = xyz_state + rotm_state @ xyz_action
                rotm_next_state = rotm_state @ rotm_action
                rpy_next_state = rotm2euler(rotm_next_state)

                action = action.numpy()
                action[:3] = xyz_next_state - xyz_state
                action[3:6] = rpy_next_state - rpy_state
                action[:6] *= [50.0, 50.0, 50.0, 20., 20., 20.]
                action[-1] = (gripper_action - 0.5) * 2
                action = torch.from_numpy(action)
                
        self.rollout_step_counter += 1
        action[-1] = 1 if action[-1] > 0 else -1
        print(f"step {self.rollout_step_counter} action {action}")
        return action

    def reset(self):
        
        if hasattr(self.model.model, 'lm_head'):
            self.model.model.lm_head.hidden_state = None
            self.model.model.lm_head.history_memory = []
        if hasattr(self.model.model, 'act_head'):
            self.model.model.act_head.hidden_state = None
            self.model.model.act_head.history_memory = []

        self.rgb_list = []
        self.hand_rgb_list = []
        self.rollout_step_counter = 0
        self.action_hist_list = []

class CustomModelFlamingoVideo(CustomModel):

    def __init__(
            self,
            ckpt_path,
            configs,
            device,
            save_dir=None,
            raw_calvin=False,
            debug=False):
        self.model = FlamingoVideoTrainer(configs=configs)
        self.init_config(ckpt_path, configs, device, save_dir, raw_calvin, debug)
        self.vision_queue = Queue(maxsize=self.window_size)
        self.vision_gripper_queue = Queue(maxsize=self.window_size)
        self.action_queue = Queue(maxsize=self.window_size-1)

    def add_element_to_queue(self, q: Queue, element):
        while q.qsize() >= q.maxsize:
            q.get()
        q.put(element)

    def get_history(self, q: Queue, pad: Literal["zero", 'first']="zero"):
        queue_list = list(q.queue)
        if len(queue_list) == 0:
            return queue_list, None
        history_type = self.configs['act_head'].get('history_type', 'pre')
        if history_type == 'pre':
            pad_len = 0
        elif history_type == 'video':
            pad_len = q.maxsize - len(queue_list)
        else:
            raise ValueError(f"Unsupported history type {history_type}")
        element = queue_list[0]
        if pad == "zero":
            if isinstance(element, torch.Tensor):
                element = torch.zeros_like(element)
            elif isinstance(element, np.ndarray):
                element = np.zeros_like(element)
            else:
                raise ValueError("This type is not supported")
            queue_list = [element for _ in range(pad_len)] + queue_list
        else:
            if isinstance(element, torch.Tensor):
                pad_list = [element.clone() for _ in range(pad_len)]
            elif isinstance(element, np.ndarray):
                pad_list = [deepcopy(element) for _ in range(pad_len)]
            queue_list = pad_list + queue_list
        pad_mask = np.ones(q.maxsize, dtype=bool)
        pad_mask[:pad_len] = False
        return queue_list, pad_mask
            
    def preprocess(self, obs, lang, mode='continuous'):
        # preprocess image
        image = obs["rgb_obs"]['rgb_static']
        image = Image.fromarray(image)
        image_x = self.image_preprocess([image]).unsqueeze(0)
        
        gripper = obs["rgb_obs"]['rgb_gripper']
        gripper = Image.fromarray(gripper)
        gripper_x = self.image_preprocess([gripper]).unsqueeze(0)
        
        # self.vision_queue.put(image_x)
        # self.vision_gripper_queue.put(gripper_x)
        self.add_element_to_queue(self.vision_queue, image_x)
        self.add_element_to_queue(self.vision_gripper_queue, gripper_x)
        
        image_x, _ = self.get_history(self.vision_queue, pad="first")
        gripper_x, _ = self.get_history(self.vision_gripper_queue, pad="first")
        image_x = torch.concatenate(image_x, dim=1)
        gripper_x = torch.concatenate(gripper_x, dim=1)
        action, action_mask = self.get_history(self.action_queue)
        if len(action) != 0:
            action = np.stack(action, axis=0)
        else:
            action = np.zeros((self.window_size, 7))
            action_mask = np.zeros(action.shape[0], dtype=bool)
        if mode == 'discrete':
            if 'llava' in self.policy.configs:
                model_name = self.policy.configs['llava']
            else:
                model_name = self.policy.configs['llm']['pretrained_model_name_or_path']
            discrete_action_history = self.act_head_configs.get('discrete_action_history', False) or self.configs.get('discrete_action_history', False)
            predict_stop_token = self.act_head_configs.get('predict_stop_token', False) or self.configs.get('predict_stop_token', False)
            text_x, _ = DiskCalvinVideoDataset.static_wrap_instruction_and_action(
                lang, action, action_mask, model_name, self.tokenizer, 
                self.action_tokenizer, self.fwd_pred_next_n, action.shape[0] + 1,
                discrete_action_history, predict_stop_token, "inference"
            )
            text_x = text_x.unsqueeze(0)
            mask = torch.ones(text_x.shape[0], dtype=torch.bool)
        else:
            text_x, mask = self.text_preprocess([lang])

        return image_x.to(self.device).to(self.dtype), gripper_x.to(self.device).to(self.dtype), text_x.to(self.device), mask.to(self.device)
    
    def val(self):
        trainer_config = init_trainer_config(self.configs)
        trainer_config['devices'] = 1
        import pdb; pdb.set_trace()
        trainer = Trainer(**trainer_config)
        trainer.validate(self.policy, self.datamodule.train_dataloader())

    def step(self, obs, goal):
        input_dict = dict()
        image_x, gripper_x, text_x, mask = self.preprocess(obs, goal, self.action_space)
        input_dict["rgb"] = image_x
        input_dict["hand_rgb"] = gripper_x
        input_dict['text'] = text_x
        input_dict['text_mask'] = mask
        if self.action_space == "discrete":
            input_dict["instr_and_action_ids"] = text_x
            input_dict['instr_and_action_mask'] = mask

        with torch.no_grad():
            action = self.policy.inference_step(input_dict)['action']
        
        if self.action_space != 'discrete':
            if action[0].ndim == action[1].ndim + 1:
                action = (action[0], action[1].unsqueeze(2))
            action = torch.cat([action[0], (torch.nn.functional.sigmoid(action[1]) > 0.5).float()], dim=-1)
        # action = action[0, 0, 0] # batch, seq_len, chunck_idx
       
        # if isinstance(action, torch.Tensor):
        #     action = action.squeeze()
        #     if action.ndim == 2:
        #         action = action[0]

        if isinstance(action, torch.Tensor): 
            if action.ndim == 3:
                action = action.unsqueeze(1) # assume seq_len=1
            action = action[0, -1, 0]
            action = action.detach().cpu()

        if self.configs.get('use_mu_law', False):
            from data.data_utils import inverse_mu_law_companding
            if isinstance(action, tuple):
                action = (inverse_mu_law_companding(action[0], self.configs['norm_min'], self.configs['norm_max']), action[1])
            else:
                action = inverse_mu_law_companding(action, self.configs.get('mu_val', 255), maintain_last=True)
        
        if self.configs.get('norm_action', False):
            from data.data_utils import unnoramalize_action
            if isinstance(action, tuple):
                action = (unnoramalize_action(action[0], self.configs['norm_min'], self.configs['norm_max']), action[1])
            else:
                action = unnoramalize_action(action, self.configs['norm_min'], self.configs['norm_max'], maintain_last=True)

        if self.action_space == 'discrete':
                action[-1] = 1 if action[-1] > 0 else -1
        else:
            # if action[0].ndim == action[1].ndim + 1:
            #     action = (action[0], action[1].unsqueeze(2))
            # action = torch.cat([action[0], (torch.nn.functional.sigmoid(action[1]) > 0.5).float()], dim=-1)
            # # action = action[0, 0, 0] # batch, seq_len, fwd_next_n
            # action = action.squeeze()
            # if action.ndim == 2:
            #     action = action[0]
            # action = action.detach().cpu()

            if self.raw_calvin:
                action[-1] = (action[-1] - 0.5) * 2
            else:
                state = obs['robot_obs'] # (15,)
                xyz_state = state[:3]
                rpy_state = state[3:6]
                rotm_state = euler2rotm(rpy_state)
                rel_action = action.numpy()
                _c_rel_action = rel_action[:6] / self.c_scaler
                xyz_action = _c_rel_action[:3]
                rpy_action = _c_rel_action[3:6]
                # xyz_action = rel_action[:3] / 50 # scale down by 50
                # rpy_action = rel_action[3:6] / 33 # scale down by 20
                gripper_action = rel_action[6]
                rotm_action = euler2rotm(rpy_action)
                xyz_next_state = xyz_state + rotm_state @ xyz_action
                rotm_next_state = rotm_state @ rotm_action
                rpy_next_state = rotm2euler(rotm_next_state)

                action = action.numpy()
                action[:3] = xyz_next_state - xyz_state
                action[3:6] = rpy_next_state - rpy_state
                action[:6] *= [50.0, 50.0, 50.0, 20., 20., 20.]
                action[-1] = (gripper_action - 0.5) * 2
                action = torch.from_numpy(action)

        print(f"step {self.rollout_step_counter}, action {action}")
        
        self.rollout_step_counter += 1
        self.add_element_to_queue(self.action_queue, action)
        return action
    
    def reset(self):
        super().reset()
        while not self.vision_queue.empty():
            self.vision_queue.get()
        while not self.vision_gripper_queue.empty():
            self.vision_gripper_queue.get()
        while not self.action_queue.empty():
            self.action_queue.get()

class CustomModelLLaVA(CustomModel):

    def __init__(
            self,
            ckpt_path,
            configs,
            device,
            save_dir=None,
            raw_calvin=False,
            debug=False):
        
        self.model = LLaVATrainer(configs=configs)
        self.init_config(ckpt_path, configs, device, save_dir, raw_calvin, debug)
    
    # def step(self, obs, goal):
    #     input_dict = dict()
    #     # import pdb; pdb.set_trace()
    #     image_x, gripper_x, text_x, mask = self.preprocess(obs, goal, self.action_space)
        
    #     input_dict["rgb"] = image_x
    #     input_dict["hand_rgb"] = gripper_x
    #     input_dict["text"] = text_x
    #     input_dict['text_mask'] = mask
        
    #     with torch.no_grad():
    #         if hasattr(self.policy.model.act_head, 'window_size'):
    #             window_size = self.policy.model.act_head.window_size
    #             self.policy.model.act_head.window_size = 1
    #         # self.policy.configs['window_size'] = 1
    #         action = self.policy.inference_step(input_dict)['action']
    #         if hasattr(self.policy.model.act_head, 'window_size'):
    #             self.policy.model.act_head.window_size = window_size
    #             # self.policy.configs['window_size'] = window_size
        
    #      if self.action_space != 'discrete':
    #         if action[0].ndim == action[1].ndim + 1:
    #             action = (action[0], action[1].unsqueeze(2))
    #         action = torch.cat([action[0], (torch.nn.functional.sigmoid(action[1]) > 0.5).float()], dim=-1)
    #     # action = action[0, 0, 0] # batch, seq_len, chunck_idx
    #     action = action.squeeze()
    #     if action.ndim == 2:
    #         action = action[0]
        
    #     action = action.detach().cpu()
        
    #     if self.configs.get('norm_action', False):
    #         from data.data_utils import unnoramalize_action
    #         action = unnoramalize_action(action, self.configs['norm_min'], self.configs['norm_max'], maintain_last=True)
    #         # if isinstance(action, tuple):
    #         #     action = (unnoramalize_action(action[0], self.configs['norm_min'], self.configs['norm_max']), action[1])
    #         # else:
    #         #     action = unnoramalize_action(action, self.configs['norm_min'], self.configs['norm_max'])

    #     if self.configs.get('use_mu_law', False):
    #         from data.data_utils import inverse_mu_law_companding
    #         action = inverse_mu_law_companding(action, self.configs.get('mu_val', 255))

    #     if self.action_space == 'discrete':
    #             action[-1] = 1 if action[-1] > 0 else -1
    #     else:
    #         # if action[0].ndim == action[1].ndim + 1:
    #         #     action = (action[0], action[1].unsqueeze(2))
    #         # action = torch.cat([action[0], (torch.nn.functional.sigmoid(action[1]) > 0.5).float()], dim=-1)
    #         # action = action[0, 0, 0] # batch, seq_len, chunck_idx
    #         # action = action.squeeze()
    #         # if action.ndim == 2:
    #         #     action = action[0]
    #         # action = action.detach().cpu()

    #         if self.raw_calvin:
    #             action[-1] = (action[-1] - 0.5) * 2
    #         else:
    #             state = obs['robot_obs'] # (15,)
    #             xyz_state = state[:3]
    #             rpy_state = state[3:6]
    #             rotm_state = euler2rotm(rpy_state)
    #             rel_action = action.numpy()
    #             _c_rel_action = rel_action[:6] / self.c_scaler
    #             xyz_action = _c_rel_action[:3]
    #             rpy_action = _c_rel_action[3:6]
    #             # xyz_action = rel_action[:3] / 50 # scale down by 50
    #             # rpy_action = rel_action[3:6] / 33 # scale down by 20
    #             gripper_action = rel_action[6]
    #             rotm_action = euler2rotm(rpy_action)
    #             xyz_next_state = xyz_state + rotm_state @ xyz_action
    #             rotm_next_state = rotm_state @ rotm_action
    #             rpy_next_state = rotm2euler(rotm_next_state)

    #             action = action.numpy()
    #             action[:3] = xyz_next_state - xyz_state
    #             action[3:6] = rpy_next_state - rpy_state
    #             action[:6] *= [50.0, 50.0, 50.0, 20., 20., 20.]
    #             action[-1] = (gripper_action - 0.5) * 2
    #             action = torch.from_numpy(action)

    #     print(f"step {self.rollout_step_counter}, action {action}")
        
    #     self.rollout_step_counter += 1
    #     return action


class CustomModelLLaVaVideo(CustomModelFlamingoVideo):
    def __init__(
            self,
            ckpt_path,
            configs,
            device,
            save_dir=None,
            raw_calvin=False,
            debug=False):
        
        self.model = LLaVATrainer(configs=configs)
        self.init_config(ckpt_path, configs, device, save_dir, raw_calvin, debug)
        self.vision_queue = Queue(maxsize=self.window_size)
        self.vision_gripper_queue = Queue(maxsize=self.window_size)
        self.action_queue = Queue(maxsize=self.window_size-1)
    


class CustomModelQwen(CustomModelLLaVA):

    def __init__(
            self,
            ckpt_path,
            configs,
            device,
            save_dir=None,
            raw_calvin=False,
            debug=False):
        
        self.model = QwenTrainer(configs=configs)
        self.init_config(ckpt_path, configs, device, save_dir, raw_calvin, debug)