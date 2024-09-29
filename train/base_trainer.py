import time
import warnings
import os

import numpy as np
import torch
import torch.nn.functional as F
from functools import partial
import math
import torch.distributed as dist
from utils.dist_train import get_rank

import lightning.pytorch as pl

from train.train_utils import adjust_learning_rate

class BaseTrainer(pl.LightningModule):
    def __init__(self, configs):
        super().__init__()
        self._main_rank_print('--------------- model configs ---------------')
        self._main_rank_print(configs)
        self.configs = configs
        self._initialize()
        self.save_hyperparameters()
        if isinstance(configs['val_dataset'], list):
            self.val_set_names = [self._parse_dataset_name(cfg) for cfg in configs['val_dataset']]
        elif isinstance(configs['val_dataset'], dict):
            # FIXME: hotfix
            self.val_set_names = None
        else:
            raise NotImplementedError

    def _parse_dataset_name(self, dataset_config):
        dataset_path = dataset_config['data_dir']
        avail_dataset = ['mode1', 'mode3', 'bridge', 'rt-1', 'ego4d', 'calvin']
        for name in avail_dataset:
            if name in dataset_path.lower():
                return name
        return 'UNKNOWN_DATA'

    @staticmethod
    def _main_rank_print(*args, **kwargs):
        if get_rank() == 0:
            print(*args, **kwargs)

    @property
    def num_gpus(self):
        return self.trainer.num_devices * self.trainer.num_nodes

    def _initialize(self):
        self.use_hand_rgb = self.configs['use_hand_rgb']
        self.use_multi_modal_emb = self.configs['use_multi_modal_emb']
        self.finetune = self.configs['finetune']
        self.no_video_pretrained_model = self.configs['no_video_pretrained_model']

        self.model = self._init_policy()
        self.cap_loss_ratio = self.configs['cap_loss_ratio']
        self.arm_gripper_loss_ratio = self.configs['arm_gripper_loss_ratio']
        self.fwd_loss_ratio = self.configs['fwd_loss_ratio']
        self.kl_div_ratio = self.configs.get('kl_div_ratio', 0.05)
        self.clip_norm_ratio = self.configs.get('clip_norm_ratio', 0.05)
        
        self.vl_cotrain_ratio = self.configs.get('vl_cotrain_ratio', 0.05)
        if self.configs['act_head'] is not None and self.configs['act_head']['action_space'] == 'discrete':
            self.vl_cotrain_ratio *= 10

        self.act_pred = self.configs['train_setup']['predict_action']
        self.fwd_pred = self.configs['train_setup']['predict_forward']
        self.fwd_pred_hand = self.configs['train_setup']['predict_forward_hand']
        self.cap_pred = self.configs['train_setup']['predict_caption']
        # self.use_hand_rgb = self.model.use_hand_rgb

        # Make sure that at least one prediction flag is on
        assert self.act_pred or self.fwd_pred or self.cap_pred

        self.start_time = time.time()

    @staticmethod
    def get_converted_fp32_paths(deepspeed_ckpt_path):
        deepspeed_ckpt_path = deepspeed_ckpt_path.rstrip('/')
        ckpt_dir = os.path.dirname(deepspeed_ckpt_path)
        ckpt_name = os.path.basename(deepspeed_ckpt_path)
        fp32_ckpt_name = f"{ckpt_name}.fp32.pt"
        converted_path = os.path.join(ckpt_dir, fp32_ckpt_name)
        return converted_path

    @classmethod
    def from_checkpoint(cls, ckpt_path=None, ckpt_source='torch', configs=None):
        if ckpt_path is None:
            assert configs is not None, "ckpt_path and configs are both None for initialization."
            return cls(configs)
        
        if os.path.isdir(ckpt_path):
            ckpt_source = 'deepspeed'
        
        if ckpt_source == 'torch':
            assert configs is not None, \
                "to initialize the model with a torch pretrained state dict, " \
                "you need to specify the configs for initialization."
            model = cls(configs)
            checkpoint = torch.load(configs['model_load_path'], map_location='cpu')
            
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                raise KeyError(f"The checkpoint must has state_dict or model_state_dict as key")
            
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            del checkpoint
            from train.train_utils import convert_old_state_dict
            state_dict = convert_old_state_dict(state_dict)
            msg = model.load_state_dict(state_dict, strict=False)
            cls._main_rank_print(msg)
            del state_dict
            return model

        if ckpt_source == 'lightning':
            checkpoint = torch.load(ckpt_path, map_location='cpu')

            model = cls(configs)
            msg = model.load_state_dict(checkpoint['state_dict'], strict=False)
            cls._main_rank_print(msg)
            return model

        if ckpt_source == 'deepspeed':
            # FIXME: currently I don't find a proper way to load sharded DeepSpeed model using pytorch lightning.
            #   Here the solution is to convert the DeepSpeed model to FP32, then just load it as a torch model.

            # convert deepspeed checkpoint to lightning
            coverted_path = cls.get_converted_fp32_paths(ckpt_path)
            
            if not os.path.exists(coverted_path):
                if dist.get_rank() == 0:
                    from utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
                    print(f"converting {ckpt_path} to {coverted_path}")
                    convert_zero_checkpoint_to_fp32_state_dict(ckpt_path, coverted_path)
                dist.barrier()
            
            # assert os.path.exists(coverted_path), \
            #     "Please use tools/convert_deepspeed_to_fp32.py [DEEPSPEED_CKPT]" \
            #     "for checkpoint conversion first."

            # remove unpretrained params
            cls._main_rank_print(f"Loading pretrained model from {coverted_path}...")
            checkpoint = torch.load(coverted_path, map_location='cpu')

            model = cls(configs)
            msg = model.load_state_dict(checkpoint['state_dict'], strict=False)
            cls._main_rank_print(msg)
            return model

        raise NotImplementedError("Unknown source of checkpoints. Legal choices: torch, lightning, or deepspeed.")

    def configure_optimizers(self):
        eff_batch_size = self.configs['batch_size'] * self.num_gpus * (self.configs['seq_len'] - 1)
        eff_lr = self.configs['learning_rate']
        self._main_rank_print('-' * 40)
        self._main_rank_print("LR SCHEDULER CONFIGS:")
        self._main_rank_print(f"effective batch size: {eff_batch_size}, effective learning rate: {eff_lr}")
        
        # optimizer = torch.optim.AdamW(self.get_grouped_params(self.model))
        optimizer = torch.optim.AdamW(
            self.get_grouped_params(self.model),
            lr=eff_lr
        )

        assert self.trainer.max_epochs is not None
        num_training_batches = self.trainer.estimated_stepping_batches
        iter_per_epoch = num_training_batches / self.trainer.max_epochs
        warmup_epochs = self.configs.get("warmup_epochs", 0)
        warmup_steps = self.configs.get("warmup_steps", 0)
        lr_scheduler_configs = {
            'warmup_iters': warmup_epochs * iter_per_epoch + warmup_steps,
            'iters': self.configs['trainer']['max_epochs'] * iter_per_epoch,
            'min_lr_scale': self.configs['min_lr_scale']
        }
        self._main_rank_print(lr_scheduler_configs)
        
        scheduler_type = self.configs.get('scheduler', 'constant')

        from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
        if scheduler_type == 'constant':
            scheduler = get_constant_schedule_with_warmup(optimizer, lr_scheduler_configs["warmup_iters"])
        elif scheduler_type == 'half-cosine':
            lr_lambda = partial(adjust_learning_rate, configs=lr_scheduler_configs)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        elif scheduler_type == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(optimizer,lr_scheduler_configs["warmup_iters"], num_training_steps=lr_scheduler_configs['iters'])
        else:
            raise NotImplementedError
            
        return {
            'optimizer': optimizer,
            'lr_scheduler':
                {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            }

    def _get_loss(self, prediction):
        # print(prediction)
        loss_arm_act = prediction.get('loss_arm_act', None)
        loss_gripper_act = prediction.get('loss_gripper_act', None)
        loss_obs = prediction.get('loss_obs_fwd', None)
        loss_hand_obs = prediction.get('loss_hand_obs_fwd', None)
        acc_arm_act = prediction.get('acc_arm_act', None)
        acc_gripper_act = prediction.get('acc_gripper_act', None)
        loss_cap = prediction.get('loss_cap', None)
        loss_kl = prediction.get('loss_kl', None)
        loss_vl_cotrain = prediction.get('loss_vl_cotrain', None)

        ### loss logout for discrete setting
        action_l1 = prediction.get('action_l1_act', None)

        clip_l1 = prediction.get('text_l1_clip', None)

        loss = torch.tensor(0.0).to(self.device)
        if self.act_pred:
            loss_act = (loss_arm_act if loss_arm_act is not None else 0) + (loss_gripper_act * self.arm_gripper_loss_ratio if loss_gripper_act is not None else 0)
            loss += loss_act
            if loss_kl is not None:
                loss += self.kl_div_ratio * loss_kl
            if clip_l1 is not None:
                loss += self.clip_norm_ratio * clip_l1
        else:
            loss_act = None

        if self.fwd_pred:
            loss += self.fwd_loss_ratio * (loss_obs if loss_obs is not None else 0)
            if self.fwd_pred_hand:
                loss += self.fwd_loss_ratio * (loss_hand_obs if loss_hand_obs is not None else 0)
        if loss_cap is not None:
            loss += self.cap_loss_ratio * loss_cap
        
        if loss_vl_cotrain is not None:
            loss += self.vl_cotrain_ratio * loss_vl_cotrain

        output = {
            'loss': loss,
            'loss_act': loss_act,
            'loss_arm_act': loss_arm_act,
            'loss_gripper_act': loss_gripper_act,
            'acc_arm_act': acc_arm_act,
            'acc_gripper_act': acc_gripper_act,
            'loss_obs': loss_obs,
            'loss_hand_obs': loss_hand_obs,
            "action_l1": action_l1,
            "loss_kl": loss_kl,
            "clip_l1": clip_l1,
            "loss_vl_cotrain": loss_vl_cotrain
        }

        return output

    def _log_output(self, output, phase, prog_bar_set=None, dataset=None, **kwargs):
        prog_bar_set = prog_bar_set or {}
        for k, v in output.items():
            if v is None:
                continue

            log_name = f"{phase}_{k}"
            if dataset is not None:
                log_name = f"{dataset}_{log_name}"

            if k in prog_bar_set:
                self.log(log_name, v, prog_bar=True, **kwargs)
            else:
                self.log(log_name, v, **kwargs)

    def _init_policy(self):
        raise NotImplementedError
    
    def _process_batch(self, batch):
        raise NotImplementedError
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        raise NotImplementedError
    
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
