import time
import warnings
import os

import numpy as np
import torch
import torch.nn.functional as F
from functools import partial
import math

from model.backbone.flamingo import RoboFlamingo
from train.base_trainer import BaseTrainer
from utils.dist_train import get_rank

import lightning.pytorch as pl

from train.train_utils import adjust_learning_rate


class FlamingoTrainer(BaseTrainer):
    def __init__(self, configs):
        super(FlamingoTrainer, self).__init__(configs)

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

        self.act_pred = self.configs['train_setup']['predict_action']
        self.fwd_pred = self.configs['train_setup']['predict_forward']
        self.fwd_pred_hand = self.configs['train_setup']['predict_forward_hand']
        self.cap_pred = self.configs['train_setup']['predict_caption']
        self.use_hand_rgb = self.model.use_hand_rgb

        # Make sure that at least one prediction flag is on
        assert self.act_pred or self.fwd_pred or self.cap_pred

        self.start_time = time.time()

    def _init_policy(self):
        model = RoboFlamingo(
            vision_encoder_configs=self.configs['vision_encoder'],
            tokenizer_configs=self.configs['tokenizer'],
            train_setup_configs=self.configs['train_setup'],
            fwd_head_configs=None,
            llm_configs=self.configs['llm'],
            window_size=self.configs['window_size'],
            use_hand_rgb=self.use_hand_rgb,
            act_head_configs=self.configs['act_head'],
            fwd_pred_next_n=self.configs['fwd_pred_next_n']
        )
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._main_rank_print(f"Model Parameters: {total_params / 1000000:.2f}M")
        return model

    @staticmethod
    def get_converted_fp32_paths(deepspeed_ckpt_path):
        deepspeed_ckpt_path = deepspeed_ckpt_path.rstrip('/')
        ckpt_dir = os.path.dirname(deepspeed_ckpt_path)
        ckpt_name = os.path.basename(deepspeed_ckpt_path)
        fp32_ckpt_name = f"{ckpt_name}.fp32.pt"
        converted_path = os.path.join(ckpt_dir, fp32_ckpt_name)
        return converted_path

    @staticmethod
    def _get_state_dict(checkpoint):
        assert isinstance(checkpoint, dict), \
            f"Checkpoint should be saved as a dict. Now it is a {type(checkpoint)}"
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            raise ValueError("No state dict is found in checkpoint. Make sure that it "
                             "has the key of 'state_dict' or 'model_state_dict'. "
                             f"Current keys in the checkpoint: {checkpoint.keys()}")
        del checkpoint
        return state_dict

    # def _convert_old_state_dict(self, state_dict):
    #     new_state_dict = {}
    #     for k, v in state_dict.items():
    #         if k.startswith('module.'):
    #             new_k = k.replace('module.', '')
    #         else:
    #             new_k = k
    #         new_state_dict[new_k] = state_dict[k]
    #     return new_state_dict

    @classmethod
    def from_checkpoint(cls, ckpt_path=None, ckpt_source='torch', configs=None):
        if ckpt_path is None:
            assert configs is not None, "ckpt_path and configs are both None for initialization."
            return cls(configs)

        if ckpt_source in {'torch', 'torch_old'}:
            assert configs is not None, \
                "to initialize the model with a torch pretrained state dict, " \
                "you need to specify the configs for initialization."
            model = cls(configs)
            checkpoint = torch.load(configs['model_load_path'], map_location='cpu')
            state_dict = cls._get_state_dict(checkpoint)
            state_dict = cls.convert_old_state_dict(state_dict)
            msg = model.load_state_dict(state_dict, strict=False)
            print(msg)
            cls._main_rank_print(msg)
            del state_dict
            return model

        if ckpt_source == 'lightning':
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            model = cls(configs)
            state_dict = cls._get_state_dict(checkpoint)
            msg = model.load_state_dict(state_dict, strict=False)
            print(msg)
            cls._main_rank_print(msg)
            return model

        if ckpt_source == 'deepspeed':
            # FIXME: currently I don't find a proper way to load sharded DeepSpeed model using pytorch lightning.
            #   Here the solution is to convert the DeepSpeed model to FP32, then just load it as a torch model.

            # convert deepspeed checkpoint to lightning
            coverted_path = cls.get_converted_fp32_paths(ckpt_path)
            assert os.path.exists(coverted_path), \
                "Please use tools/convert_deepspeed_to_fp32.py [DEEPSPEED_CKPT]" \
                "for checkpoint conversion first."

            # remove unpretrained params
            cls._main_rank_print(f"Loading pretrained model from {coverted_path}...")
            checkpoint = torch.load(coverted_path, map_location='cpu')

            model = cls(configs)
            state_dict = cls._get_state_dict(checkpoint)
            msg = model.load_state_dict(state_dict, strict=False)
            print(msg)
            cls._main_rank_print(msg)
            return model

        raise NotImplementedError("Unknown source of checkpoints. Legal choices: torch, lightning, or deepspeed.")

    def _process_batch(self, batch):
        """
        Action Prediction:
            args: rgb, language, attention_mask, hand_rgb, action
            reformat: action to input and target (seq_len = window size + chunck size)
        Video Prediction:
            args: rgb, language, attention mask, hand_rgb
            reformat: rgb, [hand_rgb] to input and target (seq_len = window size + chunck size)
        Video Caption:
            args: rgb, language, attention_mask
            reformat: Identity
        Image Caption:
            args: rgb, language, attention_mask
            reformat: Identity
            seq_len = 1
        """
        # import pdb; pdb.set_trace()
        # print(type(batch), len(batch))
        if isinstance(batch, list) or isinstance(batch, tuple):
            # print(batch[0].keys())
            batch = batch[0]
        
        rgb = batch['rgb'].to(self.device)
        if len(rgb.shape) == 4:
            rgb = rgb.unsqueeze(1)
        assert len(rgb.shape) == 5, f"current shape is {rgb.shape}"
        if isinstance(batch['text'], list) and isinstance(batch['text'][0], str):
            raise ValueError('The raw text data is not supported')
        else:
            # seq_len = batch['rgb'].shape[1]
            # if seq_len > 1:
            #     seq_len -= self.configs['fwd_pred_next_n']
            # seq_len = self.configs['window_size']
            seq_len = self.configs['window_size']
            language = batch['text'].cuda()
            text_mask = batch['text_mask'].cuda()
        
        if batch.get('action', None) is not None:
            action = batch['action'].to(self.device)
        else:
            action = None

        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = batch['attention_mask'].to(self.device)
        
        if self.use_hand_rgb and batch.get('hand_rgb', None) is not None:
            hand_rgb = batch['hand_rgb'].to(self.device)
        else:
            hand_rgb = None

        # Split arm and gripper action
        arm_action = None
        gripper_action = None
        if action is not None:
            arm_action = action[:, :, :6]  # b,len,act_dim-1
            gripper_action = action[:, :, 6]  # b,len
            gripper_action = (gripper_action + 1.0) / 2
            gripper_action = gripper_action.long()

        fwd_rgb_chunck = batch.get('fwd_rgb_chunck', None)
        fwd_hand_rgb_chunck = batch.get('fwd_hand_rgb_chunck', None)
        if fwd_rgb_chunck is not None:
            fwd_rgb_chunck = fwd_rgb_chunck.cuda()
        if fwd_hand_rgb_chunck is not None:
            fwd_hand_rgb_chunck = fwd_hand_rgb_chunck.cuda()

        arm_action_chunck = None
        gripper_action_chunck = None
        action_chunck = batch.get('action_chunck', None)
        if action_chunck is not None:
            action_chunck = action_chunck.to(self.device)
            arm_action_chunck = action_chunck[..., :6]
            gripper_action_chunck = action_chunck[..., -1]
        
        # fwd_pred_next_n = self.configs['fwd_pred_next_n']
        # 
        # caption_flag = False
        # if seq_len == 1:
        #     caption_flag = True
        # if not caption_flag:
        #     rgb = rgb[:, :-fwd_pred_next_n]
        #     if hand_rgb is not None:
        #         hand_rgb = hand_rgb[:, :-fwd_pred_next_n]
        
        rgb = rgb[:, :seq_len]
        if hand_rgb is not None:
            hand_rgb = hand_rgb[:, :seq_len]
        
        chunck_mask = batch.get('chunck_mask', None)
        if chunck_mask is not None:
            chunck_mask = chunck_mask.to(self.device)

        # data preparation for discrete action inputs and labels
        instr_and_action_ids = batch.get("instr_and_action_ids", None)
        if instr_and_action_ids is not None:
            instr_and_action_ids = instr_and_action_ids.cuda()
        
        instr_and_action_labels = batch.get("instr_and_action_labels", None)
        if instr_and_action_labels is not None:
            instr_and_action_labels = instr_and_action_labels.cuda()
        
        instr_and_action_mask = batch.get("instr_and_action_mask", None)
        if instr_and_action_mask is not None:
            instr_and_action_mask = instr_and_action_mask.cuda()

        return rgb, hand_rgb, attention_mask, language, text_mask, fwd_rgb_chunck, fwd_hand_rgb_chunck,\
        arm_action, gripper_action, arm_action_chunck, gripper_action_chunck, chunck_mask, instr_and_action_ids, instr_and_action_labels, instr_and_action_mask

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

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        with torch.no_grad():
            rgb, hand_rgb, attention_mask, language, text_mask, fwd_rgb_chunck, fwd_hand_rgb_chunck,\
            arm_action, gripper_action, arm_action_chunck, gripper_action_chunck, chunck_mask, instr_and_action_ids, instr_and_action_labels, instr_and_action_mask \
            = self._process_batch(batch)

            prediction = self.model.forward(
                rgb, 
                language,
                attention_mask=text_mask, 
                action_labels=(arm_action_chunck, gripper_action_chunck),
                action_mask=chunck_mask,
                # FIXME: hardcode here
                caption_labels=language.clone() if rgb.shape[1] == 1 else None,
                caption_mask=text_mask.clone() if rgb.shape[1] == 1 else None,
                vision_gripper=hand_rgb,
                fwd_rgb_labels=fwd_rgb_chunck,
                fwd_hand_rgb_chunck=fwd_hand_rgb_chunck,
                fwd_mask=chunck_mask,
                instr_and_action_ids=instr_and_action_ids, 
                instr_and_action_labels=instr_and_action_labels, 
                instr_and_action_mask=instr_and_action_mask
            )

            output = self._get_loss(prediction)
            
            prog_bar_set = {'loss'}
            if self.act_pred:
                prog_bar_set.add('loss_arm_act')
                prog_bar_set.add('loss_gripper_act')
                prog_bar_set.add('acc_gripper_act')
            if self.fwd_pred:
                prog_bar_set.add('loss_obs')
                if self.fwd_pred_hand:
                    prog_bar_set.add('loss_hand_obs')

            dataset = None
            if self.val_set_names is not None:
                dataset = self.val_set_names[dataloader_idx]

            self._log_output(output, phase="val", prog_bar_set=prog_bar_set,
                             sync_dist=True, on_epoch=True, on_step=False,
                             dataset=dataset)

    def training_step(self, batch, batch_idx):
        rgb, hand_rgb, attention_mask, language, text_mask, fwd_rgb_chunck, fwd_hand_rgb_chunck,\
        arm_action, gripper_action, arm_action_chunck, gripper_action_chunck, chunck_mask, instr_and_action_ids, instr_and_action_labels, instr_and_action_mask \
        = self._process_batch(batch)

        prediction = self.model.forward(
            rgb,
            language,
            attention_mask=text_mask, 
            action_labels=(arm_action_chunck, gripper_action_chunck),
            action_mask=chunck_mask,
            caption_labels=language.clone() if rgb.shape[1] == 1 else None,
            caption_mask=text_mask.clone() if rgb.shape[1] == 1 else None,
            vision_gripper=hand_rgb,
            fwd_rgb_labels=fwd_rgb_chunck,
            fwd_hand_rgb_chunck=fwd_hand_rgb_chunck,
            fwd_mask=chunck_mask,
            instr_and_action_ids=instr_and_action_ids,
            instr_and_action_labels=instr_and_action_labels,
            instr_and_action_mask=instr_and_action_mask
        )
        output = self._get_loss(prediction)
        
        prog_bar_set = {'loss'}
        if self.act_pred:
            prog_bar_set.add('loss_arm_act')
            prog_bar_set.add('loss_gripper_act')
            prog_bar_set.add('acc_gripper_act')
        if self.fwd_pred:
            prog_bar_set.add('loss_obs')
            if self.fwd_pred_hand:
                prog_bar_set.add('loss_hand_obs')

        # TODO: delete the debug dump
        # input = {"rgb": rgb, "language": language, "action_labels": (arm_action_chunck, gripper_action_chunck)}
        # self.check_keypoint_vals(output, input)

        self._log_output(output, phase="train", prog_bar_set=prog_bar_set, on_step=True, on_epoch=False)
        return output['loss']
    
    # def check_keypoint_vals(self, output, input):
    #     # check optimizer state, model gradient and loss after a training step
    #     record = {
    #         'input': input,
    #         'output': output,
    #         'loss': output,
    #         'gradients': {name: param.grad for name, param in self.model.named_parameters()},
    #         'optimizer_state': self.trainer.optimizers[0].state_dict()
    #     }

    #     torch.save(record, 'flamingo_new.pth')
    #     exit(0)

    def inference_step(self, batch):
        with torch.no_grad():
            rgb, hand_rgb, attention_mask, language, text_mask, fwd_rgb_chunck, fwd_hand_rgb_chunck,\
            arm_action, gripper_action, arm_action_chunck, gripper_action_chunck, chunck_mask, instr_and_action_ids, instr_and_action_labels, instr_and_action_mask \
            = self._process_batch(batch)
            action = self.model.forward(
                rgb,
                language,
                attention_mask=text_mask, 
                action_labels=(arm_action_chunck, gripper_action_chunck),
                action_mask=chunck_mask,
                caption_labels=language.clone() if rgb.shape[1] == 1 else None,
                caption_mask=text_mask.clone() if rgb.shape[1] == 1 else None,
                vision_gripper=hand_rgb,
                fwd_rgb_labels=fwd_rgb_chunck,
                fwd_hand_rgb_chunck=fwd_hand_rgb_chunck,
                fwd_mask=chunck_mask,
                instr_and_action_ids=instr_and_action_ids,
                instr_and_action_labels=instr_and_action_labels,
                instr_and_action_mask=instr_and_action_mask,
                mode="inference",
            )

            return {"action": action}
    
    def get_grouped_params(self, model):
            params_with_wd, params_without_wd = [], []

            def apply_decay(x):
                return (
                    "gated_cross_attn_layer" in x
                    and "ff_gate" not in x
                    and "attn_gate" not in x
                    and "norm" not in x
                    and "bias" not in x
                )

            for n, p in model.named_parameters():
                
                if apply_decay(n):
                    params_with_wd.append(p)
                else:
                    params_without_wd.append(p)

            return [
                {"params": [p for p in params_with_wd if p.requires_grad], "weight_decay": self.configs['weight_decay'], "lr": self.configs["learning_rate"]},
                {"params": [p for p in params_without_wd if p.requires_grad], "weight_decay": 0.0, "lr": self.configs["learning_rate"]},
            ]