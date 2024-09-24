import time
import warnings
import os

import numpy as np
import torch
import torch.nn.functional as F
from functools import partial
import math
import json
from train.base_trainer import BaseTrainer
from model.backbone.roboqwen import RoboQwen
from utils.dist_train import get_rank

import lightning.pytorch as pl

from train.train_utils import adjust_learning_rate


class QwenTrainer(BaseTrainer):
    def __init__(self, configs):
        super(QwenTrainer, self).__init__(configs)

    def _init_policy(self) -> RoboQwen:
        model = RoboQwen(
            qwen_model_name_or_path=self.configs['qwen'],
            train_setup_configs=self.configs['train_setup'],
            fwd_head_configs=self.configs['fwd_head'],
            window_size=self.configs['window_size'],
            use_hand_rgb=self.use_hand_rgb,
            act_head_configs=self.configs['act_head'],
            fwd_pred_next_n=self.configs['fwd_pred_next_n'],
            use_vision_resampler=self.configs.get('use_vision_resampler', False),
            vision_resampler_configs=self.configs.get('vision_resampler', None)
        )
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._main_rank_print(f"Qwen Model Parameters: {total_params / 1000000:.2f}M")
        return model

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
        
        if isinstance(batch, list):
            # print(batch[0].keys())
            batch = batch[0]
        
        if isinstance(batch['rgb'], list):
            rgb = [_.cuda() for _ in batch['rgb']]
        else:
            rgb = batch['rgb'].cuda()
            if len(rgb.shape) == 4:
                rgb = rgb.unsqueeze(1)
            assert len(rgb.shape) == 5

        if isinstance(batch['text'], list) and isinstance(batch['text'][0], str):
            raise ValueError('The raw text data is not supported')
        else:
            seq_len = self.configs['window_size']
            language = batch['text'].cuda()
            text_mask = batch['text_mask'].cuda()
        
        if batch.get('action', None) is not None:
            action = batch['action'].cuda()
        else:
            action = None

        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = batch['attention_mask'].cuda()
        
        if self.use_hand_rgb and batch.get('hand_rgb', None) is not None:
            hand_rgb = batch['hand_rgb'].cuda()
        else:
            hand_rgb = None

        # Split arm and gripper action
        arm_action = None
        gripper_action = None

        if action is not None:
            arm_action = action[:, :, :6]  # b,len,act_dim-1
            gripper_action = action[:, :, 6]  # b,len

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
            action_chunck = action_chunck.cuda()
            arm_action_chunck = action_chunck[..., :6]
            gripper_action_chunck = action_chunck[..., -1]

        if isinstance(rgb, torch.Tensor):
            rgb = rgb[:, :seq_len]
            if hand_rgb is not None:
                hand_rgb = hand_rgb[:, :seq_len]
        
        chunck_mask = batch.get('chunck_mask', None)
        if chunck_mask is not None:
            chunck_mask = chunck_mask.cuda()
        
        fwd_mask = batch.get('fwd_mask', None)
        if fwd_mask is not None:
            fwd_mask = fwd_mask.bool().cuda()

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
        
        raw_text = batch.get('raw_text', None)
        data_source = batch.get('data_source', "calvin_action")

        return rgb, hand_rgb, attention_mask, language, text_mask, fwd_rgb_chunck, fwd_hand_rgb_chunck,\
        arm_action, gripper_action, arm_action_chunck, gripper_action_chunck, chunck_mask, fwd_mask, instr_and_action_ids, instr_and_action_labels, instr_and_action_mask, raw_text, data_source

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        with torch.no_grad():
            if isinstance(batch, tuple):
                batch = batch[0]
            rgb, hand_rgb, attention_mask, language, text_mask, fwd_rgb_chunck, fwd_hand_rgb_chunck,\
            arm_action, gripper_action, arm_action_chunck, gripper_action_chunck, chunck_mask, fwd_mask, instr_and_action_ids, instr_and_action_labels, instr_and_action_mask, raw_text, data_source \
            = self._process_batch(batch)
            prediction = self.model.forward(
                rgb, 
                language,
                attention_mask=text_mask, 
                action_labels=(arm_action_chunck, gripper_action_chunck),
                action_mask=chunck_mask,
                caption_labels=language.clone() if (isinstance(rgb, torch.Tensor) and rgb.shape[1] == 1) else None,
                caption_mask=text_mask.clone() if (isinstance(rgb, torch.Tensor) and rgb.shape[1] == 1) else None,
                vision_gripper=hand_rgb,
                fwd_rgb_labels=fwd_rgb_chunck,
                fwd_hand_rgb_labels=fwd_hand_rgb_chunck,
                fwd_mask=fwd_mask,
                instr_and_action_ids=instr_and_action_ids, 
                instr_and_action_labels=instr_and_action_labels, 
                instr_and_action_mask=instr_and_action_mask,
                raw_text=raw_text,
                data_source=data_source
            )

            output = self._get_loss(prediction)
            
            prog_bar_set = {'loss'}
            if self.act_pred:
                prog_bar_set.add('loss_arm_act')
                prog_bar_set.add('loss_gripper_act')
                prog_bar_set.add('acc_arm_act')
                prog_bar_set.add('acc_gripper_act')
                prog_bar_set.add('action_l1')
                prog_bar_set.add('clip_l1')
            if self.fwd_pred:
                prog_bar_set.add('loss_obs_fwd')
                if self.fwd_pred_hand:
                    prog_bar_set.add('loss_hand_obs_fwd')

            dataset = None
            if self.val_set_names is not None:
                dataset = self.val_set_names[dataloader_idx]

            self._log_output(output, phase="val", prog_bar_set=prog_bar_set,
                             sync_dist=True, on_epoch=True, on_step=False,
                             dataset=dataset)

    def training_step(self, batch, batch_idx):
        # print(type(batch))
        # print('-'*100)
        if isinstance(batch, tuple):
            batch = batch[0]
        rgb, hand_rgb, attention_mask, language, text_mask, fwd_rgb_chunck, fwd_hand_rgb_chunck,\
        arm_action, gripper_action, arm_action_chunck, gripper_action_chunck, chunck_mask, fwd_mask, instr_and_action_ids, instr_and_action_labels, instr_and_action_mask, raw_text, data_source \
        = self._process_batch(batch)
        prediction = self.model.forward(
            rgb, 
            language,
            attention_mask=text_mask, 
            action_labels=(arm_action_chunck, gripper_action_chunck),
            action_mask=chunck_mask,
            caption_labels=language.clone() if (isinstance(rgb, torch.Tensor) and rgb.shape[1] == 1) else None,
            caption_mask=text_mask.clone() if (isinstance(rgb, torch.Tensor) and rgb.shape[1] == 1) else None,
            vision_gripper=hand_rgb,
            fwd_rgb_labels=fwd_rgb_chunck,
            fwd_hand_rgb_labels=fwd_hand_rgb_chunck,
            fwd_mask=fwd_mask,
            instr_and_action_ids=instr_and_action_ids,
            instr_and_action_labels=instr_and_action_labels,
            instr_and_action_mask=instr_and_action_mask,
            raw_text=raw_text,
            data_source=data_source
        )

        output = self._get_loss(prediction)
        
        prog_bar_set = {'loss'}
        if self.act_pred:
            prog_bar_set.add('loss_arm_act')
            prog_bar_set.add('loss_gripper_act')
            prog_bar_set.add('acc_arm_act')
            prog_bar_set.add('acc_gripper_act')
            prog_bar_set.add('action_l1')
            prog_bar_set.add('clip_l1')
        if self.fwd_pred:
            prog_bar_set.add('loss_obs')
            if self.fwd_pred_hand:
                prog_bar_set.add('loss_hand_obs')
        prog_bar_set.add('loss_vl_cotrain')

        self._log_output(output, phase="train", prog_bar_set=prog_bar_set, on_step=True, on_epoch=False)
        return output['loss']

    def inference_step(self, batch):
        with torch.no_grad():
            rgb, hand_rgb, attention_mask, language, text_mask, fwd_rgb_chunck, fwd_hand_rgb_chunck, \
            arm_action, gripper_action, arm_action_chunck, gripper_action_chunck, chunck_mask, fwd_mask, instr_and_action_ids, instr_and_action_labels, instr_and_action_mask, raw_text, data_source \
                = self._process_batch(batch)

            prediction = self.model.inference(
                rgb,
                language,
                attention_mask=text_mask,
                action_labels=(arm_action_chunck, gripper_action_chunck),
                action_mask=chunck_mask,
                # FIXME: hardcode here
                caption_labels=language.clone() if (isinstance(rgb, torch.Tensor) and rgb.shape[1] == 1) else None,
                caption_mask=text_mask.clone() if (isinstance(rgb, torch.Tensor) and rgb.shape[1] == 1) else None,
                vision_gripper=hand_rgb,
                fwd_rgb_labels=fwd_rgb_chunck,
                fwd_hand_rgb_labels=fwd_hand_rgb_chunck,
                fwd_mask=chunck_mask,
                instr_and_action_ids=instr_and_action_ids, 
                instr_and_action_labels=instr_and_action_labels, 
                instr_and_action_mask=instr_and_action_mask,
                raw_text=raw_text
            )

            return prediction
