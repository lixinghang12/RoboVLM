from os import curdir
from tkinter import N
import torch
import numpy as np
from torch import nn
import copy
from typing import Tuple
from einops import rearrange, repeat
import json
import sys, copy
sys.path.append('/mnt/bn/liuminghuan/RobotVLM')
# from model.llm.robollava import build_llava
from train.loss import calculate_vl_cross_entropy
import transformers
from transformers import GPTQConfig
from typing import Optional, Tuple, List
from data.vid_llava_constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


class RoboQwen(nn.Module):

    def __init__(
        self,
        qwen_model_name_or_path,
        train_setup_configs,
        act_encoder_configs=None,
        act_head_configs=None,
        fwd_head_configs=None,
        window_size=None,
        use_obs_queries=True,
        use_act_queries=True,
        use_hand_rgb=False,
        use_pixel_loss=True,
        use_mim_obs_loss=False,
        use_time_causal_attn=True,
        vision_masked_ratio=0.9,
        use_tube_mask=False,
        fwd_pred_next_n=1,
        use_vision_resampler=False,
        vision_resampler_configs=None,
        **kwargs
    ):
        super().__init__()
        self.window_size = window_size
        self.use_obs_queries = use_obs_queries
        self.use_act_queries = use_act_queries
        self.use_hand_rgb = use_hand_rgb
        self.use_pixel_loss = use_pixel_loss
        self.use_mim_obs_loss = use_mim_obs_loss
        self.use_time_causal_attn = use_time_causal_attn
        self.vision_masked_ratio = vision_masked_ratio
        self.use_tube_mask = use_tube_mask
        self.fwd_pred_next_n = fwd_pred_next_n

        # Copy configs
        assert isinstance(qwen_model_name_or_path, str)
        self.qwen_path = qwen_model_name_or_path

        self.train_setup_configs = train_setup_configs
        self.act_encoder_configs = act_encoder_configs
        self.act_head_configs = act_head_configs
        self.fwd_head_configs = fwd_head_configs

        self.use_hand_rgb = use_hand_rgb
    
        self.tokenizer, self.qwen_model = self._init_qwen() # Including loading pre-trained ckpts
        self.act_head, self.fwd_head = self._init_heads()
        if self.act_head is not None:
            self.action_space = self.act_head_configs.get("action_space", "continuous")
            if self.action_space == "continuous":
                self.action_token = nn.Parameter(torch.zeros(self.hidden_size))
                self.action_token.requires_grad_(True)
            elif self.action_space == "discrete":
                action_bin = self.act_head_configs.get("action_bin", 256)
                self.action_embeddings = nn.Embedding(action_bin, self.hidden_size)
            else:
                assert self.action_space == "down_sample", "Unsupported action space"
        
        if self.fwd_head_configs is not None:
            self.image_latent_num = self.fwd_head_configs.get('image_latent_num', 8)
            self.pred_image = True
            self.pred_hand_image = self.fwd_head_configs.get('pred_hand_image', False)

            global_frame_num = self.fwd_head_configs.get("global_frame_num", -1)
            if global_frame_num != -1:
                predict_image_num = global_frame_num - 1
            else:
                predict_image_num = self.fwd_pred_next_n
            
            self.static_image_tokens = nn.Parameter(torch.zeros(self.image_latent_num * predict_image_num, self.hidden_size))
            self.static_image_tokens.requires_grad_(True)
            if self.pred_hand_image:
                self.hand_image_tokens = nn.Parameter(torch.zeros(self.image_latent_num * predict_image_num, self.hidden_size))
                self.hand_image_tokens.requires_grad_(True)
        else:
            self.pred_image = False
    
        ### setup vision tower and configs
        self.vis_dim = self.hidden_size
        self.use_vision_resampler = use_vision_resampler
        self.vision_resampler_configs = vision_resampler_configs
        print(f'use vision resampler: {self.use_vision_resampler}')
        if self.use_vision_resampler:
            from model.vision_encoder.vision_resampler import PerceiverResampler
            self.vision_resampler = PerceiverResampler(dim=self.vis_dim, num_latents=self.vision_resampler_configs['num_latents'])        
        self._trainable_params_setup()

    @property
    def hidden_size(self):
        return self.qwen_model.config.hidden_size

    @property
    def visual_config(self):
        return self.qwen_model.config.visual

    def _init_qwen(self):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.qwen_path,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True,
        )
        tokenizer.pad_token_id = tokenizer.eod_id
        if self.train_setup_configs['lora_enable'] and self.train_setup_configs['q_lora']:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.qwen_path,
                trust_remote_code=True,
                quantization_config=GPTQConfig(bits=4, disable_exllama=True),
                fp32=True
            ).float()
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.qwen_path,
                trust_remote_code=True,
                fp32=True
            ).float()
        return tokenizer, model

    @property
    def qwen_vit(self):
        return self.qwen_model.transformer.visual

    def encode_images(self, images, image_sizes=None):
        # input: images: list of b,c,h,w or b,t,c,h,w
        # output: image_features: list of bx[txn, d]
        if images.ndim == 4:
            images = images.unsqueeze(1)
        bs, seq_len = images.shape[:2]
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            images = torch.cat([image for image in images], dim=0)
        image_features = self.qwen_vit(images)
        image_features = image_features.view(bs, seq_len, -1, image_features[0].shape[-1])
        
        if self.use_vision_resampler:
            ### downsample at token num dim: b, s, n, d -> b, s, v d
            # b T F v d -> b, T, n, d
            image_features = self.vision_resampler(image_features.unsqueeze(2))  # downsample v_tok per image to n_tok

        return image_features
    
    def encode_action(self, bs=None, T=None):
        action_space = self.act_head_configs.get("action_space", "continuous")
        if action_space == "continuous":
            return repeat(self.action_token, 'd -> b t n d', b=bs, t=T, n=self.latent_num)
        elif action_space == "discrete":
            # eps = 1e-6
            # min_val = self.act_head_configs.get("min_val", -1.0)
            # max_val = self.act_head_configs.get("max_val", 1.0)
            # action_bin = self.act_encoder_configs.get("action_bin", 256)
            # interval = (max_val - min_val) / action_bin
            # arm_index = (action_labels[0] - min_val + eps)  / interval
            # gripper_index = action_labels[1].long()
            
            # assume the action is tokenized in the dataset and convert to input_ids
            raise NotImplementedError
        else:
            raise ValueError(f"Unexpected action_space: {self.act_head_configs.get('action_space', 'continuous')}")

    @property
    def word_embedding(self):
        return self.qwen_model.transformer.wte

    def prepare_inputs_labels_for_action_prediction(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        action_labels: Tuple[torch.Tensor, torch.Tensor] = None,
        action_mask: torch.Tensor = None,
        past_key_values=None,
        vision_gripper = None,
        **kwargs
    ):
        bs, seq_len = vision_x.shape[:2]
        rgb_feats = self.encode_images(vision_x)
        if vision_gripper is not None:
            assert vision_gripper.shape[1] == vision_x.shape[1], "Gripper and RGB input must have the same sequence length"
            gripper_feats = self.encode_images(vision_gripper)
        else:
            gripper_feats = None
        action_tokens = None
        if 'continuous' in self.act_head_configs['action_space']:
            action_tokens = self.encode_action(bs, seq_len)
        input_ids = lang_x
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        new_input_embeds = []
        language_mask = []
        action_mask = []
        vision_mask = []
        start_token_id = torch.tensor([self.visual_config['image_start_id']]).to(self.qwen_model.device)
        end_token_id = start_token_id + 1
        
        image_start_embed = self.word_embedding(start_token_id)
        image_end_embed = self.word_embedding(end_token_id)
        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_new_input_embeds = [self.word_embedding(cur_input_ids)]
            
            cur_language_mask = [torch.ones(cur_input_ids.shape[0]).to(dtype=torch.long, device=cur_input_ids.device)]
            cur_action_mask = [torch.zeros(cur_input_ids.shape[0]).to(dtype=torch.long, device=cur_input_ids.device)]
            cur_vision_mask = [torch.zeros(cur_input_ids.shape[0]).to(dtype=torch.long, device=cur_input_ids.device)]

            for step_idx in range(seq_len):
                rgb_toks = rgb_feats[batch_idx, step_idx]
                cur_new_input_embeds.append(image_start_embed.clone())
                cur_new_input_embeds.append(rgb_toks)
                cur_new_input_embeds.append(image_end_embed.clone())
                cur_language_mask.append(torch.zeros(rgb_toks.shape[0] + 2).to(dtype=torch.long, device=cur_input_ids.device))
                cur_language_mask[-1][0] = 1
                cur_language_mask[-1][-1] = 1
                cur_action_mask.append(torch.zeros(rgb_toks.shape[0] + 2).to(dtype=torch.long, device=cur_input_ids.device))
                cur_vision_mask.append(torch.ones(rgb_toks.shape[0] + 2).to(dtype=torch.long, device=cur_input_ids.device))

                if gripper_feats is not None:
                    # gripper_toks = gripper_feats[batch_idx, step_idx]
                    # cur_new_input_embeds.append(gripper_toks)
                    # cur_language_mask.append(torch.zeros(gripper_toks.shape[0]).to(dtype=torch.long, device=cur_input_ids.device))
                    # cur_action_mask.append(torch.zeros(gripper_toks.shape[0]).to(dtype=torch.long, device=cur_input_ids.device))
                    # cur_vision_mask.append(torch.ones(gripper_toks.shape[0]).to(dtype=torch.long, device=cur_input_ids.device))

                    gripper_toks = gripper_feats[batch_idx, step_idx]
                    cur_new_input_embeds.append(image_start_embed.clone())
                    cur_new_input_embeds.append(gripper_toks)
                    cur_new_input_embeds.append(image_end_embed.clone())

                    cur_language_mask.append(torch.zeros(gripper_toks.shape[0] + 2).to(dtype=torch.long, device=cur_input_ids.device))
                    cur_language_mask[-1][0] = 1
                    cur_language_mask[-1][-1] = 1

                    cur_action_mask.append(torch.zeros(gripper_toks.shape[0] + 2).to(dtype=torch.long, device=cur_input_ids.device))
                    cur_vision_mask.append(torch.ones(gripper_toks.shape[0] + 2).to(dtype=torch.long, device=cur_input_ids.device))
                
                # add downsample support for policy head
                if action_tokens is not None:
                    cur_new_input_embeds.append(action_tokens[batch_idx, step_idx])
                    cur_language_mask.append(torch.zeros(action_tokens[batch_idx, step_idx].shape[0]).to(dtype=torch.long, device=cur_input_ids.device))
                    cur_action_mask.append(torch.ones(action_tokens[batch_idx, step_idx].shape[0]).to(dtype=torch.long, device=cur_input_ids.device))
                    cur_vision_mask.append(torch.zeros(action_tokens[batch_idx, step_idx].shape[0]).to(dtype=torch.long, device=cur_input_ids.device))

            new_input_embeds.append(torch.cat(cur_new_input_embeds, dim=0))
            language_mask.append(torch.cat(cur_language_mask, dim=0))
            action_mask.append(torch.cat(cur_action_mask, dim=0))
            vision_mask.append(torch.cat(cur_vision_mask, dim=0))
        
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_language_mask_padded = []
        new_action_mask_padded = []
        new_vision_mask_padded = []
        
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=lang_x.device)

        for i, (cur_new_embed, cur_new_lang_mask, cur_new_action_mask, cur_new_vision_mask) \
            in enumerate(zip(new_input_embeds, language_mask, action_mask, vision_mask)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.train_setup_configs, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                new_language_mask_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len), dtype=cur_new_lang_mask.dtype, device=cur_new_lang_mask.device),
                    cur_new_lang_mask
                ), dim=0))
                new_action_mask_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len), dtype=cur_new_action_mask.dtype, device=cur_new_action_mask.device),
                    cur_new_action_mask
                ), dim=0))
                new_vision_mask_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len), dtype=cur_new_vision_mask.dtype, device=cur_new_vision_mask.device),
                    cur_new_vision_mask
                ), dim=0))
                if cur_len > 0:
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                new_language_mask_padded.append(torch.cat((
                    cur_new_lang_mask,
                    torch.zeros((max_len - cur_len), dtype=cur_new_lang_mask.dtype, device=cur_new_lang_mask.device)
                ), dim=0))
                new_action_mask_padded.append(torch.cat((
                    cur_new_action_mask,
                    torch.zeros((max_len - cur_len), dtype=cur_new_action_mask.dtype, device=cur_new_action_mask.device)
                ), dim=0))
                new_vision_mask_padded.append(torch.cat((
                    cur_new_vision_mask,
                    torch.zeros((max_len - cur_len), dtype=cur_new_vision_mask.dtype, device=cur_new_vision_mask.device)
                ), dim=0))
                if cur_len > 0:
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        new_language_mask = torch.stack(new_language_mask_padded, dim=0)
        new_action_mask = torch.stack(new_action_mask_padded, dim=0)
        new_vision_mask = torch.stack(new_vision_mask_padded, dim=0)

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_language_mask, new_action_mask, new_vision_mask

    def cat_multi_modal_input(
        self,
        input_embeds: torch.Tensor,
        multimodal_embeds: torch.Tensor = None,
        insert_idx: int=0,
        masks: torch.Tensor = None,
    ):
        if insert_idx >= 0:
            return torch.cat((input_embeds[:, :insert_idx], multimodal_embeds, input_embeds[:, insert_idx:]), dim=1)
        elif insert_idx == -1 and masks is not None:
            new_embed_list = []
            for mask, input_embed, multimodal_embed in zip(masks, input_embeds, multimodal_embeds):
                # the concat index is up to mask first False index
                # concat_idx = (mask == False).nonzero()[0].item()
                indexs = (mask == False).nonzero()
                insert_idx = indexs[0].item() if len(indexs) > 0 else len(mask)
                new_embed = torch.cat((input_embed[:insert_idx], multimodal_embed, input_embed[insert_idx:]), dim=0)
                new_embed_list.append(new_embed)
            return torch.stack(new_embed_list, dim=0)
        else:
            raise Exception("insert_idx should be -1 or >= 0, and if you want to insert as last(-1), you should provide masks")
        
    def merge_multi_modal_input(
        self,
        input_embeds: torch.Tensor,
        multimodal_feats: torch.Tensor = None,
        labels: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        is_image: bool=True,
        insert_idx: int=0,
        fill_zero=False
    ):
        bs = multimodal_feats.shape[0]
        if is_image:
            rgb_feats = self.encode_images(multimodal_feats)
            
            start_image_token_id = torch.tensor([self.visual_config['image_start_id']]).to(self.qwen_model.device)
            end_image_token_id = start_image_token_id + 1
            
            image_start_embed = self.word_embedding(start_image_token_id).to(self.qwen_model.device)
            image_end_embed = self.word_embedding(end_image_token_id).to(self.qwen_model.device)
            image_start_embed = image_start_embed.unsqueeze(0).unsqueeze(0).repeat(*rgb_feats.shape[:2], 1, 1)
            image_end_embed = image_end_embed.unsqueeze(0).unsqueeze(0).repeat(*rgb_feats.shape[:2], 1, 1)

            rgb_feats = torch.cat(
                [image_start_embed, rgb_feats, image_end_embed],
                dim=2
            )
            rgb_feats = rearrange(rgb_feats, 'b l n d -> b (l n) d') #flatten seq_len and n_tok_per_img dim
        else:
            rgb_feats = multimodal_feats
        
        added_seq_len = rgb_feats.shape[1]
        # multimodal_embeds = self.cat_multi_modal_input(input_embeds, multimodal_embeds, insert_idx, attention_mask)

        multimodal_embeds = torch.cat(
            [input_embeds[:, :insert_idx], rgb_feats, input_embeds[:, insert_idx:]],
            dim=1
        )

        insert_mask = torch.cat(
            [torch.zeros(input_embeds[:, :insert_idx].shape[:2]), torch.ones(rgb_feats.shape[:2]), torch.zeros(input_embeds[:, insert_idx:].shape[:2])],
            dim=1
        ).bool().to(multimodal_embeds.device)
   
        mutlimodal_labels = None
        if labels is not None:
            mutlimodal_labels = torch.full(
                (bs, added_seq_len), -100, dtype=labels.dtype, device=labels.device
            )
            mutlimodal_labels = self.cat_multi_modal_input(labels, mutlimodal_labels, insert_idx, attention_mask)
            if is_image:
                mutlimodal_labels[:, 0] = self.visual_config['image_start_id']
                mutlimodal_labels[:, multimodal_feats.shape[1] + 1] = self.visual_config['image_start_id'] + 1
        
        multimodal_attention_mask = None
        if attention_mask is not None:
            val = False if fill_zero else True
            multimodal_attention_mask = torch.full(
                (bs, added_seq_len), val, dtype=attention_mask.dtype, device=attention_mask.device
            )
            multimodal_attention_mask = self.cat_multi_modal_input(attention_mask, multimodal_attention_mask, insert_idx, attention_mask)

        return multimodal_embeds, mutlimodal_labels, multimodal_attention_mask, insert_mask
    
    def _init_heads(self):
        action_head = None
        if self.act_head_configs is not None:
            import model.policy_head.base_policy as action_heads
            _kwargs = copy.deepcopy(self.act_head_configs)
            _kwargs.update(
                dict(# hidden_size=self.hidden_size, 
                    tokenizer=self.tokenizer,
                    in_features=self.hidden_size,
                    fwd_pred_next_n=self.fwd_pred_next_n,
                    window_size=self.window_size))
            _cls = getattr(action_heads, _kwargs.pop('type'))
            self.latent_num = self.act_head_configs.get("latent", 1)
            action_head = _cls(**_kwargs)

        fwd_decoder = None
        if self.fwd_head_configs is not None:
            import model.forward_head as fwd_heads
            _kwargs = copy.deepcopy(self.fwd_head_configs)
            _kwargs.update(dict(image_size=self.visual_config['image_size'],
                                patch_size=self.visual_config['patch_size'],
                                hidden_size=self.hidden_size,
                                chunk_size=1))
            _cls = getattr(fwd_heads, _kwargs.pop('type'))
            if self.use_mim_obs_loss:
                _kwargs['fwd_pred_next_n'] = 0
            fwd_decoder = _cls(**_kwargs)

        return action_head, fwd_decoder
    
    def test_model(self, model: nn.Module=None):
        if model is None:
            model = self.qwen_model
        images = torch.load('test.pt')
        images = images[:2]
        images = images.to(model.device)
        success = True
        with torch.no_grad():
            try:
                image_feat = model.transformer.visual(images)
                print(image_feat.shape)
            except Exception as e:
                success = False
                print(e)
        return success
    
    def _trainable_params_setup(self):
        # TODO: merge precision into trainer
        model = self.qwen_model
        # compute_dtype = torch.float32 # (torch.float16 if self.train_setup_configs['precision'] == 'fp16' else (torch.bfloat16 if self.train_setup_configs['precision'] == 'bf16' else torch.float32))
        if self.train_setup_configs['bits'] in [4, 8]: # TODO: in trainer
            from peft import prepare_model_for_kbit_training
            model.config.torch_dtype=(torch.float32 if self.train_setup_configs['precision'] == 'fp16' else (torch.bfloat16 if self.train_setup_configs['precision'] == 'bf16' else torch.float32))
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=self.train_setup_configs['gradient_checkpointing'])
            model.requires_grad_(True)

        if self.train_setup_configs['bits'] in [16]:
            if self.train_setup_configs['precision'] == 'bf16':
                model.to(torch.bfloat16)
            elif self.train_setup_configs['precision'] == 'fp16':
                model.to(torch.float16)
        print("train_vision:", self.train_setup_configs.get('train_vision', False))
        if self.train_setup_configs.get('train_vision', False):
            model.transformer.visual.requires_grad_(True)
            if hasattr(model.transformer.visual,'attn_pool'):
                model.transformer.visual.attn_pool.requires_grad_(True)
        else:
            model.transformer.visual.requires_grad_(False)
            # if hasattr(model.transformer.visual,'attn_pool'):
            #     model.transformer.visual.attn_pool.requires_grad_(True)
        
        if self.train_setup_configs['freeze_backbone']:
            # set all paramter of backbone to be not trainable
            model.requires_grad_(False)
        
        if self.train_setup_configs['train_text_embedding']:
            model.transformer.wte.requires_grad_(True)
        
        if self.train_setup_configs['lora_enable']:
            if self.train_setup_configs['q_lora'] or "chat" in self.qwen_path:
                modules_to_save = None
            else:
                modules_to_save = ["wte", "lm_head"]
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            lora_config = LoraConfig(
                r=self.train_setup_configs['lora_r'],
                lora_alpha=self.train_setup_configs['lora_alpha'],
                target_modules=["c_attn", "attn.c_proj", "w1", "w2"],  ##["in_proj","out_proj","c_fc"]
                lora_dropout=self.train_setup_configs['lora_dropout'],
                bias=self.train_setup_configs['lora_bias'],
                task_type="CAUSAL_LM",
                modules_to_save=modules_to_save  # This argument serves for adding new tokens.
            )
            if self.train_setup_configs['bits'] == 16:
                if self.train_setup_configs['precision'] == 'bf16':
                    model.to(torch.bfloat16)
                elif self.train_setup_configs['precision'] == 'fp16':
                    model.to(torch.float16)
            if self.train_setup_configs['q_lora']:
                model = prepare_model_for_kbit_training(
                    model, use_gradient_checkpointing=self.train_setup_configs['gradient_checkpointing']
                )

            model = get_peft_model(model, lora_config)
            
            if self.train_setup_configs['gradient_checkpointing']:
                model.enable_input_require_grads()

        self.qwen_model = model
        if self.act_head is not None:
            self.act_head.requires_grad_(True)
        self.train()
        self.test_trainable()


    def _forward_action_head(
            self,
            action_tokens: torch.Tensor,
            action_labels: Tuple[torch.Tensor, torch.Tensor] = None,
            action_mask: torch.Tensor = None,
            **kwargs
        ):
        # action_tokens = get_target_modal_tokens(output_hs, self._action_mask(output_hs))
        action = self.act_head(action_tokens)

        action_loss = None
        if action_labels is not None:
            action_loss = self.act_head.loss(action, action_labels, action_mask)

        return action, action_loss

    def _forward_caption(
        self,
        logits: torch.Tensor,
        caption_labels: torch.Tensor = None,
        caption_mask: torch.Tensor = None,
        **kwargs,
    ):
        
        caption_loss = {"loss": None}
        if caption_labels is not None:
            caption_loss['loss'] = calculate_vl_cross_entropy(logits, caption_labels, caption_mask)

        return logits, caption_loss

    def _action_mask(self, output_hs):
        return torch.ones(*output_hs.shape[:-1]).to(output_hs.device)

    def _fwd_mask(self, output_hs):
        return torch.ones(*output_hs.shape[:-1]).to(output_hs.device)
    
    def _caption_mask(self, output_hs):
        return torch.ones(*output_hs.shape[:-1]).to(output_hs.device)
    
    def _format_loss(self, loss):
        # for visualization and loss backward in pytorch lightning
        _loss = 0
        _keys = list(loss.keys())

        for k in _keys:
            if 'loss' in k:
                _loss += loss[k]

        loss['loss'] = _loss
        return loss
    
    @staticmethod
    def _update_loss(loss, new_loss, suffix=None):
        """
        use new_loss to update loss.
            * if suffix is not None, the key from new_loss will be reformatted as: key|suffix
            * otherwise, if the key from new_loss is not in loss, it will be directly used: key
            * otherwise, the key from the new_loss will be reformatted as: key|index, where index is
                searched from 0->+inf so that key|index is not in loss.

        """
        def get_key(k, d):
            if suffix is not None:
                new_k = f"{k}_{suffix}"
                assert new_k not in d
                return new_k

            ind = 0
            while True:
                if ind == 0:
                    new_k = k
                else:
                    new_k = f"{k}_{ind}"
                if new_k not in d:
                    return new_k
                ind += 1

        for k in new_loss:
            new_k = get_key(k, loss)
            loss[new_k] = new_loss[k]

        return loss
    
    def forward_discrete(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        use_cached_vision_x: bool = False, # TODO: Do we need this? If not we can remove it from here
        action_labels: Tuple[torch.Tensor, torch.Tensor] = None,
        action_mask: torch.Tensor = None,
        caption_labels: torch.Tensor = None,
        caption_mask: torch.Tensor = None,
        past_key_values=None,
        use_cache: bool = False,
        vision_gripper = None,
        fwd_rgb_labels: torch.Tensor = None,
        fwd_hand_rgb_labels: torch.Tensor = None,
        fwd_mask: torch.Tensor = None,
        instr_and_action_ids=None,
        instr_and_action_labels=None,
        instr_and_action_mask=None,
        **kwargs
    ):
        loss = {}
        assert (
            vision_x is not None
        )

        bs, window_size = instr_and_action_ids.shape[:2]

        if instr_and_action_ids.ndim == 2:
            instr_and_action_ids = instr_and_action_ids.unsqueeze(1).repeat(1, window_size, 1)
            instr_and_action_labels = instr_and_action_labels.unsqueeze(1).repeat(1, window_size, 1)
            instr_and_action_mask = instr_and_action_mask.unsqueeze(1).repeat(1, window_size, 1)

        instr_and_action_ids = instr_and_action_ids.flatten(0, 1)
        instr_and_action_labels = instr_and_action_labels.flatten(0, 1)
        instr_and_action_mask = instr_and_action_mask.flatten(0, 1)

        input_embeds = self.word_embedding(instr_and_action_ids)
        vision_x = vision_x.flatten(0, 1)
        
        if vision_gripper is not None:
            vision_gripper = vision_gripper.flatten(0, 1)
        
        (
            multimodal_embeds, 
            mutlimodal_labels, 
            multimodal_attention_mask,
            _
        ) = self.merge_multi_modal_input(
            input_embeds,
            vision_x,
            instr_and_action_labels,
            instr_and_action_mask
        )

        if vision_gripper is not None:
            (
                multimodal_embeds, 
                mutlimodal_labels,
                multimodal_attention_mask,
                _
            ) = self.merge_multi_modal_input(
                multimodal_embeds,
                vision_gripper,
                mutlimodal_labels,
                multimodal_attention_mask
            )
            
        multimodal_embeds, mutlimodal_labels, multimodal_attention_mask = (
            rearrange(tensor, "(bs ws) seq_len ... -> bs (ws seq_len) ...", bs=bs, ws=window_size)
            for tensor in (multimodal_embeds, mutlimodal_labels, multimodal_attention_mask)
        )
        
        output = self.qwen_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=multimodal_embeds,
                use_cache=use_cache
        )
        output_hs = output.logits
        
        _, action_loss = self._forward_action_head(output_hs, mutlimodal_labels, multimodal_attention_mask)
        self._update_loss(loss, action_loss, 'act')
        
        loss = self._format_loss(loss)

        return loss

    def _forward_image_prediction_head(
        self,
        image_latent_tokens: torch.Tensor,
        obs_target: torch.Tensor,
        image_mask: torch.Tensor = None,
        **kwargs
    ):
        # bs, window size, fwd_pred_next_n * latent_num, hidden_size 
        ws, next_n = obs_target.shape[1:3]
        obs_target = rearrange(obs_target, "bs ws next_n ... -> bs (ws next_n) 1 ...")
        obs_target = self.fwd_head.get_targets(obs_target)
        obs_target = rearrange(obs_target, "bs (ws next_n) 1 ... -> bs ws next_n ...", next_n=next_n)

        image_latent_tokens = rearrange(image_latent_tokens, "bs ws (next_n latent_num) ... -> bs (ws next_n) latent_num ...", next_n=next_n)
        image_preds = self.fwd_head(image_latent_tokens)
        image_preds = rearrange(image_preds, "bs (ws next_n) chunk_size ... -> bs ws (next_n chunk_size) ...", next_n=next_n, chunk_size=1)
        loss = self.fwd_head.loss(image_preds, obs_target, image_mask)
        return loss, image_preds

    def forward_image_prediction(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_values=None,
        use_cache: bool = False,
        vision_gripper = None,
        fwd_rgb_labels: torch.Tensor = None,
        fwd_hand_rgb_labels: torch.Tensor = None,
        fwd_mask: torch.Tensor = None,
        **kwargs
    ): 
        loss = {}
        assert (
            vision_x is not None
        )
        bs, seq_len = vision_x.shape[:2]
        eos_offset = int(self.tokenizer.eos_token is not None)
        bos_offset = int(self.tokenizer.bos_token is not None)
        
        vision_x = vision_x.reshape(bs*seq_len, *vision_x.shape[2:]).unsqueeze(1)
        lang_x = lang_x.repeat(1, seq_len, 1).flatten(0, 1)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat(1, seq_len, 1).flatten(0, 1)
        if vision_gripper is not None:
            vision_gripper = vision_gripper.reshape(bs*seq_len, *vision_gripper.shape[2:]).unsqueeze(1)
        
        input_embeds = self.word_embedding(lang_x)
        
        (
            multimodal_embeds,
            mutlimodal_labels, 
            multimodal_attention_mask,
            _
        ) = self.merge_multi_modal_input(
            input_embeds,
            vision_x,
            labels=None,
            attention_mask=attention_mask,
            insert_idx=bos_offset
        )
        
        if vision_gripper is not None:
            (
                multimodal_embeds, 
                mutlimodal_labels, 
                multimodal_attention_mask,
                _
            ) = self.merge_multi_modal_input(
                multimodal_embeds,
                vision_gripper,
                mutlimodal_labels,
                multimodal_attention_mask,
                insert_idx=bos_offset
            )
        
        insert_idx = multimodal_embeds.shape[1] - eos_offset # insert at last
        learnable_tokens = repeat(self.static_image_tokens, 'n d -> b n d', b=multimodal_embeds.shape[0])
        if self.pred_hand_image:
            hand_learnable_tokens = repeat(self.hand_image_tokens, 'n d -> b n d', b=multimodal_embeds.shape[0])
            learnable_tokens = torch.cat([hand_learnable_tokens, learnable_tokens], dim=1)

        (
            multimodal_embeds, 
            mutlimodal_labels, 
            multimodal_attention_mask,
            learnable_tokens_mask
        ) = self.merge_multi_modal_input(
            multimodal_embeds,
            learnable_tokens,
            mutlimodal_labels,
            multimodal_attention_mask,
            is_image=False,
            insert_idx=insert_idx,
            fill_zero=True
        )

        output = self.qwen_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=multimodal_embeds,
            use_cache=use_cache,
            output_hidden_states=True,
        )

        output_hs = output.hidden_states[-1].clone()
        # output_hs = rearrange(output_hs, 'b (l n) d -> (b l) n d', l=seq_len)
        learnable_hs = output_hs[learnable_tokens_mask].reshape(bs, seq_len, -1, output_hs.shape[-1])
        if self.pred_hand_image:
            static_image_hs, hand_image_hs = learnable_hs.chunk(axis=2, chunks=2)
        else:
            static_image_hs, hand_image_hs = learnable_hs, None
        static_loss, _ = self._forward_image_prediction_head(static_image_hs, fwd_rgb_labels, fwd_mask)
        if hand_image_hs is not None:
            hand_loss, _ = self._forward_image_prediction_head(hand_image_hs, fwd_hand_rgb_labels, fwd_mask)
        else:
            hand_loss = None
        loss['loss_obs_fwd'] = static_loss
        if hand_loss is not None:
            loss['loss_hand_obs_fwd'] = hand_loss
        loss = self._format_loss(loss)
        return loss
    

    def forward_video_caption(
        self,
        vision_x: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_values=None,
        use_cache: bool = False,
        vision_gripper = None,
        **kwargs
    ):
        loss = {}
        assert (
            vision_x is not None
        )
        # bs, seq_len = vision_x.shape[:2]
        # eos_offset = int(self.tokenizer.eos_token is not None)
        bos_offset = int(self.tokenizer.bos_token is not None)
        
        input_embeds = self.word_embedding(input_ids)
        
        (
            multimodal_embeds,
            mutlimodal_labels, 
            multimodal_attention_mask,
            _
        ) = self.merge_multi_modal_input(
            input_embeds,
            vision_x,
            labels=labels,
            attention_mask=attention_mask,
            insert_idx=bos_offset
        )
        
        if vision_gripper is not None:
            (
                multimodal_embeds, 
                mutlimodal_labels, 
                multimodal_attention_mask,
                _
            ) = self.merge_multi_modal_input(
                multimodal_embeds,
                vision_gripper,
                mutlimodal_labels,
                multimodal_attention_mask,
                insert_idx=bos_offset
            )
        
        output = self.qwen_model(
            input_ids=None,
            inputs_embeds=multimodal_embeds,
            attention_mask=multimodal_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=True,
        )
        
        logits = output.logits
        _, caption_loss = self._forward_caption(logits, mutlimodal_labels)
        self._update_loss(loss, caption_loss, 'cap')
        loss = self._format_loss(loss)
        return loss




    def pred_action_discrete(
        self,
        instr_and_action_ids,
        vision_x,
        vision_gripper=None,
        attention_mask=None
    ):
        assert (
            vision_x is not None
        )
        input_embeds = self.word_embedding(instr_and_action_ids)
        
        (
            multimodal_embeds, 
            _, 
            multimodal_attention_mask
        ) = self.merge_multi_modal_input(
            input_embeds,
            vision_x,
            attention_mask=attention_mask
        )

        if vision_gripper is not None:
            (
            multimodal_embeds, 
            _, 
            multimodal_attention_mask
            ) = self.merge_multi_modal_input(
                multimodal_embeds,
                vision_gripper,
                attention_mask=multimodal_attention_mask
            )
            
        action_dim = self.act_head_configs['action_dim']
        
        # self.qwen_model.main_input_name = 'inputs_embeds'
        # generated_ids = self.qwen_model.generate(inputs_embeds=multimodal_embeds, max_new_tokens=action_dim)
        
        generated_ids = []
        kv_cache = None
        for i in range(action_dim):
            if kv_cache is None:
                output_hs = self.qwen_model(inputs_embeds=multimodal_embeds, past_key_values=kv_cache, use_cache=True)
            else:
                output_hs = self.qwen_model(inputs_embeds=multimodal_embeds[:, -1:], past_key_values=kv_cache, use_cache=True)
            kv_cache = output_hs.past_key_values
            cur_id = output_hs.logits[:, -1].argmax(dim=-1)
            generated_ids.append(cur_id)
            cur_embed = self.word_embedding(cur_id)
            multimodal_embeds = torch.cat([multimodal_embeds, cur_embed.unsqueeze(1)], dim=1)
        generated_ids = torch.cat(generated_ids, dim=0).unsqueeze(0)
        
        # print(generated_ids.shape)
        predicted_action_ids = generated_ids[0, -action_dim:].cpu().numpy()
        discretized_actions = self.act_head.action_tokenizer.decode_token_ids_to_actions(predicted_action_ids)
        
        if isinstance(discretized_actions, list):
            discretized_actions = np.array(discretized_actions)
        # discretized_actions[:, -1] = np.where(discretized_actions[:, -1] > 0, 1, -1)
        discretized_actions[-1] = 1 if discretized_actions[-1] > 0 else -1
        
        return discretized_actions

    # def forward_continuous(
    #         self,
    #         vision_x: torch.Tensor,
    #         lang_x: torch.Tensor,
    #         attention_mask: torch.Tensor = None,
    #         position_ids: torch.LongTensor = None,
    #         use_cached_vision_x: bool = False, # TODO: Do we need this? If not we can remove it from here
    #         action_labels: Tuple[torch.Tensor, torch.Tensor] = None,
    #         action_mask: torch.Tensor = None,
    #         caption_labels: torch.Tensor = None,
    #         caption_mask: torch.Tensor = None,
    #         past_key_values=None,
    #         use_cache: bool = False,
    #         vision_gripper = None,
    #         fwd_rgb_labels: torch.Tensor = None,
    #         fwd_hand_rgb_labels: torch.Tensor = None,
    #         fwd_mask: torch.Tensor = None,
    #         instr_and_action_ids=None,
    #         instr_and_action_labels=None,
    #         instr_and_action_mask=None,
    #         **kwargs
    # ):  
    #     loss = {}

    #     assert (
    #         vision_x is not None
    #     )
    #     bs, seq_len = vision_x.shape[:2]
    #     action_space = self.act_head_configs.get("action_space", "continuous")
    #     # if not self.use_hand_rgb:
    #     #     vision_gripper = None
    #     if self.train_setup_configs['predict_action'] and action_labels is not None:
    #         if seq_len > 1 and self.act_head_configs.get("with_history", False):
    #             vision_x = vision_x.reshape(bs*seq_len, *vision_x.shape[2:]).unsqueeze(1)
    #             lang_x = lang_x.repeat(seq_len, 1)
    #             attention_mask = attention_mask.repeat(seq_len, 1)
    #             if vision_gripper is not None:
    #                 vision_gripper = vision_gripper.reshape(bs*seq_len, *vision_gripper.shape[2:]).unsqueeze(1)
    #             # if action_labels is not None:
    #             #     action_labels = (action_labels[0].reshape(bs*seq_len, *action_labels[0].shape[2:]), action_labels[1].reshape(bs*seq_len, *action_labels[1].shape[2:]))
    #         (
    #             input_ids,
    #             position_ids,
    #             attention_mask,
    #             past_key_values,
    #             inputs_embeds,
    #             new_lang_mask, 
    #             new_action_mask, 
    #             new_vision_mask
    #         ) = self.prepare_inputs_labels_for_action_prediction(
    #             vision_x,
    #             lang_x,
    #             attention_mask,
    #             action_labels,
    #             action_mask,
    #             vision_gripper=vision_gripper,
    #         )

    #         output = self.qwen_model(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             position_ids=position_ids,
    #             past_key_values=past_key_values,
    #             inputs_embeds=inputs_embeds,
    #             use_cache=use_cache,
    #             output_hidden_states=True
    #         )
            
    #         if action_space == "continuous":
    #             output_hs = output.hidden_states[-1].clone()
    #             # print(output_hs.shape)
    #             action_hs = torch.masked_select(output_hs, new_action_mask.cuda().bool().unsqueeze(-1)).reshape(bs, seq_len, self.latent_num, -1)
    #         elif action_space == "down_sample":
    #             action_hs = output.hidden_states[-1].clone()
    #             action_hs = action_hs.reshape(bs, seq_len, *action_hs.shape[1:])
    #         elif action_space == 'discrete':
    #             raise NotImplementedError("Discrete version of action prediction is not implemented yet")
    #         else:
    #             raise ValueError(f"Unknown action space {action_space}")
            
    #         action_logits, action_loss = self._forward_action_head(action_hs, action_labels, action_mask)
    #         self._update_loss(loss, action_loss, 'act')
            
    #         loss = self._format_loss(loss)

    #     return loss


    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        images=None,
        **kwargs
    ):
        
        loss = {}
        start_image_token_id = torch.tensor([self.visual_config['image_start_id']]).to(self.qwen_model.device)
        end_image_token_id = start_image_token_id + 1
        
        image_start_embed = self.word_embedding(start_image_token_id).to(self.qwen_model.device)
        image_end_embed = self.word_embedding(end_image_token_id).to(self.qwen_model.device)
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            image_start_embed = image_start_embed.unsqueeze(0).unsqueeze(0).repeat(*image_features.shape[:2], 1, 1)
            image_end_embed = image_end_embed.unsqueeze(0).unsqueeze(0).repeat(*image_features.shape[:2], 1, 1)
            image_features = torch.cat([image_start_embed, image_features, image_end_embed], dim=2)
            # print(image_features.shape)
            image_features = image_features.squeeze(1) # squeeze over the seq_len dim (unsqueezed in encode_images)
            # print(image_features.shape)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [_.squeeze(0) for _ in image_features]
            # print(len(image_features), image_features[0].shape)
        else:
            image_features = self.encode_images(images)
            image_start_embed = image_start_embed.unsqueeze(0).unsqueeze(0).repeat(*image_features.shape[:2], 1, 1)
            image_end_embed = image_end_embed.unsqueeze(0).unsqueeze(0).repeat(*image_features.shape[:2], 1, 1)
            image_features = torch.cat([image_start_embed, image_features, image_end_embed], dim=2)
            image_features = image_features.squeeze(1) # squeeze over the seq_len dim (unsqueezed in encode_images)
        # print(len(image_features), image_features[0].shape)
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.word_embedding(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.word_embedding(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.qwen_model.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            new_input_embeds_padded.append(torch.cat((
                cur_new_embed,
                torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
            ), dim=0))
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        new_labels = new_labels.long()
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def forward_vl_task(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        images=None,
        image_sizes: Optional[List[List[int]]] = None,
        **kwargs
    ):
        loss = {}

        if inputs_embeds is None:
            (
                _,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                labels=labels,
                images=images
            )

        output = self.qwen_model(
            input_ids=None,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_hidden_states=True
        )
        
        self._update_loss(loss, {"loss_vl": output.loss}, 'cotrain')
        return loss
    
    def forward_continuous_v2(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        use_cached_vision_x: bool = False, # TODO: Do we need this? If not we can remove it from here
        action_labels: Tuple[torch.Tensor, torch.Tensor] = None,
        action_mask: torch.Tensor = None,
        caption_labels: torch.Tensor = None,
        caption_mask: torch.Tensor = None,
        past_key_values=None,
        use_cache: bool = False,
        vision_gripper = None,
        fwd_rgb_labels: torch.Tensor = None,
        fwd_hand_rgb_labels: torch.Tensor = None,
        fwd_mask: torch.Tensor = None,
        instr_and_action_ids=None,
        instr_and_action_labels=None,
        instr_and_action_mask=None,
        mode='train',
        **kwargs
    ): 
        loss = {}
        assert (
            vision_x is not None
        )
        bs, seq_len = vision_x.shape[:2]
        action_space = self.act_head_configs.get("action_space", "continuous")

        eos_offset = int(self.tokenizer.eos_token is not None)
        bos_offset = int(self.tokenizer.bos_token is not None)
        
        history_type = self.act_head_configs.get('history_type', 'post')
        # if seq_len > 1 and self.act_head_configs.get("with_history", False):
        if history_type in ['post', 'pre']:
            vision_x = vision_x.reshape(bs*seq_len, *vision_x.shape[2:]).unsqueeze(1)
            lang_x = lang_x.unsqueeze(1).repeat(1, seq_len, 1).flatten(0, 1)
            attention_mask = attention_mask.unsqueeze(1).repeat(1, seq_len, 1).flatten(0, 1)
            if vision_gripper is not None:
                vision_gripper = vision_gripper.reshape(bs*seq_len, *vision_gripper.shape[2:]).unsqueeze(1)
        
        input_embeds = self.word_embedding(lang_x)
        
        # lang_size = lang_x.shape[-1]
        lang_size = lang_x.shape[-1] - int(self.tokenizer.eos_token is not None) - int(self.tokenizer.bos_token is not None)
        
        (
            multimodal_embeds,
            mutlimodal_labels,
            multimodal_attention_mask,
            _
        ) = self.merge_multi_modal_input(
            input_embeds,
            vision_x,
            labels=None,
            attention_mask=attention_mask,
            insert_idx=bos_offset
        )
        # print(f'step 1: multi-embed shape: {multimodal_embeds.shape} {multimodal_embeds.dtype}')
        if vision_gripper is not None:
            (
                multimodal_embeds,
                mutlimodal_labels,
                multimodal_attention_mask,
                _
            ) = self.merge_multi_modal_input(
                multimodal_embeds,
                vision_gripper,
                mutlimodal_labels,
                multimodal_attention_mask,
                insert_idx=bos_offset
            )
        
        # print(f'step 2: multi-embed shape: {multimodal_embeds.shape} {multimodal_embeds.dtype} use_gripper: {vision_gripper is not None}')
        if action_space == "continuous":
            action_token_insert_idx = multimodal_embeds.shape[1] - int(self.tokenizer.eos_token is not None) # insert at last

            action_tokens = repeat(self.action_token, 'd -> b n d', b=multimodal_embeds.shape[0], n=self.latent_num)
            (
                multimodal_embeds, 
                mutlimodal_labels, 
                multimodal_attention_mask,
                action_token_mask
            ) = self.merge_multi_modal_input(
                multimodal_embeds,
                action_tokens,
                mutlimodal_labels,
                multimodal_attention_mask,
                is_image=False,
                insert_idx=action_token_insert_idx,
                fill_zero=self.act_head_configs.get('fill_zero', False)
            )
        
        if history_type == 'pre':
            multimodal_embeds = rearrange(multimodal_embeds, '(b l) n d -> b (l n) d', l=seq_len)
            if multimodal_attention_mask is not None:
                multimodal_attention_mask = rearrange(multimodal_attention_mask, '(b l) n -> b (l n)', l=seq_len)
            
        output = self.qwen_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=multimodal_embeds,
            use_cache=use_cache,
            output_hidden_states=True
        )
        
        output_hs = output.hidden_states[-1].clone()
        # print(f'step 3: multi-embed shape: {multimodal_embeds.shape} {multimodal_embeds.dtype}')
        if history_type == 'pre':
            multimodal_embeds = rearrange(multimodal_embeds, 'b (l n) d -> (b l) n d', l=seq_len)
            output_hs = rearrange(output_hs, 'b (l n) d -> (b l) n d', l=seq_len)
        
        if history_type == 'video':
            seq_len = 1
        
        if action_space == "continuous":
            # tmp_mask = torch.all(multimodal_embeds == self.action_token, dim=-1)
            # action_hs = output_hs[tmp_mask].reshape(bs, seq_len, self.latent_num, -1)
            action_hs = output_hs[action_token_mask].reshape(bs, seq_len, self.latent_num, -1)

        elif action_space == "down_sample":
            action_hs = output_hs
            token_src = self.act_head_configs.get("token_source", "all")
            
            if token_src == 'text':
                # fetch the language tokens
                action_hs = action_hs[:, -lang_size-eos_offset:action_hs.shape[1]-eos_offset].reshape(bs, seq_len, lang_size, -1)
            elif token_src == 'vision':
                action_hs = action_hs[:, bos_offset:-lang_size-eos_offset].reshape(bs, seq_len, -1, action_hs.shape[-1])
            elif token_src == 'all':
                action_hs = action_hs.reshape(bs, seq_len, *action_hs.shape[1:])
            else:
                raise ValueError(f"Unsupported token source {token_src}")
        
        else:
            raise ValueError(f"Unsupported action space {action_space}")
        
        if history_type == 'video' and action_hs.ndim == 4:
            action_hs = action_hs.squeeze(1) # squeeze the seq_len dim
        
        action_logits, action_loss = self._forward_action_head(action_hs, action_labels, action_mask)
        
        if mode == 'train':
            self._update_loss(loss, action_loss, 'act')
            loss = self._format_loss(loss)
        else:
            return action_logits

        return loss
    
    def forward_action(
            self,
            vision_x: torch.Tensor,
            lang_x: torch.Tensor,
            attention_mask: torch.Tensor = None,
            position_ids: torch.LongTensor = None,
            use_cached_vision_x: bool = False, # TODO: Do we need this? If not we can remove it from here
            action_labels: Tuple[torch.Tensor, torch.Tensor] = None,
            action_mask: torch.Tensor = None,
            caption_labels: torch.Tensor = None,
            caption_mask: torch.Tensor = None,
            past_key_values=None,
            use_cache: bool = False,
            vision_gripper = None,
            fwd_rgb_labels: torch.Tensor = None,
            fwd_hand_rgb_labels: torch.Tensor = None,
            fwd_mask: torch.Tensor = None,
            instr_and_action_ids=None,
            instr_and_action_labels=None,
            instr_and_action_mask=None,
            raw_text=None,
            **kwargs
    ):
        action_space = self.act_head_configs.get("action_space", "continuous")

        if action_space == 'discrete':
            return self.forward_discrete(
                vision_x=vision_x, 
                lang_x=lang_x,
                attention_mask=attention_mask, 
                action_labels=action_labels,
                action_mask=action_mask,
                caption_labels=caption_labels,
                caption_mask=caption_mask,
                vision_gripper=vision_gripper,
                fwd_rgb_labels=fwd_rgb_labels,
                fwd_hand_rgb_labels=fwd_hand_rgb_labels,
                fwd_mask=fwd_mask,
                instr_and_action_ids=instr_and_action_ids, 
                instr_and_action_labels=instr_and_action_labels, 
                instr_and_action_mask=instr_and_action_mask
            )
        else:
            return self.forward_continuous_v2(
                vision_x=vision_x, 
                lang_x=lang_x,
                attention_mask=attention_mask, 
                action_labels=action_labels,
                action_mask=action_mask,
                caption_labels=caption_labels,
                caption_mask=caption_mask,
                vision_gripper=vision_gripper,
                fwd_rgb_labels=fwd_rgb_labels,
                fwd_hand_rgb_labels=fwd_hand_rgb_labels,
                fwd_mask=fwd_mask,
                instr_and_action_ids=instr_and_action_ids, 
                instr_and_action_labels=instr_and_action_labels, 
                instr_and_action_mask=instr_and_action_mask,
                raw_text=raw_text
            )
    
    def forward(
            self,
            vision_x: torch.Tensor,
            lang_x: torch.Tensor,
            attention_mask: torch.Tensor = None,
            position_ids: torch.LongTensor = None,
            use_cached_vision_x: bool = False, # TODO: Do we need this? If not we can remove it from here
            action_labels: Tuple[torch.Tensor, torch.Tensor] = None,
            action_mask: torch.Tensor = None,
            caption_labels: torch.Tensor = None,
            caption_mask: torch.Tensor = None,
            past_key_values=None,
            use_cache: bool = False,
            vision_gripper = None,
            fwd_rgb_labels: torch.Tensor = None,
            fwd_hand_rgb_labels: torch.Tensor = None,
            fwd_mask: torch.Tensor = None,
            instr_and_action_ids=None,
            instr_and_action_labels=None,
            instr_and_action_mask=None,
            raw_text=None,
            data_source=[],
            **kwargs
    ):
        loss = {}
        if isinstance(data_source, list):
            data_source = '_'.join(data_source)
        
        if 'action' in data_source:
            tmp_loss = self.forward_action(
                vision_x=vision_x, 
                lang_x=lang_x,
                attention_mask=attention_mask, 
                action_labels=action_labels,
                action_mask=action_mask,
                caption_labels=caption_labels,
                caption_mask=caption_mask,
                vision_gripper=vision_gripper,
                fwd_rgb_labels=fwd_rgb_labels,
                fwd_hand_rgb_labels=fwd_hand_rgb_labels,
                fwd_mask=fwd_mask,
                instr_and_action_ids=instr_and_action_ids, 
                instr_and_action_labels=instr_and_action_labels, 
                instr_and_action_mask=instr_and_action_mask
            )
            loss = self._update_loss(loss, tmp_loss)
        
        if 'vl_pretrain' in data_source:
            tmp_loss = self.forward_vl_task(
                input_ids=instr_and_action_ids, 
                labels=instr_and_action_labels, 
                attention_mask=instr_and_action_mask,
                images=vision_x
            )
            loss = self._update_loss(loss, tmp_loss)
        
        if 'future_frame' in data_source:
            tmp_loss = self.forward_image_prediction(
                vision_x=vision_x,
                vision_gripper=vision_gripper,
                lang_x=lang_x,
                fwd_rgb_labels=fwd_rgb_labels,
                fwd_hand_rgb_labels=fwd_hand_rgb_labels,
                fwd_mask=fwd_mask
            )
            loss = self._update_loss(loss, tmp_loss)

        if 'video_caption' in data_source:
            tmp_loss = self.forward_video_caption(
                vision_x=vision_x,
                vision_gripper=vision_gripper,
                input_ids=instr_and_action_ids,
                labels=instr_and_action_labels,
                attention_mask=instr_and_action_mask,
            )
            loss = self._update_loss(loss, tmp_loss)
        
        return loss
    
    def inference(
            self,
            vision_x: torch.Tensor,
            lang_x: torch.Tensor,
            attention_mask: torch.Tensor = None,
            position_ids: torch.LongTensor = None,
            use_cached_vision_x: bool = False, # TODO: Do we need this? If not we can remove it from here
            action_labels: Tuple[torch.Tensor, torch.Tensor] = None,
            action_mask: torch.Tensor = None,
            caption_labels: torch.Tensor = None,
            caption_mask: torch.Tensor = None,
            past_key_values=None,
            use_cache: bool = False,
            vision_gripper = None,
            **kwargs
        ):

        prediction = {}

        assert (
            vision_x is not None
        )
        bs, seq_len = vision_x.shape[:2]
        action_space = self.act_head_configs.get("action_space", "continuous")
        if self.train_setup_configs['predict_action']:
            
            if action_space == 'discrete':
                action = self.pred_action_discrete(
                    lang_x,
                    vision_x,
                    vision_gripper,
                    attention_mask
                )
                prediction['action'] = action
            
            else:
                prediction['action'] = self.forward_continuous_v2(
                    vision_x,
                    lang_x,
                    attention_mask,
                    vision_gripper=vision_gripper,
                    mode='inference'
                )
        
        return prediction

    def test_trainable(self, model=None):
        model = model or self.qwen_model
        for name, parameter in model.named_parameters():
            print(parameter.requires_grad, name)

def deep_update(d1, d2):
    # use d2 to update d1
    for k, v in d2.items():
        if isinstance(v, dict) and k in d1:
            assert isinstance(d1[k], dict)
            deep_update(d1[k], d2[k])
        else:
            d1[k] = d2[k]
    return d1

import json
def load_config(config_file):
    _config = json.load(open(config_file))
    config = {}
    if _config.get('parent', None):
        deep_update(config, load_config(_config['parent']))
    deep_update(config, _config)
    return config

if __name__ == "__main__":
    # configs = load_config('/mnt/bn/liuminghuan/RobotVLM/configs/llava/finetune_llava1.6_7b.json')
    configs = load_config('/mnt/bn/robotics-data-lxh-lq/RoboVLM/configs/llava/policy_head/finetune_llava1.6_7b_vicuna_cont_1_latent.json')
    llava_config = load_config('/mnt/bn/robotics-data-lxh-lq/LLaVA/llava-v1.6-vicuna-7b/config.json')
    use_hand_rgb = False # True
    model = RoboQwen(
        qwen_model_name_or_path='llava1.6_vicuna_7b',
        train_setup_configs=configs['train_setup'],
        fwd_head_configs=None,
        window_size=configs['window_size'],
        use_hand_rgb=use_hand_rgb,
        act_head_configs=configs['act_head'],
        fwd_pred_next_n=configs['fwd_pred_next_n']
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"LLaVA Model Parameters: {total_params / 1000000:.2f}M")
    bs, seq_len = 2, 1
    device = "cuda:0"
    device = 'cpu'
    vision_x = torch.zeros((bs, seq_len, 3, 336, 336), dtype=torch.float16).to(device)
    vision_gripper = torch.zeros((bs, seq_len, 3, 336, 336), dtype=torch.float16).to(device)
    lang_x = torch.ones((bs, 10), dtype=torch.long).to(device) * 100
    attention_mask = torch.ones((bs, 10)).bool().to(device)
    action_lables = (torch.randn(bs, seq_len, 1, 6).to(device), torch.zeros(bs, seq_len, 1).to(device))
    model = model.to(device).to(torch.float16)
    test_res = model(
        vision_x,
        lang_x,
        attention_mask= attention_mask,
        position_ids= None,
        use_cached_vision_x = False,
        action_labels = action_lables,
        action_mask = None,
        caption_labels = None,
        caption_mask = None,
        past_key_values = None,
        use_cache = False,
        vision_gripper = vision_gripper,
        fwd_rgb_labels = None,
        fwd_hand_rgb_labels = None,
        fwd_mask = None,
    )

    print(test_res)