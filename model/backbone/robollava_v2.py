import imp
import torch
from torch import nn
import copy
from typing import Tuple
from einops import rearrange, repeat
import json
import os, sys, copy
sys.path.append('/mnt/bn/liuminghuan/RobotVLM')
from model.llm.robollava import build_llava
from utils.model_utils import build_tokenizer, get_target_modal_tokens
from model.policy_head.action_tokenizer import ActionTokenizer
import numpy as np

# from llava.model.multimodal_encoder.builder import build_vision_tower
# from llava.model.multimodal_projector.builder import build_vision_projector
# from llava.train.train import find_all_linear_names
# from llava.model.llava_arch import unpad_image
# from llava.mm_utils import get_anyres_image_grid_shape
# from llava import conversation as conversation_lib
from train.loss import calculate_vl_cross_entropy

class RoboLLaVA(nn.Module):

    def __init__(
            self,
            llava_path,
            llava_config,
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
        assert isinstance(llava_path, str)
        self.llava_path = llava_path
        self.llava_config = llava_config

        self.train_setup_configs = train_setup_configs
        self.act_encoder_configs = act_encoder_configs
        self.act_head_configs = act_head_configs
        self.fwd_head_configs = fwd_head_configs
        self.vision_resampler_configs= vision_resampler_configs

        self.use_hand_rgb = use_hand_rgb
    
        self.tokenizer, self.llava_model = self._init_llava() # Including loading pre-trained ckpts
        
        self.vision_tower, self.mm_projector = self.llava_model.get_vision_tower(), self.llava_model.get_model().mm_projector
        self.lang_decoder = self.llava_model.get_model()
        self.act_head, self.fwd_head = self._init_heads()
        self.action_space = self.act_head_configs.get("action_space", "continuous")

        if self.action_space == 'discrete':
            self.action_tokenizer = ActionTokenizer(self.tokenizer, bins=self.act_head_configs['n_bin'], \
            min_action=self.act_head_configs['min_action'], max_action=self.act_head_configs['max_action'])

        if self.action_space == "continuous":
            self.action_token = nn.Parameter(torch.zeros(self.hidden_size))
            self.action_token.requires_grad_(True)
        # elif self.action_space == "discrete":
        #     action_bin = self.act_head_configs.get("action_bin", 256)
        #     self.action_embeddings = nn.Embedding(action_bin, self.hidden_size)
        # else:
        #     assert self.action_space == "down_sample", "Unsupported action space"
        
        ### setup vision tower and configs
        self.vis_dim = self.llava_model.get_vision_tower().hidden_size
        self.use_vision_resampler = use_vision_resampler
        if self.use_vision_resampler:
            from model.vision_encoder.vision_resampler import PerceiverResampler
            self.vision_resampler = PerceiverResampler(dim=self.hidden_size)

        self._trainable_params_setup()

    def _init_llava(self):
        tokenizer, llava_model = build_llava(self.llava_path)
        if hasattr(llava_model.config, "d_model"):
            self.hidden_size = llava_model.config.d_model  # mpt uses d_model
        else:
            self.hidden_size = llava_model.config.hidden_size

        return tokenizer, llava_model

    def encode_images(self, images, image_sizes=None):
        # input: images: list of b,c,h,w or b,t,c,h,w
        # output: image_features: list of bx[txn, d]
        if images.ndim == 4:
            images = images.unsqueeze(1)
        bs, seq_len = images.shape[:2]
        # import pdb; pdb.set_trace()
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.llava_model.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.llava_config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.llava_config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.llava_model.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            from llava.mm_utils import get_anyres_image_grid_shape
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.llava_config.image_grid_pinpoints, self.llava_model.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            from llava.model.llava_arch import unpad_image
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.llava_model.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.llava_model.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.llava_config.mm_patch_merge_type}")
        else:
            image_features = self.llava_model.encode_images(images)
        
        image_features = torch.stack(image_features, dim=0).view(bs, seq_len, -1, image_features[0].shape[-1])
        
        if self.use_vision_resampler:
            ### downsample at token num dim: b, s, n, d -> b, s, v d
            # b T F v d -> b, T, n, d
            # import pdb;pdb.set_trace()
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
        # import pdb; pdb.set_trace()
        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_new_input_embeds = [self.lang_decoder.embed_tokens(cur_input_ids)]
            
            cur_language_mask = [torch.ones(cur_input_ids.shape[0]).to(dtype=torch.long, device=cur_input_ids.device)]
            cur_action_mask = [torch.zeros(cur_input_ids.shape[0]).to(dtype=torch.long, device=cur_input_ids.device)]
            cur_vision_mask = [torch.zeros(cur_input_ids.shape[0]).to(dtype=torch.long, device=cur_input_ids.device)]

            for step_idx in range(seq_len):
                rgb_toks = rgb_feats[batch_idx, step_idx]
                cur_new_input_embeds.append(rgb_toks)
                cur_language_mask.append(torch.zeros(rgb_toks.shape[0]).to(dtype=torch.long, device=cur_input_ids.device))
                cur_action_mask.append(torch.zeros(rgb_toks.shape[0]).to(dtype=torch.long, device=cur_input_ids.device))
                cur_vision_mask.append(torch.ones(rgb_toks.shape[0]).to(dtype=torch.long, device=cur_input_ids.device))

                if gripper_feats is not None:
                    gripper_toks = gripper_feats[batch_idx, step_idx]
                    cur_new_input_embeds.append(gripper_toks)
                    cur_language_mask.append(torch.zeros(gripper_toks.shape[0]).to(dtype=torch.long, device=cur_input_ids.device))
                    cur_action_mask.append(torch.zeros(gripper_toks.shape[0]).to(dtype=torch.long, device=cur_input_ids.device))
                    cur_vision_mask.append(torch.ones(gripper_toks.shape[0]).to(dtype=torch.long, device=cur_input_ids.device))
                
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
            if getattr(self.llava_config, 'tokenizer_padding_side', 'right') == "left":
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

    @property
    def eos_offset(self):
        return int(self.tokenizer.eos_token is not None)

    def cat_multi_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_ids: torch.Tensor = None,
        insert_idx: int=0,
        attention_masks: torch.Tensor = None,
        return_insert_masks = False
    ):
        bs, seq_len = input_ids.shape[:2]
        device = input_ids.device
        if insert_idx >= 0:
            return_ids = torch.cat((input_ids[:, :insert_idx], multimodal_ids, input_ids[:, insert_idx:]), dim=1)
            if return_insert_masks:
                insert_masks = torch.cat((torch.zeros(bs, insert_idx), torch.ones(multimodal_ids.shape), torch.zeros(bs, seq_len - insert_idx)), dim=1)

        elif insert_idx == -1 and attention_masks is not None:
            new_id_list = []
            new_mask_list = []
            for mask, input_id, multimodal_id in zip(attention_masks, input_ids, multimodal_ids):
                indexs = (mask == False).nonzero()
                insert_idx = indexs[0].item() if len(indexs) > 0 else len(mask)
                insert_idx -= self.eos_offset
                new_id = torch.cat((input_id[:insert_idx], multimodal_id, input_id[insert_idx:]), dim=0)
                new_id_list.append(new_id)
                if return_insert_masks:
                    new_mask = torch.cat((torch.zeros(insert_idx), torch.ones(multimodal_id.shape), torch.zeros(seq_len - insert_idx)), dim=0)
                    new_mask_list.append(new_mask)
            return_ids = torch.stack(new_id_list, dim=0)
            if return_insert_masks:
                insert_masks = torch.stack(new_mask_list, dim=0)
        else:
            raise Exception("insert_idx should be -1 or >= 0, and if you want to insert as last(-1), you should provide masks")
        return_ids = return_ids.to(device)
        if return_insert_masks:
            insert_masks = insert_masks.to(device).bool()
            return return_ids, insert_masks
        else:
            return return_ids
        
    def merge_multi_modal_input(
        self,
        input_embeds: torch.Tensor,
        multi_modal_embeds: torch.Tensor = None,
        labels: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        # is_raw_feature=True,
        insert_idx=1,
    ):
        # bs, seq_len = vision_x.shape[:2]
        # vision_x = vision_x.reshape(bs*seq_len, *vision_x.shape[2:])
        
        # if is_raw_feature:
        #     rgb_feats = self.encode_images(vision_x.clone())
        #     rgb_feats = rearrange(rgb_feats, 'b l n d -> b (l n) d') #flatten seq_len and n_tok_per_img dim
        # else:
        #     rgb_feats = vision_x
        
        # multimodal_embeds = torch.cat(
        #     [input_embeds[:, :insert_idx], rgb_feats, input_embeds[:, insert_idx:]],
        #     dim=1
        # )
        
        all_embeds, insert_masks = self.cat_multi_input_ids(input_embeds, multi_modal_embeds, insert_idx, attention_mask, return_insert_masks=True)

        all_attention_mask = None
        if attention_mask is not None:
            mutli_modal_mask = torch.full(
                multi_modal_embeds.shape[:2], True, dtype=attention_mask.dtype, device=attention_mask.device
            )
            all_attention_mask = self.cat_multi_input_ids(attention_mask, mutli_modal_mask, insert_idx, return_insert_masks=False)

        mutlimodal_labels = None
        if labels is not None:
            multi_modal_labels = torch.full(
                multi_modal_embeds.shape[:2], -100, dtype=labels.dtype, device=labels.device
            )
            mutlimodal_labels = self.cat_multi_input_ids(labels, multi_modal_labels, insert_idx, return_insert_masks=False)
        
        # print("multimodal_embeds", multimodal_embeds.shape, "multimodal_attention_mask", multimodal_attention_mask.shape, "mutlimodal_labels", mutlimodal_labels.shape)
        return all_embeds, mutlimodal_labels, all_attention_mask, insert_masks
    
    def _init_heads(self):
        action_head = None
        if self.act_head_configs is not None:
            import model.policy_head as action_heads
            _kwargs = copy.deepcopy(self.act_head_configs)
            _kwargs.update(
                dict(# hidden_size=self.hidden_size, 
                    tokenizer=self.tokenizer,
                    in_features=self.hidden_size,
                    fwd_pred_next_n=self.fwd_pred_next_n,
                    window_size=self.window_size,
                    n_bin=self.act_head_configs.get('n_bin', 256),
                    min_action=self.act_head_configs.get('min_action', -1),
                    max_action=self.act_head_configs.get('max_action', 1)
                    ))
            _cls = getattr(action_heads, _kwargs.pop('type'))
            self.latent_num = self.act_head_configs.get("latent", 1)
            action_head = _cls(**_kwargs)

        fwd_decoder = None
        if self.fwd_head_configs is not None:
            import model.forward_head as fwd_heads
            _kwargs = copy.deepcopy(self.fwd_head_configs)
            _kwargs.update(dict(image_size=self.vision_encoder.image_size,
                                patch_size=self.vision_encoder.patch_size,
                                hidden_size=self.hidden_size))
            _cls = getattr(fwd_heads, _kwargs.pop('type'))
            if self.use_mim_obs_loss:
                _kwargs['fwd_pred_next_n'] = 0
            fwd_decoder = _cls(**_kwargs)

        return action_head, fwd_decoder
    
    def _trainable_params_setup(self):
        # TODO: merge precision into trainer
        model = self.llava_model
        compute_dtype = torch.float32 # (torch.float16 if self.train_setup_configs['precision'] == 'fp16' else (torch.bfloat16 if self.train_setup_configs['precision'] == 'bf16' else torch.float32))

        model.config.use_cache = False

        if self.train_setup_configs['bits'] in [4, 8]: # TODO: in trainer
            from peft import prepare_model_for_kbit_training
            model.config.torch_dtype=(torch.float32 if self.train_setup_configs['precision'] == 'fp16' else (torch.bfloat16 if self.train_setup_configs['precision'] == 'bf16' else torch.float32))
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=self.train_setup_configs['gradient_checkpointing'])
            model.model.requires_grad_(True)
            
        if self.train_setup_configs['freeze_backbone']:
            model.model.requires_grad_(False)

        if self.train_setup_configs['gradient_checkpointing']:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        
        if self.train_setup_configs['lora_enable']:
            from llava.train.train import find_all_linear_names
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=self.train_setup_configs['lora_r'],
                lora_alpha=self.train_setup_configs['lora_alpha'],
                target_modules=find_all_linear_names(model),
                lora_dropout=self.train_setup_configs['lora_dropout'],
                bias=self.train_setup_configs['lora_bias'],
                task_type="CAUSAL_LM",
            )
            if self.train_setup_configs['bits'] == 16:
                if self.train_setup_configs['precision'] == 'bf16':
                    model.to(torch.bfloat16)
                elif self.train_setup_configs['precision'] == 'fp16':
                    model.to(torch.float16)
            print("Adding LoRA adapters...")
            self.llava_model = get_peft_model(model, lora_config)

        # model.config.image_aspect_ratio = self.data_args.image_aspect_ratio # TODO: Seems useless for now, can delete in the future
        model.config.tokenizer_padding_side = self.tokenizer.padding_side
        model.config.tokenizer_model_max_length = self.tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = self.train_setup_configs['tune_mm_mlp_adapter']
        if self.train_setup_configs['tune_mm_mlp_adapter']:
            model.requires_grad_(False)
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = self.train_setup_configs['freeze_mm_mlp_adapter']
        if self.train_setup_configs['freeze_mm_mlp_adapter']:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if self.train_setup_configs['bits'] in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype) # TODO: in trainer

        model.config.mm_use_im_start_end = self.train_setup_configs['mm_use_im_start_end']
        model.config.mm_projector_lr = self.train_setup_configs['mm_projector_lr'] # TODO: in trainer
        model.config.mm_use_im_patch_token = self.train_setup_configs['mm_use_im_patch_token']
        # model.initialize_vision_tokenizer(self.model_args, tokenizer=self.tokenizer)

        self.train_setup_configs['use_im_start_end'] = self.train_setup_configs['mm_use_im_start_end'] # TODO: in trainer
        if self.train_setup_configs['bits'] in [4, 8]: # TODO: in trainer
            from peft.tuners.lora import LoraLayer
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    if self.train_setup_configs['precision'] == 'bf16':
                        module = module.to(torch.bfloat16)
                if 'norm' in name:
                    module = module.to(torch.float32)
                if 'lm_head' in name or 'embed_tokens' in name:
                    if hasattr(module, 'weight'):
                        if self.train_setup_configs['precision'] == 'bf16' and module.weight.dtype == torch.float32:
                            module = module.to(torch.bfloat16)

        if self.train_setup_configs.get('train_text_embedding', False):
            model.get_input_embeddings().requires_grad_(True)
        
        if self.use_vision_resampler and not self.train_setup_configs.get('freeze_resmapler', False):
            self.vision_resampler.requires_grad_(True)
        
        self.act_head.requires_grad_(True)
        self.train()

    def _forward_action_head(
            self,
            action_tokens: torch.Tensor,
            action_labels: Tuple[torch.Tensor, torch.Tensor] = None,
            action_mask: torch.Tensor = None,
            **kwargs
        ):

        # action_tokens = get_target_modal_tokens(output_hs, self._action_mask(output_hs))
        action = self.act_head(action_tokens, actions=action_labels, action_masks=action_mask, **kwargs)

        action_loss = None
        if action_labels is not None:
            action, action_labels, action_mask = self.act_head.get_labels(
                action, action_labels, action_mask, tok_seq=action_tokens, **kwargs)
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
        input_embeds = self.lang_decoder.embed_tokens(instr_and_action_ids)
        
        (
            multimodal_embeds, 
            mutlimodal_labels, 
            multimodal_attention_mask
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
                multimodal_attention_mask
            ) = self.merge_multi_modal_input(
                multimodal_embeds,
                vision_gripper,
                mutlimodal_labels,
                multimodal_attention_mask
            )
        # import pdb; pdb.set_trace()
        output = self.llava_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=multimodal_embeds,
                use_cache=use_cache
        )
        # import pdb; pdb.set_trace()
        
        output_hs = output.logits
        action_logits, action_loss = self._forward_action_head(output_hs, mutlimodal_labels, instr_and_action_mask)
        self._update_loss(loss, action_loss, 'act')
         
        loss = self._format_loss(loss)

        return loss

    def forward_continuous(
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
        # import pdb; pdb.set_trace()
        bs, seq_len = vision_x.shape[:2]
        action_space = self.act_head_configs.get("action_space", "continuous")
        
        if seq_len > 1 and self.act_head_configs.get("with_history", False):
            vision_x = vision_x.reshape(bs*seq_len, *vision_x.shape[2:]).unsqueeze(1)
            lang_x = lang_x.repeat(seq_len, 1)
            attention_mask = attention_mask.repeat(seq_len, 1)
            if vision_gripper is not None:
                vision_gripper = vision_gripper.reshape(bs*seq_len, *vision_gripper.shape[2:]).unsqueeze(1)
        
        input_embeds = self.lang_decoder.embed_tokens(lang_x)
        # get <bos> & <eos> offset
        lang_size = lang_x.shape[-1] - int(self.tokenizer.eos_token is not None) - int(self.tokenizer.bos_token is not None)
        
        (
        multimodal_embeds, 
        mutlimodal_labels, 
        multimodal_attention_mask
        ) = self.merge_multi_modal_input(
            input_embeds,
            vision_x,
            labels=None,
            attention_mask=attention_mask
        )
        
        if vision_gripper is not None:
            (
            multimodal_embeds, 
            mutlimodal_labels, 
            multimodal_attention_mask
            ) = self.merge_multi_modal_input(
                multimodal_embeds,
                vision_gripper,
                mutlimodal_labels,
                multimodal_attention_mask
            )
        
        if action_space == "continuous":
            # TODO delete the minus 1
            insert_idx = multimodal_embeds.shape[1] - int(self.tokenizer.eos_token is not None) # insert at last
            
            action_tokens = repeat(self.action_token, 'd -> b n d', b=multimodal_embeds.shape[0], n=self.latent_num)
            (
            multimodal_embeds, 
            mutlimodal_labels, 
            multimodal_attention_mask
            ) = self.merge_multi_modal_input(
                multimodal_embeds,
                action_tokens,
                mutlimodal_labels,
                multimodal_attention_mask,
                is_raw_feature=False,
                insert_idx=insert_idx
            )

        output = self.llava_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=multimodal_embeds,
                use_cache=use_cache,
                output_hidden_states=True
            )
        
        # action_hs = output.hidden_states[-1][:, -lang_size:] # fetch the language tokens
        if action_space == "continuous":
            output_hs = output.hidden_states[-1].clone()
            action_hs = output_hs[:, insert_idx:insert_idx+self.latent_num].reshape(bs, seq_len, self.latent_num, -1)
        elif action_space == "down_sample":
            action_hs = output.hidden_states[-1].clone()
            token_src = self.act_head_configs.get("token_source", "all")
            eos_offset = int(self.tokenizer.eos_token is not None)
            bos_offset = int(self.tokenizer.bos_token is not None)
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
        
        action_logits, action_loss = self._forward_action_head(action_hs, action_labels, action_mask)
        if mode == 'train':
            self._update_loss(loss, action_loss, 'act')
            
            loss = self._format_loss(loss)
        else:
            return action_logits

        return loss
        

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
            **kwargs
    ):
        action_space = self.act_head_configs.get("action_space", "continuous")
        ### discard the latter visual observation is with_history is False
        ### while we can maintain the multi-step action (chunk) prediction
        
        with_history = self.act_head_configs.get("with_history", False)
        if not with_history:
            vision_x = vision_x[:, :1]
            if vision_gripper is not None:
                vision_gripper = vision_gripper[:, :1]

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
            return self.forward_continuous(
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
        input_embeds = self.lang_decoder.embed_tokens(instr_and_action_ids)
        
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
        # llava model does not directly support 'generate' method
        action_dim = self.act_head_configs['action_dim']
        # import pdb; pdb.set_trace()
        # generated_ids = self.llava_model.generate(
        #     input_embeds=multimodal_embeds,
        #     max_new_tokens=action_dim
        # )
        # try:
        #     generated_ids = self.llava_model.generate(inputs_embeds=multimodal_embeds, max_new_tokens=action_dim)
        # except:
        #     generated_ids = super(type(self.llava_model), self.llava_model).generate(inputs_embeds=multimodal_embeds, max_new_tokens=action_dim)
        # import pdb; pdb.set_trace()
        generated_ids = self.llava_model.generate(inputs_embeds=multimodal_embeds, max_new_tokens=action_dim)
        predicted_action_ids = generated_ids[0, -action_dim:].cpu().numpy()
        discretized_actions = self.action_tokenizer.decode_token_ids_to_actions(predicted_action_ids)
        
        if isinstance(discretized_actions, list):
            discretized_actions = np.array(discretized_actions)
        # discretized_actions[:, -1] = np.where(discretized_actions[:, -1] > 0, 1, -1)
        discretized_actions[-1] = 1 if discretized_actions[-1] > 0 else -1
        
        return discretized_actions
    
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
        # import pdb; pdb.set_trace()
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
                prediction['action'] = self.forward_continuous(
                    vision_x,
                    lang_x,
                    attention_mask,
                    vision_gripper=vision_gripper,
                    mode='inference'
                )
                # if seq_len > 1 and self.act_head_configs.get("with_history", False):
                #     vision_x = vision_x.reshape(bs*seq_len, *vision_x.shape[2:]).unsqueeze(1)
                #     lang_x = lang_x.repeat(seq_len, 1)
                #     attention_mask = attention_mask.repeat(seq_len, 1)
                #     if vision_gripper is not None:
                #         vision_gripper = vision_gripper.reshape(bs*seq_len, *vision_gripper.shape[2:]).unsqueeze(1)
                #     # if action_labels is not None:
                #     #     action_labels = (action_labels[0].reshape(bs*seq_len, *action_labels[0].shape[2:]), action_labels[1].reshape(bs*seq_len, *action_labels[1].shape[2:]))

                # (
                # input_ids,
                # position_ids,
                # attention_mask,
                # past_key_values,
                # inputs_embeds,
                # new_lang_mask, 
                # new_action_mask, 
                # new_vision_mask
                # ) = self.prepare_inputs_labels_for_action_prediction(
                #     vision_x,
                #     lang_x,
                #     attention_mask,
                #     action_labels,
                #     action_mask,
                #     vision_gripper=vision_gripper,
                # )
                # # import pdb; pdb.set_trace()
                # output = self.llava_model(
                #     input_ids=input_ids,
                #     attention_mask=attention_mask,
                #     position_ids=position_ids,
                #     past_key_values=past_key_values,
                #     inputs_embeds=inputs_embeds,
                #     use_cache=use_cache,
                #     output_hidden_states=True
                # )
                
                # if action_space == "continuous":
                #     output_hs = output.hidden_states[-1].clone()
                #     # print(output_hs.shape)
                #     action_hs = torch.masked_select(output_hs, new_action_mask.cuda().bool().unsqueeze(-1)).reshape(bs, seq_len, self.latent_num, -1)
                #     pass
                # elif action_space == "down_sample":
                #     action_hs = output.hidden_states[-1].clone()
                #     action_hs = action_hs.reshape(bs, seq_len, *action_hs.shape[1:])
                # else:
                #     raise NotImplementedError(f"{action_space} version of action prediction is not implemented yet")
                
                # action, _ = self._forward_action_head(action_hs, action_labels, action_mask)
                # prediction['action'] = action

        return prediction


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
    model = RoboLLaVA(
        llava_path='llava1.6_vicuna_7b',
        llava_config=llava_config,
        train_setup_configs=configs['train_setup'],
        fwd_head_configs=None,
        window_size=configs['window_size'],
        use_hand_rgb=use_hand_rgb,
        act_head_configs=configs['act_head'],
        fwd_pred_next_n=configs['fwd_pred_next_n']
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"LLaVA Model Parameters: {total_params / 1000000:.2f}M")
    bs, seq_len = 2, 2
    device = "cuda:0"
    # device = 'cpu'
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