import torch
from torch import nn
from einops import rearrange, repeat
import copy
from typing import Tuple

from collections import namedtuple
from einops import rearrange, repeat
import open_clip
import numpy as np
from utils.model_utils import build_tokenizer, get_target_modal_tokens
from model.vision_encoder.vision_transformer import clip_vision_encoder
from model.llm.flamingo import build_llm_flamingo, FLAMINGO_MODEL_CONFIGS
from train.loss import calculate_vl_cross_entropy
from model.policy_head.action_tokenizer import ActionTokenizer
from open_flamingo.src.flamingo_lm import FlamingoLMMixin
from open_flamingo.src.utils import extend_instance
from open_flamingo.src.factory import _infer_decoder_layers_attr_name
from huggingface_hub import PyTorchModelHubMixin


class RoboFlamingo(nn.Module, PyTorchModelHubMixin):

    def __init__(
            self,
            vision_encoder_configs,
            tokenizer_configs,
            llm_configs,
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
        self.vision_encoder_configs = vision_encoder_configs
        clip_vision_encoder_path = vision_encoder_configs["clip_vision_encoder_path"]
        self.vis_dim = open_clip.get_model_config(clip_vision_encoder_path)["vision_cfg"]["width"]
        self.act_encoder_configs = act_encoder_configs
        self.act_head_configs = act_head_configs
        self.fwd_head_configs = fwd_head_configs
        self.train_setup_configs = train_setup_configs
        self.llm_configs = llm_configs
        self.latent_num = self.act_head_configs.get("latent", 1)
        
        # Initialize tokenizer
        self.tokenizer = build_tokenizer(tokenizer_configs)
        self.eoc_token_id = self.tokenizer.encode("<|endofchunk|>")[-1]
        self.media_token_id = self.tokenizer.encode("<image>")[-1]
        # Initialize vision encoder
        self.vision_encoder, self.clip_preprocess, self.vis_dim = self._init_vision_encoder()

        self.lang_encoder = self._init_llm()
        from open_flamingo.src.helpers import PerceiverResampler
        self.perceiver = PerceiverResampler(dim=self.vis_dim)
        
        self.lm_head, self.fwd_head = self._init_heads()
        self.action_space = self.act_head_configs.get("action_space", "continuous")
        
        if self.action_space == 'discrete':
            self.action_tokenizer = ActionTokenizer(self.tokenizer, bins=self.act_head_configs['n_bin'], \
            min_action=self.act_head_configs['min_action'], max_action=self.act_head_configs['max_action'])
        elif self.action_space == 'continuous':
            self.action_token_id = self.tokenizer.vocab_size - 1000

        self.load_openflamingo_ckpt()
        self._trainable_params_setup()
    
    def _init_llm(self):
        lang_encoder = build_llm_flamingo(self.llm_configs)
        lang_encoder_path = self.llm_configs if isinstance(self.llm_configs, str) else self.llm_configs['pretrained_model_name_or_path']
        if "mpt-1b" in lang_encoder_path or "MPT1b" in lang_encoder_path:
            class EmbeddingFnMixin:
                def get_input_embeddings(self):
                    return self.transformer.wte

                def set_input_embeddings(self, new_embeddings):
                    self.transformer.wte = new_embeddings
            extend_instance(lang_encoder, EmbeddingFnMixin)

        extend_instance(lang_encoder, FlamingoLMMixin)
        
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
        lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
        
        lang_encoder.resize_token_embeddings(len(self.tokenizer))

        if hasattr(lang_encoder.config, "d_model"):
            self.hidden_size = lang_encoder.config.d_model  # mpt uses d_model
        else:
            self.hidden_size = lang_encoder.config.hidden_size

        lang_encoder.init_flamingo(
            media_token_id=self.media_token_id,
            lang_hidden_size=self.hidden_size,
            gradient_checkpointing=False,
            vis_hidden_size=self.vis_dim,
            cross_attn_every_n_layers=FLAMINGO_MODEL_CONFIGS[self.llm_configs['name']]['cross_attn_every_n_layers'],
        )
        
        self.num_transformer_params = sum([p.numel() for p in lang_encoder.parameters()])

        return lang_encoder
    
    def load_openflamingo_ckpt(self):
        checkpoint_path = FLAMINGO_MODEL_CONFIGS[self.llm_configs['name']]['openflamingo_checkpoint']
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        msg = self.load_state_dict(ckpt, strict=False)
        print(f"OpenFlamingo Checkpoint Loaded!")

        if self.llm_configs['residual']:
            self.lang_encoder.clone_parameters()

    def _init_vision_encoder(self):
        return clip_vision_encoder(self.vision_encoder_configs['clip_vision_encoder_path'], self.vision_encoder_configs['clip_vision_encoder_pretrained'])

    def _trainable_params_setup(self):
        self.requires_grad_(False)
        if self.train_setup_configs['train_vision']:
            self.vision_encoder.requires_grad_(True)
        if self.train_setup_configs['train_decoder_layers'] == -1:
            self.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
        else:
            assert self.train_setup_configs['train_decoder_layers'] <= len(self.lang_encoder.gated_cross_attn_layers), \
            "train_decoder_layers should be less than the number of layers in the decoder"
            ix = self.train_setup_configs['train_decoder_layers']
            for layer in self.lang_encoder.gated_cross_attn_layers[-ix:]:
                layer.requires_grad_(True)

        if self.train_setup_configs['train_full_decoder']:
            self.lang_encoder.requires_grad_(True)
        if self.train_setup_configs['train_resampler']:
            self.perceiver.requires_grad_(True)
        if self.train_setup_configs['train_text_embedding']:
            self.lang_encoder.get_input_embeddings().requires_grad_(True)
        
        self.lm_head.requires_grad_(True)

    def _init_heads(self):
        action_head = None
        if self.act_head_configs is not None:
            import model.policy_head as action_heads
            _kwargs = copy.deepcopy(self.act_head_configs)
            _kwargs.update(dict(
                tokenizer=self.tokenizer,
                in_features=self.hidden_size,
                fwd_pred_next_n=self.fwd_pred_next_n,
                window_size=self.window_size,
                n_bin=self.act_head_configs.get('n_bin', 256),
                min_action=self.act_head_configs.get('min_action', -1),
                max_action=self.act_head_configs.get('max_action', 1)
            ))
            _cls = getattr(action_heads, _kwargs.pop('type'))
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

    @staticmethod
    def _get_target_modal_tokens(tok_seq, tok_mask):
        index = tok_mask.nonzero(as_tuple=True)
        return tok_seq[index]
    
    def get_modal_tokens(self, tok_seq, tok_mask_dict, modal_name):
        assert modal_name in tok_mask_dict, f"{modal_name} not in token sequence"
        return self._get_target_modal_tokens(tok_seq, tok_mask_dict[modal_name])

    def _get_obs_embed(self, rgb):

        batch_size, seq_length, c, h, w = rgb.shape
        rgb = rgb.reshape(batch_size * seq_length, c, h, w)
        # print('rgb input shape', rgb.shape)
        patch_embeddings = self.vision_encoder.visual(rgb)[1].unsqueeze(1).unsqueeze(1) # b*l, 1, 1, v, d
        # print('path_embedding shape after vit', patch_embeddings.shape)
        # patch_embeddings = patch_embeddings.view(batch_size, seq_length, *patch_embeddings.shape[1:])
        
        # patch_embeddings = patch_embeddings.unsqueeze(1).unsqueeze(1) # b*l, 1, 1, v, d
        patch_embeddings = self.perceiver(patch_embeddings) # b*l, 1, n, d

        return patch_embeddings.reshape(batch_size, seq_length, *patch_embeddings.shape[-2:])
        # return patch_embeddings

    def _encode_vision_x(self, vision_x: torch.Tensor):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """

        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]
        assert F == 1, "Only single frame supported"

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        with torch.no_grad():
            vision_x = self.vision_encoder.visual(vision_x)[1]
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)

        vision_x = self.perceiver(vision_x)  # reshapes to (b, T, n, d)

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

        return vision_x

    def _encode_multi_vision_post_fusion(self, vision_rgb: torch.Tensor, vision_gripper: torch.Tensor = None):
        vision_rgb = self._get_obs_embed(vision_rgb)
        if vision_gripper is not None:
            vision_gripper = self._get_obs_embed(vision_gripper)
            vision_rgb = torch.cat([vision_rgb, vision_gripper], dim=2) # reshapes to (b, T, 2*n, d)

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_rgb)

        return vision_rgb

    def _forward_action_head(
            self,
            action_tokens: torch.Tensor,
            action_labels: Tuple[torch.Tensor, torch.Tensor] = None,
            action_mask: torch.Tensor = None,
            **kwargs
        ):

        # action_tokens = get_target_modal_tokens(output_hs, self._action_mask(output_hs))
        action = self.lm_head(action_tokens, actions=action_labels, action_masks=action_mask, **kwargs)

        action_loss = None
        if action_labels is not None:
            action, action_labels, action_mask = self.lm_head.get_labels(
                action, action_labels, action_mask, tok_seq=action_tokens, **kwargs)
            action_loss = self.lm_head.loss(action, action_labels, action_mask)

        return action, action_loss

    def _forward_caption(
        self,
        logits: torch.Tensor,
        caption_labels: torch.Tensor = None,
        caption_mask: torch.Tensor = None,
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
    
    def forward(
            self,
            vision_x: torch.Tensor,
            lang_x: torch.Tensor,
            attention_mask: torch.Tensor = None,
            use_cached_vision_x: bool = False,
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
        action_space = self.act_head_configs.get("action_space", "down_sample")

        # with_history = self.act_head_configs.get("with_history", False)
        # if not with_history:
        #     vision_x = vision_x[:, -1:]
        #     if vision_gripper is not None:
        #         vision_gripper = vision_gripper[:, -1:]

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
                instr_and_action_mask=instr_and_action_mask,
                **kwargs
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
                instr_and_action_mask=instr_and_action_mask,
                **kwargs
            )

    def cat_multi_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_ids: torch.Tensor = None,
        insert_idx: int=0,
        attention_masks: torch.Tensor = None,
    ):
        bs, seq_len = input_ids.shape[:2]
        device = input_ids.device
        if insert_idx >= 0:
            return_ids = torch.cat((input_ids[:, :insert_idx], multimodal_ids, input_ids[:, insert_idx:]), dim=1)
            insert_masks = torch.cat((torch.zeros(bs, insert_idx), torch.ones(multimodal_ids.shape), torch.zeros(bs, seq_len - insert_idx)), dim=1)

        elif insert_idx == -1 and attention_masks is not None:
            new_id_list = []
            new_mask_list = []
            for mask, input_id, multimodal_id in zip(attention_masks, input_ids, multimodal_ids):
                indexs = (mask == False).nonzero()
                insert_idx = indexs[0].item() if len(indexs) > 0 else len(mask)
                insert_idx -= self.eos_offset
                new_embed = torch.cat((input_id[:insert_idx], multimodal_id, input_id[insert_idx:]), dim=0)
                new_mask = torch.cat((torch.zeros(insert_idx), torch.ones(multimodal_id.shape), torch.zeros(seq_len - insert_idx)), dim=0)
                new_id_list.append(new_embed)
                new_mask_list.append(new_mask)
            return_ids = torch.stack(new_id_list, dim=0)
            insert_masks = torch.stack(new_mask_list, dim=0)
        else:
            raise Exception("insert_idx should be -1 or >= 0, and if you want to insert as last(-1), you should provide masks")
        return_ids = return_ids.to(device)
        insert_masks = insert_masks.to(device).bool()

        return return_ids, insert_masks

    @property
    def eos_offset(self):
        return int(self.tokenizer.eos_token is not None)

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
        mode="train",
        **kwargs
    ):
        loss = {}
        assert (
            vision_x is not None
        ) or use_cached_vision_x, (
            "Must provide either vision_x or use_cached_vision_x to True."
        )
        bs, seq_len = vision_x.shape[:2]
        action_space = self.act_head_configs.get("action_space", "continuous")
        if seq_len > 1:
            lang_x = lang_x.unsqueeze(1).repeat(1, seq_len, 1).flatten(0, 1)
            attention_mask = attention_mask.repeat(1, seq_len, 1).flatten(0, 1)

            vision_x = vision_x.reshape(bs*seq_len, *vision_x.shape[2:]).unsqueeze(1)
            if vision_gripper is not None:
                vision_gripper = vision_gripper.reshape(bs*seq_len, *vision_gripper.shape[2:]).unsqueeze(1)
        
        if use_cached_vision_x:
            # Case: use cached; vision_x should be cached and other
            # vision-related inputs should not be provided.
            assert (
                vision_x is None
            ), "Expect vision_x to be None when use_cached_vision_x is True."
            assert self.lang_encoder.is_conditioned()

        else:
            # Case: do not use caching (i.e. this is a standard forward pass);
            self._encode_multi_vision_post_fusion(vision_x, vision_gripper)
        if action_space == "continuous":
            action_ids = torch.full((bs*seq_len, self.latent_num), self.action_token_id).to(lang_x.device)
            tmp_action_masks = torch.ones_like(action_ids)
            input_ids, action_ids_mask = self.cat_multi_input_ids(lang_x, action_ids, -1, attention_mask)
            attention_mask, _ = self.cat_multi_input_ids(attention_mask, tmp_action_masks, -1, attention_mask)
        else:
            input_ids = lang_x
        # print(lang_x.shape, attention_mask.shape)
        output = self.lang_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask.bool(),
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=True
        )

        output_hs = output.hidden_states[-1].clone()
        
        if self.train_setup_configs['predict_action'] and action_labels is not None:
            # output_hs = output.hidden_states[-1].clone()
            if action_space == "continuous":
                action_hs = output_hs[action_ids_mask].reshape(bs, seq_len, self.latent_num, -1)
            elif action_space == "down_sample":
                action_hs = output_hs.reshape(bs, seq_len, *output_hs.shape[-2:])
            
            # action_selector = self._action_mask(output_hs)
            # output_hs = get_target_modal_tokens(output_hs, action_selector)
            # print(action_hs.shape, action_labels[0].shape)
            action_logits, action_loss = self._forward_action_head(action_hs, action_labels, action_mask)
            if mode == "train":
                self._update_loss(loss, action_loss, 'act')
            else:
                return action_logits

        if self.train_setup_configs['predict_caption'] and caption_labels is not None:
            logits = output.logits.clone()
            text_selector = self._caption_mask()
            logits = get_target_modal_tokens(logits, text_selector)
            if caption_mask is None:
                caption_mask = attention_mask
            _, caption_loss = self._forward_caption(
                logits,
                caption_labels,
                caption_mask,
            )
            self._update_loss(loss, caption_loss, 'cap')

        loss = self._format_loss(loss)

        # self.lang_encoder.clear_conditioned_layers()
        # self.lang_encoder._use_cached_vision_x = False

        return loss

    def forward_discrete(    
        self,       
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        use_cached_vision_x: bool = False,
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
        mode="train",
        **kwargs
    ):
        loss = {}
        assert (
            vision_x is not None
        ) or use_cached_vision_x, (
            "Must provide either vision_x or use_cached_vision_x to True."
        )
        if use_cached_vision_x:
            # Case: use cached; vision_x should be cached and other
            # vision-related inputs should not be provided.
            assert (
                vision_x is None
            ), "Expect vision_x to be None when use_cached_vision_x is True."
            assert self.lang_encoder.is_conditioned()

        else:
            # Case: do not use caching (i.e. this is a standard forward pass);
            self._encode_multi_vision_post_fusion(vision_x, vision_gripper)

        bs, window_size = instr_and_action_ids.shape[:2]
        instr_and_action_ids = instr_and_action_ids.flatten(0, 1)
        instr_and_action_labels = instr_and_action_labels.flatten(0, 1)
        instr_and_action_mask = instr_and_action_mask.flatten(0, 1)
        media_ids = torch.full((bs*window_size, 1), self.media_token_id).to(instr_and_action_ids.device)
        media_mask = torch.ones_like(media_ids)
        media_labels = torch.full_like(media_ids, -100)

        instr_and_action_ids, _ = self.cat_multi_input_ids(instr_and_action_ids, media_ids)
        instr_and_action_labels, _ = self.cat_multi_input_ids(instr_and_action_labels, media_labels)
        instr_and_action_mask, _ = self.cat_multi_input_ids(instr_and_action_mask, media_mask)

        instr_and_action_ids, instr_and_action_labels, instr_and_action_mask = (
            rearrange(tensor, "(bs ws) seq_len ... -> bs (ws seq_len) ...", bs=bs, ws=window_size)
            for tensor in (instr_and_action_ids, instr_and_action_labels, instr_and_action_mask)
        )

        if mode != "train":
            action_dim = self.act_head_configs['action_dim']
            action_ids = self.lang_encoder.generate(input_ids=instr_and_action_ids, max_new_tokens=action_dim)
            action_ids = action_ids[0, -action_dim:].cpu().numpy()
            discretized_actions = self.action_tokenizer.decode_token_ids_to_actions(action_ids)
            action = np.array(discretized_actions)
            action[-1] = 1 if action[-1] > 0 else -1
            return action

        output = self.lang_encoder(
            input_ids=instr_and_action_ids,
            attention_mask=instr_and_action_mask.bool(),
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=True
        )

        if self.train_setup_configs['predict_action'] and instr_and_action_labels is not None:
            output_hs = output.logits
            action, action_loss = self._forward_action_head(output_hs, instr_and_action_labels)
            self._update_loss(loss, action_loss, 'act')

        if self.train_setup_configs['predict_caption'] and caption_labels is not None:
            logits = output.logits.clone()
            text_selector = self._caption_mask()
            logits = get_target_modal_tokens(logits, text_selector)
            if caption_mask is None:
                caption_mask = attention_mask
            _, caption_loss = self._forward_caption(
                logits,
                caption_labels,
                caption_mask,
            )
            self._update_loss(loss, caption_loss, 'cap')

        loss = self._format_loss(loss)

        # self.lang_encoder.clear_conditioned_layers()
        # self.lang_encoder._use_cached_vision_x = False

        return loss
