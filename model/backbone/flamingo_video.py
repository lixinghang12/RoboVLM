import torch
from einops import rearrange, repeat
from torch import nn
import copy
from typing import Tuple
import numpy as np
from collections import namedtuple
from einops import rearrange, repeat

from utils.model_utils import build_tokenizer, get_target_modal_tokens
from train.loss import calculate_vl_cross_entropy

from model.backbone.flamingo import RoboFlamingo

from typing import Optional

class RoboFlamingoVideo(RoboFlamingo):

    def __init__(self, **kwargs):
        super(RoboFlamingoVideo, self).__init__(**kwargs)
        self.frame_embs = nn.Parameter(torch.randn(self.window_size, self.vis_dim))
    
    def _get_obs_embed(self, rgb):

        batch_size, seq_length, c, h, w = rgb.shape
        rgb = rgb.reshape(batch_size * seq_length, c, h, w)
        # print('rgb input shape', rgb.shape)
        patch_embeddings = self.vision_encoder.visual(rgb)[1].unsqueeze(1).unsqueeze(1) # b*l, 1, 1, v, d
        # print('path_embedding shape after vit', patch_embeddings.shape)
        # patch_embeddings = patch_embeddings.view(batch_size, seq_length, *patch_embeddings.shape[1:])
        
        # patch_embeddings = patch_embeddings.unsqueeze(1).unsqueeze(1) # b*l, 1, 1, v, d
        patch_embeddings = self.perceiver(patch_embeddings) # b*l, 1, n, d

        return patch_embeddings
    
    def _add_frame_embedding(self, x):
        
        b, F, T, v = x.shape[:4]
        frame_embs = repeat(self.frame_embs[:F], "F d -> b F T v d", b=b, T=T, v=v)
        x = x + frame_embs
        return x
    
    def _encode_video_frame_post_only(self, vision_x):
        B, L = vision_x.shape[:2]
        vision_x = self._get_obs_embed(vision_x) # B, L, c, h, w -> B*L, 1, n, d
        vision_x = vision_x.view(B, L, *vision_x.shape[1:]) # B, L, 1, n, d
        vision_x = self._add_frame_embedding(vision_x)
        
        b, F, T = vision_x.shape[:3]
        vision_x = rearrange(vision_x, 'b F T n d -> b T (F n) d', b=b, F=F) # B, 1, L*n, d
        
        return vision_x

    def _encode_multi_vision_post_fusion(self, vision_rgb: torch.Tensor, vision_gripper: Optional[torch.Tensor]=None):
        vision_rgb = self._encode_video_frame_post_only(vision_rgb)

        if vision_gripper is not None:
            vision_gripper = self._encode_video_frame_post_only(vision_gripper)
            vision_rgb = torch.cat([vision_rgb, vision_gripper], dim=2)

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
        action = self.lm_head(action_tokens)

        action_loss = None
        action_mask = None
        if action_labels is not None:
            action_loss = self.lm_head.loss(action, action_labels, action_mask)

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

        with_history = self.act_head_configs.get("with_history", False)
        if not with_history:
            vision_x = vision_x[:, -1:]
            if vision_gripper is not None:
                vision_gripper = vision_gripper[:, -1:]

        action_space = self.act_head_configs.get("action_space", "continuous")
        # import pdb; pdb.set_trace()
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
            # TODO
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
        # print(lang_x.shape, attention_mask.shape)
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


        output_hs = output.logits
        _, action_loss = self._forward_action_head(output_hs, instr_and_action_labels, instr_and_action_mask)
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
        mode="train",
        **kwargs
    ): 
        loss = {}
        action_space = self.act_head_configs.get("action_space", "continuous")
        bs, seq_len = vision_x.shape[:2]

        assert (
            vision_x is not None
        ) or use_cached_vision_x, (
            "Must provide either vision_x or use_cached_vision_x to True."
        )
        bs, seq_len = vision_x.shape[:2]
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
            action_ids = torch.full((bs, self.latent_num), self.action_token_id).to(lang_x.device)
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

        if action_space == "continuous":
            action_hs = output_hs[action_ids_mask].reshape(bs, self.latent_num, -1)
        elif action_space == "down_sample":
            action_hs = output_hs
        action_hs = action_hs.unsqueeze(1)
        
        if action_labels is not None and action_labels[0] is not None:
            action_labels = action_labels[0].unsqueeze(1), action_labels[1].unsqueeze(1)
        
        if self.train_setup_configs['predict_action'] and action_labels is not None:
            action_logits, action_loss = self._forward_action_head(action_hs, action_labels, action_mask)
            if mode == "train":
                self._update_loss(loss, action_loss, 'act')
            else:
                return action_logits

        loss = self._format_loss(loss)

        return loss
