import argparse
import torch
from torch.autograd import Variable
from torch import nn
import numpy as np
from einops import rearrange
import copy

from .base_policy import BasePolicyHead, MLPTanhHead, MLPSigmoidHead
from .utils.transformers import Block


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class MultiBlock(nn.Module):
    def __init__(self, layer, num_layer, norm, pos_embed):
        super().__init__()
        assert isinstance(layer, Block)
        self.layers = _get_clones(layer, num_layer)
        self.norm = norm
        self.pos_embed = pos_embed

    def forward(self, x):
        # x: (bs, seq_len, hidden_size)
        x += self.pos_embed[:x.shape[1]][None, ...].to(x.device)
        for blk in self.layers:
            x = blk(x)
        x = self.norm(x)
        return x


class VAEPolicy(BasePolicyHead):
    # modified from: https://github.com/tonyzhaozh/act/blob/main/policy.py
    def __init__(self, in_features, action_dim, down_sample, latent, fwd_pred_next_n, window_size,
                 vae_latent_size=256, d_enc=3, d_dec=3, **kwargs):
        kwargs.pop('hidden_size', None)
        super().__init__(hidden_size=in_features, action_dim=action_dim, **kwargs)
        self.kl_weight = kwargs.get('kl_weight', 1.0)
        self.window_size = window_size
        print(f'KL Weight {self.kl_weight}')

        self.fwd_pred_next_n = fwd_pred_next_n
        self.down_sample = down_sample
        self.latent_dim = vae_latent_size # final size of latent z # TODO tune

        # encoder: (ENC_CLS, ACT) -> ENC_CLS -> latents
        self.cls_embed = nn.Embedding(1, self.hidden_size) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(action_dim, self.hidden_size) # project action to embedding
        self.register_buffer('enc_pos_embeddings', get_sinusoid_encoding_table(1+self.fwd_pred_next_n, self.hidden_size))
        self.encoder = MultiBlock(
            Block(self.hidden_size, 16, 4, qkv_bias=True, norm_layer=nn.LayerNorm),
            d_enc, nn.LayerNorm(self.hidden_size),
            self.enc_pos_embeddings
        )
        self.latent_proj = nn.Linear(self.hidden_size, self.latent_dim*2) # project hidden state to latent std, var

        # decoder: latents -> ENC_CLS + conditions: (OBS, QUERY) -> QUERY
        self.query_embed = nn.Embedding(fwd_pred_next_n, self.hidden_size)
        self.latent_out_proj = nn.Linear(self.latent_dim, self.hidden_size) # project latent sample to embedding
        self.global_1d_pool = nn.AdaptiveAvgPool1d(latent) # global pooling for down-sampling, typically latent is 1.
        self.register_buffer('dec_pos_embeddings', get_sinusoid_encoding_table(1024, self.hidden_size))
        self.decoder = MultiBlock(
            Block(self.hidden_size, 16, 4, qkv_bias=True, norm_layer=nn.LayerNorm),
            d_dec, nn.LayerNorm(self.hidden_size),
            self.dec_pos_embeddings
        )

        self.actions = MLPTanhHead(self.hidden_size, self.action_dim-1)
        self.gripper = MLPSigmoidHead(self.hidden_size, 1)

        self.cached_variables = {}

    def _preprocess_actions(self, actions):
        """
        Input:
            actions: either a tuple (actions, grippers) or just actions directly concatenated with grippers.

        Output:
            actions: b x l x chunk_size x act_dim
        """
        if isinstance(actions, tuple):
            actions, grippers = actions
            actions = torch.cat([actions, grippers.unsqueeze(-1).type_as(actions)], dim=-1)
        return actions

    def forward(self, tok_seq, actions=None, **kwargs):
        """
        tok_seq: b x l x n x d
        actions: only for training the auto-encoder, b x l x chunk_size x act_dim
        """
        assert tok_seq.dim() == 4
        # HACK: for now, actions may not be None during testing due to legacy code. Fix it when possible to keep codes clean.
        if not self.training:
            actions = None

        bs, seq_len, _, _  = tok_seq.shape

        if self.training:
            actions = self._preprocess_actions(actions)
            bs_a, seq_len_a, chunk_size, _ = actions.shape

            assert chunk_size == self.fwd_pred_next_n
            assert bs_a == bs and seq_len_a == seq_len

            # project action sequence to embedding dim, and concat with a CLS token
            actions = rearrange(actions, 'b l n d -> (b l) n d', b=bs, l=seq_len)
            action_embed = self.encoder_action_proj(actions) # (bs*seq, chunk_size, hidden_size)

            cls_embed = self.cls_embed.weight # (1, hidden_size)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs * seq_len, 1, 1) # (bs*seq, 1, hidden_size)

            encoder_input = torch.cat([cls_embed, action_embed], axis=1) # (bs*seq, chunk_size+1, hidden_size)

            # query model, output: (bs*seq, chunk_size + 1, hidden_size)
            encoder_output = self.encoder(encoder_input)
            encoder_output = encoder_output[:, 0] # take cls output only, (bs*seq, hidden_size)

            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample) # (bs*seq, hidden_size)

        else:
            # here for testing, we only use zero-vector as the latent variale.
            # if one wants multi-modal, this vector should be sampled from normalized gaussian distribution.
            mu = logvar = None
            latent_sample = torch.zeros([bs * seq_len, self.latent_dim], dtype=tok_seq.dtype).to(tok_seq.device)
            latent_input = self.latent_out_proj(latent_sample) # (bs*seq, hidden_size)

        if self.down_sample == 'pooling':
            tok_seq = rearrange(tok_seq, 'b l n d-> (b l) n d')
            tok_seq = self.global_1d_pool(tok_seq.permute(0, 2, 1))
            tok_seq = rearrange(tok_seq, 'b d n-> b n d')
            # tok_seq = rearrange(tok_seq, '(b l) d n -> b l n d', b=bs, l=seq_len)
        elif self.down_sample == 'none':
            tok_seq = rearrange(tok_seq, 'b l n d-> (b l) n d')
        else:
            raise NotImplementedError

        assert tok_seq.dim() == 3 and tok_seq.shape[0] == bs * seq_len

        latent_input = latent_input[:, None] # (bs*seq, 1, hidden_size)
        # (bs*seq, 1+latent+chunk_size, hidden_size)
        dec_input = torch.cat([latent_input, tok_seq, self.query_embed.weight[None, ...].repeat(bs * seq_len, 1, 1)], dim=1)
        # take the embeddings of queries only
        dec_output = self.decoder(dec_input)[:, -self.fwd_pred_next_n:, :]

        actions = self.actions(dec_output)
        gripper = self.gripper(dec_output)
        
        actions = rearrange(actions, '(b l) n d -> b l n d', b=bs, l=seq_len, n=self.fwd_pred_next_n)
        gripper = rearrange(gripper, '(b l) n d -> b l n d', b=bs, l=seq_len, n=self.fwd_pred_next_n).squeeze(-1)

        self.cached_variables
        self.cached_variables['mu'] = mu
        self.cached_variables['logvar'] = logvar

        return actions, gripper


    def loss(self, pred_action, labels, attention_mask=None):
        """
        pred_action: [bs, seq_len, chunck_size, 7], 1-6 refers to ee pose, 7 refers to gripper open/close
        lables: (pose gt [bs, seq_len, chunck_size, 6], gripper gt [bs, seq_len, chunck_size])
        attention_mask: [bs, seq_len, chunck_size]
        """
        assert 'mu' in self.cached_variables and 'logvar' in self.cached_variables
        mu, logvar = self.cached_variables.pop('mu'), self.cached_variables.pop('logvar')
        
        if mu is None or logvar is None:
            # for monitoring validation loss
            return super().loss(pred_action, labels, attention_mask)

        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
        # loss_kl = total_kld[0]
        loss_kl = mean_kld[0]

        loss_dict = super().loss(pred_action, labels, attention_mask)
        loss_dict.update({'loss_kl': loss_kl}) # weight=0.1
        return loss_dict
