"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from util.misc import top_p_sampling


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class AttentionBase(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # output projection
        if isinstance(self, CausalCrossAttention):
            self.c_proj_cross = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        else:
            self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                 .view(1, 1, config.block_size, config.block_size))

class SelfAttention(AttentionBase):

    def __init__(self, config):
        super().__init__(config)
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

    def forward(self, x, kv_cache=None, mask=None):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        if kv_cache is not None:
            key_cached, value_cached = kv_cache.unbind(dim=0)  # (2, B, T, head_size) -> 2 * (B, T, head_size)
            k = torch.cat((key_cached, k), dim=-2)  # (B, cache + T, head_size)
            v = torch.cat((value_cached, v), dim=-2)  # (B, cache + T, head_size)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0, is_causal=False)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, None if kv_cache is None else torch.stack((k, v))
    
class CausalSelfAttention(AttentionBase):

    def __init__(self, config):
        super().__init__(config)
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

    def forward(self, x, kv_cache=None, mask=None):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        if kv_cache is not None:
            key_cached, value_cached = kv_cache.unbind(dim=0)  # (2, B, T, head_size) -> 2 * (B, T, head_size)
            k = torch.cat((key_cached, k), dim=-2)  # (B, cache + T, head_size)
            v = torch.cat((value_cached, v), dim=-2)  # (B, cache + T, head_size)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0, is_causal=mask is None)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, None if kv_cache is None else torch.stack((k, v))


class CrossAttention(AttentionBase):

    def __init__(self, config):
        super().__init__(config)
        self.ckv_attn = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)
        self.cq_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x, encoding):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)4
        B_, T_, C_ = encoding.size()  # batch size, sequence length, embedding dimensionality (n_embd)4

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k, v = self.ckv_attn(encoding).split(self.n_embd, dim=2)
        q = self.cq_attn(x)

        k = k.view(B_, T_, self.n_head, C_ // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B_, T_, self.n_head, C_ // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=False)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, kv_cache=None, mask=None):
        out, kv_cache = self.attn(self.ln_1(x), kv_cache, mask)
        x = x + out
        x = x + self.mlp(self.ln_2(x))
        return x, kv_cache


class BlockWithCrossAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_3 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.xattn = CrossAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, encoding, kv_cache=None, mask=None):
        out, kv_cache = self.attn(self.ln_1(x), kv_cache, mask)
        x = x + out
        xattn = self.xattn(self.ln_2(x), encoding)
        x = x + xattn
        x = x + self.mlp(self.ln_3(x))
        return x



def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class BlockWithAdaptive(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.n_embd, 6 * config.n_embd, bias=True)
        )
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)


    def forward(self, x, c=None, kv_cache=None, mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)

        out, kv_cache = self.attn(modulate(self.ln_1(x), shift_msa, scale_msa), kv_cache, mask)
        # out, kv_cache = self.attn(self.ln_1(x), kv_cache, mask)
        x = x + gate_msa * out
        x = x + gate_mlp * self.mlp(modulate(self.ln_2(x), shift_mlp, scale_mlp))
        return x, kv_cache



class CausalCrossAttention(AttentionBase):

    def __init__(self, config):
        super().__init__(config)
        # key, query, value projections for all heads, but in a batch
        self.ckv_attn = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)
        self.cq_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x, encoding, kv_cache=None, mask=None):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        q_T = encoding.shape[1]
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k, v = self.ckv_attn(x).split(self.n_embd, dim=2)
        q = self.cq_attn(encoding)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, q_T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        # q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        if kv_cache is not None:
            key_cached, value_cached = kv_cache.unbind(dim=0)  # (2, B, T, head_size) -> 2 * (B, T, head_size)
            k = torch.cat((key_cached, k), dim=-2)  # (B, cache + T, head_size)
            v = torch.cat((value_cached, v), dim=-2)  # (B, cache + T, head_size)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0, is_causal=mask is None)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if mask is not None:
                att = att.masked_fill(self.bias[:, :, :q_T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, q_T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj_cross(y))
        return y, None if kv_cache is None else torch.stack((k, v))#, att

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.padding_idx = config.vocab_size - 1
        self.vocab_size = config.vocab_size
        print('Model Padding Index:', self.padding_idx)
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd, padding_idx=self.padding_idx),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            wce=nn.Embedding(3, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, coord, targets=None, kv_cache=None, mask_cache=None):
        use_kv_cache = kv_cache is not None
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        if kv_cache is not None and kv_cache[0].numel():
            pos = kv_cache[0].shape[-2]  # kv_cache of shape: num_layers * (2, B, nh, T, hs)
            pos_emb = self.transformer.wpe.weight[None, pos // 3]  # 1 x n_embd
            mask = mask_cache.index_select(2, torch.LongTensor([pos]).to(pos_emb.device))[:, :, :, :pos + 1]
        else:
            pos = torch.tensor([i // 3 for i in range(t)], dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
            mask = None

        # print('embs:', idx.min(), idx.max(), pos.min(), pos.max(), coord.min(), coord.max())
        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        coord_emb = self.transformer.wce(coord)  # position embeddings of shape (t, n_embd)
        # print('shapes:', tok_emb.shape, pos_emb.shape, coord_emb.shape)
        x = self.transformer.drop(tok_emb + pos_emb + coord_emb)

        # apply multiple transformer blocks
        new_kv_cache = []
        kv_cache = kv_cache or [None] * self.config.n_layer

        for block, kv_cache_layer in zip(self.transformer.h, kv_cache):
            x, new_kv = block(x, kv_cache_layer, mask)
            new_kv_cache.append(new_kv)

        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=self.padding_idx)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        if not use_kv_cache:
            return logits, loss
        else:
            return logits, new_kv_cache

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        config_args['bias'] = True  # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        return configure_optimizers(self.named_parameters(), weight_decay, learning_rate, betas, device_type)

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, coord, max_new_tokens, temperature=1.0, top_k=None, top_p=0.9, use_kv_cache=False):

        if use_kv_cache and (max_new_tokens + idx.shape[-1] - 1) > self.config.block_size:
            # print(f"Cannot generate more than {self.config.block_size} tokens with kv cache, setting max new tokens to {self.config.block_size - idx.shape[-1]}")
            max_new_tokens = self.config.block_size - idx.shape[-1]

        kv_cache = (
            [torch.empty(2, 0, device=idx.device, dtype=idx.dtype) for _ in range(self.config.n_layer)]
            if use_kv_cache
            else None
        )
        mask_cache = None
        if use_kv_cache:
            ones = torch.ones((self.config.block_size, self.config.block_size), device=idx.device, dtype=torch.bool)
            mask_cache = torch.tril(ones).unsqueeze(0).unsqueeze(0)

        current_coord = coord
        one_t = torch.LongTensor([1]).to(coord.device)
        for iteration in range(max_new_tokens):

            if not use_kv_cache or (iteration == 0 and idx.shape[-1] > 1):
                # if the sequence context is growing too long we must crop it at block_size
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
                coord_cond = current_coord if current_coord.size(0) <= self.config.block_size else current_coord[-self.config.block_size:]
            else:
                idx_cond = idx[:, -1:]
                coord_cond = current_coord[-1:]
            # forward the model to get the logits for the index in the sequence
            logits, kv_cache = self(idx_cond, coord_cond, kv_cache=kv_cache if use_kv_cache else None, mask_cache=mask_cache)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities

            use_hard_constraints = False
            if use_hard_constraints:
                # Use hard constraints
                # stop can only occur when 3 coords are generated
                if idx.shape[1] % 3 != 0:
                    logits[:, self.vocab_size - 2] = -float('Inf')

                # all z less than last z are not allowed
                if idx.shape[1] % 3 == 0:
                    if idx.shape[1] > 2:
                        last_z = idx[0, -3]
                        if last_z > 0:
                            logits[:, :last_z - 1] = -float('Inf')
                if idx.shape[1] % 3 == 1:
                    if idx.shape[1] > 3:
                        last_z = idx[0, -1]
                        last_to_last_z = idx[0, -4]
                        last_y = idx[0, -3]
                        if last_z == last_to_last_z:
                            if last_y > 0:
                                logits[:, :last_y - 1] = -float('Inf')
                if idx.shape[1] % 3 == 2:
                    if idx.shape[1] > 4:
                        last_z = idx[0, -2]
                        last_to_last_z = idx[0, -5]
                        last_y = idx[0, -1]
                        last_to_last_y = idx[0, -4]
                        last_x = idx[0, -3]
                        if last_z == last_to_last_z and last_to_last_y == last_y:
                            if last_x > 0:
                                logits[:, :last_x - 1] = -float('Inf')

            # sample from the distribution
            if top_p is not None:
                idx_next = top_p_sampling(logits, top_p)
            else:
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            current_coord = torch.cat((current_coord, one_t * (current_coord[-1] + 1) % 3), dim=0)
            if idx[0, -1] == (self.vocab_size - 2):
                break
        return idx


def configure_optimizers(named_parameters, weight_decay, learning_rate, betas, device_type, additional_params=None):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in named_parameters}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    if additional_params is not None:
        if isinstance(additional_params, dict):
            for n, p in additional_params:
                nodecay_params.append(p)
        else:
            for additional_param in additional_params:
                for n, p in additional_param:
                    nodecay_params.append(p)
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    print(f"using fused AdamW: {use_fused}")

    return optimizer
