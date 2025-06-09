"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from dataset.quantized_soup import get_shifted_sequence
from model.face_model import TransformerBase
from model.nanogpt import Block, LayerNorm
from util.misc import top_p_sampling
from tqdm import tqdm


class QuantSoupTransformer(TransformerBase):

    def __init__(self, config, vq_config):
        super().__init__()
        assert config.block_size is not None
        self.config = config
        self.padding_idx = 2
        self.tokens_per_face = config.finemb_size
        self.finemb_size = 3 + config.finemb_size  # 3 for start, stop pad, 3 for fin
        self.foutemb_size = 3 + config.foutemb_size
        vocab_size = vq_config.n_embed + 1 + 1 + 1  # +1 for start, +1 for stop, +1 for pad
        self.vocab_size = vocab_size
        print('Model Vocab Size:', vocab_size)
        print('Model Padding Index:', self.padding_idx)
        print('Model Fin Size:', self.finemb_size)
        print('Model Fout Size:', self.foutemb_size)
        self.input_layer = nn.Linear(vq_config.embed_dim, config.n_embd)
        self.extra_embeds = nn.Embedding(3, config.n_embd, padding_idx=self.padding_idx)
        self.transformer = nn.ModuleDict(dict(
            wpe=nn.Embedding(config.block_size, config.n_embd),
            wfie=nn.Embedding(self.finemb_size, config.n_embd, padding_idx=self.padding_idx),
            wfoe=nn.Embedding(self.foutemb_size, config.n_embd, padding_idx=self.padding_idx),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def forward(self, idx, fin, fout, tokenizer, targets=None, kv_cache=None, mask_cache=None):
        use_kv_cache = kv_cache is not None
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        embed = torch.zeros((b * t, self.config.n_embd), dtype=torch.float32, device=device)
        idx_in_extra = torch.isin(idx, torch.LongTensor([0, 1, 2]).to(device)).reshape(-1)
        idx_flat = idx.reshape(-1)
        embed[idx_in_extra, :] = self.extra_embeds(idx_flat[idx_in_extra])
        embed[~idx_in_extra, :] = self.input_layer(tokenizer.embed(idx_flat[~idx_in_extra] - 3))
        tok_emb = embed.reshape(b, t, -1)  # token embeddings of shape (b, t, n_embd)
        finemb = self.transformer.wfie(fin)  # face inner embeddings of shape (t, n_embd)
        foutemb = self.transformer.wfoe(fout)  # face outer embeddings of shape (t, n_embd)

        # position embedding

        if kv_cache is not None and kv_cache[0].numel():
            pos = kv_cache[0].shape[-2]  # kv_cache of shape: num_layers * (2, B, nh, T, hs)
            pos_emb = self.transformer.wpe.weight[None, pos]  # 1 x n_embd
            mask = mask_cache.index_select(2, torch.LongTensor([pos]).to(pos_emb.device))[:, :, :, :pos + 1]
        else:
            pos = torch.tensor([i for i in range(t)], dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
            mask = None

        sum_emb = tok_emb + pos_emb + finemb + foutemb
        x = self.transformer.drop(sum_emb)

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

    @torch.no_grad()
    def generate(self, idx, fin, fout, tokenizer, max_new_tokens=10000, temperature=1.0, top_k=None, top_p=0.9, use_kv_cache=False):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        if use_kv_cache and (max_new_tokens + idx.shape[-1] - 1) > self.config.block_size:
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

        current_fin = fin
        current_fout = fout
        one_t = torch.LongTensor([1]).to(fin.device)
        for iteration in range(max_new_tokens):

            if not use_kv_cache or (iteration == 0 and idx.shape[-1] > 1):
                # if the sequence context is growing too long we must crop it at block_size
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
                fin_cond = current_fin if current_fin.size(1) <= self.config.block_size else current_fin[:, -self.config.block_size:]
                fout_cond = current_fout if current_fout.size(1) <= self.config.block_size else current_fout[:, -self.config.block_size:]
                fout_cond = torch.from_numpy(get_shifted_sequence(fout_cond[0].cpu().numpy())).to(idx_cond.device).unsqueeze(0)
            else:
                idx_cond = idx[:, -1:]
                fin_cond = current_fin[:, -1:]
                fout_cond = current_fout[:, -1:]  # note: don't need shifting since we assume block_size is huge enough to not need shifting
            # forward the model to get the logits for the index in the sequence
            logits, kv_cache = self(idx_cond, fin_cond, fout_cond, tokenizer, kv_cache=kv_cache if use_kv_cache else None, mask_cache=mask_cache)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # TODO: Introduce hard constraints

            # sample from the distribution
            # apply softmax to convert logits to (normalized) probabilities
            if top_p is not None:
                idx_next = top_p_sampling(logits, top_p)
            else:
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            last_fin_cond = current_fin[0, -1]
            if (iteration == 0 and idx.shape[-1] == 2):
                current_fin = torch.cat((current_fin, ( (3) * one_t[0]).unsqueeze(0).unsqueeze(0)), dim=1)
                current_fout = torch.cat((current_fout, ((3 ) * one_t[0]).unsqueeze(0).unsqueeze(0)), dim=1)
            elif last_fin_cond == self.finemb_size - 1:
                current_fin = torch.cat((current_fin, ( (3 ) * one_t[0]).unsqueeze(0).unsqueeze(0)), dim=1)
                current_fout = torch.cat((current_fout, (current_fout[0, -1] + 1).unsqueeze(0).unsqueeze(0)), dim=1)
            else:
                current_fin = torch.cat((current_fin, (current_fin[0, -1] + 1).unsqueeze(0).unsqueeze(0)), dim=1)
                current_fout = torch.cat((current_fout, (current_fout[0, -1]).unsqueeze(0).unsqueeze(0)), dim=1)
            if idx_next == 1:
                return idx
        return None


    @torch.no_grad()
    def generate_with_beamsearch(self, idx, fin, fout, tokenizer, max_new_tokens=10000, use_kv_cache=False, beam_width=6):

        backup_beams = []
        backup_beam_prob = []
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

        current_fin = fin
        current_fout = fout
        one_t = torch.LongTensor([1]).to(fin.device)

        idx = idx.repeat((beam_width, 1, 1)).transpose(0, 1).flatten(end_dim=-2)
        current_fin = current_fin.repeat((beam_width, 1, 1)).transpose(0, 1).flatten(end_dim=-2)
        current_fout = current_fout.repeat((beam_width, 1, 1)).transpose(0, 1).flatten(end_dim=-2)

        logits, kv_cache = self(idx, fin, fout, tokenizer, kv_cache=kv_cache if use_kv_cache else None, mask_cache=mask_cache)

        vocabulary_size = logits.shape[-1]
        probabilities, top_k_indices = logits[0, 0, :].squeeze().log_softmax(-1).topk(k=beam_width, axis=-1)

        next_chars = top_k_indices.reshape(-1, 1)
        idx = torch.cat((idx, next_chars), axis=-1)

        last_fin_cond = current_fin[0, -1]  # same for all beams
        if last_fin_cond == self.finemb_size - 1 or (idx.shape[-1] == 2):
            current_fin = torch.cat((current_fin, (3 * one_t[0]).unsqueeze(0).unsqueeze(0).expand((beam_width, -1))), dim=1)
            current_fout = torch.cat((current_fout, (current_fout[0, -1] + 1).unsqueeze(0).unsqueeze(0).expand((beam_width, -1))), dim=1)
        else:
            current_fin = torch.cat((current_fin, (current_fin[0, -1] + 1).unsqueeze(0).unsqueeze(0).expand((beam_width, -1))), dim=1)
            current_fout = torch.cat((current_fout, current_fout[0, -1].unsqueeze(0).unsqueeze(0).expand((beam_width, -1))), dim=1)
        
        for iteration in tqdm(range(max_new_tokens - 1), desc='beam_search'):
            if not use_kv_cache:
                idx_cond = idx
                fin_cond = current_fin
                fout_cond = current_fout
            else:
                idx_cond = idx[:, -1:]
                fin_cond = current_fin[:, -1:]
                fout_cond = current_fout[:, -1:]

            # forward the model to get the logits for the index in the sequence
            logits, kv_cache = self(idx_cond, fin_cond, fout_cond, tokenizer, kv_cache=kv_cache if use_kv_cache else None, mask_cache=mask_cache)

            next_probabilities = logits.log_softmax(-1)

            next_probabilities = next_probabilities.reshape((-1, beam_width, next_probabilities.shape[-1]))
            probabilities = probabilities.unsqueeze(-1) + next_probabilities
            probabilities = probabilities.flatten(start_dim=1)
            probabilities, top_k_indices = probabilities.topk(k=beam_width, axis=-1)
            next_indices = torch.remainder(top_k_indices, vocabulary_size).flatten().unsqueeze(-1)
            best_candidates = (top_k_indices / vocabulary_size).long()
            best_candidates += torch.arange(idx.shape[0] // beam_width, device=idx.device).unsqueeze(-1) * beam_width
            idx = idx[best_candidates].flatten(end_dim=-2)
            for block_idx in range(len(kv_cache)):
                kv_cache[block_idx] = kv_cache[block_idx][:, best_candidates.flatten(), :, :, :]
            idx = torch.cat((idx, next_indices), axis=1)

            # update fin and fout
            last_fin_cond = current_fin[0, -1]  # same for all beams
            if last_fin_cond == self.finemb_size - 1 or (idx.shape[-1] == 2):
                current_fin = torch.cat((current_fin, (3 * one_t[0]).unsqueeze(0).unsqueeze(0).expand((beam_width, -1))), dim=1)
                current_fout = torch.cat((current_fout, (current_fout[0, -1] + 1).unsqueeze(0).unsqueeze(0).expand((beam_width, -1))), dim=1)
            else:
                current_fin = torch.cat((current_fin, (current_fin[0, -1] + 1).unsqueeze(0).unsqueeze(0).expand((beam_width, -1))), dim=1)
                current_fout = torch.cat((current_fout, current_fout[0, -1].unsqueeze(0).unsqueeze(0).expand((beam_width, -1))), dim=1)
            
            amax = probabilities.flatten().argmax()
            if idx[amax, -1] == 1:
                return idx[amax: amax + 1, :]
            for beam_idx in range(beam_width):
                if idx[beam_idx, -1] == 1:
                    backup_beams.append(idx[beam_idx: beam_idx + 1, :])
                    backup_beam_prob.append(probabilities[0, beam_idx].item())
        return None