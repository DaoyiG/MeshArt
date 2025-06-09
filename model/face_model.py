import math

import torch
from torch import nn
import torch.nn.functional as F

from model.nanogpt import LayerNorm, Block, BlockWithCrossAttention, configure_optimizers
from util.misc import top_p_sampling


class TransformerBase(nn.Module):
    
    def __init__(self):
        super().__init__()

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        # n_params = sum(p.numel() for p in self.parameters())
        n_params = sum(p.numel() for n,p in self.named_parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        return configure_optimizers(self.named_parameters(), weight_decay, learning_rate, betas, device_type)


class VertexEncoder(TransformerBase):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte_extra=nn.Embedding(3, config.n_embd),
            wte_0=nn.Embedding(config.vocab_size, config.n_embd, padding_idx=config.padding_idx),
            wte_1=nn.Embedding(config.vocab_size, config.n_embd, padding_idx=config.padding_idx),
            wte_2=nn.Embedding(config.vocab_size, config.n_embd, padding_idx=config.padding_idx),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.n_embd)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def forward(self, idx):
        # forward the GPT model itself
        outputs = []
        extra_embeddings = self.transformer.wte_extra(torch.arange(3, dtype=torch.long, device=idx.device)).unsqueeze(0)
        for split_idx in torch.split(idx, self.config.split_batch_size, dim=0):
            tok_emb_0 = self.transformer.wte_0(split_idx[:, :, 0])  # token embeddings of shape (b, t, n_embd)
            tok_emb_1 = self.transformer.wte_1(split_idx[:, :, 1])  # token embeddings of shape (b, t, n_embd)
            tok_emb_2 = self.transformer.wte_2(split_idx[:, :, 2])  # token embeddings of shape (b, t, n_embd)
            # no positional encoding in vertex encoder polygen
            # pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
            # x = self.transformer.drop(tok_emb + pos_emb)
            tok_emb = tok_emb_0 + tok_emb_1 + tok_emb_2
            tok_emb = torch.cat([extra_embeddings.expand(tok_emb.shape[0], -1, -1), tok_emb], dim=1)
            x = self.transformer.drop(tok_emb)
            for block in self.transformer.h:
                x, _ = block(x)
            x = self.transformer.ln_f(x)
            logits = self.lm_head(x)
            outputs.append(logits)
        outputs = torch.cat(outputs, dim=0)
        return outputs


class FaceDecoder(TransformerBase):

    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.use_cross_attention:
            block_class = BlockWithCrossAttention
        else:
            block_class = Block
        self.fin_size = config.finemb_size
        self.fout_size = config.foutemb_size
        if self.config.enable_in_emb:
            self.transformer = nn.ModuleDict(dict(
                wfie=nn.Embedding(config.finemb_size, config.n_embd, padding_idx=2),
                wfoe=nn.Embedding(config.foutemb_size, config.n_embd, padding_idx=2),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([block_class(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            ))
        else:
            self.transformer = nn.ModuleDict(dict(
                wfoe=nn.Embedding(config.foutemb_size, config.n_embd, padding_idx=2),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([block_class(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            ))
        self.lm_head = nn.Linear(config.n_embd, config.n_embd)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def forward(self, face_sequence, facein_idx, faceout_idx, vertex_encoding):
        # forward the GPT model itself
        embed = torch.gather(vertex_encoding, 1, face_sequence.unsqueeze(dim=-1).expand(-1, -1, vertex_encoding.shape[-1]))
        fout_emb = self.transformer.wfoe(faceout_idx)  # position embeddings of shape (b, t, n_embd)
        if self.config.enable_in_emb:
            fin_emb = self.transformer.wfie(facein_idx)  # token embeddings of shape (b, t, n_embd)
            x = self.transformer.drop(embed + fin_emb + fout_emb)
        else:
            x = self.transformer.drop(embed + fout_emb)
        for block in self.transformer.h:
            x, _ = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits


class FaceModel(torch.nn.Module):
    
    def __init__(self, vt_config, ft_config):
        super().__init__()
        self.vertex_encoder = VertexEncoder(vt_config)
        self.face_decoder = FaceDecoder(ft_config)
        self.vertex_pad = vt_config.padding_idx
        self.block_size = ft_config.block_size
        self.emb_dim = ft_config.n_embd
        self.finemb_size = ft_config.finemb_size
        self.foutemb_size = ft_config.foutemb_size

    def get_vertex_mask(self, vertex_idx):
        vertex_mask = vertex_idx[:, :, 0] != self.vertex_pad
        extra = torch.ones(vertex_mask.shape[0], 3, dtype=torch.bool, device=vertex_mask.device)
        return torch.cat((extra, vertex_mask), dim=1)

    def forward(self, vertex_idx, in_face_sequence, facein_idx, faceout_idx, out_face_sequence=None):
        vertex_mask = self.get_vertex_mask(vertex_idx)
        vertex_encoding = self.vertex_encoder(vertex_idx)
        return self.decoder(vertex_encoding, in_face_sequence, facein_idx, faceout_idx, vertex_mask, out_face_sequence)

    def decoder(self, vertex_encoding, in_face_sequence, facein_idx, faceout_idx, vertex_mask, out_face_sequence=None):
        logits = self.face_decoder(in_face_sequence, facein_idx, faceout_idx, vertex_encoding)
        scores = torch.bmm(logits, vertex_encoding.permute(0, 2, 1))
        scores = scores / math.sqrt(self.emb_dim)
        scores = scores.permute((1, 0, 2))
        scores[:, ~vertex_mask] = -1e9
        scores = scores.permute((1, 0, 2))
        loss = None
        if out_face_sequence is not None:
            loss = F.cross_entropy(scores.view(-1, scores.size(-1)), out_face_sequence.view(-1), ignore_index=2)
        return scores, loss

    @torch.no_grad()
    def generate(self, vertex_idx, in_face_sequence, facein_idx, faceout_idx, max_new_tokens, temperature=1.0, top_k=None, top_p=0.9):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        assert vertex_idx.shape[0] == 1, "only generation of a single sample at a time is supported"
        current_facein = facein_idx
        current_faceout = faceout_idx
        one_t = torch.LongTensor([1]).to(vertex_idx.device)
        vertex_encoding = self.vertex_encoder(vertex_idx)
        vertex_mask = self.get_vertex_mask(vertex_idx)
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            in_face_sequence_cond = in_face_sequence if in_face_sequence.size(1) <= self.block_size else in_face_sequence[:, -self.block_size:]
            facein_cond = current_facein if current_facein.size(1) <= self.block_size else current_facein[:, -self.block_size:]
            faceout_cond = current_faceout if current_faceout.size(1) <= self.block_size else current_faceout[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self.decoder(vertex_encoding, in_face_sequence_cond, facein_cond, faceout_cond, vertex_mask)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # TODO: Introduce hard constraints

            # sample from the distribution
            if top_p is not None:
                idx_next = top_p_sampling(logits, top_p)
            else:
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            # append sampled index to the running sequence and continue
            in_face_sequence = torch.cat((in_face_sequence, idx_next), dim=1)
            if idx_next == 0:
                current_facein = torch.cat((current_facein, idx_next), dim=1)
                current_faceout = torch.cat((current_faceout, idx_next), dim=1)
            elif idx_next != 1:
                if in_face_sequence[0, -2] == 0:
                    # the face before this face was a new face
                    current_facein = torch.cat((current_facein, (3 * one_t[0]).unsqueeze(0).unsqueeze(0)), dim=1)
                    if current_faceout[0].shape[0] == 1:
                        new_face = (3 * one_t[0])
                    else:
                        new_face = current_faceout[0, -2] + 1
                        new_face = new_face if new_face < self.foutemb_size else (2 * one_t[0])
                    current_faceout = torch.cat((current_faceout, new_face.unsqueeze(0).unsqueeze(0)), dim=1)
                else:
                    new_face = (current_facein[0, -1] + 1) if (current_facein[0, -1] + 1) < self.finemb_size else (2 * one_t[0])
                    current_facein = torch.cat((current_facein, new_face.unsqueeze(0).unsqueeze(0)), dim=1)
                    current_faceout = torch.cat((current_faceout, current_faceout[0, -1].unsqueeze(0).unsqueeze(0)), dim=1)
            elif idx_next == 1:
                break
        return in_face_sequence

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        return configure_optimizers(self.named_parameters(), weight_decay, learning_rate, betas, device_type)
