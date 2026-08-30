"""
GPT model (1B-capable rewrite).

Key differences from the legacy architecture (see ``model/gpt_legacy.py``):

- ``GPTConfig`` dataclass carries the full architecture so a checkpoint knows
  its own shape (no more module-level config globals baked into the weights).
- ``CausalSelfAttention`` fuses QKV into one ``nn.Linear`` and uses PyTorch's
  ``F.scaled_dot_product_attention`` (memory-efficient / flash backend, works on
  Turing SM 75). No per-head Python loop, no materialized BxTxT mask buffer.
- GELU MLP, tied input/output embeddings, GPT-2 style weight init with residual
  projections scaled by ``1/sqrt(2*n_layer)``.
- ``generate()`` uses a KV cache so each step only attends over the new token.

Backward-compatible construction: ``GPT(vocab_size)`` still works and builds a
``GPTConfig`` from the module-level ``config`` globals, so existing callers keep
functioning while they migrate to ``GPT(GPTConfig(...))``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    n_embd: int = 2048
    n_head: int = 16
    n_layer: int = 20
    block_size: int = 1024
    dropout: float = 0.0
    bias: bool = False
    tie_embeddings: bool = True

    @classmethod
    def from_globals(cls, vocab_size: int) -> "GPTConfig":
        """Build a config from the module-level ``config`` globals (legacy path)."""
        import config as _cfg

        return cls(
            vocab_size=vocab_size,
            n_embd=_cfg.n_embd,
            n_head=_cfg.n_head,
            n_layer=_cfg.n_layer,
            block_size=_cfg.block_size,
            dropout=_cfg.dropout,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GPTConfig":
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in fields})


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, kv_cache=None, use_cache=False):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        new_cache = (k, v) if use_cache else None

        # With a KV cache the incoming query length is 1 (or equal to full when
        # priming). Only apply the causal mask when there is no cache (full pass);
        # during incremental decode the query already only sees prior keys.
        is_causal = kv_cache is None and T > 1
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, new_cache


class FeedForward(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.sa = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.ffwd = FeedForward(cfg)

    def forward(self, x, kv_cache=None, use_cache=False):
        attn_out, new_cache = self.sa(self.ln1(x), kv_cache=kv_cache, use_cache=use_cache)
        x = x + attn_out
        x = x + self.ffwd(self.ln2(x))
        return x, new_cache


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Backward compatible: allow GPT(vocab_size:int).
        if isinstance(config, int):
            config = GPTConfig.from_globals(config)
        self.config = config
        cfg = config

        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_embedding_table.weight

        self.apply(self._init_weights)
        # Scale residual projections per GPT-2 init.
        scale = 0.02 / math.sqrt(2 * cfg.n_layer)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=scale)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.position_embedding_table.weight.numel()
            if not self.config.tie_embeddings:
                n -= self.lm_head.weight.numel()
            n -= self.token_embedding_table.weight.numel()
        return n

    def forward(self, idx, targets=None):
        _, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.token_embedding_table(idx) + self.position_embedding_table(pos)
        x = self.drop(x)
        for block in self.blocks:
            x, _ = block(x)
        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
            return logits, loss

        # Inference-time: only need logits for the last position when no targets.
        logits = self.lm_head(x)
        return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive generation with a KV cache."""
        self.eval()
        block_size = self.config.block_size
        B = idx.shape[0]

        # Prime the cache with the full prompt (cropped to block_size).
        idx_cond = idx[:, -block_size:]
        caches = [None] * len(self.blocks)
        logits = self._forward_with_cache(idx_cond, caches, cache_len=0)
        cache_len = idx_cond.shape[1]

        for _ in range(max_new_tokens):
            logits_last = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits_last, min(top_k, logits_last.size(-1)))
                logits_last[logits_last < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits_last, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            # If the cache would exceed block_size, recompute from the cropped
            # context (rare for short generations, keeps positions valid).
            if cache_len + 1 > block_size:
                idx_cond = idx[:, -block_size:]
                caches = [None] * len(self.blocks)
                logits = self._forward_with_cache(idx_cond, caches, cache_len=0)
                cache_len = idx_cond.shape[1]
            else:
                logits = self._forward_with_cache(idx_next, caches, cache_len=cache_len)
                cache_len += 1
        return idx

    def _forward_with_cache(self, idx, caches, cache_len):
        T = idx.shape[1]
        pos = torch.arange(cache_len, cache_len + T, device=idx.device)
        x = self.token_embedding_table(idx) + self.position_embedding_table(pos)
        for i, block in enumerate(self.blocks):
            x, caches[i] = block(x, kv_cache=caches[i], use_cache=True)
        x = self.ln_f(x)
        return self.lm_head(x)
