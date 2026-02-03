"""Decoder-only Transformer (GPT-style)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint as torch_checkpoint


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 1024
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1
    bias: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if past_kv is None:
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        else:
            total_len = k.size(2)
            mask = self.mask[:, :, total_len - T : total_len, :total_len]
            att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y))
        present = (k, v) if use_cache else None
        return y, present


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, present = self.attn(self.ln1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, present


class GPT(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.gradient_checkpointing = False

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = bool(enabled)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        ignore_index: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        B, T = idx.size()
        past_len = 0
        if past_kv is not None:
            past_len = past_kv[0][0].size(2)
        if past_len + T > self.config.block_size:
            raise ValueError(
                f"Sequence length {past_len + T} exceeds block_size {self.config.block_size}"
            )
        pos = torch.arange(past_len, past_len + T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        presents = [] if use_cache else None
        use_checkpoint = (
            self.gradient_checkpointing
            and self.training
            and not use_cache
            and past_kv is None
        )
        for i, block in enumerate(self.blocks):
            if use_checkpoint:
                def _ckpt_forward(inp: torch.Tensor, blk: Block = block) -> torch.Tensor:
                    out, _ = blk(inp, past_kv=None, use_cache=False)
                    return out

                x = torch_checkpoint.checkpoint(_ckpt_forward, x)
            else:
                block_past = past_kv[i] if past_kv is not None else None
                x, present = block(x, past_kv=block_past, use_cache=use_cache)
                if use_cache:
                    presents.append(present)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=ignore_index if ignore_index is not None else -100,
            )
        return logits, loss, presents

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        past_kv = None
        finished = None
        for _ in range(max_new_tokens):
            if past_kv is None:
                if idx.size(1) > self.config.block_size:
                    idx_cond = idx[:, -self.config.block_size :]
                else:
                    idx_cond = idx
            else:
                if past_kv[0][0].size(2) >= self.config.block_size:
                    past_kv = None
                    idx_cond = idx[:, -self.config.block_size :]
                else:
                    idx_cond = idx[:, -1:]
            logits, _, past_kv = self(idx_cond, past_kv=past_kv, use_cache=True)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cumprobs = torch.cumsum(probs, dim=-1)
                mask = cumprobs > top_p
                mask[:, 0] = False
                sorted_logits[mask] = -float("inf")
                logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            if eos_token_id is not None:
                if finished is None:
                    finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
                eos_fill = torch.full_like(next_id, eos_token_id)
                next_id = torch.where(finished.unsqueeze(1), eos_fill, next_id)
            idx = torch.cat([idx, next_id], dim=1)
            if eos_token_id is not None:
                finished = finished | (next_id.squeeze(1) == eos_token_id)
                if torch.all(finished):
                    break
        return idx
