"""
A decoder-only transformer with the architecture choices exposed as config
flags rather than baked in.

The point of this file is the ablation study: attention variant, positional
encoding, activation, and norm placement are all switchable, so comparing them
means changing a string and re-running, not maintaining four forks of a model.

Defaults follow current practice (GQA + RoPE + SwiGLU + pre-RMSNorm), which
makes the default config the control group.
"""
from __future__ import annotations


import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

def _detect_gqa_support() -> bool:
    """
    PyTorch >= 2.5 can group query heads inside the fused SDPA kernel. Without
    it, KV heads must be materialised with repeat_interleave, which allocates
    n_rep copies of K and V for nothing -- and made GQA slower than plain MHA
    in this repo's own sweep, which is the wrong way round.

    Probed by calling it rather than by inspecting the signature:
    scaled_dot_product_attention is a C builtin that inspect cannot read, and
    a successful call proves the path actually works, not merely that a
    keyword is accepted.
    """
    try:
        q = torch.zeros(1, 2, 1, 8)
        kv = torch.zeros(1, 1, 1, 8)
        F.scaled_dot_product_attention(q, kv, kv, enable_gqa=True)
        return True
    except (TypeError, RuntimeError):
        return False


SDPA_HAS_GQA = _detect_gqa_support()


@dataclass
class GPTConfig:
    vocab_size: int = 50304          # padded to a multiple of 64 for tensor cores
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int | None = None     # None -> equal to n_head (plain MHA)
    n_embd: int = 768
    block_size: int = 1024
    dropout: float = 0.0
    bias: bool = False               # biases in Linear/Norm; modern models omit

    # --- ablation axes -----------------------------------------------------
    attention: Literal["mha", "gqa", "mqa"] = "gqa"
    pos_encoding: Literal["rope", "alibi", "learned", "none"] = "rope"
    activation: Literal["swiglu", "gelu"] = "swiglu"
    norm: Literal["rmsnorm", "layernorm"] = "rmsnorm"
    norm_placement: Literal["pre", "post"] = "pre"
    rope_theta: float = 10000.0

    def __post_init__(self):
        # Resolve the attention variant into a concrete KV-head count so the
        # rest of the model only ever reads n_kv_head.
        if self.attention == "mha":
            self.n_kv_head = self.n_head
        elif self.attention == "mqa":
            self.n_kv_head = 1
        else:  # gqa
            if self.n_kv_head is None:
                # Floor of 2, not 1. The earlier `max(1, n_head // 4)` silently
                # collapsed to n_kv_head=1 for n_head <= 7 -- meaning "gqa" and
                # "mqa" resolved to the identical architecture, and an ablation
                # comparing them compared a config against itself. Guarded
                # below so that can never pass unnoticed again.
                self.n_kv_head = max(2, self.n_head // 4)
                while self.n_head % self.n_kv_head != 0 and self.n_kv_head < self.n_head:
                    self.n_kv_head += 1

        assert self.n_embd % self.n_head == 0, "n_embd must divide by n_head"
        assert self.n_head % self.n_kv_head == 0, "n_head must divide by n_kv_head"

        # GQA is only meaningful strictly between MHA and MQA. Degenerating to
        # either end makes the attention axis compare duplicate configurations.
        if self.attention == "gqa" and not (1 < self.n_kv_head < self.n_head):
            raise ValueError(
                f"gqa degenerates with n_head={self.n_head}: n_kv_head resolved "
                f"to {self.n_kv_head}, which is "
                f"{'mqa' if self.n_kv_head == 1 else 'mha'}. Use n_head >= 4 "
                f"with a non-prime head count, or set n_kv_head explicitly.")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """No mean subtraction and no bias -- cheaper than LayerNorm, same effect."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalise in fp32 for stability, then cast back to the input dtype.
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def make_norm(cfg: GPTConfig) -> nn.Module:
    if cfg.norm == "rmsnorm":
        return RMSNorm(cfg.n_embd)
    return nn.LayerNorm(cfg.n_embd, bias=cfg.bias)


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

def precompute_rope(head_dim: int, max_seq: int, theta: float,
                    device=None) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(max_seq, device=device).float()
    freqs = torch.outer(pos, inv_freq)           # (T, head_dim/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, n_head, T, head_dim). Rotates the two halves against each other."""
    T = x.shape[-2]
    cos = cos[:T].view(1, 1, T, -1)
    sin = sin[:T].view(1, 1, T, -1)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).type_as(x)


def alibi_slopes(n_head: int) -> torch.Tensor:
    """Geometric sequence of per-head penalties, per the ALiBi paper."""
    def pow2_slopes(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * (start ** i) for i in range(n)]

    if math.log2(n_head).is_integer():
        return torch.tensor(pow2_slopes(n_head))

    # Non power-of-two head counts: take the next power of two and interleave.
    closest = 2 ** math.floor(math.log2(n_head))
    slopes = pow2_slopes(closest)
    slopes += pow2_slopes(2 * closest)[0::2][: n_head - closest]
    return torch.tensor(slopes)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_head // cfg.n_kv_head

        # Fused QKV would be one matmul, but separate projections keep the
        # GQA/MQA shapes readable and cost nothing measurable at this scale.
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_head * self.head_dim, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=cfg.bias)
        self.o_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = cfg.dropout

    def forward(self, x, cos=None, sin=None, alibi_bias=None):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        if cos is not None:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        # GQA/MQA: query heads share KV heads. Prefer letting the fused kernel
        # handle the grouping; materialising it with repeat_interleave costs
        # n_rep copies of K and V and made GQA slower than MHA in this repo's
        # own sweep. tests/test_attention.py pins the two paths to identical
        # outputs so the fast path cannot silently diverge.
        use_fused_gqa = self.n_rep > 1 and SDPA_HAS_GQA
        if self.n_rep > 1 and not use_fused_gqa:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=alibi_bias,                     # None -> pure causal
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=alibi_bias is None,
            **({"enable_gqa": True} if use_fused_gqa else {}),
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


# ---------------------------------------------------------------------------
# Feed-forward
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.activation = cfg.activation

        if cfg.activation == "swiglu":
            # SwiGLU uses three matrices, so shrink the hidden dim to 2/3 * 4d
            # to keep the parameter count comparable to a GeLU MLP.
            hidden = int(2 * (4 * cfg.n_embd) / 3)
            hidden = 64 * ((hidden + 63) // 64)       # round up for tensor cores
            self.gate_proj = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
            self.up_proj = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
            self.down_proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        else:
            hidden = 4 * cfg.n_embd
            self.up_proj = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
            self.down_proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)

        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        if self.activation == "swiglu":
            x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        else:
            x = F.gelu(self.up_proj(x), approximate="tanh")
        return self.drop(self.down_proj(x))


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm_placement = cfg.norm_placement
        self.attn_norm = make_norm(cfg)
        self.attn = Attention(cfg)
        self.mlp_norm = make_norm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x, cos=None, sin=None, alibi_bias=None):
        if self.norm_placement == "pre":
            x = x + self.attn(self.attn_norm(x), cos, sin, alibi_bias)
            x = x + self.mlp(self.mlp_norm(x))
        else:  # post-norm: the original Transformer arrangement
            x = self.attn_norm(x + self.attn(x, cos, sin, alibi_bias))
            x = self.mlp_norm(x + self.mlp(x))
        return x


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = (nn.Embedding(cfg.block_size, cfg.n_embd)
                        if cfg.pos_encoding == "learned" else None)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.final_norm = make_norm(cfg)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying: saves vocab_size * n_embd params and usually helps.
        self.tok_emb.weight = self.lm_head.weight

        if cfg.pos_encoding == "rope":
            cos, sin = precompute_rope(cfg.head_dim, cfg.block_size, cfg.rope_theta)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

        if cfg.pos_encoding == "alibi":
            self.register_buffer("alibi", self._build_alibi(), persistent=False)

        self.apply(self._init_weights)
        # Scale residual-output projections by 1/sqrt(2*n_layer) so the residual
        # stream does not grow with depth (GPT-2 initialisation scheme).
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _build_alibi(self) -> torch.Tensor:
        T = self.cfg.block_size
        slopes = alibi_slopes(self.cfg.n_head).view(-1, 1, 1)
        pos = torch.arange(T)
        rel = pos.view(1, -1) - pos.view(-1, 1)          # (T, T), negative below diagonal
        bias = slopes * rel.unsqueeze(0)                  # (n_head, T, T)
        causal = torch.full((T, T), float("-inf")).triu(1)
        return (bias + causal).unsqueeze(0)               # (1, n_head, T, T)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        """
        Parameter count. `non_embedding` excludes the token (and learned
        positional) embeddings, which is the convention scaling-law work uses.

        This matters more than it looks at small scale: with a 50k vocabulary
        the embedding table is 19M parameters, which is two thirds of the 30M
        preset. Comparing presets on total parameters would mostly compare
        embedding tables rather than transformer capacity.
        """
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # tok_emb.weight is tied to lm_head.weight, so parameters() counts
            # it once and subtracting once is correct.
            n -= self.tok_emb.weight.numel()
            if self.pos_emb is not None:
                n -= self.pos_emb.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence {T} exceeds block_size"

        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(x)

        cos = sin = alibi_bias = None
        if self.cfg.pos_encoding == "rope":
            cos, sin = self.rope_cos, self.rope_sin
        elif self.cfg.pos_encoding == "alibi":
            alibi_bias = self.alibi[:, :, :T, :T]

        for block in self.blocks:
            x = block(x, cos, sin, alibi_bias)
        x = self.final_norm(x)

        if targets is None:
            # Inference: only the last position is needed.
            return self.lm_head(x[:, [-1], :]), None

        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               targets.reshape(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


# Named configs for the three training scales.
PRESETS = {
    "30m":  GPTConfig(n_layer=6,  n_head=6,  n_embd=384,  block_size=1024),
    "120m": GPTConfig(n_layer=12, n_head=12, n_embd=768,  block_size=1024),
    "350m": GPTConfig(n_layer=24, n_head=16, n_embd=1024, block_size=1024),
}


if __name__ == "__main__":
    print(f"{'preset':7s} {'total':>9s} {'non-embed':>11s} {'embed %':>8s}   config")
    for name, cfg in PRESETS.items():
        model = GPT(cfg)
        total = model.num_params(non_embedding=False)
        body = model.num_params(non_embedding=True)
        print(f"{name:7s} {total / 1e6:8.1f}M {body / 1e6:10.1f}M "
              f"{100 * (total - body) / total:7.1f}%   "
              f"{cfg.attention}/{cfg.pos_encoding}/{cfg.activation}/"
              f"{cfg.norm_placement}-{cfg.norm}")
