"""
Correctness tests for the attention paths and the model as a whole.

The important one is test_gqa_paths_match. Grouped-query attention can be
computed two ways -- by materialising the KV heads with repeat_interleave, or
by letting the fused SDPA kernel do the grouping internally -- and they are
only interchangeable if both use the same query-head-to-KV-head mapping. If
their grouping conventions disagreed, training would still run and loss would
still fall; the model would just be quietly wrong. So the two paths are pinned
against each other numerically rather than assumed equivalent.

Run:  .venv/bin/python -m pytest tests/ -v
"""
from __future__ import annotations

import math

import pytest
import torch

import model as M
from model import PRESETS, Attention, GPT, GPTConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _forward(attn: Attention, x, cos=None, sin=None, alibi=None, *, fused: bool):
    """Run one attention forward with the fused GQA path forced on or off."""
    original = M.SDPA_HAS_GQA
    M.SDPA_HAS_GQA = fused
    try:
        return attn(x, cos, sin, alibi)
    finally:
        M.SDPA_HAS_GQA = original


@pytest.mark.parametrize("attention,n_kv_head", [
    ("gqa", 3),      # 12 query heads over 3 KV heads -> n_rep = 4
    ("gqa", 6),      # n_rep = 2
    ("mqa", 1),      # n_rep = 12, the extreme case
])
def test_gqa_paths_match(attention, n_kv_head):
    """Fused GQA and repeat_interleave must produce identical outputs."""
    if not M.SDPA_HAS_GQA:
        pytest.skip("this PyTorch build has no enable_gqa")

    torch.manual_seed(0)
    cfg = GPTConfig(n_layer=1, n_head=12, n_embd=768, block_size=64,
                    attention=attention, n_kv_head=n_kv_head, dropout=0.0)
    attn = Attention(cfg).to(DEVICE).eval()
    x = torch.randn(2, 32, cfg.n_embd, device=DEVICE)

    with torch.no_grad():
        fused = _forward(attn, x, fused=True)
        materialised = _forward(attn, x, fused=False)

    torch.testing.assert_close(fused, materialised, rtol=1e-4, atol=1e-5)


def test_gqa_paths_match_with_alibi():
    """The equivalence must also hold when an explicit attention bias is passed."""
    if not M.SDPA_HAS_GQA:
        pytest.skip("this PyTorch build has no enable_gqa")

    torch.manual_seed(0)
    T = 32
    cfg = GPTConfig(n_layer=1, n_head=12, n_embd=768, block_size=T,
                    attention="gqa", n_kv_head=3, pos_encoding="alibi",
                    dropout=0.0)
    attn = Attention(cfg).to(DEVICE).eval()
    x = torch.randn(2, T, cfg.n_embd, device=DEVICE)

    slopes = M.alibi_slopes(cfg.n_head).view(-1, 1, 1).to(DEVICE)
    pos = torch.arange(T, device=DEVICE)
    rel = pos.view(1, -1) - pos.view(-1, 1)
    causal = torch.full((T, T), float("-inf"), device=DEVICE).triu(1)
    alibi = (slopes * rel.unsqueeze(0) + causal).unsqueeze(0)

    with torch.no_grad():
        fused = _forward(attn, x, alibi=alibi, fused=True)
        materialised = _forward(attn, x, alibi=alibi, fused=False)

    torch.testing.assert_close(fused, materialised, rtol=1e-4, atol=1e-5)


def test_mha_unaffected():
    """With n_kv_head == n_head there is no grouping, so both paths are no-ops."""
    torch.manual_seed(0)
    cfg = GPTConfig(n_layer=1, n_head=12, n_embd=768, block_size=64,
                    attention="mha", dropout=0.0)
    assert cfg.n_kv_head == cfg.n_head

    attn = Attention(cfg).to(DEVICE).eval()
    x = torch.randn(2, 32, cfg.n_embd, device=DEVICE)
    with torch.no_grad():
        torch.testing.assert_close(_forward(attn, x, fused=True),
                                   _forward(attn, x, fused=False))


@pytest.mark.parametrize("attention", ["mha", "gqa", "mqa"])
@pytest.mark.parametrize("pos_encoding", ["rope", "alibi", "learned", "none"])
def test_forward_shapes_and_finite(attention, pos_encoding):
    """Every axis combination should produce a finite loss and correct shapes."""
    torch.manual_seed(0)
    cfg = GPTConfig(n_layer=2, n_head=4, n_embd=128, block_size=32,
                    vocab_size=256, attention=attention,
                    pos_encoding=pos_encoding, dropout=0.0)
    m = GPT(cfg).to(DEVICE)

    # Targets must be the *next* token, as in training. Passing idx as its own
    # target instead would make an untrained model look better than chance:
    # tok_emb.weight is tied to lm_head.weight, so the token's own embedding
    # sitting in the residual stream inflates its logit, and the model scores
    # well on copying without having learned anything.
    tokens = torch.randint(0, cfg.vocab_size, (2, 17), device=DEVICE)
    x, y = tokens[:, :-1], tokens[:, 1:]

    logits, loss = m(x, y)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert torch.isfinite(loss), f"non-finite loss for {attention}/{pos_encoding}"
    # Untrained on random targets, loss should sit at uniform: ln(vocab_size).
    uniform = math.log(cfg.vocab_size)
    assert abs(loss.item() - uniform) < 0.5, \
        f"{attention}/{pos_encoding}: loss {loss.item():.3f} vs uniform {uniform:.3f}"


def test_causality():
    """A token must not influence any earlier position's output."""
    torch.manual_seed(0)
    cfg = GPTConfig(n_layer=2, n_head=4, n_embd=128, block_size=32,
                    vocab_size=256, dropout=0.0)
    m = GPT(cfg).to(DEVICE).eval()

    idx = torch.randint(0, cfg.vocab_size, (1, 16), device=DEVICE)
    with torch.no_grad():
        base, _ = m(idx, idx)
        changed = idx.clone()
        changed[0, 10] = (changed[0, 10] + 1) % cfg.vocab_size
        after, _ = m(changed, changed)

    # Positions before the edit must be untouched.
    torch.testing.assert_close(base[:, :10], after[:, :10], rtol=1e-4, atol=1e-5)
    # And the edited position itself must actually change something.
    assert not torch.allclose(base[:, 10], after[:, 10])


def test_weight_tying():
    cfg = PRESETS["30m"]
    m = GPT(cfg)
    assert m.tok_emb.weight is m.lm_head.weight


def test_swiglu_param_parity():
    """
    SwiGLU uses three matrices instead of two, so its hidden dim is shrunk to
    keep parameter count comparable. Without this the activation ablation would
    be measuring extra parameters instead of the activation function.
    """
    base = dict(n_layer=4, n_head=8, n_embd=512, block_size=128)
    swiglu = GPT(GPTConfig(**base, activation="swiglu")).num_params(True)
    gelu = GPT(GPTConfig(**base, activation="gelu")).num_params(True)
    assert abs(swiglu - gelu) / gelu < 0.05, \
        f"parameter counts diverge: swiglu={swiglu:,} gelu={gelu:,}"


def test_gqa_never_degenerates():
    """
    GQA must sit strictly between MHA and MQA.

    The original `max(1, n_head // 4)` collapsed to n_kv_head=1 for n_head <= 7,
    making "gqa" identical to "mqa". The 30M ablation therefore compared the
    control against a duplicate of itself and reported the expected null result
    for reasons that had nothing to do with attention. This pins the invariant.
    """
    for n_head in (4, 6, 8, 12, 16, 32):
        cfg = GPTConfig(n_layer=1, n_head=n_head, n_embd=n_head * 64,
                        attention="gqa")
        assert 1 < cfg.n_kv_head < n_head, \
            f"n_head={n_head}: gqa gave n_kv_head={cfg.n_kv_head}"
        assert n_head % cfg.n_kv_head == 0

        mqa = GPTConfig(n_layer=1, n_head=n_head, n_embd=n_head * 64,
                        attention="mqa")
        mha = GPTConfig(n_layer=1, n_head=n_head, n_embd=n_head * 64,
                        attention="mha")
        assert cfg.n_kv_head != mqa.n_kv_head, f"gqa == mqa at n_head={n_head}"
        assert cfg.n_kv_head != mha.n_kv_head, f"gqa == mha at n_head={n_head}"


def test_gqa_raises_when_impossible():
    """With too few heads there is no valid middle ground; fail loudly."""
    with pytest.raises(ValueError, match="degenerates"):
        GPTConfig(n_layer=1, n_head=2, n_embd=128, attention="gqa")
