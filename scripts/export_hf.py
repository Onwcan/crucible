"""
Export a crucible checkpoint as a HuggingFace LlamaForCausalLM.

This exists so the engine can be compared against llama.cpp and vLLM, both of
which consume HuggingFace layouts. It is a rename, not a conversion: the
architecture here is already Llama -- RMSNorm, RoPE, SwiGLU, grouped-query
attention, no biases, tied embeddings -- and every tensor has the same shape and
memory layout HF expects.

The one thing worth checking rather than assuming is the rotary convention.
HF's `apply_rotary_pos_emb` uses `rotate_half`, which pairs element i with
i + head_dim/2 and gives

    out[i]        = q[i] * cos[i] - q[i + half] * sin[i]
    out[i + half] = q[i + half] * cos[i] + q[i] * sin[i]

which is exactly `ops::RopeTable::apply`. No permutation is needed. (The
permutation people hit converting Llama checkpoints comes from Meta's original
interleaved layout, which is a different convention from both of these.)

`--verify` runs the exported model through transformers and compares logits
against reference_logits.py. Skipping that would mean benchmarking a model that
loads, runs, generates fluent text, and is not the one that was trained.

Usage:
    python scripts/export_hf.py runs/120m-main --out export/120m-hf --verify
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


def build_state_dict(state: dict, n_layer: int) -> dict:
    """Rename crucible tensors to HF Llama names."""
    out = {"model.embed_tokens.weight": state["tok_emb.weight"]}
    for i in range(n_layer):
        src = f"blocks.{i}"
        dst = f"model.layers.{i}"
        out[f"{dst}.input_layernorm.weight"] = state[f"{src}.attn_norm.weight"]
        out[f"{dst}.post_attention_layernorm.weight"] = state[f"{src}.mlp_norm.weight"]
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            out[f"{dst}.self_attn.{proj}.weight"] = state[f"{src}.attn.{proj}.weight"]
        for proj in ("gate_proj", "up_proj", "down_proj"):
            out[f"{dst}.mlp.{proj}.weight"] = state[f"{src}.mlp.{proj}.weight"]
    out["model.norm.weight"] = state["final_norm.weight"]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", help="directory containing best.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--verify", action="store_true",
                   help="load through transformers and compare logits")
    p.add_argument("--tokens", default="464,2159,318,1719")
    args = p.parse_args()

    ckpt = torch.load(Path(args.run_dir) / args.checkpoint,
                      map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    state = {k.removeprefix("_orig_mod."): v.float()
             for k, v in ckpt["model"].items()}

    if cfg["activation"] != "swiglu" or cfg["norm"] != "rmsnorm" \
            or cfg["pos_encoding"] != "rope" or cfg["norm_placement"] != "pre":
        raise SystemExit(
            f"only the Llama-shaped configuration can be exported as Llama; got "
            f"{cfg['attention']}/{cfg['pos_encoding']}/{cfg['activation']}/"
            f"{cfg['norm_placement']}-{cfg['norm']}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    hf_state = build_state_dict(state, cfg["n_layer"])
    hidden = state["blocks.0.mlp.up_proj.weight"].shape[0]

    hf_config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": cfg["n_embd"],
        "intermediate_size": hidden,
        "num_hidden_layers": cfg["n_layer"],
        "num_attention_heads": cfg["n_head"],
        "num_key_value_heads": cfg["n_kv_head"],
        "max_position_embeddings": cfg["block_size"],
        "vocab_size": cfg["vocab_size"],
        "rms_norm_eps": 1e-6,
        "rope_theta": cfg["rope_theta"],
        "attention_bias": False,
        "mlp_bias": False,
        "hidden_act": "silu",
        # lm_head is not stored: it is tied to the embedding table.
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
        "bos_token_id": 50256,
        "eos_token_id": 50256,
    }

    from safetensors.torch import save_file

    save_file({k: v.contiguous() for k, v in hf_state.items()},
              out_dir / "model.safetensors",
              metadata={"format": "pt"})
    (out_dir / "config.json").write_text(json.dumps(hf_config, indent=2) + "\n",
                                         encoding="utf-8")

    # GPT-2 tokenizer files, so llama.cpp and vLLM can build a vocabulary.
    try:
        from transformers import GPT2TokenizerFast
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        tok.save_pretrained(out_dir)
        print("wrote GPT-2 tokenizer files")
    except Exception as exc:  # noqa: BLE001
        print(f"tokenizer export skipped ({type(exc).__name__}: {exc})")
        print("throughput benchmarks do not need it; generation quality does")

    total = sum(v.numel() for v in hf_state.values())
    print(f"exported {len(hf_state)} tensors, {total / 1e6:.1f}M values -> {out_dir}")

    if not args.verify:
        print()
        print("run again with --verify before trusting any benchmark built on this")
        return

    # --- verification ------------------------------------------------------
    ids = [int(t) for t in args.tokens.split(",")]

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from model import GPT, GPTConfig  # noqa: E402

    reference = GPT(GPTConfig(**cfg))
    reference.load_state_dict(state, strict=False)
    reference.eval()
    with torch.no_grad():
        ref_logits, _ = reference(torch.tensor([ids]))
    ref = ref_logits[0, -1].double()

    from transformers import LlamaForCausalLM

    hf = LlamaForCausalLM.from_pretrained(out_dir, dtype=torch.float32)
    hf.eval()
    with torch.no_grad():
        hf_logits = hf(torch.tensor([ids])).logits
    got = hf_logits[0, -1].double()

    max_rel = ((ref - got).abs() / ref.abs().clamp(min=1e-6)).max().item()
    ref_top = ref.topk(10).indices.tolist()
    got_top = got.topk(10).indices.tolist()

    print()
    print(f"crucible logits: sum {ref.sum():.4f}, max {ref.max():.6f}")
    print(f"exported  logits: sum {got.sum():.4f}, max {got.max():.6f}")
    print(f"max relative difference: {max_rel:.3e}")
    print(f"top-10 ids match: {ref_top == got_top}")

    if ref_top != got_top:
        print()
        print(f"  crucible: {ref_top}")
        print(f"  exported: {got_top}")
        raise SystemExit("EXPORT IS WRONG -- do not benchmark this")

    print()
    print("Export verified. Benchmarks against this file measure the same model.")


if __name__ == "__main__":
    main()
