"""
Export a training checkpoint to safetensors for the Rust inference engine.

Checkpoints from train.py are torch .pt files, which are Python pickles --
readable only by PyTorch, and unsafe to load from untrusted sources. safetensors
is a flat, zero-copy, language-agnostic format that Rust can memory-map
directly, so it is the boundary between the Python training half of this repo
and the Rust serving half.

Also writes config.json, because the engine needs the architecture (head counts,
norm type, positional encoding) to build the right graph, and inferring that
from tensor shapes alone is guesswork.

Usage:
    python export.py runs/30m-control-s1337 --out export/30m
    python export.py runs/350m-main --out export/350m --dtype bf16
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from model import GPT, GPTConfig

DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", help="directory containing best.pt")
    p.add_argument("--out", default=None, help="output directory")
    p.add_argument("--dtype", default="fp32", choices=list(DTYPES))
    p.add_argument("--checkpoint", default="best.pt")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / args.checkpoint
    if not ckpt_path.exists():
        raise SystemExit(f"no checkpoint at {ckpt_path}")

    out_dir = Path(args.out or (run_dir / "export"))
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    state = ckpt["model"]

    # torch.compile prefixes every key with _orig_mod. when the module was
    # wrapped; strip it so the exported names match the plain model.
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}

    target = DTYPES[args.dtype]
    tensors = {}
    for name, tensor in state.items():
        # lm_head.weight is tied to tok_emb.weight and holds identical data.
        # safetensors cannot store aliased storage, so writing both would mean
        # duplicating the embedding table outright -- 206 MB at the 350M preset.
        # Store it once; config.json records the tying so the engine can
        # reconstruct it.
        if name == "lm_head.weight":
            continue
        t = tensor.to(target) if tensor.is_floating_point() else tensor
        tensors[name] = t.contiguous().clone()

    # Verify the export round-trips into a fresh model before writing it out.
    # A silently incomplete export would only surface as garbage generations
    # much later, in Rust, where it is far harder to diagnose.
    reference = GPT(cfg)
    restored = {k: v.float() for k, v in tensors.items()}
    restored["lm_head.weight"] = restored["tok_emb.weight"]      # re-tie
    missing, unexpected = reference.load_state_dict(restored, strict=False)
    if unexpected:
        raise SystemExit(f"unexpected keys in checkpoint: {sorted(unexpected)[:5]}")
    if missing:
        raise SystemExit(f"checkpoint is missing weights: {sorted(missing)[:5]}")

    weights_path = out_dir / "model.safetensors"
    save_file(tensors, weights_path,
              metadata={"format": "pt", "source": str(ckpt_path)})

    config = {
        **ckpt["config"],
        "dtype": args.dtype,
        "head_dim": cfg.head_dim,
        "trained_steps": ckpt.get("step"),
        "val_loss": ckpt.get("val_loss"),
        "tokenizer": "gpt2",
        "tie_word_embeddings": True,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n",
                                         encoding="utf-8")

    total = sum(t.numel() for t in tensors.values())
    size_mb = weights_path.stat().st_size / 1e6

    print(f"exported {len(tensors)} tensors, {total / 1e6:.1f}M values")
    print(f"  {weights_path}  ({size_mb:.1f} MB, {args.dtype})")
    print(f"  {out_dir / 'config.json'}")
    print(f"  step {ckpt.get('step')}, val loss {ckpt.get('val_loss'):.4f}")


if __name__ == "__main__":
    main()
