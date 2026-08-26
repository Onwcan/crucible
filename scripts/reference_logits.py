"""
Produce reference logits from PyTorch for validating the Rust engine.

The Rust CPU path and this must agree. A transformer that is subtly wrong --
a transposed projection, the other RoPE pairing convention, query heads mapped
to the wrong KV heads under GQA -- still runs, still produces a plausible
distribution, and still generates text that looks like text. None of those bugs
announce themselves. Comparing exact logits against a known-good implementation
is the only cheap way to catch them.

Runs on CPU in float32 to match the Rust reference path, so any difference is a
real discrepancy rather than a dtype artifact.

Usage:
    python scripts/reference_logits.py runs/120m-main --tokens 464,2159,318,1719
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import GPT, GPTConfig  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--tokens", default="464,2159,318,1719")
    p.add_argument("--top", type=int, default=10)
    args = p.parse_args()

    ids = [int(t.strip()) for t in args.tokens.split(",")]

    ckpt = torch.load(Path(args.run_dir) / args.checkpoint,
                      map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    state = {k.removeprefix("_orig_mod."): v.float()
             for k, v in ckpt["model"].items()}

    model = GPT(cfg)
    model.load_state_dict(state, strict=False)
    model.eval()

    idx = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        # forward() returns only the last position when targets are None,
        # which is exactly what the Rust path computes.
        logits, _ = model(idx)
    out = logits[0, -1].double()

    print(f"tokens {ids}")
    print(f"logits: sum {out.sum().item():.4f}, "
          f"min {out.min().item():.6f}, max {out.max().item():.6f}")
    print()
    print(f"top {args.top}:")
    values, indices = torch.topk(out, args.top)
    for i, v in zip(indices.tolist(), values.tolist()):
        print(f"  {i:6} {v:10.6f}")


if __name__ == "__main__":
    main()
