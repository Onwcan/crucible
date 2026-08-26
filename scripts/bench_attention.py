"""
Compare attention implementations under training conditions.

The ablation sweep reported MHA as faster than GQA, which is backwards from
theory, and suggested the cause was `repeat_interleave` materialising the KV
heads. This measures that claim directly.

Two things this gets right that a quick timing loop does not:

  - Runs under torch.compile, matching how training actually executes. An
    eager-mode measurement answers a different question and disagreed with the
    compiled one.
  - Reports spread across trials and peak memory, so a 3% difference is not
    mistaken for a result when the noise floor is wider than that.

Usage:
    .venv/bin/python scripts/bench_attention.py
    .venv/bin/python scripts/bench_attention.py --no-compile --batch 8
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import model as M                      # noqa: E402
from model import GPT, GPTConfig       # noqa: E402


def bench(attention: str, fused: bool, args) -> tuple[float, float, float]:
    """Return (median ms/step, spread %, peak GB) for one configuration."""
    torch.manual_seed(0)
    cfg = GPTConfig(n_layer=6, n_head=6, n_embd=384,
                    block_size=args.block, attention=attention)

    original = M.SDPA_HAS_GQA
    M.SDPA_HAS_GQA = fused
    model = None
    try:
        model = GPT(cfg).cuda()
        if args.compile:
            model = torch.compile(model)

        x = torch.randint(0, cfg.vocab_size, (args.batch, args.block), device="cuda")
        y = torch.randint(0, cfg.vocab_size, (args.batch, args.block), device="cuda")

        def step():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss.backward()

        for _ in range(args.warmup):       # includes compilation
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        samples = []
        for _ in range(args.trials):
            start = time.perf_counter()
            for _ in range(args.iters):
                step()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) / args.iters * 1000)

        peak = torch.cuda.max_memory_allocated() / 1e9
        median = statistics.median(samples)
        spread = (max(samples) - min(samples)) / median * 100
        return median, spread, peak
    finally:
        M.SDPA_HAS_GQA = original
        del model
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--block", type=int, default=1024)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--warmup", type=int, default=12)
    p.add_argument("--compile", action="store_true", default=True)
    p.add_argument("--no-compile", dest="compile", action="store_false")
    args = p.parse_args()

    mode = "COMPILED (matches training)" if args.compile else "EAGER"
    print(f"{mode}, batch {args.batch} x {args.block}, "
          f"median of {args.trials} trials")
    print(f"fused GQA available: {M.SDPA_HAS_GQA}")
    print()

    configs = [
        ("MHA", "mha", False),
        ("GQA repeat_interleave", "gqa", False),
        ("GQA fused", "gqa", True),
        ("MQA repeat_interleave", "mqa", False),
        ("MQA fused", "mqa", True),
    ]

    header = f"{'variant':24s} {'ms/step':>9s} {'spread':>8s} {'peak GB':>9s}"
    print(header)
    print("-" * len(header))

    results = []
    for label, attention, fused in configs:
        try:
            ms, spread, peak = bench(attention, fused, args)
            results.append((label, ms, spread, peak))
            print(f"{label:24s} {ms:9.1f} {spread:7.1f}% {peak:8.2f}")
        except Exception as exc:
            print(f"{label:24s} failed: {type(exc).__name__}: {exc}")

    if not results:
        return

    worst_spread = max(r[2] for r in results)
    fastest = min(r[1] for r in results)
    slowest = max(r[1] for r in results)
    total_range = (slowest - fastest) / fastest * 100

    print()
    print(f"noise floor (widest spread) : {worst_spread:.1f}%")
    print(f"total range across variants : {total_range:.1f}%")
    if total_range < worst_spread:
        print()
        print("The variants differ by less than the measurement noise. No "
              "attention implementation is faster than another at this size; "
              "choose on memory or KV-cache footprint instead.")


if __name__ == "__main__":
    main()
