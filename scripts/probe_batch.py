"""
Find the largest batch size a preset fits in VRAM, and how fast it runs.

A long training run that OOMs an hour in wastes the whole slot, and the memory
ceiling is not something to estimate on paper: optimiser state, gradients,
autocast copies and activation checkpointing all interact. So this just tries
increasing batch sizes until one fails, reporting peak memory and step time for
each, then recommends a batch/grad-accum pair hitting a target tokens-per-step.

Usage:
    .venv/bin/python scripts/probe_batch.py --preset 350m
    .venv/bin/python scripts/probe_batch.py --preset 350m --target-tokens 131072
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import PRESETS, GPT  # noqa: E402


def try_batch(preset: str, batch: int, iters: int = 4) -> dict | None:
    """Return timing/memory for one batch size, or None if it does not fit."""
    cfg = PRESETS[preset]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = optimizer = None
    try:
        model = GPT(cfg).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
        x = torch.randint(0, cfg.vocab_size, (batch, cfg.block_size), device="cuda")
        y = torch.randint(0, cfg.vocab_size, (batch, cfg.block_size), device="cuda")

        def step():
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss.backward()
            optimizer.step()

        step()                       # allocate steady-state memory first
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            step()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - start) / iters * 1000

        return {"batch": batch, "ms": ms,
                "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
                "tok_s": batch * cfg.block_size / (ms / 1000)}
    except torch.cuda.OutOfMemoryError:
        return None
    finally:
        del model, optimizer
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="350m", choices=list(PRESETS))
    p.add_argument("--target-tokens", type=int, default=65536,
                   help="tokens per optimiser step to aim for")
    p.add_argument("--batches", default="1,2,4,6,8,12,16")
    args = p.parse_args()

    cfg = PRESETS[args.preset]
    model = GPT(cfg)
    total, free = None, torch.cuda.mem_get_info()[1] / 1e9
    print(f"preset {args.preset}: {model.num_params(False)/1e6:.1f}M params "
          f"({model.num_params(True)/1e6:.1f}M non-embedding), "
          f"block_size {cfg.block_size}")
    print(f"attention: {cfg.attention} (n_kv_head={cfg.n_kv_head}), VRAM {free:.1f} GB")
    del model
    print()

    header = f"{'batch':>6s} {'ms/step':>9s} {'peak GB':>9s} {'tok/s':>10s}"
    print(header)
    print("-" * len(header), flush=True)

    # WSL2 does not raise OutOfMemoryError. The driver spills to host RAM over
    # PCIe instead, so an oversized batch "succeeds" while running orders of
    # magnitude slower -- 200x slower was observed at the 350M preset. Catching
    # OutOfMemoryError alone would therefore report a working configuration that
    # is unusable, so the throughput cliff is detected directly.
    ok = []
    for b in [int(x) for x in args.batches.split(",")]:
        r = try_batch(args.preset, b)
        if r is None:
            print(f"{b:6d}   OOM -- stopping", flush=True)
            break

        flag = ""
        if r["peak_gb"] > free * 0.97:
            flag = "  <- exceeds VRAM, spilling to host RAM"
        elif ok and r["tok_s"] < max(x["tok_s"] for x in ok) * 0.75:
            flag = "  <- throughput collapsed"

        print(f"{b:6d} {r['ms']:9.1f} {r['peak_gb']:9.2f} "
              f"{r['tok_s']:10,.0f}{flag}", flush=True)

        if flag:
            print(f"       stopping: larger batches only get worse", flush=True)
            break
        ok.append(r)

    if not ok:
        print("\nnothing fits; reduce block_size or the model")
        return

    # Prefer the largest batch that fits: fewer, larger kernels beat many small
    # ones, and gradient accumulation makes up the rest of the token budget.
    best = max(ok, key=lambda r: r["tok_s"])
    per_micro = best["batch"] * cfg.block_size
    accum = max(1, round(args.target_tokens / per_micro))
    tokens_per_step = per_micro * accum
    step_ms = best["ms"] * accum

    print()
    print(f"recommended: --batch-size {best['batch']} --grad-accum {accum}")
    print(f"  {tokens_per_step:,} tokens/step, ~{step_ms/1000:.2f} s/step")
    for hours in (4, 6, 8):
        steps = int(hours * 3600 / (step_ms / 1000))
        print(f"  {hours}h -> ~{steps:,} steps, "
              f"{steps * tokens_per_step / 1e9:.2f}B tokens")


if __name__ == "__main__":
    main()
