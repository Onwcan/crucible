"""
Pick the model size that best uses a fixed GPU-hour budget.

Given a few hours on one GPU, training the largest model that fits is usually
the wrong call. Chinchilla's result is that loss is minimised by balancing
parameters against tokens -- roughly 20 tokens per parameter -- and a model too
large for its token budget ends up worse than a smaller one trained properly on
the same compute.

This takes measured throughput (from probe_batch.py, not a datasheet) and
reports how far each preset lands from that balance.

Usage:
    python scripts/compute_budget.py --hours 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import PRESETS, GPT  # noqa: E402

# Measured with scripts/probe_batch.py at the best batch size for each preset.
# Replace these when the hardware or the model code changes.
MEASURED_TOK_S = {
    "30m": 250_000,     # batch 16, from training runs
    "120m": 50_827,     # batch 4
    "350m": 17_995,     # batch 4
}

CHINCHILLA_RATIO = 20      # tokens per parameter for compute-optimal training


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=float, default=6.0)
    p.add_argument("--corpus-tokens", type=float, default=1e9,
                   help="tokens available; training beyond this repeats data")
    p.add_argument("--tok-s", default=None,
                   help="override measured throughput, e.g. '120m=60000'")
    args = p.parse_args()

    measured = dict(MEASURED_TOK_S)
    if args.tok_s:
        for pair in args.tok_s.split(","):
            name, value = pair.split("=")
            measured[name.strip()] = float(value)

    seconds = args.hours * 3600
    print(f"budget: {args.hours:g} GPU-hours, corpus {args.corpus_tokens/1e9:.1f}B tokens")
    print()

    header = (f"{'preset':>7s} {'non-emb':>9s} {'tok/s':>9s} {'tokens':>9s} "
              f"{'tok/param':>10s} {'vs optimal':>11s}")
    print(header)
    print("-" * len(header))

    rows = []
    for name, cfg in PRESETS.items():
        rate = measured.get(name)
        if not rate:
            print(f"{name:>7s} {'':>9s} {'not measured':>9s}")
            continue

        params = GPT(cfg).num_params(non_embedding=True)
        tokens = min(rate * seconds, args.corpus_tokens)
        ratio = tokens / params
        optimal_fraction = ratio / CHINCHILLA_RATIO

        capped = " (corpus-capped)" if rate * seconds > args.corpus_tokens else ""
        print(f"{name:>7s} {params/1e6:8.1f}M {rate:9,.0f} "
              f"{tokens/1e9:8.2f}B {ratio:10.1f} {optimal_fraction:10.0%}{capped}")
        rows.append((name, abs(ratio - CHINCHILLA_RATIO), ratio, optimal_fraction))

    if not rows:
        print("\nno measured throughput -- run scripts/probe_batch.py first")
        return

    best = min(rows, key=lambda r: r[1])
    print()
    print(f"closest to compute-optimal: {best[0]} "
          f"({best[2]:.1f} tokens/param, {best[3]:.0%} of the {CHINCHILLA_RATIO}x target)")
    print()
    print("A model far below the target is undertrained: the same GPU-hours spent")
    print("on a smaller model would reach a lower loss. Far above it means the")
    print("model is too small to use the data available.")


if __name__ == "__main__":
    main()
