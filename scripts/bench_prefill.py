"""
Compare prefill throughput across the GEMM implementations.

Four variants:

  tiled   scalar 16x16 f32 kernel, the pre-tensor-core baseline
  small   tensor cores, 16x64 block tile
  big     tensor cores, 64x64 block tile
  auto    per-launch choice between small and big on predicted block count

`auto` exists because neither tile wins everywhere. The big tile has better
arithmetic intensity -- four times the output per block for twice the loads --
but covers that output with four times fewer blocks, so on a short prompt it
starves the GPU. The dispatch picks big only when a launch would still produce
at least two blocks per SM. That rule is a hypothesis; this script is how it
gets checked, and `auto` losing to a fixed tile would mean the rule is wrong.

Prefill is compute-bound: it multiplies a [seq, n_embd] activation block by every
weight matrix, so unlike decode it is not waiting on memory. That makes it the
one place tensor cores can help, and the one place a scalar kernel leaving 95% of
the machine idle actually costs something.

The measurement rules are the same ones the engine comparison uses, for the same
reasons:

  - Trials INTERLEAVED, one per variant per round. Running all of one variant
    then all of the other lets thermal drift masquerade as a kernel difference.
  - Median with spread shown, never a single run.
  - Power envelope recorded. This machine's limit is user-switchable between
    ~55 W and ~165 W and an earlier measurement swung 168% purely from that.

Correctness is not this script's job -- `gpu-validate` bounds the kernel error
and `gpu-eval --prefill-ctx` prices it in cross-entropy. A speedup that changed
the model's output would be worthless, and neither number is inferable from
throughput.

Usage:
    python scripts/bench_prefill.py --engine ../crucible-engine
"""
from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
from pathlib import Path

# label -> CRUCIBLE_GEMM value.
VARIANTS = [
    ("tiled", "tiled"),
    ("small", "wmma-small"),
    ("big", "wmma-big"),
    ("auto", "wmma-auto"),
]


def envelope() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,enforced.power.limit,clocks.max.sm",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:
        return "unavailable"


def run(binary: str, model: str, seq: int, gemm: str) -> float | None:
    env = dict(os.environ, CRUCIBLE_GEMM=gemm)
    out = subprocess.run(
        [binary, "gpu-logits", model, "--quant", "int8",
         "--decode", "0", "--prefill", str(seq)],
        capture_output=True, text=True, env=env, timeout=300)
    m = re.search(r"prefill\s+\d+ tokens in [\d.]+ ms\s+\(([\d.]+) tok/s", 
                  out.stdout + out.stderr)
    return float(m.group(1)) if m else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--binary", default="./target/release/llm-engine")
    p.add_argument("--model", default="/home/onur/llm-lab/export/120m")
    p.add_argument("--seqs", default="128,256,512,1024")
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--only", default=None,
                   help="comma-separated variant labels, e.g. small,big,auto")
    args = p.parse_args()

    variants = VARIANTS
    if args.only:
        wanted = {w.strip() for w in args.only.split(",")}
        unknown = wanted - {n for n, _ in VARIANTS}
        if unknown:
            raise SystemExit(f"unknown variant(s): {sorted(unknown)}")
        variants = [v for v in VARIANTS if v[0] in wanted]

    print(f"gpu     : {envelope()}")
    print(f"workload: int8 weights, prefill only, {args.trials} interleaved trials")
    print()

    names = [n for n, _ in variants]
    header = (f"{'seq':>6}" + "".join(f"{n:>10}" for n in names)
              + f"{'best':>8}")
    print(header)
    print("-" * len(header))

    for seq in [int(s) for s in args.seqs.split(",")]:
        samples: dict[str, list[float]] = {n: [] for n in names}
        for _ in range(args.trials):
            for name, flag in variants:      # interleaved, not blocked
                v = run(args.binary, args.model, seq, flag)
                if v is not None:
                    samples[name].append(v)

        if not all(samples.values()):
            print(f"{seq:>6}  measurement failed")
            continue

        med = {k: statistics.median(v) for k, v in samples.items()}
        best = max(med, key=med.get)
        print(f"{seq:>6}" + "".join(f"{med[n]:10.0f}" for n in names)
              + f"{best:>8}")
        # Per-variant spread, not the max across variants: one unstable variant
        # otherwise taints the whole row and hides which one it was.
        spread = {n: (max(v) - min(v)) / med[n] * 100 for n, v in samples.items()}
        print(f"{'':>6}" + "".join(f"{spread[n]:9.1f}%" for n in names)
              + f"{'spread':>8}")

    print()
    print("Speed alone decides nothing: see gpu-validate for kernel error and")
    print("gpu-eval --prefill-ctx for the cross-entropy this buys or costs.")


if __name__ == "__main__":
    main()
