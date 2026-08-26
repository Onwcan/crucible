"""
Measure achievable memory bandwidth, and derive the decode throughput ceiling.

Single-stream decoding reads every weight once per token and reuses nothing, so
it is bandwidth-bound: no amount of arithmetic throughput raises the ceiling.
This measures the bandwidth actually achievable, then converts it into the
tokens/second no implementation can exceed.

Records the enforced power limit alongside the result, because this machine's
power profile is user-switchable between roughly 55 W and 175 W and a number
measured under one cap says nothing about another. An earlier baseline of
479 GB/s was taken under an eco profile and understated the hardware by ~45%.

Usage:
    python scripts/bench_bandwidth.py
    python scripts/bench_bandwidth.py --mb 512 --trials 40
"""
from __future__ import annotations

import argparse
import statistics
import subprocess

import torch


def envelope() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=enforced.power.limit,clocks.max.sm,temperature.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return out or "unavailable"
    except Exception:
        return "unavailable"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mb", type=int, default=256, help="buffer size in MB")
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--params", type=float, default=113e6,
                   help="model parameters, for the ceiling calculation")
    args = p.parse_args()

    print(f"device      : {torch.cuda.get_device_name(0)}")
    print(f"power/clocks: {envelope()}")

    n = args.mb * 1024 * 1024 // 4
    a = torch.empty(n, device="cuda", dtype=torch.float32).normal_()
    b = torch.empty_like(a)

    # Sustained warm-up: clocks only boost under continuous load, and a short
    # warm-up leaves the first trials running at idle frequency.
    for _ in range(100):
        b.copy_(a)
    torch.cuda.synchronize()

    samples = []
    for _ in range(args.trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        b.copy_(a)
        end.record()
        torch.cuda.synchronize()
        secs = start.elapsed_time(end) / 1000.0
        # copy_ reads a and writes b, so it moves twice the buffer size.
        samples.append(a.numel() * 4 * 2 / secs / 1e9)

    samples.sort()
    median = statistics.median(samples)
    spread = (samples[-1] - samples[0]) / median * 100

    print()
    print(f"bandwidth   : {median:.0f} GB/s median   "
          f"[{samples[0]:.0f}-{samples[-1]:.0f}]   spread {spread:.1f}%")

    if spread > 30:
        print("              spread above 30%: check the laptop power profile")

    print()
    print("decode ceiling for a bandwidth-bound single stream:")
    for label, bytes_per_param in (("f32", 4), ("f16/bf16", 2), ("int8", 1)):
        gb = args.params * bytes_per_param / 1e9
        print(f"  {label:9} {gb:5.2f} GB/token  ->  {median / gb:7.0f} tok/s")

    print()
    print("Quantisation raises this ceiling by moving fewer bytes, not by")
    print("computing faster -- which is why it matters more than FP8 for decode.")


if __name__ == "__main__":
    main()
