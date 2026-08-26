"""
Benchmark harness with variance reporting.

Single-shot timings on a thermally-limited laptop GPU are not trustworthy:
back-to-back runs of the same matmul have shown ~8% spread. Any optimisation
claiming less than that is unfalsifiable without repeated measurement, so
everything here reports a distribution rather than one number.

Design notes:
  - CUDA events, not perf_counter: measures GPU time, immune to host jitter.
  - Many trials of many iterations: the trial is the unit of statistics.
  - Median + IQR, not mean + stdev: throughput is skewed by throttling, and
    the median is not dragged around by a single slow trial.
  - Clocks and temperature sampled before/after to make throttling visible
    instead of silently corrupting the numbers.

Usage:
    .venv/bin/python bench.py
    .venv/bin/python bench.py --trials 30 --size 8192
"""
from __future__ import annotations

import argparse
import statistics
import subprocess
from dataclasses import dataclass
from typing import Callable

import torch

GREEN, YELLOW, RESET = "\033[92m", "\033[93m", "\033[0m"


# --------------------------------------------------------------------------
# GPU state
# --------------------------------------------------------------------------

def gpu_state() -> dict[str, str]:
    """Sample clocks, temperature and throttle reasons via nvidia-smi."""
    fields = [
        "clocks.current.sm",
        "clocks.max.sm",
        "temperature.gpu",
        "power.draw",
        "power.limit",
    ]
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().split(", ")
        return dict(zip(fields, out))
    except Exception:
        return {}


def print_gpu_state(label: str) -> None:
    state = gpu_state()
    if not state:
        return
    sm = state.get("clocks.current.sm", "?")
    sm_max = state.get("clocks.max.sm", "?")
    temp = state.get("temperature.gpu", "?")
    power = state.get("power.draw", "?")
    limit = state.get("power.limit", "?")
    print(f"  [{label:5s}] SM {sm}/{sm_max} MHz | {temp}C | {power}/{limit} W")


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------

@dataclass
class Result:
    """Throughput distribution across trials, in TFLOP/s."""
    name: str
    samples: list[float]

    @property
    def median(self) -> float:
        return statistics.median(self.samples)

    @property
    def iqr(self) -> tuple[float, float]:
        ordered = sorted(self.samples)
        n = len(ordered)
        return ordered[n // 4], ordered[(3 * n) // 4]

    @property
    def spread_pct(self) -> float:
        """Full range as a percentage of the median -- the noise floor."""
        if not self.samples or self.median == 0:
            return 0.0
        return (max(self.samples) - min(self.samples)) / self.median * 100

    def __str__(self) -> str:
        lo, hi = self.iqr
        return (f"{self.median:7.1f} TFLOP/s  "
                f"[IQR {lo:6.1f}-{hi:6.1f}]  "
                f"spread {self.spread_pct:4.1f}%")


def time_trials(fn: Callable[[], object], flops: float,
                iters: int = 50, trials: int = 15,
                warmup: int = 20) -> list[float]:
    """Run fn repeatedly, returning one TFLOP/s sample per trial."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    start_ev, end_ev = torch.cuda.Event(True), torch.cuda.Event(True)

    for _ in range(trials):
        start_ev.record()
        for _ in range(iters):
            fn()
        end_ev.record()
        torch.cuda.synchronize()
        seconds = start_ev.elapsed_time(end_ev) / 1000.0 / iters
        samples.append(flops / seconds / 1e12)

    return samples


# --------------------------------------------------------------------------
# Workloads
# --------------------------------------------------------------------------

def make_matmul(dtype, n: int, tf32: bool = False):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    a = torch.randn(n, n, device="cuda", dtype=torch.float32).to(dtype)
    b = torch.randn(n, n, device="cuda", dtype=torch.float32).to(dtype)
    return lambda: a @ b


def make_fp8(n: int, fast_accum: bool):
    dtype = torch.float8_e4m3fn
    a = torch.randn(n, n, device="cuda", dtype=torch.float32).to(dtype)
    # Right operand must be column-major for the FP8 kernel to accept it.
    b = torch.randn(n, n, device="cuda", dtype=torch.float32).to(dtype)
    b = b.t().contiguous().t()
    scale = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    return lambda: torch._scaled_mm(a, b, scale_a=scale, scale_b=scale,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=fast_accum)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--trials", type=int, default=15)
    args = parser.parse_args()

    n = args.size
    flops = 2 * n ** 3

    print(f"device : {torch.cuda.get_device_name(0)}")
    print(f"config : {n}x{n} matmul, {args.iters} iters x {args.trials} trials")
    print()

    # Bring clocks up BEFORE the first workload. Without this the GPU idles at
    # ~600 MHz and whichever workload runs first is measured cold, which both
    # inflates its spread and skews every ratio computed against it.
    print_gpu_state("cold")
    warm = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(300):
        warm = warm @ warm.T if warm.shape[0] == warm.shape[1] else warm
        warm = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    del warm
    torch.cuda.empty_cache()
    print_gpu_state("warm")

    workloads = [
        ("FP32", lambda: make_matmul(torch.float32, n, tf32=False)),
        ("TF32", lambda: make_matmul(torch.float32, n, tf32=True)),
        ("FP16", lambda: make_matmul(torch.float16, n)),
        ("BF16", lambda: make_matmul(torch.bfloat16, n)),
        ("FP8 e4m3", lambda: make_fp8(n, fast_accum=False)),
        ("FP8 e4m3 fast", lambda: make_fp8(n, fast_accum=True)),
    ]

    results: list[Result] = []
    print()
    print("--- throughput (median of trials) ---")
    for name, build in workloads:
        try:
            fn = build()
            samples = time_trials(fn, flops, args.iters, args.trials)
            result = Result(name, samples)
            results.append(result)
            print(f"  {name:15s}: {result}")
            del fn
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"  {name:15s}: {YELLOW}failed: {type(exc).__name__}: {exc}{RESET}")

    print()
    print_gpu_state("end")

    if not results:
        return

    # Speedups relative to FP32, with a warning when the gap is inside noise.
    baseline = next((r for r in results if r.name == "FP32"), results[0])
    worst_noise = max(r.spread_pct for r in results)

    print()
    print(f"--- speedup vs {baseline.name} ---")
    for r in results:
        ratio = r.median / baseline.median
        print(f"  {r.name:15s}: {ratio:5.2f}x")

    print()
    print(f"Noise floor across all workloads: {worst_noise:.1f}%")
    print(f"{YELLOW}Differences smaller than this are not distinguishable. "
          f"Treat any optimisation claiming less as unproven.{RESET}")


if __name__ == "__main__":
    main()
