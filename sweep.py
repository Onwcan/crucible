"""
Architecture ablation sweep.

Changes exactly one axis at a time against a fixed control, runs every variant
across multiple seeds, and reports whether the gap between a variant and the
control is larger than the spread between seeds of the same config.

That last part is the whole point. A single training run per config produces a
number, and numbers invite conclusions -- but two seeds of an *identical* config
differ too, and if that difference is as large as the one between architectures
then the architecture comparison says nothing. This reports both, so the
distinction is visible instead of assumed.

Each run is a separate process so CUDA state, compilation caches and RNG do not
leak between configurations.

Usage:
    python sweep.py --steps 2000 --seeds 3           # run everything
    python sweep.py --axis attention --steps 2000    # one axis only
    python sweep.py --report                         # aggregate, run nothing
"""
from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
import time
from pathlib import Path

# Verdict logic lives in analyze.py and is imported rather than duplicated,
# so the sweep summary and the full report can never disagree.
from analyze import DIVERGENCE_MARGIN, verdict_for

# The control config: current standard practice. Every variant differs from
# this in exactly one axis.
CONTROL = {
    "attention": "gqa",
    "pos_encoding": "rope",
    "activation": "swiglu",
    "norm": "rmsnorm",
    "norm_placement": "pre",
}

AXES = {
    "attention": ["mha", "gqa", "mqa"],
    "pos_encoding": ["rope", "alibi", "learned", "none"],
    "activation": ["swiglu", "gelu"],
    "norm": ["rmsnorm", "layernorm"],
    "norm_placement": ["pre", "post"],
}


def variant_name(axis: str, value: str, seed: int, preset: str) -> str:
    return f"{preset}-{axis}={value}-s{seed}"


def enumerate_runs(axes: list[str], seeds: list[int], preset: str):
    """Yield (axis, value, seed) for every run, control included once per seed."""
    seen = set()
    for axis in axes:
        for value in AXES[axis]:
            for seed in seeds:
                # The control appears in every axis; run it once, share it.
                is_control = value == CONTROL[axis]
                key = ("control", seed) if is_control else (axis, value, seed)
                if key in seen:
                    continue
                seen.add(key)
                yield axis, value, seed, is_control


def run_one(axis: str, value: str, seed: int, is_control: bool,
            args) -> tuple[str, bool]:
    name = (f"{args.preset}-control-s{seed}" if is_control
            else variant_name(axis, value, seed, args.preset))
    out_dir = Path(args.out) / name

    if (out_dir / "log.csv").exists() and not args.force:
        print(f"  skip (exists): {name}")
        return name, True

    cmd = [
        sys.executable, "train.py",
        "--preset", args.preset,
        "--data", args.data,
        "--steps", str(args.steps),
        "--seed", str(seed),
        "--out", args.out,
        "--run-name", name,
        "--batch-size", str(args.batch_size),
        "--grad-accum", str(args.grad_accum),
    ]
    if not is_control:
        cmd += [f"--{axis.replace('_', '-')}", value]

    print(f"  running: {name}")
    started = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = (time.time() - started) / 60

    if result.returncode != 0:
        print(f"  FAILED ({elapsed:.1f} min): {name}")
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        return name, False

    print(f"  done ({elapsed:.1f} min): {name}")
    return name, True


def best_val_loss(run_dir: Path) -> float | None:
    log = run_dir / "log.csv"
    if not log.exists():
        return None
    best = None
    with open(log, newline="") as f:
        for row in csv.DictReader(f):
            raw = row.get("val_loss", "")
            if raw:
                v = float(raw)
                best = v if best is None else min(best, v)
    return best


def report(args) -> None:
    out = Path(args.out)

    control_seeds = []
    for d in out.glob(f"{args.preset}-control-s*"):
        v = best_val_loss(d)
        if v is not None:
            control_seeds.append(v)

    if not control_seeds:
        print("no control runs found -- nothing to compare against")
        return

    control_mean = statistics.mean(control_seeds)
    control_sd = statistics.stdev(control_seeds) if len(control_seeds) > 1 else 0.0

    print(f"control ({CONTROL['attention']}/{CONTROL['pos_encoding']}/"
          f"{CONTROL['activation']}/{CONTROL['norm_placement']}-{CONTROL['norm']})")
    print(f"  best val loss: {control_mean:.4f} +/- {control_sd:.4f} "
          f"over {len(control_seeds)} seed(s)")
    print()

    if control_sd == 0.0:
        print("WARNING: only one control seed. Seed variance is unknown, so no")
        print("         difference below can be called meaningful. Run with")
        print("         --seeds 3 to establish the noise floor.")
        print()

    header = f"{'axis':16s} {'variant':10s} {'val loss':>10s} {'sd':>7s} {'vs control':>11s}   verdict"
    print(header)
    print("-" * len(header))

    for axis, values in AXES.items():
        for value in values:
            if value == CONTROL[axis]:
                continue
            seeds = []
            for d in out.glob(f"{args.preset}-{axis}={value}-s*"):
                v = best_val_loss(d)
                if v is not None:
                    seeds.append(v)
            if not seeds:
                continue

            # Shared with analyze.py so the two reports cannot disagree.
            st = verdict_for(control_seeds, seeds)
            print(f"{axis:16s} {value:10s} {st['mean']:10.4f} {st['sd']:7.4f} "
                  f"{st['delta']:+11.4f}   {st['verdict']}")

    print()
    print(f"Verdict rule: a seed more than {DIVERGENCE_MARGIN:.1f} above the control")
    print("is treated as diverged, excluded from mean/SD, and flags the variant")
    print("UNSTABLE. Otherwise |delta| must exceed 2x the larger seed SD to count.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="30m")
    p.add_argument("--data", default="data/tiny")
    p.add_argument("--out", default="runs")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--axis", default=None, choices=list(AXES),
                   help="run one axis only (default: all)")
    p.add_argument("--axes", default=None,
                   help="comma-separated axes to run, e.g. "
                        "'pos_encoding,norm_placement'. Useful for rerunning "
                        "only the axes that produced signal at a larger scale.")
    p.add_argument("--force", action="store_true", help="rerun completed runs")
    p.add_argument("--report", action="store_true", help="aggregate only")
    args = p.parse_args()

    if args.report:
        report(args)
        return

    if args.axes:
        axes = [a.strip() for a in args.axes.split(",") if a.strip()]
        unknown = [a for a in axes if a not in AXES]
        if unknown:
            p.error(f"unknown axes: {', '.join(unknown)}. "
                    f"Valid: {', '.join(AXES)}")
    elif args.axis:
        axes = [args.axis]
    else:
        axes = list(AXES)
    seeds = [1337 + i for i in range(args.seeds)]
    runs = list(enumerate_runs(axes, seeds, args.preset))

    print(f"sweep: {len(runs)} runs "
          f"({len(axes)} axes x {args.seeds} seeds, {args.steps} steps each)")
    print()

    started = time.time()
    failed = []
    for i, (axis, value, seed, is_control) in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}]", end=" ")
        name, ok = run_one(axis, value, seed, is_control, args)
        if not ok:
            failed.append(name)

    print()
    print(f"sweep finished in {(time.time() - started) / 60:.1f} min")
    if failed:
        print(f"{len(failed)} run(s) failed:")
        for name in failed:
            print(f"  {name}")
    print()
    report(args)


if __name__ == "__main__":
    main()
