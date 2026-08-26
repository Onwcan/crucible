"""
Turn sweep output into figures and a markdown results table.

Reads every runs/*/log.csv, groups runs by ablation axis, and plots each
variant against the control with a band showing seed-to-seed spread. The band
is the point: if the bands overlap, the architectures are indistinguishable at
this scale, and the plot should make that obvious rather than inviting a
conclusion from two lines that happen not to cross.

CPU only -- safe to run while training occupies the GPU.

Usage:
    python analyze.py                      # all axes
    python analyze.py --axis attention
    python analyze.py --markdown           # table only, no figures
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless: no display under WSL
import matplotlib.pyplot as plt

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

RUN_RE = re.compile(r"^(?P<preset>[^-]+)-(?:(?P<axis>[a-z_]+)=(?P<value>[a-z]+)|control)-s(?P<seed>\d+)$")


class Run:
    def __init__(self, path: Path, axis: str | None, value: str | None, seed: int):
        self.path, self.axis, self.value, self.seed = path, axis, value, seed
        self.steps: list[int] = []
        self.train: list[float] = []
        self.val_steps: list[int] = []
        self.val: list[float] = []
        self.mfu: list[float] = []
        self._load()

    def _load(self) -> None:
        with open(self.path / "log.csv", newline="") as f:
            for row in csv.DictReader(f):
                step = int(row["step"])
                self.steps.append(step)
                self.train.append(float(row["train_loss"]))
                if row.get("mfu"):
                    self.mfu.append(float(row["mfu"]))
                if row.get("val_loss"):
                    self.val_steps.append(step)
                    self.val.append(float(row["val_loss"]))

    @property
    def best_val(self) -> float | None:
        return min(self.val) if self.val else None

    @property
    def median_mfu(self) -> float:
        # Skip step 0: it carries the torch.compile warmup.
        tail = self.mfu[1:] or self.mfu
        return statistics.median(tail) if tail else 0.0


def discover(runs_dir: Path) -> tuple[list[Run], list[Run]]:
    """Return (control_runs, variant_runs)."""
    controls, variants = [], []
    for d in sorted(runs_dir.iterdir()):
        if not (d / "log.csv").exists():
            continue
        m = RUN_RE.match(d.name)
        if not m:
            continue                      # ad-hoc run, not part of the sweep
        seed = int(m.group("seed"))
        axis, value = m.group("axis"), m.group("value")
        run = Run(d, axis, value, seed)
        (controls if axis is None else variants).append(run)
    return controls, variants


def band(runs: list[Run], key: str):
    """Mean curve plus min/max envelope across seeds, aligned on step."""
    by_step: dict[int, list[float]] = defaultdict(list)
    for r in runs:
        steps = r.val_steps if key == "val" else r.steps
        values = r.val if key == "val" else r.train
        for s, v in zip(steps, values):
            by_step[s].append(v)
    steps = sorted(by_step)
    mean = [statistics.mean(by_step[s]) for s in steps]
    lo = [min(by_step[s]) for s in steps]
    hi = [max(by_step[s]) for s in steps]
    return steps, mean, lo, hi


def plot_axis(axis: str, controls: list[Run], variants: list[Run],
              out_dir: Path) -> Path | None:
    groups: dict[str, list[Run]] = defaultdict(list)
    for r in variants:
        if r.axis == axis:
            groups[r.value].append(r)
    if not groups:
        return None

    fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(13, 5))

    series = [(f"{CONTROL[axis]} (control)", controls)] + \
             [(v, rs) for v, rs in sorted(groups.items())]

    for label, runs in series:
        if not runs:
            continue
        for ax, key in ((ax_train, "train"), (ax_val, "val")):
            steps, mean, lo, hi = band(runs, key)
            if not steps:
                continue
            line, = ax.plot(steps, mean, label=f"{label} (n={len(runs)})", lw=1.6)
            ax.fill_between(steps, lo, hi, alpha=0.18, color=line.get_color())

    ax_train.set_title(f"{axis} - training loss")
    ax_val.set_title(f"{axis} - validation loss")
    for ax in (ax_train, ax_val):
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(f"Ablation: {axis}   (shaded band = min/max across seeds)",
                 fontsize=11)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ablation_{axis}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# A run this far above the control has not "done slightly worse" -- it has
# failed to train. Averaging it in corrupts both the mean and the standard
# deviation, and an inflated SD then swallows the very difference it should
# expose, so divergence is classified before any statistics are computed.
DIVERGENCE_MARGIN = 1.0


def verdict_for(control_vals: list[float], variant_vals: list[float]) -> dict:
    """
    Compare a variant against the control, separating three distinct outcomes:
    instability, a real difference, and noise.

    Returns mean/sd over the *converged* seeds only, plus how many diverged.
    """
    ctrl_mean = statistics.mean(control_vals)
    ctrl_sd = statistics.stdev(control_vals) if len(control_vals) > 1 else 0.0

    converged = [v for v in variant_vals if v <= ctrl_mean + DIVERGENCE_MARGIN]
    diverged = len(variant_vals) - len(converged)

    if not converged:
        return {"mean": statistics.mean(variant_vals), "sd": 0.0,
                "delta": float("nan"), "diverged": diverged,
                "n": len(variant_vals),
                "verdict": f"FAILED ({diverged}/{len(variant_vals)} diverged)"}

    mean = statistics.mean(converged)
    sd = statistics.stdev(converged) if len(converged) > 1 else 0.0
    delta = mean - ctrl_mean
    noise = max(ctrl_sd, sd)

    if diverged:
        # Instability dominates: even if the surviving seeds look acceptable,
        # a config that fails on some seeds is not a config you would pick.
        verdict = f"UNSTABLE ({diverged}/{len(variant_vals)} diverged)"
    elif noise == 0.0:
        verdict = "unknown (1 seed)"
    elif abs(delta) < 2 * noise:
        verdict = "within noise"
    elif delta < 0:
        verdict = "better"
    else:
        verdict = "worse"

    return {"mean": mean, "sd": sd, "delta": delta,
            "diverged": diverged, "n": len(variant_vals), "verdict": verdict}


def markdown_table(controls: list[Run], variants: list[Run]) -> str:
    ctrl_vals = [r.best_val for r in controls if r.best_val is not None]
    if not ctrl_vals:
        return "_No control runs found._"

    ctrl_mean = statistics.mean(ctrl_vals)
    ctrl_sd = statistics.stdev(ctrl_vals) if len(ctrl_vals) > 1 else 0.0
    ctrl_mfu = statistics.median([r.median_mfu for r in controls])

    lines = [
        f"Control (`{CONTROL['attention']}/{CONTROL['pos_encoding']}/"
        f"{CONTROL['activation']}/{CONTROL['norm_placement']}-{CONTROL['norm']}`): "
        f"**{ctrl_mean:.4f}** ± {ctrl_sd:.4f} over {len(ctrl_vals)} seeds, "
        f"{ctrl_mfu:.1f}% MFU",
        "",
        "| Axis | Variant | Val loss | SD | Δ vs control | MFU | Verdict |",
        "|---|---|---:|---:|---:|---:|---|",
    ]

    by_key: dict[tuple[str, str], list[Run]] = defaultdict(list)
    for r in variants:
        by_key[(r.axis, r.value)].append(r)

    for axis in AXES:
        for value in AXES[axis]:
            if value == CONTROL[axis]:
                continue
            runs = by_key.get((axis, value), [])
            vals = [r.best_val for r in runs if r.best_val is not None]
            if not vals:
                continue
            stats = verdict_for(ctrl_vals, vals)
            mfu = statistics.median([r.median_mfu for r in runs])

            verdict = stats["verdict"]
            if verdict == "better":
                verdict = "**better**"
            note = f" ({stats['n'] - stats['diverged']}/{stats['n']})" \
                if stats["diverged"] else ""

            lines.append(f"| `{axis}` | `{value}` | {stats['mean']:.4f}{note} | "
                         f"{stats['sd']:.4f} | {stats['delta']:+.4f} | "
                         f"{mfu:.1f}% | {verdict} |")

    lines += [
        "",
        f"**Verdict rule.** A seed whose best validation loss exceeds the control "
        f"by more than {DIVERGENCE_MARGIN:.1f} is classified as *diverged* and "
        f"excluded from the mean and SD — averaging a failed run in would inflate "
        f"the SD, and the inflated SD would then mask the very difference it "
        f"should reveal. Any variant with a diverged seed is reported as "
        f"`UNSTABLE` regardless of how the surviving seeds performed.",
        "",
        f"For the remainder, |Δ| must exceed 2× the larger seed standard deviation "
        f"(2σ = {2 * ctrl_sd:.4f} for the control) to count as a difference. "
        f"Anything inside that band is `within noise`, not a finding.",
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="runs/figures")
    p.add_argument("--axis", default=None, choices=list(AXES))
    p.add_argument("--markdown", action="store_true",
                   help="print the table only, skip figures")
    args = p.parse_args()

    runs_dir = Path(args.runs)
    if not runs_dir.exists():
        print(f"no such directory: {runs_dir}")
        return

    controls, variants = discover(runs_dir)
    print(f"found {len(controls)} control runs, {len(variants)} variant runs")

    if not controls and not variants:
        print("nothing to analyse yet -- the sweep may still be on its first runs")
        return

    if not args.markdown:
        axes = [args.axis] if args.axis else list(AXES)
        for axis in axes:
            path = plot_axis(axis, controls, variants, Path(args.out))
            if path:
                print(f"  wrote {path}")
        print()

    table = markdown_table(controls, variants)
    print(table)

    out_md = runs_dir / "RESULTS.md"
    out_md.write_text("# Ablation results\n\n" + table + "\n", encoding="utf-8")
    print()
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
