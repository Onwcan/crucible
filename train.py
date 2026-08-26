"""
Training loop for the GPT presets.

BF16 autocast, gradient accumulation, cosine schedule with warmup, and a
per-step MFU figure measured against the hardware's actual BF16 throughput
rather than a datasheet number -- so a bad data pipeline or an unfused kernel
shows up immediately as low MFU instead of hiding as "training is slow".

Every run appends to a CSV so ablations can be plotted and compared later
without re-running anything.

Usage:
    python train.py --preset 30m --data data/tiny --steps 2000
    python train.py --preset 30m --attention mqa --run-name mqa-ablation
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import queue
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from model import PRESETS, GPT, GPTConfig

# Measured on this machine with scripts/bench.py (4096^3 BF16 matmul).
# MFU is reported against real achievable throughput, not spec sheet peak.
PEAK_BF16_TFLOPS = 76.4


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class ShardLoader:
    """
    Samples random windows from memory-mapped uint16 shards.

    np.memmap keeps the OS page cache in charge, so a dataset larger than RAM
    costs nothing extra and startup is instant regardless of corpus size.

    Two things keep the GPU fed:

    1. Batches are gathered with one vectorised fancy-index rather than a
       Python loop over the batch, and the (block_size + 1) window is read once
       and sliced into inputs/targets instead of being read twice.
    2. A background thread fills a bounded queue, so host-side gather and
       pinning overlap with GPU compute instead of stalling it.
    """

    def __init__(self, data_dir: str, split: str, batch_size: int,
                 block_size: int, device: str, prefetch: int = 4):
        path = Path(data_dir)
        if split == "val":
            files = [path / "val.bin"]
        else:
            files = sorted(path.glob("train_*.bin"))
        if not files or not all(f.exists() for f in files):
            raise FileNotFoundError(
                f"no {split} shards in {path} -- run data/prepare.py first")

        self.shards = [np.memmap(f, dtype=np.uint16, mode="r") for f in files]
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.total_tokens = sum(len(s) for s in self.shards)

        # Reused across batches: window offsets never change.
        self._offsets = np.arange(block_size + 1, dtype=np.int64)

        self._queue: queue.Queue = queue.Queue(maxsize=prefetch)
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._fill, daemon=True)
        self._worker.start()

    def _make_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        shard = self.shards[np.random.randint(len(self.shards))]
        starts = np.random.randint(0, len(shard) - self.block_size - 1,
                                   size=self.batch_size)
        # One gather of shape (B, block_size + 1); x and y are offset views.
        windows = shard[starts[:, None] + self._offsets[None, :]].astype(np.int64)
        x = torch.from_numpy(windows[:, :-1]).pin_memory()
        y = torch.from_numpy(windows[:, 1:]).pin_memory()
        return x, y

    def _fill(self) -> None:
        while not self._stop.is_set():
            try:
                self._queue.put(self._make_batch(), timeout=1.0)
            except queue.Full:
                continue
            except Exception:
                self._stop.set()
                raise

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self._queue.get()
        # non_blocking works because the tensors are pinned in the worker.
        return (x.to(self.device, non_blocking=True),
                y.to(self.device, non_blocking=True))

    def close(self) -> None:
        self._stop.set()


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

def lr_at(step: int, warmup: int, total: int, lr: float, min_lr: float) -> float:
    if step < warmup:
        return lr * (step + 1) / (warmup + 1)
    if step >= total:
        return min_lr
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


def make_optimizer(model: GPT, lr: float, weight_decay: float, betas):
    """Decay 2D weights only -- norms, biases and embeddings stay undecayed."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)

    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    print(f"  decayed tensors    : {len(decay)} ({sum(p.numel() for p in decay)/1e6:.1f}M)")
    print(f"  undecayed tensors  : {len(no_decay)} ({sum(p.numel() for p in no_decay)/1e6:.1f}M)")
    return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=True)


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader: ShardLoader, iters: int) -> float:
    model.eval()
    losses = torch.zeros(iters)
    for i in range(iters):
        x, y = loader.get_batch()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="30m", choices=list(PRESETS))
    p.add_argument("--data", default="data/tiny")
    p.add_argument("--out", default="runs")
    p.add_argument("--run-name", default=None)

    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--min-lr", type=float, default=6e-5)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-iters", type=int, default=50)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--compile", action="store_true", default=True)
    p.add_argument("--no-compile", dest="compile", action="store_false")

    # Ablation overrides -- anything not passed keeps the preset default.
    p.add_argument("--attention", default=None)
    p.add_argument("--pos-encoding", default=None)
    p.add_argument("--activation", default=None)
    p.add_argument("--norm", default=None)
    p.add_argument("--norm-placement", default=None)

    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"

    # --- config ------------------------------------------------------------
    overrides = {k: v for k, v in {
        "attention": args.attention,
        "pos_encoding": args.pos_encoding,
        "activation": args.activation,
        "norm": args.norm,
        "norm_placement": args.norm_placement,
    }.items() if v is not None}
    cfg = replace(PRESETS[args.preset], **overrides)
    # replace() bypasses __post_init__ resolution, so redo it.
    cfg = GPTConfig(**asdict(cfg)) if overrides else cfg

    run_name = args.run_name or (
        f"{args.preset}-{cfg.attention}-{cfg.pos_encoding}-"
        f"{cfg.activation}-{cfg.norm_placement}{cfg.norm}-s{args.seed}")
    run_dir = Path(args.out) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    tokens_per_step = args.batch_size * args.grad_accum * cfg.block_size

    print(f"run     : {run_name}")
    print(f"config  : {cfg.attention}/{cfg.pos_encoding}/{cfg.activation}/"
          f"{cfg.norm_placement}-{cfg.norm}")

    # --- model -------------------------------------------------------------
    model = GPT(cfg).to(device)
    # Capture before torch.compile wraps the module -- afterwards this would
    # mean reaching through the wrapper or rebuilding the model just to count.
    n_params = model.num_params(non_embedding=True)
    print(f"params  : {model.num_params(False)/1e6:.1f}M total, "
          f"{n_params/1e6:.1f}M non-embedding")
    print(f"tokens  : {tokens_per_step:,}/step, "
          f"{tokens_per_step * args.steps / 1e9:.2f}B total")

    optimizer = make_optimizer(model, args.lr, args.weight_decay, (0.9, 0.95))

    if args.compile:
        print("  compiling (first step will be slow)...")
        model = torch.compile(model)

    train_loader = ShardLoader(args.data, "train", args.batch_size,
                               cfg.block_size, device)
    val_loader = ShardLoader(args.data, "val", args.batch_size,
                             cfg.block_size, device)
    print(f"data    : {train_loader.total_tokens/1e6:.0f}M train, "
          f"{val_loader.total_tokens/1e6:.0f}M val tokens")

    # FLOPs per token: 6*N for forward+backward, plus the attention term.
    flops_per_token = (6 * n_params
                       + 12 * cfg.n_layer * cfg.n_head * cfg.head_dim * cfg.block_size)

    csv_path = run_dir / "log.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["step", "train_loss", "val_loss", "lr", "dt_ms", "mfu", "tokens"])

    # --- loop --------------------------------------------------------------
    print()
    print(f"{'step':>6s} {'loss':>7s} {'val':>7s} {'lr':>9s} "
          f"{'ms':>7s} {'mfu':>6s} {'tok/s':>9s}")

    best_val = float("inf")
    t0 = time.time()

    for step in range(args.steps + 1):
        lr = lr_at(step, args.warmup, args.steps, args.lr, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if step % args.eval_every == 0 or step == args.steps:
            val_loss = evaluate(model, val_loader, args.eval_iters)
            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    "model": getattr(model, "_orig_mod", model).state_dict(),
                    "config": asdict(cfg),
                    "step": step,
                    "val_loss": val_loss,
                }, run_dir / "best.pt")
        else:
            val_loss = float("nan")

        step_start = time.time()
        optimizer.zero_grad(set_to_none=True)

        for micro in range(args.grad_accum):
            x, y = train_loader.get_batch()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        torch.cuda.synchronize()

        dt = time.time() - step_start
        train_loss = loss.item() * args.grad_accum
        achieved_tflops = flops_per_token * tokens_per_step / dt / 1e12
        mfu = achieved_tflops / PEAK_BF16_TFLOPS * 100

        if step % args.log_every == 0 or not math.isnan(val_loss):
            val_str = f"{val_loss:7.4f}" if not math.isnan(val_loss) else "      -"
            print(f"{step:6d} {train_loss:7.4f} {val_str} {lr:9.2e} "
                  f"{dt*1000:7.1f} {mfu:5.1f}% {tokens_per_step/dt:9,.0f}")

        writer.writerow([step, f"{train_loss:.6f}",
                         "" if math.isnan(val_loss) else f"{val_loss:.6f}",
                         f"{lr:.3e}", f"{dt*1000:.2f}", f"{mfu:.2f}",
                         tokens_per_step * (step + 1)])
        csv_file.flush()

    csv_file.close()
    train_loader.close()
    val_loader.close()
    elapsed = (time.time() - t0) / 60
    print()
    print(f"done in {elapsed:.1f} min | best val loss {best_val:.4f}")
    print(f"log: {csv_path}")
    print(f"ckpt: {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
