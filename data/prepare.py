"""
Tokenise FineWeb-Edu into flat uint16 shards for training.

Streams from the Hub rather than downloading the full corpus first -- the
10BT sample is ~20 GB on disk and we usually want a fraction of it. Tokens
land in .bin shards that train.py memory-maps, so the training loop never
holds the dataset in RAM.

uint16 is enough because the GPT-2 vocabulary is 50257 < 65536, and it halves
both disk footprint and the host-to-device copy compared to uint32.

Usage:
    python data/prepare.py --tokens 2e9          # ~2B tokens
    python data/prepare.py --tokens 1e8 --out data/tiny
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

SHARD_TOKENS = 100_000_000        # ~200 MB per shard at uint16
VAL_TOKENS = 10_000_000           # held out from the first shard


def human(n: float) -> str:
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:.1f}{unit}"
        n /= 1000
    return f"{n:.1f}T"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=float, default=2e9,
                        help="approximate total tokens to write")
    parser.add_argument("--out", type=str, default="data/fineweb-edu")
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--subset", type=str, default="sample-10BT")
    args = parser.parse_args()

    # Imported here so --help works without the heavy deps loaded.
    import tiktoken
    from datasets import load_dataset

    target = int(args.tokens)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens["<|endoftext|>"]

    print(f"dataset : {args.dataset} / {args.subset}")
    print(f"target  : {human(target)} tokens -> {out_dir}")

    stream = load_dataset(args.dataset, name=args.subset,
                          split="train", streaming=True)

    buffer = np.empty(SHARD_TOKENS, dtype=np.uint16)
    fill = 0
    shard_idx = 0
    total = 0
    docs = 0
    started = time.time()

    def flush(count: int, is_val: bool = False) -> None:
        nonlocal shard_idx
        name = "val.bin" if is_val else f"train_{shard_idx:04d}.bin"
        path = out_dir / name
        buffer[:count].tofile(path)
        mb = count * 2 / 1e6
        print(f"  wrote {name:18s} {human(count):>8s} tokens  ({mb:.0f} MB)")
        if not is_val:
            shard_idx += 1

    val_written = False

    for doc in stream:
        # Each document is terminated so the model learns document boundaries.
        ids = enc.encode_ordinary(doc["text"])
        ids.append(eot)
        docs += 1

        pos = 0
        while pos < len(ids):
            space = SHARD_TOKENS - fill
            take = min(space, len(ids) - pos)
            buffer[fill:fill + take] = ids[pos:pos + take]
            fill += take
            pos += take
            total += take

            if fill == SHARD_TOKENS:
                if not val_written:
                    # Carve the validation split off the front of shard 0 so it
                    # is never seen during training.
                    buffer[:VAL_TOKENS].tofile(out_dir / "val.bin")
                    print(f"  wrote {'val.bin':18s} {human(VAL_TOKENS):>8s} tokens")
                    remainder = SHARD_TOKENS - VAL_TOKENS
                    buffer[:remainder] = buffer[VAL_TOKENS:]
                    fill = remainder
                    val_written = True
                    continue
                flush(SHARD_TOKENS)
                fill = 0

        if total >= target:
            break

        if docs % 10_000 == 0:
            rate = total / max(time.time() - started, 1e-9)
            pct = 100 * total / target
            print(f"  {human(total):>8s} tokens  {pct:5.1f}%  "
                  f"{human(rate)}/s  {docs:,} docs", flush=True)

    if fill > 0:
        flush(fill)

    elapsed = time.time() - started
    print()
    print(f"done: {human(total)} tokens from {docs:,} documents "
          f"in {elapsed / 60:.1f} min")
    print(f"files: {len(list(out_dir.glob('*.bin')))} in {out_dir}")

    total_bytes = sum(f.stat().st_size for f in out_dir.glob("*.bin"))
    print(f"size : {total_bytes / 1e9:.1f} GB")

    # Breaking out of a streaming dataset leaves HF's prefetch threads alive.
    # At interpreter shutdown they try to release a GIL state belonging to a
    # dead thread, which aborts with PyGILState_Release *after* all work is
    # done. Every .bin is already flushed by this point, so skip finalisation
    # entirely rather than exit nonzero on a successful run.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
