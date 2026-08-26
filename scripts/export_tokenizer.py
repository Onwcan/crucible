"""
Export the GPT-2 BPE vocabulary for the Rust engine.

The training pipeline tokenises with tiktoken's `gpt2` encoding, so the engine
must use byte-for-byte the same vocabulary and merge ranks -- a tokenizer that
differs even slightly produces ids the model was never trained on, and the
output degrades in a way that looks like a bad model rather than a bad
tokenizer.

Format is deliberately trivial so the Rust side needs no base64 or JSON
dependency:

    u32   count
    for each token, in rank order (rank == index):
        u32   byte length
        bytes token

Usage:
    python scripts/export_tokenizer.py --out export/gpt2.tok
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="export/gpt2.tok")
    p.add_argument("--encoding", default="gpt2")
    args = p.parse_args()

    import tiktoken

    enc = tiktoken.get_encoding(args.encoding)
    ranks: dict[bytes, int] = enc._mergeable_ranks

    # Rank must be a dense 0..n-1 range for the file format to use index as id.
    n = len(ranks)
    by_rank: list[bytes | None] = [None] * n
    for token, rank in ranks.items():
        if rank >= n:
            raise SystemExit(f"rank {rank} exceeds vocabulary size {n}")
        by_rank[rank] = token
    missing = [i for i, t in enumerate(by_rank) if t is None]
    if missing:
        raise SystemExit(f"ranks are not dense; {len(missing)} gaps, first {missing[:5]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "wb") as f:
        f.write(struct.pack("<I", n))
        for token in by_rank:
            f.write(struct.pack("<I", len(token)))
            f.write(token)

    # Round-trip a sample through tiktoken so the exported ids are known to be
    # the ones the model was trained on.
    probe = "The World is a stage, and 42 tokens walk onto it.\n"
    ids = enc.encode_ordinary(probe)
    assert enc.decode(ids) == probe, "tiktoken round-trip failed"

    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB, {n} tokens)")
    print(f"eot id: {enc._special_tokens['<|endoftext|>']}")
    print()
    print(f"probe   : {probe!r}")
    print(f"ids     : {ids}")
    print(f"decoded : {enc.decode(ids)!r}")
    print()
    print("use these ids to verify the Rust tokenizer matches")


if __name__ == "__main__":
    main()
