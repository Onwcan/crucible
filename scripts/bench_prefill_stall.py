"""
Does admitting a long prompt freeze the streams already decoding?

The question this answers is not "how fast is the server" but "what does an
existing client feel when somebody else shows up". So it times the *gaps*
between tokens of a stream that is already running, while a second request with
a long prompt arrives partway through.

    A: already decoding, timestamps recorded for every token
    B: arrives at T with a prompt of N tokens

If prefill is monolithic and runs on the inference thread, A's token stream
shows one gap roughly the size of B's whole prefill. If prefill is chunked, the
same work appears as several smaller gaps instead. Both cost A the same total
time; only one of them is a stall.

Reported per run:

    A's median gap          what A normally feels
    A's worst gap           the stall, if there is one
    A's p95 / p99 gap       how much of the stream the disturbance touched
    B's TTFT                what B paid to get in
    stall ratio             worst gap / median gap

Also runs a burst scenario -- several streams established, then one long prompt
-- and an all-at-once scenario, because a scheduler can fix the two-request case
and still serialise a queue.

Standard library only. Usage, against an already-running server:
    python scripts/bench_prefill_stall.py --port 8080
"""
from __future__ import annotations

import argparse
import http.client
import json
import statistics
import subprocess
import threading
import time

# Long enough that A is still decoding through the whole disturbance window:
# a stream that has already finished cannot report a stall. Bounded by the
# model context, so the server needs --max-new-tokens at least this high.
A_TOKENS = 900
# Roughly a word per token for GPT-2 BPE on ordinary prose.
FILLER = ("the quick brown fox jumps over the lazy dog while a distant engine "
          "hums and the afternoon light moves across the floor ")


def envelope() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=name,enforced.power.limit,clocks.max.sm",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "unavailable"


def conn(args):
    return http.client.HTTPConnection(args.host, args.port, timeout=600)


def prompt_of(tokens: int) -> str:
    """A prompt of approximately `tokens` GPT-2 tokens."""
    words = FILLER.split()
    return " ".join((words * (tokens // len(words) + 2))[:tokens])


def measure_tokens(args, prompt: str, max_tokens: int) -> tuple[list[float], float]:
    """Stream one request, returning (token arrival times, start time)."""
    c = conn(args)
    start = time.perf_counter()
    c.request("POST", "/v1/generate/stream",
              json.dumps({"prompt": prompt, "max_tokens": max_tokens}),
              {"Content-Type": "application/json"})
    r = c.getresponse()
    stamps = []
    for raw in r:
        if raw.startswith(b"data:") and b"token_id" in raw:
            stamps.append(time.perf_counter())
    c.close()
    return stamps, start


def exact_prompt_tokens(args, prompt: str) -> int:
    """Actual token count, or 0 if the server refuses the prompt as too long."""
    c = conn(args)
    c.request("POST", "/v1/generate", json.dumps({"prompt": prompt, "max_tokens": 1}),
              {"Content-Type": "application/json"})
    body = json.loads(c.getresponse().read())
    c.close()
    return body.get("prompt_tokens", 0)


def gaps(stamps: list[float]) -> list[float]:
    return [(b - a) * 1000 for a, b in zip(stamps, stamps[1:])]


def gaps_in(stamps: list[float], lo: float, hi: float) -> list[float]:
    """Gaps whose *end* falls inside [lo, hi)."""
    return [(b - a) * 1000 for a, b in zip(stamps, stamps[1:]) if lo <= b < hi]


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * p))]


# How long the existing streams get to settle before the intruder arrives, and
# how wide a window after its arrival counts as "the disturbance".
SETTLE_S = 0.25
WINDOW_S = 0.9


def interrupt_run(args, n_existing: int, intruder_tokens: int) -> dict:
    """`n_existing` streams decoding, then one long prompt arrives."""
    results: dict = {}
    barrier = threading.Barrier(n_existing + 1)

    def existing(i: int):
        barrier.wait()
        stamps, start = measure_tokens(args, "The capital of France is", A_TOKENS)
        results[i] = (stamps, start)

    threads = [threading.Thread(target=existing, args=(i,)) for i in range(n_existing)]
    for t in threads:
        t.start()
    barrier.wait()
    settle_end = time.perf_counter() + SETTLE_S
    time.sleep(SETTLE_S)

    b_prompt = prompt_of(intruder_tokens)
    b_start = time.perf_counter()
    b_stamps, _ = measure_tokens(args, b_prompt, 8)
    b_ttft = (b_stamps[0] - b_start) * 1000 if b_stamps else float("nan")

    for t in threads:
        t.join()

    # Baseline: gaps after the streams settled but before the intruder was
    # sent. Disturbance: gaps in the window that follows its arrival.
    baseline, during = [], []
    for i in results:
        stamps = results[i][0]
        baseline += gaps_in(stamps, settle_end - SETTLE_S / 2, b_start)
        during += gaps_in(stamps, b_start, b_start + WINDOW_S)
    med = statistics.median(baseline) if baseline else 0.0
    worst = max(during) if during else 0.0
    return {
        "median": med,
        "p95": pct(during, 0.95),
        "p99": pct(during, 0.99),
        "worst": worst,
        "ratio": worst / med if med else 0.0,
        "b_ttft": b_ttft,
        "samples": len(during),
    }


def burst_run(args, n: int, sizes: list[int]) -> dict:
    """`n` requests arriving at once with mixed prompt sizes."""
    results: dict = {}
    barrier = threading.Barrier(n)

    def one(i: int):
        p = prompt_of(sizes[i % len(sizes)])
        barrier.wait()
        start = time.perf_counter()
        stamps, _ = measure_tokens(args, p, 64)
        results[i] = ((stamps[0] - start) * 1000 if stamps else float("nan"),
                      gaps(stamps))

    threads = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    ttfts = [v[0] for v in results.values() if v[0] == v[0]]  # drop NaN
    failed = n - len(ttfts)
    all_gaps = [g for v in results.values() for g in v[1]]
    if not ttfts:
        return {"ttft_median": float("nan"), "ttft_p95": float("nan"),
                "ttft_max": float("nan"), "gap_median": 0.0, "gap_p95": 0.0,
                "aggregate": 0.0, "failed": failed}
    return {
        "failed": failed,
        "ttft_median": statistics.median(ttfts),
        "ttft_p95": pct(ttfts, 0.95),
        "ttft_max": max(ttfts),
        "gap_median": statistics.median(all_gaps) if all_gaps else 0.0,
        "gap_p95": pct(all_gaps, 0.95),
        "aggregate": sum(len(v[1]) + 1 for v in results.values()) / wall,
    }


def main() -> None:
    global A_TOKENS
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--intruders", default="32,128,512,1024")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--label", default="")
    p.add_argument("--a-tokens", type=int, default=A_TOKENS,
                   help="tokens the established streams generate")
    args = p.parse_args()

    A_TOKENS = args.a_tokens
    sizes = [int(v) for v in args.intruders.split(",") if v.strip()]
    print(f"gpu        {envelope()}")
    if args.label:
        print(f"config     {args.label}")
    print(f"workload   {A_TOKENS}-token streams, median of {args.trials}")
    print()
    print("prompt sizing check (requested vs actual tokens)")
    usable = []
    for n in sizes:
        actual = exact_prompt_tokens(args, prompt_of(n))
        if actual:
            usable.append(n)
            print(f"  {n:>5} requested -> {actual:>5} actual")
        else:
            print(f"  {n:>5} requested -> refused (over the server prompt limit); skipped")
    sizes = usable
    print()

    for n_existing in (1, 4):
        head = (f"{'existing':>9} {'intruder':>9} {'A median':>10} {'A p95':>8} "
                f"{'A p99':>8} {'A worst':>9} {'stall':>7} {'B TTFT':>9}")
        print(head)
        print("-" * len(head))
        for n in sizes:
            runs = [interrupt_run(args, n_existing, n) for _ in range(args.trials)]
            pick = sorted(runs, key=lambda r: r["worst"])[len(runs) // 2]
            if pick["samples"] == 0:
                print(f"{n_existing:>9} {n:>9}   no gaps in the window: the streams "
                      f"finished before the intruder arrived (raise --a-tokens)")
                continue
            print(f"{n_existing:>9} {n:>9} {pick['median']:>9.2f}ms {pick['p95']:>7.2f}ms "
                  f"{pick['p99']:>7.2f}ms {pick['worst']:>8.2f}ms "
                  f"{pick['ratio']:>6.1f}x {pick['b_ttft']:>8.1f}ms")
        print()

    print("burst: 16 requests arriving together, mixed prompt sizes")
    head = (f"{'n':>4} {'TTFT med':>10} {'TTFT p95':>10} {'TTFT max':>10} "
            f"{'gap med':>9} {'gap p95':>9} {'agg tok/s':>10}")
    print(head)
    print("-" * len(head))
    for n in (4, 16):
        runs = [burst_run(args, n, [8, 32, 128, 512]) for _ in range(args.trials)]
        pick = sorted(runs, key=lambda r: r["ttft_p95"])[len(runs) // 2]
        note = f"  ({pick['failed']} failed)" if pick["failed"] else ""
        print(f"{n:>4} {pick['ttft_median']:>9.1f}ms {pick['ttft_p95']:>9.1f}ms "
              f"{pick['ttft_max']:>9.1f}ms {pick['gap_median']:>8.2f}ms "
              f"{pick['gap_p95']:>8.2f}ms {pick['aggregate']:>9.0f}{note}")
    print()
    print("A median is measured on settled streams *before* the intruder is sent, so")
    print("it excludes their own admission. A worst is the largest gap in the 3 s")
    print("window after the intruder arrives, and stall is that as a multiple of the")
    print("median. Monolithic prefill puts the whole prompt in one gap, so stall")
    print("grows with prompt length; chunked prefill spreads it into smaller ones.")


if __name__ == "__main__":
    main()
