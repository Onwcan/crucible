"""
Load benchmark for the crucible HTTP inference service.

Measures what a client actually experiences -- time to first token, inter-token
latency, total request latency -- rather than what the GPU is doing. Those are
different numbers and conflating them would hide the cost of the service layer.

Uses only the standard library: adding an HTTP client dependency to measure an
HTTP server invites the two to share assumptions.

Concurrency is produced with real overlapping connections, not by sending N
requests one after another. That distinction matters here: the point of the
runtime underneath is continuous batching, and a benchmark that serialises
requests would report the batch-1 path N times and call it throughput.

Usage:
    python scripts/bench_serve.py --host 127.0.0.1 --port 8080 \
        --concurrency 1,2,4,8,16 --max-tokens 64
"""
from __future__ import annotations

import argparse
import http.client
import json
import statistics
import subprocess
import threading
import time

# Heterogeneous by construction: identical prompts would give every request the
# same sequence length and hide how the scheduler handles a mixed batch.
PROMPTS = [
    "The capital of France is",
    "In a distant galaxy, a small crew of engineers discovered that",
    "Write a short explanation of how a transformer language model works, "
    "starting from the attention mechanism and",
    "def fibonacci(n):",
    "The history of the printing press begins in",
    "Q: Why is the sky blue?\nA:",
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


class Result:
    __slots__ = ("ttft", "total", "gaps", "tokens", "reason", "error", "wall")

    def __init__(self):
        self.ttft = None
        self.total = None
        self.gaps: list[float] = []
        self.tokens = 0
        self.reason = None
        self.error = None
        self.wall = 0.0


def stream_one(host: str, port: int, prompt: str, max_tokens: int,
               barrier: threading.Barrier, out: Result) -> None:
    """One streaming request, recording per-token arrival times."""
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens})
    conn = http.client.HTTPConnection(host, port, timeout=120)
    try:
        # Every worker starts together, so the requests genuinely overlap.
        barrier.wait()
        start = time.perf_counter()
        conn.request("POST", "/v1/generate/stream", body,
                     {"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            out.error = f"HTTP {resp.status}: {resp.read()[:200]!r}"
            return

        last = start
        event = None
        for raw in resp:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                payload = json.loads(line[5:].strip())
                now = time.perf_counter()
                if event == "token":
                    if out.ttft is None:
                        out.ttft = now - start
                    else:
                        out.gaps.append(now - last)
                    last = now
                    out.tokens += 1
                elif event == "done":
                    out.reason = payload.get("finish_reason")
                    out.total = now - start
                    return
                elif event == "error":
                    out.error = payload.get("error", "unknown")
                    return
    except Exception as exc:  # noqa: BLE001
        out.error = f"{type(exc).__name__}: {exc}"
    finally:
        conn.close()


def run(host: str, port: int, n: int, max_tokens: int) -> list[Result]:
    barrier = threading.Barrier(n)
    results = [Result() for _ in range(n)]
    threads = [
        threading.Thread(
            target=stream_one,
            args=(host, port, PROMPTS[i % len(PROMPTS)], max_tokens,
                  barrier, results[i]),
        )
        for i in range(n)
    ]
    wall = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for r in results:
        r.wall = time.perf_counter() - wall
    return results


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = min(len(s) - 1, int(round(p * (len(s) - 1))))
    return s[k]


def metrics(host: str, port: int) -> dict:
    try:
        conn = http.client.HTTPConnection(host, port, timeout=5)
        conn.request("GET", "/metrics")
        return json.loads(conn.getresponse().read())
    except Exception:
        return {}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--concurrency", default="1,2,4,8,16")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--trials", type=int, default=3)
    args = p.parse_args()

    print(f"gpu         {envelope()}")
    print(f"workload    {args.max_tokens} tokens, heterogeneous prompts, "
          f"{args.trials} trials")
    print()

    header = (f"{'conc':>5} {'e2e t/s':>9} {'steady t/s':>11} {'per-req':>8} "
              f"{'TTFT med':>9} {'TTFT p95':>9} {'gap med':>8} {'gap p95':>8} "
              f"{'total med':>10} {'batch':>6}")
    print(header)
    print("-" * len(header))

    for n in [int(v) for v in args.concurrency.split(",")]:
        best = None
        for _ in range(args.trials):
            m_before = metrics(args.host, args.port)
            results = run(args.host, args.port, n, args.max_tokens)
            m_after = metrics(args.host, args.port)
            failed = [r for r in results if r.error]
            if failed:
                print(f"{n:>5}  failed: {failed[0].error}")
                best = None
                break
            wall = max(r.wall for r in results)
            total_tokens = sum(r.tokens for r in results)
            agg = total_tokens / wall
            # Mean tokens per decode step over THIS run. The server's own
            # average_batch_size is cumulative since startup, so it would
            # report the history of every earlier run instead.
            d_steps = (m_after.get("decode_steps", 0)
                       - m_before.get("decode_steps", 0))
            d_toks = (m_after.get("aggregate_tokens_generated", 0)
                      - m_before.get("aggregate_tokens_generated", 0))
            batch = d_toks / d_steps if d_steps else 0.0
            if best is None or agg > best[0]:
                best = (agg, results, wall, batch)
        if best is None:
            continue

        agg, results, wall, batch = best
        ttfts = [r.ttft for r in results if r.ttft is not None]
        gaps = [g for r in results for g in r.gaps]
        totals = [r.total for r in results if r.total is not None]
        # n / median gap: the rate once every request is decoding, excluding
        # prefill, admission ramp-up and drain. This is the number comparable
        # to a decode-only runtime benchmark; `agg` is what a client sees.
        steady = n / statistics.median(gaps) if gaps else float("nan")
        print(f"{n:>5} {agg:>9.0f} {steady:>11.0f} {agg / n:>8.0f} "
              f"{statistics.median(ttfts) * 1e3:>8.1f}m "
              f"{pct(ttfts, 0.95) * 1e3:>8.1f}m "
              f"{statistics.median(gaps) * 1e3:>7.2f}m "
              f"{pct(gaps, 0.95) * 1e3:>7.2f}m "
              f"{statistics.median(totals) * 1e3:>9.1f}m "
              f"{batch:>6.1f}")

    print()
    print("e2e t/s spans the whole overlapped window and so includes prompt")
    print("prefill, admission ramp-up and drain -- what a client experiences.")
    print("steady t/s is n / median inter-token gap, which excludes those and is")
    print("the number comparable to a decode-only runtime benchmark. Reporting")
    print("only one of them would either flatter the service or hide its real")
    print("cost. batch is mean tokens per decode step over this run alone.")


if __name__ == "__main__":
    main()
