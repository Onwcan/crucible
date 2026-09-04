"""
What the compatibility layers cost.

Each adapter parses a different request shape and emits a different event shape
over the same token stream. This measures whether that costs anything the
runtime can feel, by running the native streaming endpoint and every compatible
one against the same server, with the same prompts and the same generation
config, interleaved.

Four surfaces are compared:

    native      POST /v1/generate/stream
    completions POST /v1/completions      (stream=true)
    chat        POST /v1/chat/completions (stream=true)
    messages    POST /v1/messages         (stream=true)

The two conversation surfaces are reported separately from the other two for a
reason. They serialise messages into a transcript, so their prompt is longer
than the raw text the other two send -- more prefill, more attention, a slower
first token. That is the *template's* cost, not the HTTP layer's, and the two
must not be added together and blamed on compatibility. The prompt token counts
are printed so the difference is visible rather than asserted, and chat and
messages should agree exactly: they share one serializer.

Standard library only, like bench_serve.py: measuring an HTTP server with a
third-party HTTP client invites the two to share assumptions.

Usage:
    python scripts/bench_compat.py --port 8080 --concurrency 1,4,8,16
"""
from __future__ import annotations

import argparse
import http.client
import json
import statistics
import subprocess
import threading
import time

PROMPTS = [
    "The capital of France is",
    "In a distant galaxy, a small crew of engineers discovered that",
    "The history of the printing press begins in",
    "Q: Why is the sky blue?\nA:",
]
MODEL = "crucible-120m"


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
    __slots__ = ("ttft", "total", "tokens", "gaps")

    def __init__(self, ttft, total, tokens, gaps):
        self.ttft, self.total, self.tokens, self.gaps = ttft, total, tokens, gaps


def stream_request(args, surface: str, prompt: str, max_tokens: int) -> Result:
    """One streaming request, timed from the client's side."""
    headers = {"Content-Type": "application/json"}
    if surface == "native":
        path = "/v1/generate/stream"
        body = {"prompt": prompt, "max_tokens": max_tokens}
    elif surface == "completions":
        path = "/v1/completions"
        body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "stream": True}
    elif surface == "chat":
        path = "/v1/chat/completions"
        body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "stream": True}
    else:
        path = "/v1/messages"
        body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "stream": True}
        headers["anthropic-version"] = "2023-06-01"

    c = http.client.HTTPConnection(args.host, args.port, timeout=300)
    start = time.perf_counter()
    c.request("POST", path, json.dumps(body), headers)
    r = c.getresponse()
    ttft = None
    tokens = 0
    gaps = []
    last = start
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        # Count only events that carry generated text, so the surfaces are
        # counted alike: the chat stream opens with a role-only chunk and the
        # Anthropic stream with message_start and content_block_start, which
        # are protocol framing rather than tokens.
        if surface == "native":
            counts = True
        elif surface == "messages":
            obj = json.loads(payload)
            counts = obj.get("type") == "content_block_delta"
        else:
            obj = json.loads(payload)
            ch = obj.get("choices") or []
            if not ch:
                continue
            if surface == "chat":
                counts = ch[0]["delta"].get("role") is None
            else:
                counts = ch[0].get("finish_reason") is None
        if not counts:
            continue
        now = time.perf_counter()
        if ttft is None:
            ttft = now - start
        else:
            gaps.append(now - last)
        last = now
        tokens += 1
    total = time.perf_counter() - start
    c.close()
    return Result(ttft if ttft is not None else total, total, tokens, gaps)


def run(args, surface: str, concurrency: int) -> dict:
    results: list[Result] = []
    lock = threading.Lock()

    def worker(i: int):
        r = stream_request(args, surface, PROMPTS[i % len(PROMPTS)], args.max_tokens)
        with lock:
            results.append(r)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(concurrency)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    tokens = sum(r.tokens for r in results)
    gaps = [g for r in results for g in r.gaps]
    return {
        "aggregate": tokens / wall,
        "ttft_ms": statistics.median(r.ttft for r in results) * 1000,
        "gap_ms": statistics.median(gaps) * 1000 if gaps else 0.0,
        "tokens": tokens,
    }


def prompt_tokens(args, surface: str, prompt: str) -> int:
    """What each surface actually submits, so prefill differences are visible."""
    c = http.client.HTTPConnection(args.host, args.port, timeout=60)
    if surface == "messages":
        # count_tokens exists precisely for this and costs no generation.
        c.request("POST", "/v1/messages/count_tokens",
                  json.dumps({"model": MODEL,
                              "messages": [{"role": "user", "content": prompt}]}),
                  {"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        data = json.loads(c.getresponse().read())
        c.close()
        return data["input_tokens"]
    if surface == "chat":
        body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1}
        path = "/v1/chat/completions"
    else:
        body = {"model": MODEL, "prompt": prompt, "max_tokens": 1}
        path = "/v1/completions"
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    data = json.loads(c.getresponse().read())
    c.close()
    return data["usage"]["prompt_tokens"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--concurrency", default="1,4,8,16")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--trials", type=int, default=3)
    args = p.parse_args()

    levels = [int(v) for v in args.concurrency.split(",") if v.strip()]
    print(f"gpu          {envelope()}")
    print(f"workload     {len(PROMPTS)} prompts cycled, {args.max_tokens} tokens, "
          f"{args.trials} trials, median")
    print()

    print("prompt tokens actually submitted, per surface")
    shared = True
    for prompt in PROMPTS:
        raw = prompt_tokens(args, "completions", prompt)
        chat = prompt_tokens(args, "chat", prompt)
        msgs = prompt_tokens(args, "messages", prompt)
        shared = shared and chat == msgs
        print(f"  {raw:>3} raw -> {chat:>3} chat, {msgs:>3} messages (+{chat - raw})   "
              f"{prompt[:40]!r}")
    print(f"  chat and messages agree on every prompt: {shared}"
          "   (they share one serializer)")
    print()

    header = (f"{'clients':>8}  {'surface':>12}  {'aggregate':>12}  {'TTFT ms':>9}  "
              f"{'gap ms':>8}  {'vs native':>10}")
    print(header)
    print("-" * len(header))

    for n in levels:
        base = None
        for surface in ("native", "completions", "chat", "messages"):
            runs = [run(args, surface, n) for _ in range(args.trials)]
            agg = statistics.median(r["aggregate"] for r in runs)
            ttft = statistics.median(r["ttft_ms"] for r in runs)
            gap = statistics.median(r["gap_ms"] for r in runs)
            if surface == "native":
                base = agg
            ratio = "-" if surface == "native" else f"{agg / base:.3f}x"
            print(f"{n:>8}  {surface:>12}  {agg:>9.0f} t/s  {ttft:>9.1f}  {gap:>8.2f}  "
                  f"{ratio:>10}")
        print()

    print("TTFT includes prefill, so the conversation surfaces carry the cost of")
    print("their longer serialized prompt -- see the token counts above. gap is the")
    print("median inter-token interval once generation is under way, which is where")
    print("per-event serialisation cost would show up if it were material.")


if __name__ == "__main__":
    main()
