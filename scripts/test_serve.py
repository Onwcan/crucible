"""
HTTP-level tests for the crucible inference service.

Separate from the Rust unit tests because these need a running server and a GPU;
the pure pieces (request validation, incremental UTF-8 decoding) are tested
in-process by `cargo test` instead.

The concurrency section is the load-bearing one. Four requests completing is not
evidence of continuous batching -- they could have run one after another. So it
samples `last_batch_size` from /metrics while requests overlap, and separately
checks that a request batched with others produces exactly the token ids it
produces alone.

Usage, against an already-running server:
    python scripts/test_serve.py --port 8080
"""
from __future__ import annotations

import argparse
import http.client
import json
import threading
import time

FAILED: list[str] = []
PASSED = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSED
    if cond:
        PASSED += 1
        print(f"  ok    {name}")
    else:
        FAILED.append(name)
        print(f"  FAIL  {name}  {detail}")


def conn(args) -> http.client.HTTPConnection:
    return http.client.HTTPConnection(args.host, args.port, timeout=120)


def post(args, path: str, body, raw: bool = False):
    c = conn(args)
    payload = body if raw else json.dumps(body)
    c.request("POST", path, payload, {"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data


def get(args, path: str):
    c = conn(args)
    c.request("GET", path)
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data


def stream(args, prompt: str, max_tokens: int, stop_after=None, **extra):
    """Consume an SSE stream. stop_after closes the connection early."""
    c = conn(args)
    c.request("POST", "/v1/generate/stream",
              json.dumps({"prompt": prompt, "max_tokens": max_tokens, **extra}),
              {"Content-Type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        c.close()
        return r.status, [], None, None
    tokens, done = [], None
    ctype = r.getheader("Content-Type")
    event = None
    try:
        for raw in r:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                payload = json.loads(line[5:].strip())
                if event == "token":
                    tokens.append(payload)
                    if stop_after is not None and len(tokens) >= stop_after:
                        return 200, tokens, None, ctype
                elif event == "done":
                    done = payload
                    break
    finally:
        c.close()
    return 200, tokens, done, ctype


def metrics(args) -> dict:
    _, body = get(args, "/metrics")
    return json.loads(body)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()

    print("health and metrics")
    status, body = get(args, "/health")
    h = json.loads(body)
    check("health returns 200", status == 200, str(status))
    check("health reports model/device/max_batch",
          all(k in h for k in ("model", "device", "max_batch", "context")), str(h))
    cap = h.get("sampling")
    check("health advertises sampling capabilities",
          isinstance(cap, dict) and cap.get("greedy") and cap.get("temperature")
          and cap.get("top_k") and cap.get("seed"), str(cap))
    check("health says the default mode is greedy",
          isinstance(cap, dict) and cap.get("default_mode") == "greedy", str(cap))
    m0 = metrics(args)
    check("metrics exposes required fields",
          all(k in m0 for k in ("active_requests", "queued_requests",
                                "completed_requests", "kv_pages_used",
                                "kv_pages_free", "last_batch_size",
                                "aggregate_tokens_generated")), str(m0))
    total_pages = m0["kv_pages_used"] + m0["kv_pages_free"]

    print("\ninput validation")
    status, body = post(args, "/v1/generate", "{not json", raw=True)
    check("malformed JSON is 4xx", 400 <= status < 500, f"status {status}")
    status, _ = post(args, "/v1/generate", {"prompt": "", "max_tokens": 8})
    check("empty prompt is 400", status == 400, f"status {status}")
    status, _ = post(args, "/v1/generate", {"prompt": "hi", "max_tokens": 0})
    check("zero max_tokens is 400", status == 400, f"status {status}")
    status, _ = post(args, "/v1/generate", {"prompt": "hi", "max_tokens": 10 ** 9})
    check("huge max_tokens is 400", status == 400, f"status {status}")
    status, _ = post(args, "/v1/generate",
                     {"prompt": "word " * 600, "max_tokens": 500})
    check("prompt+max_tokens over context is 400", status == 400, f"status {status}")
    check("server still healthy after bad requests", get(args, "/health")[0] == 200)

    print("\ngeneration")
    status, body = post(args, "/v1/generate",
                        {"prompt": "The capital of France is", "max_tokens": 24})
    r = json.loads(body)
    check("non-streaming returns 200", status == 200, str(status))
    check("non-streaming produced tokens", r.get("tokens_generated") == 24, str(r))
    check("finish_reason is length", r.get("finish_reason") == "length", str(r))
    nonstream_text = r["text"]

    status, tokens, done, ctype = stream(args, "The capital of France is", 24)
    check("stream returns 200", status == 200, str(status))
    check("stream content-type is text/event-stream",
          ctype is not None and "text/event-stream" in ctype, str(ctype))
    check("stream emitted token events", len(tokens) == 24, f"{len(tokens)} tokens")
    check("stream emitted done",
          done is not None and done.get("finish_reason") == "length", str(done))
    streamed = "".join(t["text"] for t in tokens) + (done or {}).get("text", "")
    check("streamed text equals non-streamed text", streamed == nonstream_text,
          f"\n    stream={streamed!r}\n    plain ={nonstream_text!r}")

    print("\nsampling")
    # Backward compatibility: a body with no sampling fields is what every
    # client written before this feature sends, and must stay greedy.
    a = post(args, "/v1/generate", {"prompt": "The capital of France is",
                                    "max_tokens": 16})[1]
    b = post(args, "/v1/generate", {"prompt": "The capital of France is",
                                    "max_tokens": 16})[1]
    check("omitting sampling fields is deterministic (greedy)",
          json.loads(a)["text"] == json.loads(b)["text"],
          f"{json.loads(a)['text']!r} vs {json.loads(b)['text']!r}")

    m_before = metrics(args)
    s1 = post(args, "/v1/generate", {"prompt": "Once upon a time", "max_tokens": 24,
                                     "temperature": 0.8, "top_k": 40, "seed": 4242})[1]
    s2 = post(args, "/v1/generate", {"prompt": "Once upon a time", "max_tokens": 24,
                                     "temperature": 0.8, "top_k": 40, "seed": 4242})[1]
    check("same seed reproduces the same sampled text",
          json.loads(s1)["text"] == json.loads(s2)["text"],
          f"{json.loads(s1)['text']!r} vs {json.loads(s2)['text']!r}")

    s3 = post(args, "/v1/generate", {"prompt": "Once upon a time", "max_tokens": 24,
                                     "temperature": 0.8, "top_k": 40, "seed": 99})[1]
    check("a different seed gives different text",
          json.loads(s3)["text"] != json.loads(s1)["text"],
          "two seeds produced identical output")

    g = post(args, "/v1/generate", {"prompt": "Once upon a time", "max_tokens": 24})[1]
    check("sampled output differs from greedy",
          json.loads(s1)["text"] != json.loads(g)["text"],
          "sampling produced the greedy sequence")

    m_after = metrics(args)
    check("metrics count greedy and sampled requests separately",
          m_after.get("sampled_requests", 0) > m_before.get("sampled_requests", 0)
          and m_after.get("greedy_requests", 0) > m_before.get("greedy_requests", 0),
          f"{m_before} -> {m_after}")

    # Streaming carries the same parameters and must agree with non-streaming.
    _, toks, done, _ = stream(args, "Once upon a time", 24,
                              temperature=0.8, top_k=40, seed=4242)
    streamed = "".join(t["text"] for t in toks) + (done or {}).get("text", "")
    check("streamed sampled text equals non-streamed for the same seed",
          streamed == json.loads(s1)["text"],
          f"\n    stream={streamed!r}\n    plain ={json.loads(s1)['text']!r}")

    print("\nsampling parameter validation")
    for body, why in [
        ({"prompt": "hi", "max_tokens": 8, "temperature": -1.0}, "negative temperature"),
        ({"prompt": "hi", "max_tokens": 8, "temperature": 0.8, "top_k": 0}, "top_k of zero"),
        ({"prompt": "hi", "max_tokens": 8, "top_k": 40}, "top_k without temperature"),
        ({"prompt": "hi", "max_tokens": 8, "seed": 1}, "seed without temperature"),
        ({"prompt": "hi", "max_tokens": 8, "temperature": 1e9}, "absurd temperature"),
    ]:
        status, _ = post(args, "/v1/generate", body)
        check(f"{why} is rejected", status == 400, f"status {status}")
    check("server healthy after bad sampling requests", get(args, "/health")[0] == 200)

    print("\nconcurrency and batching")
    results: dict = {}
    peak = {"batch": 0}
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            try:
                peak["batch"] = max(peak["batch"], metrics(args)["last_batch_size"])
            except Exception:
                pass
            time.sleep(0.005)

    w = threading.Thread(target=watch, daemon=True)
    w.start()

    prompts = ["The capital of France is",
               "In a distant galaxy the crew found",
               "def fibonacci(n):",
               "The history of the printing press begins in",
               "Q: Why is the sky blue?\nA:",
               "Once upon a time"]

    # Long enough that later arrivals land while earlier ones are still
    # generating. At ~1500 tok/s a 32-token request finishes in ~21 ms, which
    # is shorter than the arrival stagger -- the requests would then run one
    # after another and this check would fail for a reason that has nothing to
    # do with batching.
    def worker(i: int, delay: float):
        time.sleep(delay)
        results[i] = stream(args, prompts[i % len(prompts)], 200)

    threads = [threading.Thread(target=worker, args=(i, i * 0.005)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stop.set()
    w.join(timeout=1)

    check("all concurrent requests completed",
          len(results) == 6 and all(v[2] is not None for v in results.values()),
          str({k: v[2] for k, v in results.items()}))
    check("requests actually shared decode steps (batch > 1 observed)",
          peak["batch"] > 1, f"peak observed batch {peak['batch']}")

    alone = stream(args, prompts[0], 200)
    batched = results[0]
    check("batched output equals solo output for the same prompt",
          [t["token_id"] for t in batched[1]] == [t["token_id"] for t in alone[1]],
          "token ids differ")

    print("\ncancellation")
    before = metrics(args)
    status, tokens, done, _ = stream(args, "Once upon a time", 400, stop_after=3)
    check("early disconnect returns partial tokens", len(tokens) == 3, str(len(tokens)))
    deadline = time.time() + 15
    reclaimed = False
    while time.time() < deadline:
        m = metrics(args)
        if m["active_requests"] == 0 and m["kv_pages_free"] == total_pages:
            reclaimed = True
            break
        time.sleep(0.05)
    m = metrics(args)
    check("cancelled request left the batch and freed its pages", reclaimed,
          f"active={m['active_requests']} free={m['kv_pages_free']}/{total_pages}")
    check("cancellation was counted",
          m["cancelled_requests"] > before["cancelled_requests"], str(m))

    print("\npage accounting")
    m = metrics(args)
    check("all KV pages returned when idle",
          m["kv_pages_free"] == total_pages and m["kv_pages_used"] == 0, str(m))
    check("server healthy at end", get(args, "/health")[0] == 200)

    print()
    if FAILED:
        print(f"{PASSED} passed, {len(FAILED)} FAILED: {FAILED}")
        raise SystemExit(1)
    print(f"all {PASSED} checks passed")


if __name__ == "__main__":
    main()
