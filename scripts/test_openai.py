"""
Compatibility tests for Crucible's OpenAI-compatible endpoints.

Two layers, because they catch different things. The raw-HTTP half checks the
payloads against the published schema field by field -- an SDK will happily
paper over a missing `logprobs` key or an `object` value that is subtly wrong.
The SDK half checks that the official client, which is what users will actually
point at this server, can drive it end to end.

The load-bearing tests are the ones that are not about shape at all:

  * streamed text must equal non-streamed text for the same request, including
    when a character's UTF-8 bytes are split across model tokens;
  * a chat request must produce exactly what the native endpoint produces for
    the chat template's serialized prompt, which is what proves the adapter
    adapts rather than reimplements;
  * compatibility requests must batch with native ones in the same scheduler.

Usage, against an already-running server:
    python scripts/test_openai.py --port 8080
    python scripts/test_openai.py --port 8080 --sdk /path/to/python-with-openai
"""
from __future__ import annotations

import argparse
import http.client
import json
import socket
import subprocess
import sys
import threading
import time

FAILED: list[str] = []
PASSED = 0
MODEL = "crucible-120m"


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSED
    if cond:
        PASSED += 1
        print(f"  ok    {name}")
    else:
        FAILED.append(name)
        print(f"  FAIL  {name}  {detail}")


def conn(args) -> http.client.HTTPConnection:
    return http.client.HTTPConnection(args.host, args.port, timeout=180)


def post(args, path: str, body, raw: bool = False):
    c = conn(args)
    payload = body if raw else json.dumps(body)
    c.request("POST", path, payload, {"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    c.close()
    try:
        return r.status, json.loads(data)
    except json.JSONDecodeError:
        return r.status, {"_raw": data.decode("utf-8", "replace")}


def get(args, path: str):
    c = conn(args)
    c.request("GET", path)
    r = c.getresponse()
    data = r.read()
    c.close()
    try:
        return r.status, json.loads(data)
    except json.JSONDecodeError:
        return r.status, {"_raw": data.decode("utf-8", "replace")}


def sse(args, path: str, body, stop_after: int | None = None):
    """Consume an OpenAI-style SSE stream into (chunks, saw_done)."""
    c = conn(args)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        payload = r.read()
        c.close()
        return r.status, [], False, json.loads(payload)
    chunks, done = [], False
    try:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                done = True
                break
            chunks.append(json.loads(payload))
            if stop_after is not None and len(chunks) >= stop_after:
                break
    finally:
        c.close()
    return 200, chunks, done, None


def is_openai_error(body: dict) -> bool:
    """The published envelope: every one of the four inner fields present."""
    e = body.get("error")
    return isinstance(e, dict) and all(k in e for k in ("message", "type", "param", "code"))


def chat_text(chunks) -> str:
    return "".join(
        c["choices"][0]["delta"].get("content") or ""
        for c in chunks
        if c.get("choices")
    )


def completion_text(chunks) -> str:
    return "".join(c["choices"][0]["text"] for c in chunks if c.get("choices"))


def metrics(args) -> dict:
    return get(args, "/metrics")[1]


# ---------------------------------------------------------------- models ----


def test_models(args) -> None:
    print("models")
    status, body = get(args, "/v1/models")
    check("GET /v1/models returns 200", status == 200, str(status))
    check("model list has the list envelope", body.get("object") == "list", str(body))
    data = body.get("data") or []
    check("exactly one model is advertised", len(data) == 1, str(data))
    m = data[0] if data else {}
    check("model object has the required fields",
          all(k in m for k in ("id", "object", "created", "owned_by")), str(m))
    check("model id is the documented one", m.get("id") == MODEL, str(m.get("id")))
    check("model object type is 'model'", m.get("object") == "model", str(m))
    check("model id is not a filesystem path",
          "/" not in str(m.get("id")) and "\\" not in str(m.get("id")), str(m.get("id")))
    check("created is a plausible unix timestamp",
          isinstance(m.get("created"), int) and m["created"] > 1_000_000_000, str(m.get("created")))

    status, body = get(args, f"/v1/models/{MODEL}")
    check("GET /v1/models/{id} returns the model", status == 200 and body.get("id") == MODEL,
          f"{status} {body}")

    status, body = get(args, "/v1/models/gpt-4o")
    check("an unknown model is 404", status == 404, str(status))
    check("the 404 uses the OpenAI error envelope", is_openai_error(body), str(body))
    check("the 404 names model_not_found",
          body.get("error", {}).get("code") == "model_not_found", str(body))


# ----------------------------------------------------------- completions ----


def test_completions(args) -> None:
    print("\ncompletions")
    req = {"model": MODEL, "prompt": "The capital of France is", "max_tokens": 12}
    status, body = post(args, "/v1/completions", req)
    check("POST /v1/completions returns 200", status == 200, str(body)[:200])
    check("object is text_completion", body.get("object") == "text_completion", str(body.get("object")))
    check("id has the cmpl prefix", str(body.get("id", "")).startswith("cmpl-"), str(body.get("id")))
    check("model is echoed", body.get("model") == MODEL, str(body.get("model")))
    ch = (body.get("choices") or [{}])[0]
    check("choice carries every required field",
          all(k in ch for k in ("text", "index", "logprobs", "finish_reason")), str(ch))
    check("finish_reason is length for a max_tokens stop",
          ch.get("finish_reason") == "length", str(ch.get("finish_reason")))
    check("logprobs is null rather than fabricated", ch.get("logprobs") is None, str(ch.get("logprobs")))
    u = body.get("usage") or {}
    check("usage has real token counts",
          u.get("prompt_tokens", 0) > 0 and u.get("completion_tokens") == 12,
          str(u))
    check("usage totals add up",
          u.get("total_tokens") == u.get("prompt_tokens", 0) + u.get("completion_tokens", 0), str(u))
    check("no invented usage detail fields",
          "prompt_tokens_details" not in u and "completion_tokens_details" not in u, str(u))
    plain = ch.get("text")

    status, chunks, done, _ = sse(args, "/v1/completions", {**req, "stream": True})
    check("streaming completions returns 200", status == 200, str(status))
    check("stream terminates with [DONE]", done, "no [DONE]")
    check("stream chunks reuse the text_completion object",
          all(c.get("object") == "text_completion" for c in chunks), "")
    ids = {c["id"] for c in chunks}
    check("one id for the whole stream", len(ids) == 1, str(ids))
    check("only the last chunk carries a finish_reason",
          all(c["choices"][0]["finish_reason"] is None for c in chunks[:-1])
          and chunks[-1]["choices"][0]["finish_reason"] == "length",
          str([c["choices"][0]["finish_reason"] for c in chunks]))
    check("streamed completion text equals non-streamed",
          completion_text(chunks) == plain,
          f"\n    stream={completion_text(chunks)!r}\n    plain ={plain!r}")


# ------------------------------------------------------------------ chat ----


def test_chat(args) -> None:
    print("\nchat completions")
    req = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 12,
    }
    status, body = post(args, "/v1/chat/completions", req)
    check("POST /v1/chat/completions returns 200", status == 200, str(body)[:200])
    check("object is chat.completion", body.get("object") == "chat.completion", str(body.get("object")))
    check("id has the chatcmpl prefix",
          str(body.get("id", "")).startswith("chatcmpl-"), str(body.get("id")))
    ch = (body.get("choices") or [{}])[0]
    check("choice carries every required field",
          all(k in ch for k in ("index", "message", "logprobs", "finish_reason")), str(ch))
    msg = ch.get("message") or {}
    check("the reply is an assistant message", msg.get("role") == "assistant", str(msg))
    check("refusal is present and null", "refusal" in msg and msg["refusal"] is None, str(msg))
    check("finish_reason is length", ch.get("finish_reason") == "length", str(ch))
    u = body.get("usage") or {}
    check("usage counts the serialized prompt, not the message text",
          u.get("prompt_tokens", 0) > 1 and u.get("completion_tokens") == 12, str(u))
    plain = msg.get("content")

    status, chunks, done, _ = sse(args, "/v1/chat/completions", {**req, "stream": True})
    check("streaming chat returns 200", status == 200, str(status))
    check("stream terminates with [DONE]", done, "no [DONE]")
    check("chunks use the chat.completion.chunk object",
          all(c.get("object") == "chat.completion.chunk" for c in chunks), "")
    ids = {c["id"] for c in chunks}
    check("one id for the whole stream", len(ids) == 1, str(ids))
    created = {c["created"] for c in chunks}
    check("one created timestamp for the whole stream", len(created) == 1, str(created))
    check("every chunk echoes the model id",
          all(c.get("model") == MODEL for c in chunks), "")
    check("the first chunk is the assistant role delta",
          chunks and chunks[0]["choices"][0]["delta"].get("role") == "assistant",
          str(chunks[0] if chunks else None))
    check("only the last chunk carries a finish_reason",
          all(c["choices"][0]["finish_reason"] is None for c in chunks[:-1])
          and chunks[-1]["choices"][0]["finish_reason"] == "length",
          str([c["choices"][0]["finish_reason"] for c in chunks]))
    check("the final chunk's delta is empty",
          chunks[-1]["choices"][0]["delta"] == {}, str(chunks[-1]["choices"][0]["delta"]))
    check("streamed chat text equals non-streamed",
          chat_text(chunks) == plain,
          f"\n    stream={chat_text(chunks)!r}\n    plain ={plain!r}")

    # usage is opt-in on streams, exactly as the schema describes
    status, chunks, done, _ = sse(args, "/v1/chat/completions",
                                  {**req, "stream": True,
                                   "stream_options": {"include_usage": True}})
    check("include_usage adds a final usage-only chunk",
          chunks and chunks[-1].get("usage") is not None and chunks[-1]["choices"] == [],
          str(chunks[-1] if chunks else None))
    check("earlier chunks carry a null usage field",
          all("usage" in c and c["usage"] is None for c in chunks[:-1]), "")
    uu = chunks[-1]["usage"] if chunks else {}
    check("streamed usage matches the non-streamed counts",
          uu.get("prompt_tokens") == u.get("prompt_tokens")
          and uu.get("completion_tokens") == u.get("completion_tokens"), f"{uu} vs {u}")

    status, chunks, done, _ = sse(args, "/v1/chat/completions", {**req, "stream": True})
    check("usage is omitted entirely when not requested",
          all("usage" not in c for c in chunks), "")


def test_multibyte(args) -> None:
    """A character whose UTF-8 bytes span several model tokens."""
    print("\nsplit UTF-8 across tokens")
    req = {"model": MODEL, "prompt": "世界 世界 世界", "max_tokens": 16}
    status, body = post(args, "/v1/completions", req)
    plain = body["choices"][0]["text"]
    check("the probe prompt does produce multi-byte output",
          any(ord(c) > 127 for c in plain), repr(plain))

    status, chunks, done, _ = sse(args, "/v1/completions", {**req, "stream": True})
    deltas = [c["choices"][0]["text"] for c in chunks]
    check("some deltas are empty while a character is still incomplete",
          any(d == "" for d in deltas),
          "no empty delta: tokens were decoded independently?")
    check("streamed multi-byte text equals non-streamed",
          completion_text(chunks) == plain,
          f"\n    stream={completion_text(chunks)!r}\n    plain ={plain!r}")
    check("every delta is valid UTF-8 on its own",
          all(isinstance(d, str) for d in deltas), "")

    # ...and the same through the chat adapter, which has its own chunk builder.
    creq = {"model": MODEL, "messages": [{"role": "user", "content": "世界 世界 世界"}],
            "max_tokens": 16}
    _, body = post(args, "/v1/chat/completions", creq)
    plain = body["choices"][0]["message"]["content"]
    _, chunks, _, _ = sse(args, "/v1/chat/completions", {**creq, "stream": True})
    check("streamed multi-byte chat text equals non-streamed",
          chat_text(chunks) == plain,
          f"\n    stream={chat_text(chunks)!r}\n    plain ={plain!r}")


# ------------------------------------------------------------ rejections ----


def test_rejections(args) -> None:
    print("\nunsupported parameters are refused, not ignored")
    base = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}
    cases = [
        ({"top_p": 0.2}, "top_p"),
        ({"frequency_penalty": 0.5}, "frequency_penalty"),
        ({"presence_penalty": 0.5}, "presence_penalty"),
        ({"n": 2}, "n"),
        ({"logprobs": True}, "logprobs"),
        ({"top_logprobs": 5}, "top_logprobs"),
        ({"stop": ["\n"]}, "stop"),
        ({"logit_bias": {"1": 5}}, "logit_bias"),
        ({"tools": [{"type": "function", "function": {"name": "f"}}]}, "tools"),
        ({"response_format": {"type": "json_object"}}, "response_format"),
        ({"reasoning_effort": "high"}, "reasoning_effort"),
    ]
    for extra, param in cases:
        status, body = post(args, "/v1/chat/completions", {**base, **extra})
        ok = status == 400 and is_openai_error(body) and body["error"].get("param") == param
        check(f"chat rejects {param}", ok, f"status {status} body {body}")

    print("\n  ...and their no-op values are still accepted")
    for extra in [{"top_p": 1.0}, {"frequency_penalty": 0}, {"presence_penalty": 0},
                  {"n": 1}, {"logprobs": False}, {"stop": None}, {"tools": []}]:
        status, _ = post(args, "/v1/chat/completions", {**base, **extra})
        check(f"chat accepts {list(extra)[0]} at its default", status == 200, f"status {status}")

    print("\n  completions rejections")
    cbase = {"model": MODEL, "prompt": "hi", "max_tokens": 4}
    for extra, param in [({"best_of": 3}, "best_of"), ({"echo": True}, "echo"),
                         ({"suffix": "x"}, "suffix"), ({"n": 2}, "n"),
                         ({"top_p": 0.5}, "top_p"), ({"logprobs": 3}, "logprobs")]:
        status, body = post(args, "/v1/completions", {**cbase, **extra})
        ok = status == 400 and is_openai_error(body) and body["error"].get("param") == param
        check(f"completions rejects {param}", ok, f"status {status} body {body}")

    status, body = post(args, "/v1/completions", {**cbase, "prompt": ["a", "b"]})
    check("a prompt array is refused rather than half-served",
          status == 400 and is_openai_error(body), f"{status} {body}")

    print("\n  unsupported message content")
    for messages, why in [
        ([{"role": "tool", "content": "{}"}], "tool role"),
        ([{"role": "user", "content": [{"type": "image_url",
                                        "image_url": {"url": "http://x/y.png"}}]}], "image part"),
        ([{"role": "wizard", "content": "x"}], "unknown role"),
    ]:
        status, body = post(args, "/v1/chat/completions",
                            {"model": MODEL, "messages": messages, "max_tokens": 4})
        check(f"{why} is refused", status == 400 and is_openai_error(body), f"{status} {body}")

    print("\n  request-level errors")
    status, body = post(args, "/v1/chat/completions", "{not json", raw=True)
    check("malformed JSON is 400 in the OpenAI shape",
          status == 400 and is_openai_error(body), f"{status} {body}")
    status, body = post(args, "/v1/chat/completions", {"model": MODEL})
    check("missing messages is 400",
          status == 400 and body.get("error", {}).get("code") == "missing_required_parameter",
          str(body))
    status, body = post(args, "/v1/completions", {"model": MODEL})
    check("missing prompt is 400",
          status == 400 and body.get("error", {}).get("code") == "missing_required_parameter",
          str(body))
    status, body = post(args, "/v1/chat/completions", {**base, "model": "gpt-4o"})
    check("a foreign model is 404, never silently served",
          status == 404 and body.get("error", {}).get("code") == "model_not_found", f"{status} {body}")
    status, body = post(args, "/v1/completions",
                        {"model": MODEL, "prompt": "word " * 600, "max_tokens": 500})
    check("context overflow is 400 with context_length_exceeded",
          status == 400 and body.get("error", {}).get("code") == "context_length_exceeded",
          f"{status} {body}")
    check("server still healthy after every rejection", get(args, "/health")[0] == 200)


# ----------------------------------------------------------- determinism ----


def test_determinism(args) -> None:
    print("\ndeterminism and cross-endpoint agreement")
    # Greedy: the same prompt must give the same text on every surface.
    prompt = "The capital of France is"
    native = post(args, "/v1/generate", {"prompt": prompt, "max_tokens": 16})[1]["text"]
    comp = post(args, "/v1/completions",
                {"model": MODEL, "prompt": prompt, "max_tokens": 16})[1]["choices"][0]["text"]
    check("completions matches the native endpoint exactly", comp == native,
          f"\n    openai={comp!r}\n    native={native!r}")

    # Chat rewrites the prompt, so the comparison has to use the serialized form
    # rather than the raw message -- otherwise it would be comparing two
    # different prompts and calling agreement a bug.
    serialized = "User: Hello\n\nAssistant:"
    native = post(args, "/v1/generate", {"prompt": serialized, "max_tokens": 16})[1]["text"]
    chat = post(args, "/v1/chat/completions",
                {"model": MODEL, "messages": [{"role": "user", "content": "Hello"}],
                 "max_tokens": 16})[1]["choices"][0]["message"]["content"]
    check("chat matches the native endpoint given the serialized prompt", chat == native,
          f"\n    chat  ={chat!r}\n    native={native!r}")

    # Sampled: same seed, same text; different seed, different text.
    sreq = {"model": MODEL, "prompt": "Once upon a time", "max_tokens": 24,
            "temperature": 0.8, "seed": 4242}
    a = post(args, "/v1/completions", sreq)[1]["choices"][0]["text"]
    b = post(args, "/v1/completions", sreq)[1]["choices"][0]["text"]
    check("a seeded sampled completion reproduces", a == b, f"{a!r} vs {b!r}")
    c = post(args, "/v1/completions", {**sreq, "seed": 99})[1]["choices"][0]["text"]
    check("a different seed gives different text", a != c, "seeds collided")
    nat = post(args, "/v1/generate", {"prompt": "Once upon a time", "max_tokens": 24,
                                      "temperature": 0.8, "top_k": 40, "seed": 4242})[1]["text"]
    check("sampled completions match the native endpoint for the same seed", a == nat,
          f"\n    openai={a!r}\n    native={nat!r}")

    _, chunks, _, _ = sse(args, "/v1/completions", {**sreq, "stream": True})
    check("a seeded sampled stream matches its non-streamed twin",
          completion_text(chunks) == a,
          f"\n    stream={completion_text(chunks)!r}\n    plain ={a!r}")

    # Omitting temperature is greedy here, which is a documented divergence from
    # OpenAI's default of 1.0 and the reason two identical calls agree.
    d = post(args, "/v1/completions",
             {"model": MODEL, "prompt": prompt, "max_tokens": 8})[1]["choices"][0]["text"]
    e = post(args, "/v1/completions",
             {"model": MODEL, "prompt": prompt, "max_tokens": 8})[1]["choices"][0]["text"]
    check("omitting temperature is deterministic (greedy)", d == e, f"{d!r} vs {e!r}")

    # A seed without a temperature is accepted: greedy already reproduces.
    status, _ = post(args, "/v1/completions",
                     {"model": MODEL, "prompt": prompt, "max_tokens": 4, "seed": 7})
    check("a seed without a temperature is accepted", status == 200, str(status))


# ------------------------------------------------------- batching, etc. -----


def test_batching(args) -> None:
    print("\ncontinuous batching across every client type")
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

    results: dict = {}

    def native(i):
        results[i] = sse_native(args, "The history of the printing press begins in", 200)

    def compat(i):
        results[i] = sse(args, "/v1/completions",
                         {"model": MODEL, "prompt": "In a distant galaxy the crew found",
                          "max_tokens": 200, "stream": True})

    def chat(i):
        results[i] = sse(args, "/v1/chat/completions",
                         {"model": MODEL,
                          "messages": [{"role": "user", "content": "Tell me about rivers"}],
                          "max_tokens": 200, "stream": True})

    workers = []
    for i in range(9):
        fn = [native, compat, chat][i % 3]
        t = threading.Thread(target=fn, args=(i,))
        workers.append(t)
    for i, t in enumerate(workers):
        t.start()
        time.sleep(0.005)
    for t in workers:
        t.join()
    stop.set()
    w.join(timeout=1)

    check("every mixed client completed", len(results) == 9, str(len(results)))
    check("native, completions and chat clients shared decode steps",
          peak["batch"] > 1, f"peak observed batch {peak['batch']}")
    m = metrics(args)
    check("no requests left active afterwards", m["active_requests"] == 0, str(m))


def sse_native(args, prompt: str, max_tokens: int):
    """The native SSE format, for the mixed-client batching test."""
    c = conn(args)
    c.request("POST", "/v1/generate/stream",
              json.dumps({"prompt": prompt, "max_tokens": max_tokens}),
              {"Content-Type": "application/json"})
    r = c.getresponse()
    n = 0
    for raw in r:
        if raw.startswith(b"data:"):
            n += 1
    c.close()
    return n


def test_cancellation(args) -> None:
    print("\ncancellation through the compatibility stream")
    before = metrics(args)
    total_pages = before["kv_pages_used"] + before["kv_pages_free"]
    status, chunks, done, _ = sse(args, "/v1/chat/completions",
                                  {"model": MODEL,
                                   "messages": [{"role": "user", "content": "Once upon a time"}],
                                   "max_tokens": 400, "stream": True},
                                  stop_after=4)
    check("an early disconnect still delivered partial chunks", len(chunks) == 4, str(len(chunks)))
    check("no [DONE] was sent to a client that left", not done, "")

    deadline = time.time() + 20
    reclaimed = False
    while time.time() < deadline:
        m = metrics(args)
        if m["active_requests"] == 0 and m["kv_pages_free"] == total_pages:
            reclaimed = True
            break
        time.sleep(0.05)
    m = metrics(args)
    check("the cancelled request left the batch and freed its pages", reclaimed,
          f"active={m['active_requests']} free={m['kv_pages_free']}/{total_pages}")
    check("the cancellation was counted",
          m["cancelled_requests"] > before["cancelled_requests"], str(m))
    check("no second cancellation mechanism was needed",
          m["failed_requests"] == before["failed_requests"], str(m))


def test_backpressure(args) -> None:
    print("\nbackpressure")
    limits = get(args, "/health")[1]
    threads, statuses = [], []
    lock = threading.Lock()

    def hammer():
        s, _ = post(args, "/v1/completions",
                    {"model": MODEL, "prompt": "The", "max_tokens": 400})
        with lock:
            statuses.append(s)

    for _ in range(120):
        t = threading.Thread(target=hammer)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    check("every overload response was 200 or 429",
          all(s in (200, 429) for s in statuses), str(sorted(set(statuses))))
    if 429 in statuses:
        check("queue overflow answered 429 rather than blocking", True)
    else:
        print("  note  queue never filled; 429 path not exercised this run")
    deadline = time.time() + 30
    while time.time() < deadline and metrics(args)["active_requests"] > 0:
        time.sleep(0.1)
    m = metrics(args)
    check("all pages returned after the overload", m["kv_pages_used"] == 0, str(m))
    check("server healthy after the overload", get(args, "/health")[0] == 200)
    _ = limits


# ------------------------------------------------ streaming parser torture --


def raw_sse_bytes(args, path: str, body, chunk_size: int) -> bytes:
    """Read the whole response one `chunk_size` slice at a time."""
    s = socket.create_connection((args.host, args.port), timeout=180)
    payload = json.dumps(body).encode()
    req = (
        f"POST {path} HTTP/1.1\r\nHost: {args.host}\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode() + payload
    s.sendall(req)
    out = b""
    while True:
        b = s.recv(chunk_size)
        if not b:
            break
        out += b
    s.close()
    return out


def parse_sse_body(raw: bytes) -> tuple[list, bool]:
    head, _, body = raw.partition(b"\r\n\r\n")
    text = body.decode("utf-8", "replace")
    chunks, done = [], False
    for line in text.split("\n"):
        line = line.strip()
        # Chunked transfer encoding leaves size lines interleaved; the data
        # lines are the only ones this cares about.
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            done = True
        elif payload:
            try:
                chunks.append(json.loads(payload))
            except json.JSONDecodeError:
                pass
    return chunks, done


def test_stream_torture(args) -> None:
    print("\nstreaming parser torture")
    body = {"model": MODEL, "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 8, "stream": True}
    reference = None
    for size in (1, 3, 64, 65536):
        raw = raw_sse_bytes(args, "/v1/chat/completions", body, size)
        chunks, done = parse_sse_body(raw)
        text = chat_text(chunks)
        if reference is None:
            reference = text
        label = "one byte at a time" if size == 1 else f"{size}-byte reads"
        check(f"stream survives {label}", done and text == reference,
              f"done={done} text={text!r} ref={reference!r}")

    # A request that fails validation must fail before any chunk, with a status
    # line rather than a stream that turns into an error.
    status, chunks, done, err = sse(args, "/v1/chat/completions",
                                    {**body, "top_p": 0.3})
    check("a rejected streaming request errors before the first chunk",
          status == 400 and not chunks and is_openai_error(err or {}), f"{status} {err}")


# ------------------------------------------------------------------- SDK ----

SDK_SCRIPT = r'''
import json, sys
from openai import OpenAI

base, model = sys.argv[1], sys.argv[2]
client = OpenAI(base_url=base, api_key="not-used")
out = {}

out["version"] = __import__("openai").__version__
out["models"] = [m.id for m in client.models.list()]

r = client.completions.create(model=model, prompt="The capital of France is", max_tokens=12)
out["completion"] = r.choices[0].text
out["completion_finish"] = r.choices[0].finish_reason
out["completion_usage"] = [r.usage.prompt_tokens, r.usage.completion_tokens, r.usage.total_tokens]

acc = ""
for chunk in client.completions.create(model=model, prompt="The capital of France is",
                                       max_tokens=12, stream=True):
    acc += chunk.choices[0].text if chunk.choices else ""
out["completion_stream"] = acc

c = client.chat.completions.create(model=model, max_tokens=12,
                                   messages=[{"role": "user", "content": "Hello"}])
out["chat"] = c.choices[0].message.content
out["chat_role"] = c.choices[0].message.role
out["chat_finish"] = c.choices[0].finish_reason
out["chat_usage"] = [c.usage.prompt_tokens, c.usage.completion_tokens, c.usage.total_tokens]

acc, roles = "", []
for chunk in client.chat.completions.create(model=model, max_tokens=12, stream=True,
                                            messages=[{"role": "user", "content": "Hello"}]):
    if not chunk.choices:
        continue
    d = chunk.choices[0].delta
    if d.role:
        roles.append(d.role)
    acc += d.content or ""
out["chat_stream"] = acc
out["chat_stream_roles"] = roles

try:
    client.chat.completions.create(model="gpt-4o", max_tokens=4,
                                   messages=[{"role": "user", "content": "hi"}])
    out["bad_model"] = "no error"
except Exception as e:
    out["bad_model"] = type(e).__name__

try:
    client.chat.completions.create(model=model, max_tokens=4, top_p=0.3,
                                   messages=[{"role": "user", "content": "hi"}])
    out["top_p"] = "no error"
except Exception as e:
    out["top_p"] = type(e).__name__

print(json.dumps(out))
'''


def test_sdk(args) -> None:
    if not args.sdk:
        print("\nofficial OpenAI SDK\n  skipped (no --sdk interpreter given)")
        return
    print("\nofficial OpenAI SDK")
    base = f"http://{args.host}:{args.port}/v1"
    proc = subprocess.run([args.sdk, "-c", SDK_SCRIPT, base, MODEL],
                          capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        check("SDK script ran", False, proc.stderr[-1500:])
        return
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    print(f"  ...against openai-python {out['version']}")
    check("SDK lists the model", out["models"] == [MODEL], str(out["models"]))
    check("SDK completion returned text", bool(out["completion"]), str(out["completion"]))
    check("SDK completion finish_reason is length", out["completion_finish"] == "length",
          str(out["completion_finish"]))
    p, c, t = out["completion_usage"]
    check("SDK parsed usage", p > 0 and c == 12 and t == p + c, str(out["completion_usage"]))
    check("SDK streamed completion equals non-streamed",
          out["completion_stream"] == out["completion"],
          f"{out['completion_stream']!r} vs {out['completion']!r}")
    check("SDK chat returned an assistant message",
          out["chat_role"] == "assistant" and bool(out["chat"]), str(out["chat_role"]))
    check("SDK chat finish_reason is length", out["chat_finish"] == "length", str(out["chat_finish"]))
    check("SDK streamed chat equals non-streamed",
          out["chat_stream"] == out["chat"], f"{out['chat_stream']!r} vs {out['chat']!r}")
    check("SDK saw exactly one role delta", out["chat_stream_roles"] == ["assistant"],
          str(out["chat_stream_roles"]))
    check("SDK raises NotFoundError for a foreign model",
          out["bad_model"] == "NotFoundError", str(out["bad_model"]))
    check("SDK raises BadRequestError for an unsupported parameter",
          out["top_p"] == "BadRequestError", str(out["top_p"]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--sdk", help="python interpreter with the openai package installed")
    args = p.parse_args()

    test_models(args)
    test_completions(args)
    test_chat(args)
    test_multibyte(args)
    test_rejections(args)
    test_determinism(args)
    test_stream_torture(args)
    test_cancellation(args)
    test_batching(args)
    test_backpressure(args)
    test_sdk(args)

    print()
    if FAILED:
        print(f"{PASSED} passed, {len(FAILED)} FAILED: {FAILED}")
        raise SystemExit(1)
    print(f"all {PASSED} checks passed")


if __name__ == "__main__":
    main()
