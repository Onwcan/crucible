"""
Compatibility tests for Crucible's Anthropic-compatible Messages API.

Same two-layer shape as test_openai.py, for the same reason: the raw-HTTP half
checks payloads field by field against the published schema, and the SDK half
checks that the client people will actually point at this server can drive it.
An SDK papers over a wrong event name or a missing `stop_sequence` key; raw HTTP
does not.

The load-bearing tests are not about shape:

  * streamed deltas must concatenate to exactly the non-streamed content, and
    still do when a character's UTF-8 bytes span several model tokens;
  * an Anthropic conversation and the equivalent OpenAI one must produce the
    same prompt and therefore the same tokens, which is what says the shared
    serializer is genuinely shared;
  * count_tokens must return the exact number a later request reports as
    usage.input_tokens, not an estimate of it;
  * Anthropic requests must batch with native, OpenAI and TUI traffic in one
    scheduler.

Usage, against an already-running server:
    python scripts/test_anthropic.py --port 8080
    python scripts/test_anthropic.py --port 8080 --sdk /path/to/python-with-anthropic
"""
from __future__ import annotations

import argparse
import http.client
import json
import socket
import subprocess
import threading
import time

FAILED: list[str] = []
PASSED = 0
MODEL = "crucible-120m"
VERSION = "2023-06-01"
HDRS = {"Content-Type": "application/json", "anthropic-version": VERSION,
        "x-api-key": "not-used"}


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


def post(args, path: str, body, headers=None, raw: bool = False):
    c = conn(args)
    payload = body if raw else json.dumps(body)
    c.request("POST", path, payload, headers if headers is not None else HDRS)
    r = c.getresponse()
    data = r.read()
    rid = r.getheader("request-id")
    c.close()
    try:
        return r.status, json.loads(data), rid
    except json.JSONDecodeError:
        return r.status, {"_raw": data.decode("utf-8", "replace")}, rid


def get(args, path: str, headers=None):
    c = conn(args)
    c.request("GET", path, headers=headers or {})
    r = c.getresponse()
    data = r.read()
    c.close()
    try:
        return r.status, json.loads(data)
    except json.JSONDecodeError:
        return r.status, {"_raw": data.decode("utf-8", "replace")}


def sse(args, body, stop_after: int | None = None):
    """Consume the typed event stream into [(event_name, payload), ...]."""
    c = conn(args)
    c.request("POST", "/v1/messages", json.dumps(body), HDRS)
    r = c.getresponse()
    if r.status != 200:
        payload = r.read()
        c.close()
        return r.status, [], json.loads(payload)
    events, name = [], None
    try:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                events.append((name, json.loads(line[5:].strip())))
                if stop_after is not None and len(events) >= stop_after:
                    break
    finally:
        c.close()
    return 200, events, None


def deltas(events) -> str:
    return "".join(p["delta"]["text"] for n, p in events if n == "content_block_delta")


def is_anthropic_error(body: dict) -> bool:
    return (body.get("type") == "error"
            and isinstance(body.get("error"), dict)
            and "type" in body["error"] and "message" in body["error"]
            and isinstance(body.get("request_id"), str))


def metrics(args) -> dict:
    return get(args, "/metrics")[1]


# --------------------------------------------------------------- messages ---


def test_messages(args) -> None:
    print("messages")
    req = {"model": MODEL, "max_tokens": 12,
           "messages": [{"role": "user", "content": "Hello"}]}
    status, body, rid = post(args, "/v1/messages", req)
    check("POST /v1/messages returns 200", status == 200, str(body)[:200])
    check("type is message", body.get("type") == "message", str(body.get("type")))
    check("role is assistant", body.get("role") == "assistant", str(body.get("role")))
    check("id has the msg_ prefix", str(body.get("id", "")).startswith("msg_"), str(body.get("id")))
    check("model is echoed", body.get("model") == MODEL, str(body.get("model")))
    content = body.get("content") or []
    check("content is a list of one text block",
          len(content) == 1 and content[0].get("type") == "text", str(content))
    check("stop_reason is max_tokens", body.get("stop_reason") == "max_tokens",
          str(body.get("stop_reason")))
    check("stop_sequence is present and null",
          "stop_sequence" in body and body["stop_sequence"] is None, str(body.get("stop_sequence")))
    u = body.get("usage") or {}
    check("usage carries real token counts",
          u.get("input_tokens", 0) > 0 and u.get("output_tokens") == 12, str(u))
    check("no fabricated cache counters",
          "cache_read_input_tokens" not in u and "cache_creation_input_tokens" not in u, str(u))
    check("a request-id header is returned",
          rid is not None and rid.startswith("req_"), str(rid))
    plain = content[0]["text"] if content else None

    print("\n  streaming")
    status, events, _ = sse(args, {**req, "stream": True})
    names = [n for n, _ in events]
    check("stream returns 200", status == 200, str(status))
    check("the event sequence is the documented one",
          names[0] == "message_start" and names[1] == "content_block_start"
          and names[-3:] == ["content_block_stop", "message_delta", "message_stop"],
          str(names))
    check("no OpenAI [DONE] sentinel leaked in", "[DONE]" not in str(events), "")
    start = events[0][1]
    check("message_start carries the message shell",
          start["message"]["content"] == [] and start["message"]["stop_reason"] is None,
          str(start))
    check("message_start reports input tokens",
          start["message"]["usage"]["input_tokens"] == u.get("input_tokens"), str(start))
    ids = {p["message"]["id"] for n, p in events if n == "message_start"}
    check("the message id is stable across the stream", len(ids) == 1, str(ids))
    cbs = [p for n, p in events if n == "content_block_start"]
    check("content_block_start opens an empty text block",
          cbs and cbs[0]["content_block"] == {"type": "text", "text": ""}, str(cbs[:1]))
    d = [p for n, p in events if n == "content_block_delta"]
    check("deltas are text_delta blocks at index 0",
          all(p["delta"]["type"] == "text_delta" and p["index"] == 0 for p in d), "")
    md = [p for n, p in events if n == "message_delta"][0]
    check("message_delta carries the stop reason",
          md["delta"]["stop_reason"] == "max_tokens" and md["delta"]["stop_sequence"] is None,
          str(md))
    check("message_delta carries cumulative output tokens",
          md["usage"]["output_tokens"] == 12, str(md["usage"]))
    check("streamed text equals non-streamed",
          deltas(events) == plain, f"\n    stream={deltas(events)!r}\n    plain ={plain!r}")


def test_system_and_turns(args) -> None:
    print("\nsystem prompt and multi-turn")
    with_system = {"model": MODEL, "max_tokens": 8, "system": "Be terse.",
                   "messages": [{"role": "user", "content": "Hello"}]}
    status, body, _ = post(args, "/v1/messages", with_system)
    check("a top-level system prompt is accepted", status == 200, str(body)[:160])
    sys_tokens = body["usage"]["input_tokens"]

    plain = post(args, "/v1/messages",
                 {"model": MODEL, "max_tokens": 8,
                  "messages": [{"role": "user", "content": "Hello"}]})[1]
    check("the system prompt lengthens the serialized prompt",
          sys_tokens > plain["usage"]["input_tokens"],
          f"{sys_tokens} vs {plain['usage']['input_tokens']}")

    # A system string and the equivalent one-block list must be identical.
    blocks = {**with_system, "system": [{"type": "text", "text": "Be terse."}]}
    status, body2, _ = post(args, "/v1/messages", blocks)
    check("system text blocks match the string form",
          body2["content"][0]["text"] == body["content"][0]["text"]
          and body2["usage"]["input_tokens"] == sys_tokens,
          f"{body2['usage']} vs {body['usage']}")

    multi = {"model": MODEL, "max_tokens": 8, "messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "again"}]}
    check("a multi-turn conversation is accepted",
          post(args, "/v1/messages", multi)[0] == 200)

    consecutive = {"model": MODEL, "max_tokens": 8, "messages": [
        {"role": "user", "content": "one"}, {"role": "user", "content": "two"}]}
    check("consecutive same-role messages are accepted",
          post(args, "/v1/messages", consecutive)[0] == 200)

    prefill = {"model": MODEL, "max_tokens": 8, "messages": [
        {"role": "user", "content": "count"},
        {"role": "assistant", "content": "one two"}]}
    check("a trailing assistant message (prefill) is accepted",
          post(args, "/v1/messages", prefill)[0] == 200)

    text_blocks = {"model": MODEL, "max_tokens": 8, "messages": [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]}]}
    status, body3, _ = post(args, "/v1/messages", text_blocks)
    check("message text blocks match the string form",
          status == 200 and body3["usage"]["input_tokens"] == plain["usage"]["input_tokens"],
          f"{status} {body3.get('usage')}")


def test_count_tokens(args) -> None:
    print("\ncount_tokens")
    cases = [
        {"messages": [{"role": "user", "content": "Hello"}]},
        {"system": "Be terse.", "messages": [{"role": "user", "content": "Hello"}]},
        {"messages": [{"role": "user", "content": "hi"},
                      {"role": "assistant", "content": "hello"},
                      {"role": "user", "content": "again"}]},
        {"messages": [{"role": "user", "content": "世界 世界 世界"}]},
        {"system": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}],
         "messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}]},
    ]
    for i, case in enumerate(cases):
        status, counted, rid = post(args, "/v1/messages/count_tokens", {"model": MODEL, **case})
        if status != 200:
            check(f"case {i} counted", False, f"{status} {counted}")
            continue
        _, generated, _ = post(args, "/v1/messages",
                               {"model": MODEL, "max_tokens": 1, **case})
        check(f"count_tokens[{i}] equals the reported input_tokens",
              counted.get("input_tokens") == generated["usage"]["input_tokens"],
              f"{counted} vs {generated['usage']}")
    # A third opinion: the native endpoint tokenises the serialized prompt
    # itself, so all three paths must report the same number for the same text.
    serialized = "User: Hello\n\nAssistant:"
    native = post(args, "/v1/generate", {"prompt": serialized, "max_tokens": 1},
                  headers={"Content-Type": "application/json"})[1]
    counted = post(args, "/v1/messages/count_tokens",
                   {"model": MODEL, "messages": [{"role": "user", "content": "Hello"}]})[1]
    check("count_tokens agrees with the native endpoint tokenising the same text",
          counted["input_tokens"] == native["prompt_tokens"],
          f"{counted} vs native prompt_tokens {native['prompt_tokens']}")

    check("count_tokens returns only input_tokens",
          set(post(args, "/v1/messages/count_tokens",
                   {"model": MODEL, **cases[0]})[1].keys()) == {"input_tokens"}, "")
    status, body, _ = post(args, "/v1/messages/count_tokens",
                           {"model": "claude-opus-4", **cases[0]})
    check("count_tokens rejects an unknown model",
          status == 404 and is_anthropic_error(body), f"{status} {body}")


def test_multibyte(args) -> None:
    print("\nsplit UTF-8 across tokens")
    req = {"model": MODEL, "max_tokens": 16,
           "messages": [{"role": "user", "content": "世界 世界 世界"}]}
    plain = post(args, "/v1/messages", req)[1]["content"][0]["text"]
    check("the probe prompt produces multi-byte output",
          any(ord(c) > 127 for c in plain), repr(plain))
    _, events, _ = sse(args, {**req, "stream": True})
    texts = [p["delta"]["text"] for n, p in events if n == "content_block_delta"]
    check("some deltas are empty while a character is incomplete",
          any(t == "" for t in texts), "tokens were decoded independently?")
    check("streamed multi-byte text equals non-streamed",
          deltas(events) == plain, f"\n    stream={deltas(events)!r}\n    plain ={plain!r}")


# ------------------------------------------------------------ rejections ----


def test_rejections(args) -> None:
    print("\nunsupported parameters are refused, not ignored")
    base = {"model": MODEL, "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}]}
    for extra, why in [
        ({"temperature": 0.5}, "temperature"),
        ({"top_p": 0.5}, "top_p"),
        ({"top_k": 20}, "top_k"),
        ({"stop_sequences": ["\n"]}, "stop_sequences"),
        ({"tools": [{"name": "f", "input_schema": {}}]}, "tools"),
        ({"tool_choice": {"type": "any"}}, "tool_choice"),
        ({"thinking": {"type": "enabled", "budget_tokens": 1024}}, "thinking"),
        ({"output_config": {"effort": "high"}}, "output_config"),
        ({"container": {"id": "x"}}, "container"),
        ({"cache_control": {"type": "ephemeral"}}, "cache_control"),
    ]:
        status, body, _ = post(args, "/v1/messages", {**base, **extra})
        ok = status == 400 and is_anthropic_error(body) and why in body["error"]["message"]
        check(f"rejects {why}", ok, f"status {status} body {body}")

    check("the temperature refusal points at the extension",
          "crucible_temperature" in post(args, "/v1/messages",
                                         {**base, "temperature": 0.5})[1]["error"]["message"], "")

    print("\n  no-op values are still accepted")
    for extra in [{"stop_sequences": []}, {"tools": []}, {"metadata": {"user_id": "u1"}},
                  {"service_tier": "auto"}]:
        status, _, _ = post(args, "/v1/messages", {**base, **extra})
        check(f"accepts {list(extra)[0]} at its no-op value", status == 200, f"status {status}")

    print("\n  unsupported content")
    for messages, why in [
        ([{"role": "user", "content": [{"type": "image", "source": {"type": "base64"}}]}], "image block"),
        ([{"role": "system", "content": "x"}], "system role in messages"),
        ([{"role": "tool", "content": "x"}], "tool role"),
        ([], "empty message list"),
    ]:
        status, body, _ = post(args, "/v1/messages",
                               {"model": MODEL, "max_tokens": 8, "messages": messages})
        check(f"{why} is refused", status == 400 and is_anthropic_error(body), f"{status} {body}")

    print("\n  request-level errors")
    status, body, _ = post(args, "/v1/messages", "{not json", raw=True)
    check("malformed JSON is 400 in the Anthropic shape",
          status == 400 and is_anthropic_error(body), f"{status} {body}")
    for missing, field in [({"max_tokens": 8, "messages": [{"role": "user", "content": "x"}]}, "model"),
                           ({"model": MODEL, "messages": [{"role": "user", "content": "x"}]}, "max_tokens"),
                           ({"model": MODEL, "max_tokens": 8}, "messages")]:
        status, body, _ = post(args, "/v1/messages", missing)
        check(f"missing {field} is 400",
              status == 400 and is_anthropic_error(body) and field in body["error"]["message"],
              f"{status} {body}")
    status, body, _ = post(args, "/v1/messages", {**base, "max_tokens": 0})
    check("max_tokens of 0 is 400", status == 400 and is_anthropic_error(body), f"{status} {body}")
    status, body, _ = post(args, "/v1/messages", {**base, "model": "claude-opus-4"})
    check("a foreign model is 404 with not_found_error",
          status == 404 and body["error"]["type"] == "not_found_error", f"{status} {body}")
    status, body, _ = post(args, "/v1/messages",
                           {"model": MODEL, "max_tokens": 500,
                            "messages": [{"role": "user", "content": "word " * 600}]})
    check("context overflow is 400 invalid_request_error",
          status == 400 and body["error"]["type"] == "invalid_request_error", f"{status} {body}")
    check("every error carries its own request_id",
          post(args, "/v1/messages", {**base, "max_tokens": 0})[1]["request_id"]
          != post(args, "/v1/messages", {**base, "max_tokens": 0})[1]["request_id"], "")

    print("\n  version negotiation")
    status, _, _ = post(args, "/v1/messages", base, headers={"Content-Type": "application/json"})
    check("an absent anthropic-version is accepted", status == 200, str(status))
    status, body, _ = post(args, "/v1/messages", base,
                           headers={"Content-Type": "application/json",
                                    "anthropic-version": "2099-01-01"})
    check("an unknown anthropic-version is refused",
          status == 400 and is_anthropic_error(body), f"{status} {body}")
    check("server healthy after every rejection", get(args, "/health")[0] == 200)


# ------------------------------------------------------------ models API ----


def test_models(args) -> None:
    print("\nmodels, and the two-protocol collision")
    status, oai = get(args, "/v1/models")
    check("without the version header /v1/models is unchanged OpenAI",
          status == 200 and oai.get("object") == "list"
          and oai["data"][0].get("object") == "model"
          and "owned_by" in oai["data"][0], str(oai))

    status, ant = get(args, "/v1/models", headers={"anthropic-version": VERSION})
    check("with the version header it is the Anthropic page",
          status == 200 and "data" in ant and "has_more" in ant
          and ant["data"][0].get("type") == "model", str(ant))
    m = ant["data"][0]
    check("the Anthropic model object has its own fields",
          all(k in m for k in ("id", "type", "display_name", "created_at")), str(m))
    check("created_at is RFC 3339",
          isinstance(m.get("created_at"), str) and m["created_at"].endswith("Z")
          and len(m["created_at"]) == 20, str(m.get("created_at")))
    check("both shapes report the same model id",
          m["id"] == oai["data"][0]["id"] == MODEL, "")

    status, one = get(args, f"/v1/models/{MODEL}", headers={"anthropic-version": VERSION})
    check("retrieving the model works in the Anthropic shape",
          status == 200 and one.get("type") == "model", f"{status} {one}")
    status, one = get(args, f"/v1/models/{MODEL}")
    check("retrieving it without the header keeps the OpenAI shape",
          status == 200 and one.get("object") == "model", f"{status} {one}")
    status, body = get(args, "/v1/models/claude-opus-4", headers={"anthropic-version": VERSION})
    check("an unknown model is 404 in the Anthropic shape",
          status == 404 and is_anthropic_error(body), f"{status} {body}")


# ----------------------------------------------------------- determinism ----


def test_determinism(args) -> None:
    print("\ncross-protocol determinism")
    # The comparison is on the serialized prompt, not the raw JSON: Anthropic
    # and OpenAI describe the same conversation differently and the point is
    # that both reach the same prompt.
    serialized = "System: Be terse.\n\nUser: Hello\n\nAssistant:"
    native = post(args, "/v1/generate", {"prompt": serialized, "max_tokens": 16},
                  headers={"Content-Type": "application/json"})[1]["text"]

    ant = post(args, "/v1/messages",
               {"model": MODEL, "max_tokens": 16, "system": "Be terse.",
                "messages": [{"role": "user", "content": "Hello"}]})[1]["content"][0]["text"]
    check("Anthropic matches the native endpoint for the serialized prompt",
          ant == native, f"\n    anthropic={ant!r}\n    native   ={native!r}")

    oai = post(args, "/v1/chat/completions",
               {"model": MODEL, "max_tokens": 16,
                "messages": [{"role": "system", "content": "Be terse."},
                             {"role": "user", "content": "Hello"}]},
               headers={"Content-Type": "application/json"})[1]["choices"][0]["message"]["content"]
    check("Anthropic and OpenAI agree for the same conversation", ant == oai,
          f"\n    anthropic={ant!r}\n    openai   ={oai!r}")

    _, events, _ = sse(args, {"model": MODEL, "max_tokens": 16, "system": "Be terse.",
                              "messages": [{"role": "user", "content": "Hello"}],
                              "stream": True})
    check("streaming matches non-streaming", deltas(events) == ant,
          f"\n    stream={deltas(events)!r}\n    plain ={ant!r}")

    print("\n  the Crucible sampling extension")
    sreq = {"model": MODEL, "max_tokens": 24,
            "messages": [{"role": "user", "content": "Once upon a time"}],
            "crucible_temperature": 0.8, "crucible_top_k": 40, "crucible_seed": 4242}
    a = post(args, "/v1/messages", sreq)[1]["content"][0]["text"]
    b = post(args, "/v1/messages", sreq)[1]["content"][0]["text"]
    check("a seeded sampled message reproduces", a == b, f"{a!r} vs {b!r}")
    c = post(args, "/v1/messages", {**sreq, "crucible_seed": 99})[1]["content"][0]["text"]
    check("a different seed gives different text", a != c, "seeds collided")
    greedy = post(args, "/v1/messages",
                  {"model": MODEL, "max_tokens": 24,
                   "messages": [{"role": "user", "content": "Once upon a time"}]})[1]
    check("sampled output differs from the greedy default",
          a != greedy["content"][0]["text"], "sampling produced the greedy sequence")
    nat = post(args, "/v1/generate",
               {"prompt": "User: Once upon a time\n\nAssistant:", "max_tokens": 24,
                "temperature": 0.8, "top_k": 40, "seed": 4242},
               headers={"Content-Type": "application/json"})[1]["text"]
    check("the extension matches native sampling for the same seed", a == nat,
          f"\n    anthropic={a!r}\n    native   ={nat!r}")
    _, events, _ = sse(args, {**sreq, "stream": True})
    check("a seeded sampled stream matches its non-streamed twin",
          deltas(events) == a, f"\n    stream={deltas(events)!r}\n    plain ={a!r}")
    check("omitting the extension is deterministic (greedy)",
          post(args, "/v1/messages", {"model": MODEL, "max_tokens": 8,
                                      "messages": [{"role": "user", "content": "Hello"}]}
               )[1]["content"][0]["text"]
          == post(args, "/v1/messages", {"model": MODEL, "max_tokens": 8,
                                         "messages": [{"role": "user", "content": "Hello"}]}
                  )[1]["content"][0]["text"], "")


# ------------------------------------------------------------- lifecycle ----


def test_cancellation(args) -> None:
    print("\ncancellation")
    before = metrics(args)
    total_pages = before["kv_pages_used"] + before["kv_pages_free"]
    status, events, _ = sse(args, {"model": MODEL, "max_tokens": 400,
                                   "messages": [{"role": "user", "content": "Once upon a time"}],
                                   "stream": True}, stop_after=5)
    check("an early disconnect still delivered partial events", len(events) == 5, str(len(events)))
    check("no message_stop was sent to a client that left",
          "message_stop" not in [n for n, _ in events], "")

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
    check("no request failed as a side effect",
          m["failed_requests"] == before["failed_requests"], str(m))
    check("the next request still succeeds",
          post(args, "/v1/messages", {"model": MODEL, "max_tokens": 4,
                                      "messages": [{"role": "user", "content": "hi"}]})[0] == 200)


def test_backpressure(args) -> None:
    print("\nbackpressure")
    statuses, kinds = [], []
    lock = threading.Lock()

    def hammer():
        s, b, _ = post(args, "/v1/messages",
                       {"model": MODEL, "max_tokens": 400,
                        "messages": [{"role": "user", "content": "The"}]})
        with lock:
            statuses.append(s)
            if s != 200:
                kinds.append(b.get("error", {}).get("type"))

    threads = [threading.Thread(target=hammer) for _ in range(120)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    check("every overload response was 200 or 529",
          all(s in (200, 529) for s in statuses), str(sorted(set(statuses))))
    if 529 in statuses:
        check("overload used Anthropic's 529 overloaded_error, not OpenAI's 429",
              all(k == "overloaded_error" for k in kinds), str(set(kinds)))
    else:
        print("  note  queue never filled; the 529 path was not exercised this run")
    deadline = time.time() + 30
    while time.time() < deadline and metrics(args)["active_requests"] > 0:
        time.sleep(0.1)
    m = metrics(args)
    check("all pages returned after the overload", m["kv_pages_used"] == 0, str(m))
    check("server healthy after the overload", get(args, "/health")[0] == 200)


def test_batching(args) -> None:
    print("\ncontinuous batching across every protocol")
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
    done: dict = {}

    def native(i):
        c = conn(args)
        c.request("POST", "/v1/generate/stream",
                  json.dumps({"prompt": "The history of the printing press begins in",
                              "max_tokens": 200}),
                  {"Content-Type": "application/json"})
        r = c.getresponse()
        done[i] = sum(1 for raw in r if raw.startswith(b"data:"))
        c.close()

    def openai_chat(i):
        c = conn(args)
        c.request("POST", "/v1/chat/completions",
                  json.dumps({"model": MODEL, "max_tokens": 200, "stream": True,
                              "messages": [{"role": "user", "content": "Tell me about rivers"}]}),
                  {"Content-Type": "application/json"})
        r = c.getresponse()
        done[i] = sum(1 for raw in r if raw.startswith(b"data:"))
        c.close()

    def anthropic_msg(i):
        done[i] = len(sse(args, {"model": MODEL, "max_tokens": 200, "stream": True,
                                 "messages": [{"role": "user", "content": "Explain CUDA graphs."}]})[1])

    workers = []
    for i in range(9):
        fn = [native, openai_chat, anthropic_msg][i % 3]
        workers.append(threading.Thread(target=fn, args=(i,)))
    for t in workers:
        t.start()
        time.sleep(0.005)
    for t in workers:
        t.join()
    stop.set()
    w.join(timeout=1)

    check("every client completed", len(done) == 9, str(len(done)))
    check("native, OpenAI and Anthropic clients shared decode steps",
          peak["batch"] > 1, f"peak observed batch {peak['batch']}")
    check("nothing was left active", metrics(args)["active_requests"] == 0, "")


# ---------------------------------------------------- streaming torture -----


def raw_sse(args, body, chunk_size: int) -> bytes:
    s = socket.create_connection((args.host, args.port), timeout=180)
    payload = json.dumps(body).encode()
    s.sendall((
        f"POST /v1/messages HTTP/1.1\r\nHost: {args.host}\r\n"
        f"Content-Type: application/json\r\nanthropic-version: {VERSION}\r\n"
        f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n"
    ).encode() + payload)
    out = b""
    while True:
        b = s.recv(chunk_size)
        if not b:
            break
        out += b
    s.close()
    return out


def parse_raw(raw: bytes):
    _, _, body = raw.partition(b"\r\n\r\n")
    events, name = [], None
    for line in body.decode("utf-8", "replace").split("\n"):
        line = line.strip()
        if line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            try:
                events.append((name, json.loads(line[5:].strip())))
            except json.JSONDecodeError:
                pass
    return events


def test_torture(args) -> None:
    print("\nstreaming parser torture")
    body = {"model": MODEL, "max_tokens": 8, "stream": True,
            "messages": [{"role": "user", "content": "Hello"}]}
    reference = None
    for size in (1, 3, 64, 65536):
        events = parse_raw(raw_sse(args, body, size))
        text = deltas(events)
        names = [n for n, _ in events]
        if reference is None:
            reference = text
        label = "one byte at a time" if size == 1 else f"{size}-byte reads"
        check(f"stream survives {label}",
              text == reference and names[-1] == "message_stop" and text != "",
              f"text={text!r} ref={reference!r} last={names[-1] if names else None}")

    status, events, err = sse(args, {**body, "temperature": 0.5})
    check("a rejected streaming request errors before any event",
          status == 400 and not events and is_anthropic_error(err or {}), f"{status} {err}")

    # Disconnecting right after message_start must not wedge anything.
    sse(args, {**body, "max_tokens": 400}, stop_after=1)
    deadline = time.time() + 20
    while time.time() < deadline and metrics(args)["active_requests"] > 0:
        time.sleep(0.05)
    check("disconnecting after message_start reclaims cleanly",
          metrics(args)["active_requests"] == 0 and get(args, "/health")[0] == 200, "")


# ------------------------------------------------------------------- SDK ----

SDK_SCRIPT = r'''
import json, sys
import anthropic

base, model = sys.argv[1], sys.argv[2]
client = anthropic.Anthropic(base_url=base, api_key="not-used")
out = {"version": anthropic.__version__}

m = client.messages.create(model=model, max_tokens=12,
                           messages=[{"role": "user", "content": "Hello"}])
out["text"] = m.content[0].text
out["role"] = m.role
out["type"] = m.type
out["stop_reason"] = m.stop_reason
out["stop_sequence"] = m.stop_sequence
out["usage"] = [m.usage.input_tokens, m.usage.output_tokens]
out["request_id"] = m._request_id
out["model"] = m.model

with client.messages.stream(model=model, max_tokens=12,
                            messages=[{"role": "user", "content": "Hello"}]) as s:
    out["stream_text"] = "".join(s.text_stream)
    final = s.get_final_message()
out["final_text"] = final.content[0].text
out["final_usage"] = [final.usage.input_tokens, final.usage.output_tokens]
out["final_stop"] = final.stop_reason

sysmsg = client.messages.create(model=model, max_tokens=12, system="Be terse.",
                                messages=[{"role": "user", "content": "Hello"}])
out["system_text"] = sysmsg.content[0].text
out["system_input_tokens"] = sysmsg.usage.input_tokens

multi = client.messages.create(model=model, max_tokens=12, messages=[
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "hello"},
    {"role": "user", "content": "again"}])
out["multi_turn"] = bool(multi.content[0].text)

out["count_tokens"] = client.messages.count_tokens(
    model=model, messages=[{"role": "user", "content": "Hello"}]).input_tokens
out["count_tokens_system"] = client.messages.count_tokens(
    model=model, system="Be terse.",
    messages=[{"role": "user", "content": "Hello"}]).input_tokens

out["models"] = [x.id for x in client.models.list()]
out["model_retrieve"] = client.models.retrieve(model).id

for label, fn in [
    ("bad_model", lambda: client.messages.create(
        model="claude-opus-4", max_tokens=4,
        messages=[{"role": "user", "content": "hi"}])),
    ("bad_request", lambda: client.messages.create(
        model=model, max_tokens=4,
        messages=[{"role": "user", "content": "hi"}], stop_sequences=["\n"])),
    ("bad_model_retrieve", lambda: client.models.retrieve("claude-opus-4")),
]:
    try:
        fn()
        out[label] = "no error"
    except Exception as e:
        out[label] = type(e).__name__

print(json.dumps(out))
'''


def test_sdk(args) -> None:
    if not args.sdk:
        print("\nofficial Anthropic SDK\n  skipped (no --sdk interpreter given)")
        return
    print("\nofficial Anthropic SDK")
    base = f"http://{args.host}:{args.port}"
    proc = subprocess.run([args.sdk, "-c", SDK_SCRIPT, base, MODEL],
                          capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        check("SDK script ran", False, proc.stderr[-1500:])
        return
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    print(f"  ...against anthropic-python {out['version']}")
    check("SDK non-streaming returned an assistant message",
          out["role"] == "assistant" and out["type"] == "message" and bool(out["text"]), str(out)[:200])
    check("SDK model id round-trips", out["model"] == MODEL, str(out["model"]))
    check("SDK stop_reason is max_tokens", out["stop_reason"] == "max_tokens", str(out["stop_reason"]))
    check("SDK stop_sequence is null", out["stop_sequence"] is None, str(out["stop_sequence"]))
    p, c = out["usage"]
    check("SDK parsed usage", p > 0 and c == 12, str(out["usage"]))
    check("SDK exposes a request id",
          isinstance(out["request_id"], str) and out["request_id"].startswith("req_"),
          str(out["request_id"]))
    check("SDK streamed text equals non-streamed",
          out["stream_text"] == out["text"], f"{out['stream_text']!r} vs {out['text']!r}")
    check("SDK final accumulated message equals the streamed text",
          out["final_text"] == out["stream_text"], "")
    check("SDK final message reports the same usage as non-streaming",
          out["final_usage"] == out["usage"], f"{out['final_usage']} vs {out['usage']}")
    check("SDK final message reports max_tokens", out["final_stop"] == "max_tokens", "")
    check("SDK system instruction works",
          bool(out["system_text"]) and out["system_input_tokens"] > p,
          f"{out['system_input_tokens']} vs {p}")
    check("SDK multi-turn conversation works", out["multi_turn"], "")
    check("SDK count_tokens matches the reported input tokens",
          out["count_tokens"] == p, f"{out['count_tokens']} vs {p}")
    check("SDK count_tokens tracks the system prompt",
          out["count_tokens_system"] == out["system_input_tokens"],
          f"{out['count_tokens_system']} vs {out['system_input_tokens']}")
    check("SDK models.list works", out["models"] == [MODEL], str(out["models"]))
    check("SDK models.retrieve works", out["model_retrieve"] == MODEL, str(out["model_retrieve"]))
    check("SDK raises NotFoundError for a foreign model",
          out["bad_model"] == "NotFoundError", str(out["bad_model"]))
    check("SDK raises NotFoundError retrieving a foreign model",
          out["bad_model_retrieve"] == "NotFoundError", str(out["bad_model_retrieve"]))
    check("SDK raises BadRequestError for an unsupported parameter",
          out["bad_request"] == "BadRequestError", str(out["bad_request"]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--sdk", help="python interpreter with the anthropic package installed")
    args = p.parse_args()

    test_messages(args)
    test_system_and_turns(args)
    test_count_tokens(args)
    test_multibyte(args)
    test_rejections(args)
    test_models(args)
    test_determinism(args)
    test_torture(args)
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
