"""Compare packed service results with frozen, independent reference results.

Capture against CRUCIBLE_BATCHED_PREFILL=0, then restart with packed prefill:
    python scripts/test_packed_prefill.py --record-reference /tmp/reference.json
    python scripts/test_packed_prefill.py --reference /tmp/reference.json

Use the same seed/steps/fuzz-rounds for both invocations. The fixture includes
every request body and token ID, so the reference server need not occupy GPU
memory during candidate testing. Run on a dedicated server: reclamation and
accounting assertions intentionally require no unrelated traffic.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
from pathlib import Path
import random
import time

from bench_packed_prefill import WORDS, burst, json_request, prompt_of, stream


BOUNDARIES = (1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129,
              255, 256, 257, 511, 512, 941)
POLICIES = ((0.0, 40), (0.9, 5), (0.8, 40), (0.7, 128), (0.8, 500))


def corpus(seed: int, steps: int, fuzz_rounds: int) -> tuple[dict, list[list[str]]]:
    rng = random.Random(seed)
    requests, groups = {}, []
    for index, length in enumerate(BOUNDARIES):
        temperature, top_k = POLICIES[index % len(POLICIES)]
        requests[f"boundary-{length}"] = {
            "prompt": prompt_of(length, index), "max_tokens": steps,
            "temperature": temperature, "top_k": top_k, "seed": seed + index}
    for group in range(fuzz_rounds):
        keys = []
        for index in range(rng.randint(2, 16)):
            length = rng.choice(BOUNDARIES[:-1]) if index % 2 else rng.randint(1, 512)
            temperature, top_k = rng.choice(POLICIES)
            key = f"fuzz-{group}-{index}"
            requests[key] = {
                "prompt": "".join(rng.choice(WORDS) for _ in range(length - 1)) + " x",
                "max_tokens": steps, "temperature": temperature, "top_k": top_k,
                "seed": rng.getrandbits(32)}
            keys.append(key)
        groups.append(keys)
    for body in requests.values():
        if body["temperature"] <= 0:
            for field in ("temperature", "top_k", "seed"):
                del body[field]
    return requests, groups


def await_idle(host: str, port: int) -> dict:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        metrics = json_request(host, port, "/metrics")
        if not any(metrics.get(key, 0) for key in
                   ("active_requests", "queued_requests", "prefilling_requests", "kv_pages_used")):
            return metrics
        time.sleep(0.01)
    raise AssertionError(f"server did not become idle/reclaim all pages: {metrics}")


def compare(key: str, actual: dict, reference: dict) -> None:
    for field in ("tokens", "text"):
        if actual[field] != reference[field]:
            raise AssertionError(f"{key}: {field} differs\nexpected {reference[field]!r}\ngot {actual[field]!r}")
    for field in ("tokens_generated", "finish_reason"):
        if actual["done"][field] != reference["done"][field]:
            raise AssertionError(f"{key}: completion {field} differs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--record-reference", type=Path)
    mode.add_argument("--reference", type=Path)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--fuzz-rounds", type=int, default=12)
    parser.add_argument("--allow-reference-path", action="store_true",
                        help="allow zero packed batches for harness self-check against baseline")
    args = parser.parse_args()
    if args.steps < 1 or args.fuzz_rounds < 0:
        parser.error("steps must be positive and fuzz-rounds nonnegative")
    requests, fuzz_groups = corpus(args.seed, args.steps, args.fuzz_rounds)
    health = json_request(args.host, args.port, "/health")
    idle = await_idle(args.host, args.port)
    metadata = {"seed": args.seed, "steps": args.steps, "fuzz_rounds": args.fuzz_rounds,
                "model": health["model"], "context": health["context"]}
    if args.record_reference:
        results = {}
        for index, (key, body) in enumerate(requests.items()):
            result = stream(args.host, args.port, body)
            # Non-streamed usage is independently checked against the stream;
            # the native SSE completion intentionally carries no prompt count.
            plain = json_request(args.host, args.port, "/v1/generate", body)
            if plain["text"] != result["text"] or plain["tokens_generated"] != len(result["tokens"]):
                raise AssertionError(f"{key}: reference streaming/nonstreaming mismatch")
            expected_length = body["prompt"].count(" ")
            if plain["prompt_tokens"] != expected_length:
                raise AssertionError(f"{key}: expected {expected_length} prompt tokens, got {plain['prompt_tokens']}")
            results[key] = {"tokens": result["tokens"], "text": result["text"],
                            "done": result["done"], "prompt_tokens": plain["prompt_tokens"]}
            if (index + 1) % 16 == 0:
                print(f"reference: {index + 1}/{len(requests)} requests", flush=True)
        await_idle(args.host, args.port)
        args.record_reference.write_text(json.dumps({"metadata": metadata, "requests": requests,
                                                      "results": results}, indent=2), encoding="utf-8")
        print(f"Recorded {len(results)} independent reference sequences in {args.record_reference}")
        return

    fixture = json.loads(args.reference.read_text(encoding="utf-8"))
    if fixture["metadata"] != metadata or fixture["requests"] != requests:
        raise AssertionError("fixture configuration/corpus differs; use the recording seed/steps/fuzz-rounds")
    references = fixture["results"]
    comparisons, runs = 0, 0

    def check_group(label: str, keys: list[str], stagger: bool = False) -> None:
        nonlocal comparisons, runs
        delays = [index * 0.003 for index in range(len(keys))] if stagger else None
        result = burst(args.host, args.port, [requests[key] for key in keys], delays)
        for key, actual in zip(keys, result["requests"]):
            compare(f"{label}/{key}", actual, references[key])
            comparisons += 1
        await_idle(args.host, args.port)
        runs += 1
        print(f"ok {label}: {len(keys)} exact token sequences", flush=True)

    boundary_keys = [f"boundary-{length}" for length in BOUNDARIES]
    # Both groups cross page boundaries; overlapping length941 with the first
    # fifteen gives long/short and mixed-final chunks in one scheduler run.
    for group_index, keys in enumerate((boundary_keys[:16], boundary_keys[3:])):
        for permutation_index, ordered in enumerate((keys, keys[::-1], keys[1::2] + keys[::2])):
            check_group(f"boundaries-{group_index}/permutation-{permutation_index}", ordered)
    composition = ["boundary-941"] + boundary_keys[:15]
    for count in (1, 2, 4, 8, 16):
        check_group(f"composition-{count}", composition[:count])
    for count in (16, 1, 8, 2):
        check_group(f"stale-metadata-{count}", boundary_keys[:count][::-1])
    for index, keys in enumerate(fuzz_groups):
        check_group(f"fuzz-{index}", keys, stagger=index % 2 == 1)

    def invalid_requests() -> None:
        cases = ("{broken", {"prompt": "", "max_tokens": 1},
                 {"prompt": " x", "max_tokens": 10**9},
                 {"prompt": " x", "temperature": -1.0, "max_tokens": 1})
        for index in range(32):
            connection = http.client.HTTPConnection(args.host, args.port, timeout=30)
            try:
                body = cases[index % len(cases)]
                connection.request("POST", "/v1/generate",
                                   body if isinstance(body, str) else json.dumps(body),
                                   {"Content-Type": "application/json"})
                response = connection.getresponse()
                response.read()
                if not 400 <= response.status < 500:
                    raise AssertionError(f"malformed request returned {response.status}")
            finally:
                connection.close()
            time.sleep(0.001)

    with ThreadPoolExecutor(max_workers=1) as pool:
        invalid = pool.submit(invalid_requests)
        check_group("invalid-request-isolation", ["boundary-941", "boundary-512", "boundary-257"])
        invalid.result()

    # Cancel an accepted stream with packed neighbours, let all survivors finish,
    # and immediately reuse its pages. Exact cancellation timing is a scheduler
    # boundary property covered by the GPU harness; HTTP can only control when
    # the socket is closed. Both pre-token and after-token disconnects are used.
    for cancel_after in (0, 1):
        before = await_idle(args.host, args.port)
        victim = {"prompt": prompt_of(941, 7), "max_tokens": 64}
        keys = ["boundary-511", "boundary-129", "boundary-257"]
        with ThreadPoolExecutor(max_workers=2) as pool:
            cancelled = pool.submit(stream, args.host, args.port, victim, None, None, 0.0, cancel_after)
            survivors = pool.submit(burst, args.host, args.port, [requests[key] for key in keys])
            cancelled.result()
            result = survivors.result()
        for key, actual in zip(keys, result["requests"]):
            compare(f"cancel-{cancel_after}/{key}", actual, references[key])
            comparisons += 1
        after = await_idle(args.host, args.port)
        if after["cancelled_requests"] <= before["cancelled_requests"]:
            raise AssertionError(f"disconnect after {cancel_after} tokens did not cancel a live request")
        check_group(f"cancel-{cancel_after}/page-reuse", ["boundary-941"] + keys)

    final = await_idle(args.host, args.port)
    if final["kv_pages_free"] != idle["kv_pages_free"]:
        raise AssertionError("KV page count changed after the workload")
    if final["failed_requests"] != idle["failed_requests"]:
        raise AssertionError("server recorded failed requests")
    packed = final.get("packed_prefill_batches", 0) - idle.get("packed_prefill_batches", 0)
    slices = final.get("prefill_requests", 0) - idle.get("prefill_requests", 0)
    batches = final.get("prefill_batches", 0) - idle.get("prefill_batches", 0)
    if not args.allow_reference_path and (packed <= 0 or slices <= batches):
        raise AssertionError("no packed GPU batch was recorded; concurrency alone does not prove packing")
    print(f"PASS: {comparisons} exact comparisons, {runs} workload runs, "
          f"{args.fuzz_rounds} deterministic fuzz sets, {packed} packed batches; every page reclaimed")


if __name__ == "__main__":
    main()
