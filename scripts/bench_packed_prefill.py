"""Portable real-service prefill benchmark; standard library only.

Run against one warmed server at a time, alternate reference/candidate ordering
between rounds, and retain the JSON from every round. No best-run selection.
Prompt token counts are checked through the server before measurement.

    python scripts/bench_packed_prefill.py --port 8080 --trials 3 --output run.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
from pathlib import Path
import statistics
import subprocess
import threading
import time


WORDS = (" red", " blue", " green", " white", " black", " gold", " cold", " warm")
SAMPLING_POLICIES = ((0.0, None), (0.9, 5), (0.8, 40), (0.7, 128), (0.8, 500))
MIXED_LENGTHS = (8, 17, 33, 64, 127, 256, 512, 941)
HOL_LENGTHS = (33, 127, 535, 941)


def sampling_fields(index: int, protocol: str) -> dict:
    """Cycle independently of the four adapters, including full-logit fallback."""
    temperature, top_k = SAMPLING_POLICIES[index % len(SAMPLING_POLICIES)]
    if top_k is None:
        return {}
    prefix = "crucible_" if protocol == "anthropic" else ""
    return {f"{prefix}temperature": temperature, f"{prefix}top_k": top_k,
            f"{prefix}seed": 20260905 + index}


def prompt_of(length: int, variant: int = 0) -> str:
    # Each leading-space word is one GPT-2 token; the server verifies this.
    # The common last token and unrelated prefixes make leakage more visible.
    return WORDS[variant % len(WORDS)] * (length - 1) + " x"


def json_request(host: str, port: int, path: str, body=None, headers=None) -> dict:
    conn = http.client.HTTPConnection(host, port, timeout=180)
    try:
        conn.request("GET" if body is None else "POST", path,
                     None if body is None else json.dumps(body),
                     {"Content-Type": "application/json", **(headers or {})})
        response = conn.getresponse()
        data = response.read()
        if response.status != 200:
            raise RuntimeError(f"{path}: HTTP {response.status}: {data[:300]!r}")
        return json.loads(data)
    finally:
        conn.close()


def stream(host: str, port: int, body: dict, barrier=None, ready=None,
           delay: float = 0.0, cancel_after: int | None = None, cancel_event=None,
           accepted_event=None) -> dict:
    """Record actual token IDs and arrival times, including queue/prefill wait."""
    conn = http.client.HTTPConnection(host, port, timeout=180)
    if barrier is not None:
        barrier.wait(timeout=30)
    if delay:
        time.sleep(delay)
    start = time.perf_counter()
    stamps, tokens, text = [], [], []
    done = None
    try:
        conn.request("POST", "/v1/generate/stream", json.dumps(body),
                     {"Content-Type": "application/json"})
        response = conn.getresponse()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {response.read()[:300]!r}")
        accepted = time.perf_counter()
        if accepted_event is not None:
            accepted_event.set()
        if cancel_after == 0:
            # Closing after accepted response headers exercises cancellation
            # without waiting for first-token generation.
            return {"cancelled": True, "tokens": [], "start": start, "accepted": accepted}
        event = None
        for raw in response:
            line = raw.decode("utf-8").strip()
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                payload = json.loads(line[5:].strip())
                if event == "token":
                    stamps.append(time.perf_counter())
                    tokens.append(payload["token_id"])
                    text.append(payload["text"])
                    if ready is not None and len(tokens) == 32:
                        ready.set()
                    if ((cancel_after is not None and len(tokens) >= cancel_after)
                            or (cancel_event is not None and cancel_event.is_set())):
                        return {"cancelled": True, "tokens": tokens, "start": start, "accepted": accepted}
                elif event == "done":
                    done = payload
                    text.append(payload.get("text", ""))
                    break
                elif event == "error":
                    raise RuntimeError(f"SSE error: {payload}")
        if done is None or not stamps:
            raise RuntimeError("stream ended without tokens and a done event")
        if done["tokens_generated"] != len(tokens):
            raise RuntimeError(f"usage mismatch: {done} vs {len(tokens)} events")
        return {"start": start, "accepted": accepted, "stamps": stamps, "tokens": tokens,
                "text": "".join(text), "done": done,
                "ttft_ms": (stamps[0] - start) * 1000,
                "total_ms": (time.perf_counter() - start) * 1000,
                "gaps_ms": [(b - a) * 1000 for a, b in zip(stamps, stamps[1:])]}
    finally:
        conn.close()


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    lower = int(index)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[min(lower + 1, len(ordered) - 1)] * fraction


def distribution(values: list[float]) -> dict:
    if not values:
        return {}
    return {"count": len(values), "median": statistics.median(values),
            "p95": percentile(values, 0.95),
            # A p99 over a tiny burst is merely its maximum in disguise.
            "p99": percentile(values, 0.99) if len(values) >= 100 else None,
            "min": min(values), "max": max(values)}


def gpu_envelope() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,enforced.power.limit,clocks.max.sm,clocks.sm,utilization.gpu,power.draw",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"unavailable: {error}"


class EnvelopeMonitor:
    """Sample loaded clocks while requests run, not just the idle preamble."""
    def __init__(self):
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            self.samples.append({"time": time.time(), "gpu": gpu_envelope()})
            self.stop.wait(0.5)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=11)


def metric_delta(before: dict, after: dict) -> dict:
    counters = ("prefill_tokens", "prefill_chunks", "prefill_batches", "prefill_requests",
                "packed_prefill_batches", "packed_prefill_tokens",
                "aggregate_tokens_generated", "decode_steps", "completed_requests",
                "cancelled_requests", "failed_requests", "prefill_final_rows",
                "greedy_requests", "sampled_requests", "prefill_d2h_bytes")
    return {key: after[key] - before[key] for key in counters if key in before and key in after}


def burst(host: str, port: int, bodies: list[dict], delays=None) -> dict:
    barrier = threading.Barrier(len(bodies))
    before = json_request(host, port, "/metrics")
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(bodies)) as pool:
        futures = [pool.submit(stream, host, port, body, barrier, None,
                               0.0 if delays is None else delays[i])
                   for i, body in enumerate(bodies)]
        results = [future.result() for future in futures]
    elapsed = time.perf_counter() - start
    after = json_request(host, port, "/metrics")
    return {"requests": results, "wall_s": elapsed,
            "aggregate_tok_s": sum(len(r["tokens"]) for r in results) / elapsed,
            "metrics_delta": metric_delta(before, after)}


def interruption(host: str, port: int, length: int, existing_tokens: int) -> dict:
    ready = threading.Event()
    before = json_request(host, port, "/metrics")
    with ThreadPoolExecutor(max_workers=1) as pool:
        existing = pool.submit(stream, host, port,
                               {"prompt": prompt_of(8), "max_tokens": existing_tokens}, None, ready)
        if not ready.wait(timeout=30):
            raise RuntimeError("established stream did not reach 32 tokens")
        arrival = time.perf_counter()
        intruder = stream(host, port, {"prompt": prompt_of(length, 1), "max_tokens": 8})
        result = existing.result()
    # End the disturbance window after the intruder's last token. It includes
    # the gap that straddles arrival, and excludes the established admission.
    end = intruder["stamps"][-1]
    gaps = [(b - a) * 1000 for a, b in zip(result["stamps"], result["stamps"][1:])
            if arrival <= b <= end]
    if not gaps or result["stamps"][-1] < intruder["stamps"][0]:
        raise RuntimeError("existing stream ended before the disturbance; increase --existing-tokens")
    return {"intruder": intruder, "existing": result,
            "existing_disturbance_gap_ms": distribution(gaps),
            "metrics_delta": metric_delta(before, json_request(host, port, "/metrics"))}


def head_of_line(host: str, port: int, existing_tokens: int) -> dict:
    """Measure one established decoder throughout four simultaneous prefills."""
    ready = threading.Event()
    before = json_request(host, port, "/metrics")
    with ThreadPoolExecutor(max_workers=1) as pool:
        existing = pool.submit(stream, host, port,
                               {"prompt": prompt_of(8), "max_tokens": existing_tokens}, None, ready)
        if not ready.wait(timeout=30):
            raise RuntimeError("established stream did not reach 32 tokens")
        intruders = burst(host, port,
                          [{"prompt": prompt_of(length, i + 1), "max_tokens": 64}
                           for i, length in enumerate(HOL_LENGTHS)])
        result = existing.result()
    arrival = min(request["start"] for request in intruders["requests"])
    end = max(request["stamps"][-1] for request in intruders["requests"])
    if result["stamps"][-1] < end:
        raise RuntimeError("existing stream ended before all HOL intruders drained; increase --existing-tokens")
    # Include both boundary-straddling gaps so neither the initial disturbance
    # nor the final recovery gap disappears from the measured window.
    gaps = [(b - a) * 1000 for a, b in zip(result["stamps"], result["stamps"][1:])
            if b >= arrival and a <= end]
    if not gaps:
        raise RuntimeError("HOL disturbance did not overlap established decoding")
    gap_summary = distribution(gaps)
    # This explicit small-window p99 is descriptive, with sample count retained;
    # it does not estimate a population tail from a short burst.
    gap_summary.update(p50=percentile(gaps, 0.50), p99=percentile(gaps, 0.99), worst=max(gaps))
    return {"prompt_lengths": list(HOL_LENGTHS), "intruders": intruders["requests"],
            "existing": result, "disturbance_start": arrival, "disturbance_end": end,
            "existing_disturbance_gaps_ms": gaps, "existing_disturbance_gap_ms": gap_summary,
            "metrics_delta": metric_delta(before, json_request(host, port, "/metrics"))}


def tui_overlap_probe(host: str, port: int, binary: str) -> dict:
    """Identify a real TUI prompt in a packed batch using exact token counts.

    Imports POSIX terminal support only when explicitly requested. The ordinary
    benchmark and pressure/correctness scripts remain usable on Windows.
    Requires the 941-token blocker to fit in a single prefill slice: its size
    distinguishes it from the five short client prompts in the packed batch.
    """
    from smoke_tui import Tui

    idle_keys = ("active_requests", "queued_requests", "prefilling_requests", "kv_pages_used")
    expected = {"failed_requests": 0, "prefill_requests": 6,
                "prefill_chunks": 6, "prefill_final_rows": 6, "prefill_batches": 2,
                "packed_prefill_batches": 1}
    model = json_request(host, port, "/v1/models")["data"][0]["id"]

    def compatible(protocol: str) -> dict:
        body = {"model": model, "max_tokens": 128}
        prompt = prompt_of(16, 3)
        if protocol == "openai-completions":
            body["prompt"] = prompt
            response = json_request(host, port, "/v1/completions", body)
        else:
            body["messages"] = [{"role": "user", "content": prompt}]
            response = json_request(host, port,
                "/v1/messages" if protocol == "anthropic" else "/v1/chat/completions", body,
                {"anthropic-version": "2023-06-01"} if protocol == "anthropic" else None)
        usage = response["usage"]
        return {"protocol": protocol,
                "prompt_tokens": usage["input_tokens" if protocol == "anthropic" else "prompt_tokens"],
                "generated_tokens": usage["output_tokens" if protocol == "anthropic" else "completion_tokens"]}
    # The TUI sends its input verbatim to the native stream endpoint. Verify
    # these exact text variants before the isolated six-request window.
    for length, variant in ((941, 2), (16, 1), (16, 0)):
        actual = json_request(host, port, "/v1/generate",
                              {"prompt": prompt_of(length, variant), "max_tokens": 1})["prompt_tokens"]
        if actual != length:
            raise AssertionError(f"TUI probe tokenizer mismatch: expected {length}, got {actual}")
    attempts = []
    # Arrival scheduling is nondeterministic. These are bounded functional
    # retries, not benchmark samples; preserve every unsuccessful attempt.
    for attempt in range(1, 11):
        initial = json_request(host, port, "/metrics")
        if any(initial.get(key, 0) for key in idle_keys):
            raise AssertionError("TUI participation probe requires an idle server")
        tui = Tui(binary, f"http://{host}:{port}")
        samples = []
        try:
            if not tui.wait_for("connected", 20):
                raise RuntimeError(f"TUI did not connect: {tui.screen()[-300:]}")
            # Queue only Enter during the blocker: typing the prompt itself
            # should not consume the short interval in which peers can arrive.
            tui.send(prompt_of(16, 1))
            time.sleep(0.1)
            accepted = threading.Event()
            with ThreadPoolExecutor(max_workers=5) as pool:
                blocker = pool.submit(stream, host, port,
                                      {"prompt": prompt_of(941, 2), "max_tokens": 1},
                                      accepted_event=accepted)
                if not accepted.wait(timeout=20):
                    if blocker.done():
                        blocker.result()
                    raise AssertionError("TUI prefill blocker was not accepted")
                tui_enter_sent = time.perf_counter()
                tui.send("\r")
                native = pool.submit(stream, host, port,
                                     {"prompt": prompt_of(16), "max_tokens": 128})
                compat_jobs = [pool.submit(compatible, protocol) for protocol in
                               ("openai-completions", "openai-chat", "anthropic")]
                deadline = time.monotonic() + 30
                while True:
                    state = json_request(host, port, "/metrics")
                    samples.append(state)
                    if (native.done() and blocker.done() and all(job.done() for job in compat_jobs)
                            and state["prefill_final_rows"] - initial["prefill_final_rows"] >= 6):
                        break
                    if time.monotonic() >= deadline:
                        raise AssertionError("TUI probe did not finish and reclaim pages")
                    time.sleep(0.002)
                result, blocker_result = native.result(), blocker.result()
                protocol_usage = [job.result() for job in compat_jobs]
            # Both native streams have drained and the TUI prompt has produced
            # its final row. Exercise TUI cancellation before terminating it.
            tui.send("\x1b")
            time.sleep(0.1)
        finally:
            exit_code = tui.close()
        if exit_code != 0:
            raise AssertionError(f"TUI did not shut down cleanly: {exit_code}")
        deadline = time.monotonic() + 20
        final = json_request(host, port, "/metrics")
        while any(final.get(key, 0) for key in idle_keys):
            if time.monotonic() >= deadline:
                raise AssertionError("TUI probe did not reclaim pages")
            time.sleep(0.01)
            final = json_request(host, port, "/metrics")
        delta = metric_delta(initial, final)
        peak = max((sample["last_batch_size"] for sample in samples
                    if sample["decode_steps"] > initial["decode_steps"]), default=0)
        timing = {"blocker_accepted": blocker_result["accepted"],
                  "blocker_first_token": blocker_result["stamps"][0],
                  "tui_enter_sent": tui_enter_sent, "native_accepted": result["accepted"]}
        dispatch_window_met = (tui_enter_sent < timing["blocker_first_token"]
                               and result["accepted"] < timing["blocker_first_token"])
        # Exactly six prompt slices and final rows rule out a partial blocker
        # slice masquerading as a short prompt. The single smaller packed batch
        # cannot contain the 941-token native blocker: it contains all five
        # client surfaces. Adapter usage supplies their exact formatted lengths.
        # The blocker generates one token, so it
        # also cannot be responsible for a later decode batch of size two.
        request_count = delta.get("completed_requests", 0) + delta.get("cancelled_requests", 0)
        short_tokens = 16 + 16 + sum(row["prompt_tokens"] for row in protocol_usage)
        expected.update(prefill_tokens=941 + short_tokens, packed_prefill_tokens=short_tokens)
        packed_identity_proved = (request_count == 6 and short_tokens < 941
                                 and all(delta.get(key) == value for key, value in expected.items()))
        evidence = {"attempt": attempt, "peak_decode_batch": peak,
                    "native_tokens": len(result["tokens"]), "tui_exit_code": exit_code,
                    "metrics_delta": delta, "timing": timing,
                    "input_and_native_acceptance_before_blocker_first_token": dispatch_window_met,
                    "protocol_usage": protocol_usage,
                    "all_five_protocols_packed_together": packed_identity_proved,
                    "tui_packed_participation_proved": packed_identity_proved}
        attempts.append(evidence)
        if packed_identity_proved and peak >= 2:
            return {**evidence, "attempts": attempts,
                    "proof": "Six prompts consumed exactly once in two batches: the 941-token blocker "
                             "was singleton; the smaller packed batch contains native, OpenAI completion, "
                             "OpenAI chat, Anthropic and TUI prompts together."}
    raise AssertionError(f"TUI packed-prefill participation was not proved in 10 attempts; "
                         f"requires a single-slice 941-token blocker: {json.dumps(attempts)}")


def sustained(host: str, port: int, model: str, duration: float,
              arrival_rate: float, max_inflight: int, max_tokens: int,
              cancel_every: int, queue_limit: int | None, require_packed: bool) -> dict:
    """An open arrival process, with bounded clients and explicit offered load.

    Native requests retain per-token timings. Other protocols use non-streaming
    responses and report completion latency plus real usage; their completion
    latency is deliberately never labelled TTFT. Queue samples cover the full
    arrival/drain window, including page pressure and packed/decode overlap.
    """
    snapshots, monitor_errors, stop = [], [], threading.Event()

    def watch():
        while not stop.is_set():
            try:
                snapshots.append({"time": time.perf_counter(),
                                  "metrics": json_request(host, port, "/metrics")})
            except Exception as error:
                monitor_errors.append(str(error))
            stop.wait(0.02)

    watcher = threading.Thread(target=watch, daemon=True)

    def work(index: int, barrier=None, permit_cancel: bool = True) -> dict:
        protocol = ("native", "openai-completions", "openai-chat", "anthropic")[index % 4]
        prompt = prompt_of((16, 64, 256, 512)[index % 4], index)
        fields = sampling_fields(index, protocol)
        policy = "greedy" if not fields else f"top-k-{SAMPLING_POLICIES[index % len(SAMPLING_POLICIES)][1]}"
        if barrier is not None:
            barrier.wait(timeout=30)
        started = time.perf_counter()
        try:
            if permit_cancel and cancel_every > 0 and index % cancel_every == 0:
                result = stream(host, port, {"prompt": prompt, "max_tokens": max_tokens,
                                             **sampling_fields(index, "native")},
                                cancel_after=(index // cancel_every) % 2)
                return {"protocol": "native-cancel", "generated": len(result["tokens"]),
                        "sampling_policy": policy, "total_ms": (time.perf_counter() - started) * 1000, **result}
            if protocol == "native":
                result = stream(host, port, {"prompt": prompt, "max_tokens": max_tokens, **fields})
                return {"protocol": protocol, "sampling_policy": policy,
                        "generated": len(result["tokens"]), **result}
            if protocol == "openai-completions":
                response = json_request(host, port, "/v1/completions",
                                        {"model": model, "prompt": prompt, "max_tokens": max_tokens, **fields})
                generated = response["usage"]["completion_tokens"]
            elif protocol == "openai-chat":
                response = json_request(host, port, "/v1/chat/completions",
                                        {"model": model, "messages": [{"role": "user", "content": prompt}],
                                         "max_tokens": max_tokens, **fields})
                generated = response["usage"]["completion_tokens"]
            else:
                response = json_request(host, port, "/v1/messages",
                                        {"model": model, "messages": [{"role": "user", "content": prompt}],
                                         "max_tokens": max_tokens, **fields},
                                        {"anthropic-version": "2023-06-01"})
                generated = response["usage"]["output_tokens"]
            return {"protocol": protocol, "sampling_policy": policy, "generated": generated,
                    "total_ms": (time.perf_counter() - started) * 1000}
        except Exception as error:
            return {"protocol": protocol, "sampling_policy": policy, "error": str(error),
                    "total_ms": (time.perf_counter() - started) * 1000}

    # Exactly one request per adapter, with no unrelated traffic. Therefore any
    # packed batch observed during this probe necessarily spans protocols.
    # Repeating gives HTTP arrival jitter several chances, without inventing a
    # protocol label or changing the serving scheduler just for observability.
    cross_protocol = []
    for probe in range(3):
        probe_before = json_request(host, port, "/metrics")
        barrier = threading.Barrier(4)
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs = [pool.submit(work, 4 * probe + i, barrier, False) for i in range(4)]
            probe_results = [job.result() for job in jobs]
        probe_after = json_request(host, port, "/metrics")
        cross_protocol.append({"requests": probe_results,
                               "metrics_delta": metric_delta(probe_before, probe_after)})
        if any("error" in result for result in probe_results):
            raise RuntimeError(f"mixed-protocol probe failed: {probe_results}")
    cross_packed = sum(run["metrics_delta"].get("packed_prefill_batches", 0) for run in cross_protocol)
    cross_aggregated = any(run["metrics_delta"].get("prefill_requests", 0) >
                           run["metrics_delta"].get("prefill_batches", 0) for run in cross_protocol)
    if require_packed and (cross_packed == 0 or not cross_aggregated):
        raise AssertionError("mixed-protocol probe did not execute a packed GPU batch")
    before = json_request(host, port, "/metrics")
    start = time.perf_counter()
    watcher.start()
    futures, pending_futures, client_backpressure = [], [], 0
    try:
        with ThreadPoolExecutor(max_workers=max_inflight) as pool:
            for index in range(int(duration * arrival_rate)):
                due = start + index / arrival_rate
                delay = due - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                pending_futures = [future for future in pending_futures if not future.done()]
                if len(pending_futures) >= max_inflight:
                    client_backpressure += 1
                    continue
                future = pool.submit(work, index)
                futures.append(future)
                pending_futures.append(future)
            results = [future.result() for future in futures]
        deadline = time.monotonic() + 30
        after = json_request(host, port, "/metrics")
        while any(after.get(key, 0) for key in ("active_requests", "queued_requests", "prefilling_requests", "kv_pages_used")):
            if time.monotonic() >= deadline:
                raise RuntimeError(f"sustained workload did not drain/reclaim pages: {after}")
            time.sleep(0.01)
            after = json_request(host, port, "/metrics")
        elapsed = time.perf_counter() - start
    finally:
        stop.set()
        watcher.join(timeout=181)
    errors = [r for r in results if "error" in r]
    protocols = {}
    for protocol in ("native", "openai-completions", "openai-chat", "anthropic"):
        successful = [r for r in results if r["protocol"] == protocol and "error" not in r]
        protocols[protocol] = {"completed": len(successful),
                               "total_ms": distribution([r["total_ms"] for r in successful]),
                               "ttft_ms": distribution([r["ttft_ms"] for r in successful if "ttft_ms" in r])}
    peak_queued = max((s["metrics"].get("queued_requests", 0) for s in snapshots), default=0)
    if queue_limit is not None and peak_queued > queue_limit:
        raise AssertionError(f"observed queue {peak_queued} exceeds configured limit {queue_limit}")
    delta = metric_delta(before, after)
    if require_packed and (delta.get("packed_prefill_batches", 0) == 0 or
                           delta.get("prefill_requests", 0) <= delta.get("prefill_batches", 0)):
        raise AssertionError("sustained workload did not execute packed GPU batches")
    if after["kv_pages_free"] != before["kv_pages_free"]:
        raise AssertionError("sustained workload changed free page count")
    if monitor_errors:
        raise AssertionError(f"metrics monitor failed during load: {monitor_errors}")
    unexpected_errors = [error for error in errors
                         if "HTTP 429" not in error["error"] and not
                         (error["protocol"] == "anthropic" and "HTTP 529" in error["error"])]
    if unexpected_errors or delta.get("failed_requests", 0):
        raise AssertionError(f"unexpected serving failure under load: {unexpected_errors[:3]}, metrics={delta}")
    return {"duration_s": duration, "arrival_rate": arrival_rate, "wall_including_drain_s": elapsed,
            "offered_requests": int(duration * arrival_rate), "submitted_requests": len(futures),
            "client_backpressure": client_backpressure, "errors": errors, "protocols": protocols,
            "sampling_policy_counts": {policy: sum(r["sampling_policy"] == policy for r in results)
                                       for policy in ("greedy", "top-k-5", "top-k-40", "top-k-128", "top-k-500")},
            "completed_requests_per_s": sum("error" not in r and not r.get("cancelled", False) for r in results) / elapsed,
            "submitted_requests_per_s": len(futures) / duration,
            "requested_disconnects": sum(r.get("cancelled", False) for r in results),
            "monitor_errors": monitor_errors, "final_metrics": after,
            "cross_protocol_probes": cross_protocol, "cross_protocol_packed_batches": cross_packed,
            "cross_protocol_aggregation_proven": cross_aggregated,
            "native_gap_ms": distribution([gap for r in results for gap in r.get("gaps_ms", [])]),
            "aggregate_tok_s": sum(r.get("generated", 0) for r in results) / elapsed,
            "metrics_delta": delta, "queue_samples": snapshots,
            "peak_queued": peak_queued, "asserted_queue_limit": queue_limit,
            "peak_prefilling": max((s["metrics"].get("prefilling_requests", 0) for s in snapshots), default=0),
            "peak_kv_pages": max((s["metrics"].get("kv_pages_used", 0) for s in snapshots), default=0),
            "requests": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--scenarios", default="isolated,short,mixed,long,stall,hol,fairness")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--existing-tokens", type=int, default=900)
    parser.add_argument("--fairness-arrival-spacing-ms", type=float, default=1.0,
                        help="requested spacing for sorted/reverse/shuffled fairness arrivals; actual starts are retained")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--duration", type=float, default=30.0,
                        help="seconds of offered arrivals in sustained mode")
    parser.add_argument("--arrival-rate", type=float, default=100.0,
                        help="requests per second offered in sustained mode")
    parser.add_argument("--max-inflight", type=int, default=64)
    parser.add_argument("--cancel-every", type=int, default=17,
                        help="disconnect every Nth sustained request (0 disables)")
    parser.add_argument("--queue-limit", type=int,
                        help="assert observed queued_requests does not exceed this server configuration")
    parser.add_argument("--require-packed", action="store_true",
                        help="require metric proof of packed GPU work and cross-protocol packed batches")
    parser.add_argument("--tui-binary",
                        help="prove real pty TUI packed-prefill and shared-decode participation; "
                             "requires a single-slice 941-token blocker (POSIX only)")
    parser.add_argument("--label", default="unspecified")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scenarios = args.scenarios.split(",")
    if args.trials < 1 or args.max_tokens < 1 or args.warmup < 0:
        parser.error("trials/max-tokens must be positive and warmup nonnegative")
    if args.duration <= 0 or args.arrival_rate <= 0 or args.max_inflight <= 0:
        parser.error("duration/arrival-rate/max-inflight must be positive")
    if args.fairness_arrival_spacing_ms < 0:
        parser.error("fairness-arrival-spacing-ms must be nonnegative")
    if set(scenarios) - {"isolated", "short", "tiny", "mixed", "long", "stall", "hol", "fairness", "sustained"}:
        parser.error("unknown scenario")
    workloads = []
    if "isolated" in scenarios:
        workloads.extend((f"isolated-{n}", [n]) for n in (1, 8, 16, 32, 64, 128, 256))
    if "short" in scenarios:
        workloads.extend((f"short-{n}", [(8, 16, 32, 64)[i % 4] for i in range(n)])
                         for n in (2, 4, 8, 16))
    if "mixed" in scenarios:
        workloads.append(("mixed-8", list(MIXED_LENGTHS)))
    if "tiny" in scenarios:
        workloads.extend((f"tiny-{n}x1", [1] * n) for n in (2, 4, 8, 16))
    if "long" in scenarios:
        workloads.extend((f"long-4x{length}", [length] * 4) for length in (256, 512, 941))
        workloads.append(("long-8x512", [512] * 8))
    if "fairness" in scenarios:
        workloads.extend((f"fairness-{order}", lengths) for order, lengths in (
            ("sorted", list(MIXED_LENGTHS)),
            ("reverse", list(reversed(MIXED_LENGTHS))),
            ("shuffled", [127, 8, 941, 33, 512, 17, 256, 64])))
    health = json_request(args.host, args.port, "/health")
    public_model = None
    if "sustained" in scenarios:
        models = json_request(args.host, args.port, "/v1/models").get("data", [])
        if not models or not isinstance(models[0].get("id"), str):
            raise RuntimeError("/v1/models did not advertise a public model ID")
        public_model = models[0]["id"]
    # Validate every generated prompt shape, including all vocabulary variants.
    counts = {(length, i % len(WORDS)) for _, lengths in workloads for i, length in enumerate(lengths)}
    if "stall" in scenarios:
        counts.update((length, 1) for length in (33, 535, 941))
    if "hol" in scenarios:
        counts.update((length, i + 1) for i, length in enumerate(HOL_LENGTHS))
    if "stall" in scenarios or "hol" in scenarios:
        counts.add((8, 0))
        if args.existing_tokens <= 32 or args.existing_tokens + 8 > health["context"]:
            parser.error("existing-tokens must exceed 32 and fit context with its 8-token prompt")
    if "fairness" in scenarios:
        counts.update((length, i) for i, length in enumerate(MIXED_LENGTHS))
    if "sustained" in scenarios:
        counts.update((length, variant) for length in (16, 64, 256, 512) for variant in range(8))
    for length, variant in sorted(counts):
        actual = json_request(args.host, args.port, "/v1/generate",
                              {"prompt": prompt_of(length, variant), "max_tokens": 1})["prompt_tokens"]
        if actual != length:
            raise RuntimeError(f"tokenizer mismatch: expected {length}, got {actual}, word {variant}")
    for _ in range(args.warmup):
        stream(args.host, args.port, {"prompt": prompt_of(64), "max_tokens": min(256, health["context"] - 64)})
    report = {"label": args.label, "health": health, "trials": args.trials,
              "prompt_counts_verified": len(counts), "gpu_before": gpu_envelope(), "runs": {}}
    if args.tui_binary:
        report["tui_overlap"] = tui_overlap_probe(args.host, args.port, args.tui_binary)
    with EnvelopeMonitor() as monitor:
        for trial in range(args.trials):
            # Reverse workload order every round to expose shape/thermal drift.
            for name, lengths in (workloads if trial % 2 == 0 else workloads[::-1]):
                fairness = name.startswith("fairness-")
                bodies = [{"prompt": prompt_of(length, MIXED_LENGTHS.index(length) if fairness else i),
                           "max_tokens": args.max_tokens}
                          for i, length in enumerate(lengths)]
                delays = ([i * args.fairness_arrival_spacing_ms / 1000 for i in range(len(bodies))]
                          if fairness else None)
                result = burst(args.host, args.port, bodies, delays)
                result["prompt_lengths"] = lengths
                if fairness:
                    result["requested_arrival_spacing_ms"] = args.fairness_arrival_spacing_ms
                    result["actual_start_order"] = sorted(range(len(lengths)),
                                                          key=lambda i: result["requests"][i]["start"])
                    result["actual_start_prompt_lengths"] = [lengths[i] for i in result["actual_start_order"]]
                report["runs"].setdefault(name, []).append(result)
                print(f"{args.label} round={trial + 1} {name}: "
                      f"TTFT={statistics.median(r['ttft_ms'] for r in result['requests']):.2f} ms "
                      f"aggregate={result['aggregate_tok_s']:.1f} tok/s", flush=True)
            if "stall" in scenarios:
                for length in (33, 535, 941):
                    result = interruption(args.host, args.port, length, args.existing_tokens)
                    report["runs"].setdefault(f"stall-{length}", []).append(result)
                    print(f"{args.label} round={trial + 1} stall-{length}: "
                          f"TTFT={result['intruder']['ttft_ms']:.2f} ms "
                          f"worst gap={result['existing_disturbance_gap_ms']['max']:.2f} ms", flush=True)
            if "hol" in scenarios:
                result = head_of_line(args.host, args.port, args.existing_tokens)
                report["runs"].setdefault("hol-4", []).append(result)
                print(f"{args.label} round={trial + 1} hol-4: "
                      f"TTFT={statistics.median(r['ttft_ms'] for r in result['intruders']):.2f} ms "
                      f"worst gap={result['existing_disturbance_gap_ms']['worst']:.2f} ms", flush=True)
            if "sustained" in scenarios:
                result = sustained(args.host, args.port, public_model, args.duration,
                                   args.arrival_rate, args.max_inflight, args.max_tokens,
                                   args.cancel_every, args.queue_limit, args.require_packed)
                report["runs"].setdefault("sustained", []).append(result)
                print(f"{args.label} round={trial + 1} sustained: "
                      f"{result['submitted_requests']} submitted, {len(result['errors'])} errors, "
                      f"peak queue={result['peak_queued']}, aggregate={result['aggregate_tok_s']:.1f}", flush=True)
            report["gpu_loaded"] = monitor.samples.copy()
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["gpu_after"] = gpu_envelope()
    if args.require_packed:
        deltas = [run["metrics_delta"] for runs in report["runs"].values() for run in runs]
        if (sum(delta.get("packed_prefill_batches", 0) for delta in deltas) == 0 or
                sum(delta.get("prefill_requests", 0) for delta in deltas) <=
                sum(delta.get("prefill_batches", 0) for delta in deltas)):
            raise AssertionError("benchmark did not execute packed GPU batches")
    report["summary"] = {}
    for name, runs in report["runs"].items():
        if name == "sustained":
            report["summary"][name] = {
                "aggregate_tok_s": distribution([r["aggregate_tok_s"] for r in runs]),
                "errors": sum(len(r["errors"]) for r in runs),
                "peak_queued": max(r["peak_queued"] for r in runs),
                "client_backpressure": sum(r["client_backpressure"] for r in runs)}
        elif name.startswith("stall"):
            report["summary"][name] = {
                "ttft_ms": distribution([r["intruder"]["ttft_ms"] for r in runs]),
                "worst_gap_ms": distribution([r["existing_disturbance_gap_ms"]["max"] for r in runs])}
        elif name.startswith("hol-"):
            gaps = [gap for run in runs for gap in run["existing_disturbance_gaps_ms"]]
            gap_summary = distribution(gaps)
            gap_summary.update(p50=percentile(gaps, 0.50), p99=percentile(gaps, 0.99), worst=max(gaps))
            report["summary"][name] = {
                "ttft_ms": distribution([request["ttft_ms"] for run in runs for request in run["intruders"]]),
                "existing_disturbance_gap_ms": gap_summary,
                "worst_gap_ms": distribution([r["existing_disturbance_gap_ms"]["worst"] for r in runs])}
        else:
            requests = [request for run in runs for request in run["requests"]]
            report["summary"][name] = {
                "ttft_ms": distribution([r["ttft_ms"] for r in requests]),
                "aggregate_tok_s": distribution([r["aggregate_tok_s"] for r in runs]),
                "gap_ms": distribution([gap for r in requests for gap in r["gaps_ms"]])}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
