"""Exercise HTTP admission limits, cancellation and reuse on a small KV pool.

Start a dedicated server with --kv-pages 32 --max-queue 4 and a generation limit
of at least 256, then run this script. Every valid request fits by itself. The
resident victim prevents the waiting burst from fitting until cancellation;
accepted survivors must reproduce independent reference token IDs exactly.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
from pathlib import Path
import threading
import time

from bench_packed_prefill import json_request, metric_delta, prompt_of, stream
from test_packed_prefill import await_idle, compare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--pages", type=int, default=32)
    parser.add_argument("--queue-limit", type=int, default=4)
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.requests <= args.queue_limit or args.pages != 32:
        parser.error("this pressure geometry requires 32 pages and requests > queue-limit")
    health = json_request(args.host, args.port, "/health")
    if health["kv_pages"] != args.pages:
        raise AssertionError(f"expected {args.pages} physical pages, got {health['kv_pages']}")
    before = await_idle(args.host, args.port)
    survivor = {"prompt": prompt_of(64, 1), "max_tokens": 128}
    reference = stream(args.host, args.port, survivor)
    reference = {key: reference[key] for key in ("tokens", "text", "done")}
    snapshots, monitor_errors = [], []
    stop, ready, cancel = threading.Event(), threading.Event(), threading.Event()

    def monitor() -> None:
        while not stop.is_set():
            try:
                metrics = json_request(args.host, args.port, "/metrics")
                snapshots.append(metrics)
                if metrics["queued_requests"] >= args.queue_limit:
                    cancel.set()
            except Exception as error:
                monitor_errors.append(str(error))
            stop.wait(0.002)

    barrier = threading.Barrier(args.requests)

    def request(index: int) -> dict:
        try:
            result = stream(args.host, args.port, survivor, barrier)
            compare(f"pressure survivor {index}", result, reference)
            return {"index": index, "status": 200, "tokens": len(result["tokens"])}
        except RuntimeError as error:
            if "HTTP 429" in str(error):
                return {"index": index, "status": 429}
            raise

    # 96+256 tokens reserve 22 pages, leaving ten. A 64+128-token
    # survivor needs twelve and therefore must wait until the victim releases.
    with ThreadPoolExecutor(max_workers=args.requests + 1) as pool:
        victim = pool.submit(stream, args.host, args.port,
                             {"prompt": prompt_of(96, 7), "max_tokens": 256},
                             None, ready, 0.0, None, cancel)
        if not ready.wait(timeout=30):
            raise AssertionError("victim did not become an established stream")
        watcher = threading.Thread(target=monitor, daemon=True)
        watcher.start()
        try:
            futures = [pool.submit(request, index) for index in range(args.requests)]
            results = [future.result() for future in futures]
            cancelled = victim.result()
        finally:
            cancel.set()
            stop.set()
            watcher.join(timeout=181)
    after = await_idle(args.host, args.port)
    if monitor_errors:
        raise AssertionError(f"metrics monitor failed: {monitor_errors}")
    accepted = sum(result["status"] == 200 for result in results)
    rejected = sum(result["status"] == 429 for result in results)
    if not accepted or not rejected:
        raise AssertionError(f"pressure did not exercise accepted and 429 outcomes: {results}")
    if not cancelled.get("cancelled") or after["cancelled_requests"] <= before["cancelled_requests"]:
        raise AssertionError("resident victim was not cancelled")
    peak_queue = max(snapshot["queued_requests"] for snapshot in snapshots)
    peak_pages = max(snapshot["kv_pages_used"] for snapshot in snapshots)
    if peak_queue > args.queue_limit or peak_pages > args.pages:
        raise AssertionError(f"capacity exceeded: queue={peak_queue}, pages={peak_pages}")

    # A context-valid request that cannot fit in the entire physical pool must
    # fail immediately, leaving no unserviceable head-of-line request behind.
    connection = http.client.HTTPConnection(args.host, args.port, timeout=30)
    try:
        connection.request("POST", "/v1/generate", json.dumps(
            {"prompt": prompt_of(512), "max_tokens": 2}), {"Content-Type": "application/json"})
        response = connection.getresponse()
        response.read()
        if response.status != 400:
            raise AssertionError(f"oversize physical reservation returned {response.status}, expected 400")
    finally:
        connection.close()
    compare("page reuse after oversize rejection", stream(args.host, args.port, survivor), reference)
    final = await_idle(args.host, args.port)
    if final["failed_requests"] != before["failed_requests"] or final["kv_pages_free"] != args.pages:
        raise AssertionError(f"server failed or leaked pages: {final}")
    report = {"accepted": accepted, "rejected_429": rejected, "cancelled": True,
              "peak_queue": peak_queue, "peak_pages": peak_pages,
              "queue_limit": args.queue_limit, "physical_pages": args.pages,
              "metrics_delta": metric_delta(before, final), "samples": snapshots,
              "results": results, "final_metrics": final}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PASS: {accepted} accepted exact survivors, {rejected} HTTP429, resident cancellation, "
          f"peak queue {peak_queue}/{args.queue_limit}, pages {peak_pages}/{args.pages}; every page reused/reclaimed")


if __name__ == "__main__":
    main()
