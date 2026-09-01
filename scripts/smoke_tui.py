"""
Real-server smoke test for the TUI, driven through a pty.

The TUI is an interactive terminal program, so testing it means giving it a
terminal. This allocates a pty, runs the binary in it, sends real keystrokes and
reads what gets painted.

The load-bearing check is the last one: while the TUI is generating, an
independent HTTP client generates too, and the server's own metrics must show
them sharing decode steps. A TUI that quietly got its own inference path would
pass every other check here.

Usage (server must already be running):
    python scripts/smoke_tui.py --binary ./target/release/llm-engine --port 8080
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import pty
import re
import select
import subprocess
import sys
import threading
import time

FAILED: list[str] = []
PASSED = 0

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07|\x1b[()][A-B0-2]")


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSED
    if cond:
        PASSED += 1
        print(f"  ok    {name}")
    else:
        FAILED.append(name)
        print(f"  FAIL  {name}  {detail}")


def strip(s: str) -> str:
    return ANSI.sub("", s)


class Tui:
    """The TUI running under a pty."""

    def __init__(self, binary: str, server: str, cols: int = 100, rows: int = 30):
        self.master, slave = pty.openpty()
        import fcntl
        import struct
        import termios
        fcntl.ioctl(slave, termios.TIOCSWINSZ,
                    struct.pack("HHHH", rows, cols, 0, 0))
        self.proc = subprocess.Popen(
            [binary, "tui", "--server", server, "--max-tokens", "400"],
            stdin=slave, stdout=slave, stderr=slave, close_fds=True,
        )
        os.close(slave)
        self.buf = ""
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._drain, daemon=True)
        self._t.start()

    def _drain(self):
        while not self._stop.is_set():
            r, _, _ = select.select([self.master], [], [], 0.1)
            if r:
                try:
                    data = os.read(self.master, 65536)
                except OSError:
                    return
                if not data:
                    return
                self.buf += data.decode("utf-8", "replace")

    def send(self, s: str) -> None:
        os.write(self.master, s.encode())

    def screen(self) -> str:
        return strip(self.buf)

    def wait_for(self, needle: str, timeout: float = 20.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            if needle in self.screen():
                return True
            time.sleep(0.05)
        return False

    def resize(self, cols: int, rows: int) -> None:
        import fcntl
        import struct
        import termios
        fcntl.ioctl(self.master, termios.TIOCSWINSZ,
                    struct.pack("HHHH", rows, cols, 0, 0))
        self.proc.send_signal(28)  # SIGWINCH

    def close(self) -> int | None:
        self.send("\x03")  # Ctrl+C
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)
        self._stop.set()
        try:
            os.close(self.master)
        except OSError:
            pass
        return self.proc.returncode


def metrics(host: str, port: int) -> dict:
    c = http.client.HTTPConnection(host, port, timeout=5)
    c.request("GET", "/metrics")
    d = json.loads(c.getresponse().read())
    c.close()
    return d


def background_stream(host: str, port: int, max_tokens: int, seen: dict) -> None:
    """An independent client, so the TUI is not the only request in flight."""
    c = http.client.HTTPConnection(host, port, timeout=120)
    c.request("POST", "/v1/generate/stream",
              json.dumps({"prompt": "In a distant galaxy the crew found",
                          "max_tokens": max_tokens}),
              {"Content-Type": "application/json"})
    r = c.getresponse()
    for raw in r:
        if raw.startswith(b"event: done"):
            break
        seen["tokens"] = seen.get("tokens", 0) + (1 if raw.startswith(b"data:") else 0)
    c.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--binary", default="./target/release/llm-engine")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()
    server = f"http://{args.host}:{args.port}"

    base = metrics(args.host, args.port)
    total_pages = base["kv_pages_used"] + base["kv_pages_free"]

    print("startup")
    tui = Tui(args.binary, server)
    try:
        check("TUI starts and connects", tui.wait_for("connected", 20),
              tui.screen()[-400:])
        check("header shows the model from /health", "120m" in tui.screen(),
              tui.screen()[-400:])
        check("header shows greedy sampling", "greedy" in tui.screen())

        print("\ngeneration")
        tui.send("The capital of France is")
        time.sleep(0.5)
        # A short marker only. ratatui writes just the cells that changed,
        # so a whole phrase is not guaranteed to appear contiguously in the
        # pty stream. Whether the input was really captured is proved below
        # by the server receiving a request.
        check("typed text reaches the input line", "France" in tui.screen(),
              tui.screen()[-300:])
        tui.send("\r")
        check("user message is echoed into the conversation",
              tui.wait_for("You", 10), tui.screen()[-600:])
        check("assistant reply streams in", tui.wait_for("Crucible", 15))
        # Let some tokens accumulate.
        time.sleep(2.0)
        m = metrics(args.host, args.port)
        check("server saw the TUI request",
              m["aggregate_tokens_generated"] > base["aggregate_tokens_generated"],
              f"{base['aggregate_tokens_generated']} -> {m['aggregate_tokens_generated']}")

        print("\ntelemetry panel")
        tui.send("\x1bOQ")  # F2
        check("telemetry panel toggles on",
              tui.wait_for("service", 5) or tui.wait_for("kv pages", 5),
              tui.screen()[-600:])

        print("\ncancellation")
        # Start a generation specifically to cancel. The earlier one is
        # long finished: at ~900 tok/s a 400-token request takes under half
        # a second, so cancelling it was a race the test could only lose.
        tui.send("Write a very long story about a lighthouse")
        tui.send("\r")
        # Wait for it to be resident rather than sleeping a guessed amount.
        deadline = time.time() + 10
        running = False
        while time.time() < deadline:
            if metrics(args.host, args.port)["active_requests"] > 0:
                running = True
                break
            time.sleep(0.01)
        check("a generation is in flight to cancel", running,
              str(metrics(args.host, args.port)))
        before_cancel = metrics(args.host, args.port)
        tui.send("\x1b")  # Esc
        # The authoritative signal is the server's cancellation counter, not
        # a word on screen: the UI could render "cancelled" and still have
        # failed to drop the stream.
        deadline = time.time() + 10
        counted = False
        while time.time() < deadline:
            if metrics(args.host, args.port)["cancelled_requests"] > before_cancel["cancelled_requests"]:
                counted = True
                break
            time.sleep(0.05)
        check("server registered the cancellation", counted,
              str(metrics(args.host, args.port)))
        deadline = time.time() + 15
        reclaimed = False
        while time.time() < deadline:
            m = metrics(args.host, args.port)
            if m["active_requests"] == 0 and m["kv_pages_free"] == total_pages:
                reclaimed = True
                break
            time.sleep(0.1)
        check("KV pages reclaimed after cancel", reclaimed,
              f"free={metrics(args.host, args.port)['kv_pages_free']}/{total_pages}")

        print("\nreuse after cancel")
        before_second = metrics(args.host, args.port)
        tui.send("Once upon a time")
        tui.send("\r")
        # Proof the app is usable again: the server receives another request.
        deadline = time.time() + 15
        accepted = False
        while time.time() < deadline:
            m2 = metrics(args.host, args.port)
            if m2["aggregate_tokens_generated"] > before_second["aggregate_tokens_generated"]:
                accepted = True
                break
            time.sleep(0.05)
        check("a second prompt is accepted after cancelling", accepted,
              str(metrics(args.host, args.port)))
        time.sleep(1.5)

        print("\ncontinuous batching with an independent client")
        # TUI is generating; start an external client and watch the server's
        # batch size. If the TUI had a private path, batch would stay at 1.
        peak = {"batch": 0}
        stop = threading.Event()

        def watch():
            while not stop.is_set():
                try:
                    peak["batch"] = max(peak["batch"],
                                        metrics(args.host, args.port)["last_batch_size"])
                except Exception:
                    pass
                time.sleep(0.01)

        w = threading.Thread(target=watch, daemon=True)
        w.start()
        tui.send("Explain how a transformer works")
        tui.send("\r")
        time.sleep(0.3)
        seen: dict = {}
        ext = threading.Thread(target=background_stream,
                               args=(args.host, args.port, 200, seen))
        ext.start()
        ext.join(timeout=60)
        stop.set()
        w.join(timeout=1)
        check("TUI and external client shared decode steps (batch >= 2)",
              peak["batch"] >= 2, f"peak batch {peak['batch']}")
        check("external client generated tokens", seen.get("tokens", 0) > 0, str(seen))

        print("\nresize")
        tui.resize(60, 20)
        time.sleep(0.6)
        check("survives a resize", tui.proc.poll() is None)
        tui.resize(120, 36)
        time.sleep(0.6)
        check("survives a second resize", tui.proc.poll() is None)

        print("\nshutdown")
        code = tui.close()
        check("exits cleanly on Ctrl+C", code == 0, f"exit code {code}")

    finally:
        if tui.proc.poll() is None:
            tui.proc.kill()

    # The server must be unharmed by everything above.
    m = metrics(args.host, args.port)
    check("server still healthy", m["active_requests"] == 0,
          str(m))
    check("all KV pages returned", m["kv_pages_free"] == total_pages,
          f"{m['kv_pages_free']}/{total_pages}")

    print()
    if FAILED:
        print(f"{PASSED} passed, {len(FAILED)} FAILED: {FAILED}")
        sys.exit(1)
    print(f"all {PASSED} checks passed")


if __name__ == "__main__":
    main()
