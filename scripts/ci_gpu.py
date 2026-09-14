"""Run the optional, provisioned Linux GPU CI suites without installing assets.

Use --list-commands to inspect a suite on any host; it does not claim validation.
Actual execution always performs preflight and fails on missing prerequisites.
All subprocess arguments are lists, configured asset paths are redacted, and
servers are loopback-only and stopped on failure, timeout, or cancellation.
"""
from __future__ import annotations

import argparse
import ctypes
import http.client
import json
import math
import os
import platform
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ("CRUCIBLE_MODEL_PATH", "CRUCIBLE_TOKENIZER_PATH", "CRUCIBLE_HELDOUT_PATH",
          "CRUCIBLE_OPENAI_PYTHON", "CRUCIBLE_ANTHROPIC_PYTHON")
# Keep aligned with Gpu::new's NVRTC include discovery in engine/src/gpu.rs.
CUDA_INCLUDES = ("/usr/local/cuda/include", "/usr/local/cuda-13/include",
                 "/usr/local/cuda-13.0/targets/x86_64-linux/include",
                 "/usr/local/cuda-13.3/targets/x86_64-linux/include")
MODES = ("smoke", "full", "sanitizer")


class CheckFailure(RuntimeError):
    """An actionable CI prerequisite or correctness failure."""


@dataclass(frozen=True)
class Step:
    name: str
    argv: tuple[str, ...]
    env: dict[str, str] = field(default_factory=dict)
    server: str | None = None
    ce_context: int | None = None
    timeout: int = 1200


def suite_plan(mode: str, assets: dict[str, str], binary: str,
               scratch: str, port: int = 18081) -> list[Step]:
    """Pure command construction shared by --list-commands and real execution."""
    if mode not in MODES:
        raise ValueError(f"unknown suite {mode!r}")
    model = assets[ASSETS[0]]
    python = sys.executable
    steps = [Step("release CUDA build", ("cargo", "build", "--manifest-path",
                  "engine/Cargo.toml", "--release", "--features", "cuda", "--locked"))]

    def gpu(name: str, *args: str, env: dict[str, str] | None = None,
            ce_context: int | None = None) -> None:
        steps.append(Step(name, (binary, *args), env or {}, ce_context=ce_context))

    if mode == "sanitizer":
        prefix = ("compute-sanitizer", "--tool", "memcheck", "--error-exitcode", "99", binary)
        steps.extend([
            Step("sanitizer attention tensors", prefix + ("gpu-prefill-attention-check",
                 "--fuzz", "128", "--seed", "20260908"), timeout=4200),
            Step("sanitizer packed model", prefix + ("gpu-packed-prefill-check", model,
                 "--quant", "int8", "--steps", "16", "--fuzz", "24"),
                 {"CRUCIBLE_PREFILL_ATTN": "exact-q4-k64"}, timeout=4200),
        ])
        return steps
    if mode == "full":
        steps.append(Step("CUDA-feature Rust tests", ("cargo", "test", "--manifest-path",
                          "engine/Cargo.toml", "--features", "cuda", "--locked")))
    gpu("kernel smoke and selection checks", "gpu-validate")
    for command in ("gpu-graph-check", "gpu-batch", "gpu-sampling", "gpu-prefill-check"):
        gpu(command, command, model)
    gpu("paged decode and graphs", "gpu-paged", model, "--graph")
    graph_args = (() if mode == "full" else
                  ("--lengths", "15,16,17,127,128,129,941", "--chunks", "32,37,256", "--steps", "8"))
    gpu("prefill graphs", "gpu-prefill-graph-check", model, *graph_args)
    gpu("packed int8 generation and isolation", "gpu-packed-prefill-check", model,
        "--quant", "int8", "--steps", "16", "--fuzz", "24" if mode == "full" else "4")
    gpu("attention tensor and causal isolation", "gpu-prefill-attention-check",
        "--fuzz", "128" if mode == "full" else "8", "--seed", "20260908")
    if mode == "smoke":
        return steps
    gpu("packed f32 fallback", "gpu-packed-prefill-check", model,
        "--quant", "f32", "--steps", "4", "--fuzz", "0")
    for context, tokens in ((32, 1024), (256, 16384), (941, 16384)):
        for variant in ("reference", "exact-q4-k64"):
            gpu(f"paged CE context {context} {variant}", "gpu-eval", model,
                "--data", assets[ASSETS[2]], "--tokens", str(tokens), "--quant", "int8",
                "--graph", "--paged", "--prefill-ctx", str(context),
                env={"CRUCIBLE_PREFILL_ATTN": variant}, ce_context=context)

    fixture = str(Path(scratch) / "http-reference.json")

    def client(name: str, script: str, *args: str, server: str = "default") -> None:
        steps.append(Step(name, (python, "scripts/" + script, "--port", str(port), *args),
                          server=server))

    client("record independent HTTP reference", "test_packed_prefill.py",
           "--record-reference", fixture, "--steps", "16", "--fuzz-rounds", "24",
           server="reference")
    client("packed HTTP reference equivalence", "test_packed_prefill.py",
           "--reference", fixture, "--steps", "16", "--fuzz-rounds", "24")
    client("native HTTP", "test_serve.py")
    client("OpenAI HTTP and official SDK", "test_openai.py", "--sdk", assets[ASSETS[3]])
    client("Anthropic HTTP and official SDK", "test_anthropic.py", "--sdk", assets[ASSETS[4]])
    client("real-server TUI", "smoke_tui.py", "--binary", binary)
    client("32-page cancellation and queue pressure", "test_packed_pressure.py",
           "--pages", "32", "--queue-limit", "4", "--requests", "24", server="pressure")
    return steps


def runtime_env(source: dict[str, str]) -> dict[str, str]:
    """Exercise defaults, regardless of a runner's previous tuning experiments."""
    env = {key: value for key, value in source.items() if not key.startswith("CRUCIBLE_")}
    env.update(RUSTUP_TOOLCHAIN="stable", PYTHONDONTWRITEBYTECODE="1", CARGO_TERM_COLOR="never")
    return env


def redact(text: str, assets: dict[str, str]) -> str:
    for key, value in sorted(assets.items(), key=lambda item: len(item[1]), reverse=True):
        if value:
            text = text.replace(value, f"<{key}>")
    return text


def required_assets(mode: str, source: dict[str, str]) -> dict[str, str]:
    required = ASSETS if mode == "full" else ASSETS[:1]
    result = {}
    for key in required:
        value = source.get(key, "")
        if not value:
            raise CheckFailure(f"Set {key} to a provisioned local runner asset; no download or skip is performed.")
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise CheckFailure(f"{key} must be an absolute path.")
        result[key] = str(path)
    return result


def readable_file(path: Path, label: str, minimum: int = 1) -> None:
    try:
        if not path.is_file() or path.stat().st_size < minimum:
            raise CheckFailure(f"{label} is missing, empty, or shorter than the suite requires.")
        with path.open("rb") as handle:
            handle.read(1)
    except OSError:
        raise CheckFailure(f"{label} is not readable by the runner account.") from None


def validate_assets(mode: str, assets: dict[str, str]) -> None:
    model = Path(assets[ASSETS[0]])
    readable_file(model / "config.json", "model config.json")
    readable_file(model / "model.safetensors", "model.safetensors", 9)
    try:
        config = json.loads((model / "config.json").read_text())
    except (ValueError, OSError):
        raise CheckFailure("Model config.json must be readable valid JSON.") from None
    # Existing validation corpora target the exported GPT-2-vocabulary 120M model.
    expected = {"n_layer": 12, "n_head": 12, "n_kv_head": 3, "n_embd": 768,
                "block_size": 1024, "vocab_size": 50304}
    if not isinstance(config, dict) or any(config.get(key) != value for key, value in expected.items()):
        raise CheckFailure("GPU CI requires the documented 120M model geometry: " + json.dumps(expected))
    if mode == "full":
        if "120m" not in model.name:
            raise CheckFailure("Full mode's existing TUI corpus requires a model directory name containing '120m'.")
        readable_file(Path(assets[ASSETS[1]]), "CRUCIBLE_TOKENIZER_PATH")
        heldout = Path(assets[ASSETS[2]])
        readable_file(heldout, "CRUCIBLE_HELDOUT_PATH", 2 * 16385)
        if heldout.stat().st_size % 2:
            raise CheckFailure("CRUCIBLE_HELDOUT_PATH must contain complete little-endian uint16 tokens.")
        for key in ASSETS[3:]:
            readable_file(Path(assets[key]), key)
            if not os.access(assets[key], os.X_OK):
                raise CheckFailure(f"{key} must be an executable Python interpreter.")


def stop_process(proc: subprocess.Popen) -> None:
    """Stop this command's process group, including SDK/TUI child processes."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        proc.wait()
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # The direct child can exit before an SDK/TUI descendant. Reap the
        # entire isolated process group, even if the direct child already exited.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def run_command(argv: tuple[str, ...], env: dict[str, str], assets: dict[str, str],
                timeout: int = 1200) -> str:
    proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, errors="replace",
                            start_new_session=True)
    try:
        try:
            output, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            stop_process(proc)
            output, _ = proc.communicate()
            print(redact(output, assets), end="", flush=True)
            raise CheckFailure(f"Command exceeded its {timeout}s timeout.") from None
        print(redact(output, assets), end="", flush=True)
        if proc.returncode:
            raise CheckFailure(f"Command failed with exit code {proc.returncode}.")
        return output
    finally:
        stop_process(proc)


def cuda_preflight() -> None:
    try:
        driver = ctypes.CDLL("libcuda.so.1")
        if driver.cuInit(0):
            raise CheckFailure("NVIDIA driver initialization failed; GPU runtime is unavailable.")
        count = ctypes.c_int()
        if driver.cuDeviceGetCount(ctypes.byref(count)) or count.value < 1:
            raise CheckFailure("No CUDA device is visible to the driver API.")
        device, major, minor = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        if driver.cuDeviceGet(ctypes.byref(device), 0) or driver.cuDeviceComputeCapability(
                ctypes.byref(major), ctypes.byref(minor), device):
            raise CheckFailure("Could not query CUDA device 0 compute capability.")
        if (major.value, minor.value) < (12, 0):
            raise CheckFailure("CUDA device 0 must support compute capability 12.0: engine NVRTC targets compute_120.")
        print(f"CUDA driver device 0: compute capability {major.value}.{minor.value}")
    except OSError:
        raise CheckFailure("libcuda.so.1 must be discoverable by the runner's dynamic loader.") from None
    nvrtc = None
    for name in ("libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.12"):
        try:
            nvrtc = ctypes.CDLL(name)
            break
        except OSError:
            pass
    if nvrtc is None:
        raise CheckFailure("NVRTC shared library is missing; provision CUDA and its library search path.")
    major, minor = ctypes.c_int(), ctypes.c_int()
    if nvrtc.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)):
        raise CheckFailure("Could not read the NVRTC version.")
    try:
        count = ctypes.c_int()
        if nvrtc.nvrtcGetNumSupportedArchs(ctypes.byref(count)) or count.value <= 0:
            raise CheckFailure("Could not query NVRTC supported architectures.")
        archs = (ctypes.c_int * count.value)()
        if nvrtc.nvrtcGetSupportedArchs(archs) or 120 not in archs:
            raise CheckFailure("Installed NVRTC cannot compile the engine's compute_120 target.")
    except AttributeError:
        raise CheckFailure("NVRTC is too old to query supported architectures; provision a compute_120-capable toolkit.") from None
    print(f"NVRTC {major.value}.{minor.value}; compute_120 compilation is supported")
    if not any(all((Path(folder) / header).is_file() for header in ("mma.h", "cuda_fp16.h"))
               for folder in CUDA_INCLUDES):
        raise CheckFailure("CUDA mma.h and cuda_fp16.h must be in an include location supported by engine/src/gpu.rs.")


def preflight(mode: str, assets: dict[str, str], env: dict[str, str], target: Path) -> None:
    if platform.system() != "Linux":
        raise CheckFailure("Real GPU suites require Linux; --list-commands works without a GPU.")
    if mode == "sanitizer" and ("microsoft" in platform.release().lower() or os.environ.get("WSL_INTEROP")):
        raise CheckFailure("Sanitizer mode requires native Linux; WSL is not eligible for this coverage.")
    validate_assets(mode, assets)
    commands = ("nvidia-smi", "nvcc", "rustup", "rustc", "cargo", "cc")
    if mode == "sanitizer":
        commands += ("compute-sanitizer",)
    for command in commands:
        if shutil.which(command, path=env.get("PATH")) is None:
            raise CheckFailure(f"Missing runner prerequisite: {command}. Provision it before dispatch.")
    existing = target
    while not existing.exists():
        existing = existing.parent
    free_gib = shutil.disk_usage(existing).free / (1024 ** 3)
    if free_gib < 8:
        raise CheckFailure(f"Cargo target filesystem needs at least 8 GiB free; available {free_gib:.1f} GiB.")
    print(f"Cargo target filesystem: {free_gib:.1f} GiB free")
    installed = run_command(("rustup", "toolchain", "list"), env, assets, 30)
    if not re.search(r"^stable(?:-|\s|$)", installed, re.MULTILINE):
        raise CheckFailure("Provision the stable Rust toolchain; this workflow does not install one on the runner.")
    for argv in (("rustc", "--version"), ("cargo", "--version"), ("nvcc", "--version"),
                 ("nvidia-smi", "--query-gpu=name,driver_version,compute_cap,clocks.max.sm,memory.total",
                  "--format=csv,noheader")):
        run_command(argv, env, assets, 30)
    cuda_preflight()
    if mode == "full":
        for key, package in zip(ASSETS[3:], ("openai", "anthropic")):
            print(f"Checking provisioned {package} SDK interpreter", flush=True)
            run_command((assets[key], "-c", f"import {package}; print({package}.__version__)"), env, assets, 30)
    if mode == "sanitizer":
        run_command(("compute-sanitizer", "--version"), env, assets, 30)
    print("GPU preflight passed; this is prerequisite verification, not CUDA correctness coverage.", flush=True)


def parse_ce(output: str) -> tuple[float, float, int]:
    matches = re.findall(r"^int8\s+cross-entropy\s+(\S+)\s+perplexity\s+(\S+)\s+weights\s+\S+\s+MB\s+(\d+) positions", output, re.MULTILINE)
    if len(matches) != 1:
        raise CheckFailure("Expected exactly one int8 held-out metric row.")
    ce, perplexity, count = matches[0]
    values = float(ce), float(perplexity), int(count)
    if not all(math.isfinite(value) for value in values[:2]) or values[0] < 0 or values[1] < 1 or values[2] <= 0:
        raise CheckFailure("Held-out evaluation must report finite CE/perplexity and a positive scored-position count.")
    return values


def server_command(kind: str, assets: dict[str, str], binary: str, port: int) -> tuple[str, ...]:
    args = (binary, "serve", assets[ASSETS[0]], "--tokenizer", assets[ASSETS[1]],
            "--host", "127.0.0.1", "--port", str(port),
            "--max-prompt-tokens", "1000", "--max-new-tokens", "1000")
    return args + (("--kv-pages", "32", "--max-queue", "4") if kind == "pressure" else ())


def server_env(kind: str, env: dict[str, str]) -> dict[str, str]:
    # A singleton reference fixture independently checks both packing and attention.
    return dict(env, CRUCIBLE_BATCHED_PREFILL="0", CRUCIBLE_PREFILL_ATTN="reference") if kind == "reference" else env


@contextmanager
def server(kind: str, assets: dict[str, str], binary: str, port: int,
           env: dict[str, str], scratch: Path):
    # Reject an occupied port instead of accidentally testing an unrelated service.
    with socket.socket() as probe:
        # A preceding suite's closed connections may remain in TIME_WAIT. Match
        # the Linux server listener's reuse policy while still rejecting a live
        # listener, so reference/default/pressure restarts can share this port.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))
    log_path = scratch / f"{kind}-server.log"
    with log_path.open("w+") as log:
        proc = subprocess.Popen(server_command(kind, assets, binary, port), cwd=ROOT,
                                env=server_env(kind, env), stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            deadline = time.monotonic() + 120
            while True:
                if proc.poll() is not None:
                    raise CheckFailure(f"{kind} server exited during startup with code {proc.returncode}.")
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
                try:
                    connection.request("GET", "/health")
                    response = connection.getresponse()
                    if response.status == 200:
                        health = json.loads(response.read())
                        if not health.get("context"):
                            raise CheckFailure("Server health response has no model context.")
                        break
                except (OSError, http.client.HTTPException):
                    pass
                finally:
                    connection.close()
                if time.monotonic() >= deadline:
                    raise CheckFailure(f"{kind} server did not become healthy within 120 seconds.")
                time.sleep(0.1)
            print(f"{kind} loopback server ready", flush=True)
            yield
        finally:
            stop_process(proc)
            log.seek(0)
            print(redact(log.read(), assets), end="", flush=True)


def execute(steps: list[Step], assets: dict[str, str], env: dict[str, str],
            binary: str, scratch: Path, port: int) -> None:
    from contextlib import ExitStack
    metrics: dict[int, tuple[float, float, int]] = {}
    active_server = None
    with ExitStack() as stack:
        for step in steps:
            if step.server != active_server:
                stack.close()
                active_server = step.server
                if active_server:
                    stack.enter_context(server(active_server, assets, binary, port, env, scratch))
            print(f"BEGIN {step.name}", flush=True)
            started = time.monotonic()
            output = run_command(step.argv, dict(env, **step.env), assets, step.timeout)
            if step.ce_context is not None:
                value = parse_ce(output)
                previous = metrics.setdefault(step.ce_context, value)
                if value != previous:
                    raise CheckFailure(f"Reference/exact paged CE mismatch at context {step.ce_context}: {previous} vs {value}.")
            print(f"PASS {step.name} ({time.monotonic() - started:.1f}s)", flush=True)
    print("All selected CUDA correctness checks passed. No performance thresholds were evaluated.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default="smoke")
    parser.add_argument("--list-commands", action="store_true", help="print the exact plan with placeholders; execute nothing")
    parser.add_argument("--preflight-only", action="store_true", help="validate prerequisites only; does not run correctness tests")
    parser.add_argument("--port", type=int, default=18081, help="dedicated loopback test port (full mode)")
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be between 1024 and 65535")
    if args.list_commands:
        assets = {key: f"<{key}>" for key in ASSETS}
        previous = None
        for step in suite_plan(args.mode, assets, "<CARGO_TARGET_DIR>/release/llm-engine", "<temporary directory>", args.port):
            if step.server != previous:
                if previous:
                    print(f"STOP {previous} server")
                previous = step.server
                if previous:
                    env = server_env(previous, {})
                    print(f"START {previous} server: " + shlex.join(tuple(f"{k}={v}" for k, v in env.items()) +
                          server_command(previous, assets, "<CARGO_TARGET_DIR>/release/llm-engine", args.port)))
            print(step.name + ": " + shlex.join(tuple(f"{k}={v}" for k, v in step.env.items()) + step.argv))
        if previous:
            print(f"STOP {previous} server")
        print("Plan only: no prerequisite checks or CUDA tests executed.")
        return 0
    assets = {}
    try:
        assets = required_assets(args.mode, dict(os.environ))
        env = runtime_env(dict(os.environ))
        target = Path(env.get("CARGO_TARGET_DIR", str(ROOT / "engine/target")))
        if not target.is_absolute():
            target = ROOT / target
        env["CARGO_TARGET_DIR"] = str(target)
        preflight(args.mode, assets, env, target)
        if args.preflight_only:
            return 0
        binary = str(target / "release/llm-engine")
        with tempfile.TemporaryDirectory(prefix="crucible-gpu-ci-") as directory:
            scratch = Path(directory)
            steps = suite_plan(args.mode, assets, binary, directory, args.port)
            execute(steps, assets, env, binary, scratch, args.port)
        return 0
    except (CheckFailure, OSError, ValueError, KeyboardInterrupt) as error:
        print("GPU CI FAILED: " + redact(str(error), assets), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(main())
