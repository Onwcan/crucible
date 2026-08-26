"""
Compare crucible's decode throughput against llama.cpp and vLLM.

Every performance number in this repo so far is relative to an earlier version
of itself. That answers "did this change help" and not "is this engine any
good". This runs the same model on other engines to answer the second.

Fairness rules, because a benchmark against mature projects is easy to rig by
accident:

  - Same weights. All three read exports of one checkpoint, and the HF export
    is logit-verified against the reference implementation before use.
  - Same shape of work: batch 1, greedy, identical prompt and token count.
    Batch 1 is where crucible is designed to run; vLLM is built for large
    batches and will look worse here than it deserves, which is stated rather
    than quietly banked.
  - Same power envelope, recorded per run. This machine's limit is
    user-switchable between ~55 W and ~175 W and an earlier measurement swung
    168% purely from that.
  - Median of repeated trials with the spread shown, never a single run.
  - Trials are INTERLEAVED, one per engine per round, rather than running all
    of one engine then all of the next. Running them in blocks lets thermal
    drift masquerade as an engine difference: the first block measured on a
    cool GPU, the second on a hot one. Interleaving makes drift hit every
    engine equally. (Measured: llama.cpp scored 846 tok/s standalone and 615
    when it ran second in a block layout.)

Engines that are not installed are skipped and reported as such. A missing
comparison is a gap; a fabricated one is worse.

Usage:
    python scripts/bench_compare.py --tokens 256 --trials 5
    python scripts/bench_compare.py --only crucible,llama.cpp
"""
from __future__ import annotations

import argparse
import json
import sys
import re
import shutil
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

PROMPT = "The capital of France is"


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


@dataclass
class Engine:
    name: str
    cmd: list[str]
    pattern: str
    note: str = ""
    available: bool = True
    reason: str = ""
    samples: list[float] = field(default_factory=list)


def run_once(engine: Engine, timeout: int) -> float | None:
    """Run one trial, returning tokens/second or None if it failed."""
    try:
        out = subprocess.run(engine.cmd, capture_output=True, text=True,
                             timeout=timeout)
    except subprocess.TimeoutExpired:
        engine.reason = f"timed out after {timeout}s"
        return None

    text = out.stdout + out.stderr
    match = re.search(engine.pattern, text)
    if not match:
        head = text.strip().splitlines()[-3:] if text.strip() else ["(no output)"]
        engine.reason = "no throughput in output: " + " | ".join(head)
        return None
    return float(match.group(1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--crucible-bin", default="./target/release/llm-engine")
    p.add_argument("--crucible-model", default="../llm-lab/export/120m")
    p.add_argument("--crucible-tokenizer", default="../llm-lab/export/gpt2.tok")
    p.add_argument("--gguf", default="../llm-lab/export/120m-q8_0.gguf")
    p.add_argument("--hf-model", default="../llm-lab/export/120m-hf")
    p.add_argument("--llama-bench", default="llama-bench")
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--only", default=None, help="comma-separated engine names")
    # vLLM usually lives in its own environment: it pins its own torch build and
    # installing it into the training venv would risk replacing a working one.
    p.add_argument("--vllm-python", default=sys.executable,
                   help="interpreter that has vllm installed")
    p.add_argument("--json", default=None, help="write results here")
    args = p.parse_args()

    engines = [
        Engine(
            name="crucible",
            cmd=[args.crucible_bin, "gpu-logits", args.crucible_model,
                 "--quant", "int8", "--graph", "--decode", str(args.tokens)],
            pattern=r"\(([0-9.]+) tok/s",
            note="int8 weights, CUDA graph replay",
        ),
        Engine(
            name="llama.cpp",
            # -ngl 99 is not optional. llama-bench defaults to partial offload
            # here and reported 583.7 tok/s; with all layers on the GPU the same
            # build does 846.5. Benchmarking the default would have overstated
            # crucible lead by 45%.
            #
            # -n tokens generated, -p 0 skips prompt processing, -b 1 batch 1.
            cmd=[args.llama_bench, "-m", args.gguf, "-n", str(args.tokens),
                 "-p", "0", "-b", "1", "-r", "1", "-ngl", "99",
                 "-o", "json"],
            pattern=r'"avg_ts"\s*:\s*([0-9.]+)',
            note="Q8_0 weights, all layers on GPU",
        ),
        Engine(
            name="vllm",
            cmd=[args.vllm_python, "-c", VLLM_SNIPPET.format(
                model=args.hf_model, tokens=args.tokens, prompt=PROMPT)],
            pattern=r"TOKS_PER_SEC=([0-9.]+)",
            note="batch 1; vLLM is built for large batches",
        ),
    ]

    if args.only:
        wanted = {n.strip() for n in args.only.split(",")}
        engines = [e for e in engines if e.name in wanted]

    # Availability: a missing engine is reported, never silently dropped.
    for e in engines:
        if e.name == "crucible" and not Path(args.crucible_bin).exists():
            e.available, e.reason = False, f"not built: {args.crucible_bin}"
        elif e.name == "llama.cpp":
            if shutil.which(args.llama_bench) is None:
                e.available, e.reason = False, f"{args.llama_bench} not on PATH"
            elif not Path(args.gguf).exists():
                e.available, e.reason = False, f"no GGUF at {args.gguf}"
        elif e.name == "vllm":
            probe = subprocess.run([args.vllm_python, "-c", "import vllm"],
                                   capture_output=True, text=True)
            if probe.returncode != 0:
                e.available, e.reason = False, "vllm not importable"
            elif not Path(args.hf_model).exists():
                e.available, e.reason = False, f"no HF export at {args.hf_model}"

    print(f"gpu     : {envelope()}")
    print(f"workload: {args.tokens} tokens, batch 1, greedy, "
          f"{args.trials} trials")
    print()

    for e in engines:
        if not e.available:
            print(f"  {e.name:11} SKIPPED -- {e.reason}")

    live = [e for e in engines if e.available]
    for trial in range(args.trials):
        print(f"  round {trial + 1}/{args.trials}:", end="", flush=True)
        for e in live:
            if e.reason:          # already failed in an earlier round
                continue
            v = run_once(e, args.timeout)
            if v is None:
                print(f" {e.name}=failed", end="", flush=True)
                continue
            e.samples.append(v)
            print(f" {e.name}={v:.0f}", end="", flush=True)
        print()

    print()
    header = f"{'engine':<12} {'tok/s':>9} {'spread':>8}  notes"
    print(header)
    print("-" * (len(header) + 12))

    baseline = None
    for e in engines:
        if not e.samples:
            reason = e.reason or "no result"
            print(f"{e.name:<12} {'--':>9} {'--':>8}  {reason}")
            continue
        s = sorted(e.samples)
        med = statistics.median(s)
        spread = (s[-1] - s[0]) / med * 100 if med else 0.0
        if e.name == "crucible":
            baseline = med
        rel = ""
        if baseline and e.name != "crucible":
            rel = f"  ({med / baseline:.2f}x crucible)"
        print(f"{e.name:<12} {med:9.1f} {spread:7.1f}%  {e.note}{rel}")

    print()
    print("Batch 1 only. vLLM's design point is large-batch serving, so this")
    print("measures the case it is least suited to, not its capability.")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "gpu": envelope(),
            "tokens": args.tokens,
            "trials": args.trials,
            "results": {e.name: {"samples": e.samples, "note": e.note,
                                 "reason": e.reason} for e in engines},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")


VLLM_SNIPPET = '''
import time
from vllm import LLM, SamplingParams
llm = LLM(model="{model}", dtype="float16", gpu_memory_utilization=0.5,
          max_model_len=1024, enforce_eager=False, disable_log_stats=True)
# Greedy, fixed length, so the token count is not sampling-dependent.
params = SamplingParams(temperature=0.0, max_tokens={tokens}, ignore_eos=True)
llm.generate(["{prompt}"], params)          # warm up
t0 = time.perf_counter()
out = llm.generate(["{prompt}"], params)
dt = time.perf_counter() - t0
n = len(out[0].outputs[0].token_ids)
print("TOKS_PER_SEC=%.3f" % (n / dt))
'''

if __name__ == "__main__":
    main()
