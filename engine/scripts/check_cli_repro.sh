#!/usr/bin/env bash
# Two identical sampled generations must produce identical text, and a
# different seed must not.
set -euo pipefail

MODEL=${1:-/home/onur/llm-lab/export/120m}
TOK=${2:-/home/onur/llm-lab/export/gpt2.tok}
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
cd "$HOME/crucible-engine"

gen() {
  ./target/release/llm-engine generate "$MODEL" --tokenizer "$TOK" --gpu --graph \
    --prompt "Once upon a time" --max-tokens 32 --temperature 0.8 --top-k 40 --seed "$1" \
    2>/dev/null | grep -v "^prefill\|^decode\|^device"
}

A=$(gen 4242)
B=$(gen 4242)
C=$(gen 99)
[ "$A" = "$B" ] && echo "ok    same seed reproduces" || { echo "FAIL  same seed differed"; exit 1; }
[ "$A" != "$C" ] && echo "ok    a different seed differs" || { echo "FAIL  seeds collided"; exit 1; }
echo
echo "seed 4242: $A"
