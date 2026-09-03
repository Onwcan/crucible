#!/usr/bin/env bash
# Start the service, run the HTTP test suite against it, stop it.
set -uo pipefail

MODEL=${1:-/home/onur/llm-lab/export/120m}
TOK=${2:-/home/onur/llm-lab/export/gpt2.tok}
PORT=${3:-8181}
SCRIPTS=/mnt/c/Users/Kullanici/Desktop/crucible/scripts
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
cd "$HOME/crucible-engine"

./target/release/llm-engine serve "$MODEL" --tokenizer "$TOK" --port "$PORT" \
  > /tmp/crucible-serve.log 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null' EXIT

for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then break; fi
  sleep 1
done

python3 "$SCRIPTS/test_serve.py" --port "$PORT"
RC=$?
kill $PID 2>/dev/null
exit $RC
