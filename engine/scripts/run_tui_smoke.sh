#!/usr/bin/env bash
# Start the service, drive the TUI against it in a pty, stop the service.
set -uo pipefail

MODEL=${1:-/home/onur/llm-lab/export/120m}
TOK=${2:-/home/onur/llm-lab/export/gpt2.tok}
PORT=${3:-8182}
SCRIPTS=/mnt/c/Users/Kullanici/Desktop/crucible/scripts
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
cd "$HOME/crucible-engine"

./target/release/llm-engine serve "$MODEL" --tokenizer "$TOK" --port "$PORT" \
  > /tmp/crucible-serve-tui.log 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null' EXIT

for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then break; fi
  sleep 1
done

python3 "$SCRIPTS/smoke_tui.py" --binary ./target/release/llm-engine --port "$PORT"
RC=$?
kill $PID 2>/dev/null
exit $RC
