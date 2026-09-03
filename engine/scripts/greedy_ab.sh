#!/usr/bin/env bash
# Interleaved A/B of the greedy decode path: last commit against the working
# tree.
#
# The top-k launch sits in the decode graph unconditionally, so a greedy step
# now runs one kernel that immediately exits. This is what says whether that
# costs anything.
#
# Uses gpu-serve-bench, which decodes greedily and nothing else -- gpu-sample-bench
# would measure greedy after a different amount of sampled work on each side,
# and the GPU does not forget what it just ran. Rounds alternate which binary
# goes first, so ordering cannot favour either.
set -euo pipefail

MODEL=${1:-/home/onur/llm-lab/export/120m}
ROUNDS=${2:-4}
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"

run() {
  local side=$1 round=$2 dir
  if [ "$side" = head ]; then dir="$HOME/crucible-head"; else dir="$HOME/crucible-engine"; fi
  local out env
  out=$("$dir/target/release/llm-engine" gpu-serve-bench "$MODEL" \
          --batches 1,4,8,16 --steps 64 --trials 5 2>&1)
  env=$(echo "$out" | grep -m1 -i "laptop gpu" | sed "s/.*Laptop GPU, //")
  # Columns: batch, logits agg, t/s, argmax agg, t/s, ... The argmax column is
  # the greedy production path.
  echo "$out" | awk -v s="$side" -v r="$round" -v e="$env" \
    '$1 ~ /^(1|4|8|16)$/ && $9 ~ /x$/ { printf "round %s  %-4s  batch %2s  %6s tok/s  spread %s   [%s]\n", r, s, $1, $4, $9, e }'
}

for r in $(seq 1 "$ROUNDS"); do
  if [ $((r % 2)) -eq 1 ]; then
    run head "$r"; run new "$r"
  else
    run new "$r"; run head "$r"
  fi
done
