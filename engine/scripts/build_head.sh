#!/usr/bin/env bash
# Build the last committed engine into ~/crucible-head, for before/after
# benchmarking on the same machine in the same session.
#
# Comparing against numbers recorded on another day is comparing across power
# envelopes; this builds the baseline so both sides can be measured now.
set -euo pipefail

TAR="$1"
DST="$HOME/crucible-head"
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda

rm -rf "$DST"
mkdir -p "$DST/extract"
tar -xf "$TAR" -C "$DST/extract"
# The archive keeps the repository layout; the crate is the engine directory.
cp -r "$DST/extract/engine/." "$DST/"
rm -rf "$DST/extract"
cd "$DST"
cargo build --release --features cuda 2>&1 | tail -3
echo "built $DST/target/release/llm-engine"
