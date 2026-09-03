#!/usr/bin/env bash
# Run a command in the WSL build directory with the toolchain on PATH.
#
#   bash wsl_run.sh cargo test
#
# Exists because the non-login shell used to drive WSL from the Windows side
# does not source the rustup or CUDA profile scripts.
set -euo pipefail

export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
cd "$HOME/crucible-engine"
exec "$@"
