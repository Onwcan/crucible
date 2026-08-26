#!/usr/bin/env bash
# Environment bootstrap for LLM training + CUDA kernel work.
# Target: WSL2 Ubuntu 26.04, NVIDIA RTX PRO 4000 Blackwell (sm_120).
#
# Run INSIDE WSL:  bash setup_wsl.sh
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/llm-lab}"
PY_VERSION="3.12"

log() { echo -e "\n\033[1;36m=== $* ===\033[0m"; }

log "[1/5] Build toolchain"
sudo apt-get update -qq
sudo apt-get install -y build-essential git curl wget ca-certificates ninja-build cmake

log "[2/5] uv + managed Python ${PY_VERSION}"
# Ubuntu 26.04 ships ONLY Python 3.14, which PyTorch does not yet publish wheels for.
# uv installs a standalone 3.12 independent of the distro.
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv python install "${PY_VERSION}"

log "[3/5] CUDA toolkit (WSL build)"
# CRITICAL: install the toolkit ONLY. Never install a driver inside WSL -
# the Windows driver (596.86) already provides the GPU via /dev/dxg.
if ! command -v nvcc >/dev/null 2>&1; then
  cd /tmp
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo apt-get update -qq
  # sm_120 requires CUDA >= 12.8. Prefer 13.x, fall back to 12.8.
  sudo apt-get install -y cuda-toolkit-13-0 \
    || sudo apt-get install -y cuda-toolkit-12-8
fi

CUDA_HOME_GUESS="$(ls -d /usr/local/cuda-1[23]* 2>/dev/null | sort -V | tail -1 || true)"
if [ -n "$CUDA_HOME_GUESS" ] && ! grep -q 'CUDA_HOME' "$HOME/.bashrc"; then
  {
    echo ""
    echo "# --- CUDA (added by setup_wsl.sh) ---"
    echo "export CUDA_HOME=${CUDA_HOME_GUESS}"
    echo 'export PATH=$CUDA_HOME/bin:$HOME/.local/bin:$PATH'
    echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH'
  } >> "$HOME/.bashrc"
  export CUDA_HOME="$CUDA_HOME_GUESS"
  export PATH="$CUDA_HOME/bin:$PATH"
fi

log "[4/5] Project venv + PyTorch for sm_120"
mkdir -p "$PROJECT_DIR" && cd "$PROJECT_DIR"
uv venv --python "${PY_VERSION}"
PY="$PROJECT_DIR/.venv/bin/python"

# Blackwell sm_120 needs a cu128+ build. Try newest index first.
uv pip install --python "$PY" torch --index-url https://download.pytorch.org/whl/cu130 \
  || uv pip install --python "$PY" torch --index-url https://download.pytorch.org/whl/cu128 \
  || uv pip install --python "$PY" --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

uv pip install --python "$PY" \
  numpy transformers datasets tokenizers tiktoken matplotlib pytest tqdm

log "[5/5] Verification"
"$PY" - <<'PYCHECK'
import torch
print(f"torch {torch.__version__} / cuda {torch.version.cuda}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"capability: sm_{''.join(map(str, torch.cuda.get_device_capability()))}")
PYCHECK

echo ""
echo "Done. Next:"
echo "  source ~/.bashrc"
echo "  cd $PROJECT_DIR && .venv/bin/python verify_gpu.py"
