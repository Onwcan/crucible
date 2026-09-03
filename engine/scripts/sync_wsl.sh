#!/usr/bin/env bash
# Copy the Windows-side working tree into the WSL build directory.
#
# The repository lives on the Windows filesystem (that is where git and the
# editor are) but nvcc, the driver and the model files are in WSL, and building
# across /mnt/c is slow enough to be worth avoiding. This is the one-way copy.
set -euo pipefail

SRC=/mnt/c/Users/Kullanici/Desktop/crucible/engine
DST="$HOME/crucible-engine"

mkdir -p "$DST/src/tui" "$DST/kernels" "$DST/tests" "$DST/scripts"
cp -f "$SRC"/src/*.rs "$DST/src/"
cp -f "$SRC"/src/tui/*.rs "$DST/src/tui/"
cp -f "$SRC"/kernels/*.cu "$DST/kernels/"
cp -f "$SRC"/tests/*.rs "$DST/tests/"
cp -f "$SRC"/Cargo.toml "$DST/"
echo "synced to $DST"
