#!/usr/bin/env bash
# Build causal-conv1d 1.7.0 against this venv's torch (2.10.0+cu128) on a machine whose system CUDA toolkit does
# not match (e.g. CUDA 13.x): assembles a private CUDA 12.8 toolchain from NVIDIA's nvcc redistributable plus
# the cudart/CCCL headers already shipped in the venv, applies the glibc 2.41 header fix, then builds.
#
# Tested: Arch Linux, glibc 2.41+, gcc-14 (nvcc 12.8 rejects gcc 15/16), RTX PRO 6000 Blackwell (sm_120),
# Python 3.12 venv made by uv. Takes ~10 minutes (the package compiles for sm_70..sm_120).
#
# Usage:  bash vision/setup_kernels.sh            (from the repo root, venv at .venv)
# Env:    VENV=.venv  CUDA_DIR=~/.local/cuda-12.8  CC_BIN=gcc-14  MAX_JOBS=16  PIP="uv pip install --python $VENV/bin/python"
set -euo pipefail
VENV=${VENV:-.venv}
PY=$VENV/bin/python
CUDA_DIR=${CUDA_DIR:-$HOME/.local/cuda-12.8}
CC_BIN=${CC_BIN:-gcc-14}
NVCC_VER=12.8.93
PIP=${PIP:-"uv pip install --python $PY"}

echo "== 1/4 nvcc $NVCC_VER redistributable -> $CUDA_DIR"
if [ ! -x "$CUDA_DIR/bin/nvcc" ]; then
  mkdir -p "$CUDA_DIR"
  curl -sSL "https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-$NVCC_VER-archive.tar.xz" \
    | tar xJ -C "$CUDA_DIR" --strip-components=1
fi
"$CUDA_DIR/bin/nvcc" --version | tail -1

echo "== 2/4 cudart + CCCL headers and libcudart from the venv's NVIDIA wheels"
$PIP "nvidia-cuda-cccl-cu12==12.8.90" ninja setuptools packaging >/dev/null
NV=$($PY -c "import nvidia, os; print(list(nvidia.__path__)[0])")
mkdir -p "$CUDA_DIR/lib64"
ln -sfn "$NV/cuda_runtime/lib/libcudart.so.12" "$CUDA_DIR/lib64/libcudart.so"
ln -sfn "$NV/cuda_runtime/lib/libcudart.so.12" "$CUDA_DIR/lib64/libcudart.so.12"
for f in "$NV"/cuda_runtime/include/* "$NV"/cuda_cccl/include/*; do ln -sfn "$f" "$CUDA_DIR/include/$(basename "$f")"; done

echo "== 3/4 glibc 2.41 fix: cospi/sinpi/rsqrt exception specs in crt/math_functions.h (CUDA 12.9 fixed this upstream)"
H="$CUDA_DIR/include/crt/math_functions.h"
[ -f "$H.orig" ] || cp "$H" "$H.orig"
sed -i -E 's/^(extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ (double|float) +(cospi|sinpi|rsqrt)f?\([a-z]+ x\));$/\1 noexcept(true);/' "$H"

echo "== 4/4 build causal-conv1d 1.7.0 (no build isolation, so it links this venv's torch)"
CUDA_HOME="$CUDA_DIR" PATH="$CUDA_DIR/bin:$PATH" NVCC_PREPEND_FLAGS="-ccbin $CC_BIN" CC="$CC_BIN" CXX="${CC_BIN/gcc/g++}" \
CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS="${MAX_JOBS:-16}" \
  $PIP --no-build-isolation --no-binary causal-conv1d "causal-conv1d==1.7.0"

$PY - <<'PY'
import torch, torch.nn.functional as F
from causal_conv1d import causal_conv1d_fn
x = torch.randn(2, 8, 16, device="cuda", dtype=torch.bfloat16); w = torch.randn(8, 4, device="cuda", dtype=torch.bfloat16)
y = causal_conv1d_fn(x, w, None, activation="silu")
ref = F.silu(F.conv1d(x, w.unsqueeze(1), padding=3, groups=8)[..., :16])
print("causal_conv1d OK, max diff vs reference:", (y.float() - ref.float()).abs().max().item())
PY
