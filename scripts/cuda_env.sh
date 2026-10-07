#!/usr/bin/env bash
# CUDA 13 toolchain + JAX GPU defaults for this project.
# JAX can discover the GPU plugin without this file, but XLA also needs
# ptxas, nvlink, and libdevice when compiling GPU programs.
#
# Usage:
#   source scripts/cuda_env.sh          # enable GPU-friendly env
#   JAX_PLATFORMS=cpu source ...        # force CPU before/after as needed
#
# RTX 5070 ~8GB: keep preallocate off + mem fraction <1 to avoid OOM at import.
set -euo pipefail

CUDA_ENV="${SAFE_MARL_ENV:-/home/xiang/miniconda3/envs/safe_marl_verify}"
export PATH="$CUDA_ENV/lib/python3.11/site-packages/nvidia/cu13/bin:$PATH"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_ENV/lib/python3.11/site-packages/nvidia/cu13${XLA_FLAGS:+ $XLA_FLAGS}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
# Cap JAX GPU memory (fraction of free VRAM). Override: XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}"
# Do not force platform here — leave JAX_PLATFORMS to the caller/script.
# Unset empty CUDA_VISIBLE_DEVICES if user wants all GPUs.
if [ "${FORCE_CPU:-0}" = "1" ]; then
  export JAX_PLATFORMS=cpu
  export CUDA_VISIBLE_DEVICES=""
fi
