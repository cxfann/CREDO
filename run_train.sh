#!/bin/bash
set -eo pipefail

# Launch a training run with accelerate + DeepSpeed ZeRO-2.
#   bash run_train.sh <config> <num_processes|auto> <cuda_visible_devices|-> <logfile|-> [deepspeed_config] [extra args...]
# Extra args after the 5th positional are forwarded to rl_runner.py
# (e.g. --model_name_or_path / --dataset_name / --output_dir / --run_name overrides).
#
# Optional environment overrides:
#   PYTHON_BIN      python interpreter        (default: python on PATH)
#   ACCELERATE_BIN  accelerate launcher       (default: accelerate on PATH)
#   HF_HOME         HuggingFace cache dir     (default: <repo>/.hf_cache)
#   WANDB_PROJECT   Weights & Biases project  (default: CREDO)
#   MAIN_PORT       accelerate main port      (default: 29500)
#   CUDA_HOME       CUDA toolkit root         (a version-report shim is installed if absent;
#                                              see the block below)

CONFIG="$1"
NPROC="$2"
CVD="$3"
LOGFILE="$4"
if [ -n "$5" ] && [ "${5#--}" = "$5" ]; then
  DEEPSPEED_CONFIG="$5"
  EXTRA_ARGS=("${@:6}")
else
  DEEPSPEED_CONFIG="deepspeed.yaml"
  EXTRA_ARGS=("${@:5}")
fi

if [ -z "$CONFIG" ] || [ -z "$NPROC" ] || [ -z "$CVD" ] || [ -z "$LOGFILE" ]; then
  echo "usage: bash run_train.sh <config> <num_processes|auto> <cuda_visible_devices|-> <logfile|-> [deepspeed_config] [extra rl_runner args...]"; exit 2
fi

# Resolve local modules and configuration paths from the repository root.
cd "$(cd "$(dirname "$0")" && pwd)"

export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export WANDB_PROJECT="${WANDB_PROJECT:-CREDO}"
export WANDB_DIR="${WANDB_DIR:-$PWD}"
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0

if [ "$CVD" != "-" ]; then
  export CUDA_VISIBLE_DEVICES="$CVD"
fi

# num_processes=auto -> number of visible GPUs (falls back to 1)
if [ "$NPROC" = "auto" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC="$(nvidia-smi -L | wc -l | tr -d ' ')"
  else
    NPROC=1
  fi
fi

PY="${PYTHON_BIN:-python}"
ACCELERATE="${ACCELERATE_BIN:-accelerate}"

# DeepSpeed probes for a CUDA toolkit at import; the pip wheels ship none, so install a
# version-only nvcc shim when absent (this stack never compiles a CUDA op). No-op if a real
# toolkit or CUDA_HOME already exists.
if [ -z "${CUDA_HOME:-}" ] && [ ! -x /usr/local/cuda/bin/nvcc ] && ! command -v nvcc >/dev/null 2>&1; then
  mkdir -p /tmp/credo_cuda_shim/bin
  cat > /tmp/credo_cuda_shim/bin/nvcc <<'NVCC_SHIM'
#!/bin/sh
if [ "$1" = "-V" ] || [ "$1" = "--version" ]; then
  echo "nvcc: NVIDIA (R) Cuda compiler driver"
  echo "Cuda compilation tools, release 12.4, V12.4.131"
  exit 0
fi
echo "FATAL: this nvcc is a version-report shim (no CUDA toolkit installed); real compilation attempted: $*" >&2
exit 1
NVCC_SHIM
  chmod +x /tmp/credo_cuda_shim/bin/nvcc
  export CUDA_HOME=/tmp/credo_cuda_shim DS_SKIP_CUDA_CHECK=1
  echo "[run_train] no CUDA toolkit found; installed an nvcc version shim at $CUDA_HOME (reports 12.4 to match the cu124 torch wheel)"
fi

echo "[run_train] config=$CONFIG nproc=$NPROC CUDA=${CVD} log=$LOGFILE deepspeed=$DEEPSPEED_CONFIG"

CMD=("$ACCELERATE" launch
  --num_processes "$NPROC"
  --main_process_port "${MAIN_PORT:-29500}"
  --config_file "$DEEPSPEED_CONFIG"
  rl_runner.py --config "$CONFIG")
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

if [ "$LOGFILE" = "-" ]; then
  exec "${CMD[@]}"
else
  mkdir -p "$(dirname "$LOGFILE")"
  "${CMD[@]}" 2>&1 | tee "$LOGFILE"
fi
