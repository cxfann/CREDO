#!/usr/bin/env bash
# Salvage evaluation pipeline for ONE model (mathematics protocol).
#   stage 1: fresh generation @12288 over the 7-set suite (evaluation.py)
#   stage 2: salvage of unfinished completions (salvage.py: deterministic boxed
#            continuation + strict scoring)
#   stage 3: aggregation with assertions (aggregate_salvage.py) ->
#            results/salvage/<label>/salvage_summary.json
#
# The reported accuracy is acc_anytime: every sample enters the calibration pool,
# with residual (still-unfinished) samples scored as (label 0, confidence 0).
#
# Usage:
#   bash scripts/eval_salvage.sh <label> [num_parallel_gpus]
# Prereq: eval_configs/salvage/<label>/<ds>.json exist
#         (scripts/make_eval_configs.py --protocol salvage ...).
# num_parallel_gpus (default 1): datasets are dispatched one per GPU id 0..N-1 in
# rotation; each worker holds one full model instance, so N must not exceed the
# number of visible GPUs.
set -uo pipefail
LABEL="${1:?usage: eval_salvage.sh <label> [num_parallel_gpus]}"
NPAR="${2:-1}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON_BIN:-python}"
export VLLM_USE_V1=0
export TOKENIZERS_PARALLELISM=false
DATASETS=(aime24 aime25 aime26 amc23 amc24 math-500 deepscaler-test)
LOGDIR="logs/salvage_pipeline/$LABEL"
mkdir -p "$LOGDIR"

run_ds() { # dataset gpu
  local ds="$1" gpu="$2" cfg="eval_configs/salvage/$LABEL/$1.json" rc
  [ -f "$cfg" ] || { echo "MISSING_CONFIG $cfg"; return 3; }
  echo "GEN_START $(date -Is) ds=$ds gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" evaluation.py --config "$cfg" > "$LOGDIR/gen_$ds.log" 2>&1
  rc=$?
  [ $rc -ne 0 ] && { echo "GEN_FAILED ds=$ds rc=$rc log=$LOGDIR/gen_$ds.log"; return $rc; }
  echo "SALVAGE_START $(date -Is) ds=$ds gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/salvage.py --config "$cfg" \
    --output-store "eval_outputs/salvage-fill/$LABEL/$ds" \
    --results-path "results/salvage-fill/$LABEL/$ds" \
    --overwrite > "$LOGDIR/salvage_$ds.log" 2>&1
  rc=$?
  [ $rc -ne 0 ] && { echo "SALVAGE_FAILED ds=$ds rc=$rc log=$LOGDIR/salvage_$ds.log"; return $rc; }
  echo "DS_DONE $(date -Is) ds=$ds"
}

fail=0
if [ "$NPAR" -le 1 ]; then
  for ds in "${DATASETS[@]}"; do run_ds "$ds" 0 || fail=1; done
else
  pids=()
  i=0
  for ds in "${DATASETS[@]}"; do
    run_ds "$ds" $((i % NPAR)) &
    pids+=($!)
    i=$((i + 1))
    while [ "$(jobs -rp | wc -l)" -ge "$NPAR" ]; do sleep 20; done
  done
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
fi
[ $fail -ne 0 ] && { echo "SALVAGE_PIPELINE_FAILED label=$LABEL"; exit 1; }

"$PY" scripts/aggregate_salvage.py --label "$LABEL" || { echo "SALVAGE_AGGREGATE_FAILED label=$LABEL"; exit 1; }
echo "SALVAGE_PIPELINE_DONE $(date -Is) label=$LABEL summary=results/salvage/$LABEL/salvage_summary.json"
