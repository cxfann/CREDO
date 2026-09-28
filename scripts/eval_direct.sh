#!/usr/bin/env bash
# Direct evaluation pipeline for ONE model (code / budget-limited protocol).
#   stage 1: single-pass generation at the budget recorded in the eval config (the paper's
#            coding budget is 6144) over the dataset suite (evaluation.py; scoring by
#            confidence_verifier)
#   stage 2: macro aggregation (aggregate_direct.py) -> results/<label>/direct_summary.json
#
# Direct semantics: NO salvage, NO pooling. Truncated / non-answering samples enter
# the metric pool as (label 0, confidence 0.0) via confidence_verifier's own logic.
#
# Usage:
#   bash scripts/eval_direct.sh <label> [num_parallel_gpus]
# Prereq: eval_configs/<label>/<ds>.json exist
#         (scripts/make_eval_configs.py --protocol direct [--domain code] ...).
# Env:
#   EVAL_DATASETS  space-separated subset override (default: full 7-set math suite;
#                  code runs pass the four code sets)
#   EVAL_BUDGET    generation max_tokens, recorded into the summary metadata only; the
#                  budget actually used is the one in the eval config
set -uo pipefail
LABEL="${1:?usage: eval_direct.sh <label> [num_parallel_gpus]}"
NPAR="${2:-1}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON_BIN:-python}"
export VLLM_USE_V1=0
export TOKENIZERS_PARALLELISM=false
DEFAULT_DATASETS="aime24 aime25 aime26 amc23 amc24 math-500 deepscaler-test"
read -r -a DATASETS <<< "${EVAL_DATASETS:-$DEFAULT_DATASETS}"
LOGDIR="logs/direct_pipeline/$LABEL"
mkdir -p "$LOGDIR"

run_ds() { # dataset gpu
  local ds="$1" gpu="$2" cfg="eval_configs/$LABEL/$1.json" rc
  [ -f "$cfg" ] || { echo "MISSING_CONFIG $cfg"; return 3; }
  echo "GEN_START $(date -Is) ds=$ds gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" evaluation.py --config "$cfg" > "$LOGDIR/gen_$ds.log" 2>&1
  rc=$?
  [ $rc -ne 0 ] && { echo "GEN_FAILED ds=$ds rc=$rc log=$LOGDIR/gen_$ds.log"; return $rc; }
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
[ $fail -ne 0 ] && { echo "DIRECT_PIPELINE_FAILED label=$LABEL"; exit 1; }

"$PY" scripts/aggregate_direct.py --label "$LABEL" --datasets "${DATASETS[@]}" \
  --budget "${EVAL_BUDGET:-}" || { echo "DIRECT_AGGREGATE_FAILED label=$LABEL"; exit 1; }
echo "DIRECT_PIPELINE_DONE $(date -Is) label=$LABEL summary=results/$LABEL/direct_summary.json"
