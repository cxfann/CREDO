#!/usr/bin/env bash
# Train a model, then evaluate the produced checkpoint --- in one command.
#
#   stage 1: run_train.sh <config>   (DeepSpeed ZeRO-2, multi-GPU)
#   stage 2: on training success, build eval configs for the produced model and run
#            the evaluation pipeline for the chosen protocol:
#              - salvage (math): regenerate at a large budget, salvage any unfinished
#                completions, then aggregate            (scripts/eval_salvage.sh)
#              - direct  (code): single-pass generation within budget; truncated
#                samples score as incorrect             (scripts/eval_direct.sh)
#
# Usage:
#   bash scripts/train_then_eval.sh <train_config> <label> [extra rl_runner args...]
# Example:
#   bash scripts/train_then_eval.sh configs/credo_math.yaml credo_math_r1
#
# Environment:
#   PYTHON_BIN      python interpreter (default: python on PATH)
#   OUTPUT_DIR      training output / eval model dir (default: outputs/<label>)
#   NPROC           training num_processes (default: auto = visible GPU count)
#   EVAL_NPAR       parallel GPUs for the eval stage (default: 1)
#   EVAL_PROTOCOL   salvage (default; math), direct (code / budget-limited), or both
#   EVAL_MAX_TOKENS override the eval generation budget (default: the paper budget for the
#                   protocol x domain pair --- salvage 12288, direct+code 6144,
#                   direct+math 4096). Under "both" it drives direct only.
#   EVAL_SEED       eval sampling seed (default: 42). The paper pairs the eval seed with
#                   the checkpoint's training seed, so a run trained at --seed 44 should be
#                   evaluated with EVAL_SEED=44.
#   EVAL_SYS_PROMPT override the eval sys_prompt_name (e.g. the no-analysis ablation
#                   bac_boxed_credo_noct); empty keeps the method routing
#   EVAL_DATASETS   (direct only) space-separated dataset subset, e.g. "aime25"
set -uo pipefail
TRAIN_CONFIG="${1:?usage: train_then_eval.sh <train_config> <label> [extra rl_runner args...]}"
LABEL="${2:?missing <label>}"
shift 2
EXTRA_ARGS=("$@")

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON_BIN:-python}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/$LABEL}"
EVAL_NPAR="${EVAL_NPAR:-1}"
NPROC="${NPROC:-auto}"
EVAL_PROTOCOL="${EVAL_PROTOCOL:-salvage}"
case "$EVAL_PROTOCOL" in
  direct|salvage|both) ;;
  *) echo "BAD_EVAL_PROTOCOL '$EVAL_PROTOCOL' (must be direct, salvage or both)"; exit 7 ;;
esac
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-}"   # empty = paper budget for this protocol x domain
if [ -n "$EVAL_MAX_TOKENS" ] && ! [[ "$EVAL_MAX_TOKENS" =~ ^[0-9]+$ ]]; then
  echo "BAD_EVAL_MAX_TOKENS '$EVAL_MAX_TOKENS' (must be a positive integer)"; exit 7
fi

[ -f "$TRAIN_CONFIG" ] || { echo "MISSING_TRAIN_CONFIG $TRAIN_CONFIG"; exit 2; }
METHOD="$("$PY" -c "import yaml;print(yaml.safe_load(open('$TRAIN_CONFIG')).get('method',''))")" \
  || { echo "METHOD_DETECT_FAILED config=$TRAIN_CONFIG"; exit 6; }
[ -n "$METHOD" ] || { echo "METHOD_EMPTY config=$TRAIN_CONFIG -- refusing to guess eval routing"; exit 6; }
CREDO_FLAG=(); [ "$METHOD" = "credo" ] && CREDO_FLAG=(--credo)

# Evaluation domain, derived from the config exactly like CREDO_FLAG is derived from method.
ANSWER_FORMAT="$("$PY" -c "import yaml;print(yaml.safe_load(open('$TRAIN_CONFIG')).get('answer_format','math'))")" \
  || { echo "ANSWER_FORMAT_DETECT_FAILED config=$TRAIN_CONFIG"; exit 6; }
DOMAIN_FLAG=()
if [ "$ANSWER_FORMAT" = "code" ]; then
  DOMAIN_FLAG=(--domain code)
  if [ -n "${EVAL_DATASETS:-}" ]; then
    read -r -a _EVAL_DS <<< "$EVAL_DATASETS"
    DOMAIN_FLAG+=(--datasets "${_EVAL_DS[@]}")
  fi
  # the salvage protocol injects a boxed continuation and would mis-score code
  case "$EVAL_PROTOCOL" in
    direct) ;;
    *) echo "CODE_DOMAIN_REQUIRES_DIRECT answer_format=code but EVAL_PROTOCOL=$EVAL_PROTOCOL"; exit 7 ;;
  esac
fi

# Budget shown in the banner and handed to the direct aggregator as metadata. Must mirror
# make_eval_configs.protocol_max_tokens: salvage 12288, direct+code 6144, direct+math 4096.
EFFECTIVE_EVAL_BUDGET="$EVAL_MAX_TOKENS"
if [ -z "$EFFECTIVE_EVAL_BUDGET" ]; then
  case "$EVAL_PROTOCOL" in
    salvage) EFFECTIVE_EVAL_BUDGET=12288 ;;
    *) if [ "$ANSWER_FORMAT" = "code" ]; then EFFECTIVE_EVAL_BUDGET=6144; else EFFECTIVE_EVAL_BUDGET=4096; fi ;;
  esac
fi
echo "==================== LAUNCH PARAMETERS ===================="
echo "[train] config=$TRAIN_CONFIG label=$LABEL method=$METHOD output_dir=$OUTPUT_DIR nproc=$NPROC"
echo "[train] extra rl_runner args: ${EXTRA_ARGS[*]:-<none>}"
echo "[eval ] protocol=$EVAL_PROTOCOL budget=$EFFECTIVE_EVAL_BUDGET npar=$EVAL_NPAR credo_flag=${CREDO_FLAG[*]:-<none>} domain=$ANSWER_FORMAT temp=0.7 seed=${EVAL_SEED:-42}"
echo "============================================================"

# ---- stage 1: train ----
echo "TRAIN_STAGE_START $(date -Is)"
bash run_train.sh "$TRAIN_CONFIG" "$NPROC" - - \
  --output_dir "$OUTPUT_DIR" --run_name "$LABEL" "${EXTRA_ARGS[@]}"
train_rc=$?
[ "$train_rc" -eq 0 ] || { echo "TRAIN_FAILED rc=$train_rc label=$LABEL -- skipping eval"; exit "$train_rc"; }
# final model is written by trainer.save_model at OUTPUT_DIR (weights are *.safetensors
# or pytorch_model*.bin); tokenizer is saved alongside it.
if [ ! -f "$OUTPUT_DIR/config.json" ] || { ! ls "$OUTPUT_DIR"/*.safetensors >/dev/null 2>&1 && ! ls "$OUTPUT_DIR"/pytorch_model*.bin >/dev/null 2>&1; }; then
  echo "TRAIN_NO_MODEL_AT $OUTPUT_DIR -- refusing to eval"; exit 3
fi
echo "TRAIN_STAGE_DONE $(date -Is) model=$OUTPUT_DIR"

# ---- stage 2: evaluate ----
run_eval() { # protocol
  local proto="$1" mec
  mec=(--protocol "$proto" --eval-seed "${EVAL_SEED:-42}")
  [ -n "$EVAL_MAX_TOKENS" ] && mec+=(--max-tokens "$EVAL_MAX_TOKENS")
  [ -n "${EVAL_SYS_PROMPT:-}" ] && mec+=(--sys-prompt "$EVAL_SYS_PROMPT")
  "$PY" scripts/make_eval_configs.py "${mec[@]}" --label "$LABEL" \
    --name "$LABEL" --model "$OUTPUT_DIR" "${CREDO_FLAG[@]}" "${DOMAIN_FLAG[@]}" --overwrite \
    || { echo "MAKE_EVAL_CONFIGS_FAILED proto=$proto label=$LABEL"; return 4; }
  if [ "$proto" = "salvage" ]; then
    bash scripts/eval_salvage.sh "$LABEL" "$EVAL_NPAR"
  else
    # for code, derive the dataset list from the configs just generated so the pipeline
    # (whose default is the seven math sets) can never reference a config that does not exist
    if [ "$ANSWER_FORMAT" = "code" ] && [ -z "${EVAL_DATASETS:-}" ]; then
      EVAL_DATASETS="$(cd "eval_configs/$LABEL" && ls -- *.json | sed 's/\.json$//' | tr '\n' ' ')"
      export EVAL_DATASETS
    fi
    EVAL_BUDGET="$EFFECTIVE_EVAL_BUDGET" bash scripts/eval_direct.sh "$LABEL" "$EVAL_NPAR"
  fi
}

echo "EVAL_STAGE_START $(date -Is) protocol=$EVAL_PROTOCOL budget=$EFFECTIVE_EVAL_BUDGET"
if [ "$EVAL_PROTOCOL" = "both" ]; then
  overall=0
  run_eval direct  || { echo "DIRECT_EVAL_FAILED label=$LABEL (continuing to salvage)"; overall=5; }
  run_eval salvage || { echo "SALVAGE_EVAL_FAILED label=$LABEL"; overall=5; }
  [ "$overall" -eq 0 ] && echo "TRAIN_THEN_EVAL_DONE $(date -Is) label=$LABEL" \
                       || echo "TRAIN_THEN_EVAL_PARTIAL rc=$overall label=$LABEL -- check markers above"
  exit "$overall"
fi
run_eval "$EVAL_PROTOCOL" || { echo "EVAL_FAILED proto=$EVAL_PROTOCOL label=$LABEL"; exit 5; }
echo "TRAIN_THEN_EVAL_DONE $(date -Is) label=$LABEL"
