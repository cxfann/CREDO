# CREDO: Confidence REaDOut

Official code for *"Beyond Verbalized Confidence: Calibrating Reasoners with Differentiable Readouts."*

CREDO trains calibrated reasoning models inside the RLVR loop. Instead of sampling a verbalized confidence, it reads a deterministic confidence from a dedicated token pair and trains it by differentiable regression. This repo implements CREDO and all baselines (GRPO, RLCR, DCPO) in a single framework, with configs, evaluation scripts, and datasets used in the paper.

## Setup

```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

Requires Python 3.10, CUDA 12.4, and 8×80 GB GPUs (ZeRO-2).

Prepare the base model (adds `<CONF_HIGH>` / `<CONF_LOW>` tokens to Qwen3-8B):

```bash
CONF_CKPT_SRC=/path/to/Qwen3-8B CONF_CKPT_DST=models/Qwen3-8B-conf \
    python scripts/make_conf_checkpoint.py
```

## Train

| Method | Math | Code |
|--------|------|------|
| CREDO  | `configs/credo_math.yaml` | `configs/credo_code.yaml` |
| GRPO   | `configs/grpo_math.yaml`  | `configs/grpo_code.yaml`  |
| RLCR   | `configs/rlcr_math.yaml`  | `configs/rlcr_code.yaml`  |
| DCPO   | `configs/dcpo_math.yaml`  | `configs/dcpo_code.yaml`  |

```bash
bash run_train.sh configs/credo_math.yaml 8 - - \
    --output_dir outputs/credo_math --run_name credo_math
```

Train + evaluate end-to-end:

```bash
bash scripts/train_then_eval.sh configs/credo_math.yaml credo_math
```

Default seed is 43; pass `--seed 44` or `--seed 45` for the other runs.

**Ablations** (CREDO, via CLI overrides):

- No analysis segment: `--conf_think False` + `EVAL_SYS_PROMPT=bac_boxed_credo_noct`
- No confidence-segment advantage: `--pg_channels ans_only`
- No differentiable regression: `--mse_alpha 0`
- No discrepancy weighting: `--sw_kappa 0`
- Coefficient sweeps: vary `--sw_kappa` / `--mse_alpha` / `--cal_gamma`

## Evaluate

```bash
# Math
python scripts/make_eval_configs.py --protocol salvage --label credo_math \
    --name credo_math --model outputs/credo_math --credo --eval-seed 43
bash scripts/eval_salvage.sh credo_math 8

# Code
python scripts/make_eval_configs.py --protocol direct --domain code --label credo_code \
    --name credo_code --model outputs/credo_code --credo --eval-seed 43
EVAL_DATASETS="deepcoder-test humanevalplus lcb-v5w lcb-v6w" \
    bash scripts/eval_direct.sh credo_code 8
```

Drop `--credo` for baselines. Pair each checkpoint with its training seed (`--eval-seed`). Metrics: accuracy, ECE, AUROC, Brier, AURC, selective accuracy.

## Reproduce paper tables

1. Train each method × {seed 43, 44, 45}.
2. Evaluate each checkpoint with its domain protocol and training seed.
3. Macro-average metrics over evaluation sets (7 math / 4 code), then average over seeds.

Cross-domain: evaluate a math checkpoint with `--protocol direct --domain code` (and vice versa), nothing else changed.

## Data

All datasets are bundled under `data/` and loaded directly by the training and evaluation scripts.
