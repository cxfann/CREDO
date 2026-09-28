#!/usr/bin/env python
"""Generate evaluation configs for one model over the math (7-set) or code (4-set) suite.

Global sampling parameters are fixed across all runs: temperature 0.7,
n = 4 samples per problem for the large sets and 8 for the small competition sets.
The generation budget defaults to the paper's budget for the protocol x domain pair
(salvage/math: 12288, direct/code: 6144) unless overridden. The model value may be a
literal path or an env-var reference like "$EVAL_MODEL" (evaluation.py expands it at
load time).

Pass --eval-seed to match a checkpoint's training seed (Appendix B): a run trained with
--seed 44 is evaluated with --eval-seed 44; the base model has no training seed and is
evaluated across all three. The default 42 does not reproduce the paper's per-seed pairing.

Usage:
  python scripts/make_eval_configs.py --protocol salvage --label MyRun \
      --name my-run --model outputs/credo_math --eval-seed 43
  python scripts/make_eval_configs.py --protocol direct --domain code --label MyCode \
      --name my-code --model outputs/credo_code --credo --eval-seed 43
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "deepscaler-test": {"dataset_name": "data/deepscaler-uniform", "n": 4},
    "math-500": {"dataset_name": "data/math-500", "n": 4},
    "aime24": {"dataset_name": "data/aime24", "n": 8},
    "aime25": {"dataset_name": "data/aime25", "n": 8},
    "amc23": {"dataset_name": "data/amc23", "n": 8},
    "amc24": {"dataset_name": "data/amc24", "n": 8},
    "aime26": {"dataset_name": "data/aime26", "n": 8},
}
# Coding suite: exactly the four sets that get evaluated.
#
# The two LiveCodeBench entries are the official CONTEST-DATE WINDOWS, which is what the
# literature means by "LiveCodeBench v5" and "v6": v5w = 2024.08.01-2025.02.01 (279
# problems) and v6w = 2025.02.01-2025.05.01 (131). They are disjoint, so they are two
# independent test sets rather than the nested pair the non-windowed cumulative releases
# would form; the non-windowed cumulative releases are contamination-inflated and not used here.
#
# deepcoder-test is the in-distribution held-out set (same source distribution as the
# training data, disjoint from its 9500 problems by construction), the coding analogue of
# the math suite's deepscaler-test (hence the same n=4).
CODE_DATASETS = {
    "deepcoder-test": {"dataset_name": "data/code-deepcoder", "split": "test", "n": 4},
    "humanevalplus": {"dataset_name": "data/code-evals", "split": "humanevalplus", "n": 4},
    "lcb-v5w": {"dataset_name": "data/code-evals", "split": "lcb_v5w", "n": 4},
    "lcb-v6w": {"dataset_name": "data/code-evals", "split": "lcb_v6w", "n": 4},
}
# n is uniform at 4 across the coding suite.
SUITES = {"math": DATASETS, "code": CODE_DATASETS}
PROTOCOLS = {"direct": {"max_tokens": 4096, "prefix": ""}, "salvage": {"max_tokens": 12288, "prefix": "salvage/"}}
# direct/code uses the 6144 coding budget (Appendix B); the generic direct default (4096)
# only applies to ad-hoc math direct runs. Pinned here so omitting --max-tokens can't quietly
# run the coding suite short.
PROTOCOL_DOMAIN_MAX_TOKENS = {("direct", "code"): 6144}


def protocol_max_tokens(protocol, domain):
    """Paper budget for this protocol x domain, falling back to the protocol default."""
    return PROTOCOL_DOMAIN_MAX_TOKENS.get((protocol, domain), PROTOCOLS[protocol]["max_tokens"])


def build(protocol, label, name, model, credo, dataset, max_tokens=None, digit=True,
          domain="math", eval_seed=42, sys_prompt=None):
    spec = SUITES[domain][dataset]
    proto = PROTOCOLS[protocol]
    global_args = {
        "dataset_name": spec["dataset_name"],
        "split": spec.get("split", "test"),
        "hash_key": "problem",
        "store_name": f"eval_outputs/{proto['prefix']}{label}/{dataset}",
        "gpu_memory_utilization": 0.85,
        "log_path": f"results/{proto['prefix']}{label}/{dataset}",
        "fresh": True,
    }
    local = {
        "name": name,
        "model": model,
        "check_fn": "confidence_verifier",
        "check_fn_args": {"answer_format": "code" if domain == "code" else "boxed"},
        # an explicit --sys-prompt (when given) overrides the method routing, e.g. the
        # no-analysis ablation prompts bac_boxed_credo_noct / bac_boxed_code_credo_noct
        "sys_prompt_name": sys_prompt or (("bac_boxed_code_credo" if credo else "bac_boxed_code") if domain == "code"
                                          else ("bac_boxed_credo" if credo else "bac_boxed")),
        "temperature": 0.7,
        "n": spec["n"],
        "seed": eval_seed,
        "max_tokens": max_tokens or protocol_max_tokens(protocol, domain),
        "enable_thinking": False,
        # CREDO reads the confidence from the reserved token pair (confidence_logit);
        # verbalized methods generate text, with a parallel digit-expectation readout
        "vllm_task": ["confidence_logit"] if credo else (["generate", "digit_expectation"] if digit else ["generate"]),
    }
    if credo:
        local["logprobs"] = 20
    return [global_args, local]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["direct", "salvage"], required=True)
    ap.add_argument("--label", required=True, help="eval_configs/<salvage/>LABEL/ output dir name")
    ap.add_argument("--name", required=True, help="run name (store column prefix)")
    ap.add_argument("--model", required=True, help="model path or $ENV_VAR reference")
    ap.add_argument("--credo", action="store_true",
                    help="CREDO logit-readout model (bac_boxed_credo + confidence_logit + logprobs 20)")
    ap.add_argument("--no-digit", action="store_true",
                    help="drop the digit_expectation task from non-CREDO configs")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="override the generation budget (defaults: salvage 12288, direct+code 6144, "
                         "direct+math 4096)")
    ap.add_argument("--domain", choices=["math", "code"], default="math",
                    help="evaluation suite")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--eval-seed", type=int, default=42,
                    help="vLLM sampling seed for evaluation generation (default 42). The paper "
                         "pairs this with the checkpoint's training seed: pass --eval-seed 43/44/45 "
                         "to match a run trained with that seed")
    ap.add_argument("--sys-prompt", default=None,
                    help="override the routed sys_prompt_name (e.g. the no-analysis ablation "
                         "bac_boxed_credo_noct / bac_boxed_code_credo_noct)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.sys_prompt is not None:
        sys.path.insert(0, str(ROOT))
        from system_prompts import get_sys_prompt
        get_sys_prompt(args.sys_prompt)  # fail fast on an unknown prompt name

    suite = SUITES[args.domain]
    datasets = args.datasets or list(suite)
    unknown = [d for d in datasets if d not in suite]
    if unknown:
        raise SystemExit(f"unknown {args.domain} datasets {unknown}; choose from {list(suite)}")

    out_dir = ROOT / "eval_configs" / PROTOCOLS[args.protocol]["prefix"] / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        path = out_dir / f"{dataset}.json"
        if path.exists() and not args.overwrite:
            raise SystemExit(f"refusing to overwrite {path} (pass --overwrite)")
        path.write_text(json.dumps(build(args.protocol, args.label, args.name, args.model, args.credo,
                                          dataset, args.max_tokens, digit=not args.no_digit,
                                          domain=args.domain, eval_seed=args.eval_seed,
                                          sys_prompt=args.sys_prompt), indent=2) + "\n")
        print(f"WROTE {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
