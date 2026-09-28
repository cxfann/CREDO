#!/usr/bin/env python
"""Build the training and evaluation datasets from the source files (idempotent, fixed seed).

Outputs (datasets.DatasetDict, load_from_disk compatible):
  data/deepscaler-uniform  train 9500 / test 500, fields: problem, answer, source, difficulty, orig_index
  data/aime24              test 30
  data/aime25              test 30
  data/amc23               test 46
Also verifies math-500 consistency against its source jsonl.
The bundled amc24 and aime26 datasets are not built by this script.
Run: python scripts/prepare_data.py
"""
import json
import os
import sys

import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk

import os
V3 = os.environ.get("RLCR_V3_ROOT", ".")
DCPO = os.environ.get("DCPO_DATA", "../DCPO/data")
SUFFIX = " Let's think step by step and output the final answer within \\boxed{}."
SEED = 43


def strip_suffix(content: str) -> str:
    assert content.endswith(SUFFIX), f"unexpected prompt tail: {content[-160:]!r}"
    return content[: -len(SUFFIX)].strip()


def build_deepscaler():
    df = pd.read_parquet(f"{DCPO}/deepscaler_uniform_train.parquet")
    rows = []
    for i, r in enumerate(df.itertuples()):
        problem = strip_suffix(r.prompt[0]["content"])
        answer = str(r.reward_model["ground_truth"]).strip()
        assert problem and answer, f"empty field at row {i}"
        rows.append({
            "problem": problem,
            "answer": answer,
            "source": "deepscaler",
            "difficulty": float(r.extra_info["difficulty"]),
            "orig_index": int(r.extra_info["index"]),
        })
    ds = Dataset.from_list(rows)
    split = ds.train_test_split(test_size=500, seed=SEED, shuffle=True)
    out = DatasetDict({"train": split["train"], "test": split["test"]})
    out.save_to_disk(f"{V3}/data/deepscaler-uniform")
    return out


def build_aime24():
    df = pd.read_parquet(f"{DCPO}/test_data/aime24.parquet")
    rows = []
    for i, r in enumerate(df.itertuples()):
        rows.append({
            "problem": strip_suffix(r.prompt[0]["content"]),
            "answer": str(r.reward_model["ground_truth"]).strip(),
            "source": "aime24",
            "difficulty": -1.0,
            "orig_index": i,
        })
    out = DatasetDict({"test": Dataset.from_list(rows)})
    out.save_to_disk(f"{V3}/data/aime24")
    return out


def build_aime25():
    rows = []
    with open(f"{DCPO}/test_data/aime25/aime2025.jsonl") as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            assert d["question"].strip() and str(d["answer"]).strip()
            rows.append({
                "problem": d["question"].strip(),
                "answer": str(d["answer"]).strip(),
                "source": "aime25",
                "difficulty": -1.0,
                "orig_index": i,
            })
    out = DatasetDict({"test": Dataset.from_list(rows)})
    out.save_to_disk(f"{V3}/data/aime25")
    return out


def build_amc23():
    df = pd.read_parquet(f"{DCPO}/test_data/AMC/data/amc23-00000-of-00001.parquet")
    rows = []
    for i, r in enumerate(df.itertuples()):
        assert str(r.problem).strip() and str(r.answer).strip()
        rows.append({
            "problem": str(r.problem).strip(),
            "answer": str(r.answer).strip(),
            "source": "amc23",
            "difficulty": -1.0,
            "orig_index": i,
        })
    out = DatasetDict({"test": Dataset.from_list(rows)})
    out.save_to_disk(f"{V3}/data/amc23")
    return out


def verify_math500():
    v2 = load_from_disk(f"{V3}/data/math-500")["test"]
    dcpo = [json.loads(l) for l in open(f"{DCPO}/test_data/MATH-500/test.jsonl")]
    assert len(v2) == len(dcpo) == 500
    mism_ans = sum(1 for a, b in zip(v2["answer"], (d["answer"] for d in dcpo)) if a != b)
    mism_prob = sum(1 for a, b in zip(v2["problem"], (d["problem"] for d in dcpo)) if a != b)
    return mism_prob, mism_ans


def main():
    report = {}
    ds = build_deepscaler()
    report["deepscaler"] = {"train": len(ds["train"]), "test": len(ds["test"])}
    report["aime24"] = {"test": len(build_aime24()["test"])}
    report["aime25"] = {"test": len(build_aime25()["test"])}
    report["amc23"] = {"test": len(build_amc23()["test"])}
    mp, ma = verify_math500()
    report["math500_two_source_mismatch"] = {"problem": mp, "answer": ma}
    # residual instruction substring check on all outputs
    bad = 0
    for name in ["deepscaler-uniform", "aime24", "aime25", "amc23"]:
        d = load_from_disk(f"{V3}/data/{name}")
        for split in d:
            bad += sum(1 for p in d[split]["problem"] if "output the final answer within" in p)
    report["residual_instruction_count"] = bad
    print(json.dumps(report, indent=2))
    ok = (
        report["deepscaler"] == {"train": 9500, "test": 500}
        and report["aime24"]["test"] == 30
        and report["aime25"]["test"] == 30
        and report["amc23"]["test"] == 46
        and bad == 0
    )
    print("PREPARE_DATA_ASSERTIONS:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
