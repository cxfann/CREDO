#!/usr/bin/env python
"""direct bare aggregation for ONE model.

Macro-averages the per-dataset results/<label>/<ds>/metrics.json produced by
evaluation.py (confidence_verifier) into results/<label>/direct_summary.json.

direct semantics (kept faithful to confidence_verifier's own behavior.
NO salvage, NO pooling. Samples truncated
at the generation budget simply lack \\boxed{} / <confidence> and therefore
enter the metric pool as (label 0, conf 0.0) with adherence 0 — which inflates
AUROC and deflates ECE/Brier relative to salvage's layered pool. The summary
records this in its "scoring" field; do not compare direct calibration numbers
against salvage ones without that caveat.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["deepscaler-test", "math-500", "aime24", "aime25", "amc23", "amc24", "aime26"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="results/<label>/<ds>/metrics.json store")
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--budget", default="", help="generation max_tokens (metadata only); "
                                                "omit to record the eval config as the source of truth")
    args = ap.parse_args()

    cells = {}
    for ds in args.datasets:
        mp = ROOT / "results" / args.label / ds / "metrics.json"
        m = list(json.load(open(mp)).values())[0]
        cells[ds] = {
            "acc": m["accuracy"],
            "ece": m["ece"],
            "auroc": m["auroc"],
            "brier_score": m["brier_score"],
            "coverage": m.get("confidence format adherence"),
            "mean_len_chars": m.get("completion length"),
        }
        # optional digit-expectation namespace; key-missing = not measured
        if isinstance(m.get("digit"), dict):
            d = m["digit"]
            cells[ds]["digit"] = {
                "ece": d.get("ece"),
                "auroc": d.get("auroc"),
                "brier_score": d.get("brier_score"),
                "coverage": d.get("coverage"),
                "conf_mean": d.get("confidence level"),
                "int_mass_mean_covered": d.get("int_mass_mean_covered"),
                "digit_mass_mean_covered": d.get("digit_mass_mean_covered"),
            }

    def macro(sel, k):
        vals = [cells[d][k] for d in sel if cells[d].get(k) is not None]
        return float(np.mean(vals)) if vals else None

    def macro_digit(sel, k):
        vals = [cells[d]["digit"][k] for d in sel if isinstance(cells[d].get("digit"), dict)
                and cells[d]["digit"].get(k) is not None]
        return float(np.mean(vals)) if vals else None

    entry = {
        "label": args.label,
        "protocol": "direct",
        "scoring": (
            # budget is recorded from the eval config, not hardcoded, so code (6144) isn't misreported
            f"bare direct, no salvage or pooling; budget={args.budget or 'see eval config max_tokens'}; "
            "truncated samples enter the pool as (label 0, conf 0.0, adherence 0)"
        ),
        "datasets": args.datasets,
        "cells": cells,
    }
    sets6 = [d for d in args.datasets if d != "aime26"]
    for tag, sel in [("O6", sets6), ("O7", args.datasets)]:
        entry[tag] = {k: macro(sel, k) for k in ["acc", "ece", "auroc", "brier_score", "coverage"]}
        if any(isinstance(cells[d].get("digit"), dict) for d in sel):
            entry[tag]["digit"] = {k: macro_digit(sel, k) for k in
                                   ["ece", "auroc", "brier_score", "coverage", "conf_mean",
                                    "int_mass_mean_covered", "digit_mass_mean_covered"]}

    out = ROOT / "results" / args.label / "direct_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(entry, indent=1) + "\n")
    o7 = entry["O7"]
    # macro() returns None when a field is absent from every cell (e.g. AUROC on a collapsed
    # channel); print such fields verbatim rather than crash in NoneType.__format__.
    f4 = lambda v: "None" if v is None else f"{v:.4f}"
    print(f"{args.label}: direct O7 acc={f4(o7['acc'])} ECE={f4(o7['ece'])} AUROC={f4(o7['auroc'])} "
          f"BS={f4(o7['brier_score'])} cov={f4(o7['coverage'])} (n_datasets={len(args.datasets)})")
    print("WROTE", out)


if __name__ == "__main__":
    main()
