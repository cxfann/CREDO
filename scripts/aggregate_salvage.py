#!/usr/bin/env python
"""Salvage-protocol aggregation for ONE model.

Merges the fresh-generation store (@12288) with the salvage-fill store (deterministic
boxed-continuation injection + strict first-forced-box scoring) by
idx = row_index * n_out + sample_index, and recomputes the main-table quadruple
acc/ECE/AUROC/BS over the pool (natural spontaneous + salvaged elicited + residual
counted as 0), plus composition / four-layer calibration / compliance audit /
residual acceptance (<1%).

Direct-protocol alignment diagnostics (migration matrix / overthinking control /
prefix consistency) are only computed when --direct-store-root points at existing
direct-protocol stores for this label; otherwise they are skipped.

Usage:
  python scripts/aggregate_salvage.py --label my-run
  python scripts/aggregate_salvage.py --label MyRun --direct-store-root eval_outputs
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from datasets import load_from_disk

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_utils import get_brier, get_ece, get_auroc

DATASETS = ["deepscaler-test", "math-500", "aime24", "aime25", "amc23", "amc24", "aime26"]
LAYERS = ["natural-complete", "salvaged-case(b)", "salvaged-case(a)", "residual"]
PREFIX_CAP = 2000
KEEP_FIELDS = ["layer", "case_type", "calibration_label", "ep1_confidence", "ep1_acc_natural", "ep1_acc_anytime",
               "ep1_conf_adherence", "post_forced_extra_text", "post_forced_long_extra", "post_forced_revision_flag",
               "forced_answer_changed", "second_injection_used", "residual_reason", "row_index", "sample_index",
               # Missing digit-expectation fields indicate "not measured".
               "ep1_confidence_digit", "ep1_conf_digit_adherence", "conf_digit_mode",
               "conf_digit_int_mass", "conf_digit_digit_mass"]


def digit_present(recs):
    return bool(recs) and all(r.get("ep1_confidence_digit") is not None for r in recs)


def calib_digit(recs):
    out = calib([r["calibration_label"] for r in recs], [r["ep1_confidence_digit"] for r in recs])
    out["coverage"] = float(np.mean([int(r["ep1_conf_digit_adherence"]) for r in recs]))
    out["int_mass_mean"] = float(np.mean([float(r["conf_digit_int_mass"]) for r in recs]))
    out["digit_mass_mean"] = float(np.mean([float(r["conf_digit_digit_mass"]) for r in recs]))
    return out


def flatten_store(path: Path):
    ds = load_from_disk(str(path))
    prefix = next(c[: -len("-conf_format_adherence")] for c in ds.column_names if c.endswith("-conf_format_adherence"))
    n_out = sum(1 for c in ds.column_names if c.startswith(f"{prefix}-output_"))
    y, conf, adh, lengths, texts = [], [], [], [], []
    for row in ds:
        for j in range(n_out):
            y.append(int(row[f"{prefix}-evals"][j]))
            conf.append(float(row[f"{prefix}-confidence_levels"][j]))
            adh.append(int(row[f"{prefix}-conf_format_adherence"][j]))
            lengths.append(int(row[f"{prefix}-c_lengths"][j]))
            texts.append(row[f"{prefix}-output_{j}"])
    return n_out, np.array(y, float), np.array(conf, float), np.array(adh, int), np.array(lengths, int), texts


def calib(labels, confs):
    labels = np.array(labels, dtype=float)
    confs = np.array(confs, dtype=float)
    return {
        "brier_score": float(get_brier(labels, confs)),
        "ece": float(get_ece(labels, confs)),
        "auroc": float(get_auroc(labels, confs)) if len(set(labels.tolist())) >= 2 else None,
        "conf_mean": float(np.mean(confs)) if len(confs) else None,
    }


def layer_blocks(recs):
    out = {}
    for layer in LAYERS:
        sub = [r for r in recs if r["layer"] == layer]
        block = {"n": len(sub)}
        if sub:
            block["acc"] = float(np.mean([r["calibration_label"] for r in sub]))
            block.update(calib([r["calibration_label"] for r in sub], [r["ep1_confidence"] for r in sub]))
            if digit_present(sub):
                block["digit"] = calib_digit(sub)
        out[layer] = block
    return out


def behavior_block(recs):
    case_a = [r for r in recs if r.get("case_type") == "case(a)"]
    residual = [r for r in recs if r["layer"] == "residual"]
    reasons = {}
    for r in residual:
        reasons[r["residual_reason"]] = reasons.get(r["residual_reason"], 0) + 1
    return {
        "case_a_n": len(case_a),
        "case_b_n": sum(1 for r in recs if r.get("case_type") == "case(b)"),
        "residual_n": len(residual),
        "residual_reasons": reasons,
        "post_forced_extra_rate": float(np.mean([int(bool(r.get("post_forced_extra_text"))) for r in case_a])) if case_a else None,
        "post_forced_long_extra_rate": float(np.mean([int(r.get("post_forced_long_extra", 0)) for r in case_a])) if case_a else None,
        "post_forced_revision_rate": float(np.mean([int(r.get("post_forced_revision_flag", 0)) for r in case_a])) if case_a else None,
        "forced_answer_changed_rate": float(np.mean([int(r.get("forced_answer_changed", 0)) for r in case_a])) if case_a else None,
        "second_injection_rate": float(np.mean([int(r.get("second_injection_used", 0)) for r in recs])) if recs else None,
    }


def analyze_cell(label: str, dataset: str, direct_store_root: Path | None):
    salvage_store = ROOT / "eval_outputs/salvage" / label / dataset
    salv_store = ROOT / "eval_outputs/salvage-fill" / label / dataset
    salvage_metrics = ROOT / "results/salvage" / label / dataset / "metrics.json"

    n_out2, y2, conf2, adh2, len2, texts2 = flatten_store(salvage_store)
    sds = load_from_disk(str(salv_store))
    recs = [{k: r.get(k) for k in KEEP_FIELDS} for r in sds if r["mode"] == "ep1"]
    assert len(recs) == len(y2), (str(salv_store), len(recs), len(y2))

    # assertion 1: per-sample salvage acc_natural == salvage spontaneous evals (positional merge)
    for r in recs:
        idx = int(r["row_index"]) * n_out2 + int(r["sample_index"])
        assert int(r["ep1_acc_natural"]) == int(y2[idx]), (str(salv_store), idx)
        if r["layer"] == "natural-complete":
            assert int(adh2[idx]) == 1, (str(salv_store), idx)
    # assertion 2: cell acc_natural == results/salvage metrics.json accuracy
    salvagem = json.load(open(salvage_metrics))
    salvage_acc = list(salvagem.values())[0]["accuracy"]
    acc_natural = float(np.mean([r["ep1_acc_natural"] for r in recs]))
    assert abs(acc_natural - salvage_acc) < 1e-6, (str(salvage_metrics), acc_natural, salvage_acc)

    cell = {
        "n": len(recs),
        "acc_natural": acc_natural,
        "acc_anytime": float(np.mean([r["ep1_acc_anytime"] for r in recs])),
        "coverage": float(np.mean([r["ep1_conf_adherence"] for r in recs])),
        "trunc_rate_12288": float(np.mean(adh2 == 0)),
        "pooled": calib([r["calibration_label"] for r in recs], [r["ep1_confidence"] for r in recs]),
        "layers": layer_blocks(recs),
        "behavior": behavior_block(recs),
        "mean_len_chars_salvage": float(len2.mean()),
    }
    if digit_present(recs):
        cell["pooled_digit"] = calib_digit(recs)

    direct_store = (direct_store_root / label / dataset) if direct_store_root is not None else None
    if direct_store is not None and direct_store.exists():
        n_out0, y0, conf0, adh0, len0, texts0 = flatten_store(direct_store)
        assert n_out0 == n_out2 and len(y0) == len(y2), (str(direct_store), n_out0, n_out2)
        direct_nat = adh0 == 1
        salvage_nat = adh2 == 1
        mig = {}
        for src_name, src in [("direct_natural", direct_nat), ("direct_nonadherent", ~direct_nat)]:
            mig[src_name] = {
                "n": int(src.sum()),
                "salvage_natural_correct": int((src & salvage_nat & (y2 == 1)).sum()),
                "salvage_natural_wrong": int((src & salvage_nat & (y2 == 0)).sum()),
                "salvage_still_nonadherent": int((src & ~salvage_nat).sum()),
            }
        cell["migration"] = mig
        cell["overthinking"] = {
            "direct_natural_n": int(direct_nat.sum()),
            "acc_direct_on_direct_natural": float(y0[direct_nat].mean()) if direct_nat.sum() else None,
            "acc_salvage_on_direct_natural": float(y2[direct_nat].mean()) if direct_nat.sum() else None,
        }
        matches = sum(1 for a, b in zip(texts0, texts2) if a[:PREFIX_CAP] == b[:PREFIX_CAP])
        cell["prefix_consistency_2000c"] = matches / len(texts0)
        cell["cost_char_ratio_salvage_over_direct"] = float(len2.mean() / len0.mean()) if len0.mean() else None
    return cell, recs


def macro(cells, k):
    vals = [c[k] if not isinstance(k, tuple) else c[k[0]][k[1]] for c in cells]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="eval_outputs/salvage/<label> store dir name")
    ap.add_argument("--datasets", nargs="*", default=DATASETS, choices=DATASETS)
    ap.add_argument("--direct-store-root", default=None,
                    help="root of direct stores for this label (e.g. eval_outputs); omit inside job containers to skip direct-alignment diagnostics")
    args = ap.parse_args()
    direct_root = (ROOT / args.direct_store_root) if args.direct_store_root else None

    run_cells = {}
    run_recs = []
    for dataset in args.datasets:
        cell, recs = analyze_cell(args.label, dataset, direct_root)
        run_cells[dataset] = cell
        run_recs.extend(recs)

    entry = {
        "label": args.label,
        "protocol": "salvage",
        "scoring": "fresh regeneration @12288 (temp 0.7 / seed 42) + deterministic boxed-continuation injection + strict first-forced-box scoring for residual salvage; pool = natural spontaneous + salvaged elicited + residual scored 0",
        "direct_alignment_diagnostics": direct_root is not None,
        "cells": run_cells,
    }
    sets6 = [d for d in args.datasets if d != "aime26"]
    for tag, sel in [("O6", sets6), ("O7", args.datasets)]:
        cs = [run_cells[d] for d in sel]
        entry[tag] = {
            "acc_natural": macro(cs, "acc_natural"),
            "acc_anytime": macro(cs, "acc_anytime"),
            "coverage": macro(cs, "coverage"),
            "trunc_rate_12288": macro(cs, "trunc_rate_12288"),
            "ece": macro(cs, ("pooled", "ece")),
            "auroc": macro(cs, ("pooled", "auroc")),
            "brier_score": macro(cs, ("pooled", "brier_score")),
        }
        if all("pooled_digit" in c for c in cs):
            entry[tag]["digit"] = {
                "ece": macro(cs, ("pooled_digit", "ece")),
                "auroc": macro(cs, ("pooled_digit", "auroc")),
                "brier_score": macro(cs, ("pooled_digit", "brier_score")),
                "coverage": macro(cs, ("pooled_digit", "coverage")),
                "conf_mean": macro(cs, ("pooled_digit", "conf_mean")),
                "int_mass_mean": macro(cs, ("pooled_digit", "int_mass_mean")),
                "digit_mass_mean": macro(cs, ("pooled_digit", "digit_mass_mean")),
            }
    entry["run_pooled_layers"] = layer_blocks(run_recs)
    entry["run_behavior"] = behavior_block(run_recs)
    total = len(run_recs)
    residual = entry["run_behavior"]["residual_n"]
    comp = {
        "total": total,
        "natural_rate": sum(1 for r in run_recs if r["layer"] == "natural-complete") / total,
        "case_b_rate": sum(1 for r in run_recs if r["layer"] == "salvaged-case(b)") / total,
        "case_a_rate": sum(1 for r in run_recs if r["layer"] == "salvaged-case(a)") / total,
        "residual_rate": residual / total,
    }
    assert abs(sum(comp[k] for k in ("natural_rate", "case_b_rate", "case_a_rate", "residual_rate")) - 1.0) < 1e-9
    entry["composition"] = comp
    entry["acceptance"] = {"total": total, "residual": residual, "residual_rate": residual / total, "pass_lt_1pct": residual / total < 0.01}

    out = ROOT / "results/salvage" / args.label / "salvage_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(entry, indent=2, ensure_ascii=False))
    o7 = entry["O7"]
    a = entry["acceptance"]
    # A collapsed confidence channel leaves some O7 fields None (e.g. AUROC at zero variance);
    # print them verbatim rather than crash in NoneType.__format__ after the summary is written.
    f4 = lambda v: "None" if v is None else f"{v:.4f}"
    print(f"{args.label}: O7 acc_nat={f4(o7['acc_natural'])} acc_any={f4(o7['acc_anytime'])} ECE={f4(o7['ece'])} "
          f"AUROC={f4(o7['auroc'])} BS={f4(o7['brier_score'])} cov={f4(o7['coverage'])} trunc@12288={f4(o7['trunc_rate_12288'])} "
          f"| residual {a['residual']}/{a['total']} ({a['residual_rate']:.4%}) pass={a['pass_lt_1pct']}")
    print("WROTE", out)
    print("SALVAGE_CROSSCHECK_ASSERTIONS: PASS (per-sample acc_natural==salvage evals; cell acc_natural==results/salvage metrics; composition sum==1)")


if __name__ == "__main__":
    main()
