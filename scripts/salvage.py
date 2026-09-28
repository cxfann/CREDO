#!/usr/bin/env python
"""Salvage tool: regenerate and fill in truncated completions for the salvage protocol."""

import argparse
import json
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_processing import process_dataset
from eval.eval_args import GlobalArgs, LocalConfig
from eval.eval_utils import get_brier, get_ece, get_auroc, hash_dataset
from reward_fns import extract_boxed_answer, verify_gold_pred
from segment_utils import find_segments
from system_prompts import get_sys_prompt

COMMIT_BOXED_PREFIX = "\n\nThe reasoning budget is exhausted. I must give the answer now and must not continue solving or revise it. After closing the boxed answer, I will only analyze uncertainty and report confidence.\n\nThe final answer is \\boxed{"
ANALYSIS_TAG_PREFIX = "\n<analysis>"
SECOND_INJECTION = "</analysis>\n<confidence>"
LAYERS = ["natural-complete", "salvaged-case(b)", "salvaged-case(a)", "residual"]


@dataclass
class PromptBundle:
    source_dataset: Dataset
    hashed_dataset: Dataset
    prompt_texts: list[str]
    manual_prompt_texts: list[str]


@dataclass
class Job:
    record_index: int
    prompt_text: str
    base_output: str
    first_injection: str
    case_type: str
    mode: str


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel_project_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def load_eval_config(path: Path) -> tuple[GlobalArgs, LocalConfig]:
    with path.open() as f:
        raw = json.load(f)
    if len(raw) != 2:
        raise ValueError(f"Expected exactly one GlobalArgs and one LocalConfig in {path}, got {len(raw)} entries")
    # match evaluation.py: allow env-var injection (e.g. "model": "$EVAL_MODEL");
    # identity when no $VAR present
    raw = [
        {k: (os.path.expandvars(v) if isinstance(v, str) else v) for k, v in c.items()}
        for c in raw
    ]
    return GlobalArgs(**raw[0]), LocalConfig(**raw[1])


def load_source_dataset(global_args: GlobalArgs) -> Dataset:
    ds_path = project_path(global_args.dataset_name)
    if ds_path.exists():
        dataset = load_from_disk(str(ds_path))
    else:
        dataset = load_dataset(global_args.dataset_name)
    dataset = dataset[global_args.split]
    dataset = dataset.map(lambda x: hash_dataset(x, global_args.hash_key))
    if global_args.sample_size is not None:
        dataset = dataset.select(range(global_args.sample_size))
    return dataset


def render_chat_prompts(tokenizer: Any, messages: list[list[dict[str, str]]], enable_thinking: Optional[bool]) -> list[str]:
    if enable_thinking is None:
        prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    else:
        prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=enable_thinking)
    if prompt_ids and isinstance(prompt_ids[0], int):
        prompt_ids = [prompt_ids]
    return [tokenizer.decode(ids) for ids in prompt_ids]


def manual_messages(dataset: Dataset, sys_prompt_name: str) -> list[list[dict[str, str]]]:
    sys_prompt = get_sys_prompt(sys_prompt_name)
    messages = []
    for row in dataset:
        problem = row["question"] if "question" in row else row["problem"]
        messages.append([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"\n\nPROBLEM: {problem}\n\n"},
        ])
    return messages


def build_prompts(global_args: GlobalArgs, config: LocalConfig, tokenizer: Any) -> PromptBundle:
    source_dataset = load_source_dataset(global_args)
    local_dataset = process_dataset(source_dataset, config)
    process_messages = [local_dataset[i][config.tokenize_key] for i in range(len(local_dataset))]
    process_prompts = render_chat_prompts(tokenizer, process_messages, config.enable_thinking)
    manual_prompts = render_chat_prompts(tokenizer, manual_messages(source_dataset, config.sys_prompt_name), config.enable_thinking)
    return PromptBundle(source_dataset=source_dataset, hashed_dataset=source_dataset, prompt_texts=process_prompts, manual_prompt_texts=manual_prompts)


def assert_prompt_rebuild(bundle: PromptBundle, store: Dataset, sample_count: int) -> None:
    if len(bundle.prompt_texts) != len(store):
        raise AssertionError(f"Prompt dataset/store length mismatch: {len(bundle.prompt_texts)} vs {len(store)}")
    if len(bundle.prompt_texts) == 0:
        raise AssertionError("Empty dataset cannot satisfy prompt rebuild assertion")
    indices = list(range(min(sample_count, len(store))))
    if len(store) > sample_count:
        indices = sorted(set(indices + [len(store) // 2, len(store) - 1]))[:sample_count]
    for i in indices:
        if bundle.prompt_texts[i].encode("utf-8") != bundle.manual_prompt_texts[i].encode("utf-8"):
            raise AssertionError(f"Prompt rebuild mismatch at row {i}")
        if "id" in store.column_names and int(store[i]["id"]) != int(bundle.hashed_dataset[i]["id"]):
            raise AssertionError(f"Store/source id mismatch at row {i}: {store[i]['id']} vs {bundle.hashed_dataset[i]['id']}")


def injection_for_output(output: str, variant: str) -> tuple[str, str]:
    if "<analysis>" in output:
        return "", "case(b)"
    if variant == "commit_boxed":
        return COMMIT_BOXED_PREFIX, "case(a)"
    if variant == "analysis_tag":
        return ANALYSIS_TAG_PREFIX, "case(a)"
    raise ValueError(f"Unknown variant {variant}")


def prefix_before_analysis(text: str) -> str:
    idx = text.find("<analysis>")
    return text[:idx] if idx >= 0 else text


def boxed_answer(text: str) -> Optional[str]:
    return extract_boxed_answer(prefix_before_analysis(text))


def boxed_correct(text: str, gold: str) -> int:
    answer = boxed_answer(text)
    if answer is None:
        return 0
    return int(verify_gold_pred(gold, f"\\boxed{{{answer}}}"))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def base_record(row: dict[str, Any], row_index: int, sample_index: int, config_name: str, prompt_text: str, mode: str) -> dict[str, Any]:
    output = row[f"{config_name}-output_{sample_index}"]
    eval_value = int(row[f"{config_name}-evals"][sample_index])
    conf_value = safe_float(row[f"{config_name}-confidence_levels"][sample_index])
    adh_value = int(row[f"{config_name}-conf_format_adherence"][sample_index])
    # Optional digit-expectation channel from the initial generation. Missing
    # columns produce None sentinels, indicating "not measured" downstream.
    digit_key = f"{config_name}-confidence_levels_digit"
    if digit_key in row:
        d_conf = safe_float(row[digit_key][sample_index])
        d_adh = int(row[f"{config_name}-conf_digit_adherence"][sample_index])
        d_mode = row[f"{config_name}-conf_digit_mode"][sample_index]
        d_imass = safe_float(row[f"{config_name}-conf_digit_int_mass"][sample_index])
        d_dmass = safe_float(row[f"{config_name}-conf_digit_digit_mass"][sample_index])
    else:
        d_conf = d_adh = d_mode = d_imass = d_dmass = None
    return {
        "protocol": "salvage-fill",
        "mode": mode,
        "run_name": config_name,
        "row_index": row_index,
        "sample_index": sample_index,
        "id": int(row["id"]) if "id" in row else None,
        "orig_index": int(row["orig_index"]) if "orig_index" in row else row_index,
        "problem": row["problem"] if "problem" in row else row.get("question", ""),
        "answer": str(row["answer"]),
        "prompt_text": prompt_text,
        "ep0_output": output,
        "ep0_answer_boxed": boxed_answer(output),
        "ep0_eval": eval_value,
        "ep0_confidence": conf_value,
        "ep0_conf_adherence": adh_value,
        "ep0_confidence_digit": d_conf,
        "ep0_conf_digit_adherence": d_adh,
        "ep0_length": len(output),
        "case_type": "natural" if adh_value else None,
        "layer": "natural-complete" if adh_value else None,
        "variant": "none" if adh_value else None,
        "first_injection": "",
        "second_injection_used": 0,
        "continuation_1": "",
        "continuation_2": "",
        "ep1_output": output,
        "ep1_answer_boxed": boxed_answer(output),
        "forced_answer_boxed": None,
        "final_pre_analysis_boxed": boxed_answer(output),
        "post_forced_extra_text": "",
        "post_forced_long_extra": 0,
        "post_forced_revision_flag": 0,
        "forced_answer_changed": 0,
        "ep1_acc_natural": eval_value,
        "ep1_acc_anytime": eval_value,
        "ep1_confidence": conf_value,
        "ep1_conf_adherence": adh_value,
        "ep1_confidence_digit": d_conf,
        "ep1_conf_digit_adherence": d_adh,
        "conf_digit_mode": d_mode,
        "conf_digit_int_mass": d_imass,
        "conf_digit_digit_mass": d_dmass,
        "calibration_label": eval_value,
        "residual_reason": "" if adh_value else None,
        "spontaneous_output": output if mode == "self_validation" else "",
        "spontaneous_answer_boxed": boxed_answer(output) if mode == "self_validation" else None,
        "spontaneous_confidence": conf_value if mode == "self_validation" else None,
        "truncated_output": "",
    }


def make_main_records(store: Dataset, bundle: PromptBundle, config: LocalConfig, variant: str) -> tuple[list[dict[str, Any]], list[Job]]:
    records: list[dict[str, Any]] = []
    jobs: list[Job] = []
    for row_index, row in enumerate(store):
        row_dict = dict(row)
        for sample_index in range(config.n):
            rec = base_record(row_dict, row_index, sample_index, config.name, bundle.prompt_texts[row_index], "ep1")
            if rec["ep0_conf_adherence"] == 0:
                injection, case_type = injection_for_output(rec["ep0_output"], variant)
                rec["case_type"] = case_type
                rec["layer"] = f"salvaged-{case_type}"
                rec["variant"] = variant
                rec["first_injection"] = injection
                rec["ep1_output"] = ""
                rec["ep1_answer_boxed"] = None
                rec["ep1_acc_anytime"] = 0
                rec["ep1_confidence"] = 0.0
                rec["ep1_conf_adherence"] = 0
                if rec["ep0_confidence_digit"] is not None:
                    rec["ep1_confidence_digit"] = 0.0
                    rec["ep1_conf_digit_adherence"] = 0
                    rec["conf_digit_mode"] = "pending"
                    rec["conf_digit_int_mass"] = 0.0
                    rec["conf_digit_digit_mass"] = 0.0
                rec["calibration_label"] = 0
                rec["residual_reason"] = "not_generated"
                jobs.append(Job(len(records), rec["prompt_text"], rec["ep0_output"], injection, case_type, "ep1"))
            records.append(rec)
    return records, jobs


def choose_truncation(output: str, rng: random.Random, mode: str) -> tuple[str, str]:
    analysis = output.find("<analysis>")
    confidence = output.find("<confidence>")
    usable_len = confidence if confidence > 0 else len(output)
    if usable_len <= 8:
        return output[: max(1, usable_len // 2)], "case(a)"
    desired = mode
    if mode == "mixed":
        desired = "case_b" if analysis > 0 and rng.random() < 0.5 else "case_a"
    if desired == "case_b" and analysis >= 0 and analysis + len("<analysis>") < usable_len:
        start = analysis + len("<analysis>")
        end = max(start + 1, usable_len - 1)
        cut = rng.randint(start + 1, end)
        return output[:cut], "case(b)"
    end = analysis if analysis > 8 else usable_len
    if end <= 8:
        end = usable_len
    cut = rng.randint(max(1, end // 3), max(2, end - 1))
    return output[:cut], "case(a)"


def make_self_validation_records(store: Dataset, bundle: PromptBundle, config: LocalConfig, variant: str, count: int, seed: int, truncation_mode: str) -> tuple[list[dict[str, Any]], list[Job]]:
    if count <= 0:
        return [], []
    candidates = []
    for row_index, row in enumerate(store):
        row_dict = dict(row)
        for sample_index in range(config.n):
            if int(row_dict[f"{config.name}-conf_format_adherence"][sample_index]) == 1:
                candidates.append((row_index, sample_index, row_dict))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[: min(count, len(candidates))]
    records: list[dict[str, Any]] = []
    jobs: list[Job] = []
    for row_index, sample_index, row_dict in selected:
        rec = base_record(row_dict, row_index, sample_index, config.name, bundle.prompt_texts[row_index], "self_validation")
        truncated, trunc_case = choose_truncation(rec["ep0_output"], rng, truncation_mode)
        injection, case_type = injection_for_output(truncated, variant)
        if trunc_case != case_type:
            case_type = trunc_case
            injection = "" if case_type == "case(b)" else COMMIT_BOXED_PREFIX
        rec["truncated_output"] = truncated
        rec["case_type"] = case_type
        rec["layer"] = f"self-validation-{case_type}"
        rec["variant"] = variant
        rec["first_injection"] = injection
        rec["ep1_output"] = ""
        rec["ep1_answer_boxed"] = None
        rec["ep1_acc_anytime"] = 0
        rec["ep1_confidence"] = 0.0
        rec["ep1_conf_adherence"] = 0
        rec["calibration_label"] = 0
        rec["residual_reason"] = "not_generated"
        jobs.append(Job(len(records), rec["prompt_text"], truncated, injection, case_type, "self_validation"))
        records.append(rec)
    return records, jobs


def logit_confidence_from_suffix(picked: Any, tokenizer: Any, temperature: float) -> tuple[int, float, str]:
    hid = tokenizer.convert_tokens_to_ids("<CONF_HIGH>")
    lid = tokenizer.convert_tokens_to_ids("<CONF_LOW>")
    _, _, readout = find_segments(tokenizer, list(picked.token_ids), hid, lid)
    if readout is None or picked.logprobs is None or readout >= len(picked.logprobs):
        return 0, 0.0, "no_readout"
    lp_map = picked.logprobs[readout]
    high_lp = lp_map.get(hid)
    low_lp = lp_map.get(lid)
    if high_lp is None and low_lp is None:
        return 0, 0.0, "no_pair_logprob"
    if high_lp is None or low_lp is None:
        lp_min = min(v.logprob for v in lp_map.values())
        high_value = high_lp.logprob if high_lp is not None else lp_min
        low_value = low_lp.logprob if low_lp is not None else lp_min
        reason = "one_sided_logprob"
    else:
        high_value = high_lp.logprob
        low_value = low_lp.logprob
        reason = "ok"
    conf = 1.0 / (1.0 + math.exp(-temperature * (high_value - low_value)))
    return 1, conf, reason


def local_confidence_extractor(response: str) -> tuple[int, float]:
    import re
    conf_matches = re.findall(r"<confidence>(.*?)</confidence>", response, re.DOTALL | re.MULTILINE)
    last_confidence = conf_matches[-1] if conf_matches else ""
    if last_confidence == "":
        return 0, 0.0
    try:
        confidence = float(last_confidence)
    except Exception:
        first_number = re.search(r"-?\d+(?:\.\d+)?", last_confidence)
        if not first_number:
            return 0, 0.0
        confidence = float(first_number.group())
    if 0 <= confidence <= 1:
        return 1, confidence
    if 1 < confidence <= 100:
        return 1, confidence / 100
    return 0, 0.0


def confidence_from_text(full_text: str, picked: Any, tokenizer: Any, config: LocalConfig, temperature: float) -> tuple[str, int, float, str]:
    if "confidence_logit" in config.vllm_task:
        adherence, confidence, reason = logit_confidence_from_suffix(picked, tokenizer, temperature)
        if adherence:
            return full_text + f" <confidence> {confidence} </confidence>", 1, confidence, reason
        return full_text, 0, 0.0, reason
    adherence, confidence = local_confidence_extractor(full_text)
    return full_text, int(adherence), float(confidence), "text_extractor" if adherence else "no_confidence"


def forced_box_audit(record: dict[str, Any], full_text: str) -> tuple[Optional[str], str, int, int]:
    if record.get("case_type") != "case(a)" or not record.get("first_injection"):
        return None, "", 0, 0
    injection = record["first_injection"]
    start = full_text.find(injection)
    if start < 0:
        return None, "", 0, 0
    after = full_text[start + len(injection):]
    depth = 1
    close = None
    for i, ch in enumerate(after):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                close = i
                break
    if close is None:
        return None, "", 0, 0
    forced = after[:close]
    rest = after[close + 1:]
    analysis_idx = rest.find("<analysis>")
    pre_analysis = rest[:analysis_idx] if analysis_idx >= 0 else rest
    extra = " ".join(pre_analysis.split())
    revision = int(any(marker in extra for marker in ["\\boxed", "Wait", "wait", "Let me", "Actually", "actually", "recheck", "check"]))
    return forced, extra, int(len(extra) > 40), revision


def finalize_record(record: dict[str, Any], full_text: str, adherence: int, confidence: float, reason: str) -> None:
    record["ep1_output"] = full_text
    final_box = boxed_answer(full_text)
    forced_box, extra_text, long_extra, revision_flag = forced_box_audit(record, full_text)
    label_box = forced_box if forced_box is not None else final_box
    record["ep1_answer_boxed"] = final_box
    record["forced_answer_boxed"] = forced_box
    record["final_pre_analysis_boxed"] = final_box
    record["post_forced_extra_text"] = extra_text
    record["post_forced_long_extra"] = long_extra
    record["post_forced_revision_flag"] = revision_flag
    record["forced_answer_changed"] = int(forced_box is not None and (forced_box or "").strip() != (final_box or "").strip())
    record["ep1_acc_anytime"] = int(verify_gold_pred(record["answer"], f"\\boxed{{{label_box}}}")) if label_box is not None else 0
    record["ep1_confidence"] = float(confidence)
    record["ep1_conf_adherence"] = int(adherence)
    record["calibration_label"] = int(record["ep1_acc_anytime"])
    if adherence:
        record["residual_reason"] = ""
    else:
        record["layer"] = "residual"
        record["residual_reason"] = reason


def run_jobs(records: list[dict[str, Any]], jobs: list[Job], config: LocalConfig, tokenizer: Any, args: argparse.Namespace) -> None:
    if not jobs:
        return
    from vllm import LLM, SamplingParams

    model_path = str(project_path(config.model)) if not Path(config.model).is_absolute() else config.model
    # digit pass reads top-50 next-token logprobs; engine default max_logprobs=20
    _llm_kwargs = {"max_logprobs": max(50, args.logprobs)} if "digit_expectation" in config.vllm_task else {}
    llm = LLM(model=model_path, gpu_memory_utilization=args.gpu_memory_utilization, **_llm_kwargs)
    params = SamplingParams(n=1, temperature=args.temperature, max_tokens=args.max_tokens, seed=args.seed, logprobs=args.logprobs)
    first_prompts = [job.prompt_text + job.base_output + job.first_injection for job in jobs]
    first_outputs = llm.generate(first_prompts, sampling_params=params)
    second_jobs: list[tuple[Job, str, int, float, str]] = []
    for job, output in zip(jobs, first_outputs):
        picked = output.outputs[0]
        full_text = job.base_output + job.first_injection + picked.text
        full_text, adherence, confidence, reason = confidence_from_text(full_text, picked, tokenizer, config, args.temperature)
        record = records[job.record_index]
        record["continuation_1"] = picked.text
        if adherence or args.max_rounds < 2 or "<analysis>" not in full_text:
            finalize_record(record, full_text, adherence, confidence, reason if adherence == 0 else "")
        else:
            second_jobs.append((job, full_text, adherence, confidence, reason))
    if not second_jobs:
        _digit_pass(records, jobs, config, tokenizer, llm)
        return
    second_prompts = [job.prompt_text + full_text + SECOND_INJECTION for job, full_text, _, _, _ in second_jobs]
    second_outputs = llm.generate(second_prompts, sampling_params=params)
    for (job, full_text, _, _, first_reason), output in zip(second_jobs, second_outputs):
        picked = output.outputs[0]
        combined = full_text + SECOND_INJECTION + picked.text
        combined, adherence, confidence, reason = confidence_from_text(combined, picked, tokenizer, config, args.temperature)
        record = records[job.record_index]
        record["second_injection_used"] = 1
        record["continuation_2"] = picked.text
        final_reason = reason if reason else first_reason
        finalize_record(record, combined, adherence, confidence, final_reason)
    _digit_pass(records, jobs, config, tokenizer, llm)


def _digit_pass(records: list[dict[str, Any]], jobs: list[Job], config: LocalConfig, tokenizer: Any, llm: Any) -> None:
    """Digit-expectation readout on finalized salvage records.

    Runs only when the eval config carries the digit_expectation task (non-CREDO channel).
    Anchor lives in the FINAL ep1_output (the salvage continuation usually contains the
    elicited <confidence>0.x digits, so the digit-first anchor applies; records whose
    text channel stayed silent fall back to the canonical opener when a boxed answer
    exists, mirroring the stage-1 eligibility rule).
    """
    if "digit_expectation" not in config.vllm_task:
        return
    from eval.digit_expectation import run_digit_expectation

    todo = []
    for job in jobs:
        record = records[job.record_index]
        eligible = int(record["ep1_conf_adherence"]) == 1 or boxed_answer(record["ep1_output"]) is not None
        if eligible:
            todo.append((job.record_index, record["prompt_text"], record["ep1_output"]))
        else:
            record["conf_digit_mode"] = "ineligible"
            record["conf_digit_int_mass"] = 0.0
            record["conf_digit_digit_mass"] = 0.0
    if not todo:
        return
    results = run_digit_expectation(llm, tokenizer, [(p, t) for _, p, t in todo], logprobs_k=50)
    for (record_index, _, _), res in zip(todo, results):
        record = records[record_index]
        record["ep1_confidence_digit"] = res["conf"] if res["conf"] is not None else 0.0
        record["ep1_conf_digit_adherence"] = res["adherence"]
        record["conf_digit_mode"] = res["mode"]
        record["conf_digit_int_mass"] = res["int_mass"]
        record["conf_digit_digit_mass"] = res["digit_mass"]


def scalar_mean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(np.array(values, dtype=float)))


def metric_block(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"n": 0, "coverage": None, "acc": None, "brier_score": None, "ece": None, "auroc": None, "confidence_mean": None}
    labels = np.array([int(r["calibration_label"]) for r in records], dtype=float)
    confs = np.array([float(r["ep1_confidence"]) for r in records], dtype=float)
    block = {
        "n": len(records),
        "coverage": scalar_mean([int(r["ep1_conf_adherence"]) for r in records]),
        "acc": scalar_mean([int(r["calibration_label"]) for r in records]),
        "brier_score": float(get_brier(labels, confs)),
        "ece": float(get_ece(labels, confs)),
        "auroc": None,
        "confidence_mean": float(np.mean(confs)),
    }
    if len(set(labels.tolist())) >= 2:
        block["auroc"] = float(get_auroc(labels, confs))
    digit_vals = [r.get("ep1_confidence_digit") for r in records]
    if all(v is not None for v in digit_vals):
        dconfs = np.array([float(v) for v in digit_vals], dtype=float)
        dblock = {
            "coverage": scalar_mean([int(r["ep1_conf_digit_adherence"]) for r in records]),
            "brier_score": float(get_brier(labels, dconfs)),
            "ece": float(get_ece(labels, dconfs)),
            "auroc": None,
            "confidence_mean": float(np.mean(dconfs)),
        }
        if len(set(labels.tolist())) >= 2:
            dblock["auroc"] = float(get_auroc(labels, dconfs))
        block["digit"] = dblock
    return block


def corr_or_none(a: list[float], b: list[float]) -> Optional[float]:
    if len(a) < 2 or len(b) < 2:
        return None
    if len(set(a)) < 2 or len(set(b)) < 2:
        return None
    return float(np.corrcoef(np.array(a, dtype=float), np.array(b, dtype=float))[0, 1])


def behavior_block(records: list[dict[str, Any]]) -> dict[str, Any]:
    case_a = [r for r in records if r.get("case_type") == "case(a)"]
    if not case_a:
        return {"case_a_n": 0, "post_forced_extra_rate": None, "post_forced_long_extra_rate": None, "post_forced_revision_rate": None, "forced_answer_changed_rate": None}
    return {
        "case_a_n": len(case_a),
        "post_forced_extra_rate": scalar_mean([int(bool(r.get("post_forced_extra_text"))) for r in case_a]),
        "post_forced_long_extra_rate": scalar_mean([int(r.get("post_forced_long_extra", 0)) for r in case_a]),
        "post_forced_revision_rate": scalar_mean([int(r.get("post_forced_revision_flag", 0)) for r in case_a]),
        "forced_answer_changed_rate": scalar_mean([int(r.get("forced_answer_changed", 0)) for r in case_a]),
    }


def self_answer_equiv(record: dict[str, Any]) -> int:
    salvaged = record["ep1_answer_boxed"]
    spontaneous = record["spontaneous_answer_boxed"]
    if salvaged is None or spontaneous is None:
        return 0
    return int(verify_gold_pred(spontaneous, f"\\boxed{{{salvaged}}}"))


def compute_metrics(records: list[dict[str, Any]], run_name: str, variant: str) -> dict[str, Any]:
    ep1_records = [r for r in records if r["mode"] == "ep1"]
    self_records = [r for r in records if r["mode"] == "self_validation"]
    metrics = {
        "run_name": run_name,
        "protocol": "salvage-fill",
        "variant": variant,
        "counts": {
            "total": len(ep1_records),
            "natural_complete": sum(1 for r in ep1_records if r["layer"] == "natural-complete"),
            "salvaged_case_b": sum(1 for r in ep1_records if r["layer"] == "salvaged-case(b)"),
            "salvaged_case_a": sum(1 for r in ep1_records if r["layer"] == "salvaged-case(a)"),
            "residual": sum(1 for r in ep1_records if r["layer"] == "residual"),
        },
        "acc_natural": scalar_mean([int(r["ep1_acc_natural"]) for r in ep1_records]),
        "acc_anytime": scalar_mean([int(r["ep1_acc_anytime"]) for r in ep1_records]),
        "confidence_coverage": scalar_mean([int(r["ep1_conf_adherence"]) for r in ep1_records]),
        "scoring_rule": "case(a) acc_anytime/calibration_label use the first boxed answer closed after the injection; natural and case(b) use final pre-analysis boxed answer",
        "layers": {},
        "behavior_audit": behavior_block(ep1_records),
        "note": "Calibration metrics are reported by layer; elicited confidence is not pooled with spontaneous confidence.",
    }
    for layer in LAYERS:
        metrics["layers"][layer] = metric_block([r for r in ep1_records if r["layer"] == layer])
    if self_records:
        salvaged_conf = [float(r["ep1_confidence"]) for r in self_records if int(r["ep1_conf_adherence"]) == 1 and r["spontaneous_confidence"] is not None]
        spontaneous_conf = [float(r["spontaneous_confidence"]) for r in self_records if int(r["ep1_conf_adherence"]) == 1 and r["spontaneous_confidence"] is not None]
        metrics["self_validation"] = {
            "n": len(self_records),
            "coverage": scalar_mean([int(r["ep1_conf_adherence"]) for r in self_records]),
            "conf_correlation": corr_or_none(salvaged_conf, spontaneous_conf),
            "mean_conf_bias_salvaged_minus_spontaneous": None if not salvaged_conf else float(np.mean(np.array(salvaged_conf) - np.array(spontaneous_conf))),
            "answer_exact_match_rate": scalar_mean([int((r["ep1_answer_boxed"] or "") == (r["spontaneous_answer_boxed"] or "")) for r in self_records]),
            "answer_equiv_match_rate": scalar_mean([self_answer_equiv(r) for r in self_records]),
        }
    return metrics


def write_outputs(records: list[dict[str, Any]], metrics: dict[str, Any], output_store: Path, results_path: Path, overwrite: bool) -> None:
    if output_store.exists():
        if not overwrite:
            raise FileExistsError(f"Output store already exists: {output_store}; pass --overwrite to replace it")
        shutil.rmtree(output_store)
    if results_path.exists():
        if not overwrite:
            raise FileExistsError(f"Results path already exists: {results_path}; pass --overwrite to replace it")
        shutil.rmtree(results_path)
    output_store.parent.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).save_to_disk(str(output_store))
    with (results_path / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def default_output_paths(global_args: GlobalArgs) -> tuple[Path, Path]:
    store_path = project_path(global_args.store_name)
    parts = store_path.parts
    if "eval_outputs" not in parts:
        raise ValueError(f"Cannot derive salvage-fill output path from store_name={global_args.store_name}")
    idx = parts.index("eval_outputs")
    if len(parts) < idx + 3:
        raise ValueError(f"Expected eval_outputs/<label>/<dataset> in {global_args.store_name}")
    label = parts[idx + 1]
    dataset = parts[idx + 2]
    return ROOT / "eval_outputs" / "salvage-fill" / label / dataset, ROOT / "results" / "salvage-fill" / label / dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="salvage-fill truncated-completion salvage continuation")
    parser.add_argument("--config", required=True, help="Original direct eval config JSON")
    parser.add_argument("--output-store", default=None, help="salvage-fill output store path; default derives from direct store")
    parser.add_argument("--results-path", default=None, help="salvage-fill metrics path; default derives from direct results")
    parser.add_argument("--variant", choices=["commit_boxed", "analysis_tag"], default="commit_boxed")
    parser.add_argument("--allow-deprecated-variant", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sample-limit", type=int, default=None, help="Limit salvage jobs for debugging")
    parser.add_argument("--assert-prompt-samples", type=int, default=3)
    parser.add_argument("--self-validate-count", type=int, default=0)
    parser.add_argument("--self-validation-truncation", choices=["mixed", "case_a", "case_b"], default="mixed")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=None,
                        help="Salvage sampling seed; defaults to the evaluation config's seed.")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-rounds", type=int, choices=[1, 2], default=2)
    parser.add_argument("--logprobs", type=int, default=20)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.variant != "commit_boxed" and not args.allow_deprecated_variant:
        raise SystemExit("The analysis_tag variant requires --allow-deprecated-variant.")
    global_args, config = load_eval_config(project_path(args.config))
    if args.seed is None:
        args.seed = config.seed
    print(f"SALVAGE_SEED={args.seed}", flush=True)
    output_store, results_path = default_output_paths(global_args)
    if args.output_store:
        output_store = project_path(args.output_store)
    if args.results_path:
        results_path = project_path(args.results_path)
    ep0_store = project_path(global_args.store_name)
    store = load_from_disk(str(ep0_store))
    model_path = project_path(config.model) if not Path(config.model).is_absolute() else Path(config.model)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    bundle = build_prompts(global_args, config, tokenizer)
    assert_prompt_rebuild(bundle, store, args.assert_prompt_samples)
    records, jobs = make_main_records(store, bundle, config, args.variant)
    self_records, self_jobs = make_self_validation_records(store, bundle, config, args.variant, args.self_validate_count, args.seed, args.self_validation_truncation)
    offset = len(records)
    for job in self_jobs:
        job.record_index += offset
    records.extend(self_records)
    jobs.extend(self_jobs)
    if args.sample_limit is not None:
        selected_indices = {job.record_index for job in jobs[: args.sample_limit]}
        jobs = jobs[: args.sample_limit]
        for i, record in enumerate(records):
            if record["mode"] == "ep1" and record["ep0_conf_adherence"] == 0 and i not in selected_indices:
                record["layer"] = "residual"
                record["residual_reason"] = "not_selected_by_sample_limit"
    summary = {
        "config": rel_project_path(project_path(args.config)),
        "ep0_store": rel_project_path(ep0_store),
        "output_store": rel_project_path(output_store),
        "results_path": rel_project_path(results_path),
        "run_name": config.name,
        "variant": args.variant,
        "total_records": len([r for r in records if r["mode"] == "ep1"]),
        "salvage_jobs": len([j for j in jobs if j.mode == "ep1"]),
        "self_validation_jobs": len([j for j in jobs if j.mode == "self_validation"]),
        "case_a_jobs": sum(1 for j in jobs if j.case_type == "case(a)"),
        "case_b_jobs": sum(1 for j in jobs if j.case_type == "case(b)"),
        "dry_run": args.dry_run,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.dry_run:
        return
    run_jobs(records, jobs, config, tokenizer, args)
    metrics = compute_metrics(records, config.name, args.variant)
    write_outputs(records, metrics, output_store, results_path, args.overwrite)
    print(json.dumps({"wrote_store": rel_project_path(output_store), "wrote_metrics": rel_project_path(results_path / "metrics.json")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
