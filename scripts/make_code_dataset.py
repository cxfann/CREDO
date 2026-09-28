#!/usr/bin/env python
"""Build the coding-domain datasets consumed by rl_runner / evaluation.py.

Emits the same column contract as the math datasets -- ``problem`` (statement text,
consumed by ``dataset_processing.make_generation_conversation``), ``answer`` (JSON test
spec, consumed by the judge), ``source`` and ``difficulty`` -- so the coding domain
enters through new data rather than through edits to the existing code paths.

Sources and their conversions:

  train  deepcoder/*.parquet    ``tests`` is a JSON list of ``{type, input, output}``.
                                ``stdin_stdout`` cases are already stdin payloads;
                                ``function_call`` cases carry the argument list and a
                                one-element wrapper around the return value, so they are
                                re-encoded into the APPS convention (one JSON argument
                                per line, expected value unwrapped).
  eval   lcb/test*.jsonl        public + private cases; ``functional`` problems take
                                ``fn_name`` from ``metadata.func_name``.
  eval   humanevalplus/*.jsonl  an executable ``check()`` harness, kept as harness mode.

Input files are traversed in stable order and the held-out split uses a fixed seed.
Rebuilding requires the same source files, parameters, dependencies, and judge
environment. Reference-solution filtering can vary with execution timeouts.
Evaluation builds include cumulative lcb_v5/lcb_v6 splits as well as the three
bundled splits: lcb_v5w, lcb_v6w, and humanevalplus.

Usage:
  python scripts/make_code_dataset.py --raw <code_data dir> --out data/code-deepcoder
  python scripts/make_code_dataset.py --raw <code_data dir> --out data/code-evals --split eval
"""
import argparse
import base64
import collections
import glob
import hashlib
import io
import json
import os
import pickle
import random
import re
import sys
import zlib

_PREAMBLE = re.compile(
    r"^\s*solve the following coding problem using the programming language [a-z0-9+#]+\s*:\s*",
    re.I)
_TRAILER = re.compile(r"\s*now solve the problem and return the code\.\s*\Z", re.I)
_WS = re.compile(r"\s+")

MAX_DECOMPRESSED_BYTES = 512 * 1024 * 1024
DEFAULT_MIN_TESTS = 5
DEFAULT_MAX_TESTS = 32
DEFAULT_MAX_TEST_CHARS = 8192
# Match the math dataset exactly (9500 train / 500 test), which at the shared batch
# geometry -- pdbs 2 x 8 processes x grad-accum 64 / G 8 = 128 prompts per step -- gives the
# same 148 optimizer steps, so the two domains are compared at an equal training budget.
DEFAULT_HELD_OUT = 500
DEFAULT_TRAIN_SIZE = 9500
DEFAULT_SEED = 42

LCB_RELEASES = {
    "release_v5": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"],
    "release_v6": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl",
                   "test6.jsonl"],
}

# The contest-date windows the literature reports as "LiveCodeBench v5" and "v6", as
# (date_lo, date_hi, bounding release). Half-open on the right. Expected counts: v5w 279,
# v6w 131 -- the 279 is the number a published RL-for-code paper states for this window, so a
# rebuild that produces anything else means the raw release files changed under us.
LCB_WINDOWS = {
    "lcb_v5w": ("2024-08-01", "2025-02-01", "release_v5"),
    "lcb_v6w": ("2025-02-01", "2025-05-01", "release_v6"),
}
LCB_WINDOW_EXPECTED = {"lcb_v5w": 279, "lcb_v6w": 131}


class _NoGlobals(pickle.Unpickler):
    """Unpickler that refuses to resolve any global.

    Every pickle route to executing code needs a callable, and the only opcodes that can
    produce one (GLOBAL, STACK_GLOBAL, EXT1/2/4) go through ``find_class``. With this
    raising, the remaining opcodes can build primitive containers and nothing else.
    """

    def find_class(self, module, name):
        raise pickle.UnpicklingError("refused global %s.%s" % (module, name))


def decode_lcb_private(blob):
    """Decode LiveCodeBench ``private_test_cases``: base64 -> zlib -> pickle(str) -> json."""
    decompressor = zlib.decompressobj()
    raw = decompressor.decompress(base64.b64decode(blob), MAX_DECOMPRESSED_BYTES)
    if decompressor.unconsumed_tail:
        raise ValueError("private_test_cases exceeds %d bytes" % MAX_DECOMPRESSED_BYTES)
    payload = _NoGlobals(io.BytesIO(raw)).load()
    if not isinstance(payload, str):
        raise ValueError("unexpected private_test_cases payload %s" % type(payload))
    return json.loads(payload)


def strip_framing(text):
    """Drop the harvester's language preamble and 'now solve the problem' trailer.

    The I/O-convention sentence and any starter-code block are kept: they tell the model
    which interface the tests will exercise.
    """
    body = _PREAMBLE.sub("", text or "")
    return _TRAILER.sub("", body).strip()


def statement_key(text):
    return hashlib.blake2b(_WS.sub(" ", (text or "").strip().lower()).encode("utf-8"),
                           digest_size=16).hexdigest()


def subsample_indices(count, limit):
    """Evenly spread ``limit`` indices over ``range(count)``; identity when under the cap.

    Spread rather than head-truncation because test files are often ordered from small to
    large cases, so a prefix would systematically drop the discriminating tests.
    """
    if limit is None or count <= limit:
        return list(range(count))
    step = count / float(limit)
    picked = sorted({min(count - 1, int(i * step)) for i in range(limit)})
    return picked


def _within_cap(case, cap):
    """True when both the input and the output of a test case fit the cap."""
    if cap is None:
        return True
    for key in ("input", "output"):
        value = case.get(key)
        size = len(value) if isinstance(value, str) else len(json.dumps(value))
        if size > cap:
            return False
    return True


def spec_from_deepcoder_tests(tests, max_tests=DEFAULT_MAX_TESTS,
                              min_tests=DEFAULT_MIN_TESTS,
                              max_test_chars=DEFAULT_MAX_TEST_CHARS):
    """Convert DeepCoder's ``tests`` list into an APPS-convention io spec.

    Oversized individual tests are dropped before the count logic: the corpus carries a
    few multi-megabyte stress inputs that would blow the wall-clock budget and inflate the
    dataset by two orders of magnitude, while a problem's normal-sized tests are enough to
    decide correctness.
    """
    kinds = {case.get("type") for case in tests}
    if len(kinds) != 1:
        return None, "mixed_test_types:%s" % sorted(kinds)
    kind = kinds.pop()
    usable = [case for case in tests if _within_cap(case, max_test_chars)]
    if len(usable) < min_tests:
        return None, "too_few_tests_within_cap"
    keep = subsample_indices(len(usable), max_tests)
    inputs, outputs = [], []
    if kind == "stdin_stdout":
        for index in keep:
            case = usable[index]
            if not isinstance(case.get("input"), str) or not isinstance(case.get("output"), str):
                return None, "stdin_case_not_str"
            inputs.append(case["input"])
            outputs.append(case["output"])
        return {"mode": "io", "inputs": inputs, "outputs": outputs}, kind
    if kind == "function_call":
        fn_names = {case.get("fn_name") for case in tests}
        if len(fn_names) != 1 or not next(iter(fn_names)):
            return None, "inconsistent_fn_name"
        fn_name = fn_names.pop()
        for index in keep:
            case = usable[index]
            args, expected = case.get("input"), case.get("output")
            if not isinstance(args, list) or not isinstance(expected, list):
                return None, "function_case_not_list"
            # unwrap the one-element return wrapper: keeping it would let a solution that
            # returns [x] match an expected x through the harness's list fallback
            value = expected[0] if len(expected) == 1 else expected
            inputs.append("\n".join(json.dumps(arg) for arg in args))
            outputs.append(json.dumps(value))
        return {"mode": "io", "inputs": inputs, "outputs": outputs, "fn_name": fn_name}, kind
    return None, "unknown_test_type:%s" % kind


def convert_deepcoder_row(row, orig_index, min_tests=DEFAULT_MIN_TESTS,
                          max_tests=DEFAULT_MAX_TESTS,
                          max_test_chars=DEFAULT_MAX_TEST_CHARS):
    """Return ``(record, gold_code, reason)``; ``record`` is None when the row is dropped."""
    try:
        tests = json.loads(row["tests"]) if isinstance(row["tests"], str) else row["tests"]
    except (TypeError, ValueError):
        return None, None, "tests_unparseable"
    if not isinstance(tests, list) or len(tests) < min_tests:
        return None, None, "too_few_tests"
    spec, reason = spec_from_deepcoder_tests(tests, max_tests=max_tests, min_tests=min_tests,
                                             max_test_chars=max_test_chars)
    if spec is None:
        return None, None, reason
    problem = strip_framing(row["problem"])
    if not problem:
        return None, None, "empty_problem"
    solutions = list(row.get("solutions") or ())
    record = {
        "problem": problem,
        "answer": json.dumps(spec, sort_keys=True),
        "source": "deepcoder",
        "difficulty": "unknown",
        "orig_index": orig_index,
    }
    return record, (solutions[0] if solutions else None), reason


def convert_lcb_record(rec, orig_index, max_test_chars=DEFAULT_MAX_TEST_CHARS):
    """Return ``(record, reason)`` for one LiveCodeBench problem.

    Oversized individual tests are dropped under ``max_test_chars`` and the original count
    is kept in the spec as ``n_tests_total``: the decoded suite is ~1.35 GB because a few
    stress inputs reach 4.5 MB, so the cap is what makes the set shippable, and the
    bookkeeping lets a run be scored over the capped suite and over the subset of problems
    that lost nothing (a faithful LCB subset). Pass ``None`` to keep every test.
    """
    try:
        public = json.loads(rec.get("public_test_cases") or "[]")
        private = decode_lcb_private(rec["private_test_cases"]) if rec.get("private_test_cases") else []
    except Exception as exc:  # corrupt encoding is a data problem, not a crash
        return None, "test_decode_failed:%s" % type(exc).__name__
    cases = list(public) + list(private)
    if not cases:
        return None, "no_tests"
    kinds = {case.get("testtype") for case in cases}
    if len(kinds) != 1:
        return None, "mixed_test_types:%s" % sorted(kinds)
    kind = kinds.pop()
    n_total = len(cases)
    cases = [case for case in cases if _within_cap(case, max_test_chars)]
    if not cases:
        return None, "all_tests_over_cap"
    spec = {"mode": "io",
            "inputs": [case["input"] for case in cases],
            "outputs": [case["output"] for case in cases],
            "n_tests_total": n_total}
    if kind == "functional":
        try:
            fn_name = json.loads(rec.get("metadata") or "{}").get("func_name")
        except ValueError:
            fn_name = None
        if not fn_name:
            return None, "functional_without_func_name"
        spec["fn_name"] = fn_name
    elif kind != "stdin":
        return None, "unknown_test_type:%s" % kind
    if any(not isinstance(v, str) for v in spec["inputs"] + spec["outputs"]):
        return None, "case_not_str"

    problem = (rec.get("question_content") or "").strip()
    starter = (rec.get("starter_code") or "").strip()
    if starter:
        problem += ("\n\nWrite your solution by completing this code:\n\n```python\n%s\n```"
                    % starter)
    else:
        problem += "\n\nRead from standard input and write the answer to standard output."
    return {
        "problem": problem,
        "answer": json.dumps(spec, sort_keys=True),
        "source": "lcb:%s" % (rec.get("platform") or "?"),
        "difficulty": rec.get("difficulty") or "unknown",
        "orig_index": orig_index,
    }, kind


# HumanEval+ tasks whose shipped harness cannot pass for any solution. Excluded rather
# than repaired: rewriting benchmark test code would silently redefine the metric.
HUMANEVALPLUS_EXCLUDED = {
    "HumanEval/32": ("rendered harness asserts _poly(*candidate(*inp), inp); candidate "
                     "returns a float, so the unpack raises TypeError for every solution"),
}


def convert_humanevalplus_record(rec, orig_index):
    """Return ``(record, reason)``; HumanEval+ keeps its executable harness."""
    task_id = rec.get("task_id")
    if task_id in HUMANEVALPLUS_EXCLUDED:
        return None, "excluded:%s" % task_id
    harness, entry_point = rec.get("test"), rec.get("entry_point")
    if not harness or not entry_point:
        return None, "missing_harness_or_entry_point"
    spec = {"mode": "harness", "harness": harness, "entry_point": entry_point}
    problem = ("Complete the following Python function.\n\n```python\n%s\n```"
               % (rec.get("prompt") or "").rstrip())
    return {
        "problem": problem,
        "answer": json.dumps(spec, sort_keys=True),
        "source": "humanevalplus",
        "difficulty": "unknown",
        "orig_index": orig_index,
    }, "harness"


def iter_deepcoder(raw_root, batch_size=16):
    """Yield ``(orig_index, row)`` over every DeepCoder shard in filename order.

    Streamed in small batches: a whole row group holds thousands of rows whose raw test
    payloads reach tens of megabytes each, so materialising one group at a time pushed
    resident memory past two gigabytes.
    """
    import pyarrow.parquet as pq

    index = 0
    for path in sorted(glob.glob(os.path.join(raw_root, "deepcoder", "*.parquet"))):
        handle = pq.ParquetFile(path)
        for batch in handle.iter_batches(batch_size=batch_size,
                                         columns=["problem", "solutions", "tests"]):
            for row in batch.to_pylist():
                yield index, row
                index += 1
            del batch


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _grade_gold(batch, workers, hard_cap):
    """Grade a batch of ``(record, gold_code)`` pairs; returns the judge diagnostics."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import code_judge

    return code_judge.compute_scores([(gold, rec["answer"]) for rec, gold in batch],
                                     max_workers=workers, hard_cap=hard_cap)


def build_train(raw_root, min_tests, max_tests, held_out, seed,
                max_test_chars=DEFAULT_MAX_TEST_CHARS, train_size=DEFAULT_TRAIN_SIZE,
                gold_filter=True, gold_workers=8, hard_cap=90.0, batch_size=256):
    """Filter, deduplicate and convert the training candidate. Returns (train, test, stats).

    With ``gold_filter`` on, a problem survives only if its reference solution passes its
    own tests under this judge -- the shipped dataset then has a reference pass rate of
    1.0 by construction, which is what keeps the correctness label (and therefore the
    calibration target) free of harness-induced noise.
    """
    kept, seen, stats = [], {}, collections.Counter()
    pending = []

    def flush(batch):
        if not batch:
            return
        verdicts = _grade_gold(batch, gold_workers, hard_cap) if gold_filter else None
        for position, (record, _gold) in enumerate(batch):
            if verdicts is not None:
                score, diag = verdicts[position]
                stats["gold:%s" % diag["status"]] += 1
                if not score:
                    stats["dropped:gold_failed"] += 1
                    continue
            key = statement_key(record["problem"])
            if key in seen:
                stats["dropped:duplicate_statement"] += 1
                continue
            seen[key] = record["orig_index"]
            spec = json.loads(record["answer"])
            stats["kind:%s" % ("function_call" if "fn_name" in spec else "stdin_stdout")] += 1
            stats["tests_kept"] += len(spec["inputs"])
            kept.append(record)

    for orig_index, row in iter_deepcoder(raw_root):
        stats["rows_seen"] += 1
        record, gold, reason = convert_deepcoder_row(row, orig_index, min_tests, max_tests,
                                                     max_test_chars)
        if record is None:
            stats["dropped:%s" % reason] += 1
            continue
        if gold_filter and not gold:
            stats["dropped:no_gold"] += 1
            continue
        pending.append((record, gold))
        if len(pending) >= batch_size:
            flush(pending)
            pending = []
    flush(pending)

    kept.sort(key=lambda rec: rec["orig_index"])
    rng = random.Random(seed)
    order = list(range(len(kept)))
    rng.shuffle(order)
    held = sorted(order[:held_out])
    test = [kept[i] for i in held]
    surplus = order[held_out:]
    if train_size is not None and len(surplus) > train_size:
        stats["surplus_unused"] = len(surplus) - train_size
        surplus = surplus[:train_size]
    train = [kept[i] for i in surplus]
    train.sort(key=lambda rec: rec["orig_index"])
    assert len(train) + len(test) + stats.get("surplus_unused", 0) == len(kept)
    assert not (set(rec["orig_index"] for rec in test)
                & set(rec["orig_index"] for rec in train))
    stats["kept"] = len(kept)
    stats["train"] = len(train)
    stats["test"] = len(test)
    return train, test, stats


def build_evals(raw_root, max_test_chars=DEFAULT_MAX_TEST_CHARS):
    """Convert every evaluation set. Returns ``{name: (records, stats)}``.

    LiveCodeBench is walked once: the release file lists are nested, so v5 is exactly the
    prefix of v6 and is sliced out rather than decoded again -- decoding the private tests
    twice is what previously exhausted memory.

    Two kinds of LCB split come out of that single pass:

    * ``lcb_v5`` / ``lcb_v6`` -- the cumulative releases (879 / 1054 problems here). Kept for
      offline analysis and so that artifacts already produced against them stay interpretable.
    * ``lcb_v5w`` / ``lcb_v6w`` -- the **contest-date windows**, which is what the literature
      means when it writes "LiveCodeBench v5" and "v6". The v5 window has to be bounded to the
      release_v5 files as well as the dates: test5 ends and test6 begins on the same day
      (2025-01-04), so a date-only filter over all six files pulls in 44 test6 problems and
      yields 323 instead of the canonical 279.
    """
    out = {}
    v6_files = LCB_RELEASES["release_v6"]
    v5_files = set(LCB_RELEASES["release_v5"])
    missing = [f for f in v6_files if not os.path.exists(os.path.join(raw_root, "lcb", f))]
    records, stats, v5_count = [], collections.Counter(), 0
    dates = []  # contest_date and release membership of each KEPT record, positionally aligned
    if missing:
        stats["missing_files"] = len(missing)
    else:
        index = 0
        for name in v6_files:
            for rec in iter_jsonl(os.path.join(raw_root, "lcb", name)):
                record, reason = convert_lcb_record(rec, index, max_test_chars)
                index += 1
                stats["rows_seen"] += 1
                if record is None:
                    stats["dropped:%s" % reason] += 1
                    continue
                spec = json.loads(record["answer"])
                stats["kind:%s" % reason] += 1
                stats["tests_kept"] += len(spec["inputs"])
                stats["tests_over_cap"] += spec["n_tests_total"] - len(spec["inputs"])
                if spec["n_tests_total"] == len(spec["inputs"]):
                    stats["problems_with_full_suite"] += 1
                records.append(record)
                dates.append(((rec.get("contest_date") or "")[:10], name))
                if name in v5_files:
                    v5_count += 1
        stats["kept"] = len(records)
    out["lcb_v6"] = (records, stats)
    # report only what is true of the derived split; copying v6's counters made v5 look
    # like it carried v6's test totals
    v5_records = records[:v5_count]
    v5_stats = collections.Counter({
        "kept": len(v5_records),
        "derived_as_prefix_of_v6": 1,
        "problems_with_full_suite": sum(
            1 for rec in v5_records
            if json.loads(rec["answer"])["n_tests_total"] == len(json.loads(rec["answer"])["inputs"])),
    })
    out["lcb_v5"] = (v5_records, v5_stats)

    for split_name, (lo, hi, release) in LCB_WINDOWS.items():
        allowed = set(LCB_RELEASES[release])
        picked = [rec for rec, (day, src) in zip(records, dates)
                  if day and lo <= day < hi and src in allowed]
        win_stats = collections.Counter({
            "kept": len(picked),
            "derived_from_lcb_v6_by_contest_date": 1,
            "problems_with_full_suite": sum(
                1 for rec in picked
                if json.loads(rec["answer"])["n_tests_total"] == len(json.loads(rec["answer"])["inputs"])),
        })
        out[split_name] = (picked, win_stats)

    path = os.path.join(raw_root, "humanevalplus", "test.jsonl")
    records, stats = [], collections.Counter()
    if os.path.exists(path):
        for index, rec in enumerate(iter_jsonl(path)):
            stats["rows_seen"] += 1
            record, reason = convert_humanevalplus_record(rec, index)
            if record is None:
                stats["dropped:%s" % reason] += 1
                continue
            records.append(record)
        stats["kept"] = len(records)
    out["humanevalplus"] = (records, stats)
    return out


def save(splits, out_dir):
    """Persist as a DatasetDict so rl_runner's load_from_disk path works unchanged."""
    from datasets import Dataset, DatasetDict

    bundle = DatasetDict({name: Dataset.from_list(rows) for name, rows in splits.items()
                          if rows})
    os.makedirs(os.path.dirname(os.path.abspath(out_dir)) or ".", exist_ok=True)
    bundle.save_to_disk(out_dir)
    return {name: len(rows) for name, rows in splits.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, help="directory holding deepcoder/ lcb/ humanevalplus/")
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=["train", "eval"], default="train")
    parser.add_argument("--min-tests", type=int, default=DEFAULT_MIN_TESTS)
    parser.add_argument("--max-tests", type=int, default=DEFAULT_MAX_TESTS)
    parser.add_argument("--max-test-chars", type=int, default=DEFAULT_MAX_TEST_CHARS,
                        help="drop individual tests whose input or output exceeds this")
    parser.add_argument("--held-out", type=int, default=DEFAULT_HELD_OUT)
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE,
                        help="pin the train split to exactly N problems (0 = keep all)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-gold-filter", action="store_true",
                        help="keep problems whose reference solution fails its own tests")
    parser.add_argument("--gold-workers", type=int, default=8)
    parser.add_argument("--hard-cap", type=float, default=90.0)
    parser.add_argument("--dry-run", action="store_true", help="report counts without writing")
    args = parser.parse_args()

    report = {"raw": os.path.abspath(args.raw), "out": os.path.abspath(args.out),
              "split": args.split, "seed": args.seed}
    if args.split == "train":
        train, test, stats = build_train(args.raw, args.min_tests, args.max_tests,
                                         args.held_out, args.seed,
                                         max_test_chars=args.max_test_chars,
                                         train_size=args.train_size or None,
                                         gold_filter=not args.no_gold_filter,
                                         gold_workers=args.gold_workers,
                                         hard_cap=args.hard_cap)
        report["min_tests"] = args.min_tests
        report["max_tests"] = args.max_tests
        report["max_test_chars"] = args.max_test_chars
        report["train_size"] = args.train_size
        report["gold_filter"] = not args.no_gold_filter
        report["stats"] = dict(sorted(stats.items()))
        report["kept_orig_index"] = {"train": [r["orig_index"] for r in train],
                                     "test": [r["orig_index"] for r in test]}
        splits = {"train": train, "test": test}
    else:
        built = build_evals(args.raw, args.max_test_chars or None)
        report["max_test_chars"] = args.max_test_chars
        report["stats"] = {name: dict(sorted(stats.items())) for name, (_r, stats) in built.items()}
        splits = {name: records for name, (records, _s) in built.items()}
        report["lcb_windows"] = {k: list(v) for k, v in LCB_WINDOWS.items()}
        report["kept_orig_index"] = {name: [r["orig_index"] for r in splits[name]]
                                     for name in LCB_WINDOWS if name in splits}
        if built.get("lcb_v6", ([], {}))[0]:
            for name, expected in LCB_WINDOW_EXPECTED.items():
                got = len(splits.get(name, []))
                assert got == expected, (
                    "%s has %d problems, expected %d -- the raw LiveCodeBench release files "
                    "changed, so this build is a different benchmark than the one reported"
                    % (name, got, expected))

    report["sizes"] = {name: len(rows) for name, rows in splits.items()}
    if not args.dry_run:
        save(splits, args.out)
        with open(os.path.join(args.out, "build_report.json"), "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
