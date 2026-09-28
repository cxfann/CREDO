"""Coding-domain judge: execute a candidate solution against a stored answer spec.

Public API::

    extract_code(completion)              -> str | None    last closed ```python block
    parse_answer(raw)                     -> dict | None    dict / JSON / python-repr
    compute_score(completion, answer)     -> (float, dict)  mode-aware, binary score
    compute_score_harness(c, test, entry) -> (float, dict)  explicit harness call
    compute_scores(items)                 -> list[(float, dict)]  thread-pooled batch

One stored ``answer`` field carries both dataset shapes, tagged by ``mode``:

    {"mode": "io",      "inputs": [...], "outputs": [...], "fn_name": optional}
    {"mode": "harness", "harness": "<check() code>", "entry_point": "<name>"}

``mode`` defaults to ``"io"``, so raw APPS-convention specs (the training candidates and
LiveCodeBench) work untagged; ``harness`` covers HumanEval+ / evalplus, whose benchmark
data is executable test code rather than input/output pairs.

Vendored from PRIME's ``prime_code`` (Apache-2.0). Two departures
from upstream: process isolation is rebuilt in :mod:`code_judge.sandbox_runner`, and the
score is binary all-tests-pass rather than upstream's per-test re-invocation loop (which
re-runs the whole solution once per test case).

``testing_util`` is imported **only inside the child process** -- it installs a global
``SIGALRM`` handler at import time, which must not leak into the trainer.
"""

import ast
import collections
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

from code_judge.sandbox_runner import (DEFAULT_HARD_CAP, DEFAULT_MEMORY_BYTES,
                                       DEFAULT_PER_TEST_TIMEOUT, DEFAULT_STARTUP_GRACE,
                                       JudgeResult, run_harness_in_sandbox,
                                       run_in_sandbox, total_budget)

__all__ = ["extract_code", "parse_answer", "compute_score", "compute_score_harness",
           "compute_scores", "JudgeResult", "run_in_sandbox", "run_harness_in_sandbox",
           "total_budget", "DEFAULT_PER_TEST_TIMEOUT", "DEFAULT_MEMORY_BYTES",
           "DEFAULT_HARD_CAP", "DEFAULT_STARTUP_GRACE",
           "JUDGE_STATS", "LIMITS_SEEN", "record_diagnostics", "format_stats"]

_FENCE = re.compile(r"```(?:python|py|python3)?[ \t]*\r?\n(.*?)```", re.DOTALL)

# Built from the same dataclass as the sandbox verdicts so every caller sees one schema:
# these two short-circuit before a sandbox runs, and a three-key dict used to make any
# consumer of n_passed_prefix / limits_applied raise KeyError on a no-code sample.
NO_CODE = JudgeResult(passed=False, status="no_code").as_dict()
BAD_SPEC = JudgeResult(passed=False, status="bad_spec").as_dict()

# Process-local judge telemetry. Every grading call site so far throws the per-item
# diagnostics away, which leaves the probe acceptance checklist unable to report the status
# distribution or to confirm that RLIMIT_AS took effect inside the container. Accumulating
# here keeps those readings available without threading a return channel through the reward
# and evaluation signatures.
JUDGE_STATS = collections.Counter()
LIMITS_SEEN = set()


def record_diagnostics(diagnostics):
    """Fold a batch of per-item diagnostics into the process-local counters."""
    for diag in diagnostics:
        JUDGE_STATS[diag.get("status", "unknown")] += 1
        for name in diag.get("limits_applied") or ():
            LIMITS_SEEN.add(name)


def format_stats():
    """One-line summary of everything graded so far in this process."""
    total = sum(JUDGE_STATS.values())
    spread = " ".join(f"{name}={count}" for name, count in sorted(JUDGE_STATS.items()))
    limits = ",".join(sorted(LIMITS_SEEN)) or "none"
    return f"JUDGE_STATS n={total} {spread} | limits_applied={limits}"


def extract_code(completion):
    """Return the last **closed** fenced code block, or None.

    Strict on purpose: an unclosed fence means the completion was truncated, which the
    format gate already rejects, so admitting the remainder would grade a fragment.
    """
    if not completion:
        return None
    blocks = _FENCE.findall(completion)
    if not blocks:
        return None
    code = blocks[-1]
    return code if code.strip() else None


def _as_dict(raw):
    """Decode a stored answer into a dict, accepting JSON and Python-repr strings."""
    value = raw
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError, MemoryError, RecursionError):
                return None
    return value if isinstance(value, dict) else None


def parse_answer(raw):
    """Validate and normalise a stored answer spec, or return None.

    In io mode every input and output must be a ``str``: the harness calls ``truncatefn``
    on both in both modes and that asserts ``isinstance(s, str)``, so any other type is
    guaranteed to fail grading. Rejecting it here turns a silent all-zero reward into a
    countable ``bad_spec``; normalising real datasets into string form (stdin payload for
    standard-input problems, one JSON-encoded argument per line for call-based problems)
    belongs to the preprocessing step.
    """
    spec = _as_dict(raw)
    if spec is None:
        return None
    mode = spec.get("mode", "io")
    if mode == "harness":
        harness, entry_point = spec.get("harness"), spec.get("entry_point")
        if not isinstance(harness, str) or not harness.strip():
            return None
        if not isinstance(entry_point, str) or not entry_point.strip():
            return None
        return {"mode": "harness", "harness": harness, "entry_point": entry_point}
    if mode != "io":
        return None
    inputs, outputs = spec.get("inputs"), spec.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        return None
    if len(inputs) != len(outputs) or not inputs:
        return None
    if any(not isinstance(item, str) for item in inputs):
        return None
    if any(not isinstance(item, str) for item in outputs):
        return None
    clean = {"mode": "io", "inputs": inputs, "outputs": outputs}
    fn_name = spec.get("fn_name")
    if isinstance(fn_name, str) and fn_name:
        clean["fn_name"] = fn_name
    return clean


def compute_score(completion, answer, per_test_timeout=DEFAULT_PER_TEST_TIMEOUT,
                  total_timeout=None, memory_bytes=DEFAULT_MEMORY_BYTES,
                  startup_grace=DEFAULT_STARTUP_GRACE, hard_cap=DEFAULT_HARD_CAP,
                  code=None, lenient_wrapping=False):
    """Grade one completion. Returns ``(1.0 if every test passes else 0.0, diagnostics)``.

    Pass ``code`` to skip extraction when the caller already isolated the code block.

    ``lenient_wrapping`` restores upstream's ``output == expected[0]`` acceptance for
    call-based problems (see :mod:`code_judge.testing_util`). It is a
    caller flag rather than a spec field on purpose: ``parse_answer`` whitelists spec keys,
    so no dataset row can turn grading lenient. The default uses strict comparison;
    the flag is available for comparisons with the upstream judge.
    """
    solution = code if code is not None else extract_code(completion)
    if solution is None:
        diagnostics = dict(NO_CODE)
    else:
        spec = parse_answer(answer)
        if spec is None:
            diagnostics = dict(BAD_SPEC)
        else:
            limits = dict(per_test_timeout=per_test_timeout, total_timeout=total_timeout,
                          memory_bytes=memory_bytes, startup_grace=startup_grace,
                          hard_cap=hard_cap)
            if spec["mode"] == "harness":
                result = run_harness_in_sandbox(spec["harness"], spec["entry_point"],
                                                solution, **limits)
            else:
                judged = {"inputs": spec["inputs"], "outputs": spec["outputs"]}
                if "fn_name" in spec:
                    judged["fn_name"] = spec["fn_name"]
                if lenient_wrapping:
                    judged["lenient_wrapping"] = True
                result = run_in_sandbox(judged, solution, **limits)
            diagnostics = result.as_dict()
    record_diagnostics([diagnostics])
    return (1.0 if diagnostics["passed"] else 0.0), diagnostics


def compute_score_harness(completion, harness, entry_point, **kwargs):
    """Grade one completion against an executable ``check()`` harness (HumanEval+).

    The harness is the benchmark's own test code, executed verbatim, so the verdict is
    the official one modulo sandboxing.
    """
    return compute_score(completion,
                         {"mode": "harness", "harness": harness, "entry_point": entry_point},
                         **kwargs)


def compute_scores(items, max_workers=None, **kwargs):
    """Grade ``[(completion, answer), ...]`` concurrently, preserving order.

    Each job is its own subprocess, so the pool is IO-bound from Python's point of view
    and workers default to the core count because every child runs single-threaded.
    """
    items = list(items)
    if not items:
        return []
    if max_workers is None:
        max_workers = min(16, os.cpu_count() or 4)
    if max_workers <= 1 or len(items) == 1:
        return [compute_score(c, a, **kwargs) for c, a in items]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(lambda pair: compute_score(pair[0], pair[1], **kwargs), items))
