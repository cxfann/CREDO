"""Isolated executor for one (test-spec, candidate-solution) pair.

Launched as ``python -m code_judge.sandbox_child <tmpdir> <per_test_timeout> <memory_bytes>``
by :mod:`code_judge.sandbox_runner`, which owns the process group, the wall-clock
kill and the temp directory. This module never returns to its caller: it writes a
single sentinel-prefixed JSON line and calls ``os._exit``.

Hardening applied here (the rest lives in the parent):
  * result channel moved off fd 1 so candidate ``print`` output can never corrupt it
  * fd 0/1/2 redirected to /dev/null (blind stdin, silenced stdout/stderr)
  * RLIMIT_AS/DATA/STACK/FSIZE/CORE set after the heavy imports, before candidate code
  * sockets disabled
  * every failure mode is reported as a status string, never as an exception
"""

import io
import json
import os
import sys

SENTINEL = "__JUDGE_RESULT__"
DETAIL_CAP = 2000


def _install_result_channel():
    """Move the result fd away from 1 and blind the standard streams.

    ``sys.stdout``/``sys.stderr`` are deliberately left pointing at fd 1/2 -- the
    dup2 below already redirects those fds, and rebinding the objects risks having
    a garbage-collected wrapper close the fd underneath the candidate code.
    """
    res_fd = os.dup(1)
    devnull = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(devnull, fd)
    os.close(devnull)
    return res_fd


def _emit(res_fd, payload):
    with io.open(res_fd, "wb", closefd=True) as fh:
        fh.write((SENTINEL + " " + json.dumps(payload) + "\n").encode("utf-8"))
        fh.flush()


def _set_limits(memory_bytes):
    """Cap address space, file size and core dumps. Returns the limits that stuck."""
    import resource

    applied = {}
    caps = [("RLIMIT_AS", memory_bytes), ("RLIMIT_DATA", memory_bytes),
            ("RLIMIT_STACK", memory_bytes), ("RLIMIT_FSIZE", 64 * 1024 * 1024),
            ("RLIMIT_CORE", 0)]
    for name, value in caps:
        which = getattr(resource, name, None)
        if which is None:
            continue
        try:
            resource.setrlimit(which, (value, value))
            applied[name] = value
        except (ValueError, OSError):
            pass
    return applied


def _disable_sockets():
    import socket

    def _blocked(*_args, **_kwargs):
        raise RuntimeError("network access is disabled in the code judge sandbox")

    socket.socket = _blocked
    socket.create_connection = _blocked
    socket.create_server = _blocked
    socket.socketpair = _blocked


def _truncate(text):
    text = str(text)
    return text if len(text) <= DETAIL_CAP else text[:DETAIL_CAP] + "...(truncated)"


def classify(raw_results, n_tests):
    """Map testing_util's result vector onto a status + pass prefix.

    ``run_test`` returns early on the first non-pass, so the vector is a prefix of
    the test list: ``[True, ..., True, <verdict>]``. Codes are ``-2`` compile error
    and ``-1`` runtime error/timeout; per-test verdicts are (possibly numpy) bools.
    """
    normalized = []
    for item in raw_results:
        if item is True:
            normalized.append("pass")
            continue
        if item is False:
            normalized.append("fail")
            continue
        value = item.item() if hasattr(item, "item") else item
        if value is True:
            normalized.append("pass")
        elif value is False:
            normalized.append("fail")
        elif value == -2:
            normalized.append("compile_error")
        elif value == -1:
            normalized.append("runtime_error")
        else:
            normalized.append("fail")

    n_passed = 0
    for verdict in normalized:
        if verdict != "pass":
            break
        n_passed += 1

    if not normalized:
        status = "harness_error"
    elif n_passed == len(normalized) == n_tests:
        status = "passed"
    elif n_passed == len(normalized):
        # every executed test passed but the harness stopped short of the spec
        status = "incomplete"
    else:
        status = {"fail": "wrong_answer"}.get(normalized[n_passed], normalized[n_passed])

    return status, n_passed, normalized


HARNESS_PRELUDE = (
    "import sys\n"
    "sys.setrecursionlimit(6 * 10 ** 5)\n"
    "from typing import *\n"
)


def run_harness(solution, harness, entry_point, timeout):
    """Grade against an executable ``def check(candidate)`` harness (HumanEval+ style).

    Returns the same status vocabulary as the io path so callers need no special case.
    """
    import signal

    from code_judge import testing_util

    program = "%s\n%s\n\n%s\n\ncheck(%s)\n" % (
        HARNESS_PRELUDE, solution, harness, entry_point)
    try:
        compiled = compile(program, "<harness>", "exec")
    except SyntaxError as exc:
        return "compile_error", "%s: %s" % (type(exc).__name__, exc)

    testing_util.reliability_guard()
    namespace = {"__name__": "__solution__"}
    signal.alarm(int(timeout))
    try:
        exec(compiled, namespace)
        return "passed", ""
    except AssertionError as exc:
        return "wrong_answer", _truncate("AssertionError: %s" % exc)
    except BaseException as exc:
        return "runtime_error", _truncate("%s: %s" % (type(exc).__name__, exc))
    finally:
        signal.alarm(0)


def main():
    res_fd = _install_result_channel()
    try:
        tmpdir, per_test_timeout, memory_bytes = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
        with io.open(os.path.join(tmpdir, "job.json"), "r", encoding="utf-8") as fh:
            job = json.load(fh)
        mode = job.get("mode", "io")
        solution = job["solution"]

        from code_judge import testing_util  # heavy imports (numpy) before the caps

        applied = _set_limits(memory_bytes)
        _disable_sockets()

        if mode == "harness":
            status, detail = run_harness(solution, job["harness"], job["entry_point"],
                                         per_test_timeout)
            n_tests = 1
            n_passed = 1 if status == "passed" else 0
            n_executed = 1
        else:
            spec = job["spec"]
            n_tests = len(spec["inputs"])
            raw_results, metadata = testing_util.run_test(
                in_outs=spec, test=solution, debug=False, timeout=int(per_test_timeout)
            )
            status, n_passed, normalized = classify(raw_results, n_tests)
            n_executed = len(normalized)
            detail = ""
            if isinstance(metadata, dict):
                for key in ("error", "error_message", "traceback"):
                    if metadata.get(key):
                        detail = _truncate(metadata[key])
                        break
        _emit(res_fd, {
            "status": status,
            "n_tests": n_tests,
            "n_passed_prefix": n_passed,
            "n_executed": n_executed,
            "detail": detail,
            "limits_applied": sorted(applied),
        })
        os._exit(0)
    except BaseException as exc:  # never let the child die without a verdict
        try:
            _emit(res_fd, {
                "status": "harness_error",
                "n_tests": -1,
                "n_passed_prefix": 0,
                "n_executed": 0,
                "detail": _truncate("%s: %s" % (type(exc).__name__, exc)),
                "limits_applied": [],
            })
        except BaseException:
            pass
        os._exit(1)


if __name__ == "__main__":
    main()
