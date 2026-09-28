"""Process-level isolation for the code judge (parent side).

One candidate solution is graded by one short-lived ``python -m code_judge.sandbox_child``
subprocess. A subprocess (rather than ``fork``/``multiprocessing``) is used on purpose:
the judge is called from inside the training loop of a multi-threaded, CUDA-initialised
process, where forking risks inheriting a held lock and deadlocking the child.

Hardening owned by this module:
  * ``start_new_session=True`` puts the child in its own process group, so a wall-clock
    overrun kills the whole group -- grandchildren cannot survive as orphans
  * two timeout layers: per-test ``SIGALRM`` inside the child, plus this wall-clock kill
  * a private temp directory is the child's cwd/HOME/TMPDIR and is removed afterwards
  * the accepted result payload is byte-capped
  * every failure mode maps to a status string; the caller never sees an exception
"""

import errno
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

from code_judge.sandbox_child import SENTINEL

PACKAGE_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_PER_TEST_TIMEOUT = 6.0
DEFAULT_STARTUP_GRACE = 15.0
DEFAULT_HARD_CAP = 90.0
DEFAULT_MEMORY_BYTES = 4 * 1024 ** 3
MAX_RESULT_BYTES = 256 * 1024

PASS_STATUS = "passed"


@dataclass
class JudgeResult:
    """Outcome of grading one solution. ``passed`` is the only thing the reward uses."""

    passed: bool
    status: str
    n_tests: int = -1
    n_passed_prefix: int = 0
    n_executed: int = 0
    elapsed_s: float = 0.0
    detail: str = ""
    limits_applied: list = field(default_factory=list)

    def as_dict(self):
        return {
            "passed": self.passed, "status": self.status, "n_tests": self.n_tests,
            "n_passed_prefix": self.n_passed_prefix, "n_executed": self.n_executed,
            "elapsed_s": round(self.elapsed_s, 3), "detail": self.detail,
            "limits_applied": list(self.limits_applied),
        }


def total_budget(n_tests, per_test_timeout=DEFAULT_PER_TEST_TIMEOUT,
                 startup_grace=DEFAULT_STARTUP_GRACE, hard_cap=DEFAULT_HARD_CAP):
    """Wall-clock ceiling for one grading call.

    ``SIGALRM`` cannot interrupt a pure-Python infinite loop (the vendored handler
    returns instead of raising), so ``per_test_timeout * n_tests`` is not a real
    bound and ``hard_cap`` is the load-bearing limit.
    """
    return min(hard_cap, startup_grace + per_test_timeout * max(1, n_tests))


def _child_env(tmpdir):
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": PACKAGE_PARENT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "HOME": tmpdir, "TMPDIR": tmpdir,
    })
    return env


def _kill_group(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError) as exc:
        if getattr(exc, "errno", None) not in (errno.ESRCH, errno.EPERM, None):
            raise
        try:
            proc.kill()
        except ProcessLookupError:
            pass


def _parse_payload(stdout_bytes):
    if len(stdout_bytes) > MAX_RESULT_BYTES:
        stdout_bytes = stdout_bytes[-MAX_RESULT_BYTES:]
    text = stdout_bytes.decode("utf-8", errors="replace")
    for line in reversed(text.splitlines()):
        if line.startswith(SENTINEL):
            try:
                return json.loads(line[len(SENTINEL):].strip())
            except ValueError:
                return None
    return None


def _run_job(job, n_tests, per_test_timeout, total_timeout, memory_bytes):
    """Spawn one graded subprocess and turn whatever happens into a JudgeResult."""
    tmpdir = tempfile.mkdtemp(prefix="code_judge_%d_" % os.getpid())
    started = time.monotonic()
    try:
        with open(os.path.join(tmpdir, "job.json"), "w", encoding="utf-8") as fh:
            json.dump(job, fh)
        proc = subprocess.Popen(
            [sys.executable, "-m", "code_judge.sandbox_child",
             tmpdir, str(per_test_timeout), str(int(memory_bytes))],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            cwd=tmpdir, env=_child_env(tmpdir), start_new_session=True,
        )
        try:
            stdout_bytes, _ = proc.communicate(timeout=total_timeout)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            try:
                proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            return JudgeResult(passed=False, status="timeout", n_tests=n_tests,
                               elapsed_s=time.monotonic() - started,
                               detail="wall-clock budget %.1fs exceeded" % total_timeout)
        finally:
            if proc.poll() is None:
                _kill_group(proc)

        payload = _parse_payload(stdout_bytes)
        elapsed = time.monotonic() - started
        if payload is None:
            return JudgeResult(passed=False, status="crashed", n_tests=n_tests, elapsed_s=elapsed,
                               detail="no verdict from child (returncode=%s)" % proc.returncode)
        status = payload.get("status", "harness_error")
        return JudgeResult(
            passed=(status == PASS_STATUS), status=status,
            n_tests=payload.get("n_tests", n_tests),
            n_passed_prefix=payload.get("n_passed_prefix", 0),
            n_executed=payload.get("n_executed", 0),
            elapsed_s=elapsed, detail=payload.get("detail", ""),
            limits_applied=payload.get("limits_applied", []),
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_in_sandbox(spec, solution, per_test_timeout=DEFAULT_PER_TEST_TIMEOUT,
                   total_timeout=None, memory_bytes=DEFAULT_MEMORY_BYTES,
                   startup_grace=DEFAULT_STARTUP_GRACE, hard_cap=DEFAULT_HARD_CAP):
    """Grade ``solution`` against an input/output ``spec`` in a killable subprocess."""
    n_tests = len(spec.get("inputs") or ())
    if n_tests == 0:
        return JudgeResult(passed=False, status="no_tests", n_tests=0)
    if total_timeout is None:
        total_timeout = total_budget(n_tests, per_test_timeout, startup_grace, hard_cap)
    return _run_job({"mode": "io", "spec": spec, "solution": solution}, n_tests,
                    per_test_timeout, total_timeout, memory_bytes)


def run_harness_in_sandbox(harness, entry_point, solution,
                           per_test_timeout=DEFAULT_PER_TEST_TIMEOUT, total_timeout=None,
                           memory_bytes=DEFAULT_MEMORY_BYTES,
                           startup_grace=DEFAULT_STARTUP_GRACE, hard_cap=DEFAULT_HARD_CAP):
    """Grade ``solution`` against an executable ``check()`` harness (HumanEval+ style).

    The harness is one atomic verdict rather than a test vector, so ``n_tests`` is 1.
    """
    if total_timeout is None:
        total_timeout = total_budget(1, per_test_timeout, startup_grace, hard_cap)
    job = {"mode": "harness", "harness": harness, "entry_point": entry_point,
           "solution": solution}
    return _run_job(job, 1, per_test_timeout, total_timeout, memory_bytes)
