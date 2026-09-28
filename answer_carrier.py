"""Single source of truth for the answer-channel carrier of each domain.

Both the trainer's segmented scoring path and the evaluation pipeline need to know how the
answer is carried -- a ``\\boxed{}`` expression for maths, the last closed ```python block for
code. Keeping that mapping in one place is deliberate: two copies would be free to drift, and
this module is importable without torch/vllm so the maths selection can be asserted by identity
in the invariance gate.

The heavy imports are deferred into the call so that importing this module costs nothing.
"""

MATH_CARRIER_REGEX = r"\\boxed\{"
CODE_CARRIER_REGEX = r"```"


def answer_hooks(answer_format="math"):
    """Return ``(extract, verify_one, carrier_regex)`` for the requested domain.

    ``extract`` maps the answer segment to the extracted answer or None. ``verify_one`` grades a
    single extracted answer against the gold and is None for code, where grading is a batched
    subprocess operation and the caller must use the judge's batch entry point instead.
    ``carrier_regex`` is the fragment the readability gate looks for before ``<analysis>``.

    Anything other than ``"code"`` -- including the default and the eval pipeline's ``"boxed"``
    and ``"tags"`` -- resolves to the maths pair, so existing callers are unaffected.
    """
    if answer_format == "code":
        from code_judge import extract_code
        return extract_code, None, CODE_CARRIER_REGEX
    from reward_fns import extract_boxed_answer, verify_gold_pred
    return extract_boxed_answer, verify_gold_pred, MATH_CARRIER_REGEX


def answer_presence_fn(check_fn_args=None):
    """Return the "did the model answer at all" test implied by an eval config.

    ``answer_format`` lives inside ``check_fn_args`` (see ``eval/eval_args.py``), not at the top
    level of the config, so reading it from the wrong place would silently always pick maths.
    """
    fmt = (check_fn_args or {}).get("answer_format", "boxed")
    return answer_hooks(fmt)[0]
