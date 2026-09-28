"""Digit-expectation confidence readout for text-confidence methods (base/grpo/rlcr/dcpo).

Two-step teacher-forced estimator:
  Context A = prompt + generation-prefix-to-anchor            -> P(int digit = "0"|"1")
  Context B = Context A + "0."                                -> E[first decimal digit]
  c = p1 * 1.0 + p0 * (E[d]/10),   p0/p1 renormalized over {0,1}

Anchor policy = DIGIT-FIRST: when the sample's
own text-channel parse yields a [0,1]-scale number inside a closed <confidence> tag, the
anchor sits right before that number's first digit, so the model's own spacing/newlines
stay inside the prefix and the integer-digit distribution is read exactly where the model
itself placed it. Everything else falls back to appending a canonical opener at the end
of the generation (salvage-style "</analysis>\n<confidence>" when an <analysis> block is
left unclosed, else "\n<confidence>").

Eligibility (direct-protocol semantics): a sample gets a digit reading iff the
text channel has a reading (adherence 1) OR a pre-analysis \\boxed{} answer exists.
Truncated samples with neither stay (label 0, conf 0) on BOTH channels, so the two
channels use the same coverage rule.

All contexts are built at STRING level (ans_at_end /
confidence_at_end / salvage injections all concatenate text); vLLM re-tokenizes. Qwen3
pretokenizes digits one-per-token (ids 15..24 contiguous), and
"0." never merges, so the forced suffix is tokenization-safe.

Parsing and probability helpers do not require vLLM; generation imports it on demand.
"""

import math
import re
from typing import Any, Optional

CONF_TAG = "<confidence>"
FALLBACK_INJ_ANALYSIS_OPEN = "</analysis>\n<confidence>"
FALLBACK_INJ_PLAIN = "\n<confidence>"
FORCED_DECIMAL_SUFFIX = "0."

_CLOSED_TAG_RE = re.compile(r"<confidence>(.*?)</confidence>", re.DOTALL | re.MULTILINE)
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def digit_token_ids(tokenizer: Any) -> dict:
    """Single-token ids for "0".."9" (asserted) plus the two integer candidates."""
    ids = {}
    for d in range(10):
        enc = tokenizer.encode(str(d), add_special_tokens=False)
        assert len(enc) == 1, f"digit {d!r} is not a single token: {enc}"
        ids[d] = enc[0]
    return ids


def find_anchor(text: str) -> tuple[str, str]:
    """Locate the digit-expectation anchor in a generation text.

    Returns (mode, prefix):
      mode "digit"    -> prefix = text up to (excluding) the first digit of the number the
                         text channel parses ([0,1]-scale, last closed tag only);
      mode "fallback" -> prefix = text + canonical opener injection (readout after it).
    """
    matches = list(_CLOSED_TAG_RE.finditer(text))
    if matches:
        m = matches[-1]
        content = m.group(1)
        num = _NUMBER_RE.search(content)
        if num is not None:
            try:
                val = float(num.group())
            except Exception:
                val = None
            # digit-first anchoring only for the [0,1] scale the models are trained on;
            # percent-scale (1,100] emissions keep the text-channel /100 parse but read
            # the digit channel via the canonical fallback opener.
            if val is not None and 0 <= val <= 1:
                digit_in_num = re.search(r"\d", num.group())
                anchor_char = m.start(1) + num.start() + digit_in_num.start()
                return "digit", text[:anchor_char]
    if "<analysis>" in text and "</analysis>" not in text[text.rfind("<analysis>"):]:
        return "fallback", text + FALLBACK_INJ_ANALYSIS_OPEN
    return "fallback", text + FALLBACK_INJ_PLAIN


def synthesize(p_int: dict, q_digit: dict, id_map: dict) -> tuple[Optional[float], float, float]:
    """Combine the two teacher-forced readings into a confidence.

    p_int:  {token_id: prob} at the Context-A readout position (top-k, unnormalized)
    q_digit:{token_id: prob} at the Context-B readout position
    Returns (conf | None, int_mass_A, digit_mass_B); conf is None when either target
    mass is zero at this k (recorded, upgrade path decided from measured masses).
    """
    p0 = p_int.get(id_map[0], 0.0)
    p1 = p_int.get(id_map[1], 0.0)
    int_mass = p0 + p1
    qs = {d: q_digit.get(id_map[d], 0.0) for d in range(10)}
    digit_mass = sum(qs.values())
    if int_mass <= 0.0 or digit_mass <= 0.0:
        return None, int_mass, digit_mass
    p0n, p1n = p0 / int_mass, p1 / int_mass
    e_digit = sum(d * q for d, q in qs.items()) / digit_mass
    return p1n * 1.0 + p0n * (e_digit / 10.0), int_mass, digit_mass


def probs_from_logprobs(lp_map: Any) -> dict:
    """vLLM {token_id: Logprob} -> {token_id: prob}."""
    if lp_map is None:
        return {}
    return {tid: math.exp(lp.logprob) for tid, lp in lp_map.items()}


def build_contexts(prompt_text: str, generation_text: str) -> tuple[str, str, str]:
    """(mode, ctxA, ctxB) for one sample; ctxB = ctxA + "0."."""
    mode, prefix = find_anchor(generation_text)
    ctx_a = prompt_text + prefix
    return mode, ctx_a, ctx_a + FORCED_DECIMAL_SUFFIX


def run_digit_expectation(llm: Any, tokenizer: Any, tasks: list, logprobs_k: int = 50) -> list:
    """Batched two-context readout. tasks = [(prompt_text, generation_text), ...].

    Returns per task: dict(conf, adherence, mode, int_mass, digit_mass).
    Uses one llm.generate call over 2N prompts (max_tokens=1, temperature=0), reading
    the next-token top-k distribution at each context end. Requires the LLM engine to
    have been constructed with max_logprobs >= logprobs_k.
    """
    from vllm import SamplingParams

    id_map = digit_token_ids(tokenizer)
    modes, prompts = [], []
    for prompt_text, generation_text in tasks:
        mode, ctx_a, ctx_b = build_contexts(prompt_text, generation_text)
        modes.append(mode)
        prompts.extend([ctx_a, ctx_b])
    params = SamplingParams(n=1, temperature=0, max_tokens=1, logprobs=logprobs_k)
    outs = llm.generate(prompts, sampling_params=params)
    results = []
    for i, mode in enumerate(modes):
        pa = outs[2 * i].outputs[0]
        pb = outs[2 * i + 1].outputs[0]
        lp_a = pa.logprobs[0] if pa.logprobs else None
        lp_b = pb.logprobs[0] if pb.logprobs else None
        conf, int_mass, digit_mass = synthesize(
            probs_from_logprobs(lp_a), probs_from_logprobs(lp_b), id_map)
        results.append({
            "conf": conf,
            "adherence": 0 if conf is None else 1,
            "mode": mode,
            "int_mass": int_mass,
            "digit_mass": digit_mass,
        })
    return results
