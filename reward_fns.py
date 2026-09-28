import math
import re
from math_verify import verify,parse
import numpy as np 
import string

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


# ---------- boxed helpers (nesting-safe) ----------

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return string[idx: right_brace_idx + 1]


def remove_boxed(s):
    if s is None:
        return None
    if "\\boxed " in s:
        left = "\\boxed "
        if s[:len(left)] == left:
            return s[len(left):]
        return None
    left = "\\boxed{"
    if s[:len(left)] == left and s[-1] == "}":
        return s[len(left):-1]
    return None


def extract_boxed_answer(text):
    """Return the content of the last \\boxed{...} in text, or None."""
    return remove_boxed(last_boxed_only_string(text))


_BAC_TAIL_PATTERN = r"\A<analysis>(?:(?!<analysis>).)*?</analysis>\s*<confidence>(.*?)</confidence>\s*\Z"


def bac_parts(content):
    """Split a bac completion into parts.

    Returns (pre_analysis_text, conf_str_or_None, ok_structure).
    ok_structure requires: after the FIRST <analysis>, the tail is exactly
    <analysis>...</analysis><confidence>...</confidence> (whitespace tolerant),
    AND a nesting-safe extractable \\boxed{...} exists before <analysis>.
    """
    idx = content.find("<analysis>")
    if idx < 0:
        return content, None, False
    pre, tail = content[:idx], content[idx:]
    m = re.match(_BAC_TAIL_PATTERN, tail, re.DOTALL)
    if not m:
        return pre, None, False
    if extract_boxed_answer(pre) is None:
        return pre, m.group(1), False
    return pre, m.group(1), True


def verify_gold_pred(gold, pred_text):
    """Boxed accuracy check (gold policy B)."""
    try:
        return bool(verify(parse(f"${gold}$"), parse(pred_text)))
    except Exception:
        return False


def format_reward(format_pattern,completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    if format_pattern == "bac":
        # boxed + <analysis> + <confidence> format
        completion_contents = [completion[0]["content"] for completion in completions]
        out = []
        for content in completion_contents:
            _, conf_str, ok = bac_parts(content)
            if not ok or conf_str is None:
                out.append(0.0)
                continue
            try:
                c = float(conf_str)
                out.append(1.0 if 0.0 <= c <= 1.0 else 0.0)
            except Exception:
                out.append(0.0)
        return out
    if format_pattern == "tbac":
        pattern = r".*?</think>\s*<analysis>.*?</analysis>\s*<answer>.*?</answer>\s*<confidence>.*?</confidence>\s*\Z"
    elif format_pattern == "ta":
        pattern = r".*?</think>\s*<answer>.*?</answer>\s*\Z"
    elif format_pattern == "tac":
        pattern = r".*?</think>\s*<answer>.*?</answer>\s*<confidence>.*?</confidence>\s*\Z" 
    elif format_pattern == "tabc":
        pattern = r".*?</think>\s*<answer>.*?</answer>\s*<analysis>.*?</analysis>\s*<confidence>.*?</confidence>\s*\Z"
    elif format_pattern == "abc":
        pattern = r"\s*<answer>.*?</answer>\s*<analysis>.*?</analysis>\s*<confidence>.*?</confidence>\s*\Z"
    else:
        raise ValueError(f"Invalid format pattern: {format_pattern}")
    confidence_pattern = r"<confidence>(.*?)</confidence>"

    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    matches = [1.0 if match else 0.0 for match in matches]
    
    #if it matches, check if the confidence is between 0 and 1
    for i,match in enumerate(matches):
        if match:
            content = completion_contents[i]
            if 'c' in format_pattern:
                confidence_matches = re.findall(confidence_pattern, content, re.DOTALL | re.MULTILINE)  # Get all <confidence>...</confidence> occurrences
                last_confidence = confidence_matches[-1] if confidence_matches else ""  # Get the last confidence, if exists
                if last_confidence == "":
                    matches[i] = 0.0
                else:
                    try:
                        confidence = float(last_confidence)
                        if confidence < 0 or confidence >1:
                            matches[i] = 0.0
                        else:
                            matches[i] = 1

                    except:
                        matches[i] = 0.0
    return matches

def accuracy_reward(format_pattern,completions,answer,source=None,**kwargs):
    """Reward function that extracts the last occurrence of text inside the answer tags and then checks if a label is present there"""
    completion_contents = [completion[0]["content"] for completion in completions]
    eval_contents = [e for e in answer]
    if format_pattern == "bac":
        # RLCR-style format cascade + boxed extraction + gold policy B
        format_rewards = format_reward(format_pattern, completions)
        matches = []
        for content, e, fr in zip(completion_contents, eval_contents, format_rewards):
            if fr == 0:
                matches.append(0)
                continue
            pre, _, _ = bac_parts(content)
            boxed = extract_boxed_answer(pre)
            if boxed is None:
                matches.append(0)
            else:
                matches.append(float(verify_gold_pred(e, f"\\boxed{{{boxed}}}")))
        return matches
    ans_pattern = r"<answer>(.*?)</answer>"
    matches = []
    format_rewards = format_reward(format_pattern,completions) 
    
    for content,e,fr in zip(completion_contents,eval_contents,format_rewards):
        if fr == 0:
            matches.append(0) 
        else:
            ans_matches = re.findall(ans_pattern, content, re.DOTALL | re.MULTILINE)  # Get all <answer>...</answer> occurrences
            last_answer = ans_matches[-1] if ans_matches else ""  # Get the last answer, if exists
            #if source exists in key and is equal to hotpot, then use the exact match score
            if source is not None and source[0] == 'hotpot':
                label = exact_match_score(last_answer,e)
            else:
                attempt = parse(last_answer)
                label = verify(e,attempt)
            matches.append(float(label))
    return matches

def brier_reward(format_pattern,completions,answer,source=None, **kwargs):
    """Reward function that checks if the completion is correct."""
    confidence_pattern = r"<confidence>(.*?)</confidence>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = []
    correctness_rewards = accuracy_reward(format_pattern,completions,answer,source) 
    format_rewards = format_reward(format_pattern,completions) 
    for content,cr,fr in zip(completion_contents,correctness_rewards,format_rewards):
        if fr == 0:
            matches.append(0) 
        else:
            #extract the confidence and give the reward as brier score
            confidence_matches = re.findall(confidence_pattern, content, re.DOTALL | re.MULTILINE)  # Get all <confidence>...</confidence> occurrences
            last_confidence = confidence_matches[-1] if confidence_matches else ""  # Get the last confidence, if exists
            if last_confidence == "":
                matches.append(0)
            else:
                try:
                    conf = float(last_confidence)
                    reward = 1 - (cr - conf)**2
                    matches.append(reward)
                except:
                    print("Could not parse confidence: ", last_confidence, "Something might be wrong")
                    matches.append(0)
    return matches

def mean_confidence_reward(completions,answer, **kwargs):
    """Reward function that extracts the last occurrence of text inside the answer tags and then checks if a label is present there"""
    confidence_pattern = r"<confidence>(.*?)</confidence>"
    completion_contents = [completion[0]["content"] for completion in completions]
    eval_contents = [e for e in answer] 
    matches = []

    for content,e in zip(completion_contents,eval_contents):
        confidence_matches = re.findall(confidence_pattern, content, re.DOTALL | re.MULTILINE)  # Get all <confidence>...</confidence> occurrences
        last_confidence = confidence_matches[-1] if confidence_matches else ""  # Get the last confidence, if exists
        if last_confidence == "":
            matches.append(0.0)
        else:
            try:
                confidence = float(last_confidence)
                #clip confidence to be between 0 and 1
                confidence = max(0.0, min(confidence, 1.0))
            except:
                confidence = 0.0
            matches.append(confidence)
    return matches

def confidence_one_or_zero(completions,answer, **kwargs):
    """Reward function that extracts the last occurrence of text inside the answer tags and then checks if a label is present there"""
    confidence_pattern = r"<confidence>(.*?)</confidence>"
    completion_contents = [completion[0]["content"] for completion in completions]
    eval_contents = [e for e in answer] 
    matches = []

    for content,e in zip(completion_contents,eval_contents):
        confidence_matches = re.findall(confidence_pattern, content, re.DOTALL | re.MULTILINE)  # Get all <confidence>...</confidence> occurrences
        last_confidence = confidence_matches[-1] if confidence_matches else ""  # Get the last confidence, if exists
        if last_confidence == "":
            matches.append(0.0)
        else:
            try:
                confidence = float(last_confidence)
                #clip confidence to be between 0 and 1
                confidence = max(0.0, min(confidence, 1.0))
            except:
                confidence = 0.0
            if abs(confidence - 1) < 0.01 or abs(confidence - 0) < 0.01:
                matches.append(1.0)
            else:
                matches.append(0.0)
    return matches


def bac_code_parts(content):
    """Coding-domain twin of :func:`bac_parts`.

    Identical structure gate -- after the FIRST <analysis> the tail must be exactly
    <analysis>...</analysis><confidence>...</confidence> -- with a closed ```python block
    before <analysis> in place of a nesting-safe \\boxed{...}.
    """
    from code_judge import extract_code

    idx = content.find("<analysis>")
    if idx < 0:
        return content, None, False
    pre, tail = content[:idx], content[idx:]
    m = re.match(_BAC_TAIL_PATTERN, tail, re.DOTALL)
    if not m:
        return pre, None, False
    if extract_code(pre) is None:
        return pre, m.group(1), False
    return pre, m.group(1), True


def format_code_reward(format_pattern, completions, **kwargs):
    """F' gate for the coding domain: bac structure with a fenced program as the answer.

    The coding structure gate is fixed rather than pattern-selected, but an unknown pattern
    is still rejected, mirroring :func:`format_reward`. Silently accepting ``bac_credo`` would
    apply this numeric-confidence gate to special-token output and hold F' at zero instead of
    failing loudly, which is the kind of misroute that only shows up as a dead training run.
    """
    if format_pattern not in ("bac",):
        raise ValueError(f"Invalid coding format pattern: {format_pattern} "
                         f"(the coding gate is 'bac'; 'bac_credo' output is read from logits "
                         f"and never scored through the reward registry)")
    completion_contents = [completion[0]["content"] for completion in completions]
    out = []
    for content in completion_contents:
        _, conf_str, ok = bac_code_parts(content)
        if not ok or conf_str is None:
            out.append(0.0)
            continue
        try:
            c = float(conf_str)
            out.append(1.0 if 0.0 <= c <= 1.0 else 0.0)
        except Exception:
            out.append(0.0)
    return out


# One optimisation step asks for accuracy_code and brier_code separately, and the brier
# reward needs the accuracy signal, so a naive implementation grades every completion twice
# -- structurally the same as the maths pair, but a full sandbox sweep instead of a
# millisecond of math_verify. Keyed by the exact graded text and its spec, so a hit can only
# be the same judgement; bounded because it lives for the whole run.
_CODE_SCORE_CACHE = {}
_CODE_SCORE_CACHE_MAX = 8192


def _cached_code_scores(jobs):
    """Grade ``[(graded_text, spec), ...]``, reusing verdicts already computed this run."""
    from code_judge import compute_scores

    scores = [None] * len(jobs)
    pending, positions = [], []
    for index, job in enumerate(jobs):
        hit = _CODE_SCORE_CACHE.get(job)
        if hit is None:
            pending.append(job)
            positions.append(index)
        else:
            scores[index] = hit
    if pending:
        if len(_CODE_SCORE_CACHE) + len(pending) > _CODE_SCORE_CACHE_MAX:
            _CODE_SCORE_CACHE.clear()
        for position, job, (score, _diag) in zip(positions, pending, compute_scores(pending)):
            scores[position] = float(score)
            _CODE_SCORE_CACHE[job] = float(score)
    return scores


def accuracy_code_reward(format_pattern, completions, answer, source=None, **kwargs):
    """Correctness in the coding domain: every test in the stored spec must pass.

    The whole batch goes to the judge at once because each item is graded in its own
    sandboxed subprocess, so the work parallelises across a thread pool.
    """
    completion_contents = [completion[0]["content"] for completion in completions]
    format_rewards = format_code_reward(format_pattern, completions)
    jobs, positions = [], []
    for index, (content, spec, fr) in enumerate(zip(completion_contents, answer, format_rewards)):
        if fr == 0:
            continue
        pre, _conf, _ok = bac_code_parts(content)
        jobs.append((pre, spec))
        positions.append(index)

    matches = [0.0] * len(completion_contents)
    if jobs:
        for position, score in zip(positions, _cached_code_scores(jobs)):
            matches[position] = score
    return matches


def brier_code_reward(format_pattern, completions, answer, source=None, **kwargs):
    """Coding-domain twin of :func:`brier_reward`, reusing the coding correctness signal."""
    confidence_pattern = r"<confidence>(.*?)</confidence>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = []
    correctness_rewards = accuracy_code_reward(format_pattern, completions, answer, source)
    format_rewards = format_code_reward(format_pattern, completions)
    for content, cr, fr in zip(completion_contents, correctness_rewards, format_rewards):
        if fr == 0:
            matches.append(0)
        else:
            confidence_matches = re.findall(confidence_pattern, content, re.DOTALL | re.MULTILINE)
            last_confidence = confidence_matches[-1] if confidence_matches else ""
            if last_confidence == "":
                matches.append(0)
            else:
                try:
                    conf = float(last_confidence)
                    matches.append(1 - (cr - conf) ** 2)
                except Exception:
                    matches.append(0)
    return matches


def _ungated_answer_segment(content):
    """Answer segment = text before the FIRST <analysis>; the whole text when the tag is absent.

    Byte-for-byte the segment rule of eval/check_functions.gen_correctness_reward_boxed and
    gen_correctness_reward_code, so an UNGATED training reward optimises exactly what the
    evaluation measures -- unlike the gated cascade above, where accuracy is zeroed whenever
    the bac structure fails even though the eval-side grader would still score the boxed answer.
    """
    idx = content.find("<analysis>")
    return content[:idx] if idx >= 0 else content


def accuracy_ungated_reward(completions, answer, source=None, **kwargs):
    """Pure task correctness, NO format gate (the grpo arm): last boxed of the answer segment.

    The signature deliberately has no format_pattern parameter: this reward must be unable to
    see the format even by accident, because the arm exists to measure what task-only RL does
    when nothing maintains the reporting interface.
    """
    matches = []
    for completion, gold in zip(completions, answer):
        pre = _ungated_answer_segment(completion[0]["content"])
        boxed = extract_boxed_answer(pre)
        if boxed is None:
            matches.append(0.0)
        else:
            matches.append(float(verify_gold_pred(gold, f"\\boxed{{{boxed}}}")))
    return matches


def accuracy_code_ungated_reward(completions, answer, source=None, **kwargs):
    """Coding twin of accuracy_ungated_reward: judge the answer segment, no format gate.

    Mirrors eval's gen_correctness_reward_code (pre-<analysis> text handed to the judge, which
    runs its own extractor). Shares the run-scoped verdict cache with the gated accuracy_code
    reward, so keeping that one registered at weight 0 for telemetry does not pay for a second
    sandbox sweep over the same texts.
    """
    jobs = [(_ungated_answer_segment(completion[0]["content"]), spec)
            for completion, spec in zip(completions, answer)]
    return [float(score) for score in _cached_code_scores(jobs)]


def assert_pure_acc_recipe(method, reward_funcs, reward_weights):
    """Config-time guard for method=grpo: the reward must be EXACTLY one weight-1.0 ungated acc.

    method=grpo exists to measure pure task reward; one stray nonzero weight silently turns the
    arm back into a shaped baseline and burns a ~36h run measuring the wrong construct, so a
    wrong recipe must die at parse time rather than surface in the result tables.
    """
    if method != "grpo":
        return
    pairs = [(name, float(weight)) for name, weight in zip(list(reward_funcs), list(reward_weights))]
    nonzero = [(name, weight) for name, weight in pairs if weight != 0.0]
    allowed = {"accuracy_ungated", "accuracy_code_ungated"}
    if len(nonzero) != 1 or nonzero[0][0] not in allowed or nonzero[0][1] != 1.0:
        raise ValueError(
            "method=grpo requires exactly one nonzero reward weight, equal to 1.0, placed on "
            f"accuracy_ungated or accuracy_code_ungated; got nonzero entries {nonzero!r} "
            f"from funcs {list(reward_funcs)!r}")


if __name__ == '__main__':
    s = "    h   ello whatever </think> <answer> The number of non-empty subsets 31 </answer> <confidence> 0.9 </confidence>   \n \n  "
 
    pattern = r".*?</think>\s*<answer>.*?</answer>\s*<confidence>.*?</confidence>\s*\Z" 
    match = re.match(pattern, s, re.DOTALL | re.MULTILINE)
    print(match)
    print(match[0])
