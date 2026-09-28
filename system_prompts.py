"""System prompts for the confidence-channel comparison.

Every method shares the same three-segment format --- a step-by-step solution that
ends in an answer, an <analysis> of the solution's uncertainty, and a <confidence>
slot. The channels differ only in what goes in the confidence slot:

  * verbalized channel (Base / GRPO / RLCR / DCPO): a decimal number in [0, 1];
  * readout channel (CREDO): exactly one of the reserved tokens <CONF_HIGH> /
    <CONF_LOW>, whose relative probability is read as the confidence.

The coding prompts are literal isomorphs of the math prompts: only the answer carrier
changes (a fenced ```python block instead of \\boxed{}), so that the four-method
comparison is never confounded by differences in the analysis or confidence wording.
"""

_BAC_COMMON_HEAD = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "Solve the problem step by step, and put your final answer within \\boxed{}. "
    "After the final answer, analyze the uncertainty of your solution within <analysis> </analysis> tags. "
    "This analysis is the basis for the confidence level you will report next, and must follow these rules: "
    "(1) point out specific steps that could be wrong or ambiguous, including alternative approaches that might lead to different answers; "
    "(2) do not solve the problem again and do not revise or change your answer; "
    "(3) be specific - if you cannot find more uncertainties, say so explicitly. "
)

# Verbalized channel (Base / GRPO / RLCR / DCPO): a decimal confidence in <confidence> tags.
BAC_BOXED_PROMPT = (
    _BAC_COMMON_HEAD
    + "Then provide your confidence that the final answer is correct, as a decimal number between 0 and 1 "
    "(e.g. 0.3 or 0.8), within <confidence> </confidence> tags. "
    "The final format that must be followed is: {step-by-step solution with \\boxed{final answer}} "
    "<analysis> uncertainty analysis here </analysis> <confidence> confidence here </confidence>"
)

# Readout channel (CREDO): a single reserved token in <confidence> tags.
BAC_BOXED_CREDO_PROMPT = (
    _BAC_COMMON_HEAD
    + "Then, within <confidence> </confidence> tags, output exactly one token: <CONF_HIGH> if your final answer "
    "is more likely correct than not, otherwise <CONF_LOW>. "
    "The final format that must be followed is: {step-by-step solution with \\boxed{final answer}} "
    "<analysis> uncertainty analysis here </analysis> <confidence><CONF_HIGH></confidence>"
)

# No-analysis ablation: the <analysis> segment is removed wholesale and the confidence
# readout follows the boxed answer directly. Otherwise a literal mirror of the CREDO
# prompt, so the ablation isolates exactly one component.
BAC_BOXED_CREDO_NOCT_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "Solve the problem step by step, and put your final answer within \\boxed{}. "
    "After the final answer, within <confidence> </confidence> tags, output exactly one token: "
    "<CONF_HIGH> if your final answer is more likely correct than not, otherwise <CONF_LOW>. "
    "The final format that must be followed is: {step-by-step solution with \\boxed{final answer}} "
    "<confidence><CONF_HIGH></confidence>"
)

# Coding domain. Deliberately a literal isomorph of _BAC_COMMON_HEAD: only the answer
# carrier changes (a fenced Python block instead of \boxed{}), so that the four-method
# comparison is never confounded by differences in the analysis or confidence wording.
_BAC_CODE_COMMON_HEAD = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "Reason step by step, then give your complete final program in a single ```python code block. "
    "The last code block in your response is the one that will be run, so it must be the whole program "
    "and must follow the input/output interface described in the problem. "
    "After the final answer, analyze the uncertainty of your solution within <analysis> </analysis> tags. "
    "This analysis is the basis for the confidence level you will report next, and must follow these rules: "
    "(1) point out specific steps that could be wrong or ambiguous, including alternative approaches that might lead to different answers; "
    "(2) do not solve the problem again and do not revise or change your answer; "
    "(3) be specific - if you cannot find more uncertainties, say so explicitly. "
)

BAC_BOXED_CODE_PROMPT = (
    _BAC_CODE_COMMON_HEAD
    + "Then provide your confidence that the final answer is correct, as a decimal number between 0 and 1 "
    "(e.g. 0.3 or 0.8), within <confidence> </confidence> tags. "
    "The final format that must be followed is: {step-by-step reasoning with the final program in a "
    "```python code block} "
    "<analysis> uncertainty analysis here </analysis> <confidence> confidence here </confidence>"
)

BAC_BOXED_CODE_CREDO_PROMPT = (
    _BAC_CODE_COMMON_HEAD
    + "Then, within <confidence> </confidence> tags, output exactly one token: <CONF_HIGH> if your final answer "
    "is more likely correct than not, otherwise <CONF_LOW>. "
    "The final format that must be followed is: {step-by-step reasoning with the final program in a "
    "```python code block} "
    "<analysis> uncertainty analysis here </analysis> <confidence><CONF_HIGH></confidence>"
)

# No-analysis ablation, coding domain: literal mirror of the code CREDO prompt minus the
# <analysis> instructions and slot.
BAC_BOXED_CODE_CREDO_NOCT_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "Reason step by step, then give your complete final program in a single ```python code block. "
    "The last code block in your response is the one that will be run, so it must be the whole program "
    "and must follow the input/output interface described in the problem. "
    "After the final answer, within <confidence> </confidence> tags, output exactly one token: "
    "<CONF_HIGH> if your final answer is more likely correct than not, otherwise <CONF_LOW>. "
    "The final format that must be followed is: {step-by-step reasoning with the final program in a "
    "```python code block} "
    "<confidence><CONF_HIGH></confidence>"
)


def get_sys_prompt(sys_prompt_name):
    prompts = {
        "bac_boxed": BAC_BOXED_PROMPT,
        "bac_boxed_credo": BAC_BOXED_CREDO_PROMPT,
        "bac_boxed_credo_noct": BAC_BOXED_CREDO_NOCT_PROMPT,
        "bac_boxed_code": BAC_BOXED_CODE_PROMPT,
        "bac_boxed_code_credo": BAC_BOXED_CODE_CREDO_PROMPT,
        "bac_boxed_code_credo_noct": BAC_BOXED_CODE_CREDO_NOCT_PROMPT,
    }
    if sys_prompt_name not in prompts:
        raise ValueError(f"Invalid system prompt name: {sys_prompt_name}")
    return prompts[sys_prompt_name]
