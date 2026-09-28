from eval.eval_utils import compute_pass_n, get_brier, get_ece, get_auroc, exact_match_score
import numpy as np
from math_verify import verify, parse
import re
from vllm import LLM, SamplingParams
import gc
from transformers import AutoTokenizer

def confidence_extractor(response, **kwargs):
    """Extracts the confidence from the completions
    If a float is found within confidence tags, it is processed as follows:
    If the float is between 0 and 1, it is returned as is.
    If the float is between 1 and 100, it is divided by 100 and returned.
    If float is not directly found, the first number in the string is extracted and processed as above.
    If no float is found, 0 is returned.    
    """
    conf_pattern = r"<confidence>(.*?)</confidence>"
    # Get all <confidence>...</confidence> occurrences
    conf_matches = re.findall(conf_pattern, response, re.DOTALL | re.MULTILINE)
    # Get the last confidence, if exists
    last_confidence = conf_matches[-1] if conf_matches else ""
    if last_confidence == "":
        return 0, 0.0
    else:
        try:
            confidence = float(last_confidence)
            if confidence > 1 and confidence <= 100:
                return 1, confidence/100
            elif confidence >= 0 and confidence <= 1:
                return 1, confidence
            else:
                return 0, 0.0
        except:
            # extract the first number in the string
            first_number = re.search(r'-?\d+(?:\.\d+)?', last_confidence)
            if first_number:
                first_number = float(first_number.group())
                if first_number >= 0 and first_number <= 1:
                    return 1, first_number
                elif first_number > 1 and first_number <= 100:
                    return 1, first_number/100
                else:
                    return 0, 0.0
            else:
                return 0, 0.0


def gen_correctness_reward(completions, answer, **kwargs):
    """Reward function that checks if the answer is correct or not
    The answer must be present within the answer tags.
    For math datasets, the correctness is checked using huggingface math-verify.
    For factual datasets, the correctness is checked using exact match.

    """
    ans_pattern = r"<answer>(.*?)</answer>"
    completion_contents = [completion[0]["content"]
                           for completion in completions]
    eval_contents = [e for e in answer]
    matches = []

    for content, e in zip(completion_contents, eval_contents):
        # Get all <answer>...</answer> occurrences
        ans_matches = re.findall(ans_pattern, content,
                                 re.DOTALL | re.MULTILINE)
        # Get the last answer, if exists
        last_answer = ans_matches[-1] if ans_matches else ""
        attempt = parse(last_answer)
        label = verify(e, attempt)
        if label ==0 :
            label = exact_match_score(last_answer, e)
        matches.append(float(label))

    return matches

def gen_correctness_reward_boxed(completions, answer, **kwargs):
    """Boxed correctness: last \\boxed{...} + math_verify with gold policy B.

    Note: unlike the tag-based fn, no exact-match fallback on the raw string — boxed
    content is math; policy B already handles numeric/format equivalences.
    """
    from reward_fns import extract_boxed_answer, verify_gold_pred
    matches = []
    for completion, e in zip(completions, answer):
        content = completion[0]["content"]
        # only consider text before <analysis> (answer segment) to avoid boxed inside analysis
        idx = content.find("<analysis>")
        pre = content[:idx] if idx >= 0 else content
        boxed = extract_boxed_answer(pre)
        if boxed is None:
            matches.append(0.0)
        else:
            matches.append(float(verify_gold_pred(e, f"\\boxed{{{boxed}}}")))
    return matches


def gen_correctness_reward_code(completions, answer, **kwargs):
    """Coding-domain correctness: last fenced program in the answer segment must pass all tests.

    Mirrors :func:`gen_correctness_reward_boxed`: only the text before <analysis> counts, so
    a code block written inside the analysis is never graded. The whole call is one judge
    batch -- each item runs in its own sandboxed subprocess, so the work parallelises only
    if the caller hands over every sample at once.
    """
    from code_judge import compute_scores, format_stats

    jobs = []
    for completion, spec in zip(completions, answer):
        content = completion[0]["content"]
        idx = content.find("<analysis>")
        pre = content[:idx] if idx >= 0 else content
        jobs.append((pre, spec))
    # graded in chunks so a run can be diagnosed while it is still running; the chunk is far
    # larger than the worker count, so batching still parallelises the way it should
    chunk = 512
    scores = []
    for start in range(0, len(jobs), chunk):
        scores.extend(compute_scores(jobs[start:start + chunk]))
        print("[code_judge] %d/%d graded | %s"
              % (len(scores), len(jobs), format_stats()), flush=True)
    return [float(score) for score, _diag in scores]


def confidence_verifier(local_dataset, config, format_fn="confidence_format", format_pattern="tabc", answer_format="tags", **kwargs):
    label_dict = {f"{config.name}-evals": []}
    evals = []
    c_lengths = []
    confidence_levels = []
    conf_format_levels = []
    metrics = {}
    n = config.n
    if answer_format == "code":
        correctness_fn = gen_correctness_reward_code
    else:
        correctness_fn = gen_correctness_reward_boxed if answer_format == "boxed" else gen_correctness_reward

    if f"{config.name}-class_output" in local_dataset.column_names:
        #If classification outputs are present (these come from classifier/probe)
        class_outputs = local_dataset[f"{config.name}-class_output"]
    else:
        class_outputs = None

    ### CHECK CORRECTNESS ###

    # Coding-domain grading spawns one sandboxed subprocess per sample, so it has to be
    # submitted as a single batch: called once per sample the judge takes its serial branch
    # and every launch is unparallelised, which idles the GPU for the whole judging phase.
    # The maths path keeps its per-sample call so its behaviour is untouched.
    batched_correctness = None
    if answer_format == "code":
        flat_completions, flat_answers = [], []
        for i in range(len(local_dataset)):
            row = local_dataset[i]
            for j in range(n):
                flat_completions.append([{"role": "assistant",
                                          "content": row[f"{config.name}-output_{j}"]}])
                flat_answers.append(row["answer"])
        # pure-regex snapshot before the judge runs: seconds, not hours, so a format collapse
        # is visible immediately. The three rates say where samples are being lost -- code
        # present but confidence unparsed means the completion was cut off inside <analysis>,
        # while code absent means it was cut off before the program.
        from code_judge import extract_code
        from reward_fns import bac_code_parts

        parts = [bac_code_parts(c[0]["content"]) for c in flat_completions]
        total = max(len(parts), 1)
        print("[code_eval] pre-judge snapshot n=%d structure_ok=%.4f conf_parsed=%.4f code_block=%.4f"
              % (len(parts),
                 sum(1 for _p, _c, ok in parts if ok) / total,
                 sum(1 for _p, c, _ok in parts if c is not None) / total,
                 sum(1 for p, _c, _ok in parts if extract_code(p) is not None) / total),
              flush=True)
        batched_correctness = correctness_fn(completions=flat_completions, answer=flat_answers)

    for i in range(len(local_dataset)):
        eval_list, c_len_list, conf_list, conf_format_list = [], [], [], []
        for j in range(n):
            pred_response = local_dataset[i][f"{config.name}-output_{j}"]
            answer = local_dataset[i]["answer"]
            pred = [{"role": "assistant", "content": pred_response}]

            if batched_correctness is not None:
                actual_correctness = batched_correctness[i * n + j]
            else:
                args = {"completions": [pred], "answer": [answer]}

                actual_correctness = correctness_fn(**args)[0]
            conf_format, conf_level = confidence_extractor(pred_response)
            conf_format_list.append(conf_format)

            c_len_list.append(len(pred[0]["content"]))
            conf_list.append(conf_level)
            if actual_correctness == 1:
                eval_list.append(1)
            else:
                eval_list.append(0)

        evals.append(eval_list)
        c_lengths.append(c_len_list)
        confidence_levels.append(conf_list)
        conf_format_levels.append(conf_format_list)

    ### END OF CHECK CORRECTNESS ###
     
    ### COMPUTE PASS@K ###
    if n not in config.pass_k_vals:
        config.pass_k_vals.append(n)
    if 1 not in config.pass_k_vals:
        config.pass_k_vals.append(1)
    for k in config.pass_k_vals:
        if k <= n:
            pass_k = compute_pass_n(evals, k)
            metrics[f"pass@{k}"] = pass_k

    ### END OF COMPUTE PASS@K ###

    if class_outputs is not None:
        #If classification outputs are present (these come from classifier/probe), then we use the corresponding confidence levels
        if type(class_outputs[0]) == list:
            confidence_levels = [[c[1]] for c in class_outputs]
            print("Overriding confidence levels with classification outputs")
        else:
            confidence_levels = [ [c] for c in class_outputs]

    # take mean of c_lengths
    c_length_mean = np.mean(np.array(c_lengths))

    label_dict[f"{config.name}-evals"] = evals
    label_dict[f"{config.name}-c_lengths"] = c_lengths
    label_dict[f"{config.name}-confidence_levels"] = confidence_levels
    label_dict[f"{config.name}-conf_format_adherence"] = conf_format_levels

    correctness_array = np.array(evals).flatten()
    confidence_array = np.array(confidence_levels).flatten()
    metrics["brier_score"] = get_brier(correctness_array, confidence_array) 
    metrics["ece"] = get_ece(correctness_array, confidence_array)
    metrics["auroc"] = get_auroc(correctness_array, confidence_array)

    # Parallel digit-expectation channel: same labels, second confidence
    # column, reported side-by-side under a nested namespace. A missing column
    # leaves the namespace absent, indicating "not measured".
    digit_key = f"{config.name}-confidence_levels_digit"
    if digit_key in local_dataset.column_names:
        digit_array = np.array(local_dataset[digit_key]).flatten().astype(float)
        digit_adh = np.array(local_dataset[f"{config.name}-conf_digit_adherence"]).flatten().astype(float)
        assert digit_array.shape == correctness_array.shape, (digit_array.shape, correctness_array.shape)
        int_mass = np.array(local_dataset[f"{config.name}-conf_digit_int_mass"]).flatten().astype(float)
        digit_mass = np.array(local_dataset[f"{config.name}-conf_digit_digit_mass"]).flatten().astype(float)
        _covered = digit_adh == 1
        metrics["digit"] = {
            "brier_score": get_brier(correctness_array, digit_array),
            "ece": get_ece(correctness_array, digit_array),
            "auroc": get_auroc(correctness_array, digit_array),
            "confidence level": float(np.mean(digit_array)),
            "coverage": float(np.mean(digit_adh)),
            "int_mass_mean_covered": float(np.mean(int_mass[_covered])) if _covered.any() else None,
            "digit_mass_mean_covered": float(np.mean(digit_mass[_covered])) if _covered.any() else None,
            "int_mass_p10_covered": float(np.percentile(int_mass[_covered], 10)) if _covered.any() else None,
            "digit_mass_p10_covered": float(np.percentile(digit_mass[_covered], 10)) if _covered.any() else None,
        }

    metrics["accuracy"] = metrics["pass@1"]
    metrics["completion length"] = c_length_mean
    metrics["confidence level"] = np.mean(np.array(confidence_levels))
    metrics["confidence format adherence"] = np.mean(
        np.array(conf_format_levels))

    print(f"Metrics of {config.name} =")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    return label_dict, metrics


def llm_confidence_verifier(local_dataset, config, judge_model="meta-llama/Llama-3.1-8B-Instruct", format_fn="confidence_format", **kwargs):
    label_dict = {f"{config.name}-evals": []}
    evals = []
    c_lengths = []
    confidence_levels = []
    conf_format_levels = []
    metrics = {}
    n = config.n

    if f"{config.name}-class_output" in local_dataset.column_names:
        class_outputs = local_dataset[f"{config.name}-class_output"]
    else:
        class_outputs = None

    # FIRST EXTRACT OUT ALL ANSWERS FROM THE MODEL OUTPUTS. 
    extracted_answers = []
    for i in range(len(local_dataset)):
        q_spec_ans = []
        for j in range(n):
            pred = local_dataset[i][f"{config.name}-output_{j}"]
            ans_pattern = r"<answer>(.*?)</answer>"
            # Get all <answer>...</answer> occurrences
            ans_matches = re.findall(
                ans_pattern, pred, re.DOTALL | re.MULTILINE)
            # Get the last answer, if exists
            last_answer = ans_matches[-1] if ans_matches else ""
            if last_answer == "":
                last_answer = "I don't know"
            q_spec_ans.append(last_answer)
        extracted_answers.append(q_spec_ans)

    ####### DO LLM AS JUDGE SETUP #######
    sys_prompt = """
    You are a judge that will be given a question,ground truth answers and a model generated answer. There might be multiple ground truth answers. 
    The model generated answer is correct if it matches any of the ground truth answers.
    You will need to determine if the model generated answer is correct or not. 
    Your response should be a single word. 'YES' if the answer is correct and 'NO' if it is not.
    """

    prompts = []
    chosen_key = "question" if "question" in local_dataset.column_names else "problem"
    tokenizer = AutoTokenizer.from_pretrained(judge_model, trust_remote_code=True)
    
    #Generate prompts for each example
    for i in range(len(local_dataset)):
        for j in range(n):
            prompt = f"""
            Question: {local_dataset[i][chosen_key]}
            Ground Truth Answers: {local_dataset[i]["answer"]}
            Model Generated Answer: {extracted_answers[i][j]}
            """
            processed_prompt = [{'role': 'system', 'content': sys_prompt}, {
                'role': 'user', 'content': prompt}]
            tokenized_prompt = tokenizer.apply_chat_template(
                processed_prompt, truncation=False, add_generation_prompt=True)
            decoded_prompt = tokenizer.decode(tokenized_prompt)
            prompts.append(decoded_prompt)

    # Setup LLM and send prompts
    sampling_params = SamplingParams(n=1, temperature=0, max_tokens=20)
    llm = LLM(model=judge_model, gpu_memory_utilization=0.8)
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    ####### END OF LLM AS JUDGE SETUP #######

    ####### AGGREGATE RESPONSES #######

    responses = []
    for output in outputs:
        text_r = output.outputs[0].text
        if "yes" in text_r.lower():
            responses.append(1)
        else:
            responses.append(0)

    agg_responses = []
    # agg responses by taking groups of n and making a list of them
    for i in range(0, len(responses), n):
        agg_responses.append(responses[i:i+n])

    ####### END OF AGGREGATE RESPONSES #######

    # Compute accuracy
    accuracy = np.mean(responses)
    print(f"Accuracy of {config.name} = {accuracy}")

    for i in range(len(local_dataset)):
        eval_list, c_len_list, conf_list, conf_format_list = [], [], [], []
        for j in range(n):
            pred_response = local_dataset[i][f"{config.name}-output_{j}"]
            pred = [{"role": "assistant", "content": pred_response}]

            actual_correctness = agg_responses[i][j]
            conf_format, conf_level = confidence_extractor(pred_response)
            conf_format_list.append(conf_format)

            c_len_list.append(len(pred[0]["content"]))
            conf_list.append(conf_level)
            if actual_correctness == 1:
                eval_list.append(1)
            else:
                eval_list.append(0)

        evals.append(eval_list)
        c_lengths.append(c_len_list)
        confidence_levels.append(conf_list)
        conf_format_levels.append(conf_format_list)


    if n not in config.pass_k_vals:
        config.pass_k_vals.append(n)
    if 1 not in config.pass_k_vals:
        config.pass_k_vals.append(1)
    for k in config.pass_k_vals:
        if k <= n:
            pass_k = compute_pass_n(evals, k)
            metrics[f"pass@{k}"] = pass_k

    if class_outputs is not None:
        if type(class_outputs[0]) == list:
            confidence_levels = [[c[1]] for c in class_outputs]
            print("Overriding confidence levels with class outputs")
        else:
            confidence_levels = [ [c] for c in class_outputs]

    correctness_array = np.array(evals).flatten()
    confidence_array = np.array(confidence_levels).flatten()
    metrics["brier_score"] = get_brier(correctness_array, confidence_array) 
    metrics["ece"] = get_ece(correctness_array, confidence_array)
    metrics["auroc"] = get_auroc(correctness_array, confidence_array)

    # take mean of c_lengths
    c_length_mean = np.mean(np.array(c_lengths))

    label_dict[f"{config.name}-evals"] = evals
    label_dict[f"{config.name}-c_lengths"] = c_lengths
    label_dict[f"{config.name}-confidence_levels"] = confidence_levels
    label_dict[f"{config.name}-conf_format_adherence"] = conf_format_levels

    metrics["accuracy"] = metrics["pass@1"]
    metrics["completion length"] = c_length_mean
    metrics["confidence level"] = np.mean(np.array(confidence_levels))
    metrics["confidence format adherence"] = np.mean(
        np.array(conf_format_levels))

    print(f"Metrics of {config.name} =")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    del llm
    gc.collect()
    return label_dict, metrics
