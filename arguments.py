from dataclasses import dataclass, field
import trl 
from trl import ScriptArguments 
from typing import Optional, Any
from transformers import TrainingArguments
from trl import SFTConfig

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty'.
        cosine_min_value_wrong (`float`):
            Minimum reward for cosine scaling for wrong answers.
        cosine_max_value_wrong (`float`):
            Maximum reward for cosine scaling for wrong answers.
        cosine_min_value_correct (`float`):
            Minimum reward for cosine scaling for correct answers.
        cosine_max_value_correct (`float`):
            Maximum reward for cosine scaling for correct answers.
        cosine_max_len (`int`):
            Maximum length for cosine scaling.
    """

    dataset_name: str = field(metadata={"help": "Dataset name."})
    dataset_config: Optional[str] = field(
        default=None,
        metadata={
            "help": "Dataset configuration name. Corresponds to the `name` argument of the `datasets.load_dataset` "
            "function."
        },
    )
    dataset_train_split: str = field(default="train", metadata={"help": "Dataset split to use for training."})
    dataset_test_split: str = field(default="test", metadata={"help": "Dataset split to use for evaluation."})

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty'"
        },
    )

    sys_prompt_name: str = field(default="ver", metadata={"help": "System prompt name."})
    task_spec: str = field(default="gen", metadata={"help": "Task specification."})
    set_pad_token: Optional[int] = field(default=None, metadata={"help": "Set the pad token to this id"})

    orm_key: Optional[str] = field(
        default="reward")
    
    train_subset_size : Optional[int] = field(
        default=None,
        metadata={"help": "Size of the training subset."},
    )

    eval_subset_size : Optional[int] = field(
        default=None,
        metadata={"help": "Size of the evaluation subset."},
    )

    format_pattern: Optional[str] = field(
        default="ta",
        metadata={"help": "The format pattern to use for the reward function."},
    )

    enable_thinking: Optional[bool] = field(
        default=None,
        metadata={"help": "Optional tokenizer chat template thinking flag for Qwen3 no-thinking ablations."},
    )



@dataclass
class GRPOConfig(trl.GRPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    run_name: Optional[str] = field(
        default=None,
        metadata={"help": "An optional descriptor for the run. Notably used for wandb, mlflow and comet logging."},
    )

    # Parameters that control the training
    learning_rate: float = field(
        default=1e-6,
        metadata={
            "help": "Initial learning rate for `AdamW` optimizer. The default value replaces that of "
            "`transformers.TrainingArguments`."
        },
    )

    per_device_train_batch_size: int = field(
        default=8,
        metadata={
            "help": "Number of prompts sampled per device for training. The actual batch passed into the model will "
            "be this value multiplied by `num_generations`."
        },
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={
            "help": "Number of updates steps to accumulate the gradients for, before performing a backward/update "
            "pass."
        },
    )
    steps_per_generation: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of optimization steps per generation. If `None`, it defaults to gradient_accumulation_steps."
        },
    )

    generation_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Batch size to use for generation. If `None`, it defaults to the effective training batch size: "
            "`per_device_train_batch_size * num_processes * gradient_accumulation_steps`."
        },
    )

    beta: float = field(
        default=0.04,
        metadata={"help": "KL coefficient."},
    )


    epsilon: float = field(
        default=0.2,
        metadata={"help": "Epsilon value for clipping."},
    )

    epsilon_high: Optional[float] = field(
        default=None,
        metadata={
            "help": "Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the "
            "lower-bound specified in argument `epsilon`. Paper DAPO recommends `0.28`."
        },
    )

    disable_dropout: bool = field(
        default=True,
        metadata={
            "help": "Whether to disable dropout in the model. This is useful for training with a reference model, as "
            "it prevents the model from generating different logprobs for the same input."
        },
    )

    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    system_prompt: Optional[str] = field(
        default=None, metadata={"help": "The optional system prompt to use for benchmarking."}
    )
    wandb_project: Optional[str] = field(
        default="RLCR",
        metadata={"help": ("The project to store runs under.")},
    )
    max_prompt_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left."
        },
    )
    num_generations: Optional[int] = field(
        default=4,
        metadata={"help": "Number of generations to sample."},
    )
    temperature: Optional[float] = field(
        default=0.9,
        metadata={"help": "Temperature for sampling. The higher the temperature, the more random the completions."},
    )
    max_completion_length: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum length of the generated completion."},
    )

    use_vllm: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use vLLM for generating completions. If set to `True`, ensure that a GPU is kept "
            "unused for training, as vLLM will require one for generation. vLLM must be installed "
            "(`pip install vllm`)."
        },
    )

    vllm_mode: str = field(
        default="colocate",
        metadata={
            "help": "Mode to use for vLLM integration when `use_vllm` is set to `True`. Must be one of `server` or "
            "`'colocate'`. `'server'`: The trainer will send generation requests to a separate vLLM server. Make sure a "
            "TRL vLLM server is running (start with `trl vllm-serve`). `'colocate'`: vLLM will run in the same "
            "process and share the training GPUs. This avoids the need for a separate server but may cause resource "
            "contention with training."
        },
    )

    vllm_gpu_memory_utilization: float = field(
        default=0.9,
        metadata={
            "help": "Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV "
            "cache on the device dedicated to generation powered by vLLM. Higher values will increase the KV cache "
            "size and thus improve the model's throughput. However, if the value is too high, it may cause "
            "out-of-memory (OOM) errors during initialization."
        },
    )

    vllm_tensor_parallel_size: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of tensor parallel GPUs to use."
        },
    )

    # Parameters that control the logging
    log_completions: bool = field(
        default=True,
        metadata={"help": "Whether to log the completions during training."},
    )

    completion_logging_steps: Optional[int] = field(default=5, metadata={"help": "Log completions every n steps."}) 
    num_completions_to_log: Optional[int] = field(default=3, metadata={"help": "Number of completions to log."})
    eval_log_keys: list[str] = field(
        default_factory=lambda: ["answer", "eval"],
        metadata={
            "help": "Keys to log during hf eval"
        },
    )

    mask_truncated_completions: bool = field(
        default=False,
        metadata={
            "help": "When enabled, truncated completions are excluded from the loss calculation, preventing them from "
            "being incorrectly penalized and introducing noise during training. According to the DAPO paper, this is "
            "a good practice for training stability."
        },
    )

    loss_type: str = field(
        default="bnpo",
        metadata={
            "help": "Specifies the loss formulation to use. Supported values are `grpo`, `bnpo`, and `dr_grpo`. "
            "`'grpo'`: Aggregates token-level losses by normalizing over sequence length. Not recommended due to "
            "length bias—this approach tends to prefer shorter completions with positive advantages and longer ones "
            "with negative advantages. "
            "`'bnpo'`: Aggregates token-level losses by normalizing number of active token in the local batch. "
            "Note that normalization is performed over the local batch only, so results may slightly vary depending "
            "on the local batch size, despite a constant effective batch size. When using "
            "`per_device_train_batch_size==1`, the loss is equivalent to the GRPO loss. "
            "`'dr_grpo'`: Aggregates token-level losses by normalizing with a global constant. This method was "
            "introduced in the Dr. GRPO paper to eliminate length bias. The value of the constant corresponds to "
            "`max_completion_length`."
        },
    )

    delta: Optional[float] = field(
        default=None,
        metadata={
            "help": "If set to a float value (e.g., 2.0), enables the upper clipping bound in two-sided GRPO loss. If None (default), the standard GRPO clipping is used. Recommended to be > 1 + epsilon when enabled."
        },
    )

    scale_rewards: bool = field(
        default=True,
        metadata={
            "help": "Whether to scale the rewards by dividing them by their standard deviation. If `True` (default), "
            "the rewards are normalized by the standard deviation, ensuring they have unit variance. If `False`, no "
            "scaling is applied. The Dr. GRPO paper recommends not scaling the rewards, as scaling by the standard "
            "deviation introduces a question-level difficulty bias."
        },
    )

    shuffle_dataset: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to shuffle the training dataset."},
    )

    # ---------- multi-method fields ----------
    method: str = field(
        default="rlcr",
        metadata={"help": "Training method: rlcr | dcpo | credo | grpo."},
    )
    answer_format: str = field(
        default="math",
        metadata={"help": "Answer-channel carrier on the dcpo/credo segmented path: math = "
                          "\\boxed{} + math_verify (default); code = last "
                          "closed ```python block graded by code_judge. The rlcr path selects "
                          "its domain through the reward_funcs registry instead."},
    )
    cal_gamma: float = field(
        default=0.5,
        metadata={"help": "gamma in calibration target t = gamma*acc_i + (1-gamma)*group_acc."},
    )
    cal_weight: float = field(
        default=0.5,
        metadata={"help": "w_cal for dcpo confidence-segment penalty (dcpo only)."},
    )
    mse_alpha: float = field(
        default=1.0,
        metadata={"help": "alpha weight of the CREDO MSE calibration loss."},
    )
    mse_alpha_warmup_steps: int = field(
        default=0,
        metadata={"help": "linear warmup steps for mse_alpha and sw_kappa (0=off)."},
    )
    mse_gamma: Optional[float] = field(
        default=None,
        metadata={"help": "gamma for the MSE target; defaults to cal_gamma."},
    )
    w_fmt_ans: float = field(
        default=0.5,
        metadata={"help": "format weight in the CREDO answer channel r_ans = w_fmt_ans*F + F*acc."},
    )
    conf_token_high: str = field(
        default="<CONF_HIGH>",
        metadata={"help": "high-confidence special token literal (CREDO)."},
    )
    conf_token_low: str = field(
        default="<CONF_LOW>",
        metadata={"help": "low-confidence special token literal (CREDO)."},
    )
    missing_conf_value: float = field(
        default=-1.0,
        metadata={"help": "conf value used when text confidence missing/unparsable (dcpo only)."},
    )
    vllm_logprobs: int = field(
        default=20,
        metadata={"help": "SamplingParams.logprobs for the CREDO readout channel."},
    )
    mass_loss_lambda: float = field(
        default=0.01,
        metadata={"help": "weight of an optional -lambda*log(p_H+p_L) mass regularizer on the readout "
                          "token pair (CREDO only); 0.0 disables it."},
    )
    mass_loss_mode: str = field(
        default="topk_gate",
        metadata={"help": "'always' = apply the mass regularizer to every sample; 'topk_gate' = only "
                          "samples whose readout token pair missed the vLLM top-k at the readout "
                          "position. CREDO only, like mass_loss_lambda."},
    )
    conf_think: bool = field(
        default=True,
        metadata={"help": "True (default) = the CREDO format carries the <analysis> uncertainty segment; "
                          "False = the no-analysis ablation (boxed answer then the <confidence> readout "
                          "directly). Must stay consistent with sys_prompt_name (asserted in rl_runner)."},
    )
    pg_channels: str = field(
        default="both",
        metadata={"help": "policy-gradient channel decomposition (CREDO only): 'both' (default) = score "
                          "both segments; 'ans_only' = zero the confidence-segment advantage; 'conf_only' = "
                          "zero the answer-segment advantage; 'none' = zero both. Advantages are zeroed after "
                          "computation, so rewards/telemetry stay meaningful."},
    )
    sw_signal: str = field(
        default="surprise",
        metadata={"help": "discrepancy-weighting signal (CREDO only): 'surprise' (default) = |a-c|, the "
                          "discrepancy of Section 3.4; 'error' = a hard-mining control 1[a=0]; 'shuffle' = "
                          "|a-c| with confidences permuted within the group. The controls reroute weighting "
                          "through an equivalent trainer-side hook (post-group-norm, sign-preserving clip)."},
    )
    mse_detach_scale: float = field(
        default=1.0,
        metadata={"help": "soft-detach factor s in [0,1]: the trunk gradient of the auxiliary losses is "
                          "scaled by s (the lm_head readout rows keep full gradient); 1.0 = default, 0.0 = full detach."},
    )
    ans_reward_mode: str = field(
        default="fmt_gated_acc",
        metadata={"help": "CREDO answer-channel reward form: "
                  "'fmt_gated_acc' = w_fmt_ans*F + F*acc; "
                  "'fmt_plus_acc' = w_fmt_ans*F + acc (additive format + ungated accuracy, used in the paper); "
                  "'acc_only' = acc. CREDO only; rlcr/dcpo unaffected."},
    )
    sw_kappa: float = field(
        default=0.0,
        metadata={"help": "discrepancy-weighting strength kappa (Section 3.4): the group-centered "
                  "multiplicative weight w=clip(1+kappa*(|acc-c|-group_mean), wclip) applied to the CREDO "
                  "answer-channel advantage only. 0.0 = off. CREDO only: >0 with any other method raises."},
    )
    sw_kappa_neg: Optional[float] = field(
        default=None,
        metadata={"help": "SW asymmetric variant: kappa used where adv_ans<0 (extra push-down of "
                  "surprising failures). None = symmetric (= sw_kappa)."},
    )
    sw_weight_clip: list[float] = field(
        default_factory=lambda: [0.5, 2.0],
        metadata={"help": "SW weight clamp [w_min, w_max]; requires 0 < w_min <= 1 <= w_max. "
                  "w_min>0 guarantees the advantage sign can never flip."},
    )
    sw_warmup_steps: int = field(
        default=-1,
        metadata={"help": "SW linear warmup steps for kappa. -1 = inherit mse_alpha_warmup_steps; "
                  "0 = no warmup; >0 explicit."},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.method not in ("rlcr", "dcpo", "credo", "grpo"):
            raise ValueError(f"Invalid method: {self.method}")
        if self.mse_gamma is None:
            self.mse_gamma = self.cal_gamma
        if self.mass_loss_mode not in ("always", "topk_gate"):
            raise ValueError(f"Invalid mass_loss_mode: {self.mass_loss_mode}")
        # The readout-mass regularizer is CREDO-only; neutralize it for other methods so a
        # baseline's effective config never advertises a term its loss does not include.
        if self.method != "credo":
            if float(self.mass_loss_lambda) != 0.0:
                print(f"[config] method={self.method}: mass_loss_lambda {self.mass_loss_lambda} -> 0.0 "
                      f"(the readout-mass regularizer is CREDO-only and never enters this loss)")
            self.mass_loss_lambda = 0.0
            self.mass_loss_mode = "always"
        if self.ans_reward_mode not in ("fmt_gated_acc", "fmt_plus_acc", "acc_only"):
            raise ValueError(f"Invalid ans_reward_mode: {self.ans_reward_mode}")
        if self.pg_channels not in ("both", "ans_only", "conf_only", "none"):
            raise ValueError(f"Invalid pg_channels: {self.pg_channels}")
        if self.sw_signal not in ("surprise", "error", "shuffle"):
            raise ValueError(f"Invalid sw_signal: {self.sw_signal}")
        if self.sw_kappa < 0:
            raise ValueError(f"Invalid sw_kappa: {self.sw_kappa} (must be >= 0)")
        if self.sw_kappa > 0 and self.method != "credo":
            raise ValueError(
                f"sw_kappa > 0 requires method=credo (got method={self.method}); the discrepancy "
                "weight needs the logit-readout confidence, which other methods do not provide.")
        if self.sw_kappa_neg is not None and self.sw_kappa_neg < 0:
            raise ValueError(f"Invalid sw_kappa_neg: {self.sw_kappa_neg} (must be >= 0 or None)")
        if (len(self.sw_weight_clip) != 2
                or not (0.0 < float(self.sw_weight_clip[0]) <= 1.0 <= float(self.sw_weight_clip[1]))):
            raise ValueError(
                f"Invalid sw_weight_clip: {self.sw_weight_clip} (need [w_min, w_max] with 0 < w_min <= 1 <= w_max)")
        if self.sw_warmup_steps < -1:
            raise ValueError(f"Invalid sw_warmup_steps: {self.sw_warmup_steps} (must be >= -1)")
        if not (0.0 <= float(self.mse_detach_scale) <= 1.0):
            raise ValueError(f"Invalid mse_detach_scale: {self.mse_detach_scale}")

        print("Post init on the config")

        num_processes = self.world_size
        # The current default effective batch size
        if self.generation_batch_size is not None and self.steps_per_generation is not None:
            raise ValueError(
                "'generation_batch_size' and 'steps_per_generation' can not be both configured at the same time"
            )

        if self.steps_per_generation is None:
            self.steps_per_generation = self.gradient_accumulation_steps

        if self.generation_batch_size is None:
            self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation

        if self.generation_batch_size % self.per_device_train_batch_size * num_processes != 0:
            raise ValueError(
                f"generation_batch_size ({self.generation_batch_size}) must be divisible by the global batch size "
                f"({self.per_device_train_batch_size * num_processes})."
            )

        self.steps_per_generation = self.generation_batch_size // (self.per_device_train_batch_size * num_processes)

        possible_values = [
            n_gen for n_gen in range(2, self.generation_batch_size + 1) if (self.generation_batch_size) % n_gen == 0
        ]

        if self.num_generations not in possible_values:
            raise ValueError(
                f"The effective train batch size ({num_processes} x {self.per_device_train_batch_size} x "
                f"{self.steps_per_generation}) must be evenly divisible by the number of generations per "
                f"prompt ({self.num_generations}). Given the current effective train batch size, the valid values for "
                f"the number of generations are: {possible_values}."
            )
        
        if self.eval_strategy != "no":
            global_eval_batch_size = self.per_device_eval_batch_size * num_processes
            possible_values = [
                n_gen for n_gen in range(2, global_eval_batch_size + 1) if (global_eval_batch_size) % n_gen == 0
            ]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {self.per_device_eval_batch_size}) must be "
                    f"evenly divisible by the number of generations per prompt ({self.num_generations}). Given the "
                    "current global eval batch size, the valid values for the number of generations are: "
                    f"{possible_values}."
                )



@dataclass
class ModelConfig:
    """
    Configuration class for the models.

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        model_name_or_path (`str` or `None`, *optional*, defaults to `None`):
            Model checkpoint for weights initialization.
        model_revision (`str`, *optional*, defaults to `"main"`):
            Specific model version to use. It can be a branch name, a tag name, or a commit id.
        torch_dtype (`Literal["auto", "bfloat16", "float16", "float32"]` or `None`, *optional*, defaults to `None`):
            Override the default `torch.dtype` and load the model under this dtype. Possible values are

                - `"bfloat16"`: `torch.bfloat16`
                - `"float16"`: `torch.float16`
                - `"float32"`: `torch.float32`
                - `"auto"`: Automatically derive the dtype from the model's weights.

        trust_remote_code (`bool`, *optional*, defaults to `False`):
            Whether to allow for custom models defined on the Hub in their own modeling files. This option should only
            be set to `True` for repositories you trust and in which you have read the code, as it will execute code
            present on the Hub on your local machine.
        attn_implementation (`str` or `None`, *optional*, defaults to `None`):
            Which attention implementation to use. You can run `--attn_implementation=flash_attention_2`, in which case
            you must install this manually by running `pip install flash-attn --no-build-isolation`.
        use_peft (`bool`, *optional*, defaults to `False`):
            Whether to use PEFT for training.
        lora_r (`int`, *optional*, defaults to `16`):
            LoRA R value.
        lora_alpha (`int`, *optional*, defaults to `32`):
            LoRA alpha.
        lora_dropout (`float`, *optional*, defaults to `0.05`):
            LoRA dropout.
        lora_target_modules (`Union[str, list[str]]` or `None`, *optional*, defaults to `None`):
            LoRA target modules.
        lora_modules_to_save (`list[str]` or `None`, *optional*, defaults to `None`):
            Model layers to unfreeze & train.
        lora_task_type (`str`, *optional*, defaults to `"CAUSAL_LM"`):
            Task type to pass for LoRA (use `"SEQ_CLS"` for reward modeling).
        use_rslora (`bool`, *optional*, defaults to `False`):
            Whether to use Rank-Stabilized LoRA, which sets the adapter scaling factor to `lora_alpha/√r`, instead of
            the original default value of `lora_alpha/r`.
        load_in_8bit (`bool`, *optional*, defaults to `False`):
            Whether to use 8 bit precision for the base model. Works only with LoRA.
        load_in_4bit (`bool`, *optional*, defaults to `False`):
            Whether to use 4 bit precision for the base model. Works only with LoRA.
        bnb_4bit_quant_type (`str`, *optional*, defaults to `"nf4"`):
            Quantization type (`"fp4"` or `"nf4"`).
        use_bnb_nested_quant (`bool`, *optional*, defaults to `False`):
            Whether to use nested quantization.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Model checkpoint for weights initialization."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "Specific model version to use. It can be a branch name, a tag name, or a commit id."},
    )
    torch_dtype: Optional[str] = field(
        default="auto",
        metadata={
            "help": "Override the default `torch.dtype` and load the model under this dtype.",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={
            "help": "Whether to allow for custom models defined on the Hub in their own modeling files. This option "
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        },
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which attention implementation to use. You can run `--attn_implementation=flash_attention_2`, in "
            "which case you must install this manually by running `pip install flash-attn --no-build-isolation`."
        },
    )
    use_peft: bool = field(
        default=False,
        metadata={"help": "Whether to use PEFT for training."},
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA R value."},
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha."},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout."},
    )
    lora_target_modules: Optional[list[str]] = field(
        default=None,
        metadata={"help": "LoRA target modules."},
    )
    lora_modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={"help": "Model layers to unfreeze & train."},
    )
    lora_task_type: str = field(
        default="CAUSAL_LM",
        metadata={"help": "Task type to pass for LoRA (use 'SEQ_CLS' for reward modeling)."},
    )
    use_rslora: bool = field(
        default=False,
        metadata={
            "help": "Whether to use Rank-Stabilized LoRA, which sets the adapter scaling factor to `lora_alpha/√r`, "
            "instead of the original default value of `lora_alpha/r`."
        },
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Whether to use 8 bit precision for the base model. Works only with LoRA."},
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Whether to use 4 bit precision for the base model. Works only with LoRA."},
    )
    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization type.", "choices": ["fp4", "nf4"]},
    )
    use_bnb_nested_quant: bool = field(
        default=False,
        metadata={"help": "Whether to use nested quantization."},
    )

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("You can't use 8 bit and 4 bit precision at the same time")

        if hasattr(self.lora_target_modules, "__len__") and len(self.lora_target_modules) == 1:
            self.lora_target_modules = self.lora_target_modules[0]





