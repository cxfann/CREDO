# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math
import re
import textwrap
import warnings
from collections import defaultdict, deque
from collections.abc import Sized
from contextlib import nullcontext
from typing import Any, Callable, Optional, Union

import datasets
import time 
import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available, is_peft_available

from trl import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.import_utils import is_vllm_available
from trl.models import create_reference_model, unwrap_model_for_generation
from trl.trainer.utils import prepare_deepspeed
from trl import SyncRefModelCallback
from trl import GRPOConfig
from trl.trainer.utils import (
    generate_model_card,
    get_comet_experiment_url,
    pad,
    selective_log_softmax,
)

from trainer_utils import nanstd, RepeatSampler, nanmin, nanmax, shuffle_tensor_dict,split_tensor_dict, disable_dropout_in_model,profiling_decorator,profiling_context
import gc

if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

if is_wandb_available():
    import wandb

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


def _answer_hooks(answer_format):
    """Answer-channel carrier for the segmented path; see :mod:`answer_carrier`.

    Delegated rather than duplicated: the trainer and the eval pipeline must not hold two
    copies of the domain mapping, which would be free to drift apart.
    """
    from answer_carrier import answer_hooks
    return answer_hooks(answer_format)



class CustomTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    def reward_func(completions, **kwargs):
        # Dummy reward function that rewards completions with more unique letters.
        return [float(len(set(completion))) for completion in completions]

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_func,
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None)
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )


        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
        if processing_class.pad_token is None:
            processing_class.pad_token = processing_class.eos_token


        self.reward_func_names = [] 
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
            if isinstance(reward_funcs[i], nn.Module):  # Use Module over PretrainedModel for compat w/ compiled models
                self.reward_func_names.append(reward_funcs[i].config._name_or_path.split("/")[-1])
            else:
                try:
                    self.reward_func_names.append(reward_funcs[i].__name__)
                except:
                    self.reward_func_names.append(reward_funcs[i].func.__name__)
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(len(args.reward_weights))}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32) / len(reward_funcs)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.temperature = args.temperature
        self.vllm_mode = args.vllm_mode
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size
        self.loss_type = args.loss_type
        self.scale_rewards = args.scale_rewards
        self.mask_truncated_completions = args.mask_truncated_completions 
        self.vllm_sleeping = False 

        self.log_completions = args.log_completions

        #Datasets 
        self.shuffle_dataset = args.shuffle_dataset

        #Multi-step
        self.num_iterations = args.num_iterations
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        self._step = 0 
        self._buffered_inputs = None 

        # ---------- multi-method setup ----------
        self.method = getattr(args, "method", "rlcr")
        # The math answer channel uses boxed answers and math_verify.
        self.answer_format = getattr(args, "answer_format", "math")
        self.conf_high_id = None
        self.conf_low_id = None
        if self.method == "credo":
            hid = processing_class.convert_tokens_to_ids(getattr(args, "conf_token_high", "<CONF_HIGH>"))
            lid = processing_class.convert_tokens_to_ids(getattr(args, "conf_token_low", "<CONF_LOW>"))
            if hid is None or lid is None or hid == processing_class.unk_token_id:
                raise ValueError(
                    "method=credo requires <CONF_HIGH>/<CONF_LOW> in the tokenizer; "
                    "use the Qwen3-8B-conf checkpoint built by scripts/make_conf_checkpoint.py."
                )
            self.conf_high_id, self.conf_low_id = int(hid), int(lid)
        if self.method not in ("rlcr", "grpo"):
            # segmented scoring computes its own channel columns; names used for logging only.
            # Scalar methods must keep the registry-derived names: the scalar loop zips
            # reward_funcs against these names, and a shorter list silently truncates the zip.
            self.reward_func_names = ["answer_acc", "conf_value", "r_ans", "r_conf"] + (
                ["F"] if self.method == "credo" else []
            )

        model.warnings_issued["estimate_tokens"] = True

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
            self.ref_model = self.accelerator.prepare(self.ref_model)
            self.ref_model.eval()
        else:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
            #send to accelerator device
            self.ref_model = self.ref_model.to(self.accelerator.device)
            self.ref_model.eval()

        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        self.completion_logging_steps = args.completion_logging_steps 
        self.wandb_counter = 0
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0

        self._textual_logs = {
            "step": deque(maxlen=50000),
            "prompt": deque(maxlen=50000),
            "completion": deque(maxlen=50000),
            "rewards": defaultdict(lambda: deque(maxlen=50000)),
        }

        set_seed(args.seed, device_specific=True)

        if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

        if self.vllm_mode == "colocate":
            self.llm = LLM(
                    model=model.name_or_path,
                    tensor_parallel_size=args.vllm_tensor_parallel_size,
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                    max_num_seqs=self.args.per_device_train_batch_size
                    * self.args.gradient_accumulation_steps,
                    max_model_len=self.max_prompt_length + self.max_completion_length,
                    distributed_executor_backend="external_launcher",
                    # Feed identical seed for tp groups to ensure sampling results are the same across workers
                    seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                    enable_sleep_mode=True
                )
            
            self._last_loaded_step = -1  # tag to avoid useless loading during grad accumulation
            self.model_name_or_path = model.name_or_path
            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()

        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                if self.is_deepspeed_enabled:
                    self.reward_funcs[i] = prepare_deepspeed(reward_func, self.accelerator)
                else:
                    # set device placement to True to make `prepare_model` move `reward_func` to device when using fsdp
                    self.reward_funcs[i] = self.accelerator.prepare_model(
                        reward_func, evaluation_mode=True, device_placement=True
                    )

     
    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def training_step(self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None) -> torch.Tensor:
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        del inputs
        if self.args.n_gpu > 1:
            loss = loss.mean()

        if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
            loss = loss / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss)
        return loss.detach()

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size * self.args.steps_per_generation,  # < this is the change
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self) -> Sampler:
        return RepeatSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=1,
            seed=self.args.seed,
        )
    
    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )
        if use_reentrant:
            model.enable_input_require_grads()
        return model
    
    @profiling_decorator
    def _get_last_hidden_state(self, unwrapped_model, input_ids, attention_mask, logits_to_keep=None):
        last_hidden_state = unwrapped_model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        if logits_to_keep is not None:
            last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
        return last_hidden_state

    @profiling_decorator
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None, mode = "train", readout_pos=None) -> torch.Tensor:
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        batch_size = self.args.per_device_train_batch_size ##Set small batch size to reduce memory peak
        if mode =="eval":
            batch_size = 1
        all_logps = []
        all_delta = []
        all_logmass = []
        #soft-detach: s<1 rebuilds readout logits from grad-scaled final-norm hidden
        # (full grad to lm_head rows, trunk grad ×s). s>=1 keeps the original path untouched.
        _s = float(getattr(self.args, "mse_detach_scale", 1.0))
        _soft = readout_pos is not None and _s < 1.0
        _norm_hook_handle = None
        _cap = {}
        if _soft:
            _unwrapped = self.accelerator.unwrap_model(model)
            _norm_hook_handle = _unwrapped.model.norm.register_forward_hook(
                lambda mod, inp, out: _cap.__setitem__("h", out))
            _lm_w = _unwrapped.lm_head.weight  # (V, H); tie_word_embeddings=False
        try:
            for i in range(0, input_ids.size(0), batch_size):
                input_ids_batch = input_ids[i : i + batch_size]
                attention_mask_batch = attention_mask[i : i + batch_size]

                logits = model(
                    input_ids=input_ids_batch, attention_mask=attention_mask_batch
                ).logits
                logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
                logits = logits[:, -logits_to_keep:]
                input_ids_batch = input_ids_batch[:, -logits.size(1):]
                if readout_pos is not None:
                    # raw (pre-temperature) logits at the readout position
                    rp = readout_pos[i : i + batch_size].clamp(min=0, max=logits.size(1) - 1)
                    rows = torch.arange(rp.size(0), device=logits.device)
                    if _soft:
                        hidden = _cap["h"]  # (b, L, H) final-norm output of this chunk
                        # sliced logits index j corresponds to full position j + (L - 1 - K)
                        off = hidden.size(1) - 1 - logits.size(1)
                        h_p = hidden[rows, rp + off]  # (b, H)
                        h_soft = h_p.detach() + _s * (h_p - h_p.detach())
                        sel = torch.nn.functional.linear(h_soft, _lm_w)  # (b, V) raw logits, trunk grad ×s
                    else:
                        sel = logits[rows, rp]  # (b, V) raw logits at readout
                    lh = sel[:, self.conf_high_id].float()
                    ll = sel[:, self.conf_low_id].float()
                    all_delta.append(lh - ll)
                    # log(p_H + p_L) = logaddexp(lh, ll) - logsumexp(all); only needed for optional mass loss
                    if float(getattr(self.args, "mass_loss_lambda", 0.0)) > 0:
                        lse = torch.logsumexp(sel.float(), dim=-1)
                        all_logmass.append(torch.logaddexp(lh, ll) - lse)
                # Divide logits by sampling temperature.
                # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
                logits = logits / self.temperature
                logps = selective_log_softmax(logits, input_ids_batch)  # compute logprobs for the input tokens
                all_logps.append(logps)
        finally:
            if _norm_hook_handle is not None:
                _norm_hook_handle.remove()
        if readout_pos is not None:
            delta = torch.cat(all_delta, dim=0)
            logmass = torch.cat(all_logmass, dim=0) if all_logmass else None
            return torch.cat(all_logps, dim=0), delta, logmass
        return torch.cat(all_logps, dim=0)
        
 
    @profiling_decorator
    def _move_model_to_vllm(self):
        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        gather_if_zero3 = nullcontext
        for name, param in self.model.named_parameters():
            with gather_if_zero3([param]):
                if self.vllm_mode == "server" and self.accelerator.is_main_process:
                    self.vllm_client.update_named_param(name, param.data)
                elif self.vllm_mode == "colocate":
                    llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                    llm_model.load_weights([(name, param.data)])

        if self.vllm_mode == "colocate":
            self.llm.reset_prefix_cache()

    @profiling_decorator
    def _prepare_inputs(
        self, generation_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × steps per generation)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every steps_per_generation * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                generation_batch = self._generate_and_score_completions(generation_batch)
                generation_batch = shuffle_tensor_dict(generation_batch)
                self._buffered_inputs = split_tensor_dict(generation_batch, self.args.steps_per_generation)
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
            self._step += 1
        else:
            # In evaluation, there is neither batch grouping for generation, nor multiple iterations, hence
            # local generation batch == local eval batch
            inputs = self._generate_and_score_completions(generation_batch)
        return inputs

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]], eval = False
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]
        if getattr(self.args, "enable_thinking", None) is None:
            prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        else:
            prompts_text = []
            for example in inputs:
                prompt_messages = example["prompt"]
                prompts_text.append(
                    self.processing_class.apply_chat_template(
                        prompt_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=self.args.enable_thinking,
                    )
                )
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # vLLM must condition on the exact prompt the training forward pass uses. prompt_ids is the
        # left-truncated, left-padded tensor concatenated into prompt_completion_ids below; handing
        # vLLM the raw prompts_text instead let generation see tokens the loss never conditions on --
        # it crashed when the untruncated prompt exceeded max_model_len and, more quietly, corrupted
        # the importance ratio for every prompt longer than max_prompt_length. prompt_mask marks the
        # real tokens after left padding, so strip the pad per row and feed vLLM those ids.
        vllm_prompts = [{"prompt_token_ids": row[m.bool()].tolist()} for row, m in zip(prompt_ids, prompt_mask)]

        torch.cuda.empty_cache()

        # First, have main process load weights if needed
        if self.state.global_step != self._last_loaded_step:
            if self.state.global_step>=-1:
                self.llm.wake_up()
                self.vllm_sleeping = False 
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        _sp_kwargs = dict(
                    n=1,  # vLLM on each GPU generates only 1 in colocate mode
                    temperature=self.temperature,
                    max_tokens=self.max_completion_length)
        if self.method == "credo":
            _sp_kwargs["logprobs"] = int(getattr(self.args, "vllm_logprobs", 20))
        sampling_params = SamplingParams(**_sp_kwargs)
        
        with profiling_context(self, "vLLM.generate"):
            all_outputs = self.llm.generate(vllm_prompts, sampling_params=sampling_params, use_tqdm=False)
           # put to sleep
            if mode == "train":
                self.llm.sleep(level=1)
                self.vllm_sleeping = True 
                self.accelerator.wait_for_everyone()

        completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]
        # keep vLLM per-token logprob dicts for the CREDO readout channel
        if self.method == "credo":
            gen_logprobs = [output.logprobs for outputs in all_outputs for output in outputs.outputs]
        else:
            gen_logprobs = None
        # Pad the completions, and concatenate them with the prompts
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
       
        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

         # Convert tensor to a list of lists of token IDs. This will be passed to the reward function, avoiding the need
        # to re-tokenize completions if the reward is computed from tokens.
        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m] for row, mask_row in zip(completion_ids, completion_mask)
        ]

        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

         # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad():
            # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
            # old_per_token_logps == per_token_logps, so we can skip it's computation here, and use
            # per_token_logps.detach() instead.
            if self.num_iterations > 1 or self.args.steps_per_generation > self.args.gradient_accumulation_steps:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size, mode = mode
                )
            else:
                old_per_token_logps = None

         # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # ---------------- scoring branch: rlcr/grpo scalar path vs dcpo/credo segmented path ----------------
        T_local = completion_ids.size(1)
        ours_extras = {}

        if self.method in ("rlcr", "grpo"):
            rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
            for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
            ):
                with profiling_context(self, reward_func_name):
                    if isinstance(
                        reward_func, nn.Module
                    ):  # Module instead of PretrainedModel for compat with compiled models
                        if is_conversational(inputs[0]):
                            messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                            texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                        else:
                            texts = [p + c for p, c in zip(prompts, completions)]
                        reward_inputs = reward_processing_class(
                            text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                        )
                        reward_inputs = super()._prepare_inputs(reward_inputs)
                        with torch.inference_mode():
                            rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                    else:
                        # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the number
                        # of generations
                        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
                        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                        output_reward_func = reward_func(
                            prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                        )
                        # Convert None values to NaN
                        output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                        rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

            # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
            # completions may be distributed across processes
            rewards_per_func = gather(rewards_per_func)

            # Apply weights to each reward function's output and sum
            rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

            # Compute grouped-wise rewards
            mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
            std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

            # Normalize the rewards to compute the advantages
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            advantages = rewards - mean_grouped_rewards

            if self.scale_rewards:
                advantages = advantages / (std_grouped_rewards + 1e-4)

            # Slice to keep only the local part of the data
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            advantages = advantages[process_slice]
            # per-token advantage (B,T); scalar broadcast, numerically identical to the scalar form
            advantages_bt = advantages.unsqueeze(1).repeat(1, T_local)
        else:
            # ---- segmented scoring for dcpo/credo ----
            from reward_fns import extract_boxed_answer, verify_gold_pred
            from segment_utils import find_segments, ANALYSIS_TAG
            # With answer_format="math", the hooks use the two functions imported above.
            _ans_extract, _ans_verify, _ = _answer_hooks(self.answer_format)
            _code_mode = self.answer_format == "code"
            _pending_code = []  # (row, program) for the code path, graded in one batch below
            B_local = len(prompts)
            golds = [example["answer"] for example in inputs]
            acc_l = torch.zeros(B_local, dtype=torch.float32)
            conf_l = torch.zeros(B_local, dtype=torch.float32)
            F_l = torch.zeros(B_local, dtype=torch.float32)
            boundary_l = [-1] * B_local
            readout_l = [-1] * B_local
            conf_missing_local = 0.0
            pair_cov_local = 0.0
            pair_cov_row = torch.zeros(B_local, dtype=torch.float32)  # per-sample top-20 coverage for conditional mass gate
            readable_local = 0.0
            emitted_local = 0.0
            for i in range(B_local):
                ids_row = list(completion_ids_list[i])
                text_sp, b_idx, r_idx = find_segments(
                    self.processing_class, ids_row, self.conf_high_id, self.conf_low_id
                )
                boundary_l[i] = -1 if b_idx is None else int(b_idx)
                readout_l[i] = -1 if r_idx is None else int(r_idx)
                aidx = text_sp.find(ANALYSIS_TAG)
                pre = text_sp[:aidx] if aidx >= 0 else text_sp
                if _code_mode:
                    # queue the answer segment, not the extracted program: compute_scores
                    # forwards kwargs to every item, so a per-item `code=` cannot be passed
                    # through it. The judge re-runs the same extractor on `pre`.
                    if _ans_extract(pre) is not None:
                        _pending_code.append((i, pre))
                else:
                    boxed = _ans_extract(pre)
                    acc_l[i] = float(_ans_verify(golds[i], "\\boxed{" + boxed + "}")) if boxed is not None else 0.0
                if self.method == "dcpo":
                    # DCPO-faithful: parsed value used as-is (no clamp); missing/unparsable -> missing_conf_value
                    cm = re.findall(r"<confidence>(.*?)</confidence>", text_sp, re.DOTALL)
                    conf_v = None
                    if cm:
                        try:
                            conf_v = float(cm[-1])
                        except Exception:
                            conf_v = None
                    conf_l[i] = float(getattr(self.args, "missing_conf_value", -1.0)) if conf_v is None else conf_v
                else:  # credo
                    F_l[i] = self._bac_credo_format_ok(ids_row, text_sp)
                    if any(t == self.conf_high_id or t == self.conf_low_id for t in ids_row):
                        emitted_local += 1.0
                    conf_v = 0.5  # placeholder, only consumed when F=1 (r_conf gated by F)
                    if r_idx is not None and gen_logprobs is not None and gen_logprobs[i] is not None and r_idx < len(gen_logprobs[i]):
                        readable_local += 1.0
                        lp_map = gen_logprobs[i][r_idx]
                        lp_h = lp_map.get(self.conf_high_id)
                        lp_l = lp_map.get(self.conf_low_id)
                        if lp_h is not None and lp_l is not None:
                            pair_cov_local += 1.0
                            pair_cov_row[i] = 1.0
                            delta = self.temperature * (lp_h.logprob - lp_l.logprob)
                            conf_v = 1.0 / (1.0 + math.exp(-delta))
                        elif lp_h is not None or lp_l is not None:
                            # one side missing from top-k: bound the other by the min returned logprob
                            lp_min = min(v.logprob for v in lp_map.values())
                            lh = lp_h.logprob if lp_h is not None else lp_min
                            ll = lp_l.logprob if lp_l is not None else lp_min
                            delta = self.temperature * (lh - ll)
                            conf_v = 1.0 / (1.0 + math.exp(-delta))
                            if F_l[i] > 0:
                                conf_missing_local += 1.0
                        else:
                            if F_l[i] > 0:
                                conf_missing_local += 1.0
                    conf_l[i] = conf_v

            if _code_mode and _pending_code:
                # one batched pass: each item is a sandboxed subprocess, so grading inside the
                # loop above would serialise ~B_local launches per step
                from code_judge import compute_scores
                _scored = compute_scores([(seg, golds[r]) for r, seg in _pending_code])
                for (_r, _seg), (_score, _diag) in zip(_pending_code, _scored):
                    acc_l[_r] = float(_score)
                # judge health is a first-class readout: if the timeout share climbs, the
                # wall-clock cap rather than the model is deciding the answer reward
                _statuses = [_d.get("status", "unknown") for _s, _d in _scored]
                _n_judged = float(len(_statuses))
                _n_error = sum(1 for s in _statuses
                               if s in ("compile_error", "runtime_error", "harness_error", "crashed"))
                self._metrics[mode]["judge/graded"].append(_n_judged)
                self._metrics[mode]["judge/timeout_rate"].append(
                    _statuses.count("timeout") / _n_judged)
                self._metrics[mode]["judge/error_rate"].append(_n_error / _n_judged)
                self._metrics[mode]["judge/pass_rate"].append(
                    _statuses.count("passed") / _n_judged)

            stats_local = torch.stack([acc_l, conf_l, F_l, pair_cov_row], dim=1).to(device)
            stats = gather(stats_local)
            acc_g, conf_g, F_g, pcov_g = stats[:, 0], stats[:, 1], stats[:, 2], stats[:, 3]
            G = self.num_generations
            from segment_utils import compute_channel_advantages
            # SW effective kappa: linear warmup; -1 inherits the mse warmup horizon.
            _sw_kappa = float(getattr(self.args, "sw_kappa", 0.0))
            if _sw_kappa > 0.0:
                _sww = int(getattr(self.args, "sw_warmup_steps", -1))
                if _sww < 0:
                    _sww = int(getattr(self.args, "mse_alpha_warmup_steps", 0) or 0)
                _swu = min(1.0, float(self.state.global_step + 1) / float(_sww)) if _sww > 0 else 1.0
                _sw_kappa = _sw_kappa * _swu
            # control signals reroute the weighting through _sw_external_weight with the
            # SAME effective (warmed-up) kappa; the in-library SW path is disabled bit-identically.
            _sw_sig = str(getattr(self.args, "sw_signal", "surprise"))
            ch = compute_channel_advantages(
                acc_g, conf_g, F_g, G, self.method,
                cal_gamma=float(getattr(self.args, "cal_gamma", 0.5)),
                cal_weight=float(getattr(self.args, "cal_weight", 0.5)),
                w_fmt_ans=float(getattr(self.args, "w_fmt_ans", 0.5)),
                mse_gamma=getattr(self.args, "mse_gamma", None),
                ans_reward_mode=str(getattr(self.args, "ans_reward_mode", "fmt_gated_acc")),
                sw_kappa=(0.0 if _sw_sig != "surprise" else _sw_kappa),
                sw_kappa_neg=getattr(self.args, "sw_kappa_neg", None),
                sw_valid_g=pcov_g,
                sw_weight_clip=tuple(getattr(self.args, "sw_weight_clip", (0.5, 2.0))),
            )
            group_acc = ch["group_acc"]
            target_g = ch["target_g"]
            r_ans_g = ch["r_ans_g"]
            r_conf_g = ch["r_conf_g"]
            adv_ans_g = ch["adv_ans_g"]
            adv_conf_g = ch["adv_conf_g"]
            mse_target_g = ch["mse_target_g"]
            if _sw_sig != "surprise":
                adv_ans_g, ch["sw_w_g"] = self._sw_external_weight(acc_g, conf_g, adv_ans_g, pcov_g, G, _sw_kappa)
            # PG-channel decomposition — advantages zeroed AFTER computation, so rewards,
            # targets and every telemetry stream stay meaningful.
            _pgc = str(getattr(self.args, "pg_channels", "both"))
            if _pgc in ("conf_only", "none"):
                adv_ans_g = adv_ans_g * 0.0
            if _pgc in ("ans_only", "none"):
                adv_conf_g = adv_conf_g * 0.0

            r_total_g = r_ans_g + r_conf_g
            mean_grouped_rewards = r_total_g.view(-1, G).mean(dim=1).repeat_interleave(G, dim=0)
            std_grouped_rewards = r_total_g.view(-1, G).std(dim=1).repeat_interleave(G, dim=0)
            cols = [acc_g, conf_g, r_ans_g, r_conf_g] + ([F_g] if self.method == "credo" else [])
            rewards_per_func = torch.stack(cols, dim=1)

            process_slice = slice(
                self.accelerator.process_index * B_local,
                (self.accelerator.process_index + 1) * B_local,
            )
            adv_ans = adv_ans_g[process_slice]
            adv_conf = adv_conf_g[process_slice]
            advantages_bt = torch.zeros(B_local, T_local, dtype=torch.float32, device=device)
            for i in range(B_local):
                b = boundary_l[i]
                if b < 0:
                    # boundary missing: whole seq = answer segment, conf channel lands on last unmasked token (C.2)
                    L = int(completion_mask[i].sum().item())
                    b = max(L - 1, 0)
                advantages_bt[i, :b] = adv_ans[i]
                advantages_bt[i, b:] = adv_conf[i]

            if self.method == "credo":
                pg_mask = completion_mask.clone()
                readout_pos = torch.zeros(B_local, dtype=torch.long, device=device)
                mse_valid = torch.zeros(B_local, dtype=torch.float32, device=device)
                for i in range(B_local):
                    r = readout_l[i]
                    if 0 <= r < T_local:
                        readout_pos[i] = r
                        if completion_mask[i, r] > 0:
                            mse_valid[i] = 1.0
                        pg_mask[i, r] = 0  # readout token excluded from PG (C.2)
                ours_extras = {
                    "pg_mask": pg_mask,
                    "readout_pos": readout_pos,
                    "mse_target": mse_target_g[process_slice],
                    "mse_valid": mse_valid,
                    "conf_gen": conf_l.to(device),
                    "mass_gate": (1.0 - pair_cov_row).to(device),
                }

            # segmented-path metrics
            bfound_local = torch.tensor([1.0 if b >= 0 else 0.0 for b in boundary_l], device=device)
            self._metrics[mode]["seg/boundary_found_rate"].append(
                self.accelerator.gather_for_metrics(bfound_local).mean().item())
            self._metrics[mode]["conf/mean"].append(conf_g.mean().item())
            self._metrics[mode]["cal/target_mean"].append(target_g.mean().item())
            if self.method == "dcpo":
                outlier = ((conf_g < 0) | (conf_g > 1)).float()
                self._metrics[mode]["dcpo/conf_outlier_rate"].append(outlier.mean().item())
                self._metrics[mode]["dcpo/cal_penalty_max"].append((-r_conf_g).max().item())
            else:
                miss = self.accelerator.gather_for_metrics(
                    torch.tensor([conf_missing_local], device=device)).sum()
                self._metrics[mode]["conf/missing_rate"].append(
                    (miss / F_g.sum().clamp(min=1.0)).item())
                pcv = self.accelerator.gather_for_metrics(
                    torch.tensor([pair_cov_local, readable_local, emitted_local], device=device).unsqueeze(0)).sum(dim=0)
                self._metrics[mode]["conf/pair_coverage"].append(
                    (pcv[0] / pcv[1].clamp(min=1.0)).item())
                self._metrics[mode]["conf/emitted_rate"].append(
                    (pcv[2] / max(float(B_local) * self.accelerator.num_processes, 1.0)).item())
                self._metrics[mode]["fmt/F_rate"].append(F_g.mean().item())
                # gate-ablation diagnostic: correct-but-malformed mass = P(acc=1 & F=0),
                # the exact set whose answer-channel advantage differs across ans_reward_mode.
                # acc_g/F_g are already gathered globals here.
                self._metrics[mode]["seg/acc_and_F0_rate"].append(
                    (acc_g * (1.0 - F_g)).mean().item())
                self._metrics[mode]["mse/valid_rate"].append(
                    self.accelerator.gather_for_metrics(ours_extras["mse_valid"]).mean().item())
                _sw_w = ch.get("sw_w_g")
                if _sw_w is not None:
                    # SW diagnostics: all on gathered globals; keys absent when SW off.
                    self._metrics[mode]["sw/kappa_eff"].append(_sw_kappa)
                    self._metrics[mode]["sw/w_mean"].append(_sw_w.mean().item())
                    self._metrics[mode]["sw/w_p90"].append(torch.quantile(_sw_w.float(), 0.9).item())
                    # w_p90 alone under-reads the binding tail (κ=0.5 runs showed p90≈1.03 while
                    # the active mass sits beyond p90); p99+max make clip saturation observable.
                    self._metrics[mode]["sw/w_p99"].append(torch.quantile(_sw_w.float(), 0.99).item())
                    self._metrics[mode]["sw/w_max"].append(_sw_w.float().max().item())
                    self._metrics[mode]["sw/valid_rate"].append(pcov_g.mean().item())
                    _m = pcov_g > 0
                    _s_dbg = (acc_g - conf_g).abs()
                    self._metrics[mode]["sw/surprise_mean"].append(
                        _s_dbg[_m].mean().item() if _m.any() else float("nan"))
                    self._metrics[mode]["sw/frac_confwrong"].append(
                        ((conf_g > 0.7) & (acc_g < 0.5) & _m).float().mean().item())
                    if int(_m.sum().item()) > 1:
                        _cc = torch.corrcoef(torch.stack([_s_dbg[_m], acc_g[_m]]))[0, 1].item()
                    else:
                        _cc = float("nan")
                    self._metrics[mode]["sw/corr_s_acc"].append(_cc)

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # log completion lengths, mean, min, max
        agg_completion_mask = self.accelerator.gather_for_metrics(completion_mask.sum(1))
        self._metrics[mode]["completions/mean_length"].append(agg_completion_mask.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_mask.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_mask.float().max().item())

        agg_terminated_with_eos = self.accelerator.gather_for_metrics(is_eos.any(dim=1))
        term_completion_mask = agg_completion_mask[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_mask) / len(agg_completion_mask)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_mask) == 0:
            # edge case where no completed sequences are found
            term_completion_mask = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_mask.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_mask.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_mask.float().max().item())

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        # Log prompt and completion texts
        num_completions_to_log = self.args.num_completions_to_log 
        self._textual_logs["step"].extend([str(self.state.global_step)] * num_completions_to_log)
        self._textual_logs["prompt"].extend(gather_object(prompts_text)[0:num_completions_to_log])
        if self.method == "credo":
            # Keep special tokens visible in the logged table.
            _log_texts = self.processing_class.batch_decode(completion_ids, skip_special_tokens=False)
            _log_texts = [t.replace(self.processing_class.pad_token or "", "") for t in _log_texts]
            self._textual_logs["completion"].extend(gather_object(_log_texts)[0:num_completions_to_log])
        else:
            self._textual_logs["completion"].extend(gather_object(completions_text)[0:num_completions_to_log])
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist()[0:num_completions_to_log])

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages_bt,
            "old_per_token_logps": old_per_token_logps,
            **ours_extras,
        }

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        return self._compute_loss(model, inputs)
    
    def _sw_external_weight(self, acc_g, conf_g, adv_ans_g, valid_g, G, kappa_eff):
        """SW mechanism controls (sw_signal != 'surprise'); called with the in-library SW
        path disabled (sw_kappa=0 -> bit-identical unweighted advantages). Machinery mirrors
        segment_utils' SW block line by line: signal masked to valid rows, group-mean centred
        over the valid count, w = clip(1 + kappa*(sig - sbar), w_min, w_max) with w_min > 0
        (sign never flips), invalid rows w=1, applied AFTER group normalisation. Only the
        SIGNAL differs: 'error' = 1[a=0] (hard-mining control); 'shuffle' = |a - c_pi| with
        confidences permuted within each group by a (seed, global_step)-seeded generator,
        identical on every rank."""
        import torch
        sig_kind = str(getattr(self.args, "sw_signal", "surprise"))
        assert sig_kind in ("error", "shuffle"), sig_kind
        assert getattr(self.args, "sw_kappa_neg", None) is None, \
            "controls replicate the symmetric SW only (sw_kappa_neg must be None)"
        assert acc_g.numel() % G == 0
        v = (valid_g if valid_g is not None else torch.ones_like(acc_g)).to(acc_g.dtype)
        if sig_kind == "error":
            sig = (1.0 - acc_g) * v
        else:
            gen = torch.Generator()
            gen.manual_seed(int(getattr(self.args, "seed", 0)) * 1000003 + int(self.state.global_step))
            conf_pi = conf_g.clone()
            for gi in range(acc_g.numel() // G):
                idx = (torch.randperm(G, generator=gen) + gi * G).to(conf_g.device)
                conf_pi[gi * G:(gi + 1) * G] = conf_g[idx]
            sig = (acc_g - conf_pi).abs() * v
        n_valid = v.view(-1, G).sum(dim=1).clamp(min=1.0)
        s_bar = (sig.view(-1, G).sum(dim=1) / n_valid).repeat_interleave(G, dim=0)
        lo, hi = tuple(getattr(self.args, "sw_weight_clip", (0.5, 2.0)))
        w = (1.0 + float(kappa_eff) * (sig - s_bar)).clamp(float(lo), float(hi))
        w = torch.where(v > 0, w, torch.ones_like(w))
        return adv_ans_g * w, w

    def _bac_credo_format_ok(self, ids_row, text_sp):
        """F' indicator for method=credo:
        readability-gated — does NOT require sampling a special token (sampled identity
        AND presence are both reward-irrelevant; readout is logits-based). Requires:
        nesting-safe boxed before <analysis> + intact analysis block + tail
        <confidence>{special-token literal | <=30 non-tag chars}</confidence>.
        The readout position is masked from the policy-gradient objective."""
        from reward_fns import extract_boxed_answer
        text = text_sp
        for tk in (self.processing_class.eos_token, self.processing_class.pad_token):
            if tk:
                while text.rstrip().endswith(tk):
                    text = text.rstrip()[: -len(tk)]
        high = getattr(self.args, "conf_token_high", "<CONF_HIGH>")
        low = getattr(self.args, "conf_token_low", "<CONF_LOW>")
        # The math hooks use extract_boxed_answer and the boxed-answer pattern.
        _extract, _, _carrier = _answer_hooks(getattr(self, "answer_format", "math"))
        if not bool(getattr(self.args, "conf_think", True)):
            # noct ablation: no <analysis> anywhere; the confidence tail follows the
            # answer directly. Mirrors the CT pattern with the analysis block removed.
            pat = (
                r"\A(?:(?!<analysis>|<confidence>).)*?" + _carrier + r"(?:(?!<analysis>|<confidence>).)*"
                r"<confidence>\s*(?:" + re.escape(high) + "|" + re.escape(low) + r"|[^<]{0,30})\s*</confidence>\s*\Z"
            )
            if re.match(pat, text, re.DOTALL) is None:
                return 0.0
            cidx = text.find("<confidence>")
            return 0.0 if _extract(text[:cidx]) is None else 1.0
        pat = (
            r"\A(?:(?!<analysis>).)*?" + _carrier + r"(?:(?!<analysis>).)*<analysis>(?:(?!<analysis>).)*?</analysis>\s*"
            r"<confidence>\s*(?:" + re.escape(high) + "|" + re.escape(low) + r"|[^<]{0,30})\s*</confidence>\s*\Z"
        )
        if re.match(pat, text, re.DOTALL) is None:
            return 0.0
        aidx = text.find("<analysis>")
        if _extract(text[:aidx]) is None:
            return 0.0
        return 1.0

    def _compute_loss(self, model, inputs):
        # Log the metrics
        mode = "train" if self.model.training else "eval"
        if mode == "train":
            torch.cuda.empty_cache()
            if not self.vllm_sleeping:
                self.llm.sleep()
                self.accelerator.wait_for_everyone()
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]

         #sum completion masks to find max dimension
        max_dim = max(int(completion_mask.sum(1).max().item()), 1)
        #now clip everything to that dim
        completion_mask = completion_mask[:, :max_dim]
        completion_ids = completion_ids[:, :max_dim]

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # CREDO: single forward also yields raw delta-logits at the readout position
        _readout = inputs.get("readout_pos")
        if self.method == "credo" and _readout is not None:
            per_token_logps, _delta_raw, _logmass = self._get_per_token_logps(
                model, input_ids, attention_mask, logits_to_keep, readout_pos=_readout
            )
        else:
            per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
            _delta_raw, _logmass = None, None

        if self.beta != 0.0:
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, input_ids, attention_mask, logits_to_keep
                    )
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # advantages are (B, T_gen); clip to max_dim alongside completion tensors
        advantages = inputs["advantages"][:, :max_dim]
        # CREDO: pg_mask excludes the readout token from PG; rlcr/dcpo fall back to completion_mask
        pg_mask = inputs.get("pg_mask")
        loss_mask = completion_mask if pg_mask is None else pg_mask[:, :max_dim]
        old_per_token_logps = (
            per_token_logps.detach() if inputs["old_per_token_logps"] is None else inputs["old_per_token_logps"]
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        if self.args.delta is not None:
            # Use clamp instead of min to handle tensor-float comparison
            per_token_loss1 = torch.clamp(coef_1, max=self.args.delta) * advantages
        else:
            # Original GRPO clipping (only lower bound implicitly applied by the final min)
            per_token_loss1 = coef_1 * advantages

        
        per_token_loss2 = coef_2 * advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = ((per_token_loss * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * loss_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # CREDO: direct MSE calibration loss on sigma(raw delta-logit)
        if _delta_raw is not None:
            mse_target = inputs["mse_target"].to(_delta_raw.dtype)
            mse_valid = inputs["mse_valid"].to(_delta_raw.dtype)
            conf_hf = torch.sigmoid(_delta_raw)
            se = (conf_hf - mse_target) ** 2
            mse_loss = (se * mse_valid).sum() / mse_valid.sum().clamp(min=1.0)
            # alpha linear warmup to soften the early calibration migration
            # warmup factor unified across both auxiliary losses (MSE + mass)
            _alpha = float(getattr(self.args, "mse_alpha", 1.0))
            _aw = int(getattr(self.args, "mse_alpha_warmup_steps", 0) or 0)
            _wu = min(1.0, float(self.state.global_step + 1) / float(_aw)) if _aw > 0 else 1.0
            _alpha = _alpha * _wu
            loss = loss + _alpha * mse_loss
            self._metrics[mode]["mse/alpha_eff"].append(_alpha)
            if _logmass is not None:
                _mmode = str(getattr(self.args, "mass_loss_mode", "always") or "always")
                if _mmode == "topk_gate":
                    # conditional (satisficing) mass — push only samples whose special pair
                    # missed the vLLM top-k at readout; normalize by valid count (not gate count)
                    _mgate = inputs.get("mass_gate")
                    _mgate = torch.ones_like(mse_valid) if _mgate is None else _mgate.to(_delta_raw.dtype)
                    _mw = mse_valid * _mgate
                    self._metrics[mode]["mass/gate_open_rate"].append(
                        self.accelerator.gather_for_metrics(
                            (_mw.sum() / mse_valid.sum().clamp(min=1.0)).detach()).nanmean().item())
                else:
                    _mw = mse_valid
                mass_loss = -(_logmass * _mw).sum() / mse_valid.sum().clamp(min=1.0)
                _lam = float(getattr(self.args, "mass_loss_lambda", 0.0)) * _wu
                loss = loss + _lam * mass_loss
                self._metrics[mode]["mse/lambda_eff"].append(_lam)
                self._metrics[mode]["loss/mass"].append(
                    self.accelerator.gather_for_metrics(mass_loss.detach()).nanmean().item())
            self._metrics[mode]["loss/mse"].append(
                self.accelerator.gather_for_metrics(mse_loss.detach()).nanmean().item())
            conf_gen = inputs.get("conf_gen")
            if conf_gen is not None:
                gap = ((conf_hf.detach() - conf_gen.to(conf_hf.dtype)).abs() * mse_valid).sum() / mse_valid.sum().clamp(min=1.0)
                self._metrics[mode]["conf/vllm_hf_gap"].append(
                    self.accelerator.gather_for_metrics(gap).nanmean().item())

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).nanmean().item())

         # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = (is_low_clipped * completion_mask).sum() / completion_mask.sum()
        high_clip = (is_high_clipped * completion_mask).sum() / completion_mask.sum()
        clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()

        gathered_low_clip = self.accelerator.gather_for_metrics(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather_for_metrics(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather_for_metrics(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss

    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        # Return placeholder zero loss without running compute_loss
        return torch.tensor(0.0, device=self.accelerator.device), None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process and self.log_completions:
            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd
                table = {
                    "step": self._textual_logs["step"],
                    "prompt": self._textual_logs["prompt"],
                    "completion": self._textual_logs["completion"],
                    **self._textual_logs["rewards"],
                }
                df = pd.DataFrame(table)
                wandb.log({"completions": wandb.Table(dataframe=df)})
        

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
