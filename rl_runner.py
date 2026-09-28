from arguments import GRPOScriptArguments,GRPOConfig,ModelConfig
from trl import TrlParser, get_peft_config
from transformers import set_seed
import logging
import transformers
import datasets 
from datasets import load_dataset
import sys
import os 
from transformers.trainer_utils import get_last_checkpoint
from reward_fns import (format_reward, accuracy_reward, brier_reward, mean_confidence_reward,
                        confidence_one_or_zero, format_code_reward, accuracy_code_reward,
                        brier_code_reward, accuracy_ungated_reward, accuracy_code_ungated_reward,
                        assert_pure_acc_recipe)
from system_prompts import get_sys_prompt
from dataset_processing import process_dataset 
from GRPO_Trainer import CustomTrainer
import torch
from functools import partial


logger = logging.getLogger(__name__)


def finalize_vllm_training_process(trainer, exit_code=0):
    if hasattr(trainer, "accelerator"):
        trainer.accelerator.wait_for_everyone()
    try:
        if trainer.accelerator.is_main_process:
            logger.info("Finalizing W&B before controlled process exit")
            try:
                import wandb
                if wandb.run is not None:
                    wandb.finish()
            except Exception as exc:
                logger.warning(f"W&B finalization skipped: {exc}")
    finally:
        if hasattr(trainer, "accelerator"):
            trainer.accelerator.wait_for_everyone()
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(exit_code)


def logger_setup(script_args, training_args, model_args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

def model_init(model_args, training_args):
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    return model_kwargs

def main(script_args, training_args, model_args):
    set_seed(training_args.seed)
    # Authoritative effective values (post-TrlParser, CLI>YAML) — the shell banner
    # only best-effort-merges YAML+CLI for display; THIS block is what the trainer
    # actually consumes. rank-0 only to keep logs clean under multi-process launch.
    if os.environ.get("LOCAL_RANK", "0") == "0":
        _eff_keys = ["method", "ans_reward_mode", "sw_kappa", "sw_kappa_neg", "sw_weight_clip",
                     "sw_warmup_steps", "mse_alpha", "mass_loss_lambda",
                     "mass_loss_mode", "mse_alpha_warmup_steps", "mse_gamma", "w_fmt_ans",
                     "cal_gamma", "cal_weight", "mse_detach_scale", "max_completion_length",
                     "max_prompt_length", "num_generations", "temperature", "seed",
                     "learning_rate", "num_train_epochs", "per_device_train_batch_size",
                     "gradient_accumulation_steps", "mask_truncated_completions",
                     "scale_rewards", "loss_type", "beta", "conf_think", "pg_channels", "sw_signal"]
        print("==================== EFFECTIVE PARAMETERS (post-TrlParser, CLI>YAML) ====================")
        for _k in _eff_keys:
            print(f"[effective] {_k} = {getattr(training_args, _k, '<absent>')}")
        print(f"[effective] format_pattern = {getattr(script_args, 'format_pattern', '<absent>')}")
        print(f"[effective] sys_prompt_name = {getattr(script_args, 'sys_prompt_name', '<absent>')}")
        print("==========================================================================================", flush=True)
    # fail-fast: a noct prompt with conf_think=True (or the converse) would silently
    # train against the wrong F' pattern — refuse to start rather than waste a 2-day job.
    _noct_name = "noct" in str(getattr(script_args, "sys_prompt_name", ""))
    if _noct_name == bool(getattr(training_args, "conf_think", True)):
        raise ValueError(f"conf_think={getattr(training_args, 'conf_think', True)} is inconsistent with "
                         f"sys_prompt_name={getattr(script_args, 'sys_prompt_name', '')!r} (noct prompts "
                         f"require --conf_think False and vice versa)")
    # fail-fasts: the PG-decomposition and SW-control knobs are CREDO-only, and a control
    # signal with sw_kappa=0 (inert hook) or an asymmetric kappa (unreplicated machinery)
    # would be a silent misconfiguration — die at launch instead.
    _pgc = str(getattr(training_args, "pg_channels", "both"))
    _sws = str(getattr(training_args, "sw_signal", "surprise"))
    if (_pgc != "both" or _sws != "surprise") and str(getattr(training_args, "method", "")) != "credo":
        raise ValueError(f"pg_channels={_pgc!r} / sw_signal={_sws!r} require method=credo")
    if _sws != "surprise":
        if not float(getattr(training_args, "sw_kappa", 0.0)) > 0.0:
            raise ValueError(f"sw_signal={_sws!r} requires sw_kappa > 0 (the control replaces the SW "
                             f"signal; with kappa=0 the hook would be inert and the arm meaningless)")
        if getattr(training_args, "sw_kappa_neg", None) is not None:
            raise ValueError("sw_signal controls replicate the symmetric SW only (sw_kappa_neg must be None)")
    logger_setup(script_args, training_args, model_args) 

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if os.path.isdir(script_args.dataset_name):
        dataset = datasets.load_from_disk(script_args.dataset_name)
    else:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

     # Get reward functions
    REWARD_FUNCS_REGISTRY = {
        "format": partial(format_reward, format_pattern=script_args.format_pattern),
        "accuracy": partial(accuracy_reward, format_pattern=script_args.format_pattern),
        "brier": partial(brier_reward, format_pattern=script_args.format_pattern),
        "mean_confidence": mean_confidence_reward,
        "confidence_one_or_zero": confidence_one_or_zero,
        "format_code": partial(format_code_reward, format_pattern=script_args.format_pattern),
        "accuracy_code": partial(accuracy_code_reward, format_pattern=script_args.format_pattern),
        "brier_code": partial(brier_code_reward, format_pattern=script_args.format_pattern),
        "accuracy_ungated": accuracy_ungated_reward,
        "accuracy_code_ungated": accuracy_code_ungated_reward,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]
    assert_pure_acc_recipe(getattr(training_args, "method", "rlcr"),
                           script_args.reward_funcs, training_args.reward_weights or [])

    dataset = process_dataset(dataset, script_args)  
    training_args.enable_thinking = script_args.enable_thinking

    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    model_init_kwargs = model_init(model_args, training_args)
    training_args.model_init_kwargs = model_init_kwargs

    if training_args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project

    train_dataset = dataset[script_args.dataset_train_split]
    eval_dataset = dataset[script_args.dataset_test_split]
    if script_args.train_subset_size is not None:
        train_dataset = train_dataset.select(range(script_args.train_subset_size))
    if script_args.eval_subset_size is not None:
        eval_dataset = eval_dataset.select(range(script_args.eval_subset_size))
        
    #############################
    # Initialize the GRPO trainer
    #############################
    trainer = CustomTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None)

     ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = script_args.train_subset_size
    # trainer.log_metrics("train", metrics)
    # trainer.save_metrics("train", metrics)
    try:
        trainer.save_state()
    except:
        print("Failed to save state, please debug")
        pass
    trainer.accelerator.wait_for_everyone()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    trainer.accelerator.wait_for_everyone()
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["rl-verify"],
    }
    if trainer.accelerator.is_main_process:
        try:
            trainer.create_model_card(**kwargs)
        except Exception as exc:
            logger.warning(f"Model card creation skipped: {exc}")
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)
    trainer.accelerator.wait_for_everyone()
    finalize_vllm_training_process(trainer)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)

    
