"""
Main training script for LLM Fine-Tuning Framework.
Runs on Google Colab (T4 GPU) or any CUDA machine.

Usage:
    python experiments/run_experiment.py --config experiments/configs/lora_r16.yaml
    python experiments/run_experiment.py --config experiments/configs/lora_r64.yaml
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import wandb
import yaml
from datasets import load_from_disk
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from transformers import TrainerCallback, TrainingArguments

console = Console()
load_dotenv()


# ── config loader ──────────────────────────────────────────────────────────
def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── W&B callback ───────────────────────────────────────────────────────────
class WandbMetricsCallback(TrainerCallback):
    """Log extra metrics to W&B on top of default trainer logs."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and wandb.run:
            wandb.log({"train/global_step": state.global_step, **logs})


# ── main training function ─────────────────────────────────────────────────
def run_training(config: dict):
    # guard: unsloth only works on GPU
    if not torch.cuda.is_available():
        console.print("[red]No CUDA GPU detected. Run this on Colab or a GPU machine.[/red]")
        console.print("For local inference/evaluation, use inference/benchmark.py with MPS.")
        sys.exit(1)

    # lazy import unsloth (must be first CUDA import)
    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
    except ImportError:
        console.print("[red]Unsloth not installed. Run: pip install 'unsloth[colab-new] @ git+...'[/red]")
        sys.exit(1)

    from trl import SFTTrainer

    cfg = config
    exp_name = cfg["experiment_name"]
    console.print(f"\n[bold green]Starting experiment: {exp_name}[/bold green]")

    # ── W&B init ────────────────────────────────────────────────────────
    wandb_cfg = cfg.get("wandb", {})
    wandb.init(
        project=wandb_cfg.get("project", "llm-finetuning"),
        name=exp_name,
        config=cfg,
        tags=wandb_cfg.get("tags", []),
    )

    # ── model ────────────────────────────────────────────────────────────
    model_cfg = cfg["model"]
    console.print(f"Loading model: [cyan]{model_cfg['name']}[/cyan]")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=model_cfg["max_seq_length"],
        dtype=model_cfg.get("dtype"),
        load_in_4bit=model_cfg["load_in_4bit"],
    )

    # apply chat template (Llama-3 instruct)
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3")

    # ── LoRA ─────────────────────────────────────────────────────────────
    lora_cfg = cfg["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        use_gradient_checkpointing=lora_cfg["use_gradient_checkpointing"],
        use_rslora=lora_cfg.get("use_rslora", False),
        random_state=cfg["training"]["seed"],
    )

    # log trainable params to W&B
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    console.print(f"Trainable params: [yellow]{trainable:,}[/yellow] / {total:,} ({trainable/total*100:.2f}%)")
    wandb.log({"model/trainable_params": trainable, "model/total_params": total})

    # ── dataset ──────────────────────────────────────────────────────────
    train_cfg = cfg["training"]
    console.print(f"Loading dataset from: {train_cfg['dataset_path']}")
    dataset = load_from_disk(train_cfg["dataset_path"])

    def apply_template(examples):
        texts = tokenizer.apply_chat_template(
            examples["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": texts}

    dataset = dataset.map(apply_template, batched=True)
    console.print(f"Train: {len(dataset['train']):,} | Val: {len(dataset['validation']):,}")

    # ── trainer ──────────────────────────────────────────────────────────
    output_dir = train_cfg["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg["num_train_epochs"],
        max_steps=train_cfg["max_steps"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        warmup_steps=train_cfg["warmup_steps"],
        learning_rate=train_cfg["learning_rate"],
        fp16=train_cfg["fp16"],
        bf16=train_cfg["bf16"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        eval_strategy=train_cfg["evaluation_strategy"],
        eval_steps=train_cfg["eval_steps"],
        seed=train_cfg["seed"],
        optim=train_cfg["optim"],
        weight_decay=train_cfg["weight_decay"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        report_to=train_cfg["report_to"],
        run_name=exp_name,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        dataset_text_field="text",
        max_seq_length=model_cfg["max_seq_length"],
        dataset_num_proc=2,
        packing=False,
        args=training_args,
        callbacks=[WandbMetricsCallback()],
    )

    # ── train ─────────────────────────────────────────────────────────────
    console.print("\n[bold]Starting training...[/bold]")
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    console.print(f"Training done in {elapsed/60:.1f} minutes")
    wandb.log({"train/total_minutes": elapsed / 60})

    # ── save ──────────────────────────────────────────────────────────────
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    console.print(f"Model saved to [green]{output_dir}[/green]")

    # optionally push to HuggingFace Hub
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        repo_id = f"vamsiyvk/{exp_name}"
        model.push_to_hub(repo_id, token=hf_token)
        tokenizer.push_to_hub(repo_id, token=hf_token)
        console.print(f"Pushed to HuggingFace Hub: [cyan]{repo_id}[/cyan]")
        wandb.log({"hub/repo_id": repo_id})

    wandb.finish()
    console.print("\n[bold green]Experiment complete.[/bold green]")
    console.print("Next: run evaluation/automated_eval.py to score your model.")


# ── entrypoint ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM fine-tuning experiment")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config (e.g., experiments/configs/lora_r16.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_training(config)
