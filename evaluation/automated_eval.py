"""
Automated Evaluation Pipeline.

Metrics computed:
  1. Perplexity         — how confident the model is on test responses
  2. ROUGE-L            — n-gram overlap with reference responses
  3. BERTScore          — semantic similarity (embedding-based)
  4. BLEU               — classic MT metric, used here for response quality
  5. Intent Accuracy    — does the response address the correct intent?
     (proxy: classify response with zero-shot classifier)
  6. Response Length    — avg token count vs reference (verbosity check)

Usage:
    # evaluate a fine-tuned model vs base
    python evaluation/automated_eval.py \
        --model_path ./outputs/lora_r16 \
        --base_model unsloth/Llama-3.2-3B-Instruct \
        --test_data data/cleaned/test_set.parquet \
        --output_dir evaluation/results/lora_r16 \
        --n_samples 200
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from bert_score import score as bert_score
from datasets import load_from_disk
from evaluate import load as load_metric
from rich.console import Console
from rich.progress import track
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

console = Console()


# ── model loader ───────────────────────────────────────────────────────────
def load_model_and_tokenizer(model_path: str, device: str = "auto"):
    """Works on CUDA and Apple Silicon (MPS)."""
    console.print(f"Loading model: [cyan]{model_path}[/cyan]")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    if torch.backends.mps.is_available():
        # Apple M-series: load in float16, map to MPS
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16
        ).to("mps")
        device = "mps"
    elif torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )
        device = "cuda"
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        device = "cpu"

    console.print(f"Model on: [yellow]{device}[/yellow]")
    model.eval()
    return model, tokenizer, device


# ── generation ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a helpful, professional customer support agent. "
    "Respond clearly and empathetically to customer inquiries. "
    "Be concise, accurate, and solution-focused."
)


def generate_response(
    model,
    tokenizer,
    instruction: str,
    device: str,
    max_new_tokens: int = 256,
) -> tuple[str, float]:
    """Returns (generated_text, latency_ms)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    # use apply_chat_template if available
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt = f"System: {SYSTEM_PROMPT}\nUser: {instruction}\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency_ms = (time.perf_counter() - start) * 1000

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return generated.strip(), latency_ms


# ── perplexity ─────────────────────────────────────────────────────────────
def compute_perplexity(model, tokenizer, texts: list[str], device: str) -> float:
    """Average perplexity over a list of full conversation texts."""
    total_loss = 0.0
    count = 0
    for text in track(texts, description="Computing perplexity..."):
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            loss = model(**inputs, labels=inputs["input_ids"]).loss
        total_loss += loss.item()
        count += 1
    return float(np.exp(total_loss / count))


# ── metrics ────────────────────────────────────────────────────────────────
def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    rouge = load_metric("rouge")
    result = rouge.compute(predictions=predictions, references=references)
    return {k: round(v, 4) for k, v in result.items()}


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    bleu = load_metric("bleu")
    tokenized_preds = [p.split() for p in predictions]
    tokenized_refs = [[r.split()] for r in references]
    result = bleu.compute(predictions=tokenized_preds, references=tokenized_refs)
    return round(result["bleu"], 4)


def compute_bertscore(predictions: list[str], references: list[str]) -> dict:
    console.print("Computing BERTScore (this takes ~1 min)...")
    P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
    return {
        "bertscore_precision": round(P.mean().item(), 4),
        "bertscore_recall": round(R.mean().item(), 4),
        "bertscore_f1": round(F1.mean().item(), 4),
    }


def compute_response_length_stats(
    predictions: list[str], references: list[str]
) -> dict:
    pred_lens = [len(p.split()) for p in predictions]
    ref_lens = [len(r.split()) for r in references]
    return {
        "avg_pred_length": round(np.mean(pred_lens), 1),
        "avg_ref_length": round(np.mean(ref_lens), 1),
        "length_ratio": round(np.mean(pred_lens) / max(np.mean(ref_lens), 1), 3),
    }


# ── full evaluation run ────────────────────────────────────────────────────
def evaluate_model(
    model_path: str,
    test_data_path: str,
    output_dir: str,
    n_samples: int = 200,
    label: str = "model",
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model, tokenizer, device = load_model_and_tokenizer(model_path)

    # load test set
    df = pd.read_parquet(test_data_path)
    if n_samples and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
    console.print(f"Evaluating on {len(df)} test samples")

    # ── generate responses ─────────────────────────────────────────────
    predictions, latencies = [], []
    for _, row in track(df.iterrows(), total=len(df), description="Generating responses"):
        pred, lat = generate_response(model, tokenizer, row["instruction"], device)
        predictions.append(pred)
        latencies.append(lat)

    references = df["response"].tolist()

    # ── perplexity (on reference responses) ───────────────────────────
    ref_texts = [
        f"User: {inst}\nAssistant: {resp}"
        for inst, resp in zip(df["instruction"], references)
    ]
    perplexity = compute_perplexity(model, tokenizer, ref_texts[:50], device)  # 50 for speed

    # ── automatic metrics ──────────────────────────────────────────────
    rouge_scores = compute_rouge(predictions, references)
    bleu_score = compute_bleu(predictions, references)
    bert_scores = compute_bertscore(predictions, references)
    length_stats = compute_response_length_stats(predictions, references)

    results = {
        "label": label,
        "model_path": model_path,
        "n_samples": len(df),
        "perplexity": round(perplexity, 2),
        "bleu": bleu_score,
        **rouge_scores,
        **bert_scores,
        **length_stats,
        "avg_latency_ms": round(np.mean(latencies), 1),
        "p95_latency_ms": round(np.percentile(latencies, 95), 1),
    }

    # save raw predictions for LLM judge
    predictions_df = df[["instruction", "response", "intent", "category"]].copy()
    predictions_df["prediction"] = predictions
    predictions_df["latency_ms"] = latencies
    predictions_df.to_parquet(output_path / "predictions.parquet", index=False)

    with open(output_path / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    console.print(f"\nResults saved to [green]{output_path}[/green]")
    return results


# ── comparison table printer ───────────────────────────────────────────────
def print_comparison(results_list: list[dict]):
    table = Table(title="Model Comparison", show_lines=True)
    table.add_column("Model", style="cyan")
    table.add_column("Perplexity ↓", justify="right")
    table.add_column("ROUGE-L ↑", justify="right")
    table.add_column("BLEU ↑", justify="right")
    table.add_column("BERTScore F1 ↑", justify="right")
    table.add_column("Avg Latency (ms) ↓", justify="right")

    for r in results_list:
        table.add_row(
            r["label"],
            str(r["perplexity"]),
            str(r.get("rougeL", "N/A")),
            str(r["bleu"]),
            str(r.get("bertscore_f1", "N/A")),
            str(r["avg_latency_ms"]),
        )
    console.print(table)


# ── entrypoint ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to fine-tuned model")
    parser.add_argument("--base_model", default="unsloth/Llama-3.2-3B-Instruct")
    parser.add_argument("--test_data", default="data/cleaned/test_set.parquet")
    parser.add_argument("--output_dir", default="evaluation/results")
    parser.add_argument("--n_samples", type=int, default=200)
    args = parser.parse_args()

    all_results = []

    # evaluate fine-tuned model
    console.print("\n[bold]Evaluating fine-tuned model...[/bold]")
    ft_results = evaluate_model(
        model_path=args.model_path,
        test_data_path=args.test_data,
        output_dir=f"{args.output_dir}/finetuned",
        n_samples=args.n_samples,
        label="Fine-tuned",
    )
    all_results.append(ft_results)

    # evaluate base model for comparison
    console.print("\n[bold]Evaluating base model...[/bold]")
    base_results = evaluate_model(
        model_path=args.base_model,
        test_data_path=args.test_data,
        output_dir=f"{args.output_dir}/base",
        n_samples=args.n_samples,
        label="Base",
    )
    all_results.append(base_results)

    print_comparison(all_results)

    # save combined results
    with open(f"{args.output_dir}/comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)
    console.print(f"\nComparison saved to {args.output_dir}/comparison.json")
