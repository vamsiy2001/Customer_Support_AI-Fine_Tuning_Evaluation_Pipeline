"""
LLM-as-Judge Evaluation using GPT-4o.

Each response is rated on 3 dimensions:
  - Helpfulness   (1-5): Does it actually solve the customer's problem?
  - Accuracy      (1-5): Is the information correct and specific?
  - Professionalism (1-5): Tone appropriate for customer support?

Also computes: preference rate (which model a judge prefers head-to-head).

Usage:
    python evaluation/llm_judge.py \
        --base_predictions evaluation/results/base/predictions.parquet \
        --ft_predictions evaluation/results/finetuned/predictions.parquet \
        --output_dir evaluation/results \
        --n_samples 50

Cost estimate: ~50 samples × 2 models × ~600 tokens = ~60K tokens ≈ $0.30 with GPT-4o-mini
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.progress import track
from rich.table import Table

load_dotenv()
console = Console()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── prompts ────────────────────────────────────────────────────────────────
RATING_PROMPT = """You are evaluating AI customer support responses. Score the response on three criteria, each from 1-5.

Customer question: {instruction}

Response to evaluate:
{response}

Rate the response on:
1. HELPFULNESS (1-5): Does it address and resolve the customer's actual problem?
   1=completely unhelpful, 5=fully resolves the issue
2. ACCURACY (1-5): Is the information correct, specific, and actionable?
   1=wrong/vague, 5=accurate and specific
3. PROFESSIONALISM (1-5): Is the tone appropriate for customer support?
   1=rude/robotic, 5=empathetic and professional

Return ONLY valid JSON in this exact format:
{{"helpfulness": X, "accuracy": X, "professionalism": X, "reasoning": "brief explanation"}}"""

PREFERENCE_PROMPT = """You are comparing two customer support responses. Pick the better one.

Customer question: {instruction}

Response A:
{response_a}

Response B:
{response_b}

Which response better resolves the customer's issue? Consider helpfulness, accuracy, and professionalism.
Return ONLY valid JSON: {{"winner": "A" or "B", "reasoning": "brief explanation"}}"""


# ── rating functions ────────────────────────────────────────────────────────
def rate_response(
    instruction: str,
    response: str,
    model: str = "gpt-4o-mini",
    retries: int = 3,
) -> dict:
    """Rate a single response. Returns dict with scores or None on failure."""
    prompt = RATING_PROMPT.format(instruction=instruction, response=response)

    for attempt in range(retries):
        try:
            result = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            return json.loads(result.choices[0].message.content)
        except Exception as e:
            if attempt == retries - 1:
                console.print(f"[red]Rating failed after {retries} attempts: {e}[/red]")
                return {"helpfulness": None, "accuracy": None, "professionalism": None, "reasoning": "error"}
            time.sleep(2 ** attempt)


def judge_preference(
    instruction: str,
    response_a: str,
    response_b: str,
    model: str = "gpt-4o-mini",
) -> dict:
    """Head-to-head comparison. Returns winner ('A' or 'B') and reasoning."""
    prompt = PREFERENCE_PROMPT.format(
        instruction=instruction,
        response_a=response_a,
        response_b=response_b,
    )
    try:
        result = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=150,
            response_format={"type": "json_object"},
        )
        return json.loads(result.choices[0].message.content)
    except Exception as e:
        console.print(f"[red]Preference judgment failed: {e}[/red]")
        return {"winner": None, "reasoning": "error"}


# ── aggregate results ──────────────────────────────────────────────────────
def aggregate_scores(scores: list[dict]) -> dict:
    valid = [s for s in scores if s.get("helpfulness") is not None]
    if not valid:
        return {}
    return {
        "avg_helpfulness": round(np.mean([s["helpfulness"] for s in valid]), 3),
        "avg_accuracy": round(np.mean([s["accuracy"] for s in valid]), 3),
        "avg_professionalism": round(np.mean([s["professionalism"] for s in valid]), 3),
        "avg_composite": round(
            np.mean(
                [
                    (s["helpfulness"] + s["accuracy"] + s["professionalism"]) / 3
                    for s in valid
                ]
            ),
            3,
        ),
        "n_valid": len(valid),
        "n_failed": len(scores) - len(valid),
    }


# ── main evaluation ────────────────────────────────────────────────────────
def run_llm_judge(
    base_pred_path: str,
    ft_pred_path: str,
    output_dir: str,
    n_samples: int = 50,
    judge_model: str = "gpt-4o-mini",
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    base_df = pd.read_parquet(base_pred_path)
    ft_df = pd.read_parquet(ft_pred_path)

    # align on same sample
    if n_samples < len(base_df):
        base_df = base_df.sample(n=n_samples, random_state=42).reset_index(drop=True)
        ft_df = ft_df.loc[base_df.index].reset_index(drop=True)

    console.print(f"\nRunning LLM-as-Judge on {len(base_df)} samples with [cyan]{judge_model}[/cyan]")
    console.print(f"Estimated cost: ~${len(base_df) * 0.006:.2f} (gpt-4o-mini pricing)\n")

    base_scores, ft_scores, preferences = [], [], []

    for i in track(range(len(base_df)), description="Judging responses..."):
        instruction = base_df.iloc[i]["instruction"]
        base_resp = base_df.iloc[i]["prediction"]
        ft_resp = ft_df.iloc[i]["prediction"]

        # rate each independently
        base_score = rate_response(instruction, base_resp, model=judge_model)
        ft_score = rate_response(instruction, ft_resp, model=judge_model)
        base_scores.append(base_score)
        ft_scores.append(ft_score)

        # head-to-head (randomize order to avoid position bias)
        if i % 2 == 0:  # A=base, B=ft
            pref = judge_preference(instruction, base_resp, ft_resp, model=judge_model)
            preferences.append("base" if pref.get("winner") == "A" else "finetuned")
        else:  # A=ft, B=base  (swap)
            pref = judge_preference(instruction, ft_resp, base_resp, model=judge_model)
            preferences.append("finetuned" if pref.get("winner") == "A" else "base")

        # small delay to avoid rate limits
        time.sleep(0.3)

    # aggregate
    base_agg = aggregate_scores(base_scores)
    ft_agg = aggregate_scores(ft_scores)
    ft_win_rate = preferences.count("finetuned") / len(preferences)

    results = {
        "judge_model": judge_model,
        "n_samples": len(base_df),
        "base_model": base_agg,
        "finetuned_model": ft_agg,
        "preference_win_rate_finetuned": round(ft_win_rate, 3),
        "preference_win_rate_base": round(1 - ft_win_rate, 3),
    }

    with open(output_path / "llm_judge_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # save detailed scores
    details_df = base_df[["instruction", "intent", "category"]].copy()
    details_df["base_prediction"] = base_df["prediction"]
    details_df["ft_prediction"] = ft_df["prediction"]
    details_df["base_helpfulness"] = [s.get("helpfulness") for s in base_scores]
    details_df["base_accuracy"] = [s.get("accuracy") for s in base_scores]
    details_df["base_professionalism"] = [s.get("professionalism") for s in base_scores]
    details_df["ft_helpfulness"] = [s.get("helpfulness") for s in ft_scores]
    details_df["ft_accuracy"] = [s.get("accuracy") for s in ft_scores]
    details_df["ft_professionalism"] = [s.get("professionalism") for s in ft_scores]
    details_df["preference"] = preferences
    details_df.to_parquet(output_path / "llm_judge_details.parquet", index=False)

    # print summary
    table = Table(title="LLM-as-Judge Results", show_lines=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Base Model", justify="right")
    table.add_column("Fine-tuned", justify="right")
    table.add_column("Δ", justify="right")

    for metric in ["avg_helpfulness", "avg_accuracy", "avg_professionalism", "avg_composite"]:
        base_val = base_agg.get(metric, 0)
        ft_val = ft_agg.get(metric, 0)
        delta = ft_val - base_val
        color = "green" if delta > 0 else "red"
        table.add_row(
            metric.replace("avg_", "").title(),
            str(base_val),
            str(ft_val),
            f"[{color}]{delta:+.3f}[/{color}]",
        )

    console.print(table)
    console.print(
        f"\nFine-tuned model win rate: [bold green]{ft_win_rate*100:.1f}%[/bold green] "
        f"(head-to-head preference)"
    )
    console.print(f"Results saved to [green]{output_path}[/green]")
    return results


# ── entrypoint ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_predictions", required=True)
    parser.add_argument("--ft_predictions", required=True)
    parser.add_argument("--output_dir", default="evaluation/results")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--judge_model", default="gpt-4o-mini")
    args = parser.parse_args()

    run_llm_judge(
        base_pred_path=args.base_predictions,
        ft_pred_path=args.ft_predictions,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        judge_model=args.judge_model,
    )
