"""
LLM-as-Judge Evaluation using Groq (free tier).

Judge model: llama-3.3-70b-versatile via Groq API (free, no credit card needed).
Sign up at console.groq.com → API Keys → create key → add to .env as GROQ_API_KEY.

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
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from rich.console import Console
from rich.progress import track
from rich.table import Table

load_dotenv()
console = Console()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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


# ── rate-limit helpers ─────────────────────────────────────────────────────
def _parse_retry_seconds(err_str: str) -> float | None:
    """Extract wait seconds from Groq 429 message ('try again in 6m14.976s')."""
    m = re.search(r"try again in\s+(?:(\d+)h\s*)?(?:(\d+)m\s*)?(\d+(?:\.\d+)?s)?", str(err_str))
    if not m:
        return None
    hours   = int(m.group(1) or 0)
    minutes = int(m.group(2) or 0)
    secs    = float((m.group(3) or "0s").rstrip("s"))
    return hours * 3600 + minutes * 60 + secs + 5  # +5 s buffer


def _is_daily_limit(err_str: str) -> bool:
    return "tokens per day" in str(err_str) or "TPD" in str(err_str)


def _groq_call(fn, *args, retries: int = 4, **kwargs):
    """Wrapper that retries Groq calls with smart rate-limit handling."""
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                if _is_daily_limit(err):
                    raise RuntimeError(
                        "\n[bold red]Groq daily token limit reached (100K/day).[/bold red]\n"
                        "Wait until midnight UTC, then re-run.\n"
                        "Tip: use --n_samples 25 to keep each run under 50K tokens."
                    ) from e
                wait = _parse_retry_seconds(err) or (2 ** attempt * 10)
                console.print(f"[yellow]Rate limit — sleeping {wait:.0f}s (attempt {attempt+1}/{retries})[/yellow]")
                time.sleep(wait)
            else:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
    raise RuntimeError("Groq call failed after all retries")


# ── rating functions ────────────────────────────────────────────────────────
def rate_response(
    instruction: str,
    response: str,
    model: str = "llama-3.3-70b-versatile",
) -> dict:
    """Rate a single response. Returns dict with scores or None on failure."""
    prompt = RATING_PROMPT.format(instruction=instruction, response=response)
    try:
        result = _groq_call(
            client.chat.completions.create,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        return json.loads(result.choices[0].message.content)
    except RuntimeError as e:
        console.print(str(e))
        raise SystemExit(1) from e
    except Exception as e:
        console.print(f"[red]Rating failed: {e}[/red]")
        return {"helpfulness": None, "accuracy": None, "professionalism": None, "reasoning": "error"}


def judge_preference(
    instruction: str,
    response_a: str,
    response_b: str,
    model: str = "llama-3.3-70b-versatile",
) -> dict:
    """Head-to-head comparison. Returns winner ('A' or 'B') and reasoning."""
    prompt = PREFERENCE_PROMPT.format(
        instruction=instruction,
        response_a=response_a,
        response_b=response_b,
    )
    try:
        result = _groq_call(
            client.chat.completions.create,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=150,
            response_format={"type": "json_object"},
        )
        return json.loads(result.choices[0].message.content)
    except RuntimeError as e:
        console.print(str(e))
        raise SystemExit(1) from e
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
    judge_model: str = "llama-3.3-70b-versatile",
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    base_df = pd.read_parquet(base_pred_path)
    ft_df = pd.read_parquet(ft_pred_path)

    # align on same sample — capture index BEFORE reset so both dfs use the same rows
    if n_samples < len(base_df):
        sample_idx = base_df.sample(n=n_samples, random_state=42).index
        base_df = base_df.loc[sample_idx].reset_index(drop=True)
        ft_df = ft_df.loc[sample_idx].reset_index(drop=True)

    console.print(f"\nRunning LLM-as-Judge on {len(base_df)} samples with [cyan]{judge_model}[/cyan]")
    console.print("Cost: [green]$0.00[/green] (Groq free tier)\n")

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
    parser.add_argument(
        "--experiment", default="r16",
        help="Experiment subfolder name, e.g. r16 or r64"
    )
    parser.add_argument(
        "--base_predictions", default=None,
        help="Override base predictions path (defaults to evaluation/results/<experiment>/base_predictions.parquet)"
    )
    parser.add_argument(
        "--ft_predictions", default=None,
        help="Override ft predictions path (defaults to evaluation/results/<experiment>/finetuned_predictions.parquet)"
    )
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--judge_model", default="llama-3.3-70b-versatile")
    args = parser.parse_args()

    base_path = args.base_predictions or f"evaluation/results/{args.experiment}/base_predictions.parquet"
    ft_path = args.ft_predictions or f"evaluation/results/{args.experiment}/finetuned_predictions.parquet"
    output_dir = f"evaluation/results/{args.experiment}"

    run_llm_judge(
        base_pred_path=base_path,
        ft_pred_path=ft_path,
        output_dir=output_dir,
        n_samples=args.n_samples,
        judge_model=args.judge_model,
    )
