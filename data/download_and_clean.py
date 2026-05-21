"""
Data pipeline for Bitext Customer Support LLM Fine-tuning Dataset.
Dataset: bitext/Bitext-customer-support-llm-chatbot-training-dataset

Why this dataset:
- 26,872 real customer support conversations
- 27 intents across 11 business domains (billing, orders, returns, etc.)
- 'flags' column marks quality issues: B=basic, I=irrelevant, K=keyword-stuffed
  → Real messy data you have to deal with in production
- Multiple linguistic patterns per intent (good for generalization)
"""

import json
import os
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split

# ── paths ──────────────────────────────────────────────────────────────────
RAW_DIR = Path("data/raw")
CLEANED_DIR = Path("data/cleaned")
ANALYSIS_DIR = Path("data/analysis")

for d in [RAW_DIR, CLEANED_DIR, ANALYSIS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── 1. download ────────────────────────────────────────────────────────────
def download_dataset() -> pd.DataFrame:
    print("Downloading Bitext Customer Support dataset...")
    ds = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        split="train",
    )
    df = ds.to_pandas()
    df.to_parquet(RAW_DIR / "raw_dataset.parquet", index=False)
    print(f"  Downloaded {len(df):,} rows | columns: {list(df.columns)}")
    return df


# ── 2. explore (EDA) ───────────────────────────────────────────────────────
def explore_dataset(df: pd.DataFrame) -> dict:
    """Run basic EDA and save charts to data/analysis/."""
    print("\n── EDA ──────────────────────────────────────────────")
    print(f"Shape         : {df.shape}")
    print(f"Null counts   :\n{df.isnull().sum()}")
    print(f"\nIntent counts (top 10):\n{df['intent'].value_counts().head(10)}")
    print(f"\nCategory distribution:\n{df['category'].value_counts()}")

    # Quality flags breakdown
    # flags field: each char is a flag. Common: B=basic, I=irrelevant,
    # K=keyword-stuffed, Z=zero-shot, Q=question form, P=politeness
    flag_counts: Counter = Counter()
    for flags_str in df["flags"].dropna():
        for char in str(flags_str):
            if char.strip():
                flag_counts[char] += 1
    print(f"\nQuality flags breakdown: {dict(flag_counts)}")

    stats = {
        "total_rows": len(df),
        "intents": df["intent"].nunique(),
        "categories": df["category"].nunique(),
        "null_counts": df.isnull().sum().to_dict(),
        "flag_distribution": dict(flag_counts),
        "avg_instruction_len": df["instruction"].str.len().mean(),
        "avg_response_len": df["response"].str.len().mean(),
    }

    # ── charts ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Bitext Customer Support Dataset — EDA", fontsize=14, fontweight="bold")

    # intent distribution
    intent_counts = df["intent"].value_counts()
    axes[0, 0].barh(intent_counts.index[:15], intent_counts.values[:15])
    axes[0, 0].set_title("Top 15 Intents")
    axes[0, 0].set_xlabel("Count")

    # category distribution
    cat_counts = df["category"].value_counts()
    axes[0, 1].pie(cat_counts.values, labels=cat_counts.index, autopct="%1.1f%%")
    axes[0, 1].set_title("Category Distribution")

    # instruction length distribution
    axes[1, 0].hist(df["instruction"].str.len(), bins=50, edgecolor="black")
    axes[1, 0].set_title("Instruction Length Distribution")
    axes[1, 0].set_xlabel("Characters")

    # response length distribution
    axes[1, 1].hist(df["response"].str.len(), bins=50, edgecolor="black", color="orange")
    axes[1, 1].set_title("Response Length Distribution")
    axes[1, 1].set_xlabel("Characters")

    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "eda_overview.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved EDA chart → {ANALYSIS_DIR / 'eda_overview.png'}")

    with open(ANALYSIS_DIR / "eda_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return stats


# ── 3. clean ───────────────────────────────────────────────────────────────
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning decisions (document every choice — recruiters love this):
      1. Drop rows missing instruction or response
      2. Remove rows with response < 20 chars (truncated/useless)
      3. Remove rows with instruction < 5 chars
      4. Strip leading/trailing whitespace
      5. Normalize whitespace inside text
      6. Flag rows with quality issues (don't drop — keep for analysis)
      7. Remove duplicate (instruction, response) pairs
    """
    print("\n── Cleaning ─────────────────────────────────────────")
    original_len = len(df)

    # step 1: drop nulls in key columns
    df = df.dropna(subset=["instruction", "response"])
    print(f"  After null drop     : {len(df):,} rows ({original_len - len(df)} removed)")

    # step 2 & 3: length filters
    df = df[df["response"].str.len() >= 20]
    df = df[df["instruction"].str.len() >= 5]
    print(f"  After length filter : {len(df):,} rows")

    # step 4 & 5: whitespace
    df["instruction"] = df["instruction"].str.strip().str.replace(r"\s+", " ", regex=True)
    df["response"] = df["response"].str.replace(r"\s+", " ", regex=True).str.strip()

    # step 6: quality flag column (B=basic responses, keep but mark)
    df["has_quality_flag"] = df["flags"].apply(
        lambda x: bool(re.search(r"[BIK]", str(x)))
    )
    print(f"  Rows with quality flags: {df['has_quality_flag'].sum():,}")

    # step 7: deduplicate
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["instruction", "response"])
    print(f"  After dedup         : {len(df):,} rows ({before_dedup - len(df)} dupes removed)")

    # reset index
    df = df.reset_index(drop=True)
    print(f"\n  Total removed: {original_len - len(df):,} rows ({(original_len - len(df)) / original_len * 100:.1f}%)")
    return df


# ── 4. format for instruction fine-tuning ─────────────────────────────────
SYSTEM_PROMPT = (
    "You are a helpful, professional customer support agent. "
    "Respond clearly and empathetically to customer inquiries. "
    "Be concise, accurate, and solution-focused."
)


def format_for_training(row: dict) -> dict:
    """
    Format using ChatML / Llama-3 instruct template.
    The model sees: system + user instruction → assistant response
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row["instruction"]},
        {"role": "assistant", "content": row["response"]},
    ]
    return {
        "messages": messages,
        "instruction": row["instruction"],
        "response": row["response"],
        "intent": row["intent"],
        "category": row["category"],
        "has_quality_flag": row["has_quality_flag"],
    }


# ── 5. split ───────────────────────────────────────────────────────────────
def split_and_save(df: pd.DataFrame) -> DatasetDict:
    """
    Stratified split by intent to ensure all intents appear in every split.
    Train: 80% | Val: 10% | Test: 10%
    """
    print("\n── Splitting ────────────────────────────────────────")

    # stratified split
    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["intent"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["intent"], random_state=42
    )

    print(f"  Train : {len(train_df):,}")
    print(f"  Val   : {len(val_df):,}")
    print(f"  Test  : {len(test_df):,}")

    # format all splits
    splits = {"train": train_df, "validation": val_df, "test": test_df}
    formatted = {}
    for split_name, split_df in splits.items():
        records = [format_for_training(row) for row in split_df.to_dict("records")]
        formatted[split_name] = Dataset.from_list(records)

    dataset_dict = DatasetDict(formatted)

    # save to disk
    dataset_dict.save_to_disk(str(CLEANED_DIR / "customer_support_dataset"))
    print(f"\nSaved to {CLEANED_DIR / 'customer_support_dataset'}")

    # also save test set as parquet for evaluation scripts
    test_df.to_parquet(CLEANED_DIR / "test_set.parquet", index=False)

    return dataset_dict


# ── main ───────────────────────────────────────────────────────────────────
def main():
    df_raw = download_dataset()
    stats = explore_dataset(df_raw)
    df_clean = clean_dataset(df_raw)
    dataset = split_and_save(df_clean)

    print("\n── Summary ──────────────────────────────────────────")
    print(f"  Raw rows        : {stats['total_rows']:,}")
    print(f"  Intents         : {stats['intents']}")
    print(f"  Categories      : {stats['categories']}")
    print(f"  After cleaning  : {len(df_clean):,}")
    print("\nData pipeline complete. Run experiments/run_experiment.py next.")


if __name__ == "__main__":
    main()
