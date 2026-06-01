"""
Run this once locally to generate examples.json for the HuggingFace Space:
    python deployment/prepare_examples.py

Reads evaluation/results/r64/ parquets and picks 2 examples per intent
(54 total across 27 intents) — diverse, never repeats.
"""
import json
from pathlib import Path

import pandas as pd

BASE_PATH = Path("evaluation/results/r64/base_predictions.parquet")
FT_PATH   = Path("evaluation/results/r64/finetuned_predictions.parquet")

for p in [BASE_PATH, FT_PATH]:
    if not p.exists():
        raise SystemExit(f"Missing: {p}\nDownload predictions from Colab first.")

base_df = pd.read_parquet(BASE_PATH)
ft_df   = pd.read_parquet(FT_PATH)

df = base_df.copy()
df["base_prediction"] = base_df["prediction"]
df["ft_prediction"]   = ft_df["prediction"]

examples = []
for intent, group in df.groupby("intent"):
    sample = group.sample(n=min(2, len(group)), random_state=42)
    for _, row in sample.iterrows():
        examples.append({
            "instruction":     str(row["instruction"]),
            "base_prediction": str(row["base_prediction"]),
            "ft_prediction":   str(row["ft_prediction"]),
            "response":        str(row["response"]),
            "intent":          str(row["intent"]),
            "category":        str(row["category"]),
        })

out = Path("deployment/examples.json")
out.write_text(json.dumps(examples, indent=2, ensure_ascii=False))
print(f"Saved {len(examples)} examples across {df['intent'].nunique()} intents → {out}")
