"""
Customer Support LLM — HuggingFace Spaces demo.
Shows pre-computed predictions (base vs fine-tuned) side by side.
No GPU or model loading required — instant response.

Deploy: push app.py + examples.json + requirements.txt to an HF Space (Gradio SDK).
Generate examples.json locally first:
    python deployment/prepare_examples.py
"""

import json
import random
from pathlib import Path

import gradio as gr

# ── load examples ──────────────────────────────────────────────────────────
def _load_examples():
    for p in ["examples.json", "deployment/examples.json"]:
        if Path(p).exists():
            with open(p) as f:
                return json.load(f)
    return []

EXAMPLES  = _load_examples()
INTENTS   = ["All intents"] + sorted({e["intent"]   for e in EXAMPLES})
CATEGORIES = sorted({e["category"] for e in EXAMPLES})

# ── helpers ────────────────────────────────────────────────────────────────
def pick(intent_filter: str):
    pool = EXAMPLES if intent_filter == "All intents" else [
        e for e in EXAMPLES if e["intent"] == intent_filter
    ]
    if not pool:
        return ("", "", "", "", "")
    ex = random.choice(pool)
    badge = f"**Intent:** `{ex['intent']}`  ·  **Category:** `{ex['category']}`"
    return (
        ex["instruction"],
        ex["base_prediction"],
        ex["ft_prediction"],
        ex["response"],
        badge,
    )

# ── CSS ────────────────────────────────────────────────────────────────────
CSS = """
/* question */
#question textarea {
    border-left: 4px solid #3b82f6 !important;
    background: #eff6ff !important;
    font-size: 1.05em !important;
}
/* base model */
#base-out textarea {
    border-left: 4px solid #94a3b8 !important;
    background: #f8fafc !important;
}
/* fine-tuned */
#ft-out textarea {
    border-left: 4px solid #22c55e !important;
    background: #f0fdf4 !important;
}
/* reference */
#ref-out textarea {
    border-left: 4px solid #f59e0b !important;
    background: #fffbeb !important;
    font-size: 0.92em !important;
}
.badge { font-size: 0.9em; color: #475569; margin-top: 4px; }
"""

# ── metrics markdown ───────────────────────────────────────────────────────
METRICS_MD = """
## Automatic Metrics (100 test samples)

| Metric | Base Llama 3.2-3B | Fine-tuned r=16 | Fine-tuned r=64 |
|:---|---:|---:|---:|
| BLEU ↑ | 0.0831 | 0.2292 · **+176%** | **0.2705 · +205%** |
| ROUGE-L ↑ | 0.2276 | 0.3554 · **+56%** | **0.3847 · +70%** |
| ROUGE-1 ↑ | 0.3916 | 0.5053 · **+29%** | **0.5283 · +35%** |
| ROUGE-2 ↑ | 0.1157 | 0.2438 · **+111%** | **0.2838 · +145%** |
| Avg response length | 105 words | 87 words | 91 words |

## LLM-as-Judge (50 samples · Llama 3.3-70B via Groq · free)

| Dimension (1–5) | Base | r=16 | r=64 |
|:---|---:|---:|---:|
| Helpfulness ↑ | 3.88–4.10 | **4.20** | 4.12 |
| Accuracy ↑ | 4.18–4.30 | **4.30** | 4.08 |
| Professionalism ↑ | 5.00 | 5.00 | 5.00 |
| **Composite** ↑ | 4.35–4.47 | **4.50** | 4.40 |
| Head-to-head win rate | — | 15% | **44%** |

## What the numbers mean

- **BLEU / ROUGE gap is large** because fine-tuning teaches the model the exact phrasing of the
  training set ("please allow 3–5 business days", "I sincerely apologise"). The base model answers
  correctly but in different words — valid, but n-gram metrics penalise the mismatch.
- **Judge composite is close** because Llama 3.2-3B Instruct is already a strong instruction-following
  model. Both produce professional, empathetic responses. Fine-tuning at 200 steps closes the *style*
  gap more than the *quality* gap.
- **r=64 win rate (44%) >> r=16 (15%)**: higher LoRA rank learns more of the domain vocabulary and
  is nearly competitive head-to-head. 500+ training steps would likely push this above 50%.
"""

# ── about markdown ─────────────────────────────────────────────────────────
ABOUT_MD = """
## What was built

A full **LLM fine-tuning + evaluation pipeline** on a real customer support dataset:

```
Data pipeline  →  27K Bitext conversations, cleaned + stratified 80/10/10 split
Training       →  LoRA fine-tuning on Llama 3.2-3B (Unsloth + TRL, free Colab T4)
Evaluation     →  ROUGE/BLEU (CPU) + LLM-as-judge via Groq (free tier)
Demo           →  This Gradio app on HuggingFace Spaces
```

## Training details

| Setting | r=16 | r=64 |
|:---|:---|:---|
| Base model | Llama 3.2-3B Instruct | same |
| LoRA target modules | q/k/v/o · gate/up/down | same |
| LoRA alpha | 16 | 64 |
| Training steps | 200 | 200 |
| Effective batch size | 8 (2 × 4 grad accum) | same |
| Learning rate | 2e-4 cosine | 1e-4 cosine |
| Quantization | 4-bit QLoRA | same |
| GPU | Colab T4 (free) | same |
| Training time | ~30 min | ~30 min |

## Stack

| Layer | Tool |
|:---|:---|
| Fine-tuning | Unsloth + LoRA + TRL SFTTrainer |
| Experiment tracking | Weights & Biases |
| Automatic metrics | HuggingFace `evaluate` (ROUGE, BLEU) |
| LLM judge | Groq — Llama 3.3-70B (free tier) |
| Demo | Gradio + HuggingFace Spaces |

## Source

[GitHub →](https://github.com/vamsiy2001/Customer_Support_AI-Fine_Tuning_Evaluation_Pipeline)
"""

# ── build UI ───────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Customer Support LLM — Base vs Fine-tuned",
    theme=gr.themes.Soft(),
    css=CSS,
) as demo:

    gr.Markdown(
        """
# 🤖 Customer Support LLM — Base vs Fine-tuned
Fine-tuning **Llama 3.2-3B Instruct** on 27K real customer support conversations with **LoRA**.
Pre-computed predictions from 100 held-out test examples — instant, no GPU needed.
        """
    )

    with gr.Tabs():

        # ── Tab 1: side-by-side ────────────────────────────────────────────
        with gr.Tab("🔍 Side-by-Side Comparison"):

            with gr.Row():
                intent_dd = gr.Dropdown(
                    choices=INTENTS, value="All intents",
                    label="Filter by intent", scale=3,
                )
                next_btn = gr.Button("🎲 Next example", variant="primary", scale=1)

            badge_md = gr.Markdown("", elem_classes="badge")
            question = gr.Textbox(
                label="Customer question",
                lines=2, interactive=False, elem_id="question",
            )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### ⬜ Base Llama 3.2-3B &nbsp;*(no fine-tuning)*")
                    base_out = gr.Textbox(
                        label="", lines=8, interactive=False, elem_id="base-out",
                    )
                with gr.Column():
                    gr.Markdown("### 🟢 Fine-tuned r=64 &nbsp;*(LoRA, 200 steps)*")
                    ft_out = gr.Textbox(
                        label="", lines=8, interactive=False, elem_id="ft-out",
                    )

            with gr.Accordion("📖 Reference answer from dataset", open=False):
                ref_out = gr.Textbox(
                    label="", lines=3, interactive=False, elem_id="ref-out",
                )

            outputs = [question, base_out, ft_out, ref_out, badge_md]
            next_btn.click(pick, inputs=intent_dd, outputs=outputs)
            intent_dd.change(pick, inputs=intent_dd, outputs=outputs)
            demo.load(pick, inputs=intent_dd, outputs=outputs)

        # ── Tab 2: metrics ─────────────────────────────────────────────────
        with gr.Tab("📊 Evaluation Metrics"):
            gr.Markdown(METRICS_MD)

        # ── Tab 3: about ───────────────────────────────────────────────────
        with gr.Tab("ℹ️ About"):
            gr.Markdown(ABOUT_MD)

    gr.Markdown(
        "Built by **Vamsi YVK** · "
        "[GitHub](https://github.com/vamsiy2001/Customer_Support_AI-Fine_Tuning_Evaluation_Pipeline) · "
        "[Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)"
    )


if __name__ == "__main__":
    demo.launch()
