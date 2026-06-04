# LLM Fine-Tuning Framework: Customer Support Agent

Fine-tuning **Llama 3.2 3B** on 27,000 real customer support conversations, with a full evaluation pipeline comparing base vs. fine-tuned models across automatic metrics and GPT-4 judgment.

**Live demo** → [HuggingFace Spaces](https://huggingface.co/spaces/vamsiyvk/customer-support-AI) 
**W&B experiment tracker** → [wandb.ai/vamsiyvk/llm-finetuning-customer-support](https://wandb.ai)

---

## Why this project

Every company with a support team is trying to automate first-response handling. The hard part isn't fine-tuning a model — it's knowing whether it actually got better. This project treats evaluation as a first-class problem: five different measurement axes, LLM-as-judge scoring, and a side-by-side UI so you can see the difference yourself.

The dataset is intentionally messy. The `flags` column in the Bitext dataset marks quality issues (basic responses, keyword stuffing, irrelevant patterns) — exactly the kind of thing you'd find in a production CS ticket dump. The data pipeline documents every cleaning decision, which is what you'd actually do before handing this to stakeholders.

---

## Dataset

**[Bitext Customer Support LLM Training Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)**

| Property | Value |
|---|---|
| Rows | 26,872 |
| Intents | 27 (billing, returns, cancellation, etc.) |
| Categories | 11 business domains |
| Quality flags | B (basic), I (irrelevant), K (keyword-stuffed) |
| Split | 80 / 10 / 10 (stratified by intent) |

---

## Architecture

```
data/
├── download_and_clean.py   ← EDA + cleaning pipeline
├── raw/                    ← original parquet
└── cleaned/                ← formatted splits (HuggingFace DatasetDict)

experiments/
├── configs/
│   ├── lora_r16.yaml       ← baseline experiment
│   └── lora_r64.yaml       ← comparison (higher capacity)
└── run_experiment.py       ← training orchestrator + W&B logging

evaluation/
├── automated_eval.py       ← Perplexity, ROUGE-L, BLEU, BERTScore
└── llm_judge.py            ← GPT-4o-mini rates each response (1-5)

deployment/
└── app.py                  ← Gradio: side-by-side comparison + feedback

notebooks/
└── colab_training.ipynb    ← step-by-step Colab notebook (T4 GPU)
```

---

## Results

*100 test samples, 200 training steps, LLM judge: Llama 3.3-70B via Groq (free).*

### Automatic Metrics

| Model | ROUGE-L ↑ | BLEU ↑ | ROUGE-1 ↑ | ROUGE-2 ↑ | Avg Pred Len | Avg Latency |
|---|---|---|---|---|---|---|
| Base Llama 3.2-3B | 0.2276 | 0.0831 | 0.3916 | 0.1157 | 105 words | 5,353 ms |
| Fine-tuned LoRA **r=16** | **0.3554** | **0.2292** | **0.5053** | **0.2438** | 87 words | 5,353 ms |
| Fine-tuned LoRA **r=64** | **0.3847** | **0.2705** | **0.5283** | **0.2838** | 91 words | 6,441 ms |

r=16 vs base: BLEU +176%, ROUGE-L +56%  
r=64 vs base: BLEU +205%, ROUGE-L +70%

### LLM-as-Judge (Llama 3.3-70B, scores 1–5)

| Dimension | Base | r=16 | r=64 |
|---|---|---|---|
| Helpfulness | 3.88–4.10 | 4.20 | 4.12 |
| Accuracy | 4.18–4.30 | 4.30 | 4.08 |
| Professionalism | 5.00 | 5.00 | 5.00 |
| **Composite** | 4.35–4.47 | **4.50** | **4.40** |

**Head-to-head win rate:** r=16 → FT 15%, Base 85% &nbsp;|&nbsp; r=64 → FT 44%, Base 56%

### What the numbers mean

The ROUGE/BLEU gap is large because fine-tuning teaches the *specific phrasing* of Bitext support responses ("I'm sorry to hear that", "please allow 3–5 business days"). The base model gives correct but differently-worded answers — valid, but n-gram metrics penalise the wording mismatch.

The LLM judge composites are close (both fine-tuned variants score ~4.4–4.5/5) because Llama 3.2-3B Instruct is already a strong instruction-following model. Both variants produce professional, empathetic responses. The win rate gap between r=16 (15%) and r=64 (44%) shows that **higher-rank LoRA produces noticeably more preferred responses** — r=64 is nearly competitive head-to-head — while r=16 mostly narrows the style gap rather than the quality gap. Longer training (500–1000 steps) would push both further.

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/vamsiy2001/Customer_Support_AI-Fine_Tuning_Evaluation_Pipeline
cd llm-finetuning-framework
pip install -r requirements.txt
cp .env.example .env  # fill in your keys
```

### 2. Data pipeline (run locally — no GPU needed)

```bash
python data/download_and_clean.py
# outputs: data/cleaned/customer_support_dataset + EDA charts
```

### 3. Training (Google Colab — free T4 GPU)

Open `notebooks/colab_training.ipynb` in Colab, set your W&B + HF secrets in the Secrets panel, then run all cells. Training 200 steps takes ~25 minutes on T4.

Or use the CLI (requires CUDA):
```bash
python experiments/run_experiment.py --config experiments/configs/lora_r16.yaml
python experiments/run_experiment.py --config experiments/configs/lora_r64.yaml
```

### 4. Evaluation (run locally — no GPU needed)

After downloading the parquets from Colab (cell 13), place them at `evaluation/results/r16/` and run:

```bash
# automatic metrics (ROUGE, BLEU — CPU only)
python evaluation/compute_metrics.py --experiment r16
python evaluation/compute_metrics.py --experiment r64

# LLM-as-judge (needs GROQ_API_KEY — free at console.groq.com)
python evaluation/llm_judge.py --experiment r16
python evaluation/llm_judge.py --experiment r64
```

### 5. Run the comparison app

```bash
FINETUNED_MODEL_ID=vamsiyvk/customer-support-lora-r16 python deployment/app.py
```

---

## Key design decisions

**Why Bitext over synthetic datasets?** Real customer conversations have noise — inconsistent phrasing, varying formality, quality flags. Cleaning this teaches more than working with a pre-sanitized dataset and gives you something honest to say about your data pipeline in interviews.

**Why evaluate with LLM-as-judge?** ROUGE and BLEU measure n-gram overlap, not whether the response actually solved the customer's problem. A model that says "I can help with your billing issue" scores zero ROUGE against a reference that says "Your account will be credited within 3-5 business days" — even though one is useless and one is correct. GPT-4 catches this.

**Why LoRA r=16 vs r=64?** Higher rank means more parameters updated — in theory more capacity, but also higher risk of overfitting on a small dataset. The comparison tells you whether the extra parameters help or hurt on this specific task.

**Why stratified splits?** Without stratification, some intents (especially rare ones) may not appear in the test set at all, making your evaluation miss entire failure modes.

---

## Skills demonstrated

- Parameter-efficient fine-tuning (LoRA, QLoRA via Unsloth)
- Experiment tracking with Weights & Biases
- Multi-dimensional evaluation: automatic metrics + LLM-as-judge
- Data cleaning with documented decisions (the `flags` column analysis)
- Deployment on HuggingFace Spaces with user feedback collection
- Hyperparameter comparison with controlled experiments
- Apple Silicon / MPS inference optimization

---

## Stack

| Component | Tool |
|---|---|
| Training | Unsloth + LoRA + TRL SFTTrainer |
| GPU | Google Colab T4 (free tier) |
| Local inference | PyTorch MPS (Apple M-series) |
| Experiment tracking | Weights & Biases |
| Automatic evaluation | ROUGE, BLEU, BERTScore, evaluate library |
| LLM judge | Groq (Llama 3.3 70B) — free tier |
| Deployment | Gradio + HuggingFace Spaces |

---

## Author

Vamsi YVK — [GitHub](https://github.com/vamsiy2001) | [HuggingFace](https://huggingface.co/vamsiyvk) | [LinkedIn](https://linkedin.com/in/vamsiyvk)
