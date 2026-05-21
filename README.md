# LLM Fine-Tuning Framework: Customer Support Agent

Fine-tuning **Llama 3.2 3B** on 27,000 real customer support conversations, with a full evaluation pipeline comparing base vs. fine-tuned models across automatic metrics and GPT-4 judgment.

**Live demo** → [HuggingFace Spaces](https://huggingface.co/spaces/vamsiyvk/customer-support-llm)  
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

*(Fill these in after running experiments. Replace TBD with your actual numbers.)*

| Model | Perplexity ↓ | ROUGE-L ↑ | BERTScore F1 ↑ | GPT-4 Score ↑ | Avg Latency (ms) ↓ |
|---|---|---|---|---|---|
| Base Llama 3.2-3B | TBD | TBD | TBD | TBD | TBD |
| Fine-tuned LoRA r=16 | TBD | TBD | TBD | TBD | TBD |
| Fine-tuned LoRA r=64 | TBD | TBD | TBD | TBD | TBD |

**GPT-4 judge dimensions** (each 1-5): helpfulness, accuracy, professionalism  
**Win rate**: fine-tuned model preference in head-to-head comparison = TBD%

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/vamsiyvk/llm-finetuning-framework
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

### 4. Evaluation (run locally on M2/CPU/CUDA)

```bash
# automatic metrics
python evaluation/automated_eval.py \
    --model_path ./outputs/customer-support-lora-r16 \
    --test_data data/cleaned/test_set.parquet \
    --n_samples 200

# LLM-as-judge (needs OPENAI_API_KEY, costs ~$0.30)
python evaluation/llm_judge.py \
    --base_predictions evaluation/results/base/predictions.parquet \
    --ft_predictions evaluation/results/finetuned/predictions.parquet \
    --n_samples 50
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
| LLM judge | OpenAI GPT-4o-mini |
| Deployment | Gradio + HuggingFace Spaces |

---

## Author

Vamsi YVK — [GitHub](https://github.com/vamsiyvk) | [HuggingFace](https://huggingface.co/vamsiyvk) | [LinkedIn](https://linkedin.com/in/vamsiyvk)
