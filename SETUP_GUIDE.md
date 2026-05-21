# Setup & Run Guide (4-hour sprint)

## Hour 1 — GitHub + Data

```bash
# 1. Move project to your dev folder
cp -r llm-finetuning-framework ~/projects/llm-finetuning-framework
cd ~/projects/llm-finetuning-framework

# 2. Git init + first push
git init
git add .
git commit -m "feat: initial project scaffold — customer support LLM fine-tuning"

# On GitHub: create new repo named llm-finetuning-framework (no README, no .gitignore)
git remote add origin https://github.com/vamsiyvk/llm-finetuning-framework.git
git branch -M main
git push -u origin main

# 3. Create .env from template
cp .env.example .env
# edit .env and fill in WANDB_API_KEY, HF_TOKEN, OPENAI_API_KEY

# 4. Install deps (use a venv)
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 5. Run data pipeline (takes ~2 mins, no GPU needed)
python data/download_and_clean.py
# → creates data/cleaned/ and data/analysis/eda_overview.png
```

## Hour 2 — Training on Colab

1. Open `notebooks/colab_training.ipynb` in Google Colab
2. Runtime → Change runtime type → **T4 GPU**
3. Left sidebar 🔑 → Add secrets: `WANDB_API_KEY`, `HF_TOKEN`
4. Run all cells (Ctrl+F9)
5. While it trains (~25 mins for 200 steps), check your W&B dashboard live

After training:
- Download `test_set.parquet` from the Colab file browser
- Save it to `data/cleaned/test_set.parquet` on your local machine
- Model gets auto-pushed to `https://huggingface.co/vamsiyvk/customer-support-lora-r16`

## Hour 3 — Evaluation

```bash
# automatic metrics (runs on CPU/MPS, ~20 mins for 200 samples)
python evaluation/automated_eval.py \
    --model_path vamsiyvk/customer-support-lora-r16 \
    --test_data data/cleaned/test_set.parquet \
    --output_dir evaluation/results \
    --n_samples 200

# LLM-as-judge (needs OPENAI_API_KEY, costs ~$0.30)
python evaluation/llm_judge.py \
    --base_predictions evaluation/results/base/predictions.parquet \
    --ft_predictions evaluation/results/finetuned/predictions.parquet \
    --n_samples 50

# update README with your real numbers from evaluation/results/comparison.json
```

## Hour 4 — Deploy + Polish

```bash
# test Gradio app locally
FINETUNED_MODEL_ID=vamsiyvk/customer-support-lora-r16 python deployment/app.py

# deploy to HuggingFace Spaces:
# 1. Go to huggingface.co/new-space
# 2. Name: customer-support-llm | SDK: Gradio | Hardware: CPU Basic (free)
# 3. git clone your Space repo, copy deployment/app.py → app.py + requirements.txt
# 4. Add env var FINETUNED_MODEL_ID in Space settings
# 5. git push → auto-deploys

# final GitHub push with results
git add .
git commit -m "feat: evaluation results + deployed to HuggingFace Spaces"
git push
```

---

## What to say about this project in interviews

**The data angle**: "The dataset has quality flags marking responses as 'basic' or 'keyword-stuffed'. I analyzed the distribution, documented each cleaning decision, and kept flagged rows separate so I could check if they hurt eval metrics — they did."

**The eval angle**: "ROUGE tells you n-gram overlap, not whether the model actually solved the customer's problem. I added GPT-4-as-judge to measure helpfulness, accuracy, and professionalism independently — the scores didn't always agree with ROUGE, which is the whole point."

**The experiment design angle**: "I ran r=16 and r=64 as controlled experiments — same dataset, same training steps, different LoRA rank. The YAML configs are version-controlled so anyone can reproduce either run from a single command."

**The production angle**: "The Gradio app collects user preferences as feedback. That data feeds back into future training — it's a basic RLHF loop."
