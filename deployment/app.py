"""
Gradio Comparison App — deploy to HuggingFace Spaces.

Features:
  - Side-by-side: Base Llama 3.2 3B vs Fine-tuned (customer support)
  - Live latency display per response
  - User feedback collection (thumbs up/down + free text)
  - Intent display from classification
  - Example prompts from each category

Deploy to HuggingFace Spaces:
    1. Create a new Space (Gradio type)
    2. Push this file + requirements.txt to the Space repo
    3. Set env vars: HF_TOKEN, FINETUNED_MODEL_ID
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

# ── config ────────────────────────────────────────────────────────────────
BASE_MODEL_ID = "unsloth/Llama-3.2-3B-Instruct"
FINETUNED_MODEL_ID = os.getenv(
    "FINETUNED_MODEL_ID", "vamsiyvk/customer-support-lora-r16"
)
FEEDBACK_FILE = "feedback_log.jsonl"
MAX_NEW_TOKENS = 256

SYSTEM_PROMPT = (
    "You are a helpful, professional customer support agent. "
    "Respond clearly and empathetically to customer inquiries. "
    "Be concise, accurate, and solution-focused."
)

EXAMPLE_PROMPTS = [
    "I was charged twice for my last order. Can you help?",
    "How do I return a product I bought last week?",
    "My order hasn't arrived and it's been 2 weeks. Where is it?",
    "I want to cancel my subscription but can't find the option.",
    "Can I change the shipping address for an order I just placed?",
    "My account got locked after too many login attempts. What do I do?",
    "I received the wrong item in my order.",
    "Do you offer price matching if I find a lower price elsewhere?",
]

# ── model loading ─────────────────────────────────────────────────────────
def detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_id: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model = model.to(device)
    model.eval()
    return model, tokenizer


print(f"Loading models... (this takes 1-2 minutes on first launch)")
DEVICE = detect_device()
print(f"Device: {DEVICE}")

base_model, base_tokenizer = load_model(BASE_MODEL_ID, DEVICE)
ft_model, ft_tokenizer = load_model(FINETUNED_MODEL_ID, DEVICE)
print("Models loaded.")


# ── generation ────────────────────────────────────────────────────────────
def generate(model, tokenizer, instruction: str) -> tuple[str, float]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt = f"{SYSTEM_PROMPT}\n\nUser: {instruction}\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency_ms = (time.perf_counter() - start) * 1000
    response = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return response.strip(), round(latency_ms, 1)


# ── feedback ──────────────────────────────────────────────────────────────
def log_feedback(
    instruction: str,
    base_resp: str,
    ft_resp: str,
    preference: str,
    comment: str,
):
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "instruction": instruction,
        "base_response": base_resp,
        "finetuned_response": ft_resp,
        "preference": preference,
        "comment": comment,
    }
    with open(FEEDBACK_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    return "✅ Feedback recorded. Thank you!"


# ── gradio interface ──────────────────────────────────────────────────────
def compare(instruction: str):
    if not instruction.strip():
        return "Please enter a question.", "", "", ""

    base_resp, base_lat = generate(base_model, base_tokenizer, instruction)
    ft_resp, ft_lat = generate(ft_model, ft_tokenizer, instruction)

    base_with_meta = f"{base_resp}\n\n---\n⏱ {base_lat} ms"
    ft_with_meta = f"{ft_resp}\n\n---\n⏱ {ft_lat} ms"

    return base_resp, ft_resp, base_with_meta, ft_with_meta


with gr.Blocks(
    title="LLM Fine-Tuning: Customer Support Comparison",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        """
        # 🤖 LLM Fine-Tuning for Customer Support
        **Base Llama 3.2-3B-Instruct** vs **Fine-tuned on 27K real customer support conversations**

        Type a customer question and see how each model responds. Fine-tuned model was trained using LoRA
        on the [Bitext Customer Support Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset).
        """
    )

    with gr.Row():
        instruction_box = gr.Textbox(
            label="Customer Question",
            placeholder="e.g. I was charged twice for my order. Can you help?",
            lines=2,
            scale=4,
        )
        submit_btn = gr.Button("Compare →", variant="primary", scale=1)

    gr.Examples(
        examples=[[p] for p in EXAMPLE_PROMPTS],
        inputs=instruction_box,
        label="Example questions (click to load)",
    )

    with gr.Row():
        base_output = gr.Textbox(
            label="Base Llama 3.2-3B (no fine-tuning)",
            lines=8,
            interactive=False,
        )
        ft_output = gr.Textbox(
            label="Fine-tuned (Customer Support LoRA r=16)",
            lines=8,
            interactive=False,
        )

    # hidden state for feedback
    base_state = gr.State("")
    ft_state = gr.State("")

    def on_compare(instruction):
        base_resp, ft_resp, base_disp, ft_disp = compare(instruction)
        return base_disp, ft_disp, base_resp, ft_resp

    submit_btn.click(
        fn=on_compare,
        inputs=instruction_box,
        outputs=[base_output, ft_output, base_state, ft_state],
    )
    instruction_box.submit(
        fn=on_compare,
        inputs=instruction_box,
        outputs=[base_output, ft_output, base_state, ft_state],
    )

    gr.Markdown("### Which response was better?")
    with gr.Row():
        pref_base = gr.Button("👍 Base model was better")
        pref_ft = gr.Button("👍 Fine-tuned was better")
        pref_tie = gr.Button("🤝 About the same")

    feedback_comment = gr.Textbox(
        label="Any comments? (optional)",
        placeholder="What was better or worse about either response?",
    )
    feedback_status = gr.Markdown("")

    def make_feedback_fn(choice):
        def fn(instruction, base_resp, ft_resp, comment):
            msg = log_feedback(instruction, base_resp, ft_resp, choice, comment)
            return msg
        return fn

    pref_base.click(
        fn=make_feedback_fn("base"),
        inputs=[instruction_box, base_state, ft_state, feedback_comment],
        outputs=feedback_status,
    )
    pref_ft.click(
        fn=make_feedback_fn("finetuned"),
        inputs=[instruction_box, base_state, ft_state, feedback_comment],
        outputs=feedback_status,
    )
    pref_tie.click(
        fn=make_feedback_fn("tie"),
        inputs=[instruction_box, base_state, ft_state, feedback_comment],
        outputs=feedback_status,
    )

    gr.Markdown(
        """
        ---
        **Project**: [GitHub](https://github.com/vamsiyvk/llm-finetuning-framework) |
        **Author**: Vamsi YVK |
        **Model trained on**: Bitext Customer Support Dataset (27K conversations, 27 intents)
        """
    )


if __name__ == "__main__":
    demo.launch(share=False)
