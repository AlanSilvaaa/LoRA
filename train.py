import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

from huggingface_hub import login
from config import MODEL_ID, LORA_DIR
from helpers.env_utils import load_repo_env

load_repo_env()

# Log in using HF_TOKEN from the environment or repo-local .env.
login(token=os.environ.get("HF_TOKEN"))

dataset = load_dataset("openai/gsm8k", "main")

print("Loading model in bfloat16...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map={"": 0}
)

def format_instruction(example):
    messages = [
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

training_args = SFTConfig(
    output_dir=LORA_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    optim="adamw_torch",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    save_strategy="steps",
    save_steps=25,
    save_total_limit=2,
    logging_steps=10,
    num_train_epochs=1,
    max_steps=100, # Initial test, later increase this
    bf16=True,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    peft_config=lora_config,
    formatting_func=format_instruction,
    args=training_args,
)

print("Starting training...")
last_checkpoint = get_last_checkpoint(LORA_DIR)

if last_checkpoint:
    print(f"Resuming training from checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    trainer.train()

print(f"Saving LoRA adapter to {LORA_DIR}")
trainer.model.save_pretrained(LORA_DIR)
tokenizer.save_pretrained(LORA_DIR)
print("Done!")
