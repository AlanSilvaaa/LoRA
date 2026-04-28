import torch
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

from huggingface_hub import login
from config import MODEL_ID, LORA_DIR

# Log in using the Colab Secret
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
    return f"<start_of_turn>user\n{example['question']}<end_of_turn>\n<start_of_turn>model\n{example['answer']}<end_of_turn>"

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
    save_strategy="epoch",
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
trainer.train()

print(f"Saving LoRA adapter to {LORA_DIR}")
trainer.model.save_pretrained(LORA_DIR)
tokenizer.save_pretrained(LORA_DIR)
print("Done!")
