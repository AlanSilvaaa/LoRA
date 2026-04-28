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

# Log in using the Colab Secret
login(token=os.environ.get("HF_TOKEN"))

model_id = "google/gemma-2b"
output_dir = "./gemma-gsm8k-standard-lora"
dataset = load_dataset("openai/gsm8k", "main")

print("Loading model in bfloat16...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
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
    output_dir=output_dir,
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

print(f"Saving LoRA adapter to {output_dir}")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("Done!")

