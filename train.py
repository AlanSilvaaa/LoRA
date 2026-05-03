import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer

from huggingface_hub import login
from config import LORA_CONFIG, LORA_DIR, MODEL_ID, TRAINING_CONFIG
from helpers.env_utils import load_repo_env, normalize_single_gpu_slurm_env
from helpers.test_overfitting import measure_overfitting


def _saved_lora_adapter_exists() -> bool:
    lora_path = Path(LORA_DIR)
    return (lora_path / "adapter_model.safetensors").exists() or (lora_path / "adapter_model.bin").exists()


def main():
    load_repo_env()
    normalize_single_gpu_slurm_env()

    # Log in using HF_TOKEN from the environment or repo-local .env.
    login(token=os.environ.get("HF_TOKEN"))

    dataset = load_dataset("openai/gsm8k", "main")
    train_validation_split = dataset["train"].train_test_split(test_size=0.1, seed=42)
    train_dataset = train_validation_split["train"]
    validation_dataset = train_validation_split["test"]

    print("Loading model in bfloat16...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
    )
    adapter_already_saved = _saved_lora_adapter_exists()
    if adapter_already_saved:
        print(f"Found saved LoRA adapter in {LORA_DIR}; skipping training.")
        model = PeftModel.from_pretrained(model, LORA_DIR, is_trainable=False)

    def format_instruction(example):
        messages = [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    lora_config = None if adapter_already_saved else LoraConfig(**LORA_CONFIG)

    training_args = SFTConfig(
        output_dir=LORA_DIR,
        **TRAINING_CONFIG,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        peft_config=lora_config,
        formatting_func=format_instruction,
        processing_class=tokenizer,
        args=training_args,
    )

    if not adapter_already_saved:
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

    print("Measuring overfitting...")
    overfitting_metrics = measure_overfitting(trainer=trainer)
    print("Done!")
    return overfitting_metrics


if __name__ == "__main__":
    main()
