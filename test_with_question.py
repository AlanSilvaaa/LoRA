import os
import torch
import typer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import login
from config import DECODING_CONFIG, LORA_DIR, MODEL_ID
from helpers.env_utils import load_repo_env

app = typer.Typer()


def run_question(question: str) -> dict[str, str]:
    load_repo_env()
    if os.environ.get("HF_TOKEN"):
        login(token=os.environ.get("HF_TOKEN"))

    print("Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )

    # Load base model in bfloat16 to fit in memory.
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Tokenize the input.
    inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)

    # Test the base model.
    print("\nGenerating with BASE model (this might take a few seconds)...")
    base_outputs = base_model.generate(**inputs, **DECODING_CONFIG)
    base_response = tokenizer.decode(base_outputs[0], skip_special_tokens=True)

    # Inject the LoRA adapter.
    print("\nInjecting LoRA adapter...")
    finetuned_model = PeftModel.from_pretrained(base_model, LORA_DIR)

    # Test the fine-tuned model.
    print("Generating with FINE-TUNED model...")
    ft_outputs = finetuned_model.generate(**inputs, **DECODING_CONFIG)
    ft_response = tokenizer.decode(ft_outputs[0], skip_special_tokens=True)

    # Print results.
    print("\n" + "=" * 50)
    print(f"BASE {MODEL_ID} OUTPUT:")
    print("=" * 50)
    print(base_response)

    print("\n" + "=" * 50)
    print(f"LORA FINE-TUNED OUTPUT ({LORA_DIR}):")
    print("=" * 50)
    print(ft_response)

    return {
        "prompt": question,
        "base_output": base_response,
        "finetuned_output": ft_response,
    }


@app.command()
def main(question: str = typer.Argument(..., help="Question to send to the model")):
    run_question(question)


if __name__ == "__main__":
    app()
