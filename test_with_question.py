import torch
import typer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from config import MODEL_ID, LORA_DIR

app = typer.Typer()


@app.command()
def main(question: str = typer.Argument(..., help="Question to send to the model")):
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
    base_outputs = base_model.generate(**inputs, max_new_tokens=150)
    base_response = tokenizer.decode(base_outputs[0], skip_special_tokens=True)

    # Inject the LoRA adapter.
    print("\nInjecting LoRA adapter...")
    finetuned_model = PeftModel.from_pretrained(base_model, LORA_DIR)

    # Test the fine-tuned model.
    print("Generating with FINE-TUNED model...")
    ft_outputs = finetuned_model.generate(**inputs, max_new_tokens=150)
    ft_response = tokenizer.decode(ft_outputs[0], skip_special_tokens=True)

    # Print results.
    print("\n" + "=" * 50)
    print(f"🧐 BASE {MODEL_ID} OUTPUT:")
    print("=" * 50)
    print(base_response)

    print("\n" + "=" * 50)
    print(f"🚀 LORA FINE-TUNED OUTPUT ({LORA_DIR}):")
    print("=" * 50)
    print(ft_response)


if __name__ == "__main__":
    app()
