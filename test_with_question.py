import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Paths from your training script
model_id = "google/gemma-2b"
adapter_dir = "./gemma-gsm8k-standard-lora"

# 1. Formulate the question using the EXACT formatting from training
question = "A university casino digital menu system processes 15 QR code payments every hour. If the casino is open for 4 hours, and each meal costs 3500 pesos, what is the total revenue generated through the QR system"

# Notice how we stop right after 'model\n' so the AI knows it's its turn to talk
prompt = f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"

print("Loading base model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Load base model in bfloat16 to fit in memory
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 2. Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)

# 3. Test the Base Model
print("\nGenerating with BASE model (this might take a few seconds)...")
base_outputs = base_model.generate(**inputs, max_new_tokens=150)
base_response = tokenizer.decode(base_outputs[0], skip_special_tokens=True)

# 4. Inject the LoRA Adapter
print("\nInjecting LoRA adapter...")
finetuned_model = PeftModel.from_pretrained(base_model, adapter_dir)

# 5. Test the Fine-Tuned Model
print("Generating with FINE-TUNED model...")
ft_outputs = finetuned_model.generate(**inputs, max_new_tokens=150)
ft_response = tokenizer.decode(ft_outputs[0], skip_special_tokens=True)

# 6. Print Results
print("\n" + "="*50)
print("🧐 BASE GEMMA-2B OUTPUT:")
print("="*50)
print(base_response)

print("\n" + "="*50)
print("🚀 LORA FINE-TUNED OUTPUT (500 steps):")
print("="*50)
print(ft_response)
