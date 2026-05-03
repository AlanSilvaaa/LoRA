from config import DECODING_CONFIG, LORA_CONFIG, LORA_DIR, MODEL_ID, TESTING_PROMPS, TRAINING_CONFIG
from helpers.results_utils import write_results_csv
from test_with_question import run_question
from train import main as train_main


def main():
    print("Starting LoRA training run...")
    overfitting_metrics = train_main()

    print("Running configured evaluation prompts...")
    rows = []
    for prompt in TESTING_PROMPS:
        result = run_question(prompt)
        rows.append(
            {
                "prompt": result["prompt"],
                "model_id": MODEL_ID,
                "lora_dir": LORA_DIR,
                "lora_r": LORA_CONFIG["r"],
                "lora_alpha": LORA_CONFIG["lora_alpha"],
                "lora_dropout": LORA_CONFIG["lora_dropout"],
                "learning_rate": TRAINING_CONFIG["learning_rate"],
                "num_train_epochs": TRAINING_CONFIG["num_train_epochs"],
                "per_device_train_batch_size": TRAINING_CONFIG["per_device_train_batch_size"],
                "gradient_accumulation_steps": TRAINING_CONFIG["gradient_accumulation_steps"],
                "max_new_tokens": DECODING_CONFIG["max_new_tokens"],
                **overfitting_metrics,
                "base_output": result["base_output"],
                "finetuned_output": result["finetuned_output"],
            }
        )

    csv_path = write_results_csv(rows)
    print(f"Saved evaluation results to {csv_path}")


if __name__ == "__main__":
    main()
