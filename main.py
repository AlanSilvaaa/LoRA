from config import LORA_DIR, MODEL_ID, TESTING_PROMPS
from helpers.results_utils import write_results_csv
from test_with_question import run_question
from train import main as train_main


def main():
    print("Starting LoRA training run...")
    train_main()

    print("Running configured evaluation prompts...")
    rows = []
    for prompt in TESTING_PROMPS:
        result = run_question(prompt)
        rows.append(
            {
                "prompt": result["prompt"],
                "model_id": MODEL_ID,
                "lora_dir": LORA_DIR,
                "base_output": result["base_output"],
                "finetuned_output": result["finetuned_output"],
            }
        )

    csv_path = write_results_csv(rows)
    print(f"Saved evaluation results to {csv_path}")


if __name__ == "__main__":
    main()
