import typer
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from config import (
    DECODING_CONFIG,
    LORA_CONFIG,
    LORA_DIR,
    MODEL_ID,
    TESTING_PROMPS,
    TRAINING_CONFIG,
)
from helpers.env_utils import load_repo_env
from helpers.results_utils import write_results_csv

app = typer.Typer()


class VLLMQuestionRunner:
    def __init__(self):
        load_repo_env()

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        print("Loading vLLM engine...")
        self.llm = LLM(
            model=MODEL_ID,
            dtype="bfloat16",
            enable_lora=True,
            max_lora_rank=LORA_CONFIG["r"],
        )
        self.sampling_params = SamplingParams(
            max_tokens=DECODING_CONFIG["max_new_tokens"],
            temperature=DECODING_CONFIG["temperature"],
            top_p=DECODING_CONFIG["top_p"],
        )
        self.lora_request = LoRARequest("math-lora", 1, LORA_DIR)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.llm.llm_engine.engine_core.shutdown()

    def run_question(
        self,
        question: str,
        run_original: bool = True,
        run_finetuned: bool = True,
    ) -> dict[str, str]:
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        base_response = ""
        if run_original:
            print("\nGenerating with BASE model...")
            base_outputs = self.llm.generate([prompt], self.sampling_params)
            base_response = base_outputs[0].outputs[0].text

        lora_response = ""
        if run_finetuned:
            print("Generating with LoRA adapter...")
            lora_outputs = self.llm.generate(
                [prompt],
                self.sampling_params,
                lora_request=self.lora_request,
            )
            lora_response = lora_outputs[0].outputs[0].text

        if run_original:
            print("\n" + "=" * 50)
            print(f"BASE {MODEL_ID} OUTPUT:")
            print("=" * 50)
            print(base_response)

        if run_finetuned:
            print("\n" + "=" * 50)
            print(f"VLLM LORA OUTPUT ({LORA_DIR}):")
            print("=" * 50)
            print(lora_response)

        return {
            "prompt": question,
            "base_output": base_response,
            "finetuned_output": lora_response,
        }


def run_question_vllm(
    question: str,
    run_original: bool = True,
    run_finetuned: bool = True,
) -> dict[str, str]:
    with VLLMQuestionRunner() as runner:
        return runner.run_question(question, run_original, run_finetuned)


def run_configured_prompts(
    run_original: bool = True,
    run_finetuned: bool = True,
) -> list[dict[str, str]]:
    with VLLMQuestionRunner() as runner:
        results = [
            runner.run_question(prompt, run_original, run_finetuned)
            for prompt in TESTING_PROMPS
        ]

    rows = [
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
            "base_output": result["base_output"],
            "finetuned_output": result["finetuned_output"],
        }
        for result in results
    ]
    csv_path = write_results_csv(rows)
    print(f"Saved evaluation results to {csv_path}")
    return results


@app.command()
def main(
    original: bool = typer.Option(False, "--original", help="Run the original model"),
    finetunned: bool = typer.Option(False, "--finetunned", help="Run the fine-tuned model"),
    question: str | None = typer.Argument(None, help="Question to send to the model"),
):
    run_original = original or not (original or finetunned)
    run_finetuned = finetunned or not (original or finetunned)

    if question is None:
        run_configured_prompts(run_original, run_finetuned)
    else:
        run_question_vllm(question, run_original, run_finetuned)


if __name__ == "__main__":
    app()
