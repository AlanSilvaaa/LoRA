import typer
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from config import DECODING_CONFIG, LORA_CONFIG, LORA_DIR, MODEL_ID
from helpers.env_utils import load_repo_env

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
        )
        self.lora_request = LoRARequest("math-lora", 1, LORA_DIR)

    def run_question(self, question: str) -> dict[str, str]:
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )

        print("\nGenerating with BASE model...")
        base_outputs = self.llm.generate([prompt], self.sampling_params)
        base_response = base_outputs[0].outputs[0].text

        print("Generating with LoRA adapter...")
        lora_outputs = self.llm.generate(
            [prompt],
            self.sampling_params,
            lora_request=self.lora_request,
        )
        lora_response = lora_outputs[0].outputs[0].text

        print("\n" + "=" * 50)
        print(f"BASE {MODEL_ID} OUTPUT:")
        print("=" * 50)
        print(base_response)

        print("\n" + "=" * 50)
        print(f"VLLM LORA OUTPUT ({LORA_DIR}):")
        print("=" * 50)
        print(lora_response)

        return {
            "prompt": question,
            "base_output": base_response,
            "finetuned_output": lora_response,
        }


def run_question_vllm(question: str) -> dict[str, str]:
    runner = VLLMQuestionRunner()
    return runner.run_question(question)


@app.command()
def main(question: str = typer.Argument(..., help="Question to send to the model")):
    run_question_vllm(question)


if __name__ == "__main__":
    app()
