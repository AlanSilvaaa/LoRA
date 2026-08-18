import csv
from datetime import datetime, timezone
from pathlib import Path

import typer
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from config import DECODING_CONFIG, MODEL_ID, TESTING_PROMPS
from helpers.env_utils import load_repo_env

app = typer.Typer()
RESULTS_PATH = Path("results_skill.csv")
RESULTS_FIELDNAMES = ["datetime", "skill_path", "model_id", "question", "answer"]


def write_skill_results_csv(
    results: list[dict[str, str]], skill_path: Path
) -> Path:
    file_exists = RESULTS_PATH.exists()
    executed_at = datetime.now(timezone.utc).isoformat()
    rows = [
        {
            "datetime": executed_at,
            "skill_path": str(skill_path),
            "model_id": MODEL_ID,
            "question": result["question"],
            "answer": result["answer"],
        }
        for result in results
    ]

    with RESULTS_PATH.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=RESULTS_FIELDNAMES,
            lineterminator="\n",
            quoting=csv.QUOTE_ALL,
        )
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    return RESULTS_PATH


class VLLMQuestionRunner:
    def __init__(self, skill: str):
        load_repo_env()
        self.skill = skill

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        print("Loading vLLM engine...")
        self.llm = LLM(
            model=MODEL_ID,
            dtype="bfloat16",
        )
        self.sampling_params = SamplingParams(
            max_tokens=DECODING_CONFIG["max_new_tokens"],
            temperature=DECODING_CONFIG["temperature"],
            top_p=DECODING_CONFIG["top_p"],
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.llm.llm_engine.engine_core.shutdown()

    def run_question(self, question: str) -> dict[str, str]:
        question = "\n".join(
            line.strip() for line in question.replace("\\n", "\n").splitlines()
        )
        question_with_skill = f"{self.skill}\n\n{question}"
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question_with_skill}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        print("\nGenerating with BASE model...")
        outputs = self.llm.generate([prompt], self.sampling_params)
        response = outputs[0].outputs[0].text

        print("\n" + "=" * 50)
        print(f"BASE {MODEL_ID} OUTPUT:")
        print("=" * 50)
        print(response)

        return {
            "question": question,
            "answer": response,
        }


def run_question_vllm(question: str, skill: str) -> dict[str, str]:
    with VLLMQuestionRunner(skill) as runner:
        return runner.run_question(question)


def run_configured_prompts(skill: str) -> list[dict[str, str]]:
    with VLLMQuestionRunner(skill) as runner:
        results = [runner.run_question(prompt) for prompt in TESTING_PROMPS]

    return results


@app.command()
def main(
    question: str | None = typer.Argument(None, help="Question to send to the model"),
    skill: Path = typer.Option(
        ...,
        "-skill",
        "--skill",
        help="Path to the Markdown skill file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
):
    skill_content = skill.read_text(encoding="utf-8")
    if question is None:
        results = run_configured_prompts(skill_content)
    else:
        results = [run_question_vllm(question, skill_content)]

    csv_path = write_skill_results_csv(results, skill)
    print(f"Saved evaluation results to {csv_path}")


if __name__ == "__main__":
    app()
