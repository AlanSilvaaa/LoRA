"""
Converts datasets/arma-tu-evaluacion-scrapped.json, a collection of scraped
MINEDUC mathematics questions and solutions, into instruction/response pairs
for training a model to create new questions instead of merely answering them.

For example, a source question for 1° Básico becomes an instruction such as
"Crea una pregunta original de Matemática para 1° Básico" and a response with
the generated-style question, its alternatives when applicable, and its
solution. The resulting records are written by default to
datasets/arma-tu-evaluacion-generation.json.
"""

import argparse
import json
import re
from pathlib import Path

DEFAULT_INPUT = Path("datasets/arma-tu-evaluacion-scrapped.json")
DEFAULT_OUTPUT = Path("datasets/arma-tu-evaluacion-generation.json")


def _question_type(record: dict) -> str:
    if record["question_type"] == "cerrada" and record["choices"]:
        return "cerrada"
    if record["question_type"] == "abierta" and record["choices"]:
        combined = f"{record['question']}\n{record.get('explanation') or ''}"
        has_blanks = "___" in combined or any(
            "___" in choice["text"] for choice in record["choices"]
        )
        has_subanswers = bool(
            re.search(r"(?:^|\n)B[.)]", record.get("explanation") or "")
        )
        if not has_blanks and not has_subanswers:
            return "cerrada"
    return "abierta"


def _generation_instruction(record: dict) -> str:
    grade = ", ".join(record["grades"])
    topic = ", ".join(record["topics"])
    objectives = ", ".join(record["learning_objectives"])
    difficulty = record["difficulty"]["label"].lower()
    question_format = (
        "selección múltiple"
        if _question_type(record) == "cerrada"
        else "respuesta abierta"
    )

    return (
        f"Crea una pregunta original de Matemática para {grade}.\n"
        "Requisitos:\n"
        f"- Eje: {topic}.\n"
        f"- Objetivo de aprendizaje: {objectives}.\n"
        f"- Dificultad: {difficulty}.\n"
        f"- Formato: {question_format}.\n"
        "Incluye la respuesta correcta y una explicación breve para el docente."
    )


def _question_with_supplements(record: dict) -> str:
    question = record["question"].strip()
    if _question_type(record) != "abierta" or not record["choices"]:
        return question

    # Some open exercises use the choices field for sub-exercises omitted from question.
    supplements = "\n".join(
        f"{choice['label']}. {choice['text']}" for choice in record["choices"]
    )
    return f"{question}\n{supplements}"


def _generation_response(record: dict) -> str | None:
    question = _question_with_supplements(record)
    explanation = (record.get("explanation") or "").strip()

    if _question_type(record) == "cerrada":
        if re.search(r"(?:^|\n)[A-E]\)", question):
            return None
        choices_by_label = {
            choice["label"]: choice["text"].strip() for choice in record["choices"]
        }
        correct_label = record.get("correct_answer")
        if correct_label not in choices_by_label:
            return None

        alternatives = "\n".join(
            f"{label}) {text}" for label, text in choices_by_label.items()
        )
        response = (
            f"Pregunta: {question}\n\n"
            f"Alternativas:\n{alternatives}\n\n"
            f"Respuesta correcta: {correct_label}) {choices_by_label[correct_label]}"
        )
    else:
        response = f"Pregunta: {question}"

    if explanation:
        label = "Explicación" if _question_type(record) == "cerrada" else "Solución"
        response += f"\n\n{label}: {explanation}"

    return response


def convert_dataset(input_path: Path, output_path: Path) -> tuple[int, int]:
    with input_path.open(encoding="utf-8") as source:
        records = json.load(source)

    converted = []
    skipped = 0
    for record in records:
        response = _generation_response(record)
        if response is None:
            skipped += 1
            continue

        converted.append(
            {
                "instruction": _generation_instruction(record),
                "response": response,
                "source_id": record["id"],
                "grade": record["grades"][0],
                "topic": record["topics"][0],
                "learning_objectives": record["learning_objectives"],
                "difficulty": record["difficulty"]["label"],
                "question_type": _question_type(record),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as destination:
        json.dump(converted, destination, ensure_ascii=False, indent=2)
        destination.write("\n")

    return len(converted), skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert scraped MINEDUC questions into generation training pairs."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    converted, skipped = convert_dataset(args.input, args.output)
    print(
        f"Wrote {converted} generation examples to {args.output} ({skipped} skipped)."
    )


if __name__ == "__main__":
    main()
