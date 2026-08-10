import typer
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from config import DECODING_CONFIG, MODEL_ID, TESTING_PROMPS
from helpers.env_utils import load_repo_env
from helpers.results_utils import write_results_csv

app = typer.Typer()

skill = """
# Skill: Generador de Problemas Matemáticos Chilenos

## Objetivo

Generar problemas matemáticos breves para estudiantes de educación básica en Chile, respetando el curso, contenido y objetivo de aprendizaje indicados por el usuario.

## Instrucciones

Cuando el usuario solicite generar un problema matemático:

1. Identifica:

   - Curso.
   - Contenido matemático.
   - Objetivo de aprendizaje, si fue entregado.
   - Dificultad, si fue entregada.

2. Genera exactamente **un problema matemático**.

3. El problema debe:

   - Ser apropiado para la edad y curso indicado.
   - Poder resolverse únicamente mediante texto.
   - No depender de imágenes, gráficos, tablas ni diagramas.
   - Tener toda la información necesaria para resolverlo.
   - Tener una respuesta matemática clara y única.
   - Usar números adecuados al nivel del estudiante.
   - Evitar operaciones o conceptos que excedan el nivel solicitado.
   - Usar lenguaje simple y natural.
   - Preferir contextos cotidianos comprensibles para estudiantes en Chile.
   - Evitar información innecesaria.

4. No entregues la solución a menos que el usuario la solicite explícitamente.

## Formato de salida

Entrega solamente:

**Problema:** \<enunciado>

No agregues explicaciones sobre cómo fue generado.

## Ejemplo

Solicitud:

"Genera un problema para 3° básico sobre suma hasta 1000."

Respuesta:

**Problema:** En una biblioteca había 326 libros de cuentos. La escuela recibió 248 libros nuevos. ¿Cuántos libros de cuentos hay ahora en la biblioteca?

Ahora responderás la pregunta de un usuario sobre generar un problema de matemáticas:
"""


class VLLMQuestionRunner:
    def __init__(self):
        load_repo_env()

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
        question_with_skill = f"{skill}\n\n{question}"
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
            "prompt": question,
            "base_output": response,
        }


def run_question_vllm(question: str) -> dict[str, str]:
    with VLLMQuestionRunner() as runner:
        return runner.run_question(question)


def run_configured_prompts() -> list[dict[str, str]]:
    with VLLMQuestionRunner() as runner:
        results = [runner.run_question(prompt) for prompt in TESTING_PROMPS]

    rows = [
        {
            "prompt": result["prompt"],
            "model_id": MODEL_ID,
            "max_new_tokens": DECODING_CONFIG["max_new_tokens"],
            "base_output": result["base_output"],
        }
        for result in results
    ]
    csv_path = write_results_csv(rows)
    print(f"Saved evaluation results to {csv_path}")
    return results


@app.command()
def main(
    question: str | None = typer.Argument(None, help="Question to send to the model"),
):
    if question is None:
        run_configured_prompts()
    else:
        run_question_vllm(question)


if __name__ == "__main__":
    app()
