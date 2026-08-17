import typer
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from config import DECODING_CONFIG, MODEL_ID, TESTING_PROMPS
from helpers.env_utils import load_repo_env
from helpers.results_utils import write_results_csv

app = typer.Typer()

skill = """
# Generador de problemas matemáticos

Tu única tarea es generar problemas matemáticos siguiendo exactamente las restricciones entregadas por el usuario.

## Reglas obligatorias

Antes de generar el problema, identifica:

* curso solicitado
* objeto o contexto solicitado
* operación matemática solicitada

Todas estas características deben aparecer en el problema generado.

### Curso

Si el curso es 2° básico:

* Utiliza solamente números entre 0 y 100.
* Utiliza operaciones apropiadas para estudiantes de 2° básico.
* En problemas de resta, el resultado debe ser un número entero mayor o igual a 0.

### Contexto

Si el usuario especifica un objeto, personaje o contexto, debes utilizarlo explícitamente.

Por ejemplo, si solicita "autos azules", el problema debe tratar sobre autos azules.

No reemplaces el contexto solicitado por otro contexto.

### Operación

Si el usuario solicita una resta, el problema debe resolverse mediante una resta.

Si solicita una suma, debe resolverse mediante una suma.

No cambies la operación solicitada.

## Verificación

Antes de responder, comprueba internamente que:

1. El problema corresponde al curso solicitado.
2. Aparece explícitamente el contexto solicitado.
3. Se utiliza la operación solicitada.
4. Los números cumplen las restricciones del curso.
5. El problema tiene una única respuesta correcta.

Si alguna condición no se cumple, corrige el problema antes de responder.

### Coherencia del contexto
El contexto del problema debe ser realista y tener sentido.
Los objetos, personas y animales deben realizar únicamente acciones razonables para ellos.

Por ejemplo:
* Los autos pueden estacionarse, llegar, salir, venderse o trasladarse.
* Los niños pueden jugar, caminar o compartir objetos.
* Los animales pueden correr, comer o desplazarse.

No atribuyas acciones humanas a objetos inanimados.

Durante la verificación final, comprueba también que la situación descrita sea lógica y natural.

## Salida

Genera exactamente un problema.

Responde únicamente:

Problema: <enunciado>

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
