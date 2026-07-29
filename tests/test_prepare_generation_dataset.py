import unittest

from utilities.prepare_generation_dataset import _generation_instruction, _generation_response


class GenerationDatasetTests(unittest.TestCase):
    def test_formats_multiple_choice_question_and_answer(self):
        record = {
            "question": "¿Cuánto es 2 + 2?",
            "question_type": "cerrada",
            "choices": [{"label": "A", "text": "3"}, {"label": "B", "text": "4"}],
            "correct_answer": "B",
            "explanation": "2 + 2 = 4.",
            "grades": ["1° Básico"],
            "topics": ["Números y operaciones"],
            "learning_objectives": ["MA01 OA 09"],
            "difficulty": {"label": "Medio"},
        }

        response = _generation_response(record)

        self.assertIn("Alternativas:\nA) 3\nB) 4", response)
        self.assertIn("Respuesta correcta: B) 4", response)
        self.assertIn("Explicación: 2 + 2 = 4.", response)

    def test_treats_closed_question_without_choices_as_open(self):
        record = {
            "question": "Calcula el perímetro.",
            "question_type": "cerrada",
            "choices": [],
            "correct_answer": None,
            "explanation": "El perímetro es 16 cm.",
            "grades": ["4° Básico"],
            "topics": ["Medición"],
            "learning_objectives": ["MA04 OA 21"],
            "difficulty": {"label": "Medio"},
        }

        self.assertIn("Formato: respuesta abierta", _generation_instruction(record))
        self.assertEqual(
            _generation_response(record),
            "Pregunta: Calcula el perímetro.\n\nSolución: El perímetro es 16 cm.",
        )

    def test_skips_multiple_choice_question_with_missing_correct_option(self):
        record = {
            "question": "Elige una alternativa.",
            "question_type": "cerrada",
            "choices": [{"label": "A", "text": "1"}],
            "correct_answer": "D",
            "explanation": "4",
        }

        self.assertIsNone(_generation_response(record))

    def test_corrects_open_label_on_multiple_choice_question(self):
        record = {
            "question": "1 050 en palabras es:",
            "question_type": "abierta",
            "choices": [
                {"label": "A", "text": "mil quinientos"},
                {"label": "B", "text": "mil cincuenta"},
            ],
            "correct_answer": "B",
            "explanation": "",
            "grades": ["4° Básico"],
            "topics": ["Números y operaciones"],
            "learning_objectives": ["MA04 OA 01"],
            "difficulty": {"label": "Superficial"},
        }

        self.assertIn("Formato: selección múltiple", _generation_instruction(record))
        self.assertIn("Respuesta correcta: B) mil cincuenta", _generation_response(record))


if __name__ == "__main__":
    unittest.main()
