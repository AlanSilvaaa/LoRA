MODEL_ID = "HuggingFaceTB/SmolLM3-3B"
LORA_DIR = "./checkpoints/smollm3-3b/spanish-math-r8"
DATASET_PATH = "./datasets/math-word-problems-spanish-kaggle.json"
DATASET_NAME = "math-word-problems-spanish-kaggle"

TESTING_PROMPS = [
    "Si tengo 3 cajas de chocolates y cada caja tiene 5 chocolates, ¿cuántos chocolates tengo en total?",
    "Hay 24 manzanas y se reparten por igual entre 6 niños. ¿Cuántas manzanas recibe cada niño?",
    "Lucía tenía 18 lápices, regaló 7 y después compró 5. ¿Cuántos lápices tiene ahora?",
    "Una entrada cuesta 12 pesos. ¿Cuánto cuestan 8 entradas?",
    "Pedro recorrió 2,5 kilómetros el lunes y 3,75 kilómetros el martes. ¿Cuántos kilómetros recorrió en total?",
    "Crea un problema matemático en español cuya respuesta sea 24 y resuélvelo.",
    "Crea y resuelve un problema de porcentajes sobre un descuento para un estudiante de sexto grado.",
]

LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 32,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

TRAINING_CONFIG = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "optim": "adamw_torch",
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 50,
    "eval_strategy": "steps",
    "eval_steps": 100,
    "save_strategy": "steps",
    "save_steps": 100,
    "save_total_limit": 5,
    "logging_steps": 10,
    "num_train_epochs": 3,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "bf16": True,
}

DECODING_CONFIG = {
    "max_new_tokens": 150,
}

TRAIN_EVAL_SAMPLE_SIZE = 512
