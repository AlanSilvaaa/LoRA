MODEL_ID = "HuggingFaceTB/SmolLM3-3B"
LORA_DIR = "./checkpoints/smollm3-3b/spanish-math-generation-r8"
DATASET_PATH = "./datasets/arma-tu-evaluacion-generation.json"
DATASET_NAME = "arma-tu-evaluacion-generation"

TESTING_PROMPS = [
    "Crea una pregunta original de suma para 1° Básico, con respuesta y explicación.",
    "Crea una pregunta de selección múltiple sobre multiplicación para 3° Básico.",
    "Crea una pregunta de respuesta abierta sobre fracciones para 4° Básico, con dificultad media.",
    "Crea una pregunta de geometría para 5° Básico, con respuesta y explicación para el docente.",
    "Crea una pregunta de datos y probabilidades para 6° Básico, con dificultad profunda.",
    "Crea una evaluación de Matemática para 2° Básico con 3 preguntas y una pauta de respuestas.",
    "Crea una evaluación de Matemática para 6° Básico con 5 preguntas variadas y una pauta de respuestas.",
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
    "warmup_steps": 10,
    "eval_strategy": "steps",
    "eval_steps": 25,
    "save_strategy": "steps",
    "save_steps": 25,
    "save_total_limit": 5,
    "logging_steps": 10,
    "num_train_epochs": 3,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "bf16": True,
}

DECODING_CONFIG = {
    "max_new_tokens": 400,
    "do_sample": True,
    "temperature": 0.8,
    "top_p": 0.9,
}

TRAIN_EVAL_SAMPLE_SIZE = 512
