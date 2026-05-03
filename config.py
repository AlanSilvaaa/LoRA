MODEL_ID = "google/gemma-3-270m-it"
LORA_DIR = "./checkpoints/gemma-3-270m-it-gsm8k-lora"

# MODEL_ID = "google/gemma-3-1b-it"
# LORA_DIR = "./checkpoints/gemma-3-1b-it-gsm8k-lora"

# MODEL_ID = "google/gemma-3-12b-it"
# LORA_DIR = "./checkpoints/gemma-3-12b-it-gsm8k-lora"


TESTING_PROMPS = [
    "Give me a math problem that involves addition and apples for kids of 1st grade. Also solve the problem.",
    "Give me a math problem that involves multiplication and oranges for kids of 2nd grade. Also solve the problem.",
    "If i have 3 boxes of chocolates and each box has 5 chocolates, how many chocolates do I have in total?",
]

LORA_CONFIG = {
    "r": 16,
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
