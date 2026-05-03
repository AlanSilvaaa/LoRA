import math
from typing import Any

from config import TRAIN_EVAL_SAMPLE_SIZE


def _overfit_level(loss_gap: float) -> str:
    if loss_gap < 0.2:
        return "low"
    if loss_gap <= 0.5:
        return "moderate"
    return "high"



def measure_overfitting(
    trainer: Any,
    train_dataset: Any,
    validation_dataset: Any,
    train_eval_sample_size: int = TRAIN_EVAL_SAMPLE_SIZE,
) -> dict[str, str | float | int]:
    """
    Measure overfitting by comparing the loss on a sample of the training data to the loss on the validation data.
    Args:
        trainer: The SFTTrainer instance used for training.
        train_dataset: The training dataset.
        validation_dataset: The validation dataset.
        train_eval_sample_size: The number of samples from the training dataset to use for evaluation.
    Returns:
        A dictionary containing the training evaluation loss, validation loss, loss gap, overfitting level, and other relevant metrics.
    """
    sample_size = min(train_eval_sample_size, len(train_dataset))
    train_eval_dataset = train_dataset.select(range(sample_size))

    train_metrics = trainer.evaluate(
        eval_dataset=train_eval_dataset,
        metric_key_prefix="train_eval",
    )
    validation_metrics = trainer.evaluate(
        eval_dataset=validation_dataset,
        metric_key_prefix="validation",
    )

    train_eval_loss = train_metrics["train_eval_loss"]
    validation_loss = validation_metrics["validation_loss"]
    loss_gap = validation_loss - train_eval_loss

    return {
        "train_eval_loss": train_eval_loss,
        "validation_loss": validation_loss,
        "overfit_loss_gap": loss_gap,
        "overfit_ppl_ratio": math.exp(loss_gap),
        "overfit_level": _overfit_level(loss_gap),
        "overfit_eval_dataset": "gsm8k_train_validation_split",
        "train_eval_sample_size": sample_size,
    }
