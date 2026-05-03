import csv
from datetime import datetime, timezone
from pathlib import Path


FIELDNAMES = [
    "executed_at",
    "prompt",
    "model_id",
    "lora_dir",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "learning_rate",
    "num_train_epochs",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "max_new_tokens",
    "train_eval_loss",
    "validation_loss",
    "overfit_loss_gap",
    "overfit_ppl_ratio",
    "overfit_level",
    "overfit_eval_dataset",
    "train_eval_sample_size",
    "base_output",
    "finetuned_output",
]


def _migrate_header_if_needed(csv_path: Path) -> bool:
    if not csv_path.exists():
        return False

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames == FIELDNAMES:
            return True
        existing_rows = list(reader)

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=FIELDNAMES,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(existing_rows)

    return True


def write_results_csv(rows: list[dict[str, str]]) -> Path:
    csv_path = Path("results.csv")
    file_exists = _migrate_header_if_needed(csv_path)
    executed_at = datetime.now(timezone.utc).isoformat()

    enriched_rows = []
    for row in rows:
        enriched_row = {"executed_at": executed_at}
        enriched_row.update(row)
        enriched_rows.append(enriched_row)

    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=FIELDNAMES,
            extrasaction="ignore",
        )
        if not file_exists:
            writer.writeheader()
        writer.writerows(enriched_rows)

    return csv_path
