import csv
from datetime import datetime, timezone
from pathlib import Path


FIELDNAMES = [
    "executed_at",
    "prompt",
    "model_id",
    "lora_dir",
    "base_output",
    "finetuned_output",
]


def write_results_csv(rows: list[dict[str, str]]) -> Path:
    csv_path = Path("results.csv")
    file_exists = csv_path.exists()
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
        )
        if not file_exists:
            writer.writeheader()
        writer.writerows(enriched_rows)

    return csv_path
