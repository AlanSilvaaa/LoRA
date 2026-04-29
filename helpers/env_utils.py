import os
from pathlib import Path


def load_repo_env(env_path: str = ".env") -> None:
    path = Path(env_path)
    if not path.exists():
        return

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


def normalize_single_gpu_slurm_env() -> None:
    # Running plain `python train.py` inside an interactive SLURM allocation can
    # expose rank metadata without a full distributed rendezvous configuration.
    # For a single-task job, clear those hints so Accelerate stays in single-GPU mode.
    if os.environ.get("MASTER_ADDR") or os.environ.get("WORLD_SIZE") or os.environ.get("RANK"):
        return

    if os.environ.get("SLURM_NTASKS", "1") != "1":
        return

    for key in (
        "SLURM_PROCID",
        "SLURM_LOCALID",
        "SLURM_NODEID",
        "SLURM_NTASKS",
        "LOCAL_RANK",
    ):
        os.environ.pop(key, None)
