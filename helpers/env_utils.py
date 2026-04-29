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


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def normalize_single_gpu_slurm_env() -> None:
    # Running plain `python train.py` inside an interactive SLURM allocation can
    # expose rank metadata without a full distributed rendezvous configuration.
    # For a single-task job, clear those hints so Accelerate stays in single-GPU mode.
    slurm_ntasks = _int_env("SLURM_NTASKS", 1)
    world_size = max(
        _int_env("WORLD_SIZE", 1),
        _int_env("LOCAL_WORLD_SIZE", 1),
        _int_env("PMI_SIZE", 1),
        _int_env("OMPI_COMM_WORLD_SIZE", 1),
        _int_env("MV2_COMM_WORLD_SIZE", 1),
    )

    if slurm_ntasks > 1 or world_size > 1:
        return

    for key in (
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "WORLD_SIZE",
        "SLURM_PROCID",
        "SLURM_LOCALID",
        "SLURM_NODEID",
        "SLURM_NTASKS",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "NODE_RANK",
        "PMI_RANK",
        "PMI_SIZE",
        "PMI_LOCAL_RANK",
        "OMPI_COMM_WORLD_RANK",
        "OMPI_COMM_WORLD_SIZE",
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MV2_COMM_WORLD_RANK",
        "MV2_COMM_WORLD_SIZE",
        "MV2_COMM_WORLD_LOCAL_RANK",
    ):
        os.environ.pop(key, None)
