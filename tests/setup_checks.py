import importlib
import os
import py_compile
import re
import subprocess
from pathlib import Path

from helpers.env_utils import load_repo_env


ROOT = Path(__file__).resolve().parent.parent
REQUIRED_PACKAGES = {
    "accelerate",
    "datasets",
    "huggingface-hub",
    "peft",
    "torch",
    "transformers",
    "trl",
    "typer",
    "vllm",
}


class CheckResult:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def passed(self, message: str) -> None:
        print(f"PASS: {message}")

    def error(self, message: str) -> None:
        self.errors.append(message)
        print(f"ERROR: {message}")

    def warning(self, message: str) -> None:
        self.warnings.append(message)
        print(f"WARN: {message}")


def check_python_syntax(report: CheckResult) -> None:
    python_files = sorted(ROOT.glob("*.py"))
    python_files += sorted((ROOT / "helpers").glob("*.py"))
    python_files += sorted((ROOT / "tests").glob("*.py"))
    failures = []
    for path in python_files:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as error:
            failures.append(f"{path.relative_to(ROOT)}: {error.msg}")

    if failures:
        for failure in failures:
            report.error(f"Python syntax failure in {failure}")
    else:
        report.passed(f"Python syntax ({len(python_files)} files)")


def check_requirements(report: CheckResult) -> None:
    path = ROOT / "requirements.txt"
    if not path.is_file():
        report.error("requirements.txt does not exist")
        return

    initial_errors = len(report.errors)
    packages: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s=]+)", line)
        if not match:
            report.error(f"requirements.txt:{line_number} is not an exact package pin: {line}")
            continue
        name = match.group(1).lower().replace("_", "-")
        if name in packages:
            report.error(f"requirements.txt pins {name} more than once")
        packages[name] = match.group(2)

    missing = sorted(REQUIRED_PACKAGES - packages.keys())
    if missing:
        report.error(f"requirements.txt is missing direct dependencies: {', '.join(missing)}")
    elif len(report.errors) == initial_errors:
        report.passed(f"Pinned requirements ({len(packages)} packages)")


def check_config(report: CheckResult) -> None:
    try:
        config = importlib.import_module("config")
    except Exception as error:
        report.error(f"Cannot import config.py: {error}")
        return

    required_names = (
        "MODEL_ID",
        "LORA_DIR",
        "TESTING_PROMPS",
        "LORA_CONFIG",
        "TRAINING_CONFIG",
    )
    missing_names = [name for name in required_names if not hasattr(config, name)]
    if missing_names:
        report.error(f"config.py is missing values: {', '.join(missing_names)}")
        return

    initial_errors = len(report.errors)
    if not isinstance(config.MODEL_ID, str) or not config.MODEL_ID.strip():
        report.error("MODEL_ID must be a non-empty string")
    if not isinstance(config.LORA_DIR, str) or not config.LORA_DIR.strip():
        report.error("LORA_DIR must be a non-empty string")
    if not isinstance(config.TESTING_PROMPS, list) or not config.TESTING_PROMPS or not all(
        isinstance(prompt, str) and prompt.strip() for prompt in config.TESTING_PROMPS
    ):
        report.error("TESTING_PROMPS must contain non-empty strings")

    rank = config.LORA_CONFIG.get("r")
    alpha = config.LORA_CONFIG.get("lora_alpha")
    targets = config.LORA_CONFIG.get("target_modules")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        report.error("LORA_CONFIG.r must be a positive integer")
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or alpha <= 0:
        report.error("LORA_CONFIG.lora_alpha must be positive")
    if not isinstance(targets, list) or not targets or not all(isinstance(item, str) and item for item in targets):
        report.error("LORA_CONFIG.target_modules must contain module names")
    elif len(targets) != len(set(targets)):
        report.error("LORA_CONFIG.target_modules contains duplicates")

    for key in ("per_device_train_batch_size", "gradient_accumulation_steps", "num_train_epochs"):
        value = config.TRAINING_CONFIG.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            report.error(f"TRAINING_CONFIG.{key} must be positive")

    for name, value in vars(config).items():
        if not name.isupper() or not isinstance(value, str):
            continue
        if name.endswith("_PATH"):
            path = (ROOT / value).resolve()
            if not path.exists():
                report.error(f"Configured path does not exist: {name}={value}")

    if len(report.errors) == initial_errors:
        report.passed(f"Configuration ({config.MODEL_ID})")


def check_slurm(report: CheckResult) -> None:
    path = ROOT / "train_and_test_lora.slurm"
    if not path.is_file():
        report.error("train_and_test_lora.slurm does not exist")
        return
    try:
        result = subprocess.run(
            ["bash", "-n", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        report.error(f"Cannot run bash syntax check: {error}")
        return
    if result.returncode:
        report.error(f"SLURM script has invalid shell syntax: {result.stderr.strip()}")
    else:
        report.passed("SLURM shell syntax")


def check_token(report: CheckResult, require_token: bool) -> None:
    load_repo_env(str(ROOT / ".env"))
    token = os.environ.get("HF_TOKEN", "").strip()
    valid = token and token != "hf_your_token_here"
    if valid:
        report.passed("HF_TOKEN is configured")
    elif require_token:
        report.error("HF_TOKEN is not configured in the environment or .env")
    else:
        report.warning("HF_TOKEN is not configured; the SLURM job will require it")


def check_setup(require_token: bool = False) -> int:
    report = CheckResult()
    check_python_syntax(report)
    check_requirements(report)
    check_config(report)
    check_slurm(report)
    check_token(report, require_token)

    print()
    if report.errors:
        print(f"Setup checks failed with {len(report.errors)} error(s) and {len(report.warnings)} warning(s).")
        return 1
    print(f"Setup checks passed with {len(report.warnings)} warning(s).")
    return 0

