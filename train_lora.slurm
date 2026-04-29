#!/bin/bash

#SBATCH --job-name=lora-train
#SBATCH --partition=L40
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=output/slurm-%x-%j.out
#SBATCH --error=output/slurm-%x-%j.err
#SBATCH --requeue

set -euo pipefail

CONTAINER_IMAGE="${CONTAINER_IMAGE:-nvcr.io/nvidia/pytorch:25.01-py3}"
CONTAINER_NAME="${CONTAINER_NAME:-lora}"
WORKDIR="${SLURM_SUBMIT_DIR:-$PWD}"
ENV_FILE="${WORKDIR}/.env"

if [[ -f "${ENV_FILE}" ]]; then
    set -a
    # Load repo-local secrets for the batch job.
    source "${ENV_FILE}"
    set +a
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is not set. Add it to ${ENV_FILE}."
    exit 1
fi

echo "Job ID: ${SLURM_JOB_ID}"
echo "Working directory: ${WORKDIR}"
echo "Container image: ${CONTAINER_IMAGE}"
date

srun \
    --container-workdir="${WORKDIR}" \
    --container-image="${CONTAINER_IMAGE}" \
    --container-name="${CONTAINER_NAME}" \
    --export=ALL,HF_TOKEN="${HF_TOKEN}" \
    bash -lc '
        set -euo pipefail
        python -c "import datasets, transformers, peft, trl, huggingface_hub, accelerate" \
            || python -m pip install --user datasets transformers peft trl huggingface_hub accelerate
        python -u train.py
    '
