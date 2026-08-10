# LoRA

Experimental pipeline for fine-tuning language models with LoRA. It supports
configurable training, evaluation, and comparisons between base and fine-tuned
model outputs.

Runtime settings can be changed in `config.py` as different configurations are
evaluated.

## Tests

Run the CPU-only test suite without requesting cluster resources:

```bash
python tests/main.py
```

The entry point uses Python's `unittest` framework to discover and run every
`test_*.py` file under `tests/`. The current suite checks Python and SLURM
syntax, configuration values and paths, exact dependency pins, and Hugging Face
credentials. Missing credentials are a warning locally and an error in the
batch job.

Runtime dependencies are pinned in `requirements.txt`. The SLURM job installs
those exact versions before training.

## Run

### Interactive container on patagon

To test the code before sending it to the cluster as a job, you can run a container and execute the training script directly inside it. This also sends a job to the cluster, but it will be interactive, and you can see the output in real-time, allowing you to debug and make changes as needed.

```bash
srun --partition=L40 --gpus=1 --pty --container-image='nvcr.io/nvidia/pytorch:25.01-py3' --container-name='lora' bash
```

### Leave as background job

Once you are satisfied with the code and want to run it as a background job, you can submit the job to the cluster using the following command:

```bash
sbatch train_and_test_lora.slurm
```

### Run inference

Inference uses vLLM and can compare the original model with the LoRA fine-tuned
model. Run both models on one question by omitting the model-selection flags:

```bash
python test_with_vllm.py "Genera un ejercicio sobre autos azules"
```

Use `--original` to run only the original model:

```bash
python test_with_vllm.py --original "Genera un ejercicio sobre autos azules"
```

Use `--finetunned` to run only the LoRA fine-tuned model:

```bash
python test_with_vllm.py --finetunned "Genera un ejercicio sobre autos azules"
```

Both flags can be provided explicitly to run both models:

```bash
python test_with_vllm.py --original --finetunned "Genera un ejercicio sobre autos azules"
```

When no question or model-selection flags are provided, the script preserves
the default evaluation behavior: it runs both models for every prompt in
`TESTING_PROMPS` from `config.py` and writes the results to `results.csv`.

The flags can also be used without a question to run only the selected model
for every configured prompt.

## Commands

See active jobs

```bash
squeue
```

To cancel a job, use its job ID from the `squeue` command:

```bash
scancel 90931
```
