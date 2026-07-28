# LoRA

Experimental pipeline for fine-tuning language models with LoRA on mathematical
word problems. It supports training and validation, overfitting measurements,
and comparisons between base and fine-tuned model outputs.

The model, dataset, and training parameters can be changed in `config.py` as
different configurations are evaluated.

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

## Commands

See active jobs

```bash
squeue
```

To cancel a job, use its job ID from the `squeue` command:

```bash
scancel 90931
```
