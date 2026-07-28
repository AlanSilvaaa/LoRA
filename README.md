# LoRA

Experimental pipeline for fine-tuning language models with LoRA on mathematical
word problems. It supports training and validation, overfitting measurements,
and comparisons between base and fine-tuned model outputs.

The model, dataset, and training parameters can be changed in `config.py` as
different configurations are evaluated.

## Run container on patagon

```bash
srun --partition=L40 --gpus=1 --pty --container-image='nvcr.io/nvidia/pytorch:25.01-py3' --container-name='lora' bash
```
