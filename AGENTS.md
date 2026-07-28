# AGENTS

## Context

This project is designed to retrain a pretrained model to perform a specific task, on this case, creating coherent mathematical problems and solving them, for kids ranging from 1st grade to 6th grade. The model is trained using a dataset of well formulated mathematical problems, and it learns to generate new problems.

The training happens on a supercomputer called Patagon, which is located in Chile. This repository is designed to run on Patagon, but also be edited outside of it, and then be executed on Patagon. This supercomputer works with containers and SLURM, so the code is designed to be executed in a containerized environment, and the training process is managed using SLURM.
