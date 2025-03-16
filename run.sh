#!/bin/bash
#SBATCH --ntasks=40
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/output.txt
#SBATCH --error=outputs/error.txt
#SBATCH --job-name=test
#SBATCH --mem=128000

python 2_benchmark_run.py
# nvidia-smi