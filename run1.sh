#!/bin/bash
#SBATCH --ntasks=40
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/output.txt
#SBATCH --error=outputs/error.txt
#SBATCH --job-name=test
#SBATCH --mem=128000

source ~/.bashrc               # Load your settings
source /pfs/work7/workspace/scratch/ma_ssiu-myspace/.conda/envs/test/bin/activate
ollama serve &                 # Start the server on the compute node
sleep 5                        # Wait 5 seconds for it to start
python test_test.py
# nvidia-smi