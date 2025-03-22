#!/bin/bash
#SBATCH --job-name=task1
#SBATCH --nodes=1              
#SBATCH --ntasks=1             
#SBATCH --cpus-per-task=8     
#SBATCH --gres=gpu:1           
#SBATCH --mem=16G              
#SBATCH --time=02:00:00        
#SBATCH --output=outputs/output_%j.log
#SBATCH --error=outputs/error_%j.log

# Load ollama module
module load cs/ollama

# Initialize Conda
source /opt/bwhpc/common/devel/miniconda/23.9.0-py3.9.15/etc/profile.d/conda.sh

# Activate the test environment
conda activate test

# Set Ollama environment variable to keep model loaded
export OLLAMA_DEBUG=true
export OLLAMA_KEEP_ALIVE="2h"
export OLLAMA_NUM_PARALLEL=8    # Max parallelism
export OLLAMA_MAX_QUEUE=128
export OLLAMA_CTX_SIZE=16384

# Start Ollama server in the background
ollama serve &
OLLAMA_PID=$!
sleep 5  # Wait for server to initialize

# Monitor GPU usage
nvidia-smi 
free -h
NVIDIA_PID=$!

# Run Python script
timeout 7200 python /pfs/work7/workspace/scratch/ma_ssiu-myspace/teapot/1_test_run.py
# Clean up
kill $NVIDIA_PID
kill $OLLAMA_PID