#!/bin/bash
#SBATCH --job-name=task2l
#SBATCH --nodes=1              # Keep 1 node
#SBATCH --ntasks=1             # Single task
#SBATCH --cpus-per-task=8     # Increase to 12 CPUs for better I/O and orchestration
#SBATCH --gres=gpu:1           # Keep 1 GPU (H100 is sufficient)
#SBATCH --mem=32G              # Increase to 32 GB RAM for model transitions
#SBATCH --time=04:00:00        # Increase to 3 hours for 2,580 inferences
#SBATCH --output=outputs/output_%j.log
#SBATCH --error=outputs/error_%j.log

# Load ollama module
module load cs/ollama

# Initialize Conda
source /opt/bwhpc/common/devel/miniconda/23.9.0-py3.9.15/etc/profile.d/conda.sh

# Activate the test environment
conda activate test

# Set Ollama environment variable to keep model loaded
PORT=$((11434 + (SLURM_JOB_ID % 1000)))
export OLLAMA_DEBUG=true
export OLLAMA_KEEP_ALIVE="4h"
export OLLAMA_NUM_PARALLEL=4    # Max parallelism
export OLLAMA_MAX_QUEUE=128
export OLLAMA_CTX_SIZE=16384

# Start Ollama server in the background
export OLLAMA_HOST="127.0.0.1:$PORT"
ollama serve > outputs/ollama_${SLURM_JOB_ID}.log 2>&1 &
OLLAMA_PID=$!
sleep 10  # Wait for server to initialize
# Debug
echo "PORT: $PORT" >> outputs/output_${SLURM_JOB_ID}.log

# Monitor GPU usage
nvidia-smi 
free -h
NVIDIA_PID=$!

# Run Python script
timeout 7200 python /pfs/work7/workspace/scratch/ma_ssiu-myspace/teapot/2_optimized1.py --port $PORT
# Clean up
kill $NVIDIA_PID
kill $OLLAMA_PID