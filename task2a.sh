#!/bin/bash
#SBATCH --job-name=10_100
#SBATCH --nodes=1              
#SBATCH --ntasks=1             
#SBATCH --cpus-per-task=4     
#SBATCH --gres=gpu:1           
#SBATCH --mem=8G              
#SBATCH --time=01:00:00        
#SBATCH --output=outputs/output_%j.log
#SBATCH --error=outputs/error_%j.log


# Load ollama module
module load cs/ollama

source /opt/bwhpc/common/devel/miniforge/24.11.0-py3.12/etc/profile.d/conda.sh

conda activate mythesis

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
sleep 20  # Wait for server to initialize

# Debug
echo "PORT: $PORT" >> outputs/output_${SLURM_JOB_ID}.log

# Monitor GPU usage
nvidia-smi 
free -h
NVIDIA_PID=$!

# Run Python script
timeout 7200 python /pfs/work9/workspace/scratch/ma_ssiu-thesis/Teapot/2_optimized1.py --port $PORT
# Clean up
kill $NVIDIA_PID
kill $OLLAMA_PID