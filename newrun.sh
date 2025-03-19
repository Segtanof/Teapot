#!/bin/bash
#SBATCH --job-name=ollama_test
#SBATCH --nodes=1              # Use 1 node
#SBATCH --ntasks=1             # Single task
#SBATCH --cpus-per-task=8      # 8 CPUs for parallelism
#SBATCH --gres=gpu:1           # 1 GPU
#SBATCH --mem=16G              # 16 GB RAM, sufficient for model + overhead
#SBATCH --time=01:00:00        # 2 hours, adjustable based on workload
#SBATCH --output=output_%j.log # Job-specific output
#SBATCH --error=error_%j.log   # Job-specific error

# Load ollama module
module load cs/ollama

# Initialize Conda
source /opt/bwhpc/common/devel/miniconda/23.9.0-py3.9.15/etc/profile.d/conda.sh

# Activate the test environment
conda activate test

# Debug info
echo "Active environment: $CONDA_DEFAULT_ENV" >> output_$SLURM_JOB_ID.log
echo "Python path: $(which python)" >> output_$SLURM_JOB_ID.log
pip list | grep langchain-ollama >> output_$SLURM_JOB_ID.log
echo "Ollama path: $(which ollama)" >> output_$SLURM_JOB_ID.log

# Set Ollama environment variable to keep model loaded
export OLLAMA_KEEP_ALIVE="1h"
export OLLAMA_NUM_PARALLEL=8    # Max parallelism

# Start Ollama server in the background
ollama serve &
OLLAMA_PID=$!
sleep 10  # Wait for server to initialize

# Monitor GPU usage
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1 > gpu_usage_$SLURM_JOB_ID.log &
NVIDIA_PID=$!

# Run Python script
python /pfs/work7/workspace/scratch/ma_ssiu-myspace/teapot/2_new_bench_match.py
# Clean up
kill $NVIDIA_PID
kill $OLLAMA_PID