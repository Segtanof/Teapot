#!/bin/bash
#SBATCH --job-name=100-150
#SBATCH --nodes=1              
#SBATCH --ntasks=1             
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1           
#SBATCH --mem=32G              
#SBATCH --time=10:00:00    
#SBATCH --output=outputs/output_%j.log
#SBATCH --error=outputs/error_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=godfreysiust@gmail.com

module load cs/ollama

source /opt/bwhpc/common/devel/miniforge/24.11.0-py3.12/etc/profile.d/conda.sh

conda activate mythesis

# Start Ollama server in the background
PORT=$((11434 + (SLURM_JOB_ID % 1000)))
# Set Ollama environment variable to keep model loaded
export OLLAMA_DEBUG=true
export OLLAMA_KEEP_ALIVE="12h"
export OLLAMA_NUM_PARALLEL=6    # Max parallelism
export OLLAMA_MAX_QUEUE=512
export OLLAMA_CTX_SIZE=8192

export OLLAMA_HOST="127.0.0.1:$PORT"
ollama serve > outputs/ollama_${SLURM_JOB_ID}.log 2>&1 &
OLLAMA_PID=$!
sleep 10  # Wait for server to initialize
# Debug
echo "PORT: $PORT" >> outputs/output_${SLURM_JOB_ID}.log

# Monitor GPU usage
nvidia-smi -l 180 > outputs/gpu_initial_${SLURM_JOB_ID}.log &


python /pfs/work9/workspace/scratch/ma_ssiu-thesis/Teapot/1_optimized_a.py --port $PORT
PYTHON_EXIT=$?

# Clean up
kill $OLLAMA_PID 2>/dev/null
killall nvidia-smi 2>/dev/null

echo "Python exit code: $PYTHON_EXIT" >> outputs/output_${SLURM_JOB_ID}.log