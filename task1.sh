#!/bin/bash
#SBATCH --job-name=43l2
#SBATCH --nodes=1              
#SBATCH --ntasks=1             
#SBATCH --cpus-per-task=8     
#SBATCH --gres=gpu:1           
#SBATCH --mem=32G              
#SBATCH --time=6:00:00        
#SBATCH --output=outputs/output_%j.log
#SBATCH --error=outputs/error_%j.log

# Load ollama module
module load cs/ollama
module load devel/cuda/12.2

# Initialize Conda
source /opt/bwhpc/common/devel/miniconda/23.9.0-py3.9.15/etc/profile.d/conda.sh

# Activate the test environment
conda activate test



# Start Ollama server in the background
PORT=$((11434 + (SLURM_JOB_ID % 1000)))
# Set Ollama environment variable to keep model loaded
export OLLAMA_DEBUG=true
export OLLAMA_KEEP_ALIVE="6h"
export OLLAMA_NUM_PARALLEL=4    # Max parallelism
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
monitor_gpu() {  # NEW: Function for continuous GPU logging
    local output_file="outputs/gpu_usage_${SLURM_JOB_ID}.log"
    echo "Timestamp, GPU Util (%), Memory Used (MiB), Power (W)" > "$output_file"
    while true; do
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits | \
        awk -v ts="$timestamp" '{print ts ", " $1 ", " $2 ", " $3}' >> "$output_file"
        sleep 10  # NEW: 10-second interval, less noise than 5
    done
}
monitor_gpu &  # NEW: Start GPU monitoring in background
GPU_MONITOR_PID=$!  # NEW: Store PID for cleanup

# Run Python with explicit output
echo "Starting Python script..." >> outputs/output_${SLURM_JOB_ID}.log
python /pfs/work7/workspace/scratch/ma_ssiu-myspace/teapot/1_optimized.py --port $PORT >> outputs/output_${SLURM_JOB_ID}.log 2>> outputs/error_${SLURM_JOB_ID}.log
PYTHON_EXIT=$?

# Clean up
kill $OLLAMA_PID 2>/dev/null
killall nvidia-smi 2>/dev/null

echo "Python exit code: $PYTHON_EXIT" >> outputs/output_${SLURM_JOB_ID}.log