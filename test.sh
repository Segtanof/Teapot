#!/bin/bash
#SBATCH --job-name=testo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:10:00  # Short test run
#SBATCH --output=outputs/output_%j.log
#SBATCH --error=outputs/error_%j.log

module load cs/ollama
module load devel/cuda/12.2

source /opt/bwhpc/common/devel/miniconda/23.9.0-py3.9.15/etc/profile.d/conda.sh
conda activate test

# Dynamic port
PORT=$((11434 + (SLURM_JOB_ID % 1000)))
export OLLAMA_HOST="127.0.0.1:$PORT"
ollama serve > outputs/ollama_${SLURM_JOB_ID}.log 2>&1 &
OLLAMA_PID=$!
sleep 5

# Debug
echo "PORT: $PORT" >> outputs/output_${SLURM_JOB_ID}.log

# Run Python with positional arg
python /pfs/work7/workspace/scratch/ma_ssiu-myspace/teapot/test_test.py --port $PORT

kill $OLLAMA_PID 2>/dev/null