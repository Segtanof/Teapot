#!/bin/bash
#SBATCH --job-name=embedding_test
#SBATCH --nodes=1              # Use 1 node
#SBATCH --ntasks=1             # Single task
#SBATCH --cpus-per-task=2      # 4 CPUs for parallelism
#SBATCH --gres=gpu:1           # 1 GPU
#SBATCH --mem=16G              # 16 GB RAM, sufficient for model + overhead
#SBATCH --time=02:00:00        # 2 hours, adjustable based on workload
#SBATCH --output=outputs/output_%j.log # Job-specific output
#SBATCH --error=outputs/error_%j.log   # Job-specific error


# Initialize Conda
source /opt/bwhpc/common/devel/miniconda/23.9.0-py3.9.15/etc/profile.d/conda.sh

module load devel/cuda/12.4

# Activate the test environment
conda activate test

# Log environment details
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"
echo "CUDA module loaded: $(module list 2>&1 | grep cuda)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Run Python script
python /pfs/work7/workspace/scratch/ma_ssiu-myspace/teapot/2_embedding_task.py

# Clean up
kill $NVIDIA_PID
nvidia-smi