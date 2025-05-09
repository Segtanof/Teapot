#!/bin/bash
#SBATCH --job-name=embedding_test1
#SBATCH --nodes=1                # Use 1 node
#SBATCH --ntasks=1               # Single task
#SBATCH --cpus-per-task=8        # 8 CPUs for parallelism
#SBATCH --mem=4G               # 32 GB RAM
#SBATCH --time=04:00:00          # 30 minutes, adjustable based on workload
#SBATCH --output=outputs/output_%j.log  # Job-specific output
#SBATCH --error=outputs/error_%j.log    # Job-specific error
#SBATCH --mail-type=ALL
#SBATCH --mail-user=godfreysiust@gmail.com


# Initialize Conda
source /opt/bwhpc/common/devel/miniforge/24.11.0-py3.12/etc/profile.d/conda.sh

# Activate the environment
conda activate test

# Log environment details
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"


# Run the Python script
python /pfs/work9/workspace/scratch/ma_ssiu-thesis/Teapot/2_embedding1.py