#!/bin/bash
#SBATCH --job-name=ollama_test
#SBATCH --ntasks=40
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --mem=128000

# Load Conda (might need adjustment based on cluster setup)
source /opt/bwhpc/common/devel/miniconda/23.9.0-py3.9.15/etc/profile.d/conda.sh
# Activate test environment
source /pfs/work7/workspace/scratch/ma_ssiu-myspace/.conda/envs/test/bin/activate
# Verify environment (optional, for debugging)
echo "Active environment: $CONDA_DEFAULT_ENV" >> output.log
echo "Python path: $(which python)" >> output.log
pip list | grep langchain-ollama >> output.log  # Check if langchain_ollama is there

ollama serve &  # Start server
sleep 5         # Wait for it
python /pfs/work7/workspace/scratch/ma_ssiu-myspace/teapot/test_test.py