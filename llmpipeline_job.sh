#!/bin/bash
#SBATCH --job-name=llmpipe-test
#SBATCH --output=llmpipe%j.out
#SBATCH --error=llmpipe%j.err

#SBATCH --time=15:00:00             # 15h min max
#SBATCH --partition=gpu_mig         # Use the GPU partition, testing quantization on gpu_mig like jupyter notebook
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --cpus-per-task=1           # Number of CPU cores
#SBATCH --mem=30G                   # Amount of RAM
#SBATCH --mail-type=END,FAIL        # Notification on job end or fail
#SBATCH --mail-user=jialong.bao@student.uva.com 

# Safely use the token inside the job
huggingface-cli login --token $HF_TOKEN --add-to-git-credential

# Load modules (if needed)
module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 
#module load cuda/11.8  # Adjust based on your system

# pip install transformers accelerate pandas regex torch joblib
pip install hf_xet
#pip install git+https://github.com/huggingface/transformers.git
pip install transformers
pip install git+https://github.com/huggingface/accelerate.git
# # Activate your Python environment
# source ~/.bashrc
# source $HOME/Thesis_Snellius_Job/venv/bin/activate

# Print environment details (optional)
echo "Running on $(hostname)"
echo "Job started on: $(date)"
#nvidia-smi
#which python

# Run the script
python llmpipeline.py