#!/bin/bash
#Set job requirements
# To receive email notifications
#SBATCH --mail-user=your.email@here.please
#SBATCH --mail-type=END,FAIL

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu_a100
#SBATCH --time=15:00:00
#SBATCH --gpus-per-node=2

#SBATCH --job-name=llama-final-pls
#SBATCH --output=llama_final%j.out
#SBATCH --error=llama_final%j.err

#SBATCH --requeue
#SBATCH --signal=TERM@300  # Send TERM signal 5 minutes before timeout

# Safely use the huggingface token inside the job
# Because you need to login in order to run certain models
# 1. On snellius first create a hidden env by running the following in cli
# echo 'export HF_TOKEN=your_token_here' > ~/.hf_token
# chmod 600 ~/.hf_token
# Optional, you can run this to check if the hidden file was properly created with your token
# echo $HF_TOKEN 
# 2. Then load it in before running commands/submitting jobs
# source ~/.hf_token
# 3. Add the following within your jobscript
huggingface-cli login --token $HF_TOKEN --add-to-git-credential
# 4. Submit the job by
# sbatch --export=HF_TOKEN job_script.sh

# Load modules
module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 

# Create and activate virtual environment
virtualenv --system-site-packages $HOME/venv
source $HOME/venv/bin/activate

pip install --upgrade pip
pip install hf_xet
pip install transformers pandas regex torch joblib
pip install git+https://github.com/huggingface/accelerate.git

# Copy input files to scratch/local = $TMPDIR
cp $HOME/observations.json "$TMPDIR"
cp $HOME/tables.csv "$TMPDIR"

# JSON checkpoint file paths
CHECKPOINT_HOME="$HOME/llama_summary_checkpoint.json"
CHECKPOINT_TMP="$TMPDIR/llama_summary_checkpoint.json"

# Copy checkpoint file if it exists to scratch/local
if [ -f "$CHECKPOINT_HOME" ]; then
    cp "$CHECKPOINT_HOME" "$CHECKPOINT_TMP"
fi

# Create output directory on scratch/local
mkdir -p "$TMPDIR"

# Set trap to catch SIGTERM (sent 5 minutes before timeout)
trap "echo 'Caught SIGTERM, safe exit for requeue'; exit 1" TERM

# Execute the Python program with inputs and output folder
python $HOME/llamajob.py "$TMPDIR/observations.json" "$TMPDIR/tables.csv" "$TMPDIR/llama_summaries_out"

# Check if python exited non-zero (failed or killed)
if [ $? -ne 0 ]; then
    echo "Job failed or was killed. Requeuing..."
    scontrol requeue $SLURM_JOB_ID
    exit 1
fi

# Copy output directory from scratch to home
cp -r "$TMPDIR" $HOME

# Copy checkpoint JSON back to home if it exists
if [ -f "$CHECKPOINT_TMP" ]; then
    cp "$CHECKPOINT_TMP" "$CHECKPOINT_HOME"
fi

echo "Running on $(hostname)"
echo "Job started on: $(date)"