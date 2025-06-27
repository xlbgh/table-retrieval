#!/bin/bash
#Set job requirements
#SBATCH --mail-user=your.mail@here.nl
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

# Safely use the token inside the job
huggingface-cli login --token $HF_TOKEN --add-to-git-credential

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
cp $HOME/Thesis_Snellius_Job/observations.json "$TMPDIR"
cp $HOME/Thesis_Snellius_Job/tables.csv "$TMPDIR"

# JSON checkpoint file paths
CHECKPOINT_HOME="$HOME/llama_summary_checkpoint.json"
CHECKPOINT_TMP="$TMPDIR/llama_summary_checkpoint.json"

# Copy checkpoint file if it exists to scratch/local
if [ -f "$CHECKPOINT_HOME" ]; then
    cp "$CHECKPOINT_HOME" "$CHECKPOINT_TMP"
fi

# Create output directory on scratch/local
mkdir -p "$TMPDIR/llama_summaries_out"

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
cp -r "$TMPDIR/llama_summaries_out" $HOME/Thesis_Snellius_Job

# Copy checkpoint JSON back to home if it exists
if [ -f "$CHECKPOINT_TMP" ]; then
    cp "$CHECKPOINT_TMP" "$CHECKPOINT_HOME"
fi

echo "Running on $(hostname)"
echo "Job started on: $(date)"