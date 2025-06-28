Thesis repo: LLM generated metadata against original CBS table metadata using Information Retrieval metrics (BM25 and ColBERT)

---

## Project Structure

**Key components in repo:**

- **Notebooks:**
  - `CBS_Final_Experiment.ipynb` main notebook for running final evaluations.

- **Original CBS Data:**
  - `questions_tables.csv` queries and ground truth table IDs.
  - `tables.csv` original table metadata with table IDs.

- **Available for download (link below):**

  - `/experiment_results/` ColBERT query results on generated summaries.
  - `tables_data_df.joblib` data file from CBS Open Data. 
  - `observations.json` all retrieved table data (~2GB).

- **Snellius Job Scripts:**
  `/snelliusjobscripts/` directory for clarity.

---

## Requirements to run the final evaluation notebook

Make sure you have the following in your working directory:

- `CBS_Final_Experiment.ipynb`
- `questions_tables.csv`
- `tables.csv`
- `tables_data_df.joblib` [https://drive.google.com/drive/folders/11KLN-xndUKK4CQMnlUYamij511wV9uWV?usp=sharing](Google Drive)
- all files from `/experiment_results/` [https://drive.google.com/drive/folders/11KLN-xndUKK4CQMnlUYamij511wV9uWV?usp=sharing](Google_Drive)

---

## /snelliusjobscripts/ Generating Summaries on Snellius

**Two-Phase Generation:**
- `llama_job.sh` Snellius job script for Concat + Distil.
- `llamajob.py` Python script for Concat + Distil.

**One-Phase Generation:**
- `tables_flatten_only_llama_job.sh` job script for Head.
- `tables_flatten_only_llamajob.py` Python script for Head.
- `tables_w_samples_llama_job.sh` job script for Sam.
- `tables_w_samples_llamajob.py` Python script for Sam.

Make sure you have the following in your working directory:
- `tables.csv`
- 'observations.json'
---

## Using Hugging Face Models on Snellius

Some Hugging Face models are gated, and require approval and an access token

**How to safely use your Hugging Face token in Snellius jobs:**

```bash
# 1. Store token securely
echo 'export HF_TOKEN=your_token_here' > ~/.hf_token
chmod 600 ~/.hf_token

# 2. Load token before running jobs
source ~/.hf_token

# 3. Add this in your jobscript before commands
huggingface-cli login --token $HF_TOKEN --add-to-git-credential

# 4. Submit your job (example)
sbatch --export=HF_TOKEN job_script.sh
