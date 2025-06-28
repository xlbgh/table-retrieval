Assumes you have queries + ground truth tableid + metadata for evaluation;

in order to run a llm that requires huggingface approval first on snellius:

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

Misc:
In case of no metadata; pneuma paper introduced a compelling way to create metadata https://arxiv.org/abs/2504.09207

TODO: add large table retrieval joblib file (ie generated summaries) somehow on github; for now just on google drive https://drive.google.com/file/d/1U7UepZxVR3qbA3lpJpJq1lYN1dFx9nhs/view?usp=sharing
