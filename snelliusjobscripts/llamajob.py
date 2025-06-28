# import pickle
from transformers import pipeline #, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import pandas as pd
import re
import os
import sys
#from joblib import dump, load
import json
import shutil
import time



def sample_unique_row_values(df, column_name, n=3, random_state=88):
    """
    Sample up to `n` non-na unique values from a specified column in a DataFrame.

    Parameters:
    - df: pandas DataFrame
    - column_name: str, name of the column to sample from
    - n: int, number of unique values to sample (default: 3)
    - random_state: int or None, for reproducibility

    Returns:
    - List of sampled unique values
    """
    filtered_values = df[column_name].dropna()
    unique_values = filtered_values.drop_duplicates()

    sample_size = min(n, len(unique_values)) #In case of less than 3 rows

    return unique_values.sample(n=sample_size, random_state=random_state).tolist()




# TODO later: try remove the deepseek's chain of thought output to reduce noise in docs/extract final answer, probably through skip_special_tokens parameter -> tokenizer.apply_chat_template() for role and content keys
def summarize_columns(model, df, original_metadata_text, column_generation_params):
    """Generate a description for each column with some value examples and concatonates the generated descriptions together."""

    cols = list(df.columns)
    # Optionally provide full schema context
    schema_str = " | ".join(cols)
    col_summaries = []
    all_unique_row_samples = []
    
    for col in cols: 
        unique_row_samples = sample_unique_row_values(df, col, n=3)
        all_unique_row_samples.append(unique_row_samples)
        row_samples_str = str(unique_row_samples)

        # Prompt adaptation from Pneuma paper, minor adjustments based on llama and huggingface generation strategies documentation, translated and refined to dutch
        input_text = f"""
        Gegeven de volgende context over de tabel:

        {original_metadata_text}

        De tabel heeft de volgende kolommen: {schema_str}

        Beschrijf kort wat de kolom '{col}' vertegenwoordigt. Voorbeeldwaarden uit deze kolom zijn: {row_samples_str}.

        Als een beschrijving niet mogelijk is, schrijf dan: "Geen beschrijving".

        Beschrijving:
        """

        messages = [
            {"role": "system", "content": "Je bent een behulpzame assistent."},
            {"role": "user", "content": input_text},
        ]

        outputs = model(
            messages,
            **column_generation_params
        )
        
        # Get generated output
        generated_column_summary = outputs[0]["generated_text"][-1]['content']

        col_summaries.append(f"De kolom {col} vertegenwoordigt: " + generated_column_summary)
    
    # Concatenate all column descriptions into one schema summary
    schema_summary = " ".join(col_summaries)
    return schema_summary, col_summaries, all_unique_row_samples





def generate_overall_summary(model, schema_summary, original_metadata_text, table_generation_params):
    """
    Generate a concise overall summary of the dataset using the schema summary.
    
    Parameters:
    - model: A language model capable of summarization.
    - tokenizer: Tokenizer corresponding to the model.
    - schema_summary: String containing concatenated column descriptions.
    
    Returns:
    - overall_summary: A high-level summary of the table.
    """

    input_text = f"""De dataset bevat de volgende kolombeschrijvingen: {schema_summary}

    Aanvullende context over de dataset: {original_metadata_text}

    Vat de bovenstaande tekst samen in een korte, duidelijke algemene beschrijving van de dataset.

    Samenvatting:
    """

    messages = [
            {"role": "system", "content": "Je bent een behulpzame assistent."},
            {"role": "user", "content": input_text},
        ]

    outputs = model(
        messages,
        **table_generation_params
    )
    
    # Get generated output
    generated_table_summary = outputs[0]["generated_text"][-1]['content']
    
    return generated_table_summary


def llm_summarize_all_tables_with_checkpoint(model, table_data_dfs, tables_metadata_df, summary_dict, checkpoint_path):
    """
    Summarizes tables with checkpointing.

    Parameters:
    - model: LLM (text generation pipeline)
    - table_data_dfs: dict of {table_id: DataFrame}
    - tables_metadata_df: DataFrame with metadata
    - summary_dict: dict with already processed summaries {table_id: summary}
    - checkpoint_path: path to save progress (json)

    Returns:
    - summary_dict: updated dict with summaries for all processed tables
    """
    processed_count = 0  # Initialize counter outside the loop, save checkpoint every x tables processed

    # max_new_tokens from Pneuma, temperature and top_p from https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/generation_config.json,
    # top_k + repetition_penalty + eos_token_id + num_return_sequence to avoid near infinite repetition, building on knowledge of model
    column_generation_params = {
        "max_new_tokens": 512, #
        "temperature": 0.6,
        "top_p": 0.9,
        "top_k": 30,
        "repetition_penalty": 1.15,
        "eos_token_id": model.tokenizer.eos_token_id,
        "num_return_sequences": 1
    }

    # Adding prompt_lookup_num_tokens to ground and enhance summarization task https://huggingface.co/docs/transformers/en/generation_strategies#prompt-lookup-decoding
    table_generation_params = {
        **column_generation_params,
        #"prompt_lookup_num_tokens": 10 
    }
  
    
    for table_id, df in table_data_dfs.items():
        if table_id in summary_dict:
            print(f"[{table_id}] already processed, skipping.")
            continue

        # Match metadata, if not available set to placeholder (Dutch: Not Available.)
        matched_row = tables_metadata_df[tables_metadata_df['table_id'] == table_id]
        if not matched_row.empty:
            original_metadata_text = matched_row.iloc[0]['original_metadata_text']
        else:
            original_metadata_text = "Niet Beschikbaar."

        # Generate summaries
        schema_summary, col_summaries, samples = summarize_columns(model, df, original_metadata_text, column_generation_params)
        overall_summary = generate_overall_summary(model, schema_summary, original_metadata_text, table_generation_params)

        # Collect summary dict for this table
        summary = {
            "table_id": table_id,
            "original_metadata_text": original_metadata_text,
            "column_summaries": col_summaries,
            "schema_summary": schema_summary,
            "overall_summary": overall_summary,
            "sampled_values": samples
        }

        # Save to summary dict
        summary_dict[table_id] = summary

        processed_count += 1

        # Save checkpoint every 50 tables processed
        if processed_count % 50 == 0:
            tmp_cp_path = checkpoint_path + ".tmp"
            with open(tmp_cp_path, "w") as f:
                json.dump(summary_dict, f)
            shutil.move(tmp_cp_path, checkpoint_path)

            print(f"Checkpoint saved after processing {processed_count} tables.")

            # Copy checkpoint file from scratch to home directory
            home_checkpoint_path = os.path.expanduser("~/llama_summary_checkpoint.json")
            shutil.copy(checkpoint_path, home_checkpoint_path)
            print(f"Checkpoint copied to home at {home_checkpoint_path}")


    return summary_dict





def clean_and_concat(row, columns_to_concat=['table_desc', 'expanded_desc', 'Keywords', 'Summary']):
    """Clean selected columns by removing all \\n and \n and reducing whitespace to single space."""
    pattern_newlines = re.compile(r'(\\n|\n)+')
    pattern_whitespace = re.compile(r'\s+')
    
    cleaned_parts = []
    for col in columns_to_concat:
        # Convert to string, replace newlines with space
        cleaned = pattern_newlines.sub(' ', str(row[col]))
        # Reduce multiple whitespace chars to a single space
        cleaned = pattern_whitespace.sub(' ', cleaned).strip()
        cleaned_parts.append(cleaned)
        
    return ' '.join(cleaned_parts)




if __name__ == "__main__":

    # Get command line arguments (input file, output folder)
    input_path_table_data = sys.argv[1]
    input_path_metadata = sys.argv[2]
    output_folder = sys.argv[3]

    # Load table data
    with open(input_path_table_data, "r") as f:
        tables_data = json.load(f)

    # Dictionary to hold the resulting DataFrames
    table_data_dfs = {}

    # Loop through each table, k = tables, v = rows
    for table_id, rows in tables_data.items():
        # Convert list of row dicts into a DataFrame
        table_data_dfs[table_id] = pd.DataFrame(rows)

    # Read in original metadata
    tables_metadata_df = pd.read_csv(input_path_metadata)
    
    # Concatonate original metadata
    tables_metadata_df['original_metadata_text'] = tables_metadata_df.apply(
    lambda row: clean_and_concat(row),
    axis=1)

    # Load any previous progress
    checkpoint_path = os.path.expanduser("~/llama_summary_checkpoint.json")
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        with open(checkpoint_path, "r") as f:
            summary_dict = json.load(f)
    else:
        summary_dict = {}

    # # Load the model and tokenizer in 4-bit mode on GPU
    # model_id = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    # tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left") # Left padding to continue generation

    # # quantization_config = BitsAndBytesConfig(load_in_4bit=True) #ImportError: cannot import name 'validate_bnb_backend_availability' from 'transformers.integrations' -> error on snellius
    
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     #load_in_4bit=True,     # 4-bit quantization, works on Jupyter.snellius -> Deprecated soon, need to use BitsAndBytesConfig, does the same basically -> doesnt work as well, it's caused by HF_TOKEN
    #     device_map="auto",     # auto-assign layers to GPUs
    #     # quantization_config=quantization_config # 
    # )

    # 3.1 Multilingual, not directly trained on Dutch, small parameters, fine tuned on instructrions for assistant like conversation; not MoE or unsupervised training like deepseek i.e. a simpler model
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    generator_model = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    summary_dict = llm_summarize_all_tables_with_checkpoint(
        generator_model,
        table_data_dfs,
        tables_metadata_df,
        summary_dict,
        checkpoint_path,
    )
    
    # Get the TMPDIR path from environment variable
    tmpdir = os.environ.get("TMPDIR", ".")
    save_path = os.path.join(tmpdir, "final_llama_summary_output.json")

    # Save final version as JSON
    with open(save_path, "w") as f:
        json.dump(summary_dict, f)
    
    print(f"Summary saved to {save_path}")

    shutil.move(save_path, checkpoint_path)

    # Copy checkpoint file from scratch to home directory
    home_checkpoint_path = os.path.expanduser("~/final_llama_summary_output.json")
    shutil.copy(checkpoint_path, home_checkpoint_path)
    print(f"Checkpoint copied to home at {home_checkpoint_path}")
