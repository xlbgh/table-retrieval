import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer 
import os

def test_quantized_model_loading(model_id: str):

    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    generator = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    input_text = f"""
        Gegeven de volgende context over de tabel:

        De tabel heeft de volgende kolommen: ['x2_OpleidingNiveau_3A', 'LeeftijdGroep', 'GeslachtCode', 'WoonRegioNaam', 'rowID']

        Beschrijf kort wat de kolom 'WoonRegioNaam' vertegenwoordigt. Voorbeeldwaarden uit deze kolom zijn: ['Groningen', 'Zuid-Holland', 'Noord-Brabant'].

        Als een beschrijving niet mogelijk is, schrijf dan: "Geen beschrijving".
        """
    
    messages = [
        {"role": "system", "content": "Je bent een behulpzame asssistent."},
        {"role": "user", "content": input_text},
    ]

    outputs = generator(
        messages,
        max_new_tokens=512,
        temperature=0.6,
        top_p=0.9,
        top_k=30,
        repetition_penalty=1.15,
        eos_token_id=generator.tokenizer.eos_token_id,
        num_return_sequences=1
    )
    print(outputs[0]["generated_text"][-1]['content'])
    print(type(outputs[0]["generated_text"][-1]['content']))




if __name__ == "__main__":
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    test_quantized_model_loading(model_id)
