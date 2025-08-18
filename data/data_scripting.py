'''
got the idea of n-shot text classification using a pre-trained model
https://medium.com/farmart-blog/few-shot-learning-a-simple-approach-6877c9009ca9

and utilised gemini-2.5- pro for code generation and then modified it to use with BioMistral-7B
'''

import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM 
# from transformers import AutoTokenizer, AutoModel # for BioLinkbert
from tqdm import tqdm 

# categorising data for n-shot text classification
CATEGORIES = [
    'Cardiovascular', 'Respiratory', 'Gastrointestinal', 'Neurological', 'Musculoskeletal',
    'Endocrine', 'Renal/Urological', 'Dermatological', 'Hematological', 'Oncology',
    'Reproductive/Gynecology', 'Pediatrics', 'Infectious Diseases', 'Geriatrics (Elderly Health)',
    'Ophthalmology', 'Dentistry', 'General Medicine', 'Psychiatric and Mental Health',
    'Nutrition and Metabolism', 'Substance Abuse and Toxicology', 'Therapy and Rehabilitation'
]


MODEL_ID = "BioMistral/BioMistral-7B"
# MODEL_ID = "michiyasunaga/BioLinkBERT-large"

print(f"Loading model: {MODEL_ID}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    print(f"Successfully loaded model: {MODEL_ID}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Set pad token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

#  Load Dataset
DATASET_ID = "GBaker/MedQA-USMLE-4-options"
NUM_SAMPLES = 1000 # change as per need, I tried with only 1000

try:
    dataset = load_dataset(DATASET_ID, split=f'train[:{NUM_SAMPLES}]')
    print(f"Successfully loaded {len(dataset)} samples from {DATASET_ID}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

#  "Multi-Hop step prompt" In order to leverage instruction fine-tuning, the prompt should be surrounded by [INST] and [/INST] tokens.
# https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1

PROMPT_TEMPLATE = """<s>[INST] You are an expert medical research assistant. Your task is to categorize the following medical question into one of the predefined categories.

Before selecting a category, perform a thorough analysis of the question, its options, and the provided correct answer. Consider the primary medical specialty involved, the organ systems affected, the patient population (if implied), and the nature of the medical knowledge being tested.

Medical Question Details:
Question: {question}

Options:
{options}

Correct Answer is: {answer_text} (Option {answer_idx})

Predefined Medical Categories:
{categories_list}

Based on your analysis, identify the single most appropriate category from the list above. Provide only the category name as your answer.

Category: [/INST]"""

#  Process Data and Generate Categories
results = []

for item in tqdm(dataset, desc="Processing questions"):
    question_text = item['question']
    options_dict = item['options']
    answer_idx = item['answer_idx']
    correct_answer_text = item['answer']

    options_str = "\n".join([f"{key}. {value}" for key, value in options_dict.items()])
    categories_list_str = "\n".join([f"- {cat}" for cat in CATEGORIES])

    # Construct the full prompt using the template
    full_prompt_text = PROMPT_TEMPLATE.format(
        question=question_text,
        options=options_str,
        answer_text=correct_answer_text,
        answer_idx=answer_idx,
        categories_list=categories_list_str
    )

    # Tokenize the prompt
    inputs = tokenizer(full_prompt_text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

    predicted_category = "Error: Generation failed" # Default
    try:
        with torch.no_grad():
            # Generate response
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=30,
                do_sample=False, # For deterministic output
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode only the generated part, skipping the prompt
        response_ids = outputs[0][inputs.input_ids.shape[-1]:]
        generated_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        
        # The model should output only the category due to the prompt.
        # Take the first line in case it adds a newline.
        predicted_category = generated_text.split('\n')[0].strip()

        
        # Optional: Validate if the predicted_category is in your CATEGORIES list
        '''this is the part where model gave me some extra categories which was not defined in the above list'''

        if predicted_category not in CATEGORIES:
            print(f"\nWarning: Predicted category '{predicted_category}' not in predefined list for question: {question_text[:50]}...")
            # Heuristic: if the generated text is one of the categories but has extra characters, try to find it
            for cat in CATEGORIES:
                if cat.lower() in predicted_category.lower():
                    predicted_category = cat
                    print(f"  Matched to: {cat}")
                    break

    except Exception as e:
        print(f"\nError during generation for question '{question_text[:50]}...': {e}")
        predicted_category = f"Error: {str(e)}"

    new_item = item.copy()
    new_item['category'] = predicted_category
    results.append(new_item)

#  Save the data to CSV
output_df = pd.DataFrame(results)
desired_columns = ['question', 'options', 'answer_idx', 'answer', 'meta_info', 'metamap_phrases', 'category']
output_df = output_df.reindex(columns=desired_columns)

try:
    output_df.to_csv("dataset.csv", index=False, encoding='utf-8')
    print("\ndataset.csv")
except Exception as e:
    print(f"\nError saving CSV: {e}")

print("\nSample of the first 5 rows of the output:")
print(output_df.head())