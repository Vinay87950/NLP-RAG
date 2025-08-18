'''
https://github.com/Teddy-XiongGZ/MedRAG/blob/main/src/template.py

the above template i used to generate the prompts which i have saved in prompt folder
'''
from llm import *
import time 
import gc 
import re
import os
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
# Set PyTorch memory management
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Configuration
CHECKPOINT_FILE = "cot_eval_checkpoint.pkl"
CHECKPOINT_INTERVAL = 50
DATASET_PATH = '/home/6082/Ein/chunked_wikipedia_data.csv'


# this was genearted by chatgpt, because my retriever was giving the same option answer and it could extract out the right option
def extract_answer_letter(answer_text):
    """Extract the answer letter (A, B, C, D) from generated text"""
    # Look for explicit answer statements
    explicit_patterns = [
        r'answer is ([ABCD])',
        r'option ([ABCD]) is correct',
        r'the correct answer is ([ABCD])',
        r'conclusion[^.]*([ABCD])',
        r'final answer[^.]*([ABCD])'
    ]
    
    for pattern in explicit_patterns:
        matches = re.findall(pattern, answer_text, re.IGNORECASE)
        if matches:
            return matches[-1].upper()
    
    # Find all letter mentions
    all_matches = re.findall(r'\b([ABCD])\b', answer_text, re.IGNORECASE)
    if all_matches:
        # Get the most frequent letter in the last third of the text
        last_third = all_matches[int(len(all_matches)*2/3):]
        if last_third:
            # Return the most frequent letter
            return max(set(last_third), key=last_third.count).upper()
    
    return 'X'  # Invalid answer

def load_checkpoint():
    """Load checkpoint if exists"""
    if os.path.exists(CHECKPOINT_FILE):
        print(f"Resuming from checkpoint")
        with open(CHECKPOINT_FILE, 'rb') as f:
            return pickle.load(f)
    return {"correct": 0, "total": 0, "last_idx": -1, "invalid": 0}

def save_checkpoint(data, idx):
    """Save checkpoint to a new file with the current index"""
    filename = f"checkpoints/cot_eval_checkpoint_{idx}.pkl"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def evaluate_model(retriever, dataset):
    # Load checkpoint if exists
    checkpoint = load_checkpoint()
    correct = checkpoint["correct"]
    total = checkpoint["total"]
    last_idx = checkpoint["last_idx"]
    invalid = checkpoint["invalid"]
    
    # Start from where we left off
    start_idx = last_idx + 1
    
    progress_bar = tqdm(range(start_idx, len(dataset)), 
                         initial=start_idx, 
                         total=len(dataset), 
                         desc="Evaluating")
    
    for idx in progress_bar:
        example = dataset.iloc[idx]
        
        # Extract question and options
        question = example['question']
        
        # Parse options
        try:
            options = json.loads(example['options'])
        except:
            options = example['options'].split('|') if isinstance(example['options'], str) else example['options']
        
        # Get category if available
        category = example.get('category', None)
        
        # Get correct answer index
        correct_idx = example.get('correct_index', example.get('answer_idx'))

        # Normalize the correct answer
        if isinstance(correct_idx, str):
            correct_answer_letter = correct_idx.strip().upper()
        elif isinstance(correct_idx, int):
            correct_answer_letter = chr(65 + correct_idx)
        else:
            correct_answer_letter = 'X'
        
        # Generate answer
        result = retriever.generate_answer(question, options, category)
        answer_text = result["answer"]
        
        # Extract predicted letter
        predicted_answer_letter = extract_answer_letter(answer_text)
        
        # Update counters
        if predicted_answer_letter == correct_answer_letter:
            correct += 1
        if predicted_answer_letter == 'X':
            invalid += 1
        total += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            "Correct": f"{correct}/{total}", 
            "Invalid": f"{invalid}/{total}", 
            "Current Answer": predicted_answer_letter
        })
        
        # Save checkpoint periodically
        if (idx - start_idx + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint_data = {
                "correct": correct,
                "total": total,
                "last_idx": idx,
                "invalid": invalid
            }
            save_checkpoint(checkpoint_data, idx)
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            gc.collect()
    
    # Calculate final accuracy
    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total, invalid

if __name__ == "__main__":
    # Load dataset
    import pandas as pd
    dataset = pd.read_csv(DATASET_PATH)
    print(f"Loaded dataset with {len(dataset)} questions")
    
    # Initialize retriever
    retriever = WikiRetriever()
    
    # Evaluate model
    accuracy, correct, total, invalid = evaluate_model(retriever, dataset)
    
    # Print results
    print(f"\nEvaluation complete!")
    print(f"Total questions: {total}")
    print(f"Correct answers: {correct}")
    print(f"Invalid answers: {invalid}")
    print(f"Accuracy: {accuracy:.2%}")