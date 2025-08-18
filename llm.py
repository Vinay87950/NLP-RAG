'''
https://github.com/royranadeep/LLM-with-RAG-using-MedQA-Pubmed-wiki-/blob/main/Llama3_RAG_Wiki_source.ipynb

This code snippet was partially genearted by ChatGPT and then modified with the help of the reference that helped me to understand the concept of BitsAndBytesConfig 
and furthur implementtaion of code.
'''

import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm.auto import tqdm

# Set PyTorch memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class WikiRetriever:
    def __init__(
        self, 
        index_path="faiss_index.bin", 
        metadata_dir="vector_db",
        embedding_model="all-MiniLM-L6-v2",
        llm_model="microsoft/phi-3-mini-4k-instruct"
        ):
        # Load FAISS index
        print(f"Loading FAISS index from {index_path}")
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        print(f"Loading metadata from {metadata_dir}")
        with open(os.path.join(metadata_dir, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)
        
        with open(os.path.join(metadata_dir, "chunk_map.pkl"), "rb") as f:
            self.chunk_map = pickle.load(f)
        
        # Initialize embedding model on CPU
        print(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model, device="cpu")
        
        # Clear any existing CUDA memory
        torch.cuda.empty_cache()
        gc.collect()
        
        # Configure 4-bit quantization
        print(f"Loading LLM: {llm_model} with 4-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",  # normalized float 4
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True 
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        
        # Load model with 4-bit quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model,
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        # Clear cache after loading
        torch.cuda.empty_cache()
        gc.collect()
        
        print("Retriever initialized successfully")
    
    # Rest of your methods remain the same
    def retrieve(self, query, k=5):
        """Retrieve the top k most relevant chunks for a query"""
        # Encode the query on CPU
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        
        # Search the index
        distances, indices = self.index.search(query_embedding, k)
        
        # Collect the retrieved chunks and their metadata
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
                
            results.append({
                "chunk": self.chunks[idx],
                "distance": float(distances[0][i]),
                "metadata": self.chunk_map[idx]
            })
        
        return results
    
    def format_context(self, results):
        """Format retrieved chunks into a context string"""
        context = "Here is some relevant information:\n\n"
        
        for i, result in tqdm(enumerate(results)):
            # Limit chunk size to reduce context length
            chunk = result["chunk"]
            if len(chunk) > 100000:  # Limit chunk size but keep more content
                chunk = chunk[:100000] + "..."
                
            context += f"[{i+1}] {chunk}\n\n"
        
        return context


    # this was partially genearted by gemini-2.5 pro
    def generate_answer(self, query, options=None, category=None, k=5):
        """Generate an answer using retrieved context with advanced prompting techniques"""
        # Clear memory before starting
        torch.cuda.empty_cache()
        gc.collect()

        # Retrieve relevant chunks
        enhanced_query = f"{category}: {query}" if category else query
        results = self.retrieve(enhanced_query, k)
        context = self.format_context(results)

        # Load the prompt template from file
        with open("/home/6082/Ein/prompts/cot_prompt.txt", "r") as f:
            prompt_template = f.read()
        
        # Format the prompt with the specific query details
        category_text = f"{category}" if category else "General"
        
        # Handle options
        options_text = ""
        if not options:
            for result in results:
                if 'metadata' in result and 'options' in result['metadata'] and result['metadata']['options']:
                    options = result['metadata']['options']
                    break

        if options:
            options_text = "\n".join([
                f"{chr(65+i)}. {option}" 
                for i, option in enumerate(options)
            ])
        
        # Fill in the template
        prompt = prompt_template.format(
            context=context,
            question=query,
            category=category_text,
            options=options_text
        )

        # Generate answer
        torch.cuda.empty_cache()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=800,  # Increased for multi-hop reasoning else keep it 100, for normal prompt 
                temperature=0.8,
                top_p=0.95,
                do_sample=True
            )

        answer = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Clean up memory
        del outputs, inputs
        torch.cuda.empty_cache()
        gc.collect()

        return {
            "query": query,
            "category": category,
            "options": options,
            "context": context,
            "answer": answer,
            "retrieved_chunks": results
        }



# Example usage
if __name__ == "__main__":
    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()
    
    retriever = WikiRetriever()
    
    import pandas as pd
    df = pd.read_csv('/home/6082/Ein/chunked_wikipedia_data.csv')
    
    # Select an example
    example_idx = 0
    example = df.iloc[example_idx]
    
    # Extract question, options, and category
    question = example['question']
    
    # Parse options from string format if needed
    import json
    try:
        options = json.loads(example['options'])
    except:
        options = example['options'].split('|') if isinstance(example['options'], str) else example['options']
    
    # Get category if available
    category = example.get('category', None)
    
    # Generate answer
    result = retriever.generate_answer(question, options, category)
    
    print(f"Question: {question}")
    print(f"Category: {category}")
    print(f"Options: {options}")
    print("\nRetrieved Context:")
    print(result["context"])
    print("\nGenerated Answer:")
    print(result["answer"])
    
    # If your dataset has the correct answer, you can also evaluate
    if 'correct_index' in example or 'answer_idx' in example:
        correct_idx = example.get('correct_index', example.get('answer_idx'))
        print(f"\nCorrect Answer: {correct_idx}")