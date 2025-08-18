'''
https://github.com/sn2727/medprompt-small-llms/blob/master/medprompt.ipynb

this repo have helped me to implement the chunking, the chunking function is quite similar but not as exact I have modified some of the snippets.

'''
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import re
import ast
import json

def clean_wiki_content(content):
    """Clean Wikipedia content from the specific format in your dataset"""
    if not content:
        return []
    
    # Handle string representation of list
    if isinstance(content, str):
        try:
            # Parse the string representation of a list
            content = ast.literal_eval(content)
        except:
            content = [content]
    
    # Ensure content is a list
    if not isinstance(content, list):
        content = [str(content)]
    
    # Clean each article
    cleaned_articles = []
    for article in content:
        if not article:
            continue
            
        # Remove escape characters
        article = article.replace('\\\'', "'")
        
        # Remove citations [1], [2], etc.
        article = re.sub(r'\[\d+\]', '', article)
        
        # Remove extra whitespace
        article = re.sub(r'\s+', ' ', article)
        
        cleaned_articles.append(article.strip())
    
    return cleaned_articles

def chunk_text(text, max_length=300):
    """Split text into chunks of approximately max_length words"""
    if not text or not isinstance(text, str):
        return []
    
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


'''
Utilised gemini-2.5- pro for code generation and modified a bit for my requirement.
'''
def process_dataframe(df, max_length=300):
    """Process dataframe to add chunks while maintaining original row structure"""
    # Create a copy of the dataframe to avoid modifying the original
    processed_df = df.copy()
    
    # Add new columns for chunks
    processed_df['cleaned_wiki_content'] = None
    processed_df['wiki_chunks'] = None
    processed_df['chunk_metadata'] = None
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Extract and clean Wikipedia content
        wiki_content = clean_wiki_content(row['wikipedia_content'])
        
        all_chunks = []
        chunk_metadata = []
        
        # Process each Wikipedia article
        for article_idx, article in enumerate(wiki_content):
            article_chunks = chunk_text(article, max_length)
            
            # Add metadata for each chunk
            for i, chunk in enumerate(article_chunks):
                chunk_info = {
                    'article_idx': article_idx,
                    'chunk_idx': i,
                    'chunk_char_count': len(chunk),
                    'chunk_word_count': len(chunk.split()),
                    'chunk_token_count': round(len(chunk) / 4), #1 token ~= 4 chars
                    'chunk_number_in_doc': i + 1,
                    'total_chunks_for_doc': len(article_chunks)
                }
                chunk_metadata.append(chunk_info)
                all_chunks.append(chunk)
        
        # Store the cleaned content and chunks in the dataframe
        processed_df.at[idx, 'cleaned_wiki_content'] = json.dumps(wiki_content)
        processed_df.at[idx, 'wiki_chunks'] = json.dumps(all_chunks)
        # processed_df.at[idx, 'chunk_metadata'] = json.dumps(chunk_metadata)
    
    return processed_df

if __name__ == "__main__":
    # Load the data with Wikipedia content
    df = pd.read_csv('/home/6082/Ein/data_with_wikipedia_content.csv')
    print(f"Loaded dataframe with {len(df)} rows")
    
    # Process and chunk the data
    processed_df = process_dataframe(df)
    print(f"Processed {len(processed_df)} rows with chunks")
    
    # Save the processed data
    processed_df.to_csv('chunked_wikipedia_data.csv', index=False)
    print("Saved processed data to 'chunked_wikipedia_data.csv'")
    
    # Display sample
    print("\nSample processed row:")
    sample_row = processed_df.iloc[0]
    # print(f"Question: {sample_row['question']}")
    print(f"Number of chunks: {len(json.loads(sample_row['wiki_chunks']))}")
    print(f'Chunked data: {json.load(sample_row["wiki_chunks"])[0]}')
    # chunked_dataset = pd.read_csv('/home/6082/Ein/chunked_wikipedia_data.csv')
    # print(f'Length of wiki_chunks: {len(chunked_dataset["wiki_chunks"][0])}')
    # print(chunked_dataset['wiki_chunks'][0])
