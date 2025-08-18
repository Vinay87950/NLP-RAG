'''
1. https://zilliz.com/learn/beginner-guide-to-website-chunking-and-embedding-for-your-genai-applications
this reference has played a important role for writing this script.

2. https://github.com/attardi/wikiextractor/tree/master/wikiextractor

3. https://moussakam.github.io/demo/2024/09/26/arabic-rag.html


for this particular script to implement i utilised 'https://www.microsoft.com/en-us/download/details.aspx?id=52419' - Microsoft Research WikiQA Corpus code
their code is not openly avialble, so one should need to download it 'script "process_data.py" there didn't particularly implemented the same but I got idea from it.

'''

import pandas as pd
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor # 3. repo have helped me to understand the concept of threading and how to use it for efficient web scraping
import re
from urllib.parse import quote
from pprint import pprint
from tqdm.auto import tqdm

# Global cache for efficient requests
PAGE_CACHE = {}

def clean_phrase(phrase):
    """Prepare phrase for Wikipedia URL"""
    return quote(re.sub(r'[^\w\s-]', '', phrase.strip()).replace(' ', '_'))

def fetch_wiki_page(title):
    """Retrieve and parse Wikipedia content"""
    if title in PAGE_CACHE:
        return PAGE_CACHE[title]
    
    try:
        url = f'https://en.wikipedia.org/wiki/{title}'
        response = requests.get(url, headers={'User-Agent': 'wikipedia content for multi-hop questions'}, timeout=5)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        content_div = soup.find('div', id='bodyContent')
        
        if not content_div:
            return ""
        
        # Remove non-content elements
        for element in content_div.find_all(['table', 'div.infobox', 'span.reference']):
            element.decompose()
        
        # Extract text content
        return ' '.join(p.get_text().strip() for p in content_div.find_all('p'))
    
    except Exception:
        return ""

def get_wiki_content(phrases, max_pages=5):
    """Fetch content for multiple phrases with threading"""
    unique_phrases = []
    for phrase in tqdm(set(phrases)):
        if isinstance(phrase, str) and phrase.strip():
            clean = clean_phrase(phrase)
            if clean and clean not in unique_phrases:
                unique_phrases.append(clean)

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(fetch_wiki_page, unique_phrases[:max_pages]))
    
    valid_results = [res for res in results if res.strip()]
    for phrase, content in zip(unique_phrases, results):
        PAGE_CACHE[phrase] = content
    return valid_results[:max_pages]

def process_questions(df):
    """Main function to process dataframe"""
    results = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0]): 
        phrases = []
        if 'metamap_phrases' in df.columns and pd.notna(row.metamap_phrases):
            phrases.extend(str(row.metamap_phrases).split(','))
        if 'category' in df.columns and pd.notna(row.category):
            phrases.append(str(row.category))
        
        content = get_wiki_content(phrases)
        results.append({
            'question': row.question,
            'options': row.options,
            'correct_index': row.answer_idx,
            'category':row.category,
            'wikipedia_content': content
        })
    return results

# Example usage
if __name__ == "__main__":

    # loading the original dataset in csv form
    df = pd.read_csv('/home/6082/Ein/data/dataset.csv') 
    print("Dataset loaded successfully. Processing questions...")

    # 2. Process questions and get results
    enriched_data = process_questions(df)  
    print(f"Processed {len(enriched_data)} questions.")

    # 3. Convert the list of dictionaries to a pandas DataFrame
    enriched_df = pd.DataFrame(enriched_data)
    
    # 4. Save the DataFrame to a CSV file 
    csv_file_path = 'data_with_wikipedia_content.csv'
    enriched_df.to_csv(csv_file_path, index=False)
    print(f"Enriched data saved to {csv_file_path}")

    # You can still pprint the first two entries to check
    print("\nFirst two entries from the processed data:")
    pprint(enriched_data[:2])

    'once the data is stored through above script, you can load it and check the content'
    # df = pd.read_csv('/home/6082/Ein/data_with_wikipedia_content.csv')
    # print(df.head())


