import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import seaborn as sns

# Load the processed data
chunked_dataset = pd.read_csv('/home/6082/Ein/chunked_wikipedia_data.csv')
print(f"Loaded dataset with {len(chunked_dataset)} rows")

# Function to extract chunk information
def analyze_chunks(dataset):
    # Lists to store analysis data
    chunks_per_row = []
    chunk_lengths = []
    chunk_word_counts = []
    
    # Process each row
    for i, row in enumerate(dataset['wiki_chunks']):
        try:
            # Parse the JSON string to get the chunks
            chunks = json.loads(row)
            
            # Count chunks per row
            chunks_per_row.append(len(chunks))
            
            # Analyze individual chunks
            for chunk in chunks:
                chunk_lengths.append(len(chunk))
                chunk_word_counts.append(len(chunk.split()))
                
        except Exception as e:
            print(f"Error processing row {i}: {e}")
    
    return chunks_per_row, chunk_lengths, chunk_word_counts

# Analyze the data
chunks_per_row, chunk_lengths, chunk_word_counts = analyze_chunks(chunked_dataset)

# Create a figure with multiple subplots
plt.figure(figsize=(15, 12))

# 1. Distribution of chunks per row
plt.subplot(2, 2, 1)
plt.hist(chunks_per_row, bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Chunks per Wikipedia Entry')
plt.xlabel('Number of Chunks')
plt.ylabel('Frequency')

# 2. Distribution of chunk lengths (characters)
plt.subplot(2, 2, 2)
plt.hist(chunk_lengths, bins=20, color='lightgreen', edgecolor='black')
plt.title('Distribution of Chunk Lengths')
plt.xlabel('Number of Characters')
plt.ylabel('Frequency')

# 3. Distribution of word counts per chunk
plt.subplot(2, 2, 3)
plt.hist(chunk_word_counts, bins=20, color='salmon', edgecolor='black')
plt.title('Distribution of Word Counts per Chunk')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')

# 4. Scatter plot of chunk length vs. word count
plt.subplot(2, 2, 4)
plt.scatter(chunk_lengths[:1000], chunk_word_counts[:1000], alpha=0.5, color='purple')
plt.title('Chunk Length vs. Word Count (Sample of 1000 chunks)')
plt.xlabel('Number of Characters')
plt.ylabel('Number of Words')

# Adjust layout and save
plt.tight_layout()
# plt.savefig('chunk_analysis.png')
plt.show()

