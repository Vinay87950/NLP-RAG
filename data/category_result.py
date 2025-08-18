'''
to use this please load the dataset which we obtained from the file name data_scripting.py
'''

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('/home/6082/Ein/data/dataset.csv')

top_categories = df['category'].value_counts().head(21)
# low_categories = df['category'].value_counts().tail(10)

# histogram
plt.figure(figsize=(12, 8))
top_categories.plot(kind='bar')
# low_categories.plot(kind='bar')
plt.title('Categories')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('Categories')
plt.show()