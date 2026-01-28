import pandas as pd

file_path = r'd:\project-work\01_nlp_toxicity_classification\data\train.csv'

print(f"Reading {file_path}...")
df = pd.read_csv(file_path, usecols=['target'])
total = len(df)

toxic_mask = df['target'] >= 0.5
clean_mask = df['target'] == 0
hard_negative_mask = (df['target'] > 0) & (df['target'] < 0.5)

n_toxic = toxic_mask.sum()
n_clean = clean_mask.sum()
n_hard_negative = hard_negative_mask.sum()

print(f"Total: {total}")
print("-" * 30)
print(f"Toxic (>= 0.5):            {n_toxic:<10} ({n_toxic/total*100:.2f}%)")
print(f"Non-Toxic (< 0.5):         {total - n_toxic:<10} ({(total - n_toxic)/total*100:.2f}%)")
print("-" * 30)
print("Detailed Breakdown:")
print(f"  Pure Non-Toxic (= 0):    {n_clean:<10} ({n_clean/total*100:.2f}%)")
print(f"  Hard Negative (0<x<0.5): {n_hard_negative:<10} ({n_hard_negative/total*100:.2f}%)")
