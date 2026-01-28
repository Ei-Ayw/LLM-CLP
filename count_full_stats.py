import pandas as pd
import os

file_path = r'd:\project-work\01_nlp_toxicity_classification\data\train.csv'

# List of columns to check based on typical Civil Comments dataset structure
toxicity_cols = [
    'target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat', 'sexual_explicit'
]

identity_cols = [
    'asian', 'atheist', 'bisexual', 'black', 'buddhist', 'christian', 'female', 
    'heterosexual', 'hindu', 'homosexual_gay_or_lesbian', 'intellectual_or_learning_disability', 
    'jewish', 'latino', 'male', 'muslim', 'other_disability', 'other_gender', 
    'other_race_or_ethnicity', 'other_religion', 'other_sexual_orientation', 
    'physical_disability', 'psychiatric_or_mental_illness', 'transgender', 'white'
]

# Reaction columns are counts, not probabilities, but might be interesting to summary stats
reaction_cols = ['funny', 'wow', 'sad', 'likes', 'disagree']

print(f"Processing {file_path}...")

try:
    # Check headers first
    df_head = pd.read_csv(file_path, nrows=0)
    all_cols = toxicity_cols + identity_cols
    
    # Filter to only existing columns
    existing_cols = [c for c in all_cols if c in df_head.columns]
    existing_reaction = [c for c in reaction_cols if c in df_head.columns]
    
    print("Reading file...")
    # Load data
    df = pd.read_csv(file_path, usecols=existing_cols + existing_reaction)
    total_rows = len(df)
    print(f"Total Rows: {total_rows}\n")
    
    # 1. Toxicity Categories
    print("--- Toxicity Categories (Score >= 0.5) ---")
    print(f"{'Category':<35} | {'Count':<10} | {'Percentage':<10}")
    print("-" * 65)
    for col in toxicity_cols:
        if col in df.columns:
            count = (df[col] >= 0.5).sum()
            pct = (count / total_rows) * 100
            print(f"{col:<35} | {count:<10} | {pct:.2f}%")
    print("\n")

    # 2. Identity Attributes
    print("--- Identity Attributes (Score >= 0.5) ---")
    print("Note: Identity tags are only present in a subset of the data (annotated rows).")
    # Identities are often NaN for rows that weren't annotated for identity. 
    # Valid count is non-null rows.
    
    # Check how many rows have ANY identity annotation (checking one common col like 'male' for nulls)
    identity_annotated_count = df['male'].notna().sum()
    print(f"Rows with Identity Annotations: {identity_annotated_count} ({(identity_annotated_count/total_rows)*100:.2f}% of total)")
    print(f"{'Category':<35} | {'Count':<10} | {'% of Total':<10} | {'% of Annotated'}")
    print("-" * 80)
    
    for col in identity_cols:
        if col in df.columns:
            count = (df[col] >= 0.5).sum()
            pct_total = (count / total_rows) * 100
            pct_annotated = (count / identity_annotated_count * 100) if identity_annotated_count > 0 else 0
            print(f"{col:<35} | {count:<10} | {pct_total:.2f}%      | {pct_annotated:.2f}%")
    print("\n")
    
    # 3. Reactions (Stats)
    if existing_reaction:
        print("--- User Reactions (Non-zero Counts) ---")
        print(f"{'Category':<35} | {'Count > 0':<10} | {'Max Value':<10}")
        print("-" * 65)
        for col in existing_reaction:
            count = (df[col] > 0).sum()
            max_val = df[col].max()
            print(f"{col:<35} | {count:<10} | {max_val}")
    
except Exception as e:
    print(f"Error: {e}")
