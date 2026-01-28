import pandas as pd
import os

data_dir = r'd:\project-work\01_nlp_toxicity_classification\data'
files = ['train.csv', 'all_data.csv']
target_cols = [
    'target', 'toxicity', 'severe_toxicity', 'obscene', 
    'sexual_explicit', 'identity_attack', 'insult', 'threat'
]

print(f"Checking files in {data_dir}...\n")

for f in files:
    path = os.path.join(data_dir, f)
    if not os.path.exists(path):
        continue
    
    print(f"Processing {f}...")
    try:
        use_cols = [c for c in target_cols if c in pd.read_csv(path, nrows=0).columns]
        if not use_cols:
            continue
            
        df = pd.read_csv(path, usecols=use_cols)
        
        print("Category Stats (Min | Max | Count >= 0.5):")
        for col in use_cols:
            c_min = df[col].min()
            c_max = df[col].max()
            count = (df[col] >= 0.5).sum()
            print(f"  {col:<15}: Min={c_min:.4f} | Max={c_max:.4f} | Count={count}")
        print("\n" + "="*40 + "\n")
        
    except Exception as e:
        print(f"Error: {e}\n")
