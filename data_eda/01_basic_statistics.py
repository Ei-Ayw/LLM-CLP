"""
Basic Statistics and Data Overview
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Paths
DATA_DIR = Path(r"d:/project-work/01_nlp_toxicity_classification/data")
OUTPUT_DIR = Path(r"d:/project-work/01_nlp_toxicity_classification/data_eda")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("Loading train.csv (this may take a moment)...")
df = pd.read_csv(DATA_DIR / "train.csv")

print(f"\n{'='*60}")
print(f"DATASET OVERVIEW")
print(f"{'='*60}")
print(f"Total samples: {len(df):,}")
print(f"Total columns: {len(df.columns)}")
print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Basic info
print(f"\n{'='*60}")
print(f"COLUMN INFORMATION")
print(f"{'='*60}")
print(df.info())

# Check missing values
print(f"\n{'='*60}")
print(f"MISSING VALUES")
print(f"{'='*60}")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Column': missing.index,
    'Missing Count': missing.values,
    'Missing %': missing_pct.values
})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
print(missing_df.to_string(index=False))

# Save basic statistics
print(f"\n{'='*60}")
print(f"SAVING RESULTS")
print(f"{'='*60}")

with open(OUTPUT_DIR / "01_basic_statistics.txt", "w", encoding="utf-8") as f:
    f.write("="*60 + "\n")
    f.write("BASIC STATISTICS - TRAIN DATASET\n")
    f.write("="*60 + "\n\n")
    
    f.write(f"Total samples: {len(df):,}\n")
    f.write(f"Total columns: {len(df.columns)}\n")
    f.write(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n")
    
    f.write("="*60 + "\n")
    f.write("MISSING VALUES\n")
    f.write("="*60 + "\n")
    f.write(missing_df.to_string(index=False))
    f.write("\n\n")
    
    f.write("="*60 + "\n")
    f.write("SAMPLE DATA (first 5 rows)\n")
    f.write("="*60 + "\n")
    f.write(df.head().to_string())

print(f"[OK] Basic statistics saved to {OUTPUT_DIR / '01_basic_statistics.txt'}")
print("\nDone!")
