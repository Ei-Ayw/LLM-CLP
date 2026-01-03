"""
Target Distribution Analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Paths
DATA_DIR = Path(r"d:/project-work/01_nlp_toxicity_classification/data")
OUTPUT_DIR = Path(r"d:/project-work/01_nlp_toxicity_classification/data_eda")

print("Loading train.csv...")
df = pd.read_csv(DATA_DIR / "train.csv")

print(f"\n{'='*60}")
print(f"TARGET DISTRIBUTION ANALYSIS")
print(f"{'='*60}")

# Target statistics
target = df['target']
print(f"\nTarget Statistics:")
print(f"  Mean: {target.mean():.4f}")
print(f"  Median: {target.median():.4f}")
print(f"  Std Dev: {target.std():.4f}")
print(f"  Min: {target.min():.4f}")
print(f"  Max: {target.max():.4f}")

# Binary classification (threshold = 0.5)
toxic = (target >= 0.5).sum()
non_toxic = (target < 0.5).sum()
toxic_pct = (toxic / len(df)) * 100
non_toxic_pct = (non_toxic / len(df)) * 100

print(f"\nBinary Classification (threshold=0.5):")
print(f"  Toxic (>=0.5): {toxic:,} ({toxic_pct:.2f}%)")
print(f"  Non-Toxic (<0.5): {non_toxic:,} ({non_toxic_pct:.2f}%)")
print(f"  Imbalance Ratio: 1:{non_toxic/toxic:.2f}")

# Toxicity subtypes
subtypes = ['severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack', 'sexual_explicit']
print(f"\nToxicity Subtypes (mean scores):")
for subtype in subtypes:
    if subtype in df.columns:
        mean_score = df[subtype].mean()
        positive_count = (df[subtype] >= 0.5).sum()
        positive_pct = (positive_count / len(df)) * 100
        print(f"  {subtype:25s}: {mean_score:.4f} (>=0.5: {positive_count:,} = {positive_pct:.2f}%)")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Target distribution histogram
axes[0, 0].hist(target, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold=0.5')
axes[0, 0].set_xlabel('Target Score', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('Target Distribution', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Binary class distribution
class_labels = ['Non-Toxic\n(<0.5)', 'Toxic\n(>=0.5)']
class_counts = [non_toxic, toxic]
colors = ['#2ecc71', '#e74c3c']
axes[0, 1].bar(class_labels, class_counts, color=colors, edgecolor='black', alpha=0.8)
axes[0, 1].set_ylabel('Count', fontsize=12)
axes[0, 1].set_title('Binary Class Distribution', fontsize=14, fontweight='bold')
for i, (count, pct) in enumerate(zip(class_counts, [non_toxic_pct, toxic_pct])):
    axes[0, 1].text(i, count + 50000, f'{count:,}\n({pct:.1f}%)', 
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 3. Subtypes comparison
subtype_data = []
for subtype in subtypes:
    if subtype in df.columns:
        positive_count = (df[subtype] >= 0.5).sum()
        positive_pct = (positive_count / len(df)) * 100
        subtype_data.append({'Subtype': subtype.replace('_', ' ').title(), 'Percentage': positive_pct})

subtype_df = pd.DataFrame(subtype_data).sort_values('Percentage', ascending=True)
axes[1, 0].barh(subtype_df['Subtype'], subtype_df['Percentage'], color='coral', edgecolor='black', alpha=0.8)
axes[1, 0].set_xlabel('Percentage (%)', fontsize=12)
axes[1, 0].set_title('Toxicity Subtypes Distribution (>=0.5)', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='x')

# 4. Log-scale histogram for better visibility
axes[1, 1].hist(target, bins=50, color='steelblue', edgecolor='black', alpha=0.7, log=True)
axes[1, 1].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold=0.5')
axes[1, 1].set_xlabel('Target Score', fontsize=12)
axes[1, 1].set_ylabel('Frequency (log scale)', fontsize=12)
axes[1, 1].set_title('Target Distribution (Log Scale)', fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_target_distribution.png", dpi=150, bbox_inches='tight')
print(f"\n[OK] Visualization saved to {OUTPUT_DIR / '02_target_distribution.png'}")

# Save text report
with open(OUTPUT_DIR / "02_target_distribution.txt", "w", encoding="utf-8") as f:
    f.write("="*60 + "\n")
    f.write("TARGET DISTRIBUTION ANALYSIS\n")
    f.write("="*60 + "\n\n")
    
    f.write("Target Statistics:\n")
    f.write(f"  Mean: {target.mean():.4f}\n")
    f.write(f"  Median: {target.median():.4f}\n")
    f.write(f"  Std Dev: {target.std():.4f}\n")
    f.write(f"  Min: {target.min():.4f}\n")
    f.write(f"  Max: {target.max():.4f}\n\n")
    
    f.write("Binary Classification (threshold=0.5):\n")
    f.write(f"  Toxic (>=0.5): {toxic:,} ({toxic_pct:.2f}%)\n")
    f.write(f"  Non-Toxic (<0.5): {non_toxic:,} ({non_toxic_pct:.2f}%)\n")
    f.write(f"  Imbalance Ratio: 1:{non_toxic/toxic:.2f}\n\n")
    
    f.write("Toxicity Subtypes (percentage >=0.5):\n")
    for subtype in subtypes:
        if subtype in df.columns:
            positive_count = (df[subtype] >= 0.5).sum()
            positive_pct = (positive_count / len(df)) * 100
            f.write(f"  {subtype:25s}: {positive_pct:.2f}%\n")

print(f"[OK] Report saved to {OUTPUT_DIR / '02_target_distribution.txt'}")
print("\nDone!")
