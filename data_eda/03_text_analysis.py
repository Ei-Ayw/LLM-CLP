"""
Text Analysis - Length, Complexity, etc.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Paths
DATA_DIR = Path(r"d:/project-work/01_nlp_toxicity_classification/data")
OUTPUT_DIR = Path(r"d:/project-work/01_nlp_toxicity_classification/data_eda")

print("Loading train.csv...")
df = pd.read_csv(DATA_DIR / "train.csv")

print(f"\n{'='*60}")
print(f"TEXT ANALYSIS")
print(f"{'='*60}")

# Text statistics
print("\nCalculating text statistics...")
df['text_length'] = df['comment_text'].fillna('').str.len()
df['word_count'] = df['comment_text'].fillna('').str.split().str.len()
df['avg_word_length'] = df['comment_text'].fillna('').apply(
    lambda x: np.mean([len(word) for word in x.split()]) if len(x.split()) > 0 else 0
)

# Binary target
df['is_toxic'] = (df['target'] >= 0.5).astype(int)

print(f"\nText Length Statistics:")
print(f"  Mean: {df['text_length'].mean():.2f} chars")
print(f"  Median: {df['text_length'].median():.2f} chars")
print(f"  Max: {df['text_length'].max()} chars")

print(f"\nWord Count Statistics:")
print(f"  Mean: {df['word_count'].mean():.2f} words")
print(f"  Median: {df['word_count'].median():.2f} words")
print(f"  Max: {df['word_count'].max()} words")

# Compare toxic vs non-toxic
print(f"\n{'='*60}")
print(f"TOXIC vs NON-TOXIC COMPARISON")
print(f"{'='*60}")

toxic_df = df[df['is_toxic'] == 1]
non_toxic_df = df[df['is_toxic'] == 0]

print(f"\nText Length:")
print(f"  Toxic:     Mean={toxic_df['text_length'].mean():.2f}, Median={toxic_df['text_length'].median():.2f}")
print(f"  Non-Toxic: Mean={non_toxic_df['text_length'].mean():.2f}, Median={non_toxic_df['text_length'].median():.2f}")

print(f"\nWord Count:")
print(f"  Toxic:     Mean={toxic_df['word_count'].mean():.2f}, Median={toxic_df['word_count'].median():.2f}")
print(f"  Non-Toxic: Mean={non_toxic_df['word_count'].mean():.2f}, Median={non_toxic_df['word_count'].median():.2f}")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Text length distribution
axes[0, 0].hist([non_toxic_df['text_length'], toxic_df['text_length']], 
                bins=50, label=['Non-Toxic', 'Toxic'], 
                color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('Text Length (characters)', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('Text Length Distribution', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=11)
axes[0, 0].set_xlim(0, 2000)  # Limit x-axis for better visualization
axes[0, 0].grid(True, alpha=0.3)

# 2. Word count distribution
axes[0, 1].hist([non_toxic_df['word_count'], toxic_df['word_count']], 
                bins=50, label=['Non-Toxic', 'Toxic'], 
                color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('Word Count', fontsize=12)
axes[0, 1].set_ylabel('Frequency', fontsize=12)
axes[0, 1].set_title('Word Count Distribution', fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=11)
axes[0, 1].set_xlim(0, 400)  # Limit x-axis
axes[0, 1].grid(True, alpha=0.3)

# 3. Box plot comparison - Text Length
data_to_plot = [non_toxic_df['text_length'].clip(upper=1500), 
                toxic_df['text_length'].clip(upper=1500)]
bp1 = axes[1, 0].boxplot(data_to_plot, labels=['Non-Toxic', 'Toxic'],
                          patch_artist=True, showmeans=True)
for patch, color in zip(bp1['boxes'], ['#2ecc71', '#e74c3c']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1, 0].set_ylabel('Text Length (characters)', fontsize=12)
axes[1, 0].set_title('Text Length Comparison (Box Plot)', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 4. Box plot comparison - Word Count
data_to_plot = [non_toxic_df['word_count'].clip(upper=300), 
                toxic_df['word_count'].clip(upper=300)]
bp2 = axes[1, 1].boxplot(data_to_plot, labels=['Non-Toxic', 'Toxic'],
                          patch_artist=True, showmeans=True)
for patch, color in zip(bp2['boxes'], ['#2ecc71', '#e74c3c']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1, 1].set_ylabel('Word Count', fontsize=12)
axes[1, 1].set_title('Word Count Comparison (Box Plot)', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_text_analysis.png", dpi=150, bbox_inches='tight')
print(f"\n[OK] Visualization saved to {OUTPUT_DIR / '03_text_analysis.png'}")

# Save text report
with open(OUTPUT_DIR / "03_text_analysis.txt", "w", encoding="utf-8") as f:
    f.write("="*60 + "\n")
    f.write("TEXT ANALYSIS\n")
    f.write("="*60 + "\n\n")
    
    f.write("Overall Statistics:\n")
    f.write(f"  Text Length - Mean: {df['text_length'].mean():.2f}, Median: {df['text_length'].median():.2f}\n")
    f.write(f"  Word Count - Mean: {df['word_count'].mean():.2f}, Median: {df['word_count'].median():.2f}\n\n")
    
    f.write("Toxic vs Non-Toxic Comparison:\n")
    f.write(f"  Text Length:\n")
    f.write(f"    Toxic:     Mean={toxic_df['text_length'].mean():.2f}, Median={toxic_df['text_length'].median():.2f}\n")
    f.write(f"    Non-Toxic: Mean={non_toxic_df['text_length'].mean():.2f}, Median={non_toxic_df['text_length'].median():.2f}\n\n")
    f.write(f"  Word Count:\n")
    f.write(f"    Toxic:     Mean={toxic_df['word_count'].mean():.2f}, Median={toxic_df['word_count'].median():.2f}\n")
    f.write(f"    Non-Toxic: Mean={non_toxic_df['word_count'].mean():.2f}, Median={non_toxic_df['word_count'].median():.2f}\n")

print(f"[OK] Report saved to {OUTPUT_DIR / '03_text_analysis.txt'}")
print("\nDone!")
