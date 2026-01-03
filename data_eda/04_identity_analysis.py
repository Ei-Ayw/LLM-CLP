"""
Identity Attributes Analysis
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

# Identity columns (bold ones in description are evaluated in competition)
identity_columns = [
    'male', 'female', 'transgender', 'other_gender',
    'heterosexual', 'homosexual_gay_or_lesbian', 'bisexual', 'other_sexual_orientation',
    'christian', 'jewish', 'muslim', 'hindu', 'buddhist', 'atheist', 'other_religion',
    'black', 'white', 'asian', 'latino', 'other_race_or_ethnicity',
    'physical_disability', 'intellectual_or_learning_disability', 
    'psychiatric_or_mental_illness', 'other_disability'
]

# Filter to existing columns
identity_columns = [col for col in identity_columns if col in df.columns]

print(f"\n{'='*60}")
print(f"IDENTITY ATTRIBUTES ANALYSIS")
print(f"{'='*60}")
print(f"Total identity columns: {len(identity_columns)}")

# Calculate statistics
identity_stats = []
for col in identity_columns:
    non_null = df[col].notna().sum()
    non_null_pct = (non_null / len(df)) * 100
    mentioned = (df[col] >= 0.5).sum()  # Mentioned if >= 0.5
    mentioned_pct = (mentioned / len(df)) * 100
    
    # Average toxicity for comments mentioning this identity
    if mentioned > 0:
        avg_toxicity = df[df[col] >= 0.5]['target'].mean()
    else:
        avg_toxicity = 0
    
    identity_stats.append({
        'Identity': col,
        'Non-Null Count': non_null,
        'Non-Null %': non_null_pct,
        'Mentioned Count': mentioned,
        'Mentioned %': mentioned_pct,
        'Avg Toxicity': avg_toxicity
    })

identity_df = pd.DataFrame(identity_stats).sort_values('Mentioned Count', ascending=False)

print(f"\nTop 10 Most Mentioned Identities:")
print(identity_df.head(10).to_string(index=False))

# Binary target
df['is_toxic'] = (df['target'] >= 0.5).astype(int)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# 1. Top identities by mention count
top_identities = identity_df.head(15)
axes[0, 0].barh(top_identities['Identity'], top_identities['Mentioned Count'], 
                color='steelblue', edgecolor='black', alpha=0.8)
axes[0, 0].set_xlabel('Mention Count', fontsize=12)
axes[0, 0].set_title('Top 15 Identity Mentions', fontsize=14, fontweight='bold')
axes[0, 0].invert_yaxis()
axes[0, 0].grid(True, alpha=0.3, axis='x')

# 2. Identity mention percentage
axes[0, 1].barh(top_identities['Identity'], top_identities['Mentioned %'], 
                color='coral', edgecolor='black', alpha=0.8)
axes[0, 1].set_xlabel('Mention Percentage (%)', fontsize=12)
axes[0, 1].set_title('Top 15 Identity Mention Percentage', fontsize=14, fontweight='bold')
axes[0, 1].invert_yaxis()
axes[0, 1].grid(True, alpha=0.3, axis='x')

# 3. Average toxicity by identity
toxicity_sorted = identity_df[identity_df['Mentioned Count'] > 1000].sort_values('Avg Toxicity', ascending=False).head(15)
colors = ['#e74c3c' if x > 0.5 else '#f39c12' if x > 0.3 else '#2ecc71' for x in toxicity_sorted['Avg Toxicity']]
axes[1, 0].barh(toxicity_sorted['Identity'], toxicity_sorted['Avg Toxicity'], 
                color=colors, edgecolor='black', alpha=0.8)
axes[1, 0].axvline(x=0.5, color='red', linestyle='--', linewidth=2, alpha=0.5)
axes[1, 0].set_xlabel('Average Toxicity Score', fontsize=12)
axes[1, 0].set_title('Avg Toxicity for Top Identities (>1000 mentions)', fontsize=14, fontweight='bold')
axes[1, 0].invert_yaxis()
axes[1, 0].grid(True, alpha=0.3, axis='x')

# 4. Distribution of identity mentions per comment
df['total_identities_mentioned'] = df[identity_columns].fillna(0).ge(0.5).sum(axis=1)
identity_counts = df['total_identities_mentioned'].value_counts().sort_index()
axes[1, 1].bar(identity_counts.index, identity_counts.values, 
               color='mediumpurple', edgecolor='black', alpha=0.8)
axes[1, 1].set_xlabel('Number of Identities Mentioned', fontsize=12)
axes[1, 1].set_ylabel('Frequency', fontsize=12)
axes[1, 1].set_title('Distribution of Identity Mentions per Comment', fontsize=14, fontweight='bold')
axes[1, 1].set_xlim(-0.5, 10.5)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_identity_analysis.png", dpi=150, bbox_inches='tight')
print(f"\n[OK] Visualization saved to {OUTPUT_DIR / '04_identity_analysis.png'}")

# Save detailed report
with open(OUTPUT_DIR / "04_identity_analysis.txt", "w", encoding="utf-8") as f:
    f.write("="*60 + "\n")
    f.write("IDENTITY ATTRIBUTES ANALYSIS\n")
    f.write("="*60 + "\n\n")
    
    f.write(f"Total identity columns analyzed: {len(identity_columns)}\n\n")
    
    f.write("All Identities (sorted by mention count):\n")
    f.write("-"*60 + "\n")
    f.write(identity_df.to_string(index=False))
    f.write("\n\n")
    
    f.write("="*60 + "\n")
    f.write("KEY INSIGHTS\n")
    f.write("="*60 + "\n")
    f.write(f"Most mentioned identity: {identity_df.iloc[0]['Identity']} ({identity_df.iloc[0]['Mentioned Count']:,} mentions)\n")
    f.write(f"Identity with highest avg toxicity: {toxicity_sorted.iloc[0]['Identity']} ({toxicity_sorted.iloc[0]['Avg Toxicity']:.4f})\n")
    f.write(f"Comments with 0 identities: {(df['total_identities_mentioned'] == 0).sum():,}\n")
    f.write(f"Comments with 1+ identities: {(df['total_identities_mentioned'] >= 1).sum():,}\n")

print(f"[OK] Report saved to {OUTPUT_DIR / '04_identity_analysis.txt'}")
print("\nDone!")
