import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

def plot_comparison(result_dir, output_path):
    all_files = glob.glob(os.path.join(result_dir, "metrics_fair_*.csv"))
    if not all_files:
        print("No fairness metrics files found.")
        return
    
    dfs = []
    for f in all_files:
        model_name = os.path.basename(f).replace("metrics_fair_", "").replace(".csv", "")
        df = pd.read_csv(f)
        df['Model'] = model_name
        dfs.append(df)
    
    total_df = pd.concat(dfs)
    
    # Calculate Mean Bias AUC for each model
    summary = []
    for model in total_df['Model'].unique():
        m_df = total_df[total_df['Model'] == model]
        mean_auc = m_df[['subgroup_auc', 'bpsn_auc', 'bnsp_auc']].values.mean()
        summary.append({'Model': model, 'Mean_Bias_AUC': mean_auc})
    
    summary_df = pd.DataFrame(summary).sort_values(by='Mean_Bias_AUC', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=summary_df, x='Mean_Bias_AUC', y='Model', hue='Model', palette='viridis', legend=False)
    plt.title('Comparison of Mean Bias AUC across Models', fontsize=14)
    plt.xlim(0.7, 1.0)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Comparison plot saved to {output_path}")

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RESULT_DIR = os.path.join(BASE_DIR, "src_result")
    OUTPUT_PLOT = os.path.join(RESULT_DIR, "comparison_fairness_auc.png")
    
    plot_comparison(RESULT_DIR, OUTPUT_PLOT)

if __name__ == "__main__":
    main()
