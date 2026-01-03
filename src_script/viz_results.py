import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_fairness_metrics(metrics_csv, output_path):
    if not os.path.exists(metrics_csv):
        print(f"File {metrics_csv} not found.")
        return
        
    df = pd.read_csv(metrics_csv)
    # Melting for easier plotting with seaborn
    df_melted = df.melt(id_vars='subgroup', var_name='Metric', value_name='AUC')
    
    plt.figure(figsize=(15, 8))
    sns.barplot(data=df_melted, x='subgroup', y='AUC', hue='Metric')
    plt.title('Fairness Metrics per Identity Subgroup')
    plt.xticks(rotation=45)
    plt.ylim(0.5, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Fairness plot saved to {output_path}")

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RESULT_DIR = os.path.join(BASE_DIR, "src_result")
    FAIRNESS_CSV = os.path.join(RESULT_DIR, "res_fairness_metrics.csv")
    OUTPUT_PLOT = os.path.join(RESULT_DIR, "viz_fairness_auc.png")
    
    plot_fairness_metrics(FAIRNESS_CSV, OUTPUT_PLOT)

if __name__ == "__main__":
    main()
