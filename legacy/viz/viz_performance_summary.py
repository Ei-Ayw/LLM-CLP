import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import json
import sys

# 设置项目路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src_script", "utils"))

from path_config import get_eval_path, EVAL_DIR, get_viz_path

def collect_metrics():
    """从 eval 目录收集所有模型的评估结果 (每种模型类型取最新的)"""
    all_json = glob.glob(os.path.join(EVAL_DIR, "*_metrics.json"))
    if not all_json:
        print("未发现评估结果文件 (.json)")
        return None
    
    # --- 新逻辑：按模型前缀分组，每种类型取最新的 ---
    MODEL_PREFIXES = [
        "DebertaV3MTL_S2",
        "BertCNNBiLSTM", 
        "VanillaBERT",
        "VanillaRoBERTa",
        "VanillaDeBERTa",
    ]
    
    # 按前缀分组
    grouped = {prefix: [] for prefix in MODEL_PREFIXES}
    for f in all_json:
        filename = os.path.basename(f)
        for prefix in MODEL_PREFIXES:
            if filename.startswith(prefix):
                grouped[prefix].append(f)
                break
    
    # 每组取修改时间最新的
    latest_files = []
    for prefix, files in grouped.items():
        if files:
            files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            latest_files.append(files[0])
    
    records = []
    for f in latest_files:
        try:
            with open(f, 'r', encoding='utf-8') as jf:
                data = json.load(jf)
            
            # 提取模型名称 (简化版)
            filename = os.path.basename(f)
            model_name = filename.replace("_metrics.json", "").split("_Sample")[0]
            
            # 提取指标
            records.append({
                'Model': model_name,
                'F1 (Optimal)': data['primary_metrics_optimal']['f1'],
                'F1 (Fixed 0.5)': data['primary_metrics_fixed_0.5']['f1'],
                'ROC-AUC': data['primary_metrics_optimal']['roc_auc'],
                'Mean Bias AUC': data['bias_metrics']['mean_bias_auc']
            })
        except Exception as e:
            print(f"解析 {f} 出错: {e}")
            
    return pd.DataFrame(records)

def plot_performance_summary(df):
    """绘制性能对比摘要图"""
    if df is None or len(df) == 0: return

    # 绘制 Mean Bias AUC 对比 (公平性核心指标)
    plt.figure(figsize=(12, 6))
    df_sorted = df.sort_values(by='Mean Bias AUC', ascending=False)
    
    # 使用 Seaborn 绘制
    sns.set_style("whitegrid")
    ax = sns.barplot(data=df_sorted, x='Mean Bias AUC', y='Model', palette='viridis')
    
    plt.title('Performance Comparison: Mean Bias AUC (Higher is Better)', fontsize=14, fontweight='bold')
    plt.xlim(0.8, 1.0) # 假设都在这个区间
    plt.xlabel('Mean Bias AUC Score')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # 添加数值标注
    for i, v in enumerate(df_sorted['Mean Bias AUC']):
        ax.text(v + 0.005, i, f"{v:.4f}", color='black', va='center', fontweight='bold')

    plt.tight_layout()
    output_path = get_viz_path("model_performance_summary.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f">>> 性能总结图表已保存: {output_path}")

def main():
    print(">>> 启动实验性能总结报告生成器...")
    df = collect_metrics()
    if df is not None:
        print("\n实验指标汇总:")
        print(df.to_string(index=False))
        plot_performance_summary(df)
    else:
        print("未发现足够的数据来生成报告。")

if __name__ == "__main__":
    main()
