"""
计算完整的偏见指标汇总：
1. Subgroup AUC
2. BPSN AUC  
3. BNSP AUC
4. Mean Bias AUC (平均偏见AUC)
5. Worst-group Bias AUC (最差子群偏见AUC)
"""
import os
import pandas as pd

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "src_result")
    
    # 找到所有公平性指标文件
    files = [f for f in os.listdir(results_dir) if f.startswith('metrics_fair_')]
    
    print("=" * 70)
    print("偏见指标汇总计算 (Bias Metrics Summary)")
    print("=" * 70)
    
    summary_records = []
    
    for f in sorted(files):
        df = pd.read_csv(os.path.join(results_dir, f))
        
        # 计算 Mean Bias AUC: 所有子群的三种AUC的平均值
        all_aucs = df[['subgroup_auc', 'bpsn_auc', 'bnsp_auc']].values.flatten()
        mean_bias_auc = all_aucs.mean()
        
        # 计算 Worst-group Bias AUC: 所有子群×三种AUC中的最小值
        worst_bias_auc = all_aucs.min()
        
        # 找出最差的子群和指标
        min_idx = all_aucs.argmin()
        worst_subgroup_idx = min_idx // 3
        worst_metric_idx = min_idx % 3
        worst_subgroup = df.iloc[worst_subgroup_idx]['subgroup']
        worst_metric = ['subgroup_auc', 'bpsn_auc', 'bnsp_auc'][worst_metric_idx]
        
        model_name = f.replace('metrics_fair_', '').replace('.csv', '')
        
        print(f"\n【{model_name}】")
        print("-" * 50)
        print(f"  Mean Bias AUC:        {mean_bias_auc:.4f}")
        print(f"  Worst-group Bias AUC: {worst_bias_auc:.4f}")
        print(f"  最差子群: {worst_subgroup} ({worst_metric})")
        
        # 打印各子群详细数据
        print("\n  各子群详细指标:")
        print(f"  {'Subgroup':<35} {'Sub-AUC':>10} {'BPSN-AUC':>10} {'BNSP-AUC':>10}")
        for _, row in df.iterrows():
            print(f"  {row['subgroup']:<35} {row['subgroup_auc']:>10.4f} {row['bpsn_auc']:>10.4f} {row['bnsp_auc']:>10.4f}")
        
        summary_records.append({
            'Model': model_name,
            'Mean_Bias_AUC': mean_bias_auc,
            'Worst_Bias_AUC': worst_bias_auc,
            'Worst_Subgroup': worst_subgroup,
            'Worst_Metric': worst_metric
        })
    
    # 创建汇总表
    print("\n" + "=" * 70)
    print("模型对比汇总表 (Model Comparison Summary)")
    print("=" * 70)
    
    summary_df = pd.DataFrame(summary_records)
    summary_df = summary_df.sort_values('Mean_Bias_AUC', ascending=False)
    
    print(f"\n{'Model':<45} {'Mean Bias AUC':>15} {'Worst Bias AUC':>15}")
    print("-" * 75)
    for _, row in summary_df.iterrows():
        print(f"{row['Model']:<45} {row['Mean_Bias_AUC']:>15.4f} {row['Worst_Bias_AUC']:>15.4f}")
    
    # 保存汇总结果
    output_path = os.path.join(results_dir, "summary_bias_metrics.csv")
    summary_df.to_csv(output_path, index=False)
    print(f"\n汇总结果已保存到: {output_path}")

if __name__ == "__main__":
    main()
