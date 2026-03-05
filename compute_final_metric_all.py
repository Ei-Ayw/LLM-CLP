"""
从已有评估 JSON 文件中，计算 Jigsaw 官方 Final Metric (Borkan et al., 2019).
无需重新推理，直接利用已保存的 per-subgroup AUC 数据.
"""
import os, json, glob
import numpy as np

EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src_result", "eval")


def power_mean(values, p=-5):
    values = np.array(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan
    values = np.clip(values, 1e-10, None)
    return float(np.power(np.mean(np.power(values, p)), 1.0 / p))


def compute_from_json(path):
    with open(path) as f:
        data = json.load(f)

    overall_auc = data['primary_metrics_optimal']['roc_auc']
    f1 = data['primary_metrics_optimal']['f1']
    details = data['bias_metrics']['per_subgroup_details']

    subgroup_aucs = [d['subgroup_auc'] for d in details]
    bpsn_aucs = [d['bpsn_auc'] for d in details]
    bnsp_aucs = [d['bnsp_auc'] for d in details]

    pm_sub = power_mean(subgroup_aucs, p=-5)
    pm_bpsn = power_mean(bpsn_aucs, p=-5)
    pm_bnsp = power_mean(bnsp_aucs, p=-5)
    bias_score = (pm_sub + pm_bpsn + pm_bnsp) / 3.0
    final_metric = 0.25 * overall_auc + 0.75 * bias_score

    # 找 worst identity + metric
    worst_val, worst_id, worst_type = 1.0, '', ''
    for d in details:
        for k in ['subgroup_auc', 'bpsn_auc', 'bnsp_auc']:
            if d[k] < worst_val:
                worst_val = d[k]
                worst_id = d['subgroup']
                worst_type = k

    return {
        'name': os.path.basename(path).replace('_metrics.json', ''),
        'Final': final_metric,
        'BiasScore': bias_score,
        'OverallAUC': overall_auc,
        'F1': f1,
        'PM_Subgroup': pm_sub,
        'PM_BPSN': pm_bpsn,
        'PM_BNSP': pm_bnsp,
        'MeanBias': data['bias_metrics']['mean_bias_auc'],
        'WorstBias': data['bias_metrics']['worst_group_bias_auc'],
        'WorstIdentity': worst_id,
        'WorstMetric': worst_type,
    }


def main():
    jsons = sorted(glob.glob(os.path.join(EVAL_DIR, "*_metrics.json")))
    if not jsons:
        print("No eval JSON files found.")
        return

    results = []
    for path in jsons:
        try:
            r = compute_from_json(path)
            results.append(r)
        except Exception as e:
            print(f"  [Skip] {os.path.basename(path)}: {e}")

    # 按 Final Metric 降序排列
    results.sort(key=lambda x: x['Final'], reverse=True)

    # 打印排名表
    print(f"\n{'='*120}")
    print(f"{'Rank':<5} {'Model':<58} {'Final':>7} {'BiasScore':>10} {'OvlAUC':>8} {'F1':>7} {'PM_Sub':>8} {'PM_BPSN':>9} {'PM_BNSP':>9} {'Worst':>12}")
    print(f"{'='*120}")
    for i, r in enumerate(results, 1):
        worst_info = f"{r['WorstIdentity'][:8]}({r['WorstMetric'][:3]})"
        print(f"{i:<5} {r['name']:<58} {r['Final']:.4f} {r['BiasScore']:.4f}     {r['OverallAUC']:.4f}  {r['F1']:.4f}  {r['PM_Subgroup']:.4f}   {r['PM_BPSN']:.4f}   {r['PM_BNSP']:.4f}  {worst_info}")
    print(f"{'='*120}")

    # 对比关键模型
    key_models = {}
    for r in results:
        name_lower = r['name'].lower()
        if 'vanilla' in name_lower and 'deberta' in name_lower and 'seed42' in name_lower:
            key_models['Vanilla DeBERTa'] = r
        elif 'paramfix' in name_lower:
            key_models['MTL ParamFix'] = r
        elif 'optimalv1' in name_lower:
            key_models['MTL OptimalV1'] = r
        elif 'iacd' in name_lower and 'e1v3' in name_lower:
            key_models['IACD E1v3'] = r

    if key_models:
        print(f"\n>>> 关键模型在新指标下的对比:")
        for name, r in key_models.items():
            print(f"  {name:<20}: Final={r['Final']:.4f} | BiasScore={r['BiasScore']:.4f} | AUC={r['OverallAUC']:.4f}")


if __name__ == "__main__":
    main()
