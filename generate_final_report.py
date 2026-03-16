#!/usr/bin/env python3
"""汇总所有 7metrics 评估结果，生成论文表格"""
import json, glob, os, re
import numpy as np
from collections import defaultdict

EVAL_DIR = "src_result/eval"

def parse_filename(fname):
    """从文件名提取 method, dataset, seed, cf_method"""
    base = os.path.basename(fname)
    # 新格式: Method_dataset_seedXX_MMDD_HHMM_7metrics_cf.json
    # Ours格式: dataset_llm_clp1.0_con0.0_seedXX_MMDD_HHMM_7metrics_cf.json

    cf_method = None
    if "_7metrics_swap.json" in base:
        cf_method = "swap"
    elif "_7metrics_llm.json" in base:
        cf_method = "llm"
    else:
        return None  # 旧格式，跳过

    # 提取 seed
    seed_match = re.search(r'seed(\d+)', base)
    if not seed_match:
        return None
    seed = int(seed_match.group(1))

    # 提取 method 和 dataset
    methods_map = {
        'Vanilla': 'Vanilla', 'EAR': 'EAR', 'GetFair': 'GetFair',
        'CCDF': 'CCDF', 'Davani': 'Davani', 'Ramponi': 'Ramponi'
    }

    method = None
    dataset = None
    for m in methods_map:
        if base.startswith(m + "_"):
            method = m
            # dataset 在 method 后面
            rest = base[len(m)+1:]
            for ds in ['hatexplain', 'toxigen', 'dynahate']:
                if rest.startswith(ds):
                    dataset = ds
                    break
            break

    # Ours: 文件名以 dataset 开头
    if method is None:
        for ds in ['hatexplain', 'toxigen', 'dynahate']:
            if base.startswith(ds + "_llm_clp"):
                method = 'Ours'
                dataset = ds
                break

    if method and dataset and cf_method:
        return method, dataset, seed, cf_method
    return None

# 收集所有结果
results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
# results[method][dataset][cf_method] = [(seed, metrics_dict), ...]

files = glob.glob(os.path.join(EVAL_DIR, "*_7metrics_*.json"))
print(f"找到 {len(files)} 个评估文件")

skipped = 0
for f in sorted(files):
    parsed = parse_filename(f)
    if not parsed:
        skipped += 1
        continue
    method, dataset, seed, cf_method = parsed
    with open(f) as fp:
        data = json.load(fp)
    results[method][dataset][cf_method].append((seed, data))

print(f"跳过 {skipped} 个旧格式文件")

# 指标列表
METRICS = ['accuracy', 'macro_f1', 'auc_roc', 'cfr', 'ctfg', 'fped', 'fned']
METRIC_NAMES = ['Accuracy', 'Macro-F1', 'AUC-ROC', 'CFR↓', 'CTFG↓', 'FPED↓', 'FNED↓']
METHOD_ORDER = ['Vanilla', 'EAR', 'GetFair', 'CCDF', 'Davani', 'Ramponi', 'Ours']
DATASETS = ['hatexplain', 'toxigen', 'dynahate']

def compute_avg_std(entries, metric):
    vals = []
    for seed, data in entries:
        v = data.get(metric)
        if v is not None:
            vals.append(v)
    if not vals:
        return None, None
    return np.mean(vals), np.std(vals)

# 生成表格
for cf in ['swap', 'llm']:
    cf_label = "Swap-based" if cf == "swap" else "LLM-based"
    print(f"\n{'='*120}")
    print(f"Counterfactual Method: {cf_label}")
    print(f"{'='*120}")

    for ds in DATASETS:
        print(f"\n--- {ds.upper()} ---")
        header = f"| {'Method':<10} |"
        for mn in METRIC_NAMES:
            header += f" {mn:>16} |"
        print(header)
        print("|" + "-"*12 + "|" + ("|".join(["-"*18]*7)) + "|")

        for method in METHOD_ORDER:
            entries = results.get(method, {}).get(ds, {}).get(cf, [])
            if not entries:
                row = f"| {method:<10} |" + " N/A              |" * 7
                print(row)
                continue

            row = f"| {method:<10} |"
            for metric in METRICS:
                avg, std = compute_avg_std(entries, metric)
                if avg is not None:
                    row += f" {avg:.4f}±{std:.4f} |"
                else:
                    row += f" {'N/A':>16} |"
            print(row)

# LaTeX 表格
print("\n\n" + "="*120)
print("LaTeX Tables for Paper")
print("="*120)

for cf in ['llm']:  # 论文主表用 LLM counterfactual
    cf_label = "LLM-based"
    for ds in DATASETS:
        print(f"\n% Table: {ds} ({cf_label} counterfactual)")
        print(r"\begin{table}[t]")
        print(r"\centering")
        print(r"\small")
        print(f"\\caption{{Results on {ds.upper()} with {cf_label} counterfactuals. Best in \\textbf{{bold}}.}}")
        print(r"\begin{tabular}{l|ccc|cccc}")
        print(r"\toprule")
        print(r"Method & Acc & F1 & AUC & CFR$\downarrow$ & CTFG$\downarrow$ & FPED$\downarrow$ & FNED$\downarrow$ \\")
        print(r"\midrule")

        # 找每列最优值
        best = {}
        for metric in METRICS:
            vals = []
            for method in METHOD_ORDER:
                entries = results.get(method, {}).get(ds, {}).get(cf, [])
                avg, _ = compute_avg_std(entries, metric)
                if avg is not None:
                    vals.append((avg, method))
            if vals:
                if metric in ['cfr', 'ctfg', 'fped', 'fned']:
                    best[metric] = min(vals, key=lambda x: x[0])[1]
                else:
                    best[metric] = max(vals, key=lambda x: x[0])[1]

        for method in METHOD_ORDER:
            entries = results.get(method, {}).get(ds, {}).get(cf, [])
            if not entries:
                continue

            row = f"{method} "
            for metric in METRICS:
                avg, std = compute_avg_std(entries, metric)
                if avg is not None:
                    val_str = f"{avg:.4f}"
                    if best.get(metric) == method:
                        val_str = f"\\textbf{{{val_str}}}"
                    row += f"& {val_str} "
                else:
                    row += "& -- "
            row += r"\\"
            if method == 'Ramponi':
                row += r" \midrule"
            print(row)

        print(r"\bottomrule")
        print(r"\end{tabular}")
        print(r"\end{table}")

# 汇总 JSON
summary = {}
for method in METHOD_ORDER:
    summary[method] = {}
    for ds in DATASETS:
        summary[method][ds] = {}
        for cf in ['swap', 'llm']:
            entries = results.get(method, {}).get(ds, {}).get(cf, [])
            metrics_avg = {}
            for metric in METRICS:
                avg, std = compute_avg_std(entries, metric)
                if avg is not None:
                    metrics_avg[metric] = {'mean': round(avg, 4), 'std': round(std, 4)}
            metrics_avg['n_seeds'] = len(entries)
            summary[method][ds][cf] = metrics_avg

with open("src_result/eval/summary_all_7metrics.json", 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print("\n\n汇总 JSON 已保存至 src_result/eval/summary_all_7metrics.json")
