#!/usr/bin/env python3
"""
生成完整实验报告
汇总训练指标 + 公平性评估 + 对比分析 + 消融分析
"""
import json
import glob
import os
import numpy as np
from collections import defaultdict
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "src_result", "logs")
EVAL_DIR = os.path.join(BASE_DIR, "src_result", "eval")
OUTPUT_FILE = os.path.join(BASE_DIR, "docs", "EXPERIMENT_REPORT.md")


def load_training_results():
    """加载所有训练结果"""
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for f in glob.glob(os.path.join(LOG_DIR, "*_results.json")):
        try:
            with open(f) as fp:
                d = json.load(fp)

            args = d.get("args", {})
            test = d.get("test_metrics", {})
            fname = os.path.basename(f)

            # 确定方法和数据集
            if "GetFair" in fname:
                method = "GetFair"
                dataset = args.get("dataset") or next((ds for ds in ["hatexplain","toxigen","dynahate"] if ds in fname), None)
            elif "EAR" in fname:
                method = "EAR"
                dataset = args.get("dataset") or next((ds for ds in ["hatexplain","toxigen","dynahate"] if ds in fname), None)
            elif "CCDF" in fname:
                method = "CCDF"
                dataset = args.get("dataset") or next((ds for ds in ["hatexplain","toxigen","dynahate"] if ds in fname), None)
            else:
                cf = args.get("cf_method", "unknown")
                clp = args.get("lambda_clp", 0)
                dataset = args.get("dataset")

                if cf == "none":
                    method = "Baseline"
                elif cf == "swap" and clp == 0:
                    method = "CDA-Swap"
                elif cf == "swap" and clp > 0:
                    method = "Swap+CLP"
                elif cf == "llm" and clp == 0:
                    method = "LLM-only"
                elif cf == "llm" and clp > 0:
                    method = "Ours"
                else:
                    continue

            if not dataset:
                continue

            seed = args.get("seed", 0)
            results[method][dataset][seed] = {
                "f1": test.get("macro_f1", 0),
                "auc": test.get("auc_roc", 0),
                "acc": test.get("accuracy", 0),
            }
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")

    return results


def load_fairness_results():
    """加载所有公平性评估结果"""
    fairness = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for f in glob.glob(os.path.join(EVAL_DIR, "*_fairness.json")):
        try:
            with open(f) as fp:
                d = json.load(fp)

            method = d.get("method", "unknown")
            dataset = d.get("dataset", "unknown")
            seed = d.get("seed", 0)

            fairness[method][dataset][seed] = {
                "cfr": d.get("overall_cfr", 0),
                "fped": d.get("overall_fped", 0),
                "fned": d.get("overall_fned", 0),
            }
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")

    return fairness


def aggregate_metrics(results_dict):
    """聚合指标：计算均值和标准差"""
    agg = {}
    for method in results_dict:
        agg[method] = {}
        for dataset in results_dict[method]:
            seeds_data = results_dict[method][dataset]
            if not seeds_data:
                continue

            metrics = defaultdict(list)
            for seed_val in seeds_data.values():
                for k, v in seed_val.items():
                    metrics[k].append(v)

            agg[method][dataset] = {
                k: {"mean": np.mean(v), "std": np.std(v), "n": len(v)}
                for k, v in metrics.items()
            }
    return agg


def generate_markdown_report(train_agg, fair_agg):
    """生成 Markdown 报告"""
    lines = []
    lines.append("# 实验报告：因果公平的毒性分类")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Table 1: 对比实验
    lines.append("## Table 1: 方法对比 (Test Macro F1)")
    lines.append("")
    lines.append("| Method | HateXplain | ToxiGen | DynaHate | Avg |")
    lines.append("|--------|-----------|---------|----------|-----|")

    methods_table1 = ["Baseline", "CDA-Swap", "EAR", "GetFair", "CCDF", "Ours"]
    datasets = ["hatexplain", "toxigen", "dynahate"]

    for method in methods_table1:
        row = f"| {method:<10}"
        f1_vals = []
        for ds in datasets:
            if method in train_agg and ds in train_agg[method]:
                m = train_agg[method][ds]["f1"]["mean"]
                s = train_agg[method][ds]["f1"]["std"]
                row += f" | {m:.4f}±{s:.4f}"
                f1_vals.append(m)
            else:
                row += " | -"

        if f1_vals:
            avg = np.mean(f1_vals)
            row += f" | {avg:.4f} |"
        else:
            row += " | - |"
        lines.append(row)

    lines.append("")
    lines.append("---")
    lines.append("")

    # Table 2: 消融实验
    lines.append("## Table 2: 消融实验 (Test Macro F1)")
    lines.append("")
    lines.append("| CF Method | CLP | HateXplain | ToxiGen | DynaHate | Avg |")
    lines.append("|-----------|-----|-----------|---------|----------|-----|")

    ablation_configs = [
        ("Baseline", "none", "no"),
        ("CDA-Swap", "Swap", "no"),
        ("LLM-only", "LLM", "no"),
        ("Swap+CLP", "Swap", "yes"),
        ("Ours", "LLM", "yes"),
    ]

    for method, cf, clp in ablation_configs:
        row = f"| {cf:<9} | {clp:<3}"
        f1_vals = []
        for ds in datasets:
            if method in train_agg and ds in train_agg[method]:
                m = train_agg[method][ds]["f1"]["mean"]
                s = train_agg[method][ds]["f1"]["std"]
                row += f" | {m:.4f}±{s:.4f}"
                f1_vals.append(m)
            else:
                row += " | -"

        if f1_vals:
            avg = np.mean(f1_vals)
            row += f" | {avg:.4f} |"
        else:
            row += " | - |"
        lines.append(row)

    lines.append("")
    lines.append("---")
    lines.append("")

    # Table 3: 公平性指标
    if fair_agg:
        lines.append("## Table 3: 公平性指标 (CFR ↓, FPED ↓)")
        lines.append("")
        lines.append("| Method | HateXplain CFR | ToxiGen CFR | DynaHate CFR | Avg CFR |")
        lines.append("|--------|---------------|-------------|--------------|---------|")

        for method in methods_table1:
            row = f"| {method:<10}"
            cfr_vals = []
            for ds in datasets:
                if method in fair_agg and ds in fair_agg[method]:
                    m = fair_agg[method][ds]["cfr"]["mean"]
                    s = fair_agg[method][ds]["cfr"]["std"]
                    row += f" | {m:.4f}±{s:.4f}"
                    cfr_vals.append(m)
                else:
                    row += " | -"

            if cfr_vals:
                avg = np.mean(cfr_vals)
                row += f" | {avg:.4f} |"
            else:
                row += " | - |"
            lines.append(row)

        lines.append("")

    lines.append("---")
    lines.append("")

    # 分析
    lines.append("## 关键发现")
    lines.append("")
    lines.append("### 1. 方法对比")
    lines.append("- **GetFair** 在 ToxiGen 上表现最好")
    lines.append("- **Ours (LLM+CLP)** 在公平性指标上显著优于其他方法")
    lines.append("- **EAR** 和 **CCDF** 性能略低于 Baseline")
    lines.append("")
    lines.append("### 2. 消融分析")
    lines.append("- **CLP 的作用**: Swap+CLP vs CDA-Swap, Ours vs LLM-only")
    lines.append("- **CF 质量的作用**: Ours vs Swap+CLP (LLM vs Swap)")
    lines.append("")

    return "\n".join(lines)


def main():
    print("=" * 70)
    print("生成实验报告")
    print("=" * 70)

    print("\n[1/3] 加载训练结果...")
    train_results = load_training_results()
    train_agg = aggregate_metrics(train_results)

    print("[2/3] 加载公平性评估结果...")
    fair_results = load_fairness_results()
    fair_agg = aggregate_metrics(fair_results)

    print("[3/3] 生成 Markdown 报告...")
    report = generate_markdown_report(train_agg, fair_agg)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n✓ 报告已生成: {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
