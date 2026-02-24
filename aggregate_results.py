#!/usr/bin/env python3
"""
=============================================================================
### 结果汇总脚本：aggregate_results.py ###
=============================================================================
功能：
1. 读取 src_result/eval/ 下所有 *_metrics.json
2. 按实验类型分组（多seed / 消融 / 敏感性 / baseline）
3. 多seed 实验计算均值 +/- 标准差
4. 输出 Markdown 表格 + LaTeX 表格
=============================================================================
"""

import os
import json
import sys
from collections import defaultdict
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(BASE_DIR, "src_result", "eval")
OUTPUT_DIR = os.path.join(BASE_DIR, "src_result")

# 关键指标
KEY_METRICS = [
    ("roc_auc", "ROC-AUC"),
    ("best_f1", "F1"),
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("mean_bias_auc", "Mean Bias AUC"),
    ("worst_group_bias_auc", "Worst Group AUC"),
]


def load_all_metrics():
    """加载所有评估结果 JSON 文件。"""
    results = {}
    if not os.path.exists(EVAL_DIR):
        print(f"[Error] 评估目录不存在: {EVAL_DIR}")
        return results

    for f in sorted(os.listdir(EVAL_DIR)):
        if not f.endswith("_metrics.json"):
            continue
        name = f.replace("_metrics.json", "")
        with open(os.path.join(EVAL_DIR, f), 'r') as fp:
            try:
                data = json.load(fp)
                results[name] = data
            except json.JSONDecodeError:
                print(f"[Warning] 无法解析: {f}")
    return results


def extract_seed(name):
    """从文件名提取 seed 值。"""
    import re
    m = re.search(r"Seed(\d+)", name)
    if m:
        return int(m.group(1))
    return 42  # 旧命名无 Seed 标识，默认 42


def classify_results(results):
    """将结果按实验类型分组。"""
    groups = {
        "mtl_s2_seeds": [],       # MTL S2 多seed (含seed=42)
        "vanilla_deberta_seeds": [],  # Vanilla DeBERTa 多seed
        "ablation": [],           # 消融实验
        "sensitivity": [],        # 超参敏感性
        "baselines": [],          # 其他baseline
    }

    for name, data in results.items():
        seed = extract_seed(name)

        if "DebertaV3MTL_S2" in name:
            # 区分消融 vs 正常 vs 敏感性
            if any(tag in name for tag in ["NoPooling", "NoFocal", "OnlyTox", "NoAug", "NoReweight"]):
                groups["ablation"].append((name, data, seed))
            elif any(tag in name for tag in ["W30", "W35"]):
                groups["sensitivity"].append((name, data, seed))
            elif "AblationBCE" in name:
                groups["ablation"].append((name, data, seed))
            else:
                groups["mtl_s2_seeds"].append((name, data, seed))
        elif "VanillaDeBERTa" in name:
            groups["vanilla_deberta_seeds"].append((name, data, seed))
        else:
            groups["baselines"].append((name, data, seed))

    return groups


def get_metric(data, key):
    """安全提取指标值。"""
    if key in data:
        return data[key]
    # 嵌套查找
    for section in ["classification", "bias", "fairness"]:
        if section in data and key in data[section]:
            return data[section][key]
    return None


def mean_std(values):
    """计算均值和标准差。"""
    n = len(values)
    if n == 0:
        return None, None
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(variance)
    return mean, std


def format_mean_std(mean, std, precision=4):
    """格式化为 mean +/- std 字符串。"""
    if mean is None:
        return "N/A"
    if std is None or std == 0:
        return f"{mean:.{precision}f}"
    return f"{mean:.{precision}f} +/- {std:.{precision}f}"


def generate_markdown_report(groups):
    """生成 Markdown 格式的汇总报告。"""
    lines = []
    lines.append("# 补充实验结果汇总\n")
    lines.append(f"生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # --- 1. 多seed统计表 ---
    lines.append("## 1. 多随机种子实验（统计显著性）\n")

    for group_name, group_label in [
        ("mtl_s2_seeds", "所提方法 (DeBERTa-v3 MTL S2)"),
        ("vanilla_deberta_seeds", "基线对照 (Vanilla DeBERTa-v3)")
    ]:
        entries = groups[group_name]
        if not entries:
            continue

        lines.append(f"### {group_label}\n")
        seeds = sorted(set(e[2] for e in entries))
        lines.append(f"Seeds: {seeds} (共{len(entries)}次实验)\n")

        # 表头
        header = "| 指标 |"
        separator = "|------|"
        for seed in seeds:
            header += f" Seed={seed} |"
            separator += "----------|"
        header += " 均值 +/- 标准差 |"
        separator += "----------------|"
        lines.append(header)
        lines.append(separator)

        for metric_key, metric_label in KEY_METRICS:
            row = f"| {metric_label} |"
            values = []
            for seed in seeds:
                entry = next((e for e in entries if e[2] == seed), None)
                if entry:
                    val = get_metric(entry[1], metric_key)
                    if val is not None:
                        values.append(val)
                        row += f" {val:.4f} |"
                    else:
                        row += " N/A |"
                else:
                    row += " - |"
            m, s = mean_std(values)
            row += f" {format_mean_std(m, s)} |"
            lines.append(row)
        lines.append("")

    # --- 2. 细粒度消融 ---
    lines.append("## 2. 细粒度消融实验\n")
    ablation_entries = groups["ablation"]
    if ablation_entries:
        header = "| 消融案例 |"
        separator = "|----------|"
        for _, label in KEY_METRICS:
            header += f" {label} |"
            separator += "-------|"
        lines.append(header)
        lines.append(separator)

        # 先放完整模型作为参照
        full_model = [e for e in groups["mtl_s2_seeds"] if e[2] == 42]
        if full_model:
            row = "| **完整模型 (Full)** |"
            for metric_key, _ in KEY_METRICS:
                val = get_metric(full_model[0][1], metric_key)
                row += f" **{val:.4f}** |" if val else " N/A |"
            lines.append(row)

        for name, data, seed in sorted(ablation_entries, key=lambda x: x[0]):
            # 提取消融标签
            label = name
            for tag in ["NoPooling", "NoFocal", "OnlyTox", "NoAug", "NoReweight", "AblationBCE"]:
                if tag in name:
                    label = tag
                    break
            row = f"| {label} |"
            for metric_key, _ in KEY_METRICS:
                val = get_metric(data, metric_key)
                row += f" {val:.4f} |" if val else " N/A |"
            lines.append(row)
        lines.append("")

    # --- 3. 超参敏感性 ---
    lines.append("## 3. 超参数敏感性分析 (w_identity)\n")
    sensitivity_entries = groups["sensitivity"]
    if sensitivity_entries:
        header = "| w_identity |"
        separator = "|------------|"
        for _, label in KEY_METRICS:
            header += f" {label} |"
            separator += "-------|"
        lines.append(header)
        lines.append(separator)

        # 默认值 2.5 作为参照
        full_model = [e for e in groups["mtl_s2_seeds"] if e[2] == 42]
        if full_model:
            row = "| **2.5 (默认)** |"
            for metric_key, _ in KEY_METRICS:
                val = get_metric(full_model[0][1], metric_key)
                row += f" **{val:.4f}** |" if val else " N/A |"
            lines.append(row)

        for name, data, _ in sorted(sensitivity_entries, key=lambda x: x[0]):
            w_label = name
            if "W30" in name:
                w_label = "3.0"
            elif "W35" in name:
                w_label = "3.5"
            row = f"| {w_label} |"
            for metric_key, _ in KEY_METRICS:
                val = get_metric(data, metric_key)
                row += f" {val:.4f} |" if val else " N/A |"
            lines.append(row)
        lines.append("")

    # --- 4. 全模型对比总表 ---
    lines.append("## 4. 全模型对比总表\n")
    all_entries = []
    for group in groups.values():
        all_entries.extend(group)

    if all_entries:
        header = "| 模型 |"
        separator = "|------|"
        for _, label in KEY_METRICS:
            header += f" {label} |"
            separator += "-------|"
        lines.append(header)
        lines.append(separator)

        for name, data, seed in sorted(all_entries, key=lambda x: x[0]):
            row = f"| {name} |"
            for metric_key, _ in KEY_METRICS:
                val = get_metric(data, metric_key)
                row += f" {val:.4f} |" if val else " N/A |"
            lines.append(row)
        lines.append("")

    return "\n".join(lines)


def generate_latex_table(groups):
    """生成 LaTeX 格式的多seed对比表（可直接粘贴到论文）。"""
    lines = []
    lines.append("% === 多随机种子实验结果 (LaTeX) ===")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{多随机种子实验结果 (均值 $\\pm$ 标准差)}")
    lines.append("\\label{tab:multi_seed}")

    # 选取论文常用的核心指标
    core_metrics = [
        ("roc_auc", "ROC-AUC"),
        ("best_f1", "F1"),
        ("mean_bias_auc", "Mean Bias AUC"),
        ("worst_group_bias_auc", "Worst Group AUC"),
    ]

    cols = "l" + "c" * len(core_metrics)
    lines.append(f"\\begin{{tabular}}{{{cols}}}")
    lines.append("\\toprule")

    header = "Model"
    for _, label in core_metrics:
        header += f" & {label}"
    header += " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    for group_name, group_label in [
        ("mtl_s2_seeds", "MTL S2 (Ours)"),
        ("vanilla_deberta_seeds", "Vanilla DeBERTa-v3"),
    ]:
        entries = groups[group_name]
        if not entries:
            continue

        row = f"{group_label}"
        for metric_key, _ in core_metrics:
            values = [get_metric(e[1], metric_key) for e in entries if get_metric(e[1], metric_key) is not None]
            m, s = mean_std(values)
            if m is not None and s is not None and s > 0:
                row += f" & ${m:.4f} \\pm {s:.4f}$"
            elif m is not None:
                row += f" & {m:.4f}"
            else:
                row += " & N/A"
        row += " \\\\"
        lines.append(row)

    # 加入其他baseline (单次结果)
    for name, data, seed in sorted(groups["baselines"], key=lambda x: x[0]):
        short_name = name.split("_Sample")[0]
        row = f"{short_name}"
        for metric_key, _ in core_metrics:
            val = get_metric(data, metric_key)
            row += f" & {val:.4f}" if val else " & N/A"
        row += " \\\\"
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    # === 消融实验表 ===
    lines.append("% === 消融实验结果 (LaTeX) ===")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{消融实验结果}")
    lines.append("\\label{tab:ablation}")
    lines.append(f"\\begin{{tabular}}{{{cols}}}")
    lines.append("\\toprule")
    lines.append(header)
    lines.append("\\midrule")

    # Full model
    full = [e for e in groups["mtl_s2_seeds"] if e[2] == 42]
    if full:
        row = "Full Model (Ours)"
        for metric_key, _ in core_metrics:
            val = get_metric(full[0][1], metric_key)
            row += f" & \\textbf{{{val:.4f}}}" if val else " & N/A"
        row += " \\\\"
        lines.append(row)

    for name, data, _ in sorted(groups["ablation"], key=lambda x: x[0]):
        label = name
        for tag in ["NoPooling", "NoFocal", "OnlyTox", "NoAug", "NoReweight", "AblationBCE"]:
            if tag in name:
                label = f"w/o {tag}"
                break
        row = f"{label}"
        for metric_key, _ in core_metrics:
            val = get_metric(data, metric_key)
            row += f" & {val:.4f}" if val else " & N/A"
        row += " \\\\"
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("  补充实验结果汇总")
    print("=" * 60)

    results = load_all_metrics()
    if not results:
        print("[Error] 未找到任何评估结果文件")
        sys.exit(1)

    print(f"\n加载了 {len(results)} 个评估结果:")
    for name in sorted(results.keys()):
        print(f"  - {name}")

    groups = classify_results(results)

    # 生成 Markdown 报告
    md_report = generate_markdown_report(groups)
    md_path = os.path.join(OUTPUT_DIR, "SUPPLEMENT_RESULTS.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_report)
    print(f"\n[OK] Markdown 报告已保存: {md_path}")

    # 生成 LaTeX 表格
    latex_tables = generate_latex_table(groups)
    latex_path = os.path.join(OUTPUT_DIR, "latex_tables.tex")
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex_tables)
    print(f"[OK] LaTeX 表格已保存: {latex_path}")

    # 保存原始汇总 JSON
    summary = {}
    for name, data in results.items():
        summary[name] = {}
        for metric_key, _ in KEY_METRICS:
            val = get_metric(data, metric_key)
            if val is not None:
                summary[name][metric_key] = val
    json_path = os.path.join(OUTPUT_DIR, "all_results_summary.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[OK] JSON 汇总已保存: {json_path}")


if __name__ == "__main__":
    main()
