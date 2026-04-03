"""
=============================================================================
build_unified_table.py
把所有实验结果（DeBERTa 3seeds + BERT/RoBERTa seed=42）汇总成统一表格
格式: 有多个seed → mean±std，只有1个seed → 单值

输出:
  src_result/eval/unified_results_table.txt   (论文风格文本表格)
  src_result/eval/unified_results_table.json  (机器可读)

用法:
  python build_unified_table.py [--cf_method llm]
=============================================================================
"""
import os, sys, json, glob, argparse
import numpy as np
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_DIR = os.path.join(BASE_DIR, "src_result", "eval")

METHODS   = ["Vanilla", "EAR", "GetFair", "CCDF", "Davani", "Ramponi", "Ours"]
BACKBONES = ["bert", "roberta", "deberta"]
DATASETS  = ["hatexplain", "toxigen", "dynahate"]
CF_METHODS = ["swap", "llm"]

# 7 core metrics
METRICS = [
    ("macro_f1",         "Macro-F1",  True),
    ("auc_roc",          "AUC-ROC",   True),
    ("cfr",              "CFR",       False),
    ("ctfg",             "CTFG",      False),
    ("fped",             "FPED",      False),
    ("fned",             "FNED",      False),
    ("per_group_f1_std", "F1-Std",    False),
]


def fmt(mean, std=None, higher_is_better=True, best=None):
    """Format mean±std or single value. Mark best with *"""
    if mean is None:
        return "-"
    s = f"{mean:.4f}"
    if std is not None:
        s = f"{mean:.4f}±{std:.4f}"
    if best is not None:
        try:
            if higher_is_better and abs(float(mean) - best) < 1e-6:
                s += "*"
            elif not higher_is_better and abs(float(mean) - best) < 1e-6:
                s += "*"
        except Exception:
            pass
    return s


def load_deberta_summary(eval_dir, cf_method):
    """
    从 summary_all_7metrics.json 读取 DeBERTa 汇总数据（含 Ours，3 seeds mean±std）
    返回 dict: (method, 'deberta', dataset) → 单条 pre-aggregated dict
    """
    summary_path = os.path.join(eval_dir, "summary_all_7metrics.json")
    if not os.path.exists(summary_path):
        return {}
    with open(summary_path) as f:
        summary = json.load(f)

    # Also need per_group_f1_std — get from individual JSONs
    # Pattern for Ours DeBERTa: {dataset}_{cf_method}_clp1.0_con*_seed*_*_7metrics_{cf_method}.json
    ours_pattern = os.path.join(eval_dir, f"*_clp1.0_con*_seed*_*_7metrics_{cf_method}.json")
    ours_files = glob.glob(ours_pattern)
    ours_by_dataset = defaultdict(list)
    for fp in ours_files:
        fname = os.path.basename(fp)
        # skip bert/roberta
        if "_bert_" in fname or "_roberta_" in fname:
            continue
        for ds in DATASETS:
            if fname.startswith(ds + "_"):
                try:
                    d = json.load(open(fp))
                    ours_by_dataset[ds].append(d.get("per_group_f1_std"))
                except Exception:
                    pass
                break

    result = {}
    for method in METHODS:
        if method not in summary:
            continue
        for dataset in DATASETS:
            if dataset not in summary[method]:
                continue
            entry = summary[method][dataset].get(cf_method, {})
            if not entry:
                continue
            mdict = {}
            for key, _, _ in METRICS:
                sub = entry.get(key, {})
                if isinstance(sub, dict):
                    mdict[key] = (sub.get("mean"), sub.get("std"))
                else:
                    mdict[key] = (sub, None)
            # Fill per_group_f1_std for Ours from individual files
            if method == "Ours" and mdict.get("per_group_f1_std", (None, None))[0] is None:
                vals = [v for v in ours_by_dataset.get(dataset, []) if v is not None]
                if vals:
                    mdict["per_group_f1_std"] = (
                        round(float(np.mean(vals)), 4),
                        round(float(np.std(vals)), 4) if len(vals) > 1 else None
                    )
            mdict["n_seeds"] = entry.get("n_seeds", 3)
            result[(method, "deberta", dataset)] = mdict
    return result


def load_all_jsons(eval_dir, cf_method):
    """
    从所有 *_7metrics_{cf_method}.json 文件中读取 BERT/RoBERTa 数据
    返回 dict: (method, backbone, dataset) → list of metric dicts
    """
    pattern = os.path.join(eval_dir, f"*_7metrics_{cf_method}.json")
    files = sorted(glob.glob(pattern))

    bucket = defaultdict(list)

    for fp in files:
        fname = os.path.basename(fp)
        try:
            data = json.load(open(fp))
        except Exception:
            continue

        # --- Detect backbone (only BERT and RoBERTa from individual files) ---
        if "_roberta_" in fname:
            backbone = "roberta"
        elif "_bert_" in fname:
            backbone = "bert"
        else:
            continue  # DeBERTa handled via summary

        # --- Detect method ---
        method = None
        fname_lower = fname.lower()
        for m in METHODS:
            if fname_lower.startswith(m.lower() + "_"):
                method = m
                break
        if method is None:
            continue

        # Skip sensitivity runs (clp != 1.0) for the backbone comparison table
        if method == "Ours" and "_clp" in fname:
            import re
            m_clp = re.search(r'_clp([\d.]+)_', fname)
            if m_clp and abs(float(m_clp.group(1)) - 1.0) > 1e-6:
                continue  # sensitivity run, skip for main table

        # --- Detect dataset ---
        dataset = None
        for ds in DATASETS:
            if f"_{ds}_" in fname_lower:
                dataset = ds
                break
        if dataset is None:
            continue

        # Extract metrics
        mdict = {k: data.get(k) for k, _, _ in METRICS}
        bucket[(method, backbone, dataset)].append(mdict)

    return bucket


def aggregate(records):
    """Aggregate list of metric dicts → {metric: (mean, std or None)}"""
    result = {}
    for key, _, _ in METRICS:
        vals = [r[key] for r in records if r.get(key) is not None]
        if not vals:
            result[key] = (None, None)
        elif len(vals) == 1:
            result[key] = (round(vals[0], 4), None)
        else:
            result[key] = (round(np.mean(vals), 4), round(np.std(vals), 4))
    result["n_seeds"] = len(records)
    return result


def get_agg(bucket, deberta_summary, method, backbone, dataset):
    """Get aggregated entry for (method, backbone, dataset)."""
    if backbone == "deberta":
        entry = deberta_summary.get((method, "deberta", dataset))
        if entry is None:
            return None
        # entry is already {key: (mean, std), n_seeds: n}
        return entry
    else:
        records = bucket.get((method, backbone, dataset), [])
        if not records:
            return None
        agg = aggregate(records)
        # convert to same format: {key: (mean, std)}
        result = {k: agg[k] for k, _, _ in METRICS}
        result["n_seeds"] = agg["n_seeds"]
        return result


def find_best(agg_by_method, metric_key, higher_is_better):
    vals = []
    for agg in agg_by_method.values():
        if agg is None:
            continue
        m = agg.get(metric_key, (None, None))[0]
        if m is not None:
            vals.append(float(m))
    if not vals:
        return None
    return max(vals) if higher_is_better else min(vals)


def build_table(bucket, deberta_summary, cf_method, dataset):
    """Build per-dataset table across all backbones and methods."""
    lines = []

    # Header
    bb_labels = ["BERT", "RoBERTa", "DeBERTa"]
    metric_names = [name for _, name, _ in METRICS]

    # We'll do one sub-table per backbone for readability
    for backbone, bb_label in zip(BACKBONES, bb_labels):
        lines.append(f"\n  [{bb_label}]")
        col_w = 16
        header = f"  {'Method':10s}"
        for _, name, _ in METRICS:
            header += f"  {name:^{col_w}}"
        lines.append(header)
        lines.append("  " + "-" * (12 + (col_w + 2) * len(METRICS)))

        # Collect aggregated entries for all methods
        agg_by_method = {}
        for method in METHODS:
            agg_by_method[method] = get_agg(bucket, deberta_summary, method, backbone, dataset)

        # Find best per metric
        bests = {}
        for key, _, hib in METRICS:
            bests[key] = find_best(agg_by_method, key, hib)

        for method in METHODS:
            agg = agg_by_method.get(method)
            row = f"  {method:10s}"
            for key, _, hib in METRICS:
                if agg is None:
                    row += f"  {'N/A':^{col_w}}"
                else:
                    mean, std = agg.get(key, (None, None))
                    cell = fmt(mean, std, hib, bests.get(key))
                    row += f"  {cell:^{col_w}}"
            n = agg["n_seeds"] if agg else 0
            row += f"  (n={n})"
            lines.append(row)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cf_method", type=str, default="llm", choices=CF_METHODS)
    parser.add_argument("--eval_dir",  type=str, default=EVAL_DIR)
    args = parser.parse_args()

    print(f"[Load] Scanning {args.eval_dir} for cf_method={args.cf_method} ...")
    bucket = load_all_jsons(args.eval_dir, args.cf_method)
    deberta_summary = load_deberta_summary(args.eval_dir, args.cf_method)
    print(f"[Load] BERT/RoBERTa combos: {len(bucket)}  DeBERTa entries: {len(deberta_summary)}")

    lines = []
    lines.append("=" * 110)
    lines.append(f"  综合实验结果汇总 (CF={args.cf_method.upper()})")
    lines.append(f"  格式: 多seed → mean±std   单seed → 单值   * = 该列最优")
    lines.append(f"  DeBERTa: 3 seeds (42/123/2024)   BERT/RoBERTa: seed=42")
    lines.append("=" * 110)
    lines.append("""
  指标说明:
    Macro-F1↑  AUC-ROC↑  CFR↓(反事实翻转率)  CTFG↓(反事实公平差距)
    FPED↓(假阳性平等差异)  FNED↓(假阴性平等差异)  F1-Std↓(群体间F1标准差)
""")

    for dataset in DATASETS:
        lines.append("\n" + "=" * 110)
        lines.append(f"  数据集: {dataset.upper()}")
        lines.append("=" * 110)
        lines.append(build_table(bucket, deberta_summary, args.cf_method, dataset))

    # Also build swap table
    if args.cf_method == "llm":
        bucket_swap = load_all_jsons(args.eval_dir, "swap")
        deberta_summary_swap = load_deberta_summary(args.eval_dir, "swap")
        lines.append("\n\n" + "=" * 110)
        lines.append("  综合实验结果汇总 (CF=SWAP)")
        lines.append("=" * 110)
        for dataset in DATASETS:
            lines.append("\n" + "=" * 110)
            lines.append(f"  数据集: {dataset.upper()}")
            lines.append("=" * 110)
            lines.append(build_table(bucket_swap, deberta_summary_swap, "swap", dataset))

    report = "\n".join(lines)

    # Save
    txt_path  = os.path.join(args.eval_dir, "unified_results_table.txt")
    json_path = os.path.join(args.eval_dir, "unified_results_table.json")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report)

    # JSON version
    json_out = {}
    for cf in CF_METHODS:
        bkt = bucket if cf == args.cf_method else load_all_jsons(args.eval_dir, cf)
        db_sum = deberta_summary if cf == args.cf_method else load_deberta_summary(args.eval_dir, cf)
        json_out[cf] = {}
        for method in METHODS:
            json_out[cf][method] = {}
            for backbone in BACKBONES:
                json_out[cf][method][backbone] = {}
                for dataset in DATASETS:
                    agg = get_agg(bkt, db_sum, method, backbone, dataset)
                    if agg:
                        json_out[cf][method][backbone][dataset] = {
                            k: {"mean": agg[k][0], "std": agg[k][1]}
                            for k, _, _ in METRICS
                            if agg.get(k, (None, None))[0] is not None
                        }
                        json_out[cf][method][backbone][dataset]["n_seeds"] = agg["n_seeds"]

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2, default=str)

    print(report)
    print(f"\n保存至:\n  {txt_path}\n  {json_path}")


if __name__ == "__main__":
    main()
