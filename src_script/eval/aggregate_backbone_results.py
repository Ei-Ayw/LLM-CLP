"""
=============================================================================
聚合 BERT / RoBERTa / DeBERTa 骨干对比实验结果
输出格式与 summary_all_7metrics.json 一致:
  {method: {backbone: {dataset: {cf_method: {metric: {mean, std}, n_seeds}}}}}

7 个指标: macro_f1, auc_roc, cfr, ctfg, fped, fned, per_group_f1_std
额外保留: accuracy, binary_f1

用法:
  python aggregate_backbone_results.py \
      --eval_dir  ../../src_result/eval \
      --output    ../../src_result/eval/summary_backbone_7metrics.json \
      [--backbone bert roberta deberta]

文件命名约定 (eval_backbone_baselines.py 生成):
  {ExpName}_7metrics_{cf_method}.json
  ExpName 格式: {Method}_{backbone}_{dataset}_{cf_method?}_..._seed{seed}_{timestamp}
  例: Ours_bert_hatexplain_llm_clp1.0_con0.5_seed42_0402_1000
      Vanilla_bert_hatexplain_seed42_0402_1000
      CCDF_roberta_toxigen_seed123_0402_1100
=============================================================================
"""
import os, sys, json, glob, re, argparse
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


METHODS   = ["Vanilla", "EAR", "GetFair", "CCDF", "Davani", "Ramponi", "Ours"]
BACKBONES = ["bert", "roberta", "deberta"]
DATASETS  = ["hatexplain", "toxigen", "dynahate"]
CF_METHODS = ["swap", "llm"]
SEEDS     = [42, 123, 2024]

METRICS_7 = ["macro_f1", "auc_roc", "cfr", "ctfg", "fped", "fned", "per_group_f1_std"]
METRICS_ALL = ["accuracy", "binary_f1"] + METRICS_7


def parse_meta(filepath):
    """
    从 JSON 文件的 _meta 字段读取元信息.
    回退: 从文件名解析.
    返回 dict 或 None.
    """
    try:
        with open(filepath) as f:
            data = json.load(f)
    except Exception as e:
        print(f"  [WARN] Cannot read {os.path.basename(filepath)}: {e}")
        return None, None

    meta = data.get('_meta', {})
    method   = meta.get('method', None)
    backbone = meta.get('backbone', None)
    dataset  = meta.get('dataset', None)
    cf_method = meta.get('cf_method', None)

    # 若 _meta 不完整, 从文件名解析
    fname = os.path.basename(filepath)
    # 尝试 pattern: Method_backbone_dataset_..._seed\d+_\d+_7metrics_cfmethod.json
    # 或: Method_backbone_dataset_seed\d+_\d+_7metrics_cfmethod.json
    if not all([method, backbone, dataset, cf_method]):
        # Extract cf_method from filename suffix
        m_cf = re.search(r'_7metrics_(swap|llm)\.json$', fname)
        if m_cf and not cf_method:
            cf_method = m_cf.group(1)

        # Find backbone
        for bb in BACKBONES:
            if f'_{bb}_' in fname.lower() and not backbone:
                backbone = bb
                break

        # Find method (case-insensitive prefix)
        fname_lower = fname.lower()
        for mt in [m.lower() for m in METHODS]:
            if fname_lower.startswith(mt + '_'):
                if not method:
                    method = mt.capitalize() if mt != 'ear' else 'EAR'
                    # Normalize
                    for orig in METHODS:
                        if orig.lower() == mt:
                            method = orig
                            break
                break
        # Also handle "ours" which is method="ours" in eval output
        if not method:
            if fname_lower.startswith('ours_'):
                method = 'ours'

        # Find dataset
        for ds in DATASETS:
            if f'_{ds}_' in fname_lower and not dataset:
                dataset = ds
                break

    if not all([method, backbone, dataset, cf_method]):
        return None, None

    # Normalize method capitalization to match METHODS list
    method_norm = None
    for m in METHODS:
        if m.lower() == method.lower():
            method_norm = m
            break
    if method_norm is None:
        # ours → Ours
        if method.lower() == 'ours':
            method_norm = 'Ours'
        else:
            method_norm = method

    return {
        'method':    method_norm,
        'backbone':  backbone,
        'dataset':   dataset,
        'cf_method': cf_method,
    }, data


def extract_metrics(data):
    """从评估 JSON 提取所有指标值 (含 None)."""
    out = {}
    for m in METRICS_ALL:
        out[m] = data.get(m, None)
    return out


def aggregate(records):
    """
    records: list of metric dicts (possibly with None values)
    Returns: {metric: {mean, std}} for metrics that have >=1 valid value
    """
    result = {}
    for m in METRICS_ALL:
        vals = [r[m] for r in records if r[m] is not None]
        if vals:
            result[m] = {
                'mean': round(float(np.mean(vals)), 4),
                'std':  round(float(np.std(vals)),  4),
            }
        else:
            result[m] = {'mean': None, 'std': None}
    return result


def main():
    parser = argparse.ArgumentParser(description="聚合 backbone 对比实验结果 (7指标)")
    parser.add_argument("--eval_dir", type=str,
                        default=os.path.join(BASE_DIR, "src_result", "eval"),
                        help="评估结果 JSON 目录")
    parser.add_argument("--output",   type=str,
                        default=os.path.join(BASE_DIR, "src_result", "eval",
                                             "summary_backbone_7metrics.json"),
                        help="输出聚合 JSON 路径")
    parser.add_argument("--backbone", type=str, nargs='+',
                        default=BACKBONES,
                        choices=BACKBONES,
                        help="要聚合的 backbone 列表")
    parser.add_argument("--pattern",  type=str, default="*_7metrics_*.json",
                        help="文件名 glob 匹配模式")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.eval_dir, args.pattern)))
    print(f"[Scan] {len(files)} files matching '{args.pattern}' in {args.eval_dir}")

    # Bucket: method → backbone → dataset → cf_method → [metric_dicts]
    bucket = {}

    for fp in files:
        meta, data = parse_meta(fp)
        if meta is None:
            print(f"  [SKIP] Cannot parse meta: {os.path.basename(fp)}")
            continue

        method   = meta['method']
        backbone = meta['backbone']
        dataset  = meta['dataset']
        cf_method = meta['cf_method']

        if backbone not in args.backbone:
            continue

        mvals = extract_metrics(data)

        bucket.setdefault(method, {}) \
              .setdefault(backbone, {}) \
              .setdefault(dataset, {}) \
              .setdefault(cf_method, []) \
              .append(mvals)

    # Build summary
    summary = {}
    for method in METHODS:
        if method not in bucket:
            continue
        summary[method] = {}
        for backbone in args.backbone:
            if backbone not in bucket.get(method, {}):
                continue
            summary[method][backbone] = {}
            for dataset in DATASETS:
                if dataset not in bucket[method][backbone]:
                    continue
                summary[method][backbone][dataset] = {}
                for cf_method in CF_METHODS:
                    records = bucket[method][backbone].get(dataset, {}).get(cf_method, [])
                    if not records:
                        continue
                    agg = aggregate(records)
                    agg['n_seeds'] = len(records)
                    summary[method][backbone][dataset][cf_method] = agg

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Print coverage report
    print(f"\n{'='*70}")
    print(f"  Backbone 实验结果聚合报告")
    print(f"{'='*70}")
    for method in METHODS:
        if method not in summary:
            print(f"  {method}: [未找到结果]")
            continue
        for backbone in args.backbone:
            if backbone not in summary.get(method, {}):
                continue
            for dataset in DATASETS:
                if dataset not in summary[method][backbone]:
                    continue
                for cf_method in CF_METHODS:
                    if cf_method not in summary[method][backbone][dataset]:
                        continue
                    entry = summary[method][backbone][dataset][cf_method]
                    n = entry.get('n_seeds', 0)
                    f1 = entry.get('macro_f1', {})
                    auc = entry.get('auc_roc', {})
                    cfr = entry.get('cfr', {})
                    std = entry.get('per_group_f1_std', {})
                    print(f"  {method:10s} {backbone:8s} {dataset:12s} {cf_method:4s} | "
                          f"n={n} | F1={f1.get('mean','?'):.4f}±{f1.get('std','?'):.4f} | "
                          f"AUC={auc.get('mean','?'):.4f} | "
                          f"CFR={cfr.get('mean','?') if cfr.get('mean') is not None else '?'} | "
                          f"F1std={std.get('mean','?') if std.get('mean') is not None else '?'}")

    print(f"\n保存至: {args.output}")


if __name__ == "__main__":
    main()
