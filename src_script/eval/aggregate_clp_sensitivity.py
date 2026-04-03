"""
=============================================================================
聚合 CausalFair CLP 灵敏度分析结果
λ_clp ∈ {0.2, 0.4, 0.6, 0.8, 1.0}, λ_con=0.5, cf_method=llm
所有 3 个数据集 × 3 个 seed × 3 个 backbone

输出格式:
  {
    "bert": {
      "hatexplain": {
        "llm": {
          "0.2": { macro_f1: {mean, std}, ..., n_seeds: 3 },
          "0.4": { ... },
          ...
        }
      }
    },
    "roberta": { ... },
    "deberta": { ... }
  }

文件命名约定 (eval_backbone_baselines.py 输出 with method=ours):
  Ours_{backbone}_{dataset}_{cf_method}_clp{lambda_clp}_con{lambda_con}_seed{seed}_{ts}_7metrics_{cf_method}.json

用法:
  python aggregate_clp_sensitivity.py \
      --eval_dir  ../../src_result/eval \
      --output    ../../src_result/eval/summary_clp_sensitivity.json
=============================================================================
"""
import os, sys, json, glob, re, argparse
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BACKBONES  = ["bert", "roberta", "deberta"]
DATASETS   = ["hatexplain", "toxigen", "dynahate"]
CF_METHODS = ["swap", "llm"]
LAMBDA_CLPS = ["0.2", "0.4", "0.6", "0.8", "1.0"]
SEEDS      = [42, 123, 2024]

METRICS_7  = ["macro_f1", "auc_roc", "cfr", "ctfg", "fped", "fned", "per_group_f1_std"]
METRICS_ALL = ["accuracy", "binary_f1"] + METRICS_7


def parse_clp_meta(filepath):
    """
    从 JSON _meta 或文件名解析 backbone/dataset/cf_method/lambda_clp.
    返回 (meta_dict, data_dict) 或 (None, None)
    """
    try:
        with open(filepath) as f:
            data = json.load(f)
    except Exception as e:
        print(f"  [WARN] Cannot read {os.path.basename(filepath)}: {e}")
        return None, None

    meta = data.get('_meta', {})
    backbone  = meta.get('backbone', None)
    dataset   = meta.get('dataset', None)
    cf_method = meta.get('cf_method', None)
    # lambda_clp is NOT stored in _meta by default; parse from checkpoint filename or exp name
    checkpoint = meta.get('checkpoint', '') or ''
    lambda_clp = None

    # Try to extract from checkpoint filename: clp{val}
    m_clp = re.search(r'_clp([\d.]+)_', os.path.basename(checkpoint))
    if m_clp:
        lambda_clp = m_clp.group(1)

    # Fallback: from eval result file name
    fname = os.path.basename(filepath)
    if lambda_clp is None:
        m_clp2 = re.search(r'_clp([\d.]+)_', fname)
        if m_clp2:
            lambda_clp = m_clp2.group(1)

    # Ensure this is an "Ours" / CausalFair result
    method = meta.get('method', '')
    if method.lower() not in ('ours', 'causalfair'):
        # Check filename
        if not (fname.lower().startswith('ours_') or 'causalfair' in fname.lower()):
            return None, None

    # If still missing backbone/dataset/cf_method, parse from filename
    if not backbone:
        for bb in BACKBONES:
            if f'_{bb}_' in fname.lower():
                backbone = bb
                break
    if not dataset:
        for ds in DATASETS:
            if f'_{ds}_' in fname.lower():
                dataset = ds
                break
    if not cf_method:
        m_cf = re.search(r'_7metrics_(swap|llm)\.json$', fname)
        if m_cf:
            cf_method = m_cf.group(1)

    if not all([backbone, dataset, cf_method, lambda_clp]):
        return None, None

    # Normalize lambda_clp to standard keys
    try:
        clp_float = float(lambda_clp)
        # Round to avoid floating point mismatch
        lambda_clp = f"{clp_float:.1f}"
    except ValueError:
        pass

    return {
        'backbone':    backbone,
        'dataset':     dataset,
        'cf_method':   cf_method,
        'lambda_clp':  lambda_clp,
    }, data


def extract_metrics(data):
    out = {}
    for m in METRICS_ALL:
        out[m] = data.get(m, None)
    return out


def aggregate(records):
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
    parser = argparse.ArgumentParser(description="聚合 CLP 灵敏度分析结果 (7指标)")
    parser.add_argument("--eval_dir", type=str,
                        default=os.path.join(BASE_DIR, "src_result", "eval"))
    parser.add_argument("--output",   type=str,
                        default=os.path.join(BASE_DIR, "src_result", "eval",
                                             "summary_clp_sensitivity.json"))
    parser.add_argument("--pattern",  type=str, default="Ours_*_7metrics_*.json")
    parser.add_argument("--backbone", type=str, nargs='+', default=BACKBONES)
    parser.add_argument("--cf_method", type=str, nargs='+', default=CF_METHODS)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.eval_dir, args.pattern)))
    print(f"[Scan] {len(files)} files matching '{args.pattern}'")

    # Bucket: backbone → dataset → cf_method → lambda_clp → [metric_dicts]
    bucket = {}

    for fp in files:
        meta, data = parse_clp_meta(fp)
        if meta is None:
            continue
        if meta['backbone'] not in args.backbone:
            continue
        if meta['cf_method'] not in args.cf_method:
            continue
        if meta['lambda_clp'] not in LAMBDA_CLPS:
            # Accept if value is close to one of the defined values
            try:
                v = float(meta['lambda_clp'])
                matched = False
                for lc in LAMBDA_CLPS:
                    if abs(v - float(lc)) < 1e-6:
                        meta['lambda_clp'] = lc
                        matched = True
                        break
                if not matched:
                    print(f"  [SKIP] Unexpected lambda_clp={meta['lambda_clp']}: {os.path.basename(fp)}")
                    continue
            except ValueError:
                continue

        mvals = extract_metrics(data)
        backbone  = meta['backbone']
        dataset   = meta['dataset']
        cf_method = meta['cf_method']
        lc        = meta['lambda_clp']

        bucket.setdefault(backbone, {}) \
              .setdefault(dataset, {}) \
              .setdefault(cf_method, {}) \
              .setdefault(lc, []) \
              .append(mvals)

    # Build summary
    summary = {}
    for backbone in args.backbone:
        if backbone not in bucket:
            continue
        summary[backbone] = {}
        for dataset in DATASETS:
            if dataset not in bucket[backbone]:
                continue
            summary[backbone][dataset] = {}
            for cf_method in args.cf_method:
                if cf_method not in bucket[backbone][dataset]:
                    continue
                summary[backbone][dataset][cf_method] = {}
                for lc in LAMBDA_CLPS:
                    records = bucket[backbone][dataset][cf_method].get(lc, [])
                    if not records:
                        continue
                    agg = aggregate(records)
                    agg['n_seeds'] = len(records)
                    summary[backbone][dataset][cf_method][lc] = agg

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Print report
    print(f"\n{'='*80}")
    print(f"  CLP 灵敏度分析聚合 (λ_clp ∈ {LAMBDA_CLPS})")
    print(f"{'='*80}")
    for backbone in args.backbone:
        if backbone not in summary:
            print(f"  {backbone}: [未找到结果]")
            continue
        for dataset in DATASETS:
            if dataset not in summary[backbone]:
                continue
            for cf_method in args.cf_method:
                if cf_method not in summary[backbone][dataset]:
                    continue
                print(f"\n  {backbone} | {dataset} | {cf_method}")
                print(f"  {'λ_clp':6s}  {'Macro-F1':>10s}  {'AUC-ROC':>10s}  {'CFR':>8s}  "
                      f"{'CTFG':>8s}  {'FPED':>8s}  {'FNED':>8s}  {'F1-Std':>8s}  n")
                print(f"  {'-'*85}")
                for lc in LAMBDA_CLPS:
                    if lc not in summary[backbone][dataset][cf_method]:
                        print(f"  {lc:6s}  [missing]")
                        continue
                    e = summary[backbone][dataset][cf_method][lc]
                    def fmt(key):
                        v = e.get(key, {}).get('mean')
                        return f"{v:.4f}" if v is not None else "  N/A "
                    print(f"  {lc:6s}  {fmt('macro_f1'):>10s}  {fmt('auc_roc'):>10s}  "
                          f"{fmt('cfr'):>8s}  {fmt('ctfg'):>8s}  {fmt('fped'):>8s}  "
                          f"{fmt('fned'):>8s}  {fmt('per_group_f1_std'):>8s}  "
                          f"{e.get('n_seeds', 0)}")

    print(f"\n保存至: {args.output}")


if __name__ == "__main__":
    main()
