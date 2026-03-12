"""
=============================================================================
反事实质量验证
过滤低质量的 LLM 生成反事实
=============================================================================
"""
import pandas as pd
import numpy as np
from tqdm import tqdm


def validate_counterfactual(original, cf_text, source_group, target_group):
    """
    验证单条反事实质量

    返回: dict with quality scores and pass/fail
    """
    checks = {}

    # 1. 非空且不等于原文
    checks['not_empty'] = bool(cf_text and len(cf_text.strip()) > 5)
    checks['not_same'] = (cf_text.strip() != original.strip())

    if not checks['not_empty'] or not checks['not_same']:
        checks['valid'] = False
        return checks

    # 2. 长度比 (应接近 1.0)
    len_ratio = len(cf_text.split()) / max(len(original.split()), 1)
    checks['length_ratio'] = len_ratio
    checks['length_ok'] = 0.5 < len_ratio < 2.0

    # 3. 源群体关键词应被移除 (大部分情况)
    cf_lower = cf_text.lower()
    orig_lower = original.lower()
    checks['source_reduced'] = cf_lower.count(source_group) <= orig_lower.count(source_group)

    # 4. 不应该添加太多新内容
    orig_words = set(orig_lower.split())
    cf_words = set(cf_lower.split())
    new_words = cf_words - orig_words
    removed_words = orig_words - cf_words
    # 新增词数不应超过原文的 50%
    checks['not_too_different'] = len(new_words) < len(orig_words) * 0.5

    # 5. 综合判定
    checks['valid'] = all([
        checks['not_empty'],
        checks['not_same'],
        checks['length_ok'],
        checks['not_too_different'],
    ])

    return checks


def validate_and_filter(cf_df, verbose=True):
    """
    批量验证并过滤反事实数据

    Args:
        cf_df: 反事实 DataFrame, 需要 original_text, cf_text, source_group, target_group 列

    Returns:
        过滤后的 DataFrame
    """
    valid_records = []
    stats = {'total': 0, 'valid': 0, 'invalid': 0, 'reasons': {}}

    for _, row in tqdm(cf_df.iterrows(), total=len(cf_df), desc="验证反事实"):
        stats['total'] += 1
        checks = validate_counterfactual(
            row['original_text'], row['cf_text'],
            row.get('source_group', ''), row.get('target_group', '')
        )

        if checks['valid']:
            stats['valid'] += 1
            valid_records.append(row.to_dict())
        else:
            stats['invalid'] += 1
            for key, val in checks.items():
                if key != 'valid' and val == False:
                    stats['reasons'][key] = stats['reasons'].get(key, 0) + 1

    if verbose:
        print(f"\n验证结果:")
        print(f"  总数: {stats['total']}")
        print(f"  通过: {stats['valid']} ({stats['valid']/max(stats['total'],1)*100:.1f}%)")
        print(f"  过滤: {stats['invalid']}")
        if stats['reasons']:
            print(f"  过滤原因:")
            for reason, count in sorted(stats['reasons'].items(), key=lambda x: -x[1]):
                print(f"    {reason}: {count}")

    return pd.DataFrame(valid_records)


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="反事实文件路径")
    parser.add_argument("--output", type=str, default=None, help="过滤后保存路径")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    print(f"加载 {len(df)} 条反事实")

    filtered = validate_and_filter(df)

    output = args.output or args.input.replace('.parquet', '_filtered.parquet')
    filtered.to_parquet(output, index=False)
    print(f"保存至: {output}")
