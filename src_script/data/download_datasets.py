"""
=============================================================================
数据集下载脚本: HateXplain + ToxiGen + HateCheck
运行: python src_script/data/download_datasets.py

使用 huggingface_hub 直接下载 parquet/csv 文件 (绕过 datasets 库兼容性问题)
=============================================================================
"""
import os
import json
import pandas as pd
import numpy as np
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "causal_fair")
os.makedirs(DATA_DIR, exist_ok=True)


def download_hatexplain():
    """
    下载并处理 HateXplain 数据集 (~20K)
    使用 HuggingFace Hub 的 refs/convert/parquet 分支直接下载 parquet 文件
    """
    print("=" * 60)
    print("下载 HateXplain 数据集...")
    print("=" * 60)

    from huggingface_hub import hf_hub_download

    TARGET_GROUP_MAP = {
        'African': 'race', 'Arab': 'race', 'Asian': 'race',
        'Caucasian': 'race', 'Hispanic': 'race', 'Indian': 'race',
        'Islam': 'religion', 'Jewish': 'religion',
        'Christian': 'religion', 'Buddhist': 'religion',
        'Hindu': 'religion',
        'Women': 'gender', 'Men': 'gender',
        'Homosexual': 'sexual_orientation',
        'Refugee': 'other', 'Disability': 'disability',
        'Economic': 'other', 'Other': 'other',
    }

    LABEL_MAP = {0: 'hatespeech', 1: 'normal', 2: 'offensive'}

    # 从 HF refs/convert/parquet 分支下载 parquet 文件
    split_files = {
        'train': 'plain_text/train/0000.parquet',
        'val': 'plain_text/validation/0000.parquet',
        'test': 'plain_text/test/0000.parquet',
    }

    for split_name, filename in split_files.items():
        print(f"  下载 {split_name} 集...")
        parquet_path = hf_hub_download(
            repo_id='Hate-speech-CNERG/hatexplain',
            repo_type='dataset',
            filename=filename,
            revision='refs/convert/parquet',
        )
        raw_df = pd.read_parquet(parquet_path)

        # 处理每条数据
        records = []
        for idx, item in raw_df.iterrows():
            text = " ".join(item['post_tokens'])

            # 多数投票确定标签
            # parquet 中 annotators 是 dict: {'label': array, 'annotator_id': array, 'target': array}
            annotator_labels = list(item['annotators']['label'])
            label_counts = Counter(annotator_labels)
            majority_label = label_counts.most_common(1)[0][0]
            label_str = LABEL_MAP[majority_label]

            # 收集所有标注者标注的目标群体
            all_targets = []
            for targets in item['annotators']['target']:
                if targets is not None:
                    all_targets.extend(list(targets))
            unique_targets = list(set(all_targets))

            # 粗粒度群体类别
            coarse_groups = list(set(
                TARGET_GROUP_MAP.get(t, 'other') for t in unique_targets
                if t and TARGET_GROUP_MAP.get(t, 'other') != 'other'
            ))

            # 多数投票 rationale
            rationales = item['rationales']
            if rationales is not None and len(rationales) > 0:
                n_tokens = len(item['post_tokens'])
                agg_rationale = []
                for i in range(n_tokens):
                    votes = sum(1 for r in rationales if r is not None and i < len(r) and r[i] == 1)
                    agg_rationale.append(1 if votes >= 2 else 0)
            else:
                agg_rationale = [0] * len(item['post_tokens'])

            records.append({
                'post_id': item['id'],
                'text': text,
                'label': label_str,
                'label_id': 1 if label_str == 'hatespeech' else (2 if label_str == 'offensive' else 0),
                'binary_label': 1 if label_str in ['hatespeech', 'offensive'] else 0,
                'target_groups': unique_targets,
                'coarse_groups': coarse_groups,
                'has_identity': len(coarse_groups) > 0,
                'rationale': agg_rationale,
                'n_annotators': len(annotator_labels),
                'annotator_agreement': label_counts.most_common(1)[0][1] / len(annotator_labels),
            })

        df = pd.DataFrame(records)
        save_path = os.path.join(DATA_DIR, f"hatexplain_{split_name}.parquet")
        df.to_parquet(save_path, index=False)
        n_hate = (df['binary_label'] == 1).sum()
        n_id = df['has_identity'].sum()
        print(f"  [{split_name}] {len(df):,} 条 | "
              f"有害: {n_hate:,} ({n_hate/len(df)*100:.1f}%) | "
              f"有身份提及: {n_id:,} ({n_id/len(df)*100:.1f}%)")

    print(f"  保存至: {DATA_DIR}/hatexplain_*.parquet\n")


def download_toxigen():
    """
    下载并处理 ToxiGen 数据集 (人工标注子集 ~9K)
    直接通过 huggingface_hub 下载 parquet 文件
    """
    print("=" * 60)
    print("下载 ToxiGen 数据集...")
    print("=" * 60)

    from huggingface_hub import hf_hub_download

    # ToxiGen annotated 子集有 parquet 文件
    parquet_path = hf_hub_download(
        repo_id='toxigen/toxigen-data',
        repo_type='dataset',
        filename='annotated/train-00000-of-00001.parquet',
    )
    raw_df = pd.read_parquet(parquet_path)

    records = []
    for _, item in raw_df.iterrows():
        text = item.get('text', '')
        target_group = item.get('target_group', 'unknown')
        toxicity_human = item.get('toxicity_human', None)

        if not text or toxicity_human is None or pd.isna(toxicity_human):
            continue

        # 人工标注的毒性 (1-5 scale), >= 2.5 视为有毒
        binary_label = 1 if toxicity_human >= 2.5 else 0

        records.append({
            'text': text,
            'target_group': target_group,
            'toxicity_human': toxicity_human,
            'binary_label': binary_label,
            'has_identity': True,  # ToxiGen 所有样本都有目标群体
        })

    df = pd.DataFrame(records)

    # 按 target_group 分层划分
    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['target_group']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['target_group']
    )

    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        save_path = os.path.join(DATA_DIR, f"toxigen_{split_name}.parquet")
        split_df.to_parquet(save_path, index=False)
        n_toxic = (split_df['binary_label'] == 1).sum()
        print(f"  [{split_name}] {len(split_df):,} 条 | "
              f"有害: {n_toxic:,} ({n_toxic/len(split_df)*100:.1f}%) | "
              f"群体数: {split_df['target_group'].nunique()}")

    print(f"  保存至: {DATA_DIR}/toxigen_*.parquet\n")


def download_hatecheck():
    """
    下载 HateCheck 诊断测试集 (~3.7K, 仅测试用)
    直接通过 huggingface_hub 下载 CSV 文件
    """
    print("=" * 60)
    print("下载 HateCheck 诊断测试集...")
    print("=" * 60)

    from huggingface_hub import hf_hub_download

    # HateCheck 有 test.csv
    csv_path = hf_hub_download(
        repo_id='Paul/hatecheck',
        repo_type='dataset',
        filename='test.csv',
    )
    raw_df = pd.read_csv(csv_path)

    # 检查列名并适配
    print(f"  原始列: {raw_df.columns.tolist()}")

    records = []
    for _, item in raw_df.iterrows():
        # HateCheck CSV 列名可能是: test_case, label_gold, target_ident, functionality
        text = item.get('test_case', '')
        label_gold = item.get('label_gold', '')
        target_ident = item.get('target_ident', 'unknown')
        functionality = item.get('functionality', '')

        if not text:
            continue

        records.append({
            'text': text,
            'binary_label': 1 if label_gold == 'hateful' else 0,
            'target_group': target_ident if pd.notna(target_ident) else 'unknown',
            'functionality': functionality if pd.notna(functionality) else '',
        })

    df = pd.DataFrame(records)
    save_path = os.path.join(DATA_DIR, "hatecheck_test.parquet")
    df.to_parquet(save_path, index=False)

    n_hate = (df['binary_label'] == 1).sum()
    print(f"  [test] {len(df):,} 条 | "
          f"有害: {n_hate:,} ({n_hate/len(df)*100:.1f}%) | "
          f"群体数: {df['target_group'].nunique()}")
    print(f"  保存至: {save_path}\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  数据集下载与预处理")
    print("=" * 60 + "\n")

    download_hatexplain()
    download_toxigen()
    download_hatecheck()

    print("=" * 60)
    print("  全部完成!")
    print("=" * 60)
