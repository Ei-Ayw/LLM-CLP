import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import argparse

# =============================================================================
# 数据预处理脚本
# 功能: 读取原始CSV → 采样 → 标签处理 → 分层划分 → 保存parquet
# 已移除所有离线数据增强 (SR/RI/RD/BT), 原因:
#   1. 同义词替换/随机删词会破坏毒性文本的关键语义, 引入标签噪声
#   2. 回译质量差且耗时, 性价比极低
#   3. 增强仅针对toxic样本, 会打破1:1平衡采样
#   4. DeBERTa 预训练已具备足够的泛化能力, 无需离线增强
# =============================================================================

def preprocess_data(input_path, output_dir, sample_size=None, seed=42, keep_all_labeled=False):
    print(f"Loading data from {input_path}...")

    identity_cols = [
        'male', 'female', 'black', 'white', 'muslim', 'jewish', 'christian',
        'homosexual_gay_or_lesbian', 'psychiatric_or_mental_illness'
    ]
    subtype_cols = [
        'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack', 'sexual_explicit'
    ]

    target_col = 'target'
    text_col = 'comment_text'
    use_cols = [text_col, target_col] + subtype_cols + identity_cols

    df = pd.read_csv(input_path, usecols=use_cols)

    # 二值标签 (用于分层采样和评估)
    df['y_tox'] = (df[target_col] >= 0.5).astype(int)
    # 保留原始连续值用于 soft label 训练 (值域 [0, 1])
    df['y_tox_soft'] = df[target_col].clip(0, 1)

    if keep_all_labeled:
        # =====================================================================
        # 新策略：保留所有有身份/子类别标签的样本 + 填充至 1:1
        # 目的：最大化身份标签覆盖率，提升 Worst Group AUC
        # =====================================================================
        df[identity_cols] = df[identity_cols].fillna(0)
        df[subtype_cols] = df[subtype_cols].fillna(0)

        has_label = (df[identity_cols].max(axis=1) >= 0.5) | (df[subtype_cols].max(axis=1) >= 0.5)
        labeled_df = df[has_label].copy()
        unlabeled_df = df[~has_label].copy()

        n_labeled_toxic = (labeled_df['y_tox'] == 1).sum()
        n_labeled_normal = (labeled_df['y_tox'] == 0).sum()
        print(f">>> 标签保留策略: 有标签样本 {len(labeled_df)} 条 (有毒: {n_labeled_toxic}, 正常: {n_labeled_normal})")

        # 从无标签样本中补充，使有毒:无毒 = 1:1
        if n_labeled_toxic > n_labeled_normal:
            n_fill = n_labeled_toxic - n_labeled_normal
            pool = unlabeled_df[unlabeled_df['y_tox'] == 0]
            fill_df = pool.sample(n=min(n_fill, len(pool)), random_state=seed)
        else:
            n_fill = n_labeled_normal - n_labeled_toxic
            pool = unlabeled_df[unlabeled_df['y_tox'] == 1]
            fill_df = pool.sample(n=min(n_fill, len(pool)), random_state=seed)

        df = pd.concat([labeled_df, fill_df], ignore_index=True)
        n_final_toxic = (df['y_tox'] == 1).sum()
        n_final_normal = (df['y_tox'] == 0).sum()
        print(f"  填充后: 总计 {len(df)} 条 (有毒: {n_final_toxic}, 正常: {n_final_normal})")
        print(f"  比例: {n_final_toxic/len(df)*100:.1f}% : {n_final_normal/len(df)*100:.1f}%")
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    elif sample_size and 0 < sample_size < len(df):
        print(f">>> 类别平衡采样: 目标 50:50 比例，总数 {sample_size}...")

        toxic_df = df[df['y_tox'] == 1]
        normal_df = df[df['y_tox'] == 0]

        n_toxic = len(toxic_df)
        n_normal = len(normal_df)
        print(f"  原始数据: 有毒 {n_toxic} 条, 正常 {n_normal} 条")

        n_per_class = sample_size // 2

        if n_toxic >= n_per_class:
            sampled_toxic = toxic_df.sample(n=n_per_class, random_state=seed)
        else:
            sampled_toxic = toxic_df
            n_per_class = n_toxic

        if n_normal >= n_per_class:
            sampled_normal = normal_df.sample(n=n_per_class, random_state=seed)
        else:
            sampled_normal = normal_df

        df = pd.concat([sampled_toxic, sampled_normal], ignore_index=True)
        print(f"  平衡采样后: 有毒 {len(sampled_toxic)} 条, 正常 {len(sampled_normal)} 条, 总计 {len(df)} 条")
        print(f"  比例: {len(sampled_toxic)/len(df)*100:.1f}% : {len(sampled_normal)/len(df)*100:.1f}%")

        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    print("Processing labels...")
    df[identity_cols] = df[identity_cols].fillna(0)
    df[subtype_cols] = df[subtype_cols].fillna(0)
    df['has_identity'] = (df[identity_cols].max(axis=1) >= 0.5).astype(int)

    # Split 80/10/10
    train_df, temp_df = train_test_split(df, test_size=0.20, random_state=seed, stratify=df['y_tox'])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=seed, stratify=temp_df['y_tox'])

    # === 打印最终数据统计 ===
    print("\n" + "="*60)
    print(">>> 数据预处理完成 - 最终统计")
    print("="*60)

    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        n_total = len(split_df)
        n_toxic = (split_df['y_tox'] == 1).sum()
        n_normal = (split_df['y_tox'] == 0).sum()
        ratio = n_normal / n_toxic if n_toxic > 0 else float('inf')
        pct_toxic = n_toxic / n_total * 100
        print(f"  {split_name:6s}: {n_total:>8,} 条 | 有毒: {n_toxic:>7,} ({pct_toxic:5.1f}%) | 正常: {n_normal:>7,} | 比例 1:{ratio:.2f}")

    total_all = len(train_df) + len(val_df) + len(test_df)
    total_toxic = (train_df['y_tox'] == 1).sum() + (val_df['y_tox'] == 1).sum() + (test_df['y_tox'] == 1).sum()
    total_normal = total_all - total_toxic
    print("-"*60)
    print(f"  {'Total':6s}: {total_all:>8,} 条 | 有毒: {total_toxic:>7,} ({total_toxic/total_all*100:5.1f}%) | 正常: {total_normal:>7,}")
    print("="*60 + "\n")

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_parquet(os.path.join(output_dir, 'train_processed.parquet'), index=False)
    val_df.to_parquet(os.path.join(output_dir, 'val_processed.parquet'), index=False)
    test_df.to_parquet(os.path.join(output_dir, 'test_processed.parquet'), index=False)
    print(f"Preprocessing completed. Files saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep_all_labeled", action="store_true",
                        help="保留所有有身份/子类别标签的样本，从无标签样本中填充至1:1")
    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    INPUT_CSV = os.path.join(DATA_DIR, "train.csv")
    OUTPUT_DIR = DATA_DIR

    # 自动解压逻辑
    if not os.path.exists(INPUT_CSV):
        import zipfile
        zip_path = os.path.join(DATA_DIR, "jigsaw-unintended-bias-in-toxicity-classification.zip")
        if os.path.exists(zip_path):
            print(f"Detected dataset zip file. Unzipping {zip_path}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(DATA_DIR)
                print("Unzip successful.")
            except Exception as e:
                print(f"Failed to unzip: {e}")
                exit(1)
        else:
            print(f"[Error] Data not found! Please upload 'train.csv' or the dataset zip file to: {DATA_DIR}")
            print(f"Expected path: {INPUT_CSV}")
            exit(1)

    preprocess_data(INPUT_CSV, OUTPUT_DIR, sample_size=args.sample_size, seed=args.seed,
                    keep_all_labeled=args.keep_all_labeled)
