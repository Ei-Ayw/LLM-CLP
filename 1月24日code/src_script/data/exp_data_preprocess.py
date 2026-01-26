import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def preprocess_data(input_path, output_dir):
    print(f"Loading data from {input_path}...")
    # 由于数据量大，仅读取需要的列
    # =========================================================================
    # 3.3 辅助标签 2: identity（多标签）- 用于偏见评估与 Stage 2 身份感知重加权
    # 选取 9 个经典且覆盖面大的身份属性列：
    #   - 性别: male, female
    #   - 种族: black, white
    #   - 宗教: muslim, jewish, christian
    #   - 性取向: homosexual_gay_or_lesbian
    #   - 心理健康: psychiatric_or_mental_illness
    # 处理方式: NaN 填 0, 保留原始 0~1 软标签 (更稳健)
    # =========================================================================
    identity_cols = [
        'male', 'female', 'black', 'white', 'muslim', 'jewish', 'christian', 
        'homosexual_gay_or_lesbian', 'psychiatric_or_mental_illness'
    ]
    # =========================================================================
    # 3.2 辅助标签 1: subtypes（多标签）- 用于多任务学习 (MTL)
    # 选取 6 个够用且常见的毒性子类别：
    #   - severe_toxicity (严重毒性)
    #   - obscene (淫秽)
    #   - threat (威胁)
    #   - insult (侮辱)
    #   - identity_attack (身份攻击)
    #   - sexual_explicit (性暗示)
    # 处理方式: NaN 填 0, 直接用原始 0~1 作为软标签 (更稳)
    # =========================================================================
    subtype_cols = [
        'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack', 'sexual_explicit'
    ]
    # 3.1 主任务标签: toxicity 二分类 (用 target >= 0.5 二值化)
    target_col = 'target'
    text_col = 'comment_text'
    
    use_cols = [text_col, target_col] + subtype_cols + identity_cols
    
    # 读取数据
    df = pd.read_csv(input_path, usecols=use_cols)
    
    print("Processing labels...")
    # 填充 NaN 为 0
    df[identity_cols] = df[identity_cols].fillna(0)
    df[subtype_cols] = df[subtype_cols].fillna(0)
    
    # 计算 has_identity 标识 (max >= 0.5)
    df['has_identity'] = (df[identity_cols].max(axis=1) >= 0.5).astype(int)
    
    # 主任务二分类标签
    df['y_tox'] = (df[target_col] >= 0.5).astype(int)
    
    # =========================================================================
    # 三分数据划分 (学术严谨性)
    # Train: 80% 用于模型训练
    # Val:   10% 用于训练过程中的验证与早停 (Early Stopping)
    # Test:  10% 用于最终评估，训练期间从未见过
    # =========================================================================
    # 第一步：先分出 80% Train 和 20% 临时集
    train_df, temp_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df['y_tox'])
    # 第二步：将临时集对半分为 Val 和 Test (各占总体的 10%)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['y_tox'])
    
    print(f"Train size: {len(train_df)} | Val size: {len(val_df)} | Test size: {len(test_df)}")
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_parquet(os.path.join(output_dir, 'train_processed.parquet'), index=False)
    val_df.to_parquet(os.path.join(output_dir, 'val_processed.parquet'), index=False)
    test_df.to_parquet(os.path.join(output_dir, 'test_processed.parquet'), index=False)
    print("Preprocessing completed. Train/Val/Test files saved to parquet.")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    INPUT_CSV = os.path.join(BASE_DIR, "data", "train.csv")
    OUTPUT_DIR = os.path.join(BASE_DIR, "data")
    preprocess_data(INPUT_CSV, OUTPUT_DIR)
