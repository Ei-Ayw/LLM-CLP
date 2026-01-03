import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def preprocess_data(input_path, output_dir):
    print(f"Loading data from {input_path}...")
    # 由于数据量大，仅读取需要的列
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
    
    # 训练/验证集划分 (95/5 比例，因为数据量巨大)
    train_df, val_df = train_test_split(df, test_size=0.05, random_state=42, stratify=df['y_tox'])
    
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_parquet(os.path.join(output_dir, 'train_processed.parquet'), index=False)
    val_df.to_parquet(os.path.join(output_dir, 'val_processed.parquet'), index=False)
    print("Preprocessing completed. Files saved to parquet.")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_CSV = os.path.join(BASE_DIR, "data", "train.csv")
    OUTPUT_DIR = os.path.join(BASE_DIR, "data")
    preprocess_data(INPUT_CSV, OUTPUT_DIR)
