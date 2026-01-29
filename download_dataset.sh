#!/bin/bash
# 自动从 Kaggle 下载数据集到 data/ 目录

# 1. 设置目录
PROJECT_DIR=$(pwd)
DATA_DIR="$PROJECT_DIR/data"
mkdir -p "$DATA_DIR"

# 2. 检查并安装 Kaggle CLI
# 2. 使用 Hugging Face Datasets 下载 (最稳方案)
echo "Installing/Updating 'datasets' library..."
pip install datasets pandas pyarrow -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "Downloading dataset from Hugging Face Mirror..."
cd "$PROJECT_DIR"

# 创建一个临时的 Python 脚本来处理下载和转换
cat <<EOF > download_hf.py
import os
import pandas as pd
from datasets import load_dataset, set_caching_enabled

DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

# 启用缓存防止重复下载中断
set_caching_enabled(True)

# 设置 HF 镜像端点 (关键!)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print(f"Downloading from Hugging Face to {DATA_DIR}...")
try:
    # 尝试加载 Google Jigsaw Unintended Bias 数据集
    ds = load_dataset("google/jigsaw_unintended_bias", split="train")
    print("Download complete. Converting to CSV...")
    
    # 转换为 DataFrame
    df = ds.to_pandas()
    
    # 打印列名确认
    print("Columns found:", df.columns.tolist())
    
    # 保存为 train.csv
    csv_path = os.path.join(DATA_DIR, "train.csv")
    df.to_csv(csv_path, index=False)
    print(f"Success! Saved to {csv_path}")
    
except Exception as e:
    print(f"Error: {e}")
EOF

# 运行 Python 下载脚本
python download_hf.py

# 清理
rm download_hf.py

if [ -f "$DATA_DIR/train.csv" ]; then
    echo "Dataset is ready: $DATA_DIR/train.csv"
else
    echo "Download failed."
fi
