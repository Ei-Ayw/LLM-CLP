#!/bin/bash
# 自动从 Kaggle 下载数据集到 data/ 目录

# 1. 设置目录
PROJECT_DIR=$(pwd)
DATA_DIR="$PROJECT_DIR/data"
mkdir -p "$DATA_DIR"

# 2. 检查并安装 Kaggle CLI
echo "Checking Kaggle API Client..."
pip install kaggle -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 自动配置 kaggle.json 凭证
# 3. 配置 Kaggle 凭证 (使用您的 API Token)
echo "Configuring Kaggle API Token..."
export KAGGLE_API_TOKEN=KGAT_5e14fb44dabe299ff6edf8456d989c93

# 验证 Token 是否有效 (可选)
# kaggle competitions list --head 5

# 4. 下载数据集
echo "Downloading dataset to $DATA_DIR..."
cd "$DATA_DIR"

# 使用 Kaggle API 下载
kaggle competitions download -c jigsaw-unintended-bias-in-toxicity-classification

# 5. 解压
if [ -f "jigsaw-unintended-bias-in-toxicity-classification.zip" ]; then
    echo "Unzipping dataset..."
    unzip -o jigsaw-unintended-bias-in-toxicity-classification.zip
    echo "Done! Data is ready in $DATA_DIR"
    ls -lh "$DATA_DIR/train.csv"
else
    echo "Download failed. Please check your network or Kaggle API permissions."
fi
