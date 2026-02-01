#!/bin/bash

# 1. 设置镜像站端点
export HF_ENDPOINT="https://hf-mirror.com"
# 开启增强下载模式
export HF_HUB_ENABLE_HF_TRANSFER=1
# 禁用离线模式
export TRANSFORMERS_OFFLINE=0
export HF_HUB_OFFLINE=0

# 2. 设置路径
BASE_DIR=$(pwd)
CACHE_DIR="${BASE_DIR}/pretrained_models"
mkdir -p "$CACHE_DIR"

echo "=========================================================="
echo "Hugging Face 模型镜像下载工具 (抗 429 频率限制版)"
echo "当前端点: $HF_ENDPOINT"
echo "如果遇到 429 报错，脚本会自动等待并重试"
echo "=========================================================="

# 安装/更新下载增强组件
echo "正在检查下载组件..."
pip install -U huggingface_hub hf_transfer

# 待下载模型列表
models=(
    "bert-base-uncased"
    "roberta-base"
    "microsoft/deberta-base"
    "microsoft/deberta-v3-base"
)

for model in "${models[@]}"
do
    echo ""
    echo ">>>> 正在处理: $model"
    
    # 无限循环重试，直到成功
    retries=0
    while true; do
        echo "尝试下载 (第 $((retries+1)) 次)..."
        
        # 核心下载命令 (改用 python -m 调用以避开 PATH 问题)
        python -m huggingface_hub.commands.huggingface_cli download "$model" --cache-dir "${CACHE_DIR}/hub" --resume-download
        
        if [ $? -eq 0 ]; then
            echo "✅ $model 下载成功！"
            break
        else
            echo "⚠️  遇到频率限制 (429) 或网络波动。等待 30 秒后再次尝试..."
            # 增加随机抖动，防止固定频率被检查
            sleep_time=$((30 + RANDOM % 30))
            sleep $sleep_time
            retries=$((retries+1))
        fi
    done
done

echo ""
echo "=========================================================="
echo "所有模型已成功同步到本地缓存！"
echo "=========================================================="
