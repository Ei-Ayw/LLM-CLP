#!/bin/bash
# ============================================================================
# 创建全新的 Conda 环境来解决 SIGSEGV 问题
# 使用方法: bash create_new_env.sh
# ============================================================================

ENV_NAME="nlp_new"
PYTHON_VERSION="3.9"

echo "=============================================="
echo ">>> [1/7] 初始化 Conda..."
echo "=============================================="
# 尝试初始化 Conda (适配常见安装路径)
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || \
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null

# 检查是否有同名环境存在
if conda env list | grep -q "^${ENV_NAME} "; then
    echo ">>> [Warning] 环境 ${ENV_NAME} 已存在，将先删除..."
    conda deactivate 2>/dev/null
    conda env remove -n ${ENV_NAME} -y
fi

echo "=============================================="
echo ">>> [2/7] 创建新环境: ${ENV_NAME} (Python ${PYTHON_VERSION})..."
echo "=============================================="
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
if [ $? -ne 0 ]; then
    echo "❌ [Error] 创建 Conda 环境失败！"
    exit 1
fi

echo "=============================================="
echo ">>> [3/7] 激活新环境..."
echo "=============================================="
conda activate ${ENV_NAME}
if [ $? -ne 0 ]; then
    echo "❌ [Error] 激活环境失败！"
    exit 1
fi
echo ">>> Python 路径: $(which python)"
echo ">>> Python 版本: $(python --version)"

echo "=============================================="
echo ">>> [4/7] 安装 PyTorch (CUDA 12.1)..."
echo "=============================================="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if [ $? -ne 0 ]; then
    echo "❌ [Error] PyTorch 安装失败！"
    exit 1
fi

echo "=============================================="
echo ">>> [5/7] 安装 Transformers 及相关依赖 (锁定稳定版本)..."
echo "=============================================="
# 关键：锁定 transformers 和 protobuf 版本以避免 SIGSEGV
pip install transformers==4.36.2 \
            sentencepiece==0.1.99 \
            protobuf==3.20.3 \
            tokenizers==0.15.2 \
            sacremoses \
            -i https://pypi.tuna.tsinghua.edu.cn/simple
if [ $? -ne 0 ]; then
    echo "❌ [Error] Transformers 安装失败！"
    exit 1
fi

echo "=============================================="
echo ">>> [6/7] 安装数据科学与可视化工具..."
echo "=============================================="
pip install pandas \
            scikit-learn \
            pyarrow \
            tqdm \
            numpy \
            matplotlib \
            nltk \
            -i https://pypi.tuna.tsinghua.edu.cn/simple
if [ $? -ne 0 ]; then
    echo "❌ [Error] 数据科学库安装失败！"
    exit 1
fi

echo "=============================================="
echo ">>> [7/7] 下载 NLTK 资源并验证环境..."
echo "=============================================="
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('omw-1.4'); nltk.download('stopwords')"

# 验证环境
echo ""
echo ">>> 验证关键库导入..."
python -c "
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
print('✓ [1/5] Core libs OK')
import pandas; import numpy
print('✓ [2/5] Pandas/Numpy OK')
from sklearn import metrics
print('✓ [3/5] Sklearn OK')
from transformers import AutoTokenizer, DebertaV2Model
print('✓ [4/5] Transformers OK')
import torch
print(f'✓ [5/5] Torch OK (CUDA: {torch.cuda.is_available()})')
print('')
print('🎉 所有关键库导入成功！环境验证通过！')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "✅ 环境创建成功！"
    echo "=============================================="
    echo ""
    echo ">>> 使用方法:"
    echo "    conda activate ${ENV_NAME}"
    echo "    bash start_experiment.sh"
    echo ""
else
    echo ""
    echo "=============================================="
    echo "❌ 环境验证失败！请检查上方错误信息。"
    echo "=============================================="
    exit 1
fi
