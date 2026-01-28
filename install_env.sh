#!/bin/bash
# 自动安装 PyTorch (CUDA 12.1 for CUDA 12.x drivers)
# 使用 nohup 后台运行防止 SSH 断开中断安装

# 1. 尝试初始化 Conda (适配常见安装路径)
# 注意：脚本中直接 conda activate 往往需要先 source conda.sh
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null

# 2. 激活指定环境 pyt38
echo "Activating conda environment: pyt38..."
conda activate pyt38 || source activate pyt38

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'pyt38'. Make sure it exists."
    exit 1
fi

echo "Environment activated. Python path: $(which python)"
echo "Starting PyTorch Installation..."
echo "Log file: install_pytorch.log"

# 3. 执行安装 (包含所有项目依赖)
echo "Installing dependencies..." >> install_pytorch.log

# 基础库与深度学习
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 >> install_pytorch.log 2>&1
pip install transformers sentencepiece sacremoses >> install_pytorch.log 2>&1

# 数据处理与科学计算
pip install pandas scikit-learn pyarrow tqdm numpy >> install_pytorch.log 2>&1

# 文本增强工具
pip install nltk >> install_pytorch.log 2>&1

# 可视化
pip install matplotlib >> install_pytorch.log 2>&1

if [ $? -eq 0 ]; then
    echo "Installation Successful!" >> install_pytorch.log
    # 简单的验证与资源准备
    python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')" >> install_pytorch.log
    # 预下载 NLTK 资源避免多卡训练时竞争报错
    python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('omw-1.4'); nltk.download('stopwords')" >> install_pytorch.log 2>&1
else
    echo "Installation Failed!" >> install_pytorch.log
fi
