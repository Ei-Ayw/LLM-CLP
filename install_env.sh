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

# 3. 执行安装 (使用清华镜像加速)
echo "Installing dependencies with TUNA mirror..." >> install_pytorch.log

# 基础库与深度学习
echo "> Installing torch suite..." >> install_pytorch.log
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 >> install_pytorch.log 2>&1

echo "> Installing transformers suite..." >> install_pytorch.log
pip install transformers sentencepiece sacremoses -i https://pypi.tuna.tsinghua.edu.cn/simple >> install_pytorch.log 2>&1

# 数据处理与科学计算
echo "> Installing data science stack..." >> install_pytorch.log
pip install pandas scikit-learn pyarrow tqdm numpy -i https://pypi.tuna.tsinghua.edu.cn/simple >> install_pytorch.log 2>&1

# 文本增强工具
echo "> Installing nltk..." >> install_pytorch.log
pip install nltk -i https://pypi.tuna.tsinghua.edu.cn/simple >> install_pytorch.log 2>&1

# 可视化
echo "> Installing matplotlib..." >> install_pytorch.log
pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple >> install_pytorch.log 2>&1

if [ $? -eq 0 ]; then
    echo "Summary: Global dependencies installed successfully!" >> install_pytorch.log
    # 资源下载 (NLTK)
    echo "> Downloading NLTK resources..." >> install_pytorch.log
    python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('omw-1.4'); nltk.download('stopwords')" >> install_pytorch.log 2>&1
else
    echo "Summary: Some installations failed. Please check install_pytorch.log" >> install_pytorch.log
fi
