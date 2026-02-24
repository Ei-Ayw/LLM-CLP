import os
import subprocess
import sys

# 1. 基础配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "pretrained_models")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = CACHE_DIR

def run_download(model_id):
    print(f"\n>>>> 开始下载模型: {model_id}")
    # 使用 huggingface-cli download 命令
    # --resume-download: 支持断点续传
    # --local-dir-use-symlinks False: 直接下载文件，不使用软连接（这样文件结构最清晰）
    cmd = [
        sys.executable, "-m", "huggingface_hub.commands.cli", "download",
        model_id,
        "--cache-dir", CACHE_DIR,
        "--resume-download"
    ]
    
    try:
        # 使用 subprocess 运行，以便实时看到进度
        subprocess.run(cmd, check=True)
        print(f"✅ 模型 {model_id} 下载完成！")
    except subprocess.CalledProcessError as e:
        print(f"❌ 模型 {model_id} 下载失败。")
        print(f"尝试手动运行以下命令查看报错：\nexport HF_ENDPOINT=https://hf-mirror.com && huggingface-cli download {model_id} --cache-dir {CACHE_DIR}")

if __name__ == "__main__":
    # 确保文件夹存在
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    models = [
        "microsoft/deberta-v3-base",
        "bert-base-uncased",
        "roberta-base",
        "microsoft/deberta-base"
    ]
    
    for m in models:
        run_download(m)

    print("\n所有任务已尝试完成。")
