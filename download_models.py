import os
import subprocess
import sys

# 1. 基础配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "pretrained_models")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = CACHE_DIR

from huggingface_hub import snapshot_download

def run_download(model_id):
    print(f"\n>>>> 开始下载模型: {model_id}")
    try:
        # 使用 snapshot_download API，支持断点续传和镜像站
        snapshot_download(
            repo_id=model_id,
            cache_dir=CACHE_DIR,
            resume_download=True,
            local_files_only=False
        )
        print(f"✅ 模型 {model_id} 下载/校验完成！")
    except Exception as e:
        print(f"❌ 模型 {model_id} 下载失败: {e}")
        print(f"提示: 请检查网络连接或手动重试。")

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
