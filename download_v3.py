import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/data/dell/workspace/01_nlp_toxicity_classification/pretrained_models'
from huggingface_hub import snapshot_download
import time
max_retries = 30
for attempt in range(max_retries):
    try:
        print(f'\n>>> 尝试下载 microsoft/deberta-v3-base (第 {attempt+1}/{max_retries} 次)...')
        snapshot_download(
            repo_id='microsoft/deberta-v3-base',
            cache_dir='/data/dell/workspace/01_nlp_toxicity_classification/pretrained_models',
            resume_download=True,
            local_files_only=False
        )
        print('✅ 下载成功！')
        break
    except Exception as e:
        print(f'⚠️ 下载失败: {e}')
        if attempt < max_retries - 1:
            wait = 10 + attempt * 3
            print(f'⏳ 等待 {wait} 秒后重试...')
            time.sleep(wait)
        else:
            print('❌ 达到最大重试次数，下载失败。')