import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/data/dell/workspace/01_nlp_toxicity_classification/pretrained_models'

from huggingface_hub import snapshot_download
import time

max_retries = 30
for attempt in range(max_retries):
    try:
        print(f'\n>>> 尝试下载 microsoft/deberta-v3-base (第 {attempt+1}/{max_retries} 次)...')
        print('>>> 注意：将下载所有文件，包括权重文件 (*.bin, *.safetensors)...')
        
        snapshot_download(
            repo_id='microsoft/deberta-v3-base',
            cache_dir='/data/dell/workspace/01_nlp_toxicity_classification/pretrained_models',
            resume_download=True,
            local_files_only=False,
            # 明确要求下载所有文件，不过滤任何格式
            allow_patterns=None,  
            ignore_patterns=None,
        )
        
        # 验证权重文件是否存在
        import glob
        weight_files = glob.glob('/data/dell/workspace/01_nlp_toxicity_classification/pretrained_models/hub/models--microsoft--deberta-v3-base/snapshots/**/*.bin', recursive=True)
        weight_files += glob.glob('/data/dell/workspace/01_nlp_toxicity_classification/pretrained_models/hub/models--microsoft--deberta-v3-base/snapshots/**/*.safetensors', recursive=True)
        
        if weight_files:
            print(f'\n✅ 下载成功！找到权重文件：')
            for f in weight_files:
                size = os.path.getsize(f) / (1024**3)
                print(f'   - {os.path.basename(f)} ({size:.2f} GB)')
        else:
            print('\n⚠️ 警告：下载完成但未找到权重文件，尝试使用原始 HuggingFace Hub...')
            raise Exception("权重文件缺失")
        break
        
    except Exception as e:
        print(f'⚠️ 下载失败: {e}')
        if attempt < max_retries - 1:
            wait = 10 + attempt * 3
            print(f'⏳ 等待 {wait} 秒后重试...')
            time.sleep(wait)
        else:
            print('\n❌ 达到最大重试次数。')
            print('💡 建议：如果镜像站无法下载权重，请尝试：')
            print('   1. 使用 VPN 直接从 huggingface.co 下载')
            print('   2. 或手动从其他渠道获取 pytorch_model.bin')
