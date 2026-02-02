import os
# 不设置镜像，直连 HuggingFace 官方
# 如果需要代理，请先执行: export https_proxy="your_proxy_address"
os.environ['HF_HOME'] = '/data/dell/workspace/01_nlp_toxicity_classification/pretrained_models'

from transformers import AutoModel, AutoTokenizer
import time

print('>>> 正在从 HuggingFace 官方下载 microsoft/deberta-v3-base...')
print('>>> 如果网络不通，请设置代理: export https_proxy="your_proxy"')

max_retries = 10
for attempt in range(max_retries):
    try:
        print(f'\n>>> 尝试 {attempt+1}/{max_retries}...')
        
        # 使用 transformers 库下载，它会自动下载权重
        tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
        model = AutoModel.from_pretrained('microsoft/deberta-v3-base')
        
        print(f'\n✅ 下载成功！')
        print(f'>>> 模型已缓存到: {os.environ["HF_HOME"]}')
        
        # 验证权重文件
        import glob
        weight_files = glob.glob(f'{os.environ["HF_HOME"]}/hub/models--microsoft--deberta-v3-base/snapshots/**/*.bin', recursive=True)
        if weight_files:
            for f in weight_files:
                size = os.path.getsize(f) / (1024**3)
                print(f'   ✓ {os.path.basename(f)} ({size:.2f} GB)')
        break
        
    except Exception as e:
        print(f'⚠️ 下载失败: {e}')
        if attempt < max_retries - 1:
            wait = 5 + attempt * 2
            print(f'⏳ 等待 {wait} 秒后重试...')
            time.sleep(wait)
        else:
            print('\n❌ 下载失败。请检查：')
            print('   1. 网络连接是否正常')
            print('   2. 是否需要配置代理')
            print('   3. 或联系管理员开放 huggingface.co 访问权限')