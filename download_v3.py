from modelscope import snapshot_download
print('>>> 正在从 ModelScope 下载 deberta-v3-base...')
model_dir = snapshot_download(
    'AI-ModelScope/deberta-v3-base', 
    cache_dir='/data/dell/workspace/01_nlp_toxicity_classification/pretrained_models_modelscope'
)
print(f'✅ 下载完成！模型路径: {model_dir}')