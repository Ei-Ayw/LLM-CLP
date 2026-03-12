"""
=============================================================================
反事实生成: LLM 方法 (核心创新)
支持智谱 GLM API / OpenAI 兼容 API
支持多 API Key 轮询 + 高并发
=============================================================================
"""
import os
import json
import time
import argparse
import threading
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =====================================================
# Prompt 模板
# =====================================================
PROMPT_TEMPLATE_EN = """You are an expert NLP data augmentation specialist. Rewrite the following text by replacing references to {source_group} with {target_group}.

STRICT RULES:
1. ONLY change identity-related terms (group names, cultural markers, proper nouns)
2. Preserve the EXACT syntactic structure, sentiment, and toxicity level
3. Make culturally appropriate substitutions (e.g., mosque→church, hijab→cross necklace)
4. Do NOT add, remove, or rephrase other content
5. Return ONLY the rewritten text, nothing else

Original text: "{text}"
Source group: {source_group}
Target group: {target_group}

Rewritten text:"""

# 群体替换对
IDENTITY_SWAP_PAIRS = [
    ('muslim', 'christian'), ('christian', 'muslim'),
    ('muslim', 'jewish'), ('jewish', 'muslim'),
    ('black', 'white'), ('white', 'black'),
    ('african', 'european'), ('asian', 'european'),
    ('women', 'men'), ('men', 'women'),
    ('gay', 'straight'), ('lesbian', 'straight'),
    ('homosexual', 'heterosexual'),
    ('islam', 'christianity'),
    ('lgbtq', 'heterosexual'),
    ('disabled', 'abled'),
]

# 文本中的群体检测关键词
GROUP_KEYWORDS = {
    'muslim': ['muslim', 'islam', 'mosque', 'quran', 'hijab', 'allah', 'muhammad'],
    'christian': ['christian', 'church', 'bible', 'jesus', 'christ', 'pastor'],
    'jewish': ['jewish', 'jew', 'synagogue', 'torah', 'rabbi', 'israel'],
    'black': ['black', 'african', 'negro', 'nigger', 'nigga'],
    'white': ['white', 'caucasian', 'european'],
    'asian': ['asian', 'chinese', 'japanese', 'korean', 'oriental'],
    'women': ['woman', 'women', 'female', 'girl', 'she', 'her', 'feminist'],
    'men': ['man', 'men', 'male', 'boy', 'he', 'his'],
    'gay': ['gay', 'homosexual', 'lgbtq', 'queer', 'lesbian', 'bisexual', 'transgender'],
    'disabled': ['disabled', 'disability', 'mental illness', 'mentally ill', 'retard'],
}


def detect_groups(text):
    """检测文本中提及的身份群体"""
    text_lower = text.lower()
    detected = []
    for group, keywords in GROUP_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            detected.append(group)
    return detected


def get_swap_targets(source_group, max_targets=2):
    """获取可替换的目标群体"""
    targets = []
    for src, tgt in IDENTITY_SWAP_PAIRS:
        if src == source_group and tgt not in targets:
            targets.append(tgt)
    return targets[:max_targets]


# =====================================================
# API 调用: 智谱 GLM (支持多 Key 轮询)
# =====================================================
class ZhipuGeneratorPool:
    """多 API Key 轮询的智谱生成器池, 线程安全"""

    def __init__(self, api_keys, model="glm-4-flash"):
        if isinstance(api_keys, str):
            api_keys = [api_keys]
        self.model = model
        self.clients = []
        self._lock = threading.Lock()
        self._counter = 0

        # 优先尝试新版 zai-sdk
        try:
            from zai import ZhipuAiClient
            for key in api_keys:
                self.clients.append(ZhipuAiClient(api_key=key))
            print(f"  [SDK] zai-sdk | {len(self.clients)} 个 API Key")
        except ImportError:
            try:
                from zhipuai import ZhipuAI
                for key in api_keys:
                    self.clients.append(ZhipuAI(api_key=key))
                print(f"  [SDK] zhipuai | {len(self.clients)} 个 API Key")
            except ImportError:
                raise ImportError("请安装智谱 SDK: pip install zai-sdk 或 pip install zhipuai")

    def _get_client(self):
        """轮询获取下一个 client"""
        with self._lock:
            client = self.clients[self._counter % len(self.clients)]
            self._counter += 1
            return client

    def generate(self, text, source_group, target_group):
        prompt = PROMPT_TEMPLATE_EN.format(
            text=text, source_group=source_group, target_group=target_group
        )
        client = self._get_client()
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=512,
                )
                result = response.choices[0].message.content.strip()
                result = result.strip('"').strip("'").strip()
                return result
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1.0 * (attempt + 1))
                    client = self._get_client()  # 换个 key 重试
                else:
                    return None


# 保持向后兼容的别名
class ZhipuGenerator(ZhipuGeneratorPool):
    def __init__(self, api_key, model="glm-4-flash"):
        super().__init__(api_keys=[api_key], model=model)


# =====================================================
# API 调用: OpenAI 兼容 (备用)
# =====================================================
class OpenAICompatGenerator:
    def __init__(self, api_key, base_url=None, model="gpt-4o-mini"):
        from openai import OpenAI
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model

    def generate(self, text, source_group, target_group):
        prompt = PROMPT_TEMPLATE_EN.format(
            text=text, source_group=source_group, target_group=target_group
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=512,
            )
            result = response.choices[0].message.content.strip()
            result = result.strip('"').strip("'").strip()
            return result
        except Exception as e:
            print(f"  [API Error] {e}")
            return None


# =====================================================
# 主流程: 并发批量生成反事实
# =====================================================
def generate_counterfactuals_for_dataset(
    df, generator, text_col='text', id_col='post_id',
    max_cf_per_sample=2, save_path=None, resume=True,
    max_workers=20,
):
    """
    为数据集并发生成 LLM 反事实

    Args:
        df: 包含文本的 DataFrame
        generator: ZhipuGeneratorPool / ZhipuGenerator / OpenAICompatGenerator
        max_cf_per_sample: 每条样本最多生成几个反事实
        save_path: 保存路径 (支持断点续传)
        resume: 是否从已有结果继续
        max_workers: 并发线程数
    """
    # 断点续传: 加载已有结果
    existing = set()
    existing_records = []
    if resume and save_path and os.path.exists(save_path):
        existing_df = pd.read_parquet(save_path)
        for _, row in existing_df.iterrows():
            key = f"{row['post_id']}_{row['source_group']}_{row['target_group']}"
            existing.add(key)
            existing_records.append(row.to_dict())
        print(f"  [Resume] 已有 {len(existing)} 条结果，跳过已生成的")

    # 1. 先收集所有需要调用 API 的任务
    tasks = []
    for idx, row in df.iterrows():
        text = row[text_col]
        pid = row[id_col] if id_col in df.columns else f"sample_{idx}"

        detected_groups = detect_groups(text)
        if not detected_groups:
            continue

        for source_group in detected_groups:
            targets = get_swap_targets(source_group, max_targets=max_cf_per_sample)
            for target_group in targets:
                key = f"{pid}_{source_group}_{target_group}"
                if key in existing:
                    continue
                tasks.append({
                    'pid': pid,
                    'text': text,
                    'source_group': source_group,
                    'target_group': target_group,
                    'key': key,
                })

    print(f"  [Tasks] 需要生成 {len(tasks)} 条 (已跳过 {len(existing)} 条)")

    if not tasks:
        print("  [Done] 无需生成新数据")
        if existing_records:
            return pd.DataFrame(existing_records)
        return pd.DataFrame()

    # 2. 并发调用 API
    records = list(existing_records)  # 从已有结果开始
    failed = 0
    lock = threading.Lock()
    save_lock = threading.Lock()

    def process_one(task):
        cf_text = generator.generate(task['text'], task['source_group'], task['target_group'])
        if cf_text and cf_text != task['text'] and len(cf_text) > 5:
            return {
                'post_id': task['pid'],
                'original_text': task['text'],
                'cf_text': cf_text,
                'source_group': task['source_group'],
                'target_group': task['target_group'],
                'method': 'llm',
            }
        return None

    pbar = tqdm(total=len(tasks), desc="LLM反事实生成")
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one, t): t for t in tasks}

        for future in as_completed(futures):
            result = future.result()
            with lock:
                if result:
                    records.append(result)
                else:
                    failed += 1
                completed += 1
                pbar.update(1)

                # 每 500 条自动保存
                if save_path and len(records) > 0 and completed % 500 == 0:
                    with save_lock:
                        pd.DataFrame(records).to_parquet(save_path, index=False)

    pbar.close()

    result_df = pd.DataFrame(records)
    if save_path:
        result_df.to_parquet(save_path, index=False)
        new_count = len(result_df) - len(existing_records)
        print(f"\n  [完成] 新生成 {new_count} 条, 总计 {len(result_df)} 条 "
              f"(失败 {failed})")
        print(f"  保存至: {save_path}")

    return result_df


# =====================================================
# CLI
# =====================================================
def main():
    parser = argparse.ArgumentParser(description="LLM 反事实生成")
    parser.add_argument("--dataset", type=str, default="hatexplain",
                        choices=["hatexplain", "toxigen"])
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--api", type=str, default="zhipu",
                        choices=["zhipu", "openai"])
    parser.add_argument("--api_key", type=str, required=True,
                        help="API key, 多个用逗号分隔")
    parser.add_argument("--model", type=str, default=None,
                        help="模型名称 (默认: zhipu=glm-4-flash, openai=gpt-4o-mini)")
    parser.add_argument("--base_url", type=str, default=None,
                        help="OpenAI 兼容 API 的 base_url")
    parser.add_argument("--max_cf", type=int, default=2,
                        help="每条样本最多生成几个反事实")
    parser.add_argument("--max_workers", type=int, default=20,
                        help="并发线程数")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(BASE_DIR, "data", "causal_fair"))
    args = parser.parse_args()

    # 加载数据
    data_path = os.path.join(args.data_dir, f"{args.dataset}_{args.split}.parquet")
    if not os.path.exists(data_path):
        print(f"[Error] 数据文件不存在: {data_path}")
        print(f"  请先运行: python src_script/data/download_datasets.py")
        return

    df = pd.read_parquet(data_path)
    print(f"[Data] 加载 {args.dataset}/{args.split}: {len(df)} 条")

    # 支持多 key (逗号分隔)
    api_keys = [k.strip() for k in args.api_key.split(",") if k.strip()]

    # 初始化生成器
    if args.api == "zhipu":
        model = args.model or "glm-4-flash"
        generator = ZhipuGeneratorPool(api_keys=api_keys, model=model)
    else:
        model = args.model or "gpt-4o-mini"
        generator = OpenAICompatGenerator(
            api_key=api_keys[0], base_url=args.base_url, model=model
        )

    print(f"[API] {args.api} / {model} / {len(api_keys)} keys / {args.max_workers} workers")

    # 保存路径
    save_path = os.path.join(
        args.data_dir, f"{args.dataset}_{args.split}_cf_llm.parquet"
    )

    # 确定 id_col
    id_col = 'post_id' if 'post_id' in df.columns else None
    if id_col is None:
        df['post_id'] = [f"sample_{i}" for i in range(len(df))]
        id_col = 'post_id'

    # 生成
    generate_counterfactuals_for_dataset(
        df, generator,
        text_col='text', id_col=id_col,
        max_cf_per_sample=args.max_cf,
        save_path=save_path,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
