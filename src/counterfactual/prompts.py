"""
反事实生成的提示模板与身份替换映射表。
"""
from typing import Dict, List, Tuple

# =====================================================
# 提示模板
# =====================================================

PROMPT_FIXED_MAPPING = """You are an expert NLP data augmentation specialist. Rewrite the following text by replacing references to {source_group} with {target_group}.

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

PROMPT_FREE_FORM = """Replace the identity group in the following text with a contextually plausible alternative identity, preserving syntax and sentiment.

STRICT RULES:
1. Identify the demographic/identity group being referenced
2. Replace it with a different but contextually appropriate group
3. Preserve the EXACT syntactic structure, sentiment, and toxicity level
4. Make culturally appropriate substitutions
5. Do NOT add, remove, or rephrase other content
6. Return ONLY the rewritten text, nothing else

Original text: "{text}"

Rewritten text:"""

# =====================================================
# 16 个双向身份替换对
# 用于固定映射模式
# =====================================================

IDENTITY_SWAP_PAIRS: List[Tuple[str, str]] = [
    ("muslim", "christian"),
    ("christian", "muslim"),
    ("muslim", "jewish"),
    ("jewish", "muslim"),
    ("black", "white"),
    ("white", "black"),
    ("african", "european"),
    ("asian", "european"),
    ("women", "men"),
    ("men", "women"),
    ("gay", "straight"),
    ("lesbian", "straight"),
    ("homosexual", "heterosexual"),
    ("islam", "christianity"),
    ("lgbtq", "heterosexual"),
    ("disabled", "abled"),
]

# =====================================================
# 身份检测关键词
# 用于触发反事实生成
# =====================================================

GROUP_KEYWORDS: Dict[str, List[str]] = {
    "muslim": ["muslim", "islam", "mosque", "quran", "hijab", "allah", "muhammad"],
    "christian": ["christian", "church", "bible", "jesus", "christ", "pastor"],
    "jewish": ["jewish", "jew", "synagogue", "torah", "rabbi", "israel"],
    "black": ["black", "african", "negro", "nigger", "nigga"],
    "white": ["white", "caucasian", "european"],
    "asian": ["asian", "chinese", "japanese", "korean", "oriental"],
    "women": ["woman", "women", "female", "girl", "she", "her", "feminist"],
    "men": ["man", "men", "male", "boy", "he", "his"],
    "gay": ["gay", "homosexual", "lgbtq", "queer", "lesbian", "bisexual", "transgender"],
    "disabled": ["disabled", "disability", "mental illness", "mentally ill", "retard"],
}


def detect_groups(text: str) -> List[str]:
    """检测文本中出现的身份群体。

    Args:
        text: 输入文本。

    Returns:
        检测到的群体名称列表（例如 ["muslim", "women"]）。
    """
    text_lower = text.lower()
    detected = []
    for group, keywords in GROUP_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            detected.append(group)
    return detected


def get_swap_targets(source_group: str, max_targets: int = 2) -> List[str]:
    """获取给定源群体的可用目标群体。

    Args:
        source_group: 源身份群体名称。
        max_targets: 返回的最大目标数量。

    Returns:
        目标群体名称列表。
    """
    targets = []
    for src, tgt in IDENTITY_SWAP_PAIRS:
        if src == source_group and tgt not in targets:
            targets.append(tgt)
    return targets[:max_targets]


def get_all_identity_groups() -> List[str]:
    """返回所有支持的身份群体名称。"""
    return list(GROUP_KEYWORDS.keys())