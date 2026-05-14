"""
Prompt templates and identity swap mappings for counterfactual generation.
"""
from typing import Dict, List, Tuple

# =====================================================
# Prompt Templates
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
# 16 Bidirectional Identity Swap Pairs
# Used in fixed mapping mode
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
# Identity Detection Keywords
# Used for triggering counterfactual generation
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
    """Detect identity groups present in a text.

    Args:
        text: Input text.

    Returns:
        List of detected group names (e.g. ["muslim", "women"]).
    """
    text_lower = text.lower()
    detected = []
    for group, keywords in GROUP_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            detected.append(group)
    return detected


def get_swap_targets(source_group: str, max_targets: int = 2) -> List[str]:
    """Get available target groups for a given source group.

    Args:
        source_group: Source identity group name.
        max_targets: Maximum number of targets to return.

    Returns:
        List of target group names.
    """
    targets = []
    for src, tgt in IDENTITY_SWAP_PAIRS:
        if src == source_group and tgt not in targets:
            targets.append(tgt)
    return targets[:max_targets]


def get_all_identity_groups() -> List[str]:
    """Return all supported identity group names."""
    return list(GROUP_KEYWORDS.keys())