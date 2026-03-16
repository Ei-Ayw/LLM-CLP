"""
=============================================================================
反事实生成: 传统换词方法 (CDA-Swap, 对照组 baseline)
简单粗暴地把身份词替换为其他群体的词
=============================================================================
"""


# 双向身份词替换映射
IDENTITY_SWAP_GROUPS = {
    'race': {
        'black': ['white', 'asian', 'hispanic'],
        'white': ['black', 'asian', 'hispanic'],
        'african': ['european', 'asian'],
        'asian': ['african', 'european'],
        'hispanic': ['black', 'white'],
    },
    'religion': {
        'muslim': ['christian', 'jewish', 'buddhist'],
        'christian': ['muslim', 'jewish', 'buddhist'],
        'jewish': ['muslim', 'christian'],
        'islam': ['christianity', 'judaism'],
        'islamic': ['christian', 'jewish'],
        'mosque': ['church', 'synagogue', 'temple'],
        'church': ['mosque', 'synagogue', 'temple'],
        'quran': ['bible', 'torah'],
        'bible': ['quran', 'torah'],
    },
    'gender': {
        'women': ['men'],
        'men': ['women'],
        'woman': ['man'],
        'man': ['woman'],
        'she': ['he'],
        'he': ['she'],
        'her': ['his'],
        'his': ['her'],
        'mother': ['father'],
        'father': ['mother'],
    },
    'sexual_orientation': {
        'gay': ['straight', 'heterosexual'],
        'lesbian': ['straight', 'heterosexual'],
        'homosexual': ['heterosexual'],
        'lgbtq': ['heterosexual'],
        'queer': ['straight'],
    },
    'disability': {
        'disabled': ['abled', 'healthy'],
        'mental illness': ['physical health'],
        'mentally ill': ['physically healthy'],
    },
}

# 扁平化映射: source_word -> [(target_word, target_group), ...]
FLAT_SWAP_MAP = {}
for category, group_map in IDENTITY_SWAP_GROUPS.items():
    for source, targets in group_map.items():
        FLAT_SWAP_MAP[source] = targets


def generate_swap_counterfactuals(text, max_cf=3):
    """
    传统换词法生成反事实
    返回: [(counterfactual_text, source_word, target_word), ...]
    """
    text_lower = text.lower()
    results = []

    for source_word, target_words in FLAT_SWAP_MAP.items():
        if source_word in text_lower:
            for target_word in target_words[:max_cf]:
                # 保持原文大小写
                cf_text = text
                # 尝试多种大小写匹配
                for original_form in [source_word, source_word.capitalize(), source_word.upper()]:
                    if original_form in cf_text:
                        target_form = target_word
                        if original_form[0].isupper():
                            target_form = target_word.capitalize()
                        if original_form.isupper():
                            target_form = target_word.upper()
                        cf_text = cf_text.replace(original_form, target_form)
                        break

                if cf_text != text:
                    results.append({
                        'cf_text': cf_text,
                        'source_word': source_word,
                        'target_word': target_word,
                        'method': 'swap',
                    })

    return results[:max_cf]


def batch_generate_swap(texts, post_ids=None, max_cf_per_sample=3):
    """
    批量生成换词反事实
    返回: DataFrame with columns [post_id, original_text, cf_text, source_word, target_word, method]
    """
    import pandas as pd

    records = []
    for i, text in enumerate(texts):
        pid = post_ids[i] if post_ids is not None else f"sample_{i}"
        cfs = generate_swap_counterfactuals(text, max_cf=max_cf_per_sample)
        for cf in cfs:
            records.append({
                'post_id': pid,
                'original_text': text,
                **cf,
            })

    return pd.DataFrame(records)


if __name__ == "__main__":
    # 测试
    test_texts = [
        "Muslims are destroying this country",
        "She wore a hijab to the mosque",
        "Black people are always causing trouble",
        "Gay people should not have rights",
        "This is a normal sentence without identity words",
    ]

    for text in test_texts:
        cfs = generate_swap_counterfactuals(text)
        print(f"\n原始: {text}")
        for cf in cfs:
            print(f"  换词 → {cf['cf_text']}  ({cf['source_word']} → {cf['target_word']})")
