"""
反事实对数据结构定义。

定义反事实生成输入/输出的标准结构。
"""
from dataclasses import dataclass, field
from typing import Optional, List
import json


@dataclass
class CounterfactualPair:
    """单条原始-反事实配对。

    Attributes:
        original_text: 原始输入文本。
        cf_text: 身份替换后的反事实版本。
        source_group: 被替换的身份群体（例如 "muslim"）。
        target_group: 引入的目标身份群体（例如 "christian"）。
        method: 生成方法（"llm"、"swap"、"qwen"、"free"）。
        post_id: 原始数据样本 ID（可选）。
        quality_passed: 是否通过质量验证（可选）。
        toxicity_orig: 原始文本的 P(toxic) 预测值（可选）。
        toxicity_cf: 反事实文本的 P(toxic) 预测值（可选）。
        has_identity_reference: 原始文本是否包含身份引用。
        source_identity: 被替换的具体身份词。
        source_identity_type: 源身份的类型（例如 "religion"、"gender"）。
        source_span: 源身份词在原始文本中的精确片段。
        implicit_identity: 身份引用是否为隐式。
        target_identity: 引入的具体目标身份词。
        target_identity_type: 目标身份的类型。
        counterfactual: 反事实文本（cf_text 的别名）。
        changed_span_original: 原始文本中被修改的片段。
        changed_span_counterfactual: 反事实文本中被修改的片段。
        generation_valid: 是否通过生成有效性检查。
        judge_valid: 是否通过判别有效性检查。
        toxicity_preserved: 替换后毒性是否保留。
        sentiment_preserved: 情感是否保留。
        minimal_edit: 是否为最小化编辑。
        identity_changed: 身份是否成功替换。
        contextually_plausible: 反事实是否在语境上合理。
        semantic_similarity: 原始与反事实文本的相似度分数。
        toxicity_drift: 毒性分数的漂移量。
        normalized_edit_distance: 文本间的归一化编辑距离。
        identity_change_success: 身份替换是否成功。
        delta_gen: 生成差异指标。
        delta_sem: 语义差异指标。
        generator_backend: 生成所用后端（例如 "anthropic"）。
        generator_model: 生成所用模型。
        judge_backend: 判别所用后端。
        judge_model: 判别所用模型。
        temperature: 生成的温度参数。
        top_p: 生成的 top-p 参数。
        max_tokens: 生成的最大 token 数。
        prompt_version: 所用提示词版本。
    """

    original_text: str
    cf_text: str
    source_group: str = ""
    target_group: str = ""
    method: str = "unknown"
    post_id: str = ""
    quality_passed: Optional[bool] = None
    toxicity_orig: Optional[float] = None
    toxicity_cf: Optional[float] = None

    # Identity reference fields
    has_identity_reference: bool = False
    source_identity: str = ""
    source_identity_type: str = ""
    source_span: str = ""
    implicit_identity: bool = False
    target_identity: str = ""
    target_identity_type: str = ""
    counterfactual: str = ""
    changed_span_original: str = ""
    changed_span_counterfactual: str = ""

    # Validation fields
    generation_valid: bool = False
    judge_valid: bool = False
    toxicity_preserved: bool = False
    sentiment_preserved: bool = False
    minimal_edit: bool = False
    identity_changed: bool = False
    contextually_plausible: bool = False
    semantic_similarity: float = 0.0
    toxicity_drift: float = 0.0
    normalized_edit_distance: float = 0.0
    identity_change_success: bool = False

    # Delta metrics
    delta_gen: int = 0
    delta_sem: int = 0

    # Generation parameters
    generator_backend: str = ""
    generator_model: str = ""
    judge_backend: str = ""
    judge_model: str = ""
    temperature: float = 0.0
    top_p: float = 0.0
    max_tokens: int = 0
    prompt_version: str = ""

    def to_dict(self) -> dict:
        return {
            "original_text": self.original_text,
            "cf_text": self.cf_text,
            "source_group": self.source_group,
            "target_group": self.target_group,
            "method": self.method,
            "post_id": self.post_id,
            "quality_passed": self.quality_passed,
            "toxicity_orig": self.toxicity_orig,
            "toxicity_cf": self.toxicity_cf,
            "has_identity_reference": self.has_identity_reference,
            "source_identity": self.source_identity,
            "source_identity_type": self.source_identity_type,
            "source_span": self.source_span,
            "implicit_identity": self.implicit_identity,
            "target_identity": self.target_identity,
            "target_identity_type": self.target_identity_type,
            "counterfactual": self.counterfactual,
            "changed_span_original": self.changed_span_original,
            "changed_span_counterfactual": self.changed_span_counterfactual,
            "generation_valid": self.generation_valid,
            "judge_valid": self.judge_valid,
            "toxicity_preserved": self.toxicity_preserved,
            "sentiment_preserved": self.sentiment_preserved,
            "minimal_edit": self.minimal_edit,
            "identity_changed": self.identity_changed,
            "contextually_plausible": self.contextually_plausible,
            "semantic_similarity": self.semantic_similarity,
            "toxicity_drift": self.toxicity_drift,
            "normalized_edit_distance": self.normalized_edit_distance,
            "identity_change_success": self.identity_change_success,
            "delta_gen": self.delta_gen,
            "delta_sem": self.delta_sem,
            "generator_backend": self.generator_backend,
            "generator_model": self.generator_model,
            "judge_backend": self.judge_backend,
            "judge_model": self.judge_model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "prompt_version": self.prompt_version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CounterfactualPair":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "CounterfactualPair":
        return cls.from_dict(json.loads(s))


def validate_cf_pair_schema(data: dict) -> List[str]:
    """验证字典是否符合 CounterfactualPair 结构定义。

    Args:
        data: 待验证的字典。

    Returns:
        验证错误信息列表。列表为空表示验证通过。
    """
    errors = []
    required = ["original_text", "cf_text"]

    for field in required:
        if field not in data:
            errors.append(f"Missing required field: {field}")
        elif not isinstance(data[field], str) or len(data[field].strip()) == 0:
            errors.append(f"Field '{field}' must be a non-empty string")

    return errors