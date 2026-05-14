"""
Counterfactual pair data schema.

Defines the canonical structure for counterfactual generation input/output.
"""
from dataclasses import dataclass, field
from typing import Optional, List
import json


@dataclass
class CounterfactualPair:
    """A single original-counterfactual pair.

    Attributes:
        original_text: The original input text.
        cf_text: The counterfactual version after identity substitution.
        source_group: The identity group replaced (e.g. "muslim").
        target_group: The identity group introduced (e.g. "christian").
        method: Generation method ("llm", "swap", "qwen", "free").
        post_id: Original data sample ID (optional).
        quality_passed: Whether quality validation passed (optional).
        toxicity_orig: Predicted P(toxic) for original (optional).
        toxicity_cf: Predicted P(toxic) for counterfactual (optional).
        has_identity_reference: Whether the original contains an identity reference.
        source_identity: The specific identity term replaced.
        source_identity_type: Type of the source identity (e.g., "religion", "gender").
        source_span: The exact span of the source identity in original text.
        implicit_identity: Whether the identity reference is implicit.
        target_identity: The specific identity term introduced.
        target_identity_type: Type of the target identity.
        counterfactual: The counterfactual text (alias for cf_text).
        changed_span_original: The span changed in original text.
        changed_span_counterfactual: The span changed in counterfactual text.
        generation_valid: Whether generation passed validity checks.
        judge_valid: Whether judge passed validity checks.
        toxicity_preserved: Whether toxicity is preserved after substitution.
        sentiment_preserved: Whether sentiment is preserved.
        minimal_edit: Whether the edit is minimal.
        identity_changed: Whether the identity was successfully changed.
        contextually_plausible: Whether the counterfactual is contextually plausible.
        semantic_similarity: Similarity score between original and counterfactual.
        toxicity_drift: Drift in toxicity score.
        normalized_edit_distance: Normalized edit distance between texts.
        identity_change_success: Whether identity change was successful.
        delta_gen: Generation delta metric.
        delta_sem: Semantic delta metric.
        generator_backend: Backend used for generation (e.g., "anthropic").
        generator_model: Model used for generation.
        judge_backend: Backend used for judge.
        judge_model: Model used for judge.
        temperature: Temperature parameter for generation.
        top_p: Top-p parameter for generation.
        max_tokens: Max tokens for generation.
        prompt_version: Version of the prompt used.
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
    """Validate a dict against CounterfactualPair schema.

    Args:
        data: Dictionary to validate.

    Returns:
        List of validation error messages. Empty list = valid.
    """
    errors = []
    required = ["original_text", "cf_text"]

    for field in required:
        if field not in data:
            errors.append(f"Missing required field: {field}")
        elif not isinstance(data[field], str) or len(data[field].strip()) == 0:
            errors.append(f"Field '{field}' must be a non-empty string")

    return errors