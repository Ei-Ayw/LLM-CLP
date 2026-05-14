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