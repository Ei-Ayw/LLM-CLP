# Future Method Changes

> Interface definitions for planned v2 improvements.
> These are **not implemented yet** — this document defines the target API.

## Counterfactual Generation

### Open-Set Controlled Rewriting

```python
from enum import Enum

class CounterfactualMode(str, Enum):
    """Generation modes for counterfactual texts."""
    FIXED16 = "fixed16"           # Current: 16 bidirectional pairs
    OPEN_SET = "open_set"         # Planned: LLM decides identity group
    FREE_FORM = "free_form"       # Planned: No identity constraint
    LLM_GUIDED = "llm_guided"    # Planned: LLM detects and substitutes
```

### Planned Generator Interface

```python
class CounterfactualGenerator(ABC):
    """Abstract interface for all CF generation backends."""

    @abstractmethod
    def generate(
        self,
        text: str,
        source_group: Optional[str] = None,
        target_group: Optional[str] = None,
        mode: CounterfactualMode = CounterfactualMode.FIXED16,
    ) -> CounterfactualPair:
        """Generate a counterfactual text.

        Args:
            text: Original text.
            source_group: Source identity (required for FIXED16).
            target_group: Target identity (required for FIXED16).
            mode: Generation mode.

        Returns:
            CounterfactualPair with generated text.
        """
        pass

    @abstractmethod
    def batch_generate(
        self,
        texts: List[str],
        mode: CounterfactualMode = CounterfactualMode.FIXED16,
        show_progress: bool = True,
    ) -> List[CounterfactualPair]:
        """Batch generate counterfactuals."""
        pass


class QwenGenerator(CounterfactualGenerator):
    """Local Qwen-based generator (exp2)."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        backend: str = "huggingface",
        quantization: Optional[str] = None,  # "awq", "gptq"
        **kwargs,
    ):
        ...


class ZhipuGenerator(CounterfactualGenerator):
    """Zhipu GLM API generator (current main)."""

    def __init__(
        self,
        api_keys: List[str],
        model: str = "glm-4-flash",
        max_workers: int = 20,
        **kwargs,
    ):
        ...
```

## Semantic Validity Gate

```python
class SemanticGate(ABC):
    """Filter for counterfactual quality."""

    @abstractmethod
    def filter_pair(self, pair: CounterfactualPair) -> bool:
        """Return True if pair passes semantic validity checks."""
        pass


class DualStageGate(SemanticGate):
    """Two-stage filtering: LLM Judge + Toxicity Drift."""

    def __init__(
        self,
        judge_model: str = "Qwen/Qwen2.5-7B-Instruct",
        toxicity_threshold: float = 0.15,
        classifier_checkpoint: Optional[str] = None,
    ):
        self.judge = LLMJudge(judge_model)
        self.drift_detector = ToxicityDriftDetector(classifier_checkpoint)
        self.toxicity_threshold = toxicity_threshold

    def filter_pair(self, pair: CounterfactualPair) -> bool:
        # Stage 1: LLM Judge
        if not self.judge.is_valid(pair):
            return False
        # Stage 2: Toxicity drift
        if not self.drift_detector.is_valid(pair):
            return False
        return True
```

## Baseline Hyperparameter Tuning

```python
# Planned: Grid search for LogitPairing
LAMBDA_LP_GRID = [0.5, 1.0, 2.0, 3.0]
SEEDS = [42, 123, 2024]

# Planned: Auto-detect best baseline params from original papers
BASELINE_PAPER_PARAMS = {
    "EAR": {"lambda_ear": 0.1},
    "GetFair": {"lambda_gf": 0.5},
    "CCDF": {"lambda_kl": 1.0, "tde_alpha": 0.5},
    "AdvDebias": {"lambda_adv": 0.1},
}
```

## Model Architecture Variants

```python
class CounterfactualClassifier(ABC):
    """Base interface for all classifier variants."""

    @abstractmethod
    def forward(self, input_ids, attention_mask, return_features=True):
        pass


class DebertaV3CausalFair(CounterfactualClassifier):
    """Current canonical model (see classifier.py)."""
    ...


# Future: configurable projector dim
class DebertaV3CausalFairV2(CounterfactualClassifier):
    """V2 with configurable projection dim and attention pooling."""
    pass
```