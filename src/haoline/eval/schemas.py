"""
Eval Result Schemas

Task-agnostic and task-specific schemas for importing evaluation results
from external tools like Ultralytics, HuggingFace evaluate, lm-eval, etc.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class EvalMetric:
    """A single evaluation metric."""

    name: str  # e.g., "mAP@50", "top1_accuracy", "f1_macro"
    value: float  # The metric value
    unit: str = ""  # e.g., "%", "ms", "" (dimensionless)
    higher_is_better: bool = True  # For ranking/comparison
    category: str = ""  # e.g., "accuracy", "speed", "size"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "higher_is_better": self.higher_is_better,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalMetric":
        return cls(
            name=data["name"],
            value=data["value"],
            unit=data.get("unit", ""),
            higher_is_better=data.get("higher_is_better", True),
            category=data.get("category", ""),
        )


@dataclass
class EvalResult:
    """
    Base class for evaluation results.

    Task-agnostic fields that all eval results share.
    """

    model_id: str  # Identifier for the model (path, name, or hash)
    task_type: str  # "detection", "classification", "nlp", "llm", "segmentation"
    timestamp: str = ""  # ISO format timestamp of eval run
    dataset: str = ""  # Dataset used for evaluation (e.g., "coco_val2017")
    metrics: list[EvalMetric] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)  # Tool-specific extras

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def get_metric(self, name: str) -> EvalMetric | None:
        """Get a metric by name."""
        for m in self.metrics:
            if m.name == name:
                return m
        return None

    def get_metric_value(self, name: str, default: float = 0.0) -> float:
        """Get a metric value by name, with default."""
        m = self.get_metric(name)
        return m.value if m else default

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "task_type": self.task_type,
            "timestamp": self.timestamp,
            "dataset": self.dataset,
            "metrics": [m.to_dict() for m in self.metrics],
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalResult":
        return cls(
            model_id=data["model_id"],
            task_type=data["task_type"],
            timestamp=data.get("timestamp", ""),
            dataset=data.get("dataset", ""),
            metrics=[EvalMetric.from_dict(m) for m in data.get("metrics", [])],
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "EvalResult":
        return cls.from_dict(json.loads(json_str))


# =============================================================================
# Task-Specific Schemas
# =============================================================================


@dataclass
class DetectionEvalResult(EvalResult):
    """
    Object detection evaluation results.

    Standard metrics: mAP@50, mAP@50:95, precision, recall, F1 per class.
    Compatible with: Ultralytics YOLO, Detectron2, MMDetection
    """

    task_type: str = "detection"

    # Per-class metrics
    class_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    # e.g., {"person": {"precision": 0.92, "recall": 0.88, "f1": 0.90, "ap50": 0.91}}

    # IoU thresholds used
    iou_thresholds: list[float] = field(default_factory=lambda: [0.5, 0.75])

    # Confidence threshold
    confidence_threshold: float = 0.5

    @classmethod
    def create(
        cls,
        model_id: str,
        dataset: str,
        map50: float,
        map50_95: float,
        precision: float,
        recall: float,
        f1: float,
        class_metrics: dict[str, dict[str, float]] | None = None,
        **kwargs,
    ) -> "DetectionEvalResult":
        """Convenience constructor with standard detection metrics."""
        metrics = [
            EvalMetric("mAP@50", map50, "%", True, "accuracy"),
            EvalMetric("mAP@50:95", map50_95, "%", True, "accuracy"),
            EvalMetric("precision", precision, "%", True, "accuracy"),
            EvalMetric("recall", recall, "%", True, "accuracy"),
            EvalMetric("f1", f1, "%", True, "accuracy"),
        ]
        return cls(
            model_id=model_id,
            dataset=dataset,
            metrics=metrics,
            class_metrics=class_metrics or {},
            **kwargs,
        )


@dataclass
class ClassificationEvalResult(EvalResult):
    """
    Image/text classification evaluation results.

    Standard metrics: top-1 accuracy, top-5 accuracy, per-class accuracy.
    Compatible with: timm, torchvision, HuggingFace
    """

    task_type: str = "classification"

    # Per-class accuracy
    class_accuracy: dict[str, float] = field(default_factory=dict)

    # Confusion matrix (optional)
    confusion_matrix: list[list[int]] | None = None
    class_names: list[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        model_id: str,
        dataset: str,
        top1_accuracy: float,
        top5_accuracy: float,
        class_accuracy: dict[str, float] | None = None,
        **kwargs,
    ) -> "ClassificationEvalResult":
        """Convenience constructor with standard classification metrics."""
        metrics = [
            EvalMetric("top1_accuracy", top1_accuracy, "%", True, "accuracy"),
            EvalMetric("top5_accuracy", top5_accuracy, "%", True, "accuracy"),
        ]
        return cls(
            model_id=model_id,
            dataset=dataset,
            metrics=metrics,
            class_accuracy=class_accuracy or {},
            **kwargs,
        )


@dataclass
class NLPEvalResult(EvalResult):
    """
    NLP task evaluation results.

    Standard metrics: accuracy, F1, exact match, BLEU, ROUGE.
    Compatible with: HuggingFace evaluate, SacreBLEU
    """

    task_type: str = "nlp"

    # Task subtype
    nlp_task: str = ""  # "classification", "ner", "qa", "translation", "summarization"

    @classmethod
    def create(
        cls,
        model_id: str,
        dataset: str,
        nlp_task: str,
        accuracy: float | None = None,
        f1: float | None = None,
        exact_match: float | None = None,
        bleu: float | None = None,
        rouge_l: float | None = None,
        **kwargs,
    ) -> "NLPEvalResult":
        """Convenience constructor with standard NLP metrics."""
        metrics = []
        if accuracy is not None:
            metrics.append(EvalMetric("accuracy", accuracy, "%", True, "accuracy"))
        if f1 is not None:
            metrics.append(EvalMetric("f1", f1, "%", True, "accuracy"))
        if exact_match is not None:
            metrics.append(EvalMetric("exact_match", exact_match, "%", True, "accuracy"))
        if bleu is not None:
            metrics.append(EvalMetric("bleu", bleu, "", True, "accuracy"))
        if rouge_l is not None:
            metrics.append(EvalMetric("rouge_l", rouge_l, "", True, "accuracy"))

        return cls(
            model_id=model_id,
            dataset=dataset,
            metrics=metrics,
            nlp_task=nlp_task,
            **kwargs,
        )


@dataclass
class LLMEvalResult(EvalResult):
    """
    Large Language Model evaluation results.

    Standard metrics: perplexity, MMLU, HellaSwag, TruthfulQA, etc.
    Compatible with: lm-eval-harness, EleutherAI eval
    """

    task_type: str = "llm"

    # Benchmark scores (0-100 or 0-1 depending on benchmark)
    benchmark_scores: dict[str, float] = field(default_factory=dict)
    # e.g., {"mmlu": 0.72, "hellaswag": 0.81, "truthfulqa": 0.45}

    @classmethod
    def create(
        cls,
        model_id: str,
        perplexity: float | None = None,
        mmlu: float | None = None,
        hellaswag: float | None = None,
        truthfulqa: float | None = None,
        arc_challenge: float | None = None,
        winogrande: float | None = None,
        **kwargs,
    ) -> "LLMEvalResult":
        """Convenience constructor with standard LLM benchmarks."""
        metrics = []
        benchmark_scores = {}

        if perplexity is not None:
            metrics.append(
                EvalMetric("perplexity", perplexity, "", False, "accuracy")
            )  # Lower is better

        benchmarks = {
            "mmlu": mmlu,
            "hellaswag": hellaswag,
            "truthfulqa": truthfulqa,
            "arc_challenge": arc_challenge,
            "winogrande": winogrande,
        }

        for name, value in benchmarks.items():
            if value is not None:
                metrics.append(EvalMetric(name, value, "%", True, "accuracy"))
                benchmark_scores[name] = value

        return cls(
            model_id=model_id,
            dataset="multiple",
            metrics=metrics,
            benchmark_scores=benchmark_scores,
            **kwargs,
        )


@dataclass
class SegmentationEvalResult(EvalResult):
    """
    Semantic/instance segmentation evaluation results.

    Standard metrics: mIoU, dice coefficient, per-class IoU.
    Compatible with: MMSegmentation, Detectron2
    """

    task_type: str = "segmentation"

    # Per-class IoU
    class_iou: dict[str, float] = field(default_factory=dict)

    # Segmentation type
    segmentation_type: str = "semantic"  # "semantic", "instance", "panoptic"

    @classmethod
    def create(
        cls,
        model_id: str,
        dataset: str,
        miou: float,
        dice: float | None = None,
        class_iou: dict[str, float] | None = None,
        segmentation_type: str = "semantic",
        **kwargs,
    ) -> "SegmentationEvalResult":
        """Convenience constructor with standard segmentation metrics."""
        metrics = [
            EvalMetric("mIoU", miou, "%", True, "accuracy"),
        ]
        if dice is not None:
            metrics.append(EvalMetric("dice", dice, "%", True, "accuracy"))

        return cls(
            model_id=model_id,
            dataset=dataset,
            metrics=metrics,
            class_iou=class_iou or {},
            segmentation_type=segmentation_type,
            **kwargs,
        )


# =============================================================================
# JSON Schema for Validation
# =============================================================================

EVAL_RESULT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "HaoLine Eval Result",
    "type": "object",
    "required": ["model_id", "task_type"],
    "properties": {
        "model_id": {"type": "string", "description": "Model identifier"},
        "task_type": {
            "type": "string",
            "enum": ["detection", "classification", "nlp", "llm", "segmentation"],
        },
        "timestamp": {"type": "string", "format": "date-time"},
        "dataset": {"type": "string"},
        "metrics": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "value"],
                "properties": {
                    "name": {"type": "string"},
                    "value": {"type": "number"},
                    "unit": {"type": "string"},
                    "higher_is_better": {"type": "boolean"},
                    "category": {"type": "string"},
                },
            },
        },
        "metadata": {"type": "object"},
    },
}


def validate_eval_result(data: dict[str, Any]) -> bool:
    """
    Basic validation of eval result data.

    For full JSON Schema validation, use jsonschema library.
    """
    if "model_id" not in data:
        return False
    if "task_type" not in data:
        return False
    if data["task_type"] not in [
        "detection",
        "classification",
        "nlp",
        "llm",
        "segmentation",
    ]:
        return False
    return True
