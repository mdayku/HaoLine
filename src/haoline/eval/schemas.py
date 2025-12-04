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


@dataclass
class GenericEvalResult(EvalResult):
    """
    Generic evaluation results with user-defined metrics.

    Use this when no task-specific schema fits, or for custom evaluation tasks.
    The user provides metric definitions explicitly.
    """

    task_type: str = "generic"

    # User can specify what metrics mean
    metric_definitions: dict[str, str] = field(default_factory=dict)
    # e.g., {"custom_score": "Higher values indicate better model performance"}

    @classmethod
    def create(
        cls,
        model_id: str,
        dataset: str = "",
        metrics: dict[str, float] | None = None,
        metric_definitions: dict[str, str] | None = None,
        higher_is_better: dict[str, bool] | None = None,
        **kwargs,
    ) -> "GenericEvalResult":
        """
        Convenience constructor for generic metrics.

        Args:
            model_id: Model identifier.
            dataset: Dataset name.
            metrics: Dict of metric_name -> value.
            metric_definitions: Dict of metric_name -> description.
            higher_is_better: Dict of metric_name -> bool (default True).
        """
        metric_list = []
        higher_map = higher_is_better or {}

        for name, value in (metrics or {}).items():
            metric_list.append(
                EvalMetric(
                    name=name,
                    value=value,
                    higher_is_better=higher_map.get(name, True),
                    category="custom",
                )
            )

        return cls(
            model_id=model_id,
            dataset=dataset,
            metrics=metric_list,
            metric_definitions=metric_definitions or {},
            **kwargs,
        )


# =============================================================================
# Combined Report (Architecture + Eval)
# =============================================================================


@dataclass
class CombinedReport:
    """
    Combines architecture analysis with evaluation results.

    Links an InspectionReport (model structure, FLOPs, params) with
    EvalResult (accuracy, speed benchmarks) for unified comparison.
    """

    model_id: str
    model_path: str = ""

    # Architecture analysis (from haoline inspect)
    architecture: dict[str, Any] = field(default_factory=dict)
    # Keys: params_total, flops_total, memory_bytes, architecture_type, etc.

    # Evaluation results (from external tools)
    eval_results: list[EvalResult] = field(default_factory=list)

    # Computed summaries
    primary_accuracy_metric: str = ""  # e.g., "mAP@50" or "top1_accuracy"
    primary_accuracy_value: float = 0.0

    # Hardware estimates (from haoline)
    hardware_profile: str = ""
    latency_ms: float = 0.0
    throughput_fps: float = 0.0

    # Deployment cost (if calculated)
    cost_per_day_usd: float = 0.0
    cost_per_month_usd: float = 0.0

    def add_eval_result(self, result: EvalResult) -> None:
        """Add an evaluation result."""
        self.eval_results.append(result)

    def get_eval_by_task(self, task_type: str) -> EvalResult | None:
        """Get eval result by task type."""
        for r in self.eval_results:
            if r.task_type == task_type:
                return r
        return None

    def get_all_metrics(self) -> list[EvalMetric]:
        """Get all metrics from all eval results."""
        metrics = []
        for r in self.eval_results:
            metrics.extend(r.metrics)
        return metrics

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_id": self.model_id,
            "model_path": self.model_path,
            "architecture": self.architecture,
            "eval_results": [r.to_dict() for r in self.eval_results],
            "primary_accuracy_metric": self.primary_accuracy_metric,
            "primary_accuracy_value": self.primary_accuracy_value,
            "hardware_profile": self.hardware_profile,
            "latency_ms": self.latency_ms,
            "throughput_fps": self.throughput_fps,
            "cost_per_day_usd": self.cost_per_day_usd,
            "cost_per_month_usd": self.cost_per_month_usd,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_inspection_report(
        cls,
        report: Any,  # InspectionReport
        model_path: str = "",
    ) -> "CombinedReport":
        """
        Create from an InspectionReport.

        Args:
            report: InspectionReport from haoline.
            model_path: Path to the model file.
        """
        # Extract key architecture metrics
        arch_summary = {
            "params_total": (report.param_counts.total if report.param_counts else 0),
            "flops_total": (report.flop_counts.total if report.flop_counts else 0),
            "memory_bytes": (report.memory_estimates.total_bytes if report.memory_estimates else 0),
            "architecture_type": report.architecture_type,
            "num_nodes": (report.graph_summary.num_nodes if report.graph_summary else 0),
        }

        # Hardware estimates if available
        hw_profile = ""
        latency = 0.0
        throughput = 0.0
        if report.hardware_estimates:
            hw_profile = report.hardware_profile.name if report.hardware_profile else ""
            latency = report.hardware_estimates.latency_ms
            throughput = report.hardware_estimates.throughput_samples_per_sec

        return cls(
            model_id=report.metadata.name if report.metadata else "",
            model_path=model_path,
            architecture=arch_summary,
            hardware_profile=hw_profile,
            latency_ms=latency,
            throughput_fps=throughput,
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
            "enum": ["detection", "classification", "nlp", "llm", "segmentation", "generic"],
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
        "generic",
    ]:
        return False
    return True
