"""
Eval Result Schemas (Pydantic v2)

Task-agnostic and task-specific schemas for importing evaluation results
from external tools like Ultralytics, HuggingFace evaluate, lm-eval, etc.

All schemas use Pydantic for validation, serialization, and JSON Schema generation.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """Supported evaluation task types."""

    detection = "detection"
    classification = "classification"
    nlp = "nlp"
    llm = "llm"
    segmentation = "segmentation"
    generic = "generic"


class EvalMetric(BaseModel):
    """A single evaluation metric."""

    name: Annotated[str, Field(description="Metric name, e.g., 'mAP@50', 'top1_accuracy'")]
    value: Annotated[float, Field(description="The metric value")]
    unit: Annotated[str, Field(default="", description="Unit, e.g., '%', 'ms', '' (dimensionless)")]
    higher_is_better: Annotated[
        bool, Field(default=True, description="Whether higher values are better")
    ]
    category: Annotated[
        str, Field(default="", description="Category, e.g., 'accuracy', 'speed', 'size'")
    ]


class EvalResult(BaseModel):
    """
    Base class for evaluation results.

    Task-agnostic fields that all eval results share.
    """

    model_id: Annotated[str, Field(description="Identifier for the model (path, name, or hash)")]
    task_type: Annotated[str, Field(description="Task type: detection, classification, etc.")]
    timestamp: Annotated[str, Field(default="", description="ISO format timestamp of eval run")] = (
        ""
    )
    dataset: Annotated[str, Field(default="", description="Dataset used for evaluation")] = ""
    metrics: Annotated[
        list[EvalMetric], Field(default_factory=list, description="List of evaluation metrics")
    ]
    metadata: Annotated[
        dict[str, Any], Field(default_factory=dict, description="Tool-specific extras")
    ]

    def model_post_init(self, __context: Any) -> None:
        """Set timestamp if not provided."""
        if not self.timestamp:
            object.__setattr__(self, "timestamp", datetime.now().isoformat())

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

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> EvalResult:
        """Deserialize from JSON string."""
        return cls.model_validate_json(json_str)


# =============================================================================
# Task-Specific Schemas
# =============================================================================


class DetectionEvalResult(EvalResult):
    """
    Object detection evaluation results.

    Standard metrics: mAP@50, mAP@50:95, precision, recall, F1 per class.
    Compatible with: Ultralytics YOLO, Detectron2, MMDetection
    """

    task_type: str = "detection"

    # Per-class metrics
    class_metrics: Annotated[
        dict[str, dict[str, float]],
        Field(
            default_factory=dict,
            description="Per-class metrics, e.g., {'person': {'precision': 0.92}}",
        ),
    ]

    # IoU thresholds used
    iou_thresholds: Annotated[
        list[float], Field(default_factory=lambda: [0.5, 0.75], description="IoU thresholds")
    ]

    # Confidence threshold
    confidence_threshold: Annotated[float, Field(default=0.5, description="Confidence threshold")]

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
        **kwargs: Any,
    ) -> DetectionEvalResult:
        """Convenience constructor with standard detection metrics."""
        metrics = [
            EvalMetric(
                name="mAP@50", value=map50, unit="%", higher_is_better=True, category="accuracy"
            ),
            EvalMetric(
                name="mAP@50:95",
                value=map50_95,
                unit="%",
                higher_is_better=True,
                category="accuracy",
            ),
            EvalMetric(
                name="precision",
                value=precision,
                unit="%",
                higher_is_better=True,
                category="accuracy",
            ),
            EvalMetric(
                name="recall", value=recall, unit="%", higher_is_better=True, category="accuracy"
            ),
            EvalMetric(name="f1", value=f1, unit="%", higher_is_better=True, category="accuracy"),
        ]
        return cls(
            model_id=model_id,
            dataset=dataset,
            metrics=metrics,
            class_metrics=class_metrics or {},
            **kwargs,
        )


class ClassificationEvalResult(EvalResult):
    """
    Image/text classification evaluation results.

    Standard metrics: top-1 accuracy, top-5 accuracy, per-class accuracy.
    Compatible with: timm, torchvision, HuggingFace
    """

    task_type: str = "classification"

    # Per-class accuracy
    class_accuracy: Annotated[
        dict[str, float],
        Field(default_factory=dict, description="Per-class accuracy"),
    ]

    # Confusion matrix (optional)
    confusion_matrix: Annotated[
        list[list[int]] | None,
        Field(default=None, description="Confusion matrix"),
    ]
    class_names: Annotated[
        list[str], Field(default_factory=list, description="Class names for confusion matrix")
    ]

    @classmethod
    def create(
        cls,
        model_id: str,
        dataset: str,
        top1_accuracy: float,
        top5_accuracy: float,
        class_accuracy: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> ClassificationEvalResult:
        """Convenience constructor with standard classification metrics."""
        metrics = [
            EvalMetric(
                name="top1_accuracy",
                value=top1_accuracy,
                unit="%",
                higher_is_better=True,
                category="accuracy",
            ),
            EvalMetric(
                name="top5_accuracy",
                value=top5_accuracy,
                unit="%",
                higher_is_better=True,
                category="accuracy",
            ),
        ]
        return cls(
            model_id=model_id,
            dataset=dataset,
            metrics=metrics,
            class_accuracy=class_accuracy or {},
            **kwargs,
        )


class NLPEvalResult(EvalResult):
    """
    NLP task evaluation results.

    Standard metrics: accuracy, F1, exact match, BLEU, ROUGE.
    Compatible with: HuggingFace evaluate, SacreBLEU
    """

    task_type: str = "nlp"

    # Task subtype
    nlp_task: Annotated[
        str,
        Field(
            default="",
            description="NLP task: classification, ner, qa, translation, summarization",
        ),
    ] = ""

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
        **kwargs: Any,
    ) -> NLPEvalResult:
        """Convenience constructor with standard NLP metrics."""
        metrics = []
        if accuracy is not None:
            metrics.append(
                EvalMetric(name="accuracy", value=accuracy, unit="%", category="accuracy")
            )
        if f1 is not None:
            metrics.append(EvalMetric(name="f1", value=f1, unit="%", category="accuracy"))
        if exact_match is not None:
            metrics.append(
                EvalMetric(name="exact_match", value=exact_match, unit="%", category="accuracy")
            )
        if bleu is not None:
            metrics.append(EvalMetric(name="bleu", value=bleu, category="accuracy"))
        if rouge_l is not None:
            metrics.append(EvalMetric(name="rouge_l", value=rouge_l, category="accuracy"))

        return cls(
            model_id=model_id,
            dataset=dataset,
            metrics=metrics,
            nlp_task=nlp_task,
            **kwargs,
        )


class LLMEvalResult(EvalResult):
    """
    Large Language Model evaluation results.

    Standard metrics: perplexity, MMLU, HellaSwag, TruthfulQA, etc.
    Compatible with: lm-eval-harness, EleutherAI eval
    """

    task_type: str = "llm"

    # Benchmark scores (0-100 or 0-1 depending on benchmark)
    benchmark_scores: Annotated[
        dict[str, float],
        Field(
            default_factory=dict,
            description="Benchmark scores, e.g., {'mmlu': 0.72, 'hellaswag': 0.81}",
        ),
    ]

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
        **kwargs: Any,
    ) -> LLMEvalResult:
        """Convenience constructor with standard LLM benchmarks."""
        metrics = []
        benchmark_scores = {}

        if perplexity is not None:
            metrics.append(
                EvalMetric(
                    name="perplexity", value=perplexity, higher_is_better=False, category="accuracy"
                )
            )

        benchmarks = {
            "mmlu": mmlu,
            "hellaswag": hellaswag,
            "truthfulqa": truthfulqa,
            "arc_challenge": arc_challenge,
            "winogrande": winogrande,
        }

        for name, value in benchmarks.items():
            if value is not None:
                metrics.append(EvalMetric(name=name, value=value, unit="%", category="accuracy"))
                benchmark_scores[name] = value

        return cls(
            model_id=model_id,
            dataset="multiple",
            metrics=metrics,
            benchmark_scores=benchmark_scores,
            **kwargs,
        )


class SegmentationEvalResult(EvalResult):
    """
    Semantic/instance segmentation evaluation results.

    Standard metrics: mIoU, dice coefficient, per-class IoU.
    Compatible with: MMSegmentation, Detectron2
    """

    task_type: str = "segmentation"

    # Per-class IoU
    class_iou: Annotated[
        dict[str, float],
        Field(default_factory=dict, description="Per-class IoU values"),
    ]

    # Segmentation type
    segmentation_type: Annotated[
        str,
        Field(default="semantic", description="Type: semantic, instance, or panoptic"),
    ] = "semantic"

    @classmethod
    def create(
        cls,
        model_id: str,
        dataset: str,
        miou: float,
        dice: float | None = None,
        class_iou: dict[str, float] | None = None,
        segmentation_type: str = "semantic",
        **kwargs: Any,
    ) -> SegmentationEvalResult:
        """Convenience constructor with standard segmentation metrics."""
        metrics = [
            EvalMetric(name="mIoU", value=miou, unit="%", category="accuracy"),
        ]
        if dice is not None:
            metrics.append(EvalMetric(name="dice", value=dice, unit="%", category="accuracy"))

        return cls(
            model_id=model_id,
            dataset=dataset,
            metrics=metrics,
            class_iou=class_iou or {},
            segmentation_type=segmentation_type,
            **kwargs,
        )


class GenericEvalResult(EvalResult):
    """
    Generic evaluation results with user-defined metrics.

    Use this when no task-specific schema fits, or for custom evaluation tasks.
    """

    task_type: str = "generic"

    # User can specify what metrics mean
    metric_definitions: Annotated[
        dict[str, str],
        Field(
            default_factory=dict,
            description="Metric definitions, e.g., {'custom_score': 'Higher is better'}",
        ),
    ]

    @classmethod
    def create(
        cls,
        model_id: str,
        dataset: str = "",
        metrics: dict[str, float] | None = None,
        metric_definitions: dict[str, str] | None = None,
        higher_is_better: dict[str, bool] | None = None,
        **kwargs: Any,
    ) -> GenericEvalResult:
        """Convenience constructor for generic metrics."""
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


class CombinedReport(BaseModel):
    """
    Combines architecture analysis with evaluation results.

    Links an InspectionReport (model structure, FLOPs, params) with
    EvalResult (accuracy, speed benchmarks) for unified comparison.
    """

    model_id: Annotated[str, Field(description="Model identifier")]
    model_path: Annotated[str, Field(default="", description="Path to model file")]

    # Architecture analysis (from haoline inspect)
    architecture: Annotated[
        dict[str, Any],
        Field(
            default_factory=dict,
            description="Architecture summary: params_total, flops_total, etc.",
        ),
    ]

    # Evaluation results (from external tools)
    eval_results: Annotated[
        list[EvalResult],
        Field(default_factory=list, description="Evaluation results from external tools"),
    ]

    # Computed summaries
    primary_accuracy_metric: Annotated[
        str, Field(default="", description="Primary accuracy metric name")
    ] = ""
    primary_accuracy_value: Annotated[
        float, Field(default=0.0, description="Primary accuracy metric value")
    ] = 0.0

    # Hardware estimates (from haoline)
    hardware_profile: Annotated[str, Field(default="", description="Hardware profile name")] = ""
    latency_ms: Annotated[float, Field(default=0.0, description="Latency in milliseconds")] = 0.0
    throughput_fps: Annotated[
        float, Field(default=0.0, description="Throughput in frames per second")
    ] = 0.0

    # Deployment cost (if calculated)
    cost_per_day_usd: Annotated[
        float, Field(default=0.0, description="Estimated cost per day in USD")
    ] = 0.0
    cost_per_month_usd: Annotated[
        float, Field(default=0.0, description="Estimated cost per month in USD")
    ] = 0.0

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

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_inspection_report(
        cls,
        report: Any,  # InspectionReport
        model_path: str = "",
    ) -> CombinedReport:
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
# Schema Generation and Validation
# =============================================================================


def get_eval_schema() -> dict[str, Any]:
    """Get JSON Schema for EvalResult."""
    return EvalResult.model_json_schema()


def get_combined_report_schema() -> dict[str, Any]:
    """Get JSON Schema for CombinedReport."""
    return CombinedReport.model_json_schema()


def validate_eval_result(data: dict[str, Any]) -> bool:
    """
    Validate eval result data using Pydantic.

    Returns True if valid, False otherwise.
    """
    try:
        EvalResult.model_validate(data)
        return True
    except Exception:
        return False


def is_valid_task_type(task_type: str) -> bool:
    """Check if a task type is valid."""
    return task_type in [t.value for t in TaskType]
