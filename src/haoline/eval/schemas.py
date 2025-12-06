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
    metrics: list[EvalMetric] = Field(
        default_factory=list, description="List of evaluation metrics"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Tool-specific extras")

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
        result: str = self.model_dump_json(indent=indent)
        return result

    @classmethod
    def from_json(cls, json_str: str) -> EvalResult:
        """Deserialize from JSON string."""
        result: EvalResult = cls.model_validate_json(json_str)
        return result


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
                EvalMetric(
                    name="accuracy",
                    value=accuracy,
                    unit="%",
                    higher_is_better=True,
                    category="accuracy",
                )
            )
        if f1 is not None:
            metrics.append(
                EvalMetric(
                    name="f1", value=f1, unit="%", higher_is_better=True, category="accuracy"
                )
            )
        if exact_match is not None:
            metrics.append(
                EvalMetric(
                    name="exact_match",
                    value=exact_match,
                    unit="%",
                    higher_is_better=True,
                    category="accuracy",
                )
            )
        if bleu is not None:
            metrics.append(
                EvalMetric(
                    name="bleu", value=bleu, unit="", higher_is_better=True, category="accuracy"
                )
            )
        if rouge_l is not None:
            metrics.append(
                EvalMetric(
                    name="rouge_l",
                    value=rouge_l,
                    unit="",
                    higher_is_better=True,
                    category="accuracy",
                )
            )

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
                    name="perplexity",
                    value=perplexity,
                    unit="",
                    higher_is_better=False,
                    category="accuracy",
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
                metrics.append(
                    EvalMetric(
                        name=name, value=value, unit="%", higher_is_better=True, category="accuracy"
                    )
                )
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
            EvalMetric(
                name="mIoU", value=miou, unit="%", higher_is_better=True, category="accuracy"
            ),
        ]
        if dice is not None:
            metrics.append(
                EvalMetric(
                    name="dice", value=dice, unit="%", higher_is_better=True, category="accuracy"
                )
            )

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
                    unit="",
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
    model_path: Annotated[str, Field(default="", description="Path to model file")] = ""

    # Architecture analysis (from haoline inspect)
    architecture: dict[str, Any] = Field(
        default_factory=dict,
        description="Architecture summary: params_total, flops_total, etc.",
    )

    # Evaluation results (from external tools)
    eval_results: list[EvalResult] = Field(
        default_factory=list, description="Evaluation results from external tools"
    )

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
        result: str = self.model_dump_json(indent=indent)
        return result

    @classmethod
    def from_inspection_report(
        cls,
        report: Any,  # InspectionReport
        model_path: str = "",
        eval_results: list[EvalResult] | None = None,
    ) -> CombinedReport:
        """
        Create from an InspectionReport.

        Args:
            report: InspectionReport from haoline.
            model_path: Path to the model file.
            eval_results: Optional list of eval results to attach.
        """
        from pathlib import Path

        # Extract key architecture metrics
        mem_bytes = 0
        if report.memory_estimates:
            mem_bytes = (
                report.memory_estimates.model_size_bytes
                + report.memory_estimates.peak_activation_bytes
            )

        arch_summary = {
            "params_total": (report.param_counts.total if report.param_counts else 0),
            "flops_total": (report.flop_counts.total if report.flop_counts else 0),
            "memory_bytes": mem_bytes,
            "model_size_bytes": (
                report.memory_estimates.model_size_bytes if report.memory_estimates else 0
            ),
            "peak_activation_bytes": (
                report.memory_estimates.peak_activation_bytes if report.memory_estimates else 0
            ),
            "architecture_type": report.architecture_type,
            "num_nodes": (report.graph_summary.num_nodes if report.graph_summary else 0),
        }

        # Hardware estimates if available
        hw_profile = ""
        latency = 0.0
        throughput = 0.0
        if report.hardware_estimates:
            hw_profile = report.hardware_profile.name if report.hardware_profile else ""
            latency = getattr(report.hardware_estimates, "latency_ms", 0.0)
            throughput = getattr(report.hardware_estimates, "throughput_samples_per_sec", 0.0)

        # Model ID: use filename stem or path
        model_id = ""
        if model_path:
            model_id = Path(model_path).stem
        elif report.metadata:
            model_id = Path(report.metadata.path).stem if report.metadata.path else ""

        # Set primary accuracy from first eval result
        primary_metric = ""
        primary_value = 0.0
        evals = eval_results or []
        if evals and evals[0].metrics:
            # Use first accuracy-type metric as primary
            for m in evals[0].metrics:
                if m.higher_is_better and m.category in ("accuracy", ""):
                    primary_metric = m.name
                    primary_value = m.value
                    break

        return cls(
            model_id=model_id,
            model_path=model_path or (report.metadata.path if report.metadata else ""),
            architecture=arch_summary,
            eval_results=evals,
            primary_accuracy_metric=primary_metric,
            primary_accuracy_value=primary_value,
            hardware_profile=hw_profile,
            latency_ms=latency,
            throughput_fps=throughput,
        )


# =============================================================================
# Model Linking Utilities (Task 12.4.1)
# =============================================================================


def compute_model_hash(model_path: str, algorithm: str = "sha256") -> str:
    """
    Compute a hash of a model file for unique identification.

    Args:
        model_path: Path to the model file.
        algorithm: Hash algorithm ("sha256", "md5", "sha1").

    Returns:
        Hex digest of the file hash.

    Example:
        >>> hash_id = compute_model_hash("model.onnx")
        >>> print(hash_id[:12])  # First 12 chars as short ID
        'a1b2c3d4e5f6'
    """
    import hashlib
    from pathlib import Path

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    hash_func = hashlib.new(algorithm)

    # Read in chunks to handle large files
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_func.update(chunk)

    return hash_func.hexdigest()


def link_eval_to_model(
    model_path: str,
    eval_result: EvalResult,
    use_hash: bool = False,
) -> EvalResult:
    """
    Link an evaluation result to a model file.

    Updates the eval_result's model_id to match the model file identifier
    (either path or hash).

    Args:
        model_path: Path to the model file.
        eval_result: EvalResult to link.
        use_hash: If True, use file hash as model_id. If False, use filename.

    Returns:
        Updated EvalResult with linked model_id.

    Example:
        >>> eval_result = parse_ultralytics_val(data)
        >>> linked = link_eval_to_model("yolov8n.onnx", eval_result)
        >>> print(linked.model_id)  # 'yolov8n'
    """
    from pathlib import Path

    if use_hash:
        model_id = compute_model_hash(model_path)[:12]  # Short hash
    else:
        model_id = Path(model_path).stem

    # Update the eval result's model_id
    eval_result.model_id = model_id
    eval_result.metadata["linked_model_path"] = model_path

    return eval_result


def create_combined_report(
    model_path: str,
    eval_results: list[EvalResult] | None = None,
    inspection_report: Any = None,  # InspectionReport
    run_inspection: bool = True,
) -> CombinedReport:
    """
    Create a CombinedReport by linking model analysis with eval results.

    If inspection_report is not provided and run_inspection is True,
    runs haoline analysis on the model first.

    Args:
        model_path: Path to the model file.
        eval_results: List of evaluation results to attach.
        inspection_report: Pre-computed InspectionReport (optional).
        run_inspection: Whether to run inspection if not provided.

    Returns:
        CombinedReport combining architecture analysis and eval metrics.

    Example:
        >>> # Import eval, then combine with architecture analysis
        >>> eval_result = load_ultralytics_json("val_results.json")
        >>> combined = create_combined_report("yolov8n.onnx", [eval_result])
        >>> print(combined.architecture["params_total"])
        >>> print(combined.eval_results[0].metrics[0].value)
    """
    from pathlib import Path

    # Run inspection if needed
    if inspection_report is None and run_inspection:
        try:
            from haoline.report import ModelInspector

            inspector = ModelInspector()
            inspection_report = inspector.inspect(Path(model_path))
        except Exception as e:
            # Can't import or run haoline - create minimal combined report
            print(f"Warning: Could not run model inspection: {e}")
            return CombinedReport(
                model_id=Path(model_path).stem,
                model_path=model_path,
                architecture={},
                eval_results=eval_results or [],
            )

    # Link eval results to model
    linked_evals: list[EvalResult] = []
    if eval_results:
        for er in eval_results:
            linked = link_eval_to_model(model_path, er)
            linked_evals.append(linked)

    # Create combined report
    if inspection_report:
        return CombinedReport.from_inspection_report(
            inspection_report,
            model_path=model_path,
            eval_results=linked_evals,
        )
    else:
        return CombinedReport(
            model_id=Path(model_path).stem,
            model_path=model_path,
            architecture={},
            eval_results=linked_evals,
        )


# =============================================================================
# Schema Generation and Validation
# =============================================================================


def get_eval_schema() -> dict[str, Any]:
    """Get JSON Schema for EvalResult."""
    schema: dict[str, Any] = EvalResult.model_json_schema()
    return schema


def get_combined_report_schema() -> dict[str, Any]:
    """Get JSON Schema for CombinedReport."""
    schema: dict[str, Any] = CombinedReport.model_json_schema()
    return schema


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
