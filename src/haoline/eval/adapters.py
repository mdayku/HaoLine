"""
Eval Adapters

Parse evaluation results from external tools into HaoLine's schema.

Supported adapters:
- Ultralytics YOLO (detection): parse_ultralytics_val, load_ultralytics_json
- HuggingFace evaluate (classification/NLP): parse_hf_evaluate, load_hf_evaluate
- lm-eval-harness (LLM benchmarks): parse_lm_eval, load_lm_eval
- timm (image classification): parse_timm_benchmark, load_timm_benchmark
- Generic CSV/JSON: parse_generic_json, parse_generic_csv, load_generic_json, load_generic_csv

Auto-detection: detect_and_parse() tries to identify the format automatically.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .schemas import (
    ClassificationEvalResult,
    DetectionEvalResult,
    EvalMetric,
    EvalResult,
    GenericEvalResult,
    LLMEvalResult,
    NLPEvalResult,
)

# =============================================================================
# Ultralytics YOLO Adapter (Task 12.3.1)
# =============================================================================


def parse_ultralytics_val(
    data: dict[str, Any],
    model_id: str = "",
) -> DetectionEvalResult:
    """
    Parse Ultralytics YOLO validation results.

    Ultralytics outputs validation metrics in various formats. This parser
    handles the JSON output from `yolo val` or results from `model.val()`.

    Expected fields (from results.results_dict or JSON):
        - metrics/mAP50(B): float
        - metrics/mAP50-95(B): float
        - metrics/precision(B): float
        - metrics/recall(B): float
        - fitness: float (optional)

    Args:
        data: Dictionary from YOLO validation output.
        model_id: Model identifier (defaults to extracting from data).

    Returns:
        DetectionEvalResult with parsed metrics.
    """

    # Try different key formats (Ultralytics uses inconsistent naming)
    def get_metric(keys: list[str], default: float = 0.0) -> float:
        for key in keys:
            if key in data:
                val = data[key]
                return float(val) if val is not None else default
            # Check nested metrics dict
            if "metrics" in data and key in data["metrics"]:
                val = data["metrics"][key]
                return float(val) if val is not None else default
        return default

    # Extract metrics with various key formats
    map50 = get_metric(
        [
            "metrics/mAP50(B)",
            "mAP50",
            "map50",
            "mAP@50",
            "box/mAP50",
        ]
    )
    map50_95 = get_metric(
        [
            "metrics/mAP50-95(B)",
            "mAP50-95",
            "map50_95",
            "mAP@50:95",
            "box/mAP50-95",
            "map",
        ]
    )
    precision = get_metric(
        [
            "metrics/precision(B)",
            "precision",
            "box/precision",
            "p",
        ]
    )
    recall = get_metric(
        [
            "metrics/recall(B)",
            "recall",
            "box/recall",
            "r",
        ]
    )

    # Calculate F1 if not provided
    f1 = get_metric(["f1", "box/f1"])
    if f1 == 0.0 and precision > 0 and recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)

    # Extract model ID
    if not model_id:
        model_id = data.get("model", data.get("name", "unknown"))

    # Extract dataset
    dataset = data.get("data", data.get("dataset", ""))
    if isinstance(dataset, dict):
        dataset = dataset.get("path", dataset.get("name", ""))

    # Per-class metrics if available
    class_metrics: dict[str, dict[str, float]] = {}
    if "per_class" in data:
        for cls_name, cls_data in data["per_class"].items():
            class_metrics[cls_name] = {
                "precision": cls_data.get("precision", 0.0),
                "recall": cls_data.get("recall", 0.0),
                "ap50": cls_data.get("ap50", cls_data.get("mAP50", 0.0)),
            }

    # Build the result
    result = DetectionEvalResult.create(
        model_id=str(model_id),
        dataset=str(dataset),
        map50=map50,
        map50_95=map50_95,
        precision=precision,
        recall=recall,
        f1=f1,
        class_metrics=class_metrics,
    )

    # Add extra metrics from metadata
    speed = data.get("speed", {})
    if speed:
        if "inference" in speed:
            result.metrics.append(
                EvalMetric(
                    name="inference_ms",
                    value=speed["inference"],
                    unit="ms",
                    higher_is_better=False,
                    category="speed",
                )
            )
        if "preprocess" in speed:
            result.metrics.append(
                EvalMetric(
                    name="preprocess_ms",
                    value=speed["preprocess"],
                    unit="ms",
                    higher_is_better=False,
                    category="speed",
                )
            )
        if "postprocess" in speed:
            result.metrics.append(
                EvalMetric(
                    name="postprocess_ms",
                    value=speed["postprocess"],
                    unit="ms",
                    higher_is_better=False,
                    category="speed",
                )
            )

    # Store raw data in metadata
    result.metadata["raw_ultralytics"] = data

    return result


def load_ultralytics_json(path: Path, model_id: str = "") -> DetectionEvalResult:
    """
    Load Ultralytics validation results from JSON file.

    Args:
        path: Path to JSON file.
        model_id: Optional model identifier.

    Returns:
        DetectionEvalResult with parsed metrics.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return parse_ultralytics_val(data, model_id)


# =============================================================================
# Generic CSV/JSON Adapter (Task 12.3.5)
# =============================================================================


def parse_generic_json(
    data: dict[str, Any],
    model_id: str = "",
    metric_mapping: dict[str, str] | None = None,
    higher_is_better: dict[str, bool] | None = None,
) -> GenericEvalResult:
    """
    Parse generic JSON evaluation results.

    Extracts numeric fields as metrics. User can provide mapping to rename fields.

    Args:
        data: Dictionary with metric values.
        model_id: Model identifier.
        metric_mapping: Optional dict to rename fields (json_key -> metric_name).
        higher_is_better: Optional dict specifying direction (metric_name -> bool).

    Returns:
        GenericEvalResult with extracted metrics.

    Example:
        >>> data = {"acc": 0.95, "loss": 0.12, "model": "resnet50"}
        >>> result = parse_generic_json(
        ...     data,
        ...     metric_mapping={"acc": "accuracy", "loss": "val_loss"},
        ...     higher_is_better={"accuracy": True, "val_loss": False}
        ... )
    """
    mapping = metric_mapping or {}
    better_map = higher_is_better or {}

    # Extract model_id from data if not provided
    if not model_id:
        model_id = str(data.get("model_id", data.get("model", data.get("name", "unknown"))))

    # Extract dataset
    dataset = str(data.get("dataset", data.get("data", "")))

    # Find all numeric fields
    metrics: dict[str, float] = {}
    for key, value in data.items():
        # Skip non-numeric and metadata fields
        if key in ("model_id", "model", "name", "dataset", "data", "timestamp", "metadata"):
            continue

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            # Apply mapping if provided
            metric_name = mapping.get(key, key)
            metrics[metric_name] = float(value)

    # Build result
    return GenericEvalResult.create(
        model_id=model_id,
        dataset=dataset,
        metrics=metrics,
        higher_is_better=better_map,
    )


def load_generic_json(
    path: Path,
    model_id: str = "",
    metric_mapping: dict[str, str] | None = None,
    higher_is_better: dict[str, bool] | None = None,
) -> GenericEvalResult:
    """
    Load generic evaluation results from JSON file.

    Args:
        path: Path to JSON file.
        model_id: Optional model identifier.
        metric_mapping: Optional dict to rename fields.
        higher_is_better: Optional dict specifying metric direction.

    Returns:
        GenericEvalResult with extracted metrics.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return parse_generic_json(data, model_id, metric_mapping, higher_is_better)


def parse_generic_csv(
    rows: list[dict[str, str]],
    model_id_column: str = "model",
    metric_columns: list[str] | None = None,
    higher_is_better: dict[str, bool] | None = None,
) -> list[GenericEvalResult]:
    """
    Parse generic CSV evaluation results.

    Each row becomes one EvalResult. Specify which columns are metrics.

    Args:
        rows: List of row dicts (from csv.DictReader).
        model_id_column: Column name containing model identifier.
        metric_columns: List of column names to treat as metrics (None = auto-detect numeric).
        higher_is_better: Dict specifying metric direction.

    Returns:
        List of GenericEvalResult, one per row.

    Example CSV:
        model,accuracy,f1,loss
        resnet50,0.95,0.94,0.12
        mobilenet,0.91,0.90,0.18

    >>> with open("results.csv") as f:
    ...     rows = list(csv.DictReader(f))
    >>> results = parse_generic_csv(rows, metric_columns=["accuracy", "f1", "loss"])
    """
    better_map = higher_is_better or {}
    results = []

    for row in rows:
        model_id = row.get(model_id_column, "unknown")

        # Extract metrics
        metrics: dict[str, float] = {}

        if metric_columns:
            # Use specified columns
            for col in metric_columns:
                if col in row:
                    try:
                        metrics[col] = float(row[col])
                    except (ValueError, TypeError):
                        pass  # Skip non-numeric
        else:
            # Auto-detect numeric columns
            for key, value in row.items():
                if key == model_id_column:
                    continue
                try:
                    metrics[key] = float(value)
                except (ValueError, TypeError):
                    pass  # Skip non-numeric

        result = GenericEvalResult.create(
            model_id=model_id,
            metrics=metrics,
            higher_is_better=better_map,
        )
        results.append(result)

    return results


def load_generic_csv(
    path: Path,
    model_id_column: str = "model",
    metric_columns: list[str] | None = None,
    higher_is_better: dict[str, bool] | None = None,
) -> list[GenericEvalResult]:
    """
    Load generic evaluation results from CSV file.

    Args:
        path: Path to CSV file.
        model_id_column: Column name containing model identifier.
        metric_columns: List of column names to treat as metrics.
        higher_is_better: Dict specifying metric direction.

    Returns:
        List of GenericEvalResult, one per row.
    """
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return parse_generic_csv(rows, model_id_column, metric_columns, higher_is_better)


# =============================================================================
# HuggingFace Evaluate Adapter (Task 12.3.2)
# =============================================================================


def parse_hf_evaluate(
    data: dict[str, Any],
    model_id: str = "",
    task_type: str = "classification",
) -> ClassificationEvalResult | NLPEvalResult | GenericEvalResult:
    """
    Parse HuggingFace evaluate library output.

    HuggingFace evaluate returns a dict with metric names as keys.
    Common output formats:
        - Classification: {"accuracy": 0.95, "f1": 0.94, "precision": 0.93, "recall": 0.95}
        - NER: {"precision": 0.9, "recall": 0.88, "f1": 0.89, "accuracy": 0.95}
        - QA: {"exact_match": 80.5, "f1": 85.3}

    Args:
        data: Dictionary from evaluate.compute() or JSON output.
        model_id: Model identifier.
        task_type: One of "classification", "nlp", or "generic".

    Returns:
        Appropriate EvalResult subtype.

    Example:
        >>> import evaluate
        >>> metric = evaluate.load("accuracy")
        >>> result = metric.compute(predictions=[1,1,0], references=[1,0,0])
        >>> eval_result = parse_hf_evaluate(result, model_id="bert-base")
    """
    # Extract model_id from data if not provided
    if not model_id:
        model_id = str(data.get("model", data.get("model_id", "unknown")))

    dataset = str(data.get("dataset", data.get("data", "")))

    # Try to auto-detect task from metric names
    has_exact_match = "exact_match" in data or "em" in data
    has_bleu = "bleu" in data or "sacrebleu" in data
    has_rouge = any(k.startswith("rouge") for k in data.keys())

    if task_type == "nlp" or has_exact_match or has_bleu or has_rouge:
        # NLP task - determine specific task from metrics
        nlp_task = "qa" if has_exact_match else ("translation" if has_bleu else "classification")
        return NLPEvalResult.create(
            model_id=model_id,
            dataset=dataset,
            nlp_task=nlp_task,
            accuracy=data.get("accuracy"),
            f1=data.get("f1", data.get("f1_score")),
            exact_match=data.get("exact_match", data.get("em")),
            bleu=data.get("bleu", data.get("sacrebleu")),
        )

    elif task_type == "classification":
        # Classification task - default to 0.0 if not found
        top1 = data.get("accuracy", data.get("top1", data.get("top_1_accuracy", 0.0)))
        top5 = data.get("top5", data.get("top_5_accuracy", 0.0))
        return ClassificationEvalResult.create(
            model_id=model_id,
            dataset=dataset,
            top1_accuracy=float(top1) if top1 is not None else 0.0,
            top5_accuracy=float(top5) if top5 is not None else 0.0,
        )

    else:
        # Generic fallback
        return parse_generic_json(data, model_id)


def load_hf_evaluate(
    path: Path,
    model_id: str = "",
    task_type: str = "classification",
) -> ClassificationEvalResult | NLPEvalResult | GenericEvalResult:
    """
    Load HuggingFace evaluate results from JSON file.

    Args:
        path: Path to JSON file.
        model_id: Optional model identifier.
        task_type: One of "classification", "nlp", or "generic".

    Returns:
        Appropriate EvalResult subtype.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return parse_hf_evaluate(data, model_id, task_type)


# =============================================================================
# lm-eval-harness Adapter (Task 12.3.3)
# =============================================================================


def parse_lm_eval(
    data: dict[str, Any],
    model_id: str = "",
) -> LLMEvalResult:
    """
    Parse lm-eval-harness (EleutherAI) output.

    lm-eval-harness outputs JSON with results per task/benchmark.
    Format varies but typically:
        {
            "results": {
                "hellaswag": {"acc": 0.7, "acc_norm": 0.75},
                "mmlu": {"acc": 0.65},
                "arc_easy": {"acc": 0.8, "acc_norm": 0.82}
            },
            "config": {"model": "llama-7b", ...}
        }

    Args:
        data: Dictionary from lm-eval JSON output.
        model_id: Model identifier (extracted from config if not provided).

    Returns:
        LLMEvalResult with benchmark scores.

    Example:
        >>> # After running: lm_eval --model hf --model_args ... --output_path results.json
        >>> result = load_lm_eval("results.json")
    """
    # Extract model_id from config
    if not model_id:
        config = data.get("config", {})
        model_id = str(
            config.get("model", config.get("model_args", {}).get("pretrained", "unknown"))
        )

    # Extract results - can be at top level or nested
    results = data.get("results", data)

    # Standard LLM benchmark scores
    def get_task_score(task_name: str, metric: str = "acc_norm") -> float | None:
        """Get score for a task, trying multiple metric names."""
        if task_name not in results:
            # Try case variations
            for key in results:
                if key.lower() == task_name.lower():
                    task_name = key
                    break
            else:
                return None

        task_data = results[task_name]
        if isinstance(task_data, dict):
            # Try acc_norm first, then acc, then the raw value
            for m in [metric, "acc_norm", "acc", "accuracy"]:
                if m in task_data:
                    val = task_data[m]
                    # Handle both raw scores (0-1) and percentages (0-100)
                    return float(val) * 100 if float(val) <= 1 else float(val)
        elif isinstance(task_data, (int, float)):
            val = float(task_data)
            return val * 100 if val <= 1 else val
        return None

    # Extract common benchmarks
    mmlu = get_task_score("mmlu") or get_task_score("mmlu_pro")
    hellaswag = get_task_score("hellaswag")
    truthfulqa = get_task_score("truthfulqa") or get_task_score("truthfulqa_mc")
    arc_easy = get_task_score("arc_easy")
    arc_challenge = get_task_score("arc_challenge")
    winogrande = get_task_score("winogrande")

    # Calculate average if we have benchmarks
    benchmark_scores: dict[str, float] = {}
    for name, score in [
        ("mmlu", mmlu),
        ("hellaswag", hellaswag),
        ("truthfulqa", truthfulqa),
        ("arc_easy", arc_easy),
        ("arc_challenge", arc_challenge),
        ("winogrande", winogrande),
    ]:
        if score is not None:
            benchmark_scores[name] = score

    # Add any other tasks not in our standard list
    for task_name, _task_data in results.items():
        if task_name.lower() not in [
            "mmlu",
            "mmlu_pro",
            "hellaswag",
            "truthfulqa",
            "truthfulqa_mc",
            "arc_easy",
            "arc_challenge",
            "winogrande",
        ]:
            score = get_task_score(task_name)
            if score is not None:
                benchmark_scores[task_name] = score

    # Try to get perplexity if available
    perplexity = None
    if "perplexity" in results:
        perplexity = results["perplexity"]
        if isinstance(perplexity, dict):
            perplexity = perplexity.get("word_perplexity", perplexity.get("perplexity"))

    return LLMEvalResult.create(
        model_id=model_id,
        perplexity=float(perplexity) if perplexity else None,
        mmlu=mmlu,
        hellaswag=hellaswag,
        truthfulqa=truthfulqa,
        benchmark_scores=benchmark_scores,
    )


def load_lm_eval(path: Path, model_id: str = "") -> LLMEvalResult:
    """
    Load lm-eval-harness results from JSON file.

    Args:
        path: Path to JSON file (lm-eval output).
        model_id: Optional model identifier.

    Returns:
        LLMEvalResult with benchmark scores.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return parse_lm_eval(data, model_id)


# =============================================================================
# timm Adapter (Task 12.3.4)
# =============================================================================


def parse_timm_benchmark(
    data: dict[str, Any],
    model_id: str = "",
) -> ClassificationEvalResult:
    """
    Parse timm (PyTorch Image Models) benchmark output.

    timm's validate.py outputs JSON/CSV with classification metrics.
    Common fields:
        - top1: top-1 accuracy (%)
        - top5: top-5 accuracy (%)
        - model: model name
        - param_count: number of parameters

    Args:
        data: Dictionary from timm validation output.
        model_id: Model identifier.

    Returns:
        ClassificationEvalResult with accuracy metrics.

    Example:
        >>> # After: python validate.py --data imagenet --model resnet50 --results-file results.json
        >>> result = load_timm_benchmark("results.json")
    """
    # Extract model_id
    if not model_id:
        model_id = str(data.get("model", data.get("arch", "unknown")))

    # Dataset
    dataset = str(data.get("dataset", data.get("data", "imagenet")))

    # Extract accuracy metrics
    def get_accuracy(keys: list[str]) -> float | None:
        for key in keys:
            if key in data:
                val = data[key]
                if val is not None:
                    # timm outputs percentages (0-100)
                    return float(val)
        return None

    top1 = get_accuracy(["top1", "top1_acc", "accuracy", "acc1", "prec1"])
    top5 = get_accuracy(["top5", "top5_acc", "acc5", "prec5"])

    # Create result - default to 0.0 if not found
    result = ClassificationEvalResult.create(
        model_id=model_id,
        dataset=dataset,
        top1_accuracy=float(top1) if top1 is not None else 0.0,
        top5_accuracy=float(top5) if top5 is not None else 0.0,
    )

    # Add extra metrics if available
    if "param_count" in data:
        result.metrics.append(
            EvalMetric(
                name="param_count",
                value=float(data["param_count"]),
                unit="params",
                higher_is_better=False,
                category="size",
            )
        )

    if "img_size" in data:
        result.metrics.append(
            EvalMetric(
                name="img_size",
                value=float(data["img_size"]),
                unit="px",
                higher_is_better=False,
                category="input",
            )
        )

    if "batch_size" in data:
        result.metrics.append(
            EvalMetric(
                name="batch_size",
                value=float(data["batch_size"]),
                unit="",
                higher_is_better=False,
                category="config",
            )
        )

    # Throughput/latency if available
    for key in ["samples_per_sec", "throughput", "samples_sec"]:
        if key in data:
            result.metrics.append(
                EvalMetric(
                    name="throughput",
                    value=float(data[key]),
                    unit="samples/sec",
                    higher_is_better=True,
                    category="speed",
                )
            )
            break

    for key in ["latency_ms", "inference_time"]:
        if key in data:
            result.metrics.append(
                EvalMetric(
                    name="latency_ms",
                    value=float(data[key]),
                    unit="ms",
                    higher_is_better=False,
                    category="speed",
                )
            )
            break

    # Store raw data
    result.metadata["raw_timm"] = data

    return result


def load_timm_benchmark(path: Path, model_id: str = "") -> ClassificationEvalResult:
    """
    Load timm benchmark results from JSON file.

    Args:
        path: Path to JSON file (timm validate.py output).
        model_id: Optional model identifier.

    Returns:
        ClassificationEvalResult with accuracy metrics.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Handle both single result and array of results
    if isinstance(data, list):
        if not data:
            return ClassificationEvalResult.create(
                model_id=model_id or "unknown",
                dataset="",
                top1_accuracy=0.0,
                top5_accuracy=0.0,
            )
        data = data[0]  # Take first result

    return parse_timm_benchmark(data, model_id)


# =============================================================================
# Auto-detect Adapter (Task 12.3.6)
# =============================================================================


def detect_and_parse(path: Path, model_id: str = "") -> EvalResult | None:
    """
    Auto-detect file format and parse with appropriate adapter.

    Detection heuristics:
    - Ultralytics: Has mAP50/mAP50-95 fields (YOLO format)
    - lm-eval-harness: Has "results" dict with benchmark names (mmlu, hellaswag, etc.)
    - timm: Has "top1"/"top5" fields (image classification)
    - HuggingFace evaluate: Has standard metric names (accuracy, f1, precision, recall)
    - Generic: Fallback for any JSON/CSV with numeric fields

    Args:
        path: Path to eval results file.
        model_id: Optional model identifier.

    Returns:
        EvalResult or None if format not recognized.

    Example:
        >>> result = detect_and_parse(Path("yolo_val.json"))
        >>> print(result.task_type)  # "detection"
    """
    suffix = path.suffix.lower()

    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Handle array of results (take first)
        if isinstance(data, list):
            if not data:
                return None
            data = data[0]

        # Check for Ultralytics signature (YOLO detection)
        ultralytics_keys = ["metrics/mAP50(B)", "box/mAP50", "mAP50", "map50", "mAP50-95"]
        if any(key in data or key in data.get("metrics", {}) for key in ultralytics_keys):
            return parse_ultralytics_val(data, model_id)

        # Check for lm-eval-harness signature (LLM benchmarks)
        lm_eval_tasks = ["mmlu", "hellaswag", "truthfulqa", "arc_easy", "winogrande"]
        if "results" in data:
            results = data["results"]
            if any(
                task in results or task.lower() in [k.lower() for k in results]
                for task in lm_eval_tasks
            ):
                return parse_lm_eval(data, model_id)
        # Also check if tasks are at top level
        if any(task in data or task.lower() in [k.lower() for k in data] for task in lm_eval_tasks):
            return parse_lm_eval(data, model_id)

        # Check for timm signature (image classification)
        timm_keys = ["top1", "top5", "top1_acc", "top5_acc", "prec1", "prec5"]
        if any(key in data for key in timm_keys):
            return parse_timm_benchmark(data, model_id)

        # Check for HuggingFace evaluate signature
        hf_keys = ["accuracy", "f1", "precision", "recall", "exact_match", "bleu", "rouge1"]
        if any(key in data for key in hf_keys):
            # Determine if NLP or classification based on keys
            nlp_keys = ["exact_match", "em", "bleu", "rouge1", "rouge2", "rougeL"]
            task_type = "nlp" if any(k in data for k in nlp_keys) else "classification"
            return parse_hf_evaluate(data, model_id, task_type)

        # Fall back to generic
        return parse_generic_json(data, model_id)

    elif suffix == ".csv":
        results = load_generic_csv(path)
        return results[0] if results else None

    return None
