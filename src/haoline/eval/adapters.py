"""
Eval Adapters

Parse evaluation results from external tools into HaoLine's schema.
Supported: Ultralytics YOLO, generic CSV/JSON.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .schemas import (
    DetectionEvalResult,
    EvalMetric,
    EvalResult,
    GenericEvalResult,
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
                EvalMetric("inference_ms", speed["inference"], "ms", False, "speed")
            )
        if "preprocess" in speed:
            result.metrics.append(
                EvalMetric("preprocess_ms", speed["preprocess"], "ms", False, "speed")
            )
        if "postprocess" in speed:
            result.metrics.append(
                EvalMetric("postprocess_ms", speed["postprocess"], "ms", False, "speed")
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
# Auto-detect Adapter (Task 12.3.6 preview)
# =============================================================================


def detect_and_parse(path: Path, model_id: str = "") -> EvalResult | None:
    """
    Auto-detect file format and parse with appropriate adapter.

    Args:
        path: Path to eval results file.
        model_id: Optional model identifier.

    Returns:
        EvalResult or None if format not recognized.
    """
    suffix = path.suffix.lower()

    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Check for Ultralytics signature
        if any(
            key in data
            for key in [
                "metrics/mAP50(B)",
                "box/mAP50",
                "mAP50",
                "map50",
            ]
        ):
            return parse_ultralytics_val(data, model_id)

        # Fall back to generic
        return parse_generic_json(data, model_id)

    elif suffix == ".csv":
        results = load_generic_csv(path)
        return results[0] if results else None

    return None
