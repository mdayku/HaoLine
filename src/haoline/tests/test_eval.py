"""Tests for the eval module: schemas, adapters, and linking utilities."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from haoline.eval.schemas import (
    CombinedReport,
    DetectionEvalResult,
    EvalMetric,
    EvalResult,
    compute_model_hash,
    create_combined_report,
    link_eval_to_model,
    validate_eval_result,
)


class TestEvalMetric:
    """Tests for EvalMetric Pydantic model."""

    def test_create_metric(self) -> None:
        """Test creating an EvalMetric."""
        metric = EvalMetric(
            name="accuracy",
            value=95.5,
            unit="%",
            higher_is_better=True,
            category="accuracy",
        )
        assert metric.name == "accuracy"
        assert metric.value == 95.5
        assert metric.unit == "%"
        assert metric.higher_is_better is True

    def test_metric_json_serialization(self) -> None:
        """Test EvalMetric serialization."""
        metric = EvalMetric(
            name="loss",
            value=0.05,
            unit="",
            higher_is_better=False,
            category="loss",
        )
        data = json.loads(metric.model_dump_json())
        assert data["name"] == "loss"
        assert data["higher_is_better"] is False


class TestEvalResult:
    """Tests for EvalResult base class."""

    def test_create_eval_result(self) -> None:
        """Test creating an EvalResult."""
        result = EvalResult(
            model_id="test-model",
            task_type="classification",
            dataset="imagenet",
            metrics=[
                EvalMetric(
                    name="top1",
                    value=76.5,
                    unit="%",
                    higher_is_better=True,
                    category="accuracy",
                )
            ],
        )
        assert result.model_id == "test-model"
        assert result.task_type == "classification"
        assert len(result.metrics) == 1

    def test_to_json(self) -> None:
        """Test JSON serialization."""
        result = EvalResult(
            model_id="model",
            task_type="detection",
            metrics=[],
        )
        json_str = result.to_json()
        data = json.loads(json_str)
        assert data["model_id"] == "model"
        assert data["task_type"] == "detection"


class TestDetectionEvalResult:
    """Tests for detection-specific eval result."""

    def test_create_with_factory(self) -> None:
        """Test using the create() convenience method."""
        result = DetectionEvalResult.create(
            model_id="yolov8n",
            dataset="coco",
            map50=0.65,
            map50_95=0.48,
            precision=0.72,
            recall=0.68,
            f1=0.70,
        )
        assert result.model_id == "yolov8n"
        assert result.dataset == "coco"
        assert len(result.metrics) == 5


class TestLinkingUtilities:
    """Tests for model-eval linking functions."""

    def test_compute_model_hash(self, tmp_path: Path) -> None:
        """Test computing file hash."""
        # Create a temporary file
        test_file = tmp_path / "model.onnx"
        test_file.write_bytes(b"fake model content")

        hash_result = compute_model_hash(str(test_file))
        assert len(hash_result) == 64  # SHA-256 hex length
        assert hash_result.isalnum()

    def test_compute_model_hash_not_found(self) -> None:
        """Test hash of non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            compute_model_hash("/nonexistent/path/model.onnx")

    def test_link_eval_to_model(self, tmp_path: Path) -> None:
        """Test linking eval result to model file."""
        # Create a temporary model file
        model_file = tmp_path / "yolov8n.onnx"
        model_file.write_bytes(b"model content")

        result = EvalResult(
            model_id="",
            task_type="detection",
            metrics=[],
        )

        linked = link_eval_to_model(str(model_file), result, use_hash=False)
        assert linked.model_id == "yolov8n"
        assert "linked_model_path" in linked.metadata

    def test_link_eval_to_model_with_hash(self, tmp_path: Path) -> None:
        """Test linking with hash-based model ID."""
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"unique content")

        result = EvalResult(
            model_id="",
            task_type="classification",
            metrics=[],
        )

        linked = link_eval_to_model(str(model_file), result, use_hash=True)
        assert len(linked.model_id) == 12  # Short hash

    def test_create_combined_report_no_inspection(self, tmp_path: Path) -> None:
        """Test creating combined report without running inspection."""
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"model")

        eval_result = DetectionEvalResult.create(
            model_id="",
            dataset="coco",
            map50=0.65,
            map50_95=0.48,
            precision=0.72,
            recall=0.68,
            f1=0.70,
        )

        combined = create_combined_report(
            str(model_file),
            eval_results=[eval_result],
            run_inspection=False,
        )

        assert combined.model_id == "model"
        assert len(combined.eval_results) == 1
        assert combined.eval_results[0].model_id == "model"


class TestValidation:
    """Tests for schema validation."""

    def test_validate_valid_eval_result(self) -> None:
        """Test validation of valid data."""
        data = {
            "model_id": "test",
            "task_type": "classification",
            "metrics": [],
        }
        assert validate_eval_result(data) is True

    def test_validate_invalid_eval_result(self) -> None:
        """Test validation of invalid data."""
        data = {"invalid": "data"}
        assert validate_eval_result(data) is False


class TestCombinedReport:
    """Tests for CombinedReport model."""

    def test_create_combined_report(self) -> None:
        """Test creating a CombinedReport manually."""
        combined = CombinedReport(
            model_id="resnet50",
            model_path="/path/to/resnet50.onnx",
            architecture={
                "params_total": 25_000_000,
                "flops_total": 4_000_000_000,
            },
            eval_results=[],
        )
        assert combined.model_id == "resnet50"
        assert combined.architecture["params_total"] == 25_000_000

    def test_add_eval_result(self) -> None:
        """Test adding eval results to combined report."""
        combined = CombinedReport(
            model_id="model",
            architecture={},
        )
        eval_result = EvalResult(
            model_id="model",
            task_type="classification",
            metrics=[],
        )
        combined.add_eval_result(eval_result)
        assert len(combined.eval_results) == 1

    def test_get_eval_by_task(self) -> None:
        """Test retrieving eval by task type."""
        combined = CombinedReport(
            model_id="model",
            architecture={},
            eval_results=[
                EvalResult(model_id="m", task_type="detection", metrics=[]),
                EvalResult(model_id="m", task_type="classification", metrics=[]),
            ],
        )
        det = combined.get_eval_by_task("detection")
        assert det is not None
        assert det.task_type == "detection"

        missing = combined.get_eval_by_task("segmentation")
        assert missing is None
