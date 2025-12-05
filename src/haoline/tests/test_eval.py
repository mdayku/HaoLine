"""Tests for the eval module: schemas, adapters, and linking utilities."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from haoline.eval.deployment import (
    HARDWARE_TIERS,
    DeploymentScenario,
    DeploymentTarget,
    calculate_deployment_cost,
    estimate_latency_from_flops,
    get_hardware_tier,
    list_hardware_tiers,
    select_hardware_tier_for_latency,
)
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


# =============================================================================
# Deployment Cost Calculator Tests
# =============================================================================


class TestDeploymentScenario:
    """Tests for DeploymentScenario dataclass."""

    def test_default_scenario(self) -> None:
        """Test creating scenario with defaults."""
        scenario = DeploymentScenario()
        assert scenario.target_fps == 30.0
        assert scenario.hours_per_day == 24.0
        assert scenario.target == DeploymentTarget.CLOUD_GPU

    def test_realtime_video_preset(self) -> None:
        """Test realtime video preset."""
        scenario = DeploymentScenario.realtime_video(fps=60.0)
        assert scenario.target_fps == 60.0
        assert scenario.max_latency_ms == pytest.approx(1000.0 / 60, rel=0.01)
        assert scenario.name == "realtime_video"

    def test_edge_device_preset(self) -> None:
        """Test edge device preset."""
        scenario = DeploymentScenario.edge_device(fps=10.0)
        assert scenario.target == DeploymentTarget.EDGE_GPU
        assert scenario.target_fps == 10.0

    def test_serialization(self) -> None:
        """Test to_dict and from_dict."""
        original = DeploymentScenario(
            target_fps=15.0,
            hours_per_day=8.0,
            precision="fp16",
        )
        data = original.to_dict()
        restored = DeploymentScenario.from_dict(data)
        assert restored.target_fps == 15.0
        assert restored.hours_per_day == 8.0
        assert restored.precision == "fp16"


class TestHardwareTiers:
    """Tests for hardware tier lookups."""

    def test_get_hardware_tier(self) -> None:
        """Test getting a tier by name."""
        tier = get_hardware_tier("t4")
        assert tier is not None
        assert tier.name == "T4"
        assert tier.cost_per_hour_usd > 0

    def test_get_hardware_tier_case_insensitive(self) -> None:
        """Test case-insensitive lookup."""
        tier = get_hardware_tier("A10G")
        assert tier is not None
        assert tier.name == "A10G"

    def test_get_unknown_tier(self) -> None:
        """Test getting non-existent tier returns None."""
        tier = get_hardware_tier("nonexistent")
        assert tier is None

    def test_list_hardware_tiers(self) -> None:
        """Test listing all tiers."""
        tiers = list_hardware_tiers()
        assert len(tiers) > 0
        # Should be sorted by cost
        costs = [t.cost_per_hour_usd for t in tiers]
        assert costs == sorted(costs)

    def test_list_hardware_tiers_filtered(self) -> None:
        """Test filtering by target."""
        gpu_tiers = list_hardware_tiers(DeploymentTarget.CLOUD_GPU)
        for tier in gpu_tiers:
            assert tier.target == DeploymentTarget.CLOUD_GPU

        edge_tiers = list_hardware_tiers(DeploymentTarget.EDGE_GPU)
        for tier in edge_tiers:
            assert tier.target == DeploymentTarget.EDGE_GPU


class TestCostCalculation:
    """Tests for deployment cost calculation."""

    def test_estimate_latency_from_flops(self) -> None:
        """Test latency estimation."""
        tier = get_hardware_tier("t4")
        assert tier is not None

        # 1 GFLOP model
        flops = 1_000_000_000
        latency = estimate_latency_from_flops(flops, tier, "fp32")

        # Should be a reasonable latency value
        assert latency > 0
        assert latency < 10000  # Less than 10 seconds

    def test_select_hardware_for_latency(self) -> None:
        """Test hardware selection based on latency SLA."""
        flops = 10_000_000_000  # 10 GFLOP model

        # Strict latency requirement - should pick faster hardware
        tier = select_hardware_tier_for_latency(
            flops,
            target_latency_ms=10.0,
            precision="fp16",
        )
        # May or may not find suitable tier
        if tier:
            assert tier.cost_per_hour_usd > 0

    def test_calculate_deployment_cost(self) -> None:
        """Test full cost calculation."""
        scenario = DeploymentScenario(
            target_fps=10.0,
            hours_per_day=8.0,
            days_per_month=22,  # Business days
            target=DeploymentTarget.CLOUD_GPU,
        )

        flops = 5_000_000_000  # 5 GFLOP model
        estimate = calculate_deployment_cost(flops, scenario)

        # Check basic fields are populated
        assert estimate.hardware_tier is not None
        assert estimate.cost_per_hour_usd >= 0
        assert estimate.cost_per_day_usd >= 0
        assert estimate.cost_per_month_usd >= 0
        assert estimate.estimated_latency_ms > 0

        # Costs should scale correctly
        assert estimate.cost_per_day_usd == pytest.approx(
            estimate.cost_per_hour_usd * 8.0, rel=0.01
        )
        assert estimate.cost_per_month_usd == pytest.approx(
            estimate.cost_per_day_usd * 22, rel=0.01
        )

    def test_cost_estimate_summary(self) -> None:
        """Test human-readable summary generation."""
        scenario = DeploymentScenario(
            target_fps=30.0,
            hours_per_day=24.0,
            name="test_scenario",
        )

        estimate = calculate_deployment_cost(1_000_000_000, scenario)
        summary = estimate.summary()

        assert "test_scenario" in summary
        assert "Per hour:" in summary
        assert "Per month:" in summary
