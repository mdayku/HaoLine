# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
Unit tests for the report module (ModelInspector, InspectionReport).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from ..report import InspectionReport, ModelInspector


def create_simple_model() -> onnx.ModelProto:
    """Create a simple model for testing."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])

    W = helper.make_tensor(
        "W",
        TensorProto.FLOAT,
        [16, 3, 3, 3],
        np.random.randn(16, 3, 3, 3).astype(np.float32).flatten().tolist(),
    )

    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, 6, 6])

    conv = helper.make_node("Conv", ["X", "W"], ["conv_out"], kernel_shape=[3, 3])
    relu = helper.make_node("Relu", ["conv_out"], ["Y"])

    graph = helper.make_graph([conv, relu], "test", [X], [Y], [W])

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        producer_name="test_producer",
        producer_version="1.0",
    )
    return model


class TestModelInspector:
    """Tests for ModelInspector class."""

    def test_inspect_returns_report(self):
        """Inspect should return an InspectionReport."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            inspector = ModelInspector()
            report = inspector.inspect(model_path)

            assert isinstance(report, InspectionReport)
            assert report.metadata is not None
            assert report.graph_summary is not None
            assert report.param_counts is not None
            assert report.flop_counts is not None
            assert report.memory_estimates is not None
        finally:
            model_path.unlink()

    def test_metadata_extraction(self):
        """Test that metadata is extracted correctly."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            inspector = ModelInspector()
            report = inspector.inspect(model_path)

            assert report.metadata.producer_name == "test_producer"
            assert report.metadata.producer_version == "1.0"
            assert "ai.onnx" in report.metadata.opsets
        finally:
            model_path.unlink()

    def test_graph_summary(self):
        """Test graph summary extraction."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            inspector = ModelInspector()
            report = inspector.inspect(model_path)

            assert report.graph_summary.num_nodes == 2  # Conv + Relu
            assert report.graph_summary.num_inputs == 1
            assert report.graph_summary.num_outputs == 1
            assert "Conv" in report.graph_summary.op_type_counts
            assert "Relu" in report.graph_summary.op_type_counts
        finally:
            model_path.unlink()


class TestInspectionReport:
    """Tests for InspectionReport class."""

    def test_to_json(self):
        """Test JSON serialization."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            inspector = ModelInspector()
            report = inspector.inspect(model_path)

            json_str = report.to_json()
            assert json_str is not None

            # Should be valid JSON
            data = json.loads(json_str)
            assert "metadata" in data
            assert "graph_summary" in data
            assert "param_counts" in data
        finally:
            model_path.unlink()

    def test_to_dict(self):
        """Test dictionary serialization."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            inspector = ModelInspector()
            report = inspector.inspect(model_path)

            data = report.to_dict()
            assert isinstance(data, dict)
            assert "metadata" in data
            assert data["metadata"]["producer_name"] == "test_producer"
        finally:
            model_path.unlink()

    def test_to_markdown(self):
        """Test Markdown generation."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            inspector = ModelInspector()
            report = inspector.inspect(model_path)

            md = report.to_markdown()
            assert isinstance(md, str)
            assert "# Model Card:" in md
            assert "## Metadata" in md
            assert "## Graph Summary" in md
        finally:
            model_path.unlink()

    def test_to_html_basic(self):
        """Test HTML generation produces valid HTML structure."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            inspector = ModelInspector()
            report = inspector.inspect(model_path)

            html = report.to_html()
            assert isinstance(html, str)
            assert "<!DOCTYPE html>" in html
            assert "<html" in html
            assert "</html>" in html
            assert "HaoLine" in html
        finally:
            model_path.unlink()

    def test_to_html_contains_sections(self):
        """Test HTML contains expected sections."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            inspector = ModelInspector()
            report = inspector.inspect(model_path)

            html = report.to_html()
            # Should have key sections
            assert "Model Details" in html
            assert "Metadata" in html
            assert "Graph Summary" in html
        finally:
            model_path.unlink()

    def test_to_html_with_llm_summary(self):
        """Test HTML includes LLM summary when provided."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            inspector = ModelInspector()
            report = inspector.inspect(model_path)

            # Add mock LLM summary
            report.llm_summary = {
                "success": True,
                "short_summary": "Test short summary.",
                "detailed_summary": "Test detailed summary paragraph.",
                "model": "test-model",
            }

            html = report.to_html()
            assert "Executive Summary" in html
            assert "Test short summary" in html
            assert "Test detailed summary" in html
        finally:
            model_path.unlink()

    def test_to_markdown_with_llm_summary(self):
        """Test Markdown includes executive summary when LLM summary provided."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            inspector = ModelInspector()
            report = inspector.inspect(model_path)

            # Add mock LLM summary
            report.llm_summary = {
                "success": True,
                "short_summary": "This is a test model for unit testing.",
                "detailed_summary": "This model contains Conv and ReLU operations for testing.",
                "model": "test-llm-model",
            }

            md = report.to_markdown()
            assert "## Executive Summary" in md
            assert "Executive Summary" in md
            assert "This is a test model for unit testing." in md
            assert "This model contains Conv and ReLU operations for testing." in md
            assert "*Generated by test-llm-model*" in md
        finally:
            model_path.unlink()

    def test_to_markdown_no_executive_summary_without_llm(self):
        """Test Markdown omits executive summary when no LLM summary."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            inspector = ModelInspector()
            report = inspector.inspect(model_path)

            # No LLM summary set
            md = report.to_markdown()
            assert "## Executive Summary" not in md
        finally:
            model_path.unlink()

    def test_to_html_embeds_images(self):
        """Test HTML embeds images as base64 when provided."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            inspector = ModelInspector()
            report = inspector.inspect(model_path)

            # Create a temp image file
            with tempfile.TemporaryDirectory() as tmpdir:
                # Create a minimal PNG (1x1 pixel)
                import base64

                # Minimal valid PNG
                png_data = base64.b64decode(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                )
                img_path = Path(tmpdir) / "test.png"
                img_path.write_bytes(png_data)

                html = report.to_html(image_paths={"test_chart": img_path})
                assert "data:image/png;base64," in html
        finally:
            model_path.unlink()

    def test_format_number(self):
        """Test number formatting utility."""
        assert InspectionReport._format_number(1_000) == "1.00K"
        assert InspectionReport._format_number(1_000_000) == "1.00M"
        assert InspectionReport._format_number(1_000_000_000) == "1.00B"
        assert InspectionReport._format_number(500) == "500"

    def test_format_bytes(self):
        """Test byte formatting utility."""
        assert InspectionReport._format_bytes(1024) == "1.02 KB"
        assert InspectionReport._format_bytes(1024 * 1024) == "1.05 MB"
        assert InspectionReport._format_bytes(1024 * 1024 * 1024) == "1.07 GB"
        assert InspectionReport._format_bytes(500) == "500 bytes"


class TestReportRiskSignals:
    """Tests for risk signal inclusion in reports."""

    def test_report_includes_risk_signals(self):
        """Reports should include analyzed risk signals."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            inspector = ModelInspector()
            report = inspector.inspect(model_path)

            # risk_signals is always a list (may be empty for small models)
            assert isinstance(report.risk_signals, list)
        finally:
            model_path.unlink()

    def test_report_includes_blocks(self):
        """Reports should include detected blocks."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            inspector = ModelInspector()
            report = inspector.inspect(model_path)

            assert isinstance(report.detected_blocks, list)
            assert report.architecture_type in (
                "cnn",
                "mlp",
                "transformer",
                "hybrid",
                "unknown",
            )
        finally:
            model_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
