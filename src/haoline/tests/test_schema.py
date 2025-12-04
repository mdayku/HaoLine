# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
Unit tests for the schema module (JSON schema validation).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from ..report import ModelInspector
from ..schema import (
    ValidationError,
    get_schema,
    validate_report,
    validate_report_strict,
)

# Check if jsonschema is available
try:
    from jsonschema import Draft7Validator  # noqa: F401

    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False


def create_simple_model() -> onnx.ModelProto:
    """Create a simple ONNX model for testing."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 64])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 64])

    W = helper.make_tensor(
        "W",
        TensorProto.FLOAT,
        [64, 64],
        np.random.randn(64, 64).astype(np.float32).flatten().tolist(),
    )

    matmul = helper.make_node("MatMul", ["X", "W"], ["matmul_out"], name="matmul")
    relu = helper.make_node("Relu", ["matmul_out"], ["Y"], name="relu")

    graph = helper.make_graph([matmul, relu], "simple_model", [X], [Y], [W])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return model


class TestSchemaDefinition:
    """Tests for the JSON schema definition."""

    def test_schema_has_required_fields(self):
        """Verify schema has the required top-level structure."""
        schema = get_schema()
        assert "$schema" in schema
        assert "properties" in schema
        assert "required" in schema

    def test_schema_required_fields(self):
        """Verify required fields are defined."""
        schema = get_schema()
        required = schema["required"]
        assert "metadata" in required
        assert "generated_at" in required
        assert "autodoc_version" in required

    def test_schema_has_all_sections(self):
        """Verify schema includes all report sections."""
        schema = get_schema()
        props = schema["properties"]

        expected_sections = [
            "metadata",
            "generated_at",
            "autodoc_version",
            "graph_summary",
            "param_counts",
            "flop_counts",
            "memory_estimates",
            "detected_blocks",
            "architecture_type",
            "risk_signals",
            "hardware_profile",
            "hardware_estimates",
            "llm_summary",
            "dataset_info",
        ]

        for section in expected_sections:
            assert section in props, f"Missing schema section: {section}"


@pytest.mark.skipif(not JSONSCHEMA_AVAILABLE, reason="jsonschema not installed")
class TestSchemaValidation:
    """Tests for JSON schema validation (requires jsonschema)."""

    def test_valid_report_passes_validation(self):
        """A properly generated report should pass validation."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            inspector = ModelInspector()
            report = inspector.inspect(model_path)

            is_valid, errors = report.validate()
            assert is_valid, f"Validation failed: {errors}"
            assert len(errors) == 0
        finally:
            model_path.unlink()

    def test_validate_report_function(self):
        """Test the validate_report function directly."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            inspector = ModelInspector()
            report = inspector.inspect(model_path)
            report_dict = report.to_dict()

            is_valid, errors = validate_report(report_dict)
            assert is_valid
            assert len(errors) == 0
        finally:
            model_path.unlink()

    def test_invalid_report_fails_validation(self):
        """An invalid report should fail validation."""
        # Missing required fields
        invalid_report = {
            "generated_at": "2025-01-01T00:00:00Z",
            # Missing metadata and autodoc_version
        }

        is_valid, errors = validate_report(invalid_report)
        assert not is_valid
        assert len(errors) > 0
        assert any("metadata" in e for e in errors)

    def test_invalid_metadata_fails_validation(self):
        """Invalid metadata should fail validation."""
        invalid_report = {
            "metadata": {
                "path": "/test/model.onnx",
                "ir_version": "not_an_integer",  # Should be int
                "producer_name": "test",
                "opsets": {"": 17},
            },
            "generated_at": "2025-01-01T00:00:00Z",
            "autodoc_version": "0.1.0",
        }

        is_valid, errors = validate_report(invalid_report)
        assert not is_valid
        assert any("ir_version" in e for e in errors)

    def test_validate_strict_raises_on_invalid(self):
        """validate_report_strict should raise ValidationError."""
        invalid_report = {"not_valid": True}

        with pytest.raises(ValidationError) as exc_info:
            validate_report_strict(invalid_report)

        assert len(exc_info.value.errors) > 0

    def test_report_validate_strict_method(self):
        """Test the validate_strict method on InspectionReport."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            inspector = ModelInspector()
            report = inspector.inspect(model_path)

            # Should not raise
            report.validate_strict()
        finally:
            model_path.unlink()

    def test_architecture_type_enum_validation(self):
        """Architecture type should be one of the allowed values."""
        valid_report = {
            "metadata": {
                "path": "/test/model.onnx",
                "ir_version": 9,
                "producer_name": "test",
                "producer_version": "1.0",
                "domain": "",
                "model_version": 0,
                "doc_string": "",
                "opsets": {"": 17},
            },
            "generated_at": "2025-01-01T00:00:00Z",
            "autodoc_version": "0.1.0",
            "architecture_type": "invalid_type",  # Not in enum
        }

        is_valid, errors = validate_report(valid_report)
        assert not is_valid
        assert any("architecture_type" in e for e in errors)

    def test_risk_signal_severity_enum(self):
        """Risk signal severity should be one of the allowed values."""
        valid_report = {
            "metadata": {
                "path": "/test/model.onnx",
                "ir_version": 9,
                "producer_name": "test",
                "producer_version": "",
                "domain": "",
                "model_version": 0,
                "doc_string": "",
                "opsets": {"": 17},
            },
            "generated_at": "2025-01-01T00:00:00Z",
            "autodoc_version": "0.1.0",
            "risk_signals": [
                {
                    "id": "test_risk",
                    "severity": "critical",  # Invalid - should be info/warning/high
                    "description": "Test risk",
                }
            ],
        }

        is_valid, errors = validate_report(valid_report)
        assert not is_valid
        assert any("severity" in e for e in errors)


class TestSchemaWithoutJsonschema:
    """Tests for behavior when jsonschema is not installed."""

    def test_get_schema_always_works(self):
        """get_schema should work regardless of jsonschema."""
        schema = get_schema()
        assert isinstance(schema, dict)
        assert "properties" in schema


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
