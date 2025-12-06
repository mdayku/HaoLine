# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
Unit tests for the schema module (JSON schema validation).

Tests cover both Pydantic validation (preferred) and jsonschema fallback.
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
    validate_with_pydantic,
)

# Pydantic is always available now (required dependency)
PYDANTIC_AVAILABLE = True

# Check if jsonschema is available (fallback)
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
        # Note: Pydantic-generated schemas don't include $schema field
        # but do include all the structural elements we need
        assert "properties" in schema
        assert "required" in schema
        assert "title" in schema  # Pydantic includes title

    def test_schema_required_fields(self):
        """Verify required fields are defined."""
        schema = get_schema()
        required = schema["required"]
        # Only metadata is required - generated_at and autodoc_version have defaults
        assert "metadata" in required

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


@pytest.mark.skipif(
    not PYDANTIC_AVAILABLE and not JSONSCHEMA_AVAILABLE,
    reason="Neither pydantic nor jsonschema installed",
)
class TestSchemaValidation:
    """Tests for schema validation (Pydantic preferred, jsonschema fallback)."""

    def test_valid_report_passes_validation(self):
        """A properly generated report should pass validation."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            inspector = ModelInspector()
            report = inspector.inspect(model_path)

            is_valid, errors = report.validate_schema()
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
            report.validate_schema_strict()
        finally:
            model_path.unlink()

    def test_architecture_type_accepts_string(self):
        """Architecture type accepts any string value (flexible schema)."""
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
            "architecture_type": "custom_type",  # Any string is valid now
        }

        is_valid, errors = validate_report(valid_report)
        assert is_valid

    def test_risk_signal_severity_accepts_string(self):
        """Risk signal severity accepts any string value (flexible schema)."""
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
                    "severity": "critical",  # Any string is valid now
                    "description": "Test risk",
                }
            ],
        }

        is_valid, errors = validate_report(valid_report)
        assert is_valid


@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="pydantic not installed")
class TestPydanticValidation:
    """Tests specific to Pydantic validation."""

    def test_validate_with_pydantic_returns_model(self):
        """validate_with_pydantic should return a Pydantic model instance."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            inspector = ModelInspector()
            report = inspector.inspect(model_path)
            report_dict = report.to_dict()

            pydantic_model = validate_with_pydantic(report_dict)
            assert pydantic_model is not None
            assert hasattr(pydantic_model, "metadata")
            assert hasattr(pydantic_model, "generated_at")
            assert hasattr(pydantic_model, "autodoc_version")
        finally:
            model_path.unlink()

    def test_validate_with_pydantic_invalid_raises(self):
        """validate_with_pydantic should raise ValidationError on invalid input."""
        invalid_report = {"not_valid": True}

        with pytest.raises(ValidationError) as exc_info:
            validate_with_pydantic(invalid_report)

        assert len(exc_info.value.errors) > 0

    def test_pydantic_model_has_correct_types(self):
        """Pydantic model should have correct field types after parsing."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            inspector = ModelInspector()
            report = inspector.inspect(model_path)
            report_dict = report.to_dict()

            pydantic_model = validate_with_pydantic(report_dict)
            assert pydantic_model is not None

            # Check metadata structure
            assert pydantic_model.metadata is not None
            assert isinstance(pydantic_model.metadata.path, str)
            assert isinstance(pydantic_model.metadata.ir_version, int)

            # Check autodoc_version is a string
            assert isinstance(pydantic_model.autodoc_version, str)
        finally:
            model_path.unlink()

    def test_pydantic_schema_matches_structure(self):
        """Pydantic-generated schema should have expected structure."""
        schema = get_schema()

        # Pydantic schema should have these elements
        assert "properties" in schema
        assert "required" in schema

        # Check key properties exist
        props = schema["properties"]
        assert "metadata" in props
        assert "generated_at" in props
        assert "autodoc_version" in props

    def test_pydantic_validation_error_messages_are_readable(self):
        """Pydantic validation errors should be human-readable."""
        invalid_report = {
            "metadata": {
                "path": "/test/model.onnx",
                "ir_version": "invalid",  # Should be int
                "producer_name": "test",
                "opsets": {"": 17},
            },
            "generated_at": "2025-01-01T00:00:00Z",
            "autodoc_version": "0.1.0",
        }

        is_valid, errors = validate_report(invalid_report)
        assert not is_valid
        assert len(errors) > 0
        # Error message should contain the field path
        error_str = " ".join(errors)
        assert "ir_version" in error_str.lower()

    def test_pydantic_validates_nested_objects(self):
        """Pydantic should validate nested objects correctly."""
        # Valid nested structure
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
            "param_counts": {
                "total": 1000,
                "trainable": 1000,
                "non_trainable": 0,
                "by_op_type": {"Conv": 500, "MatMul": 500},
            },
        }

        is_valid, errors = validate_report(valid_report)
        assert is_valid, f"Validation failed: {errors}"

    def test_pydantic_accepts_nested_dicts_loosely(self):
        """Nested objects are validated loosely (Any type) until full type hints are added.

        Note: After Story 40.6, when InspectionReport type hints are updated from Any
        to actual types (ParamCounts, etc.), this test should be updated to verify
        strict type checking.
        """
        # This would fail with strict type checking, but currently param_counts is Any
        report_with_loose_nested = {
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
            "param_counts": {
                "total": "not_a_number",  # Accepted because param_counts is Any
                "trainable": 1000,
            },
        }

        # Currently accepts loose typing - will be stricter after Story 40.6
        is_valid, errors = validate_report(report_with_loose_nested)
        assert is_valid  # Loose validation until type hints are updated


class TestSchemaWithoutJsonschema:
    """Tests for behavior when jsonschema is not installed."""

    def test_get_schema_always_works(self):
        """get_schema should work regardless of jsonschema."""
        schema = get_schema()
        assert isinstance(schema, dict)
        assert "properties" in schema


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
