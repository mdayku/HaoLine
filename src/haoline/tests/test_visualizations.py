# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
Unit tests for the visualization module.

Tests chart generation, theming, and graceful fallback behavior.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ..analyzer import FlopCounts, MemoryEstimates, ParamCounts
from ..report import GraphSummary, InspectionReport, ModelMetadata
from ..visualizations import (
    THEME,
    ChartTheme,
    VisualizationGenerator,
    _format_count,
    generate_visualizations,
    is_available,
)


class TestVisualizationAvailability:
    """Tests for matplotlib availability detection."""

    def test_is_available_returns_bool(self):
        """is_available() should return a boolean."""
        result = is_available()
        assert isinstance(result, bool)

    def test_theme_has_required_colors(self):
        """Theme should have all required color properties."""
        assert hasattr(THEME, "background")
        assert hasattr(THEME, "text")
        assert hasattr(THEME, "accent_primary")
        assert hasattr(THEME, "palette")
        assert len(THEME.palette) >= 5  # Need enough colors for charts


class TestChartTheme:
    """Tests for the ChartTheme dataclass."""

    def test_default_theme_values(self):
        """Default theme should have sensible values."""
        theme = ChartTheme()
        assert theme.background.startswith("#")
        assert theme.text.startswith("#")
        assert theme.figure_dpi >= 72
        assert theme.figure_width > 0
        assert theme.figure_height > 0

    def test_custom_theme(self):
        """Should be able to create custom themes."""
        theme = ChartTheme(
            background="#000000",
            text="#ffffff",
            accent_primary="#ff0000",
        )
        assert theme.background == "#000000"
        assert theme.text == "#ffffff"
        assert theme.accent_primary == "#ff0000"


class TestFormatCount:
    """Tests for the _format_count helper function."""

    def test_format_small_numbers(self):
        """Small numbers should be formatted as-is."""
        assert _format_count(0) == "0"
        assert _format_count(1) == "1"
        assert _format_count(999) == "999"

    def test_format_thousands(self):
        """Thousands should use K suffix."""
        assert _format_count(1000) == "1.0K"
        assert _format_count(1500) == "1.5K"
        assert _format_count(999999) == "1000.0K"  # Just under 1M

    def test_format_millions(self):
        """Millions should use M suffix."""
        assert _format_count(1000000) == "1.0M"
        assert _format_count(1500000) == "1.5M"
        assert _format_count(25000000) == "25.0M"

    def test_format_billions(self):
        """Billions should use B suffix."""
        assert _format_count(1000000000) == "1.0B"
        assert _format_count(7500000000) == "7.5B"


class TestVisualizationGenerator:
    """Tests for the VisualizationGenerator class."""

    def test_generator_initialization(self):
        """Generator should initialize without errors."""
        gen = VisualizationGenerator()
        assert gen is not None
        assert gen.logger is not None

    def test_generator_with_custom_logger(self):
        """Generator should accept custom logger."""
        import logging

        logger = logging.getLogger("test")
        gen = VisualizationGenerator(logger=logger)
        assert gen.logger is logger


@pytest.mark.skipif(not is_available(), reason="matplotlib not installed")
class TestChartGeneration:
    """Tests for actual chart generation (requires matplotlib)."""

    def test_operator_histogram_generation(self):
        """Should generate operator histogram PNG."""
        gen = VisualizationGenerator()
        op_counts = {
            "Conv": 50,
            "Relu": 48,
            "Add": 25,
            "MatMul": 12,
            "Softmax": 6,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "op_histogram.png"
            result = gen.operator_histogram(op_counts, output_path)

            assert result is not None
            assert result.exists()
            assert result.stat().st_size > 0  # File has content

    def test_param_distribution_generation(self):
        """Should generate parameter distribution chart."""
        gen = VisualizationGenerator()
        params_by_op = {
            "Conv": 5000000,
            "Gemm": 2000000,
            "MatMul": 1500000,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "param_dist.png"
            result = gen.param_distribution(params_by_op, output_path)

            assert result is not None
            assert result.exists()

    def test_flops_distribution_generation(self):
        """Should generate FLOPs distribution chart."""
        gen = VisualizationGenerator()
        flops_by_op = {
            "Conv": 500000000,
            "MatMul": 200000000,
            "Gemm": 150000000,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "flops_dist.png"
            result = gen.flops_distribution(flops_by_op, output_path)

            assert result is not None
            assert result.exists()

    def test_empty_data_returns_none(self):
        """Charts should return None for empty data."""
        gen = VisualizationGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Empty dict
            result = gen.operator_histogram({}, Path(tmpdir) / "empty.png")
            assert result is None

            # All zeros
            result = gen.param_distribution({"Conv": 0, "Relu": 0}, Path(tmpdir) / "zeros.png")
            assert result is None

    def test_generate_all_creates_multiple_charts(self):
        """generate_all should create multiple chart files."""
        # Create a mock report
        metadata = ModelMetadata(
            path="test.onnx",
            ir_version=8,
            producer_name="test",
            producer_version="1.0",
            domain="",
            model_version=1,
            doc_string="",
            opsets={"ai.onnx": 17},
        )

        graph_summary = GraphSummary(
            num_nodes=100,
            num_inputs=1,
            num_outputs=1,
            num_initializers=50,
            input_shapes={"input": [1, 3, 224, 224]},
            output_shapes={"output": [1, 1000]},
            op_type_counts={"Conv": 50, "Relu": 48, "Add": 25},
        )

        param_counts = ParamCounts(
            total=25000000,
            trainable=25000000,
            by_op_type={"Conv": 20000000, "Gemm": 5000000},
        )

        flop_counts = FlopCounts(
            total=4000000000,
            by_op_type={"Conv": 3500000000, "Gemm": 500000000},
        )

        memory_estimates = MemoryEstimates(
            model_size_bytes=100000000,
            peak_activation_bytes=50000000,
        )

        report = InspectionReport(
            metadata=metadata,
            graph_summary=graph_summary,
            param_counts=param_counts,
            flop_counts=flop_counts,
            memory_estimates=memory_estimates,
        )

        gen = VisualizationGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = gen.generate_all(report, Path(tmpdir))

            assert len(paths) >= 3  # Should generate at least 3 charts
            assert "op_histogram" in paths
            assert "param_distribution" in paths
            assert "complexity_summary" in paths

            # All files should exist
            for name, path in paths.items():
                assert path.exists(), f"{name} file should exist"


@pytest.mark.skipif(not is_available(), reason="matplotlib not installed")
class TestConvenienceFunction:
    """Tests for the generate_visualizations convenience function."""

    def test_generate_visualizations_function(self):
        """Convenience function should work correctly."""
        metadata = ModelMetadata(
            path="test.onnx",
            ir_version=8,
            producer_name="test",
            producer_version="1.0",
            domain="",
            model_version=1,
            doc_string="",
            opsets={"ai.onnx": 17},
        )

        graph_summary = GraphSummary(
            num_nodes=10,
            num_inputs=1,
            num_outputs=1,
            num_initializers=5,
            input_shapes={},
            output_shapes={},
            op_type_counts={"Conv": 5, "Relu": 5},
        )

        param_counts = ParamCounts(total=1000, trainable=1000, by_op_type={"Conv": 1000})
        flop_counts = FlopCounts(total=10000, by_op_type={"Conv": 10000})
        memory_estimates = MemoryEstimates(model_size_bytes=4000, peak_activation_bytes=2000)

        report = InspectionReport(
            metadata=metadata,
            graph_summary=graph_summary,
            param_counts=param_counts,
            flop_counts=flop_counts,
            memory_estimates=memory_estimates,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_visualizations(report, tmpdir)
            assert isinstance(paths, dict)
            assert len(paths) > 0


class TestGracefulDegradation:
    """Tests for graceful degradation when matplotlib is unavailable."""

    def test_generator_handles_missing_matplotlib(self):
        """Generator should not crash if matplotlib is unavailable."""
        # This test runs regardless of matplotlib availability
        gen = VisualizationGenerator()

        # Even with matplotlib, empty data should return empty dict
        metadata = ModelMetadata(
            path="test.onnx",
            ir_version=8,
            producer_name="test",
            producer_version="1.0",
            domain="",
            model_version=1,
            doc_string="",
            opsets={},
        )

        # Report with no metrics - should return empty dict
        report = InspectionReport(metadata=metadata)

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = gen.generate_all(report, Path(tmpdir))
            assert isinstance(paths, dict)
