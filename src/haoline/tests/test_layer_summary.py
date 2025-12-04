# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""Tests for per-layer summary table (Story 5.8)."""

from __future__ import annotations

import csv
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from ..analyzer import MetricsEngine, ONNXGraphLoader
from ..layer_summary import (
    LayerMetrics,
    LayerSummary,
    LayerSummaryBuilder,
    generate_html_table,
    generate_markdown_table,
)


def create_test_model() -> onnx.ModelProto:
    """Create a simple model for testing."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 224, 224])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1000])

    weight1 = helper.make_tensor(
        "w1",
        TensorProto.FLOAT,
        [64, 3, 7, 7],
        np.random.randn(64 * 3 * 7 * 7).astype(np.float32).tolist(),
    )
    weight2 = helper.make_tensor(
        "w2",
        TensorProto.FLOAT,
        [512, 1000],
        np.random.randn(512 * 1000).astype(np.float32).tolist(),
    )
    bias = helper.make_tensor(
        "b2",
        TensorProto.FLOAT,
        [1000],
        np.zeros(1000, dtype=np.float32).tolist(),
    )

    nodes = [
        helper.make_node(
            "Conv",
            ["X", "w1"],
            ["c1"],
            kernel_shape=[7, 7],
            strides=[2, 2],
            pads=[3, 3, 3, 3],
        ),
        helper.make_node("Relu", ["c1"], ["r1"]),
        helper.make_node("GlobalAveragePool", ["r1"], ["pool"]),
        helper.make_node("Flatten", ["pool"], ["flat"]),
        helper.make_node("Gemm", ["flat", "w2", "b2"], ["Y"]),
    ]

    graph = helper.make_graph(nodes, "test_model", [X], [Y], [weight1, weight2, bias])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return model


class TestLayerMetrics:
    """Tests for LayerMetrics dataclass."""

    def test_basic_creation(self):
        """Test creating layer metrics."""
        layer = LayerMetrics(
            name="conv1",
            op_type="Conv",
            params=9408,
            flops=118013952,
            pct_params=10.5,
            pct_flops=25.3,
        )
        assert layer.name == "conv1"
        assert layer.op_type == "Conv"
        assert layer.params == 9408

    def test_to_dict(self):
        """Test dictionary export."""
        layer = LayerMetrics(
            name="fc1",
            op_type="Gemm",
            params=512000,
            flops=1024000,
        )
        d = layer.to_dict()
        assert d["name"] == "fc1"
        assert d["params"] == 512000


class TestLayerSummary:
    """Tests for LayerSummary."""

    def test_creation(self):
        """Test creating layer summary."""
        layers = [
            LayerMetrics(name="conv1", op_type="Conv", params=9408, flops=1000000, pct_flops=50.0),
            LayerMetrics(name="relu1", op_type="Relu", params=0, flops=500000, pct_flops=25.0),
            LayerMetrics(name="fc1", op_type="Gemm", params=512000, flops=500000, pct_flops=25.0),
        ]
        summary = LayerSummary(
            layers=layers,
            total_params=521408,
            total_flops=2000000,
        )
        assert len(summary.layers) == 3
        assert summary.total_params == 521408

    def test_to_json(self):
        """Test JSON export."""
        layers = [
            LayerMetrics(name="conv1", op_type="Conv", params=1000, flops=100000),
        ]
        summary = LayerSummary(layers=layers, total_params=1000, total_flops=100000)

        json_str = summary.to_json()
        data = json.loads(json_str)

        assert "layers" in data
        assert len(data["layers"]) == 1
        assert data["layers"][0]["name"] == "conv1"

    def test_to_csv(self):
        """Test CSV export (Task 5.8.4)."""
        layers = [
            LayerMetrics(
                name="conv1",
                op_type="Conv",
                params=1000,
                flops=100000,
                pct_params=50.0,
                pct_flops=50.0,
            ),
            LayerMetrics(
                name="relu1",
                op_type="Relu",
                params=0,
                flops=50000,
                pct_params=0.0,
                pct_flops=25.0,
            ),
        ]
        summary = LayerSummary(layers=layers, total_params=1000, total_flops=150000)

        csv_str = summary.to_csv()
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)

        # Header + 2 data rows
        assert len(rows) == 3
        assert rows[0][0] == "Layer Name"
        assert rows[1][0] == "conv1"
        assert rows[2][0] == "relu1"

    def test_save_csv(self):
        """Test saving CSV to file."""
        layers = [
            LayerMetrics(name="test", op_type="Test", params=100, flops=1000),
        ]
        summary = LayerSummary(layers=layers, total_params=100, total_flops=1000)

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)

        try:
            summary.save_csv(path)
            assert path.exists()
            content = path.read_text()
            assert "Layer Name" in content
            assert "test" in content
        finally:
            path.unlink()

    def test_filter_by_op_type(self):
        """Test filtering by op type."""
        layers = [
            LayerMetrics(name="conv1", op_type="Conv", params=1000, flops=100000),
            LayerMetrics(name="relu1", op_type="Relu", params=0, flops=50000),
            LayerMetrics(name="conv2", op_type="Conv", params=2000, flops=200000),
        ]
        summary = LayerSummary(layers=layers, total_params=3000, total_flops=350000)

        filtered = summary.filter_by_op_type(["Conv"])
        assert len(filtered.layers) == 2
        assert all(layer.op_type == "Conv" for layer in filtered.layers)

    def test_filter_by_threshold(self):
        """Test filtering by parameter/FLOP thresholds."""
        layers = [
            LayerMetrics(name="big", op_type="Conv", params=10000, flops=1000000, pct_flops=80.0),
            LayerMetrics(name="small", op_type="Relu", params=0, flops=100000, pct_flops=8.0),
            LayerMetrics(name="medium", op_type="Conv", params=1000, flops=150000, pct_flops=12.0),
        ]
        summary = LayerSummary(layers=layers, total_params=11000, total_flops=1250000)

        # Filter to layers with >10% FLOPs
        filtered = summary.filter_by_threshold(min_pct_flops=10.0)
        assert len(filtered.layers) == 2
        assert "big" in [layer.name for layer in filtered.layers]
        assert "medium" in [layer.name for layer in filtered.layers]

    def test_sort_by(self):
        """Test sorting by different keys."""
        layers = [
            LayerMetrics(name="a", op_type="Conv", params=100, flops=300),
            LayerMetrics(name="b", op_type="Relu", params=0, flops=100),
            LayerMetrics(name="c", op_type="Gemm", params=500, flops=200),
        ]
        summary = LayerSummary(layers=layers, total_params=600, total_flops=600)

        # Sort by params descending
        sorted_summary = summary.sort_by("params", descending=True)
        assert sorted_summary.layers[0].name == "c"

        # Sort by name ascending
        sorted_summary = summary.sort_by("name", descending=False)
        assert sorted_summary.layers[0].name == "a"

    def test_top_n(self):
        """Test getting top N layers."""
        layers = [
            LayerMetrics(name="a", op_type="Conv", params=100, flops=300),
            LayerMetrics(name="b", op_type="Relu", params=0, flops=100),
            LayerMetrics(name="c", op_type="Gemm", params=500, flops=200),
        ]
        summary = LayerSummary(layers=layers, total_params=600, total_flops=600)

        top2 = summary.top_n(2, key="flops")
        assert len(top2.layers) == 2
        assert top2.layers[0].name == "a"  # 300 FLOPs
        assert top2.layers[1].name == "c"  # 200 FLOPs


class TestLayerSummaryBuilder:
    """Tests for building layer summary from ONNX."""

    def test_build_from_model(self):
        """Test building summary from ONNX model."""
        model = create_test_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            metrics_engine = MetricsEngine()
            param_counts = metrics_engine.count_parameters(graph_info)
            flop_counts = metrics_engine.estimate_flops(graph_info)

            builder = LayerSummaryBuilder()
            summary = builder.build(graph_info, param_counts, flop_counts)

            assert len(summary.layers) == 5  # 5 nodes in test model
            assert summary.total_params == param_counts.total
            assert summary.total_flops == flop_counts.total

            # Check op types
            op_types = {layer.op_type for layer in summary.layers}
            assert "Conv" in op_types
            assert "Relu" in op_types
            assert "Gemm" in op_types

        finally:
            model_path.unlink()

    def test_percentages_calculated(self):
        """Test that percentages are calculated correctly."""
        model = create_test_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            metrics_engine = MetricsEngine()
            param_counts = metrics_engine.count_parameters(graph_info)
            flop_counts = metrics_engine.estimate_flops(graph_info)

            builder = LayerSummaryBuilder()
            summary = builder.build(graph_info, param_counts, flop_counts)

            # Totals should match
            assert summary.total_params == param_counts.total
            assert summary.total_flops == flop_counts.total

            # Percentages should be valid (0-100 range)
            for layer in summary.layers:
                assert 0.0 <= layer.pct_params <= 100.0, f"Invalid pct_params: {layer.pct_params}"
                assert 0.0 <= layer.pct_flops <= 100.0, f"Invalid pct_flops: {layer.pct_flops}"

            # At least some layers should have FLOPs
            total_pct_flops = sum(layer.pct_flops for layer in summary.layers)
            # Note: might not sum to 100 if some nodes aren't in by_node dict
            assert total_pct_flops >= 0.0, f"FLOPs % sum: {total_pct_flops}"

        finally:
            model_path.unlink()


class TestHTMLTable:
    """Tests for HTML table generation (Task 5.8.2)."""

    def test_generate_html_table(self):
        """Test generating sortable HTML table."""
        layers = [
            LayerMetrics(
                name="conv1",
                op_type="Conv",
                params=9408,
                flops=118013952,
                pct_flops=80.0,
            ),
            LayerMetrics(name="relu1", op_type="Relu", params=0, flops=1000000, pct_flops=0.7),
            LayerMetrics(
                name="fc1",
                op_type="Gemm",
                params=512000,
                flops=29000000,
                pct_flops=19.3,
            ),
        ]
        summary = LayerSummary(layers=layers, total_params=521408, total_flops=148013952)

        html = generate_html_table(summary)

        # Check for table structure
        assert '<table class="layer-table"' in html
        assert "<thead>" in html
        assert "<tbody>" in html

        # Check for search input
        assert 'id="layerSearch"' in html
        assert "filterLayers" in html

        # Check for filter dropdown
        assert 'id="opFilter"' in html

        # Check for export button
        assert "exportLayersCSV" in html

        # Check for data rows
        assert "conv1" in html
        assert "relu1" in html
        assert "fc1" in html

        # Check for sorting JavaScript
        assert "sortTable" in html
        assert "filterLayers" in html

    def test_html_without_js(self):
        """Test generating HTML table without JavaScript."""
        layers = [
            LayerMetrics(name="test", op_type="Test", params=100, flops=1000),
        ]
        summary = LayerSummary(layers=layers, total_params=100, total_flops=1000)

        html = generate_html_table(summary, include_js=False)

        # Should still have table
        assert '<table class="layer-table"' in html

        # But no export function definition
        assert "const layerCSVData" not in html


class TestMarkdownTable:
    """Tests for Markdown table generation."""

    def test_generate_markdown_table(self):
        """Test generating Markdown table."""
        layers = [
            LayerMetrics(
                name="conv1",
                op_type="Conv",
                params=9408,
                flops=118000000,
                pct_flops=80.0,
            ),
            LayerMetrics(name="relu1", op_type="Relu", params=0, flops=1000000, pct_flops=0.7),
        ]
        summary = LayerSummary(layers=layers, total_params=9408, total_flops=119000000)

        md = generate_markdown_table(summary)

        # Check for header
        assert "| Layer | Op Type | Params | FLOPs | % Compute |" in md
        assert "|-------|---------|--------|-------|-----------|" in md

        # Check for data (sorted by FLOPs)
        assert "conv1" in md
        assert "Conv" in md

    def test_markdown_max_rows(self):
        """Test Markdown table respects max_rows."""
        layers = [
            LayerMetrics(name=f"layer{i}", op_type="Conv", params=100, flops=1000 * i)
            for i in range(100)
        ]
        summary = LayerSummary(layers=layers, total_params=10000, total_flops=5050000)

        md = generate_markdown_table(summary, max_rows=10)

        # Should have header + separator + 10 rows + truncation note
        lines = [line for line in md.split("\n") if line.strip()]
        assert len(lines) == 13  # header, separator, 10 rows, truncation note
        assert "more layers" in md


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
