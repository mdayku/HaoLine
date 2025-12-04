# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""Tests for HTML export."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ..edge_analysis import EdgeAnalysisResult
from ..hierarchical_graph import HierarchicalGraph, HierarchicalNode
from ..html_export import HTMLExporter, generate_html


def create_test_graph() -> HierarchicalGraph:
    """Create a test hierarchical graph."""
    child1 = HierarchicalNode(
        id="op_1",
        name="Conv1",
        node_type="op",
        op_type="Conv",
        total_flops=1000000,
        total_memory_bytes=1024 * 1024,
    )
    child2 = HierarchicalNode(
        id="op_2",
        name="Relu1",
        node_type="op",
        op_type="Relu",
    )
    block = HierarchicalNode(
        id="block_1",
        name="ConvBlock",
        node_type="block",
        children=[child1, child2],
        is_collapsed=False,
        attributes={"block_type": "ConvRelu"},
    )
    root = HierarchicalNode(
        id="root",
        name="TestModel",
        node_type="model",
        children=[block],
        is_collapsed=False,
    )

    return HierarchicalGraph(
        root=root,
        nodes_by_id={"root": root, "block_1": block, "op_1": child1, "op_2": child2},
        total_nodes=4,
        depth=2,
    )


def create_test_edge_result() -> EdgeAnalysisResult:
    """Create a test edge analysis result."""
    return EdgeAnalysisResult(
        edges=[],
        total_activation_bytes=10 * 1024 * 1024,
        peak_activation_bytes=5 * 1024 * 1024,
        peak_activation_node="Conv1",
        bottleneck_edges=["conv_out"],
        attention_edges=[],
        skip_connection_edges=[],
        memory_profile=[("Conv1", 5 * 1024 * 1024)],
    )


class TestHTMLGeneration:
    """Tests for HTML generation."""

    def test_generate_html_basic(self):
        """Test basic HTML generation."""
        graph = create_test_graph()
        html = generate_html(graph, title="Test Model")

        # Should contain key elements
        assert "<!DOCTYPE html>" in html
        assert "<title>Test Model - Neural Architecture</title>" in html
        assert "d3.v7.min.js" in html
        assert "graphData" in html

    def test_generate_html_with_edges(self):
        """Test HTML generation with edge data."""
        graph = create_test_graph()
        edge_result = create_test_edge_result()
        html = generate_html(graph, edge_result, title="Test Model")

        assert "edgeData" in html
        assert "peak_activation_bytes" in html

    def test_generate_html_to_file(self):
        """Test HTML generation to file."""
        graph = create_test_graph()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.html"
            html = generate_html(graph, title="Test", output_path=output_path)

            assert output_path.exists()
            content = output_path.read_text(encoding="utf-8")
            assert content == html

    def test_html_contains_controls(self):
        """Test that HTML contains interactive controls."""
        graph = create_test_graph()
        html = generate_html(graph)

        assert "expandAll()" in html
        assert "collapseAll()" in html
        assert "resetZoom()" in html
        assert "fitToScreen()" in html

    def test_html_contains_legend(self):
        """Test that HTML contains op type legend."""
        graph = create_test_graph()
        html = generate_html(graph)

        # New design uses "Op Types" instead of "Legend"
        assert "Op Types" in html
        assert "Convolution" in html
        assert "Attention" in html
        assert "legend-item" in html  # CSS class for legend items

    def test_html_contains_stats_panel(self):
        """Test that HTML contains stats panel."""
        graph = create_test_graph()
        html = generate_html(graph)

        assert "node-count" in html
        assert "edge-count" in html
        assert "peak-memory" in html


class TestHTMLExporter:
    """Tests for HTMLExporter class."""

    def test_export_basic(self):
        """Test basic export."""
        graph = create_test_graph()
        exporter = HTMLExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.html"
            result = exporter.export(graph, output_path=output_path)

            assert result == output_path
            assert output_path.exists()

    def test_export_with_edges(self):
        """Test export with edge data."""
        graph = create_test_graph()
        edge_result = create_test_edge_result()
        exporter = HTMLExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.html"
            exporter.export(graph, edge_result, output_path=output_path)

            content = output_path.read_text(encoding="utf-8")
            assert "peak_activation_bytes" in content

    def test_export_custom_title(self):
        """Test export with custom title."""
        graph = create_test_graph()
        exporter = HTMLExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.html"
            exporter.export(graph, output_path=output_path, title="My Custom Model")

            content = output_path.read_text(encoding="utf-8")
            assert "My Custom Model" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
