# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""Tests for hierarchical graph view."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from ..analyzer import ONNXGraphLoader
from ..hierarchical_graph import (
    HierarchicalGraph,
    HierarchicalGraphBuilder,
    HierarchicalNode,
    generate_summary,
)
from ..patterns import PatternAnalyzer


def create_test_model() -> onnx.ModelProto:
    """Create a simple model for hierarchical testing."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 64, 56, 56])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 64, 56, 56])

    weight1 = helper.make_tensor(
        "w1",
        TensorProto.FLOAT,
        [64, 64, 3, 3],
        np.random.randn(64, 64, 3, 3).astype(np.float32).flatten().tolist(),
    )
    weight2 = helper.make_tensor(
        "w2",
        TensorProto.FLOAT,
        [64, 64, 3, 3],
        np.random.randn(64, 64, 3, 3).astype(np.float32).flatten().tolist(),
    )

    nodes = [
        helper.make_node("Conv", ["X", "w1"], ["c1"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c1"], ["r1"]),
        helper.make_node("Conv", ["r1", "w2"], ["c2"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Add", ["X", "c2"], ["add_out"]),
        helper.make_node("Relu", ["add_out"], ["Y"]),
    ]

    graph = helper.make_graph(nodes, "test_model", [X], [Y], [weight1, weight2])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return model


class TestHierarchicalNode:
    """Tests for HierarchicalNode."""

    def test_basic_creation(self):
        """Test basic node creation."""
        node = HierarchicalNode(
            id="test_1",
            name="TestNode",
            node_type="op",
            op_type="Conv",
        )
        assert node.id == "test_1"
        assert node.name == "TestNode"
        assert node.is_leaf()
        assert node.is_collapsed

    def test_display_name_with_repeat(self):
        """Test xN notation for repeated blocks."""
        node = HierarchicalNode(
            id="block_1",
            name="TransformerLayer",
            node_type="block",
            repeat_count=12,
            is_repeated=True,
        )
        assert node.get_display_name() == "TransformerLayer x12"

    def test_collapse_expand(self):
        """Test collapse/expand state management."""
        parent = HierarchicalNode(
            id="parent",
            name="Parent",
            node_type="block",
            children=[
                HierarchicalNode(id="child1", name="Child1", node_type="op"),
                HierarchicalNode(id="child2", name="Child2", node_type="op"),
            ],
        )

        assert parent.is_collapsed
        parent.expand()
        assert not parent.is_collapsed
        parent.toggle()
        assert parent.is_collapsed

    def test_aggregate_stats(self):
        """Test stats aggregation from children."""
        child1 = HierarchicalNode(
            id="c1",
            name="C1",
            node_type="op",
            total_flops=1000,
            total_params=100,
            total_memory_bytes=400,
        )
        child2 = HierarchicalNode(
            id="c2",
            name="C2",
            node_type="op",
            total_flops=2000,
            total_params=200,
            total_memory_bytes=800,
        )
        parent = HierarchicalNode(
            id="parent",
            name="Parent",
            node_type="block",
            children=[child1, child2],
        )

        parent.aggregate_stats()

        assert parent.total_flops == 3000
        assert parent.total_params == 300
        assert parent.total_memory_bytes == 1200
        assert parent.node_count == 2

    def test_aggregate_with_repeat(self):
        """Test stats aggregation with repeat multiplier."""
        child = HierarchicalNode(
            id="c1",
            name="C1",
            node_type="op",
            total_flops=1000,
            total_params=100,
            total_memory_bytes=400,
        )
        parent = HierarchicalNode(
            id="parent",
            name="Parent",
            node_type="block",
            children=[child],
            repeat_count=12,
        )

        parent.aggregate_stats()

        assert parent.total_flops == 12000  # Multiplied
        assert parent.total_params == 100  # Shared, not multiplied
        assert parent.total_memory_bytes == 4800  # Multiplied

    def test_to_dict(self):
        """Test JSON export."""
        node = HierarchicalNode(
            id="test",
            name="TestNode",
            node_type="op",
            op_type="Conv",
        )
        d = node.to_dict()

        assert d["id"] == "test"
        assert d["name"] == "TestNode"
        assert d["op_type"] == "Conv"
        assert "children" not in d  # No children for leaf


class TestHierarchicalGraph:
    """Tests for HierarchicalGraph."""

    def test_graph_creation(self):
        """Test creating a hierarchical graph."""
        root = HierarchicalNode(id="root", name="Model", node_type="model")
        graph = HierarchicalGraph(root=root, nodes_by_id={"root": root})

        assert graph.root.name == "Model"
        assert graph.get_node("root") == root

    def test_visible_nodes_collapsed(self):
        """Test visible nodes when parent is collapsed."""
        child = HierarchicalNode(id="child", name="Child", node_type="op")
        root = HierarchicalNode(
            id="root",
            name="Root",
            node_type="model",
            children=[child],
            is_collapsed=True,
        )
        graph = HierarchicalGraph(
            root=root,
            nodes_by_id={"root": root, "child": child},
        )

        visible = graph.get_visible_nodes()
        assert len(visible) == 1
        assert visible[0].id == "root"

    def test_visible_nodes_expanded(self):
        """Test visible nodes when parent is expanded."""
        child = HierarchicalNode(id="child", name="Child", node_type="op")
        root = HierarchicalNode(
            id="root",
            name="Root",
            node_type="model",
            children=[child],
            is_collapsed=False,
        )
        graph = HierarchicalGraph(
            root=root,
            nodes_by_id={"root": root, "child": child},
        )

        visible = graph.get_visible_nodes()
        assert len(visible) == 2

    def test_expand_to_depth(self):
        """Test expanding to a specific depth."""
        grandchild = HierarchicalNode(id="gc", name="GC", node_type="op", depth=2)
        child = HierarchicalNode(
            id="child",
            name="Child",
            node_type="block",
            children=[grandchild],
            depth=1,
        )
        root = HierarchicalNode(
            id="root",
            name="Root",
            node_type="model",
            children=[child],
            depth=0,
        )
        graph = HierarchicalGraph(
            root=root,
            nodes_by_id={"root": root, "child": child, "gc": grandchild},
        )

        graph.expand_to_depth(1)

        assert not root.is_collapsed
        assert not child.is_collapsed
        assert grandchild.is_collapsed

    def test_to_json(self):
        """Test JSON export."""
        root = HierarchicalNode(id="root", name="Model", node_type="model")
        graph = HierarchicalGraph(root=root, total_nodes=1, depth=0)

        json_str = graph.to_json()
        data = json.loads(json_str)

        assert data["root"]["name"] == "Model"
        assert data["total_nodes"] == 1


class TestHierarchicalGraphBuilder:
    """Tests for building hierarchical graphs from ONNX."""

    def test_build_from_onnx(self):
        """Test building hierarchy from ONNX model."""
        model = create_test_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            pattern_analyzer = PatternAnalyzer()
            blocks = pattern_analyzer.group_into_blocks(graph_info)

            builder = HierarchicalGraphBuilder()
            graph = builder.build(graph_info, blocks, "TestModel")

            assert graph.root.name == "TestModel"
            assert graph.total_nodes > 0
            assert len(graph.root.children) > 0
        finally:
            model_path.unlink()


class TestSummaryGeneration:
    """Tests for summary generation."""

    def test_generate_summary(self):
        """Test multi-level summary generation."""
        child1 = HierarchicalNode(
            id="c1",
            name="ConvBlock",
            node_type="block",
            attributes={"block_type": "ConvRelu"},
        )
        child2 = HierarchicalNode(
            id="c2",
            name="Layer",
            node_type="layer",
            is_repeated=True,
            repeat_count=12,
        )
        root = HierarchicalNode(
            id="root",
            name="Model",
            node_type="model",
            children=[child1, child2],
            is_collapsed=False,
        )
        graph = HierarchicalGraph(root=root, total_nodes=3, depth=1)

        summary = generate_summary(graph)

        assert "Model" in summary
        assert "ConvRelu" in summary
        assert "x12" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
