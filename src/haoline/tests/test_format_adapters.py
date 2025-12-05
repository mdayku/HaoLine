# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""Tests for format adapter system."""

from pathlib import Path

import numpy as np
import pytest

from haoline.format_adapters import (
    ConversionLevel,
    FormatAdapter,
    OnnxAdapter,
    PyTorchAdapter,
    can_convert,
    get_adapter,
    get_conversion_level,
    list_adapters,
    list_conversion_paths,
    map_onnx_op_to_universal,
    register_adapter,
)
from haoline.universal_ir import (
    DataType,
    GraphMetadata,
    SourceFormat,
    TensorOrigin,
    UniversalGraph,
    UniversalNode,
    UniversalTensor,
)


class TestAdapterRegistry:
    """Tests for adapter registration and lookup."""

    def test_list_adapters_includes_onnx(self) -> None:
        """ONNX adapter should be registered by default."""
        adapters = list_adapters()
        names = [a["name"] for a in adapters]
        assert "onnx" in names

    def test_list_adapters_includes_pytorch(self) -> None:
        """PyTorch adapter should be registered by default."""
        adapters = list_adapters()
        names = [a["name"] for a in adapters]
        assert "pytorch" in names

    def test_get_adapter_onnx(self) -> None:
        """get_adapter should return OnnxAdapter for .onnx files."""
        adapter = get_adapter("model.onnx")
        assert isinstance(adapter, OnnxAdapter)

    def test_get_adapter_pytorch(self) -> None:
        """get_adapter should return PyTorchAdapter for .pt files."""
        adapter = get_adapter("model.pt")
        assert isinstance(adapter, PyTorchAdapter)

    def test_get_adapter_unknown_extension_raises(self) -> None:
        """get_adapter should raise ValueError for unknown extensions."""
        with pytest.raises(ValueError, match="No adapter registered"):
            get_adapter("model.unknown")

    def test_register_custom_adapter(self) -> None:
        """Custom adapters can be registered."""

        @register_adapter
        class TestAdapter(FormatAdapter):
            name = "test_format"
            extensions = [".testfmt"]
            source_format = SourceFormat.UNKNOWN

            def can_read(self, path: Path) -> bool:
                return path.suffix.lower() == ".testfmt"

            def read(self, path: Path) -> UniversalGraph:
                return UniversalGraph()

        # Should be able to get the adapter
        adapter = get_adapter("model.testfmt")
        assert adapter.name == "test_format"


class TestOnnxAdapter:
    """Tests for ONNX adapter."""

    def test_can_read_onnx_file(self) -> None:
        """OnnxAdapter.can_read should return True for .onnx files."""
        adapter = OnnxAdapter()
        assert adapter.can_read(Path("model.onnx")) is True
        assert adapter.can_read(Path("model.pt")) is False

    def test_can_write(self) -> None:
        """OnnxAdapter should support writing."""
        adapter = OnnxAdapter()
        assert adapter.can_write() is True

    def test_read_nonexistent_file_raises(self) -> None:
        """Reading non-existent file should raise FileNotFoundError."""
        adapter = OnnxAdapter()
        with pytest.raises(FileNotFoundError):
            adapter.read(Path("nonexistent_model.onnx"))

    @pytest.mark.skipif(
        not Path("src/haoline/tests/fixtures").exists(),
        reason="No test fixtures available",
    )
    def test_read_simple_onnx_model(self) -> None:
        """Test reading a simple ONNX model."""
        # This test would use a fixture model
        pass


class TestPyTorchAdapter:
    """Tests for PyTorch adapter."""

    def test_can_read_pytorch_files(self) -> None:
        """PyTorchAdapter.can_read should return True for .pt/.pth files."""
        adapter = PyTorchAdapter()
        assert adapter.can_read(Path("model.pt")) is True
        assert adapter.can_read(Path("model.pth")) is True
        assert adapter.can_read(Path("model.onnx")) is False

    def test_cannot_write(self) -> None:
        """PyTorchAdapter should not support direct writing."""
        adapter = PyTorchAdapter()
        assert adapter.can_write() is False


class TestOpMapping:
    """Tests for ONNX to Universal op mapping."""

    def test_conv_mapping(self) -> None:
        """ONNX Conv should map to Conv2D."""
        assert map_onnx_op_to_universal("Conv") == "Conv2D"

    def test_gemm_mapping(self) -> None:
        """ONNX Gemm should map to MatMul."""
        assert map_onnx_op_to_universal("Gemm") == "MatMul"

    def test_relu_mapping(self) -> None:
        """ONNX Relu should map to Relu."""
        assert map_onnx_op_to_universal("Relu") == "Relu"

    def test_unknown_op_passthrough(self) -> None:
        """Unknown ops should pass through unchanged."""
        assert map_onnx_op_to_universal("CustomOp") == "CustomOp"

    def test_pooling_mapping(self) -> None:
        """Pooling ops should map correctly."""
        assert map_onnx_op_to_universal("MaxPool") == "MaxPool2D"
        assert map_onnx_op_to_universal("AveragePool") == "AvgPool2D"
        assert map_onnx_op_to_universal("GlobalAveragePool") == "GlobalAvgPool"


class TestUniversalGraphCreation:
    """Tests for creating UniversalGraph manually."""

    def test_create_empty_graph(self) -> None:
        """Should be able to create an empty graph."""
        graph = UniversalGraph()
        assert graph.num_nodes == 0
        assert graph.num_tensors == 0

    def test_create_graph_with_metadata(self) -> None:
        """Should be able to create graph with metadata."""
        metadata = GraphMetadata(
            name="test_model",
            source_format=SourceFormat.ONNX,
            opset_version=17,
        )
        graph = UniversalGraph(metadata=metadata)
        assert graph.metadata.name == "test_model"
        assert graph.metadata.source_format == SourceFormat.ONNX

    def test_create_node(self) -> None:
        """Should be able to create a node."""
        node = UniversalNode(
            id="conv1",
            op_type="Conv2D",
            inputs=["input", "weight"],
            outputs=["output"],
            attributes={"kernel_shape": [3, 3]},
        )
        assert node.id == "conv1"
        assert node.op_type == "Conv2D"
        assert node.is_compute_op is True

    def test_create_tensor(self) -> None:
        """Should be able to create a tensor."""
        tensor = UniversalTensor(
            name="weight",
            shape=[64, 3, 3, 3],
            dtype=DataType.FLOAT32,
            origin=TensorOrigin.WEIGHT,
        )
        assert tensor.name == "weight"
        assert tensor.num_elements == 64 * 3 * 3 * 3
        assert tensor.size_bytes == 64 * 3 * 3 * 3 * 4

    def test_graph_with_nodes_and_tensors(self) -> None:
        """Should be able to create a complete graph."""
        weight_data = np.random.randn(64, 3, 3, 3).astype(np.float32)

        tensors = {
            "input": UniversalTensor(
                name="input",
                shape=[1, 3, 224, 224],
                dtype=DataType.FLOAT32,
                origin=TensorOrigin.INPUT,
            ),
            "conv1.weight": UniversalTensor(
                name="conv1.weight",
                shape=[64, 3, 3, 3],
                dtype=DataType.FLOAT32,
                origin=TensorOrigin.WEIGHT,
                data=weight_data,
            ),
            "output": UniversalTensor(
                name="output",
                shape=[1, 64, 222, 222],
                dtype=DataType.FLOAT32,
                origin=TensorOrigin.OUTPUT,
            ),
        }

        nodes = [
            UniversalNode(
                id="conv1",
                op_type="Conv2D",
                inputs=["input", "conv1.weight"],
                outputs=["output"],
                attributes={"kernel_shape": [3, 3]},
            )
        ]

        graph = UniversalGraph(
            nodes=nodes,
            tensors=tensors,
            metadata=GraphMetadata(name="simple_conv"),
        )

        assert graph.num_nodes == 1
        assert graph.num_tensors == 3
        assert graph.total_parameters == 64 * 3 * 3 * 3
        assert len(graph.weight_tensors) == 1
        assert len(graph.input_tensors) == 1
        assert len(graph.output_tensors) == 1


class TestGraphComparison:
    """Tests for graph structural comparison."""

    def test_empty_graphs_equal(self) -> None:
        """Two empty graphs should be structurally equal."""
        g1 = UniversalGraph()
        g2 = UniversalGraph()
        assert g1.is_structurally_equal(g2) is True

    def test_different_node_count_not_equal(self) -> None:
        """Graphs with different node counts are not equal."""
        g1 = UniversalGraph(nodes=[UniversalNode(id="n1", op_type="Conv2D")])
        g2 = UniversalGraph()
        assert g1.is_structurally_equal(g2) is False

    def test_same_structure_equal(self) -> None:
        """Graphs with same structure should be equal."""
        g1 = UniversalGraph(
            nodes=[
                UniversalNode(id="conv1", op_type="Conv2D", inputs=["a"], outputs=["b"]),
                UniversalNode(id="relu1", op_type="Relu", inputs=["b"], outputs=["c"]),
            ]
        )
        g2 = UniversalGraph(
            nodes=[
                UniversalNode(id="c1", op_type="Conv2D", inputs=["x"], outputs=["y"]),
                UniversalNode(id="r1", op_type="Relu", inputs=["y"], outputs=["z"]),
            ]
        )
        assert g1.is_structurally_equal(g2) is True

    def test_different_op_types_not_equal(self) -> None:
        """Graphs with different op types are not equal."""
        g1 = UniversalGraph(nodes=[UniversalNode(id="n1", op_type="Conv2D")])
        g2 = UniversalGraph(nodes=[UniversalNode(id="n1", op_type="MatMul")])
        assert g1.is_structurally_equal(g2) is False


class TestGraphDiff:
    """Tests for graph diff functionality."""

    def test_diff_empty_graphs(self) -> None:
        """Diff of two empty graphs."""
        g1 = UniversalGraph()
        g2 = UniversalGraph()
        diff = g1.diff(g2)
        assert diff["structurally_equal"] is True
        assert diff["node_count_diff"] == (0, 0)

    def test_diff_shows_node_count(self) -> None:
        """Diff should show node count difference."""
        g1 = UniversalGraph(nodes=[UniversalNode(id="n1", op_type="Conv2D")])
        g2 = UniversalGraph()
        diff = g1.diff(g2)
        assert diff["structurally_equal"] is False
        assert diff["node_count_diff"] == (1, 0)

    def test_diff_shows_op_type_diff(self) -> None:
        """Diff should show op type count differences."""
        g1 = UniversalGraph(
            nodes=[
                UniversalNode(id="c1", op_type="Conv2D"),
                UniversalNode(id="c2", op_type="Conv2D"),
            ]
        )
        g2 = UniversalGraph(
            nodes=[
                UniversalNode(id="c1", op_type="Conv2D"),
            ]
        )
        diff = g1.diff(g2)
        assert "Conv2D" in diff["op_type_diff"]
        assert diff["op_type_diff"]["Conv2D"] == (2, 1)


class TestGraphSerialization:
    """Tests for graph JSON serialization."""

    def test_to_dict_empty_graph(self) -> None:
        """Empty graph should serialize to dict."""
        graph = UniversalGraph()
        data = graph.to_dict()
        assert "metadata" in data
        assert "nodes" in data
        assert "tensors" in data
        assert "summary" in data

    def test_to_dict_with_tensors(self) -> None:
        """Graph with tensors should serialize correctly."""
        graph = UniversalGraph(
            tensors={
                "w": UniversalTensor(
                    name="w",
                    shape=[3, 3],
                    dtype=DataType.FLOAT32,
                    origin=TensorOrigin.WEIGHT,
                    data=np.ones((3, 3), dtype=np.float32),
                )
            }
        )
        data = graph.to_dict(include_weights=False)
        # Weight data should be stripped
        assert data["tensors"]["w"]["data"] is None

    def test_round_trip_json(self, tmp_path: Path) -> None:
        """Graph should survive JSON round-trip."""
        original = UniversalGraph(
            nodes=[UniversalNode(id="n1", op_type="Conv2D", inputs=["a"], outputs=["b"])],
            tensors={
                "a": UniversalTensor(name="a", shape=[1, 3, 224, 224], origin=TensorOrigin.INPUT)
            },
            metadata=GraphMetadata(name="test", source_format=SourceFormat.ONNX),
        )

        json_path = tmp_path / "graph.json"
        original.to_json(json_path)

        loaded = UniversalGraph.from_json(json_path)
        assert loaded.num_nodes == 1
        assert loaded.metadata.name == "test"
        assert loaded.is_structurally_equal(original)


class TestDataType:
    """Tests for DataType enum."""

    def test_bytes_per_element(self) -> None:
        """DataType should report correct bytes per element."""
        assert DataType.FLOAT32.bytes_per_element == 4
        assert DataType.FLOAT16.bytes_per_element == 2
        assert DataType.INT8.bytes_per_element == 1
        assert DataType.FLOAT64.bytes_per_element == 8

    def test_from_numpy_dtype(self) -> None:
        """DataType should convert from numpy dtype."""
        assert DataType.from_numpy_dtype(np.dtype(np.float32)) == DataType.FLOAT32
        assert DataType.from_numpy_dtype(np.dtype(np.float16)) == DataType.FLOAT16
        assert DataType.from_numpy_dtype(np.dtype(np.int8)) == DataType.INT8


class TestConversionMatrix:
    """Tests for conversion matrix functionality."""

    def test_conversion_level_enum(self) -> None:
        """ConversionLevel enum should have expected values."""
        assert ConversionLevel.FULL.value == "full"
        assert ConversionLevel.PARTIAL.value == "partial"
        assert ConversionLevel.LOSSY.value == "lossy"
        assert ConversionLevel.NONE.value == "none"

    def test_identity_conversion(self) -> None:
        """Converting to same format should be FULL."""
        assert get_conversion_level(SourceFormat.ONNX, SourceFormat.ONNX) == ConversionLevel.FULL
        assert get_conversion_level("pytorch", "pytorch") == ConversionLevel.FULL

    def test_pytorch_to_onnx(self) -> None:
        """PyTorch to ONNX should be FULL."""
        level = get_conversion_level(SourceFormat.PYTORCH, SourceFormat.ONNX)
        assert level == ConversionLevel.FULL

    def test_onnx_to_tensorrt(self) -> None:
        """ONNX to TensorRT should be PARTIAL."""
        level = get_conversion_level(SourceFormat.ONNX, SourceFormat.TENSORRT)
        assert level == ConversionLevel.PARTIAL

    def test_tensorrt_to_onnx(self) -> None:
        """TensorRT to ONNX should be NONE (no export)."""
        level = get_conversion_level(SourceFormat.TENSORRT, SourceFormat.ONNX)
        assert level == ConversionLevel.NONE

    def test_unknown_conversion(self) -> None:
        """Unknown conversions should return NONE."""
        level = get_conversion_level(SourceFormat.GGUF, SourceFormat.TENSORRT)
        assert level == ConversionLevel.NONE

    def test_string_format_input(self) -> None:
        """Should accept string format names."""
        level = get_conversion_level("onnx", "openvino")
        assert level == ConversionLevel.FULL

    def test_can_convert_true(self) -> None:
        """can_convert should return True for valid conversions."""
        assert can_convert("pytorch", "onnx") is True
        assert can_convert("onnx", "tflite") is True

    def test_can_convert_false(self) -> None:
        """can_convert should return False for impossible conversions."""
        assert can_convert("safetensors", "onnx") is False
        assert can_convert("tensorrt", "onnx") is False

    def test_list_conversion_paths(self) -> None:
        """list_conversion_paths should return available conversions."""
        paths = list_conversion_paths()
        assert len(paths) > 0
        # Each path should have source, target, level
        for path in paths:
            assert "source" in path
            assert "target" in path
            assert "level" in path

    def test_list_conversion_paths_filtered_source(self) -> None:
        """list_conversion_paths should filter by source."""
        paths = list_conversion_paths(source="onnx")
        assert all(p["source"] == "onnx" for p in paths)

    def test_list_conversion_paths_filtered_target(self) -> None:
        """list_conversion_paths should filter by target."""
        paths = list_conversion_paths(target="onnx")
        assert all(p["target"] == "onnx" for p in paths)
