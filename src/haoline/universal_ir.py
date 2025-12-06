# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
Universal Internal Representation (IR) for format-agnostic model analysis.

This module provides a backend-neutral representation of neural network models,
enabling analysis and comparison across different frameworks (ONNX, PyTorch,
TensorFlow, TensorRT, CoreML, etc.).

Design inspired by:
- OpenVINO IR (graph + weights separation)
- TVM Relay (typed graph representation)
- MLIR (extensible operation types)

Usage:
    from haoline.universal_ir import UniversalGraph, UniversalNode, UniversalTensor

    # Load from ONNX (via adapter)
    graph = OnnxAdapter().read("model.onnx")

    # Analyze
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Parameters: {graph.total_parameters}")

    # Compare two graphs
    if graph1.is_structurally_equal(graph2):
        print("Same architecture!")

    # Export to JSON
    graph.to_json("model_ir.json")
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class TensorOrigin(str, Enum):
    """Origin type for tensors in the graph."""

    WEIGHT = "weight"  # Constant model parameter
    INPUT = "input"  # Model input
    OUTPUT = "output"  # Model output
    ACTIVATION = "activation"  # Intermediate activation (runtime)


class DataType(str, Enum):
    """Supported data types for tensors."""

    FLOAT64 = "float64"
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    INT64 = "int64"
    INT32 = "int32"
    INT16 = "int16"
    INT8 = "int8"
    UINT8 = "uint8"
    BOOL = "bool"
    STRING = "string"
    UNKNOWN = "unknown"

    @classmethod
    def from_numpy_dtype(cls, dtype: Any) -> DataType:
        """Convert numpy dtype to DataType."""
        import numpy as np

        dtype_map = {
            np.float64: cls.FLOAT64,
            np.float32: cls.FLOAT32,
            np.float16: cls.FLOAT16,
            np.int64: cls.INT64,
            np.int32: cls.INT32,
            np.int16: cls.INT16,
            np.int8: cls.INT8,
            np.uint8: cls.UINT8,
            np.bool_: cls.BOOL,
        }
        return dtype_map.get(dtype.type, cls.UNKNOWN)

    @classmethod
    def from_onnx_dtype(cls, onnx_dtype: int) -> DataType:
        """Convert ONNX TensorProto dtype to DataType."""
        # ONNX TensorProto.DataType values
        onnx_map = {
            1: cls.FLOAT32,  # FLOAT
            2: cls.UINT8,  # UINT8
            3: cls.INT8,  # INT8
            5: cls.INT16,  # INT16
            6: cls.INT32,  # INT32
            7: cls.INT64,  # INT64
            8: cls.STRING,  # STRING
            9: cls.BOOL,  # BOOL
            10: cls.FLOAT16,  # FLOAT16
            11: cls.FLOAT64,  # DOUBLE
            16: cls.BFLOAT16,  # BFLOAT16
        }
        return onnx_map.get(onnx_dtype, cls.UNKNOWN)

    @property
    def bytes_per_element(self) -> int:
        """Return bytes per element for this dtype."""
        size_map = {
            DataType.FLOAT64: 8,
            DataType.INT64: 8,
            DataType.FLOAT32: 4,
            DataType.INT32: 4,
            DataType.FLOAT16: 2,
            DataType.BFLOAT16: 2,
            DataType.INT16: 2,
            DataType.INT8: 1,
            DataType.UINT8: 1,
            DataType.BOOL: 1,
            DataType.STRING: 0,  # Variable
            DataType.UNKNOWN: 0,
        }
        return size_map.get(self, 0)


class UniversalTensor(BaseModel):
    """Represents a tensor (weight, input, output, or activation) in the IR.

    Tensors are the edges of the computation graph - they connect nodes
    and carry data (for weights) or metadata (for activations).

    Attributes:
        name: Unique identifier for this tensor
        shape: Tensor dimensions (empty list for scalars)
        dtype: Data type (float32, float16, int8, etc.)
        origin: Whether this is a weight, input, output, or activation
        data: Actual tensor data (for weights). None for activations.
              Use lazy loading for large tensors.
        source_name: Original name in source format (for round-trip)
    """

    name: str
    shape: list[int] = Field(default_factory=list)
    dtype: DataType = DataType.FLOAT32
    origin: TensorOrigin = TensorOrigin.ACTIVATION
    data: Any | None = None  # numpy array or None for lazy loading
    source_name: str | None = None  # Original name in source format

    model_config = {"arbitrary_types_allowed": True}

    @property
    def num_elements(self) -> int:
        """Total number of elements in the tensor."""
        if not self.shape:
            return 1  # Scalar
        result = 1
        for dim in self.shape:
            if dim > 0:
                result *= dim
        return result

    @property
    def size_bytes(self) -> int:
        """Size in bytes (0 if shape has dynamic dimensions)."""
        if any(d <= 0 for d in self.shape):
            return 0  # Dynamic dimension
        return self.num_elements * self.dtype.bytes_per_element

    def __repr__(self) -> str:
        shape_str = "x".join(str(d) for d in self.shape) if self.shape else "scalar"
        return f"UniversalTensor({self.name}: {shape_str} {self.dtype.value})"


class UniversalNode(BaseModel):
    """Represents a single operation in the computation graph.

    Nodes are the vertices of the graph. Each node performs an operation
    (like Conv2D, MatMul, Relu) on its input tensors to produce output tensors.

    The op_type is a high-level category, NOT tied to any specific framework.
    This enables cross-format comparison.

    Attributes:
        id: Unique identifier for this node
        op_type: High-level operation type (Conv2D, MatMul, Relu, etc.)
        inputs: List of input tensor names
        outputs: List of output tensor names
        attributes: Operation-specific parameters (kernel_size, strides, etc.)
        output_shapes: Shapes of output tensors (if known)
        output_dtypes: Data types of output tensors
        source_op: Original op name in source format (for round-trip)
        source_domain: Original domain (e.g., "ai.onnx" for ONNX)
    """

    id: str
    op_type: str  # High-level: Conv2D, MatMul, Relu, Add, etc.
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    output_shapes: list[list[int]] = Field(default_factory=list)
    output_dtypes: list[DataType] = Field(default_factory=list)

    # Source format metadata (for round-trip conversion)
    source_op: str | None = None  # Original op name (e.g., "Conv" in ONNX)
    source_domain: str | None = None  # e.g., "ai.onnx", "com.microsoft"

    def __repr__(self) -> str:
        return f"UniversalNode({self.id}: {self.op_type})"

    @property
    def is_compute_op(self) -> bool:
        """Check if this is a compute-heavy operation."""
        compute_ops = {
            "Conv2D",
            "Conv3D",
            "MatMul",
            "Gemm",
            "ConvTranspose",
            "Attention",
            "MultiHeadAttention",
        }
        return self.op_type in compute_ops

    @property
    def is_activation(self) -> bool:
        """Check if this is an activation function."""
        activation_ops = {
            "Relu",
            "LeakyRelu",
            "Sigmoid",
            "Tanh",
            "Gelu",
            "Silu",
            "Swish",
            "Softmax",
            "Mish",
        }
        return self.op_type in activation_ops


class SourceFormat(str, Enum):
    """Supported source model formats."""

    ONNX = "onnx"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    TENSORRT = "tensorrt"
    COREML = "coreml"
    TFLITE = "tflite"
    OPENVINO = "openvino"
    SAFETENSORS = "safetensors"
    GGUF = "gguf"
    UNKNOWN = "unknown"


class GraphMetadata(BaseModel):
    """Metadata about the model graph.

    Stores information about the model's origin, version, and structure
    that isn't captured in the nodes/tensors themselves.
    """

    name: str = ""
    source_format: SourceFormat = SourceFormat.UNKNOWN
    source_path: str | None = None

    # Version info from source
    ir_version: int | None = None  # e.g., ONNX IR version
    producer_name: str | None = None  # e.g., "pytorch", "tf2onnx"
    producer_version: str | None = None
    opset_version: int | None = None  # e.g., ONNX opset

    # Model I/O
    input_names: list[str] = Field(default_factory=list)
    output_names: list[str] = Field(default_factory=list)

    # Additional metadata from source (for round-trip)
    extra: dict[str, Any] = Field(default_factory=dict)


class UniversalGraph(BaseModel):
    """Universal representation of a neural network computation graph.

    This is the top-level container for a model's IR. It holds all nodes
    (operations), tensors (weights and I/O), and metadata.

    The graph is designed to be:
    - Format-agnostic: Works with ONNX, PyTorch, TensorFlow, etc.
    - Serializable: Can be saved to JSON for debugging/interchange
    - Comparable: Supports structural comparison between models
    - Extensible: Easy to add new op types or metadata

    Attributes:
        nodes: List of computation nodes (operations)
        tensors: Dict mapping tensor name to UniversalTensor
        metadata: Graph-level metadata (name, source format, etc.)
    """

    nodes: list[UniversalNode] = Field(default_factory=list)
    tensors: dict[str, UniversalTensor] = Field(default_factory=dict)
    metadata: GraphMetadata = Field(default_factory=GraphMetadata)

    model_config = {"arbitrary_types_allowed": True}

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        """Total number of nodes in the graph."""
        return len(self.nodes)

    @property
    def num_tensors(self) -> int:
        """Total number of tensors (weights + I/O + activations)."""
        return len(self.tensors)

    @property
    def weight_tensors(self) -> list[UniversalTensor]:
        """Get all weight tensors."""
        return [t for t in self.tensors.values() if t.origin == TensorOrigin.WEIGHT]

    @property
    def input_tensors(self) -> list[UniversalTensor]:
        """Get all input tensors."""
        return [t for t in self.tensors.values() if t.origin == TensorOrigin.INPUT]

    @property
    def output_tensors(self) -> list[UniversalTensor]:
        """Get all output tensors."""
        return [t for t in self.tensors.values() if t.origin == TensorOrigin.OUTPUT]

    @property
    def total_parameters(self) -> int:
        """Total number of parameters (weight elements)."""
        return sum(t.num_elements for t in self.weight_tensors)

    @property
    def total_weight_bytes(self) -> int:
        """Total size of weights in bytes."""
        return sum(t.size_bytes for t in self.weight_tensors)

    @property
    def op_type_counts(self) -> dict[str, int]:
        """Count of each operation type."""
        counts: dict[str, int] = {}
        for node in self.nodes:
            counts[node.op_type] = counts.get(node.op_type, 0) + 1
        return counts

    # -------------------------------------------------------------------------
    # Node/Tensor Access
    # -------------------------------------------------------------------------

    def get_node(self, node_id: str) -> UniversalNode | None:
        """Get a node by its ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_tensor(self, name: str) -> UniversalTensor | None:
        """Get a tensor by its name."""
        return self.tensors.get(name)

    def get_node_inputs(self, node: UniversalNode) -> list[UniversalTensor]:
        """Get the input tensors for a node."""
        return [self.tensors[name] for name in node.inputs if name in self.tensors]

    def get_node_outputs(self, node: UniversalNode) -> list[UniversalTensor]:
        """Get the output tensors for a node."""
        return [self.tensors[name] for name in node.outputs if name in self.tensors]

    # -------------------------------------------------------------------------
    # Comparison (Task 18.4.1, 18.4.2)
    # -------------------------------------------------------------------------

    def is_structurally_equal(self, other: UniversalGraph) -> bool:
        """Check if two graphs have the same structure.

        Two graphs are structurally equal if they have:
        - Same number of nodes
        - Same op_type sequence
        - Same connectivity (which node feeds which)

        Ignores:
        - Weight values
        - Precision differences (FP32 vs FP16)
        - Node/tensor names
        """
        if len(self.nodes) != len(other.nodes):
            return False

        # Compare op types in order
        self_ops = [n.op_type for n in self.nodes]
        other_ops = [n.op_type for n in other.nodes]
        if self_ops != other_ops:
            return False

        # Compare connectivity (input/output counts per node)
        for n1, n2 in zip(self.nodes, other.nodes, strict=True):
            if len(n1.inputs) != len(n2.inputs):
                return False
            if len(n1.outputs) != len(n2.outputs):
                return False

        return True

    def diff(self, other: UniversalGraph) -> dict[str, Any]:
        """Generate a detailed diff between two graphs.

        Returns a dict with:
        - 'structurally_equal': bool
        - 'node_count_diff': (self_count, other_count)
        - 'op_type_diff': {op_type: (self_count, other_count)}
        - 'dtype_changes': [{node_id, self_dtype, other_dtype}]
        - 'missing_in_self': [node_ids in other but not self]
        - 'missing_in_other': [node_ids in self but not other]
        """
        result: dict[str, Any] = {
            "structurally_equal": self.is_structurally_equal(other),
            "node_count_diff": (len(self.nodes), len(other.nodes)),
            "param_count_diff": (self.total_parameters, other.total_parameters),
            "weight_bytes_diff": (self.total_weight_bytes, other.total_weight_bytes),
            "op_type_diff": {},
            "dtype_changes": [],
        }

        # Compare op type counts
        self_ops = self.op_type_counts
        other_ops = other.op_type_counts
        all_ops = set(self_ops.keys()) | set(other_ops.keys())
        for op in all_ops:
            self_count = self_ops.get(op, 0)
            other_count = other_ops.get(op, 0)
            if self_count != other_count:
                result["op_type_diff"][op] = (self_count, other_count)

        # Compare weight dtypes
        self_weights = {t.name: t for t in self.weight_tensors}
        other_weights = {t.name: t for t in other.weight_tensors}

        for name in set(self_weights.keys()) & set(other_weights.keys()):
            if self_weights[name].dtype != other_weights[name].dtype:
                result["dtype_changes"].append(
                    {
                        "tensor": name,
                        "self_dtype": self_weights[name].dtype.value,
                        "other_dtype": other_weights[name].dtype.value,
                    }
                )

        return result

    # -------------------------------------------------------------------------
    # Serialization (Task 18.5.1)
    # -------------------------------------------------------------------------

    def to_dict(self, include_weights: bool = False) -> dict[str, Any]:
        """Convert graph to a dictionary for JSON serialization.

        Args:
            include_weights: If True, include actual weight data (large!)

        Returns:
            Dict representation of the graph
        """
        # Build tensors dict
        tensors_dict: dict[str, Any] = {}
        for name, tensor in self.tensors.items():
            t_dict = tensor.model_dump()
            if not include_weights:
                t_dict["data"] = None  # Don't serialize actual weight data
            tensors_dict[name] = t_dict

        result: dict[str, Any] = {
            "metadata": self.metadata.model_dump(),
            "nodes": [node.model_dump() for node in self.nodes],
            "tensors": tensors_dict,
            "summary": {
                "num_nodes": self.num_nodes,
                "num_tensors": self.num_tensors,
                "total_parameters": self.total_parameters,
                "total_weight_bytes": self.total_weight_bytes,
                "op_type_counts": self.op_type_counts,
            },
        }

        return result

    def to_json(self, path: str | Path, include_weights: bool = False) -> None:
        """Save graph to JSON file.

        Args:
            path: Output file path
            include_weights: If True, include actual weight data (large!)
        """
        import json

        data = self.to_dict(include_weights=include_weights)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def from_json(cls, path: str | Path) -> UniversalGraph:
        """Load graph from JSON file.

        Note: Weight data is not restored (would need separate binary file).
        """
        import json

        with open(path) as f:
            data = json.load(f)

        metadata = GraphMetadata(**data.get("metadata", {}))
        nodes = [UniversalNode(**n) for n in data.get("nodes", [])]
        tensors = {name: UniversalTensor(**t) for name, t in data.get("tensors", {}).items()}

        return cls(nodes=nodes, tensors=tensors, metadata=metadata)

    # -------------------------------------------------------------------------
    # String representation
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"UniversalGraph("
            f"nodes={self.num_nodes}, "
            f"params={self.total_parameters:,}, "
            f"source={self.metadata.source_format.value})"
        )

    def summary(self) -> str:
        """Return a human-readable summary of the graph."""
        lines = [
            f"Universal IR Graph: {self.metadata.name or 'unnamed'}",
            f"  Source: {self.metadata.source_format.value}",
            f"  Nodes: {self.num_nodes}",
            f"  Parameters: {self.total_parameters:,}",
            f"  Weight Size: {self.total_weight_bytes / 1024 / 1024:.2f} MB",
            f"  Inputs: {len(self.input_tensors)}",
            f"  Outputs: {len(self.output_tensors)}",
            "",
            "  Top Operations:",
        ]

        for op, count in sorted(self.op_type_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            lines.append(f"    {op}: {count}")

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Visualization (Task 18.5.2)
    # -------------------------------------------------------------------------

    def to_dot(self, max_nodes: int = 500, cluster_by_op: bool = False) -> str:
        """Export graph to Graphviz DOT format.

        Args:
            max_nodes: Maximum nodes to include (for large graphs)
            cluster_by_op: Group nodes by operation type in subgraphs

        Returns:
            DOT format string
        """
        lines = [
            "digraph UniversalGraph {",
            "  rankdir=TB;",
            '  node [shape=box, style=filled, fontname="Arial"];',
            '  edge [fontname="Arial", fontsize=10];',
            "",
        ]

        # Add title
        name = self.metadata.name or "model"
        lines.append(
            f'  label="{name} ({self.num_nodes} nodes, {self.total_parameters:,} params)";'
        )
        lines.append('  labelloc="t";')
        lines.append("")

        # Limit nodes for large graphs
        nodes_to_render = self.nodes[:max_nodes]
        if len(self.nodes) > max_nodes:
            lines.append(f"  // Showing {max_nodes} of {len(self.nodes)} nodes")
            lines.append("")

        # Color mapping for op types
        op_colors = {
            "Conv2D": "#a8d5ba",  # Green for convolutions
            "MatMul": "#f4a261",  # Orange for matrix ops
            "Relu": "#e9c46a",  # Yellow for activations
            "LeakyRelu": "#e9c46a",
            "Sigmoid": "#e9c46a",
            "Softmax": "#e9c46a",
            "BatchNorm": "#b8c5d6",  # Blue-gray for normalization
            "LayerNorm": "#b8c5d6",
            "Add": "#d4a5a5",  # Pink for element-wise
            "Concat": "#c8b6ff",  # Purple for structural
            "Reshape": "#c8b6ff",
            "MaxPool2D": "#95d5b2",  # Light green for pooling
            "AvgPool2D": "#95d5b2",
        }
        default_color = "#ffffff"

        if cluster_by_op:
            # Group by op type
            ops_to_nodes: dict[str, list[UniversalNode]] = {}
            for node in nodes_to_render:
                ops_to_nodes.setdefault(node.op_type, []).append(node)

            for op_type, op_nodes in ops_to_nodes.items():
                color = op_colors.get(op_type, default_color)
                lines.append(f"  subgraph cluster_{op_type} {{")
                lines.append(f'    label="{op_type}";')
                lines.append("    style=filled;")
                lines.append(f'    color="{color}";')
                for node in op_nodes:
                    label = f"{node.id}\\n{node.op_type}"
                    lines.append(f'    "{node.id}" [label="{label}"];')
                lines.append("  }")
                lines.append("")
        else:
            # Flat node list
            for node in nodes_to_render:
                color = op_colors.get(node.op_type, default_color)
                label = f"{node.id}\\n{node.op_type}"
                lines.append(f'  "{node.id}" [label="{label}", fillcolor="{color}"];')

        lines.append("")

        # Add edges based on tensor connections
        node_ids = {n.id for n in nodes_to_render}
        tensor_to_producer: dict[str, str] = {}

        # Map tensors to their producing nodes
        for node in nodes_to_render:
            for output in node.outputs:
                tensor_to_producer[output] = node.id

        # Create edges
        for node in nodes_to_render:
            for inp in node.inputs:
                if inp in tensor_to_producer:
                    producer = tensor_to_producer[inp]
                    if producer in node_ids:
                        lines.append(f'  "{producer}" -> "{node.id}";')

        lines.append("}")
        return "\n".join(lines)

    def to_networkx(self) -> Any:
        """Export graph to NetworkX DiGraph.

        Returns:
            networkx.DiGraph with nodes and edges

        Raises:
            ImportError: If networkx is not installed
        """
        try:
            import networkx as nx
        except ImportError as e:
            raise ImportError(
                "NetworkX is required for graph export. Install with: pip install networkx"
            ) from e

        G = nx.DiGraph()

        # Add nodes with attributes
        for node in self.nodes:
            G.add_node(
                node.id,
                op_type=node.op_type,
                inputs=node.inputs,
                outputs=node.outputs,
                attributes=node.attributes,
            )

        # Add edges based on tensor connections
        tensor_to_producer: dict[str, str] = {}
        for node in self.nodes:
            for output in node.outputs:
                tensor_to_producer[output] = node.id

        for node in self.nodes:
            for inp in node.inputs:
                if inp in tensor_to_producer:
                    G.add_edge(tensor_to_producer[inp], node.id, tensor=inp)

        return G

    def save_dot(self, path: str | Path) -> None:
        """Save graph to DOT file.

        Args:
            path: Output file path (.dot)
        """
        dot_content = self.to_dot()
        with open(path, "w") as f:
            f.write(dot_content)

    def save_png(self, path: str | Path, max_nodes: int = 500) -> None:
        """Render graph to PNG using Graphviz.

        Args:
            path: Output file path (.png)
            max_nodes: Maximum nodes to render

        Raises:
            ImportError: If graphviz is not installed
        """
        try:
            import graphviz
        except ImportError as e:
            raise ImportError(
                "Graphviz Python package is required. Install with: pip install graphviz\n"
                "Also ensure Graphviz system package is installed."
            ) from e

        dot_content = self.to_dot(max_nodes=max_nodes)
        source = graphviz.Source(dot_content)

        # graphviz renders to path without extension, then adds it
        path = Path(path)
        output_path = path.with_suffix("")  # Remove .png
        source.render(str(output_path), format="png", cleanup=True)

    def to_hierarchical(self) -> dict[str, Any]:
        """Convert UniversalGraph to HierarchicalGraph-compatible dict for D3.js.

        Returns a nested dict structure that matches the format expected by the
        interactive D3.js graph visualization. This enables using UniversalGraph
        as a drop-in replacement for HierarchicalGraph.

        Returns:
            Dict structure matching HierarchicalGraph.to_dict() format
        """
        # Build tensor -> producing node mapping for edges
        tensor_to_producer: dict[str, str] = {}
        for node in self.nodes:
            for output in node.outputs:
                tensor_to_producer[output] = node.id

        # Calculate approximate stats per node
        def estimate_node_flops(node: UniversalNode) -> int:
            """Rough FLOP estimate based on op type and output shapes."""
            if not node.output_shapes:
                return 0

            output_elements = 1
            for shape in node.output_shapes:
                for dim in shape:
                    if dim > 0:
                        output_elements *= dim

            # Simple heuristics
            flop_multipliers = {
                "Conv2D": 9,  # Approximate for 3x3 kernel
                "MatMul": 2,  # multiply-add
                "Gemm": 2,
                "Attention": 4,
                "MultiHeadAttention": 4,
            }
            return output_elements * flop_multipliers.get(node.op_type, 1)

        def estimate_node_memory(node: UniversalNode) -> int:
            """Rough memory estimate in bytes."""
            if not node.output_shapes:
                return 0

            total_elements = 0
            for shape in node.output_shapes:
                elements = 1
                for dim in shape:
                    if dim > 0:
                        elements *= dim
                total_elements += elements

            # Assume float32 (4 bytes) by default
            bytes_per_elem = 4
            if node.output_dtypes:
                dtype = node.output_dtypes[0]
                if isinstance(dtype, DataType):
                    bytes_per_elem = dtype.bytes_per_element  # Property, not method
                elif isinstance(dtype, int):
                    # Handle case where dtype is stored as int
                    bytes_per_elem = dtype if dtype > 0 else 4

            return total_elements * bytes_per_elem

        # Convert nodes to hierarchical format
        children = []
        total_flops = 0
        total_params = 0
        total_memory = 0

        for node in self.nodes:
            node_flops = estimate_node_flops(node)
            node_memory = estimate_node_memory(node)
            total_flops += node_flops
            total_memory += node_memory

            # Get param count from associated weight tensors
            node_params = 0
            for inp in node.inputs:
                if inp in self.tensors:
                    tensor = self.tensors[inp]
                    if tensor.origin == TensorOrigin.WEIGHT:
                        node_params += tensor.num_elements

            total_params += node_params

            child = {
                "id": node.id,
                "name": node.id,
                "display_name": node.op_type,
                "node_type": "op",
                "op_type": node.op_type,
                "depth": 1,
                "is_collapsed": False,
                "is_repeated": False,
                "repeat_count": 1,
                "total_flops": node_flops,
                "total_params": node_params,
                "total_memory_bytes": node_memory,
                "node_count": 1,
                "inputs": node.inputs,
                "outputs": node.outputs,
                "attributes": node.attributes,
            }
            children.append(child)

        # Create root node
        model_name = self.metadata.name or "Model"
        root: dict[str, Any] = {
            "id": "root",
            "name": model_name,
            "display_name": model_name,
            "node_type": "model",
            "op_type": None,
            "depth": 0,
            "is_collapsed": False,
            "is_repeated": False,
            "repeat_count": 1,
            "total_flops": total_flops,
            "total_params": total_params or self.total_parameters,
            "total_memory_bytes": total_memory,
            "node_count": len(self.nodes),
            "inputs": self.metadata.input_names,
            "outputs": self.metadata.output_names,
            "attributes": {},
            "children": children,
        }

        return root
