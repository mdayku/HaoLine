# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
Core analysis engine for HaoLine.

This module provides:
- ONNXGraphLoader: Load ONNX models and extract graph structure
- MetricsEngine: Compute parameters, FLOPs, and memory estimates
- GraphInfo: Internal representation of the parsed graph
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, ClassVar

import numpy as np
import onnx
from pydantic import BaseModel, ConfigDict, Field


# Standalone implementations that work without onnxruntime
def get_opsets_imported(model: onnx.ModelProto) -> dict:
    """Get the opsets imported by the model."""
    opsets = {}
    for entry in model.opset_import:
        domain = entry.domain or "ai.onnx"
        opsets[domain] = entry.version
    return opsets


def iterate_graph_per_node_func(graph, per_node_func, **func_args):
    """Iterate the graph including subgraphs calling the per_node_func for each node."""
    for node in graph.node:
        per_node_func(node, **func_args)
        for attr in node.attribute:
            if attr.HasField("g"):
                iterate_graph_per_node_func(attr.g, per_node_func, **func_args)


# ORT utilities not available in standalone package - use onnx fallback
_HAS_ORT_UTILS = False
ModelProtoWithShapeInfo = None  # type: ignore


class NodeInfo(BaseModel):
    """Information about a single ONNX node."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    op_type: str
    domain: str
    inputs: list[str]
    outputs: list[str]
    attributes: dict[str, Any]
    # Computed during analysis
    param_count: float = 0.0  # Float for fractional shared weight attribution
    flops: int = 0


class GraphInfo(BaseModel):
    """Parsed graph structure with extracted metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    nodes: list[NodeInfo]
    inputs: list[str]
    outputs: list[str]
    initializers: dict[str, Any]  # name -> np.ndarray (Any for Pydantic compat)
    value_shapes: dict[str, list[int | str]]  # name -> shape (may have symbolic dims)

    # Computed summaries
    num_nodes: int = 0
    input_shapes: dict[str, list[int | str]] = Field(default_factory=dict)
    output_shapes: dict[str, list[int | str]] = Field(default_factory=dict)
    op_type_counts: dict[str, int] = Field(default_factory=dict)

    # Node lookup
    node_by_name: dict[str, NodeInfo] = Field(default_factory=dict)
    node_by_output: dict[str, NodeInfo] = Field(default_factory=dict)


class ParamCounts(BaseModel):
    """Parameter count breakdown."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    total: int = 0
    trainable: int = 0  # Assumed: all initializers are trainable unless marked
    non_trainable: int = 0
    by_node: dict[str, float] = Field(
        default_factory=dict
    )  # Float for fractional shared attribution
    by_op_type: dict[str, float] = Field(
        default_factory=dict
    )  # Float for fractional shared attribution

    # Shared weight tracking
    shared_weights: dict[str, list[str]] = Field(
        default_factory=dict
    )  # initializer -> nodes using it
    num_shared_weights: int = 0  # Count of weights used by 2+ nodes

    # Quantization info
    precision_breakdown: dict[str, int] = Field(default_factory=dict)  # dtype -> param count
    is_quantized: bool = False  # True if model has quantized weights or ops
    quantized_ops: list[str] = Field(default_factory=list)  # Quantized op types detected

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "trainable": self.trainable,
            "non_trainable": self.non_trainable,
            "by_op_type": {k: round(v, 2) for k, v in self.by_op_type.items()},
            "shared_weights": {
                "count": self.num_shared_weights,
                "details": {k: v for k, v in self.shared_weights.items() if len(v) > 1},
            },
            "precision_breakdown": self.precision_breakdown,
            "is_quantized": self.is_quantized,
            "quantized_ops": self.quantized_ops,
        }


class FlopCounts(BaseModel):
    """FLOP estimate breakdown."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    total: int = 0
    by_node: dict[str, int] = Field(default_factory=dict)
    by_op_type: dict[str, int] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "by_op_type": self.by_op_type,
        }


class MemoryBreakdown(BaseModel):
    """Detailed memory breakdown by component type."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Weights by operation type
    weights_by_op_type: dict[str, int] = Field(default_factory=dict)  # op -> bytes
    # Top weight tensors
    largest_weights: list[tuple[str, int]] = Field(default_factory=list)  # (name, bytes)
    # Activation breakdown
    activations_by_op_type: dict[str, int] = Field(default_factory=dict)  # op -> bytes
    largest_activations: list[tuple[str, int]] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights_by_op_type": self.weights_by_op_type,
            "largest_weights": [
                {"name": name, "bytes": size} for name, size in self.largest_weights[:10]
            ],
            "activations_by_op_type": self.activations_by_op_type,
            "largest_activations": [
                {"name": name, "bytes": size} for name, size in self.largest_activations[:10]
            ],
        }


class MemoryEstimates(BaseModel):
    """Memory usage estimates."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_size_bytes: int = 0  # Size of parameters/initializers
    peak_activation_bytes: int = 0  # Estimated peak activation memory (batch=1)
    per_layer_activation_bytes: dict[str, int] = Field(default_factory=dict)
    # KV cache estimates for transformer models
    kv_cache_bytes_per_token: int = 0  # KV cache per token (for streaming inference)
    kv_cache_bytes_full_context: int = 0  # Total KV cache at max seq length
    kv_cache_config: dict[str, int] = Field(default_factory=dict)  # num_layers, hidden_dim, etc.
    # Detailed breakdown
    breakdown: MemoryBreakdown | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "model_size_bytes": self.model_size_bytes,
            "peak_activation_bytes": self.peak_activation_bytes,
        }
        if self.kv_cache_bytes_per_token > 0:
            result["kv_cache_bytes_per_token"] = self.kv_cache_bytes_per_token
            result["kv_cache_bytes_full_context"] = self.kv_cache_bytes_full_context
            result["kv_cache_config"] = self.kv_cache_config
        if self.breakdown:
            result["breakdown"] = self.breakdown.to_dict()
        return result


class ONNXGraphLoader:
    """
    Load ONNX models and extract graph structure.

    Handles shape inference and creates a GraphInfo representation
    suitable for analysis.
    """

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger("haoline.loader")

    def load(self, model_path: str | pathlib.Path) -> tuple[onnx.ModelProto, GraphInfo]:
        """
        Load an ONNX model and extract graph information.

        Args:
            model_path: Path to the ONNX model file.

        Returns:
            Tuple of (ModelProto, GraphInfo)
        """
        model_path = pathlib.Path(model_path)
        self.logger.debug(f"Loading model from {model_path}")

        # Use ORT's helper if available, otherwise fall back to onnx
        if _HAS_ORT_UTILS and ModelProtoWithShapeInfo is not None:
            wrapper = ModelProtoWithShapeInfo(model_path)
            model = wrapper.model_with_shape_info
        else:
            # Fallback: load with onnx and run shape inference
            model = onnx.load(str(model_path))
            try:
                model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
            except Exception as e:
                self.logger.warning(f"Shape inference failed: {e}. Proceeding without shape info.")

        graph_info = self._extract_graph_info(model.graph, model)

        self.logger.debug(f"Loaded graph with {graph_info.num_nodes} nodes")
        return model, graph_info

    def _extract_graph_info(self, graph: onnx.GraphProto, model: onnx.ModelProto) -> GraphInfo:
        """Extract GraphInfo from an ONNX GraphProto."""

        # Extract initializers (weights/biases)
        initializers = {}
        for init in graph.initializer:
            try:
                initializers[init.name] = onnx.numpy_helper.to_array(init)
            except Exception as e:
                self.logger.warning(f"Could not convert initializer {init.name}: {e}")
                # Store shape info at minimum
                initializers[init.name] = np.zeros(init.dims, dtype=np.float32)

        # Build value shape map from value_info, inputs, and outputs
        value_shapes = {}

        def _extract_shape(value_info: onnx.ValueInfoProto) -> list[int | str]:
            shape = []
            if value_info.type.HasField("tensor_type"):
                tensor_type = value_info.type.tensor_type
                if tensor_type.HasField("shape"):
                    for dim in tensor_type.shape.dim:
                        if dim.HasField("dim_value"):
                            shape.append(dim.dim_value)
                        elif dim.HasField("dim_param"):
                            shape.append(dim.dim_param)
                        else:
                            shape.append("?")
            return shape

        for vi in graph.input:
            value_shapes[vi.name] = _extract_shape(vi)
        for vi in graph.output:
            value_shapes[vi.name] = _extract_shape(vi)
        for vi in graph.value_info:
            value_shapes[vi.name] = _extract_shape(vi)

        # For initializers without explicit value_info, use their tensor shapes
        for name, arr in initializers.items():
            if name not in value_shapes:
                value_shapes[name] = list(arr.shape)

        # Extract nodes
        nodes: list[NodeInfo] = []
        op_type_counts: dict[str, int] = {}
        node_by_name: dict[str, NodeInfo] = {}
        node_by_output: dict[str, NodeInfo] = {}

        for node in graph.node:
            # Extract attributes
            attributes = {}
            for attr in node.attribute:
                if attr.HasField("i"):
                    attributes[attr.name] = attr.i
                elif attr.HasField("f"):
                    attributes[attr.name] = attr.f
                elif attr.HasField("s"):
                    attributes[attr.name] = (
                        attr.s.decode("utf-8") if isinstance(attr.s, bytes) else attr.s
                    )
                elif attr.ints:
                    attributes[attr.name] = list(attr.ints)
                elif attr.floats:
                    attributes[attr.name] = list(attr.floats)
                # Skip subgraphs and other complex types for now

            node_info = NodeInfo(
                name=node.name or f"unnamed_{len(nodes)}",
                op_type=node.op_type,
                domain=node.domain or "ai.onnx",
                inputs=list(node.input),
                outputs=list(node.output),
                attributes=attributes,
            )
            nodes.append(node_info)
            node_by_name[node_info.name] = node_info
            for output in node_info.outputs:
                node_by_output[output] = node_info

            # Count op types
            op_type_counts[node.op_type] = op_type_counts.get(node.op_type, 0) + 1

        # Build input/output shape maps (excluding initializers from inputs)
        input_names = [i.name for i in graph.input if i.name not in initializers]
        output_names = [o.name for o in graph.output]

        input_shapes = {name: value_shapes.get(name, []) for name in input_names}
        output_shapes = {name: value_shapes.get(name, []) for name in output_names}

        return GraphInfo(
            name=graph.name or "main",
            nodes=nodes,
            inputs=input_names,
            outputs=output_names,
            initializers=initializers,
            value_shapes=value_shapes,
            num_nodes=len(nodes),
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            op_type_counts=op_type_counts,
            node_by_name=node_by_name,
            node_by_output=node_by_output,
        )


class MetricsEngine:
    """
    Compute model complexity metrics.

    Provides parameter counts, FLOP estimates, and memory estimates
    for ONNX graphs.
    """

    # FLOP multipliers per operation type
    # These are rough estimates; actual FLOPs depend on implementation
    FLOP_FORMULAS: ClassVar[dict[str, str]] = {
        # Conv: 2 * K_h * K_w * C_in * C_out * H_out * W_out
        "Conv": "conv",
        # MatMul: 2 * M * N * K
        "MatMul": "matmul",
        "Gemm": "gemm",
        # Element-wise ops: N elements
        "Add": "elementwise",
        "Sub": "elementwise",
        "Mul": "elementwise",
        "Div": "elementwise",
        "Relu": "elementwise",
        "Sigmoid": "elementwise",
        "Tanh": "elementwise",
        "Sqrt": "elementwise",
        "Exp": "elementwise",
        "Log": "elementwise",
        "Gelu": "elementwise",
        "Silu": "elementwise",
        # Softmax: ~5N (exp, sum, div)
        "Softmax": "softmax",
        # Reduction ops: N elements
        "ReduceMean": "elementwise",
        "ReduceSum": "elementwise",
        "ReduceMax": "elementwise",
        # Normalization layers
        "LayerNormalization": "layernorm",
        "BatchNormalization": "batchnorm",
        # Attention ops (ONNX contrib / custom)
        "Attention": "attention",
        "MultiHeadAttention": "attention",
        "com.microsoft.Attention": "attention",
        "com.microsoft.MultiHeadAttention": "attention",
    }

    # Quantized operation types in ONNX
    QUANTIZED_OPS: ClassVar[set[str]] = {
        "QuantizeLinear",
        "DequantizeLinear",
        "QLinearConv",
        "QLinearMatMul",
        "QLinearAdd",
        "QGemm",
        "ConvInteger",
        "MatMulInteger",
        "DynamicQuantizeLinear",
        "QLinearSigmoid",
        "QLinearLeakyRelu",
        "QLinearAveragePool",
        "QLinearGlobalAveragePool",
        "QLinearConcat",
    }

    # Quantized dtypes
    QUANTIZED_DTYPES: ClassVar[set[type]] = {np.int8, np.uint8, np.int16, np.uint16}

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger("haoline.metrics")

    def count_parameters(self, graph_info: GraphInfo) -> ParamCounts:
        """
        Count parameters in the model.

        Parameters are counted from initializers. All initializers are
        assumed trainable unless specifically marked otherwise.

        Handles edge cases:
        - Shared weights: Uses fractional attribution so by_op_type sums to total
        - Quantized params: Detects INT8/UINT8 weights and quantized ops

        Args:
            graph_info: Parsed graph information.

        Returns:
            ParamCounts with total and per-node breakdowns.
        """
        counts = ParamCounts()

        # First pass: build usage map (which nodes use each initializer)
        usage_map: dict[str, list[str]] = {name: [] for name in graph_info.initializers}
        for node in graph_info.nodes:
            for inp in node.inputs:
                if inp in graph_info.initializers:
                    usage_map[inp].append(node.name)

        # Track shared weights (used by 2+ nodes)
        counts.shared_weights = {k: v for k, v in usage_map.items() if len(v) > 1}
        counts.num_shared_weights = len(counts.shared_weights)

        # Detect quantized ops in the graph
        quantized_ops_found = set()
        for node in graph_info.nodes:
            if node.op_type in self.QUANTIZED_OPS:
                quantized_ops_found.add(node.op_type)
        counts.quantized_ops = sorted(quantized_ops_found)

        # Second pass: count parameters with fractional attribution
        for name, tensor in graph_info.initializers.items():
            param_count = int(np.prod(tensor.shape)) if tensor.shape else 1
            counts.total += param_count
            counts.by_node[name] = float(param_count)

            # Track precision breakdown
            dtype_name = self._get_dtype_name(tensor)
            counts.precision_breakdown[dtype_name] = (
                counts.precision_breakdown.get(dtype_name, 0) + param_count
            )

            # Check if this is a quantized weight
            if hasattr(tensor, "dtype") and tensor.dtype in self.QUANTIZED_DTYPES:
                counts.is_quantized = True

            # Fractional attribution to nodes sharing this weight
            using_nodes = usage_map[name]
            num_users = len(using_nodes) if using_nodes else 1
            fractional_count = param_count / num_users

            for node in graph_info.nodes:
                if node.name in using_nodes:
                    counts.by_op_type[node.op_type] = (
                        counts.by_op_type.get(node.op_type, 0.0) + fractional_count
                    )
                    node.param_count += fractional_count

        # Mark as quantized if quantized ops are present
        if counts.quantized_ops:
            counts.is_quantized = True

        # For now, assume all are trainable
        # Could be refined with graph analysis (e.g., constants, frozen layers)
        counts.trainable = counts.total
        counts.non_trainable = 0

        return counts

    def _get_dtype_name(self, tensor: np.ndarray) -> str:
        """Get a human-readable dtype name for a tensor."""
        if not hasattr(tensor, "dtype"):
            return "unknown"
        dtype = tensor.dtype
        dtype_map = {
            np.float32: "fp32",
            np.float64: "fp64",
            np.float16: "fp16",
            np.int8: "int8",
            np.uint8: "uint8",
            np.int16: "int16",
            np.uint16: "uint16",
            np.int32: "int32",
            np.int64: "int64",
        }
        return dtype_map.get(dtype.type, str(dtype))

    def estimate_flops(self, graph_info: GraphInfo) -> FlopCounts:
        """
        Estimate FLOPs for each operation in the graph.

        Uses shape information to compute FLOPs. Falls back to
        rough estimates when shapes are unavailable.

        Args:
            graph_info: Parsed graph information.

        Returns:
            FlopCounts with total and per-node breakdowns.
        """
        counts = FlopCounts()

        for node in graph_info.nodes:
            flops = self._estimate_node_flops(node, graph_info)
            node.flops = flops
            counts.total += flops
            counts.by_node[node.name] = flops
            counts.by_op_type[node.op_type] = counts.by_op_type.get(node.op_type, 0) + flops

        return counts

    def _estimate_node_flops(self, node: NodeInfo, graph_info: GraphInfo) -> int:
        """Estimate FLOPs for a single node."""
        formula_type = self.FLOP_FORMULAS.get(node.op_type, "unknown")

        if formula_type == "conv":
            return self._estimate_conv_flops(node, graph_info)
        elif formula_type == "matmul":
            return self._estimate_matmul_flops(node, graph_info)
        elif formula_type == "gemm":
            return self._estimate_gemm_flops(node, graph_info)
        elif formula_type == "elementwise":
            return self._estimate_elementwise_flops(node, graph_info)
        elif formula_type == "softmax":
            return self._estimate_elementwise_flops(node, graph_info) * 5
        elif formula_type == "layernorm":
            return self._estimate_elementwise_flops(node, graph_info) * 5
        elif formula_type == "batchnorm":
            return self._estimate_elementwise_flops(node, graph_info) * 2
        elif formula_type == "attention":
            return self._estimate_attention_flops(node, graph_info)
        else:
            # Unknown op - estimate based on output size
            return self._estimate_elementwise_flops(node, graph_info)

    def _estimate_conv_flops(self, node: NodeInfo, graph_info: GraphInfo) -> int:
        """Estimate FLOPs for Conv operation: 2 * K_h * K_w * C_in * C_out * H_out * W_out"""
        if len(node.inputs) < 2:
            return 0

        # Get weight shape
        weight_name = node.inputs[1]
        if weight_name in graph_info.initializers:
            weight_shape = list(graph_info.initializers[weight_name].shape)
        elif weight_name in graph_info.value_shapes:
            weight_shape = graph_info.value_shapes[weight_name]
        else:
            return 0

        # Weight shape: [C_out, C_in/groups, K_h, K_w] for 2D conv
        if len(weight_shape) < 4 or not all(isinstance(d, int) for d in weight_shape):
            return 0

        c_out, c_in_per_group, k_h, k_w = weight_shape[:4]

        # Get output shape
        if node.outputs and node.outputs[0] in graph_info.value_shapes:
            output_shape = graph_info.value_shapes[node.outputs[0]]
            if len(output_shape) >= 4 and all(isinstance(d, int) for d in output_shape[-2:]):
                h_out, w_out = output_shape[-2], output_shape[-1]
            else:
                h_out, w_out = 1, 1
        else:
            h_out, w_out = 1, 1

        node.attributes.get("group", 1)
        flops = 2 * k_h * k_w * c_in_per_group * c_out * h_out * w_out

        # Add bias if present
        if len(node.inputs) > 2:
            flops += c_out * h_out * w_out

        return int(flops)

    def _estimate_matmul_flops(self, node: NodeInfo, graph_info: GraphInfo) -> int:
        """Estimate FLOPs for MatMul: 2 * M * N * K"""
        if len(node.inputs) < 2:
            return 0

        # Get shapes of both inputs
        shapes = []
        for inp in node.inputs[:2]:
            if inp in graph_info.initializers:
                shapes.append(list(graph_info.initializers[inp].shape))
            elif inp in graph_info.value_shapes:
                shapes.append(graph_info.value_shapes[inp])
            else:
                return 0

        if len(shapes) < 2:
            return 0

        # MatMul: A[..., M, K] @ B[..., K, N] = C[..., M, N]
        shape_a, shape_b = shapes[0], shapes[1]

        # Handle broadcasting and get M, K, N
        if len(shape_a) < 2 or len(shape_b) < 2:
            return 0

        if not all(isinstance(d, int) for d in shape_a[-2:]) or not all(
            isinstance(d, int) for d in shape_b[-2:]
        ):
            return 0

        m, k = shape_a[-2], shape_a[-1]
        k2, n = shape_b[-2], shape_b[-1]

        if k != k2:
            self.logger.warning(f"MatMul shape mismatch in node {node.name}: K={k} vs K={k2}")
            return 0

        # Handle batch dimensions
        batch = 1
        for dim in shape_a[:-2]:
            if isinstance(dim, int):
                batch *= dim

        return int(2 * batch * m * n * k)

    def _estimate_gemm_flops(self, node: NodeInfo, graph_info: GraphInfo) -> int:
        """Estimate FLOPs for Gemm: 2 * M * N * K + M * N (bias)"""
        flops = self._estimate_matmul_flops(node, graph_info)

        # Add bias computation if present
        if len(node.inputs) > 2 and node.outputs and node.outputs[0] in graph_info.value_shapes:
            output_shape = graph_info.value_shapes[node.outputs[0]]
            if output_shape and all(isinstance(d, int) for d in output_shape):
                int_shape: list[int] = [d for d in output_shape if isinstance(d, int)]
                bias_flops = int(np.prod(int_shape))
                flops += bias_flops

        return flops

    def _estimate_elementwise_flops(self, node: NodeInfo, graph_info: GraphInfo) -> int:
        """Estimate FLOPs for element-wise operations: N elements"""
        # Use output shape to determine element count
        if node.outputs and node.outputs[0] in graph_info.value_shapes:
            shape = graph_info.value_shapes[node.outputs[0]]
            if shape and all(isinstance(d, int) for d in shape):
                int_shape: list[int] = [d for d in shape if isinstance(d, int)]
                return int(np.prod(int_shape))

        # Fallback: use first input shape
        if node.inputs and node.inputs[0] in graph_info.value_shapes:
            shape = graph_info.value_shapes[node.inputs[0]]
            if shape and all(isinstance(d, int) for d in shape):
                int_shape2: list[int] = [d for d in shape if isinstance(d, int)]
                return int(np.prod(int_shape2))

        return 0

    def _estimate_attention_flops(self, node: NodeInfo, graph_info: GraphInfo) -> int:
        """
        Estimate FLOPs for attention operations.

        Standard multi-head attention FLOPs:
        - QKV projections: 3 * batch * seq_len * d_model * d_model
        - Attention scores (Q @ K^T): batch * num_heads * seq_len * seq_len * d_head
        - Softmax: batch * num_heads * seq_len * seq_len * 5
        - Attention output (scores @ V): batch * num_heads * seq_len * seq_len * d_head
        - Output projection: batch * seq_len * d_model * d_model

        Simplified formula: 4 * seq_len * d_model^2 + 2 * num_heads * seq_len^2 * d_head
        """
        # Try to get dimensions from node attributes or input shapes
        num_heads = 1
        seq_len = 512  # Default assumption
        d_model = 768  # Default assumption

        # Try to extract from node attributes (ONNX Attention op)
        for attr_name, attr_value in node.attributes.items():
            if attr_name == "num_heads" and isinstance(attr_value, int):
                num_heads = attr_value
            elif attr_name == "hidden_size" and isinstance(attr_value, int):
                d_model = attr_value

        # Try to infer from input shapes
        if node.inputs and node.inputs[0] in graph_info.value_shapes:
            input_shape = graph_info.value_shapes[node.inputs[0]]
            if input_shape and len(input_shape) >= 2:
                # Shape is typically [batch, seq_len, d_model] or [batch, seq_len, ...]
                if len(input_shape) >= 3:
                    if isinstance(input_shape[1], int):
                        seq_len = input_shape[1]
                    if isinstance(input_shape[2], int):
                        d_model = input_shape[2]
                elif len(input_shape) == 2:
                    if isinstance(input_shape[0], int):
                        seq_len = input_shape[0]
                    if isinstance(input_shape[1], int):
                        d_model = input_shape[1]

        d_head = d_model // num_heads if num_heads > 0 else d_model

        # Compute FLOPs using standard attention formula
        # QKV projections: 3 * seq * d_model * d_model
        qkv_flops = 3 * seq_len * d_model * d_model

        # Attention scores and output: 2 * num_heads * seq^2 * d_head
        attention_flops = 2 * num_heads * seq_len * seq_len * d_head

        # Output projection: seq * d_model * d_model
        output_flops = seq_len * d_model * d_model

        # Softmax on attention scores: 5 * num_heads * seq^2
        softmax_flops = 5 * num_heads * seq_len * seq_len

        total_flops = qkv_flops + attention_flops + output_flops + softmax_flops

        self.logger.debug(
            f"Attention FLOPs: seq={seq_len}, d_model={d_model}, "
            f"heads={num_heads}, total={total_flops:,}"
        )

        return total_flops

    def estimate_memory(self, graph_info: GraphInfo) -> MemoryEstimates:
        """
        Estimate memory usage for the model.

        Computes model size (parameters), peak activation memory,
        KV cache size for transformer models, and detailed breakdown.

        Args:
            graph_info: Parsed graph information.

        Returns:
            MemoryEstimates with size, activation memory, KV cache, and breakdown.
        """
        estimates = MemoryEstimates()
        breakdown = MemoryBreakdown()

        # Build mapping: initializer name -> op type that uses it
        init_to_op: dict[str, str] = {}
        for node in graph_info.nodes:
            for inp in node.inputs:
                if inp in graph_info.initializers and inp not in init_to_op:
                    init_to_op[inp] = node.op_type

        # Model size: sum of initializer sizes with breakdown by op type
        weight_sizes: list[tuple[str, int]] = []
        for name, tensor in graph_info.initializers.items():
            # Determine bytes per element based on dtype
            bytes_per_elem = 4
            if hasattr(tensor, "dtype"):
                if tensor.dtype == np.float16:
                    bytes_per_elem = 2
                elif tensor.dtype == np.float64:
                    bytes_per_elem = 8
                elif tensor.dtype in (np.int8, np.uint8):
                    bytes_per_elem = 1
                elif tensor.dtype in (np.int16, np.uint16):
                    bytes_per_elem = 2

            tensor_bytes = (
                int(np.prod(tensor.shape)) * bytes_per_elem if tensor.shape else bytes_per_elem
            )
            estimates.model_size_bytes += tensor_bytes
            weight_sizes.append((name, tensor_bytes))

            # Categorize by op type
            op_type = init_to_op.get(name, "Other")
            breakdown.weights_by_op_type[op_type] = (
                breakdown.weights_by_op_type.get(op_type, 0) + tensor_bytes
            )

        # Store top 10 largest weights
        breakdown.largest_weights = sorted(weight_sizes, key=lambda x: -x[1])[:10]

        # Peak activation memory: rough estimate based on intermediate tensor sizes
        # Build mapping: activation name -> op type that produces it
        activation_to_op: dict[str, str] = {}
        for node in graph_info.nodes:
            for out in node.outputs:
                activation_to_op[out] = node.op_type

        activation_sizes: list[tuple[str, int]] = []
        for name, shape in graph_info.value_shapes.items():
            # Skip initializers (they're counted in model size)
            if name in graph_info.initializers:
                continue

            if shape:
                # Handle symbolic dimensions (e.g., 'N' for batch) by treating as 1
                int_shape: list[int] = [d if isinstance(d, int) else 1 for d in shape]
                # Skip if all dims are symbolic (no meaningful size)
                if all(d == 1 for d in int_shape) and len(int_shape) > 1:
                    continue
                # Assume float32 for activations
                tensor_bytes = int(np.prod(int_shape)) * 4
                activation_sizes.append((name, tensor_bytes))
                estimates.per_layer_activation_bytes[name] = tensor_bytes

                # Categorize by producing op type
                op_type = activation_to_op.get(name, "Input")
                breakdown.activations_by_op_type[op_type] = (
                    breakdown.activations_by_op_type.get(op_type, 0) + tensor_bytes
                )

        # Peak is approximate: sum of largest activations that might coexist
        sorted_activations = sorted(activation_sizes, key=lambda x: -x[1])
        # Rough heuristic: top 3 largest activations might coexist
        top_n = min(3, len(sorted_activations))
        estimates.peak_activation_bytes = sum(size for _, size in sorted_activations[:top_n])

        # Store top 10 largest activations
        breakdown.largest_activations = sorted_activations[:10]

        # Store breakdown
        estimates.breakdown = breakdown

        # Estimate KV cache for transformer models
        kv_config = self._estimate_kv_cache_config(graph_info)
        if kv_config:
            estimates.kv_cache_config = kv_config
            estimates.kv_cache_bytes_per_token = self._compute_kv_cache_per_token(kv_config)
            estimates.kv_cache_bytes_full_context = (
                estimates.kv_cache_bytes_per_token * kv_config["seq_len"]
            )

        return estimates

    def _estimate_kv_cache_config(self, graph_info: GraphInfo) -> dict[str, int]:
        """
        Detect transformer architecture and extract KV cache config.

        Returns dict with num_layers, hidden_dim, num_heads, seq_len, bytes_per_elem
        or empty dict if not a transformer.
        """
        # Check for attention ops
        attention_ops = {"Attention", "MultiHeadAttention", "Softmax"}
        attention_count = sum(graph_info.op_type_counts.get(op, 0) for op in attention_ops)

        if attention_count == 0:
            return {}

        # Try to detect transformer parameters
        num_layers = 0
        hidden_dim = 768  # Default
        num_heads = 12  # Default
        seq_len = 512  # Default
        bytes_per_elem = 4  # FP32 default

        # Count attention ops to estimate number of layers
        # Each transformer layer typically has one attention block
        mha_count = graph_info.op_type_counts.get("Attention", 0) + graph_info.op_type_counts.get(
            "MultiHeadAttention", 0
        )
        softmax_count = graph_info.op_type_counts.get("Softmax", 0)

        # Use MHA count if available, otherwise estimate from Softmax
        if mha_count > 0:
            num_layers = mha_count
        elif softmax_count > 0:
            # Softmax in attention: typically one per layer (or two with cross-attention)
            num_layers = max(1, softmax_count // 2)

        if num_layers == 0:
            return {}

        # Try to infer hidden_dim from weight shapes
        for node in graph_info.nodes:
            if node.op_type in ("MatMul", "Gemm"):
                for inp in node.inputs:
                    if inp in graph_info.initializers:
                        weight = graph_info.initializers[inp]
                        if len(weight.shape) == 2:
                            # Dense layer weights: [in_features, out_features] or vice versa
                            dim = max(weight.shape)
                            if 256 <= dim <= 16384 and dim % 64 == 0:
                                hidden_dim = dim
                                break
                break

        # Try to infer sequence length from input shapes
        for shape in graph_info.input_shapes.values():
            if len(shape) >= 2:
                # Look for typical transformer input shape [batch, seq_len, ...] or [batch, seq_len]
                for dim in shape[1:3]:
                    if isinstance(dim, int) and 16 <= dim <= 32768:
                        seq_len = dim
                        break

        # Estimate num_heads from hidden_dim (typical: 64-128 per head)
        if hidden_dim >= 256:
            num_heads = max(1, hidden_dim // 64)

        self.logger.debug(
            f"KV cache config: layers={num_layers}, hidden={hidden_dim}, "
            f"heads={num_heads}, seq={seq_len}"
        )

        return {
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "seq_len": seq_len,
            "bytes_per_elem": bytes_per_elem,
        }

    def _compute_kv_cache_per_token(self, config: dict[str, int]) -> int:
        """
        Compute KV cache memory per token.

        Formula: 2 * num_layers * hidden_dim * bytes_per_elem
        (2 for K and V, each of size [hidden_dim])

        For multi-head attention with head_dim = hidden_dim / num_heads:
        KV cache per token per layer = 2 * hidden_dim * bytes_per_elem

        Total per token = 2 * num_layers * hidden_dim * bytes_per_elem
        """
        num_layers = config.get("num_layers", 0)
        hidden_dim = config.get("hidden_dim", 0)
        bytes_per_elem = config.get("bytes_per_elem", 4)

        # KV cache: each layer stores K and V for each token
        # K and V each have shape [batch, num_heads, seq_len, head_dim]
        # Per token: 2 * hidden_dim * bytes_per_elem per layer
        return 2 * num_layers * hidden_dim * bytes_per_elem
