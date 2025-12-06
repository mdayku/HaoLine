# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
Edge-Centric Analysis for graph visualization.

Task 5.6: Analyze tensor flow between nodes to identify bottlenecks,
memory hotspots, and data flow patterns.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from .analyzer import GraphInfo


# Data type sizes in bytes
DTYPE_SIZES: dict[str, int] = {
    "float32": 4,
    "float": 4,
    "float16": 2,
    "half": 2,
    "bfloat16": 2,
    "float64": 8,
    "double": 8,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "uint8": 1,
    "uint16": 2,
    "uint32": 4,
    "uint64": 8,
    "bool": 1,
    "string": 8,  # Pointer size estimate
    "complex64": 8,
    "complex128": 16,
}


class EdgeInfo(BaseModel):
    """Information about an edge (tensor) between nodes."""

    model_config = ConfigDict(frozen=False)  # Allow mutation for analysis

    tensor_name: str
    source_node: str | None  # None if graph input
    target_nodes: list[str]  # Nodes consuming this tensor
    shape: list[int | str]  # Shape with possible symbolic dims
    dtype: str
    size_bytes: int  # Total size in bytes
    is_weight: bool  # True if initializer/constant
    precision: str  # fp32, fp16, int8, etc.

    # Analysis results
    is_bottleneck: bool = False
    is_skip_connection: bool = False
    is_attention_qk: bool = False  # Q @ K^T output (O(seq^2))
    memory_intensity: float = 0.0  # 0-1 scale

    def to_dict(self) -> dict:
        return {
            "tensor_name": self.tensor_name,
            "source_node": self.source_node,
            "target_nodes": self.target_nodes,
            "shape": self.shape,
            "dtype": self.dtype,
            "size_bytes": self.size_bytes,
            "is_weight": self.is_weight,
            "precision": self.precision,
            "is_bottleneck": self.is_bottleneck,
            "is_skip_connection": self.is_skip_connection,
            "is_attention_qk": self.is_attention_qk,
            "memory_intensity": self.memory_intensity,
        }


class EdgeAnalysisResult(BaseModel):
    """Complete edge analysis for a graph."""

    model_config = ConfigDict(frozen=True)

    edges: list[EdgeInfo]
    total_activation_bytes: int
    peak_activation_bytes: int
    peak_activation_node: str | None
    bottleneck_edges: list[str]  # Tensor names of bottleneck edges
    attention_edges: list[str]  # O(seq^2) attention edges
    skip_connection_edges: list[str]

    # Memory profile along execution
    memory_profile: list[tuple[str, int]]  # (node_name, cumulative_memory)

    def to_dict(self) -> dict:
        return {
            "total_activation_bytes": self.total_activation_bytes,
            "peak_activation_bytes": self.peak_activation_bytes,
            "peak_activation_node": self.peak_activation_node,
            "bottleneck_edges": self.bottleneck_edges,
            "attention_edges": self.attention_edges,
            "skip_connection_edges": self.skip_connection_edges,
            "num_edges": len(self.edges),
        }


class EdgeAnalyzer:
    """
    Analyze edges (tensors) in the computation graph.

    Identifies memory bottlenecks, attention patterns, and skip connections.
    """

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger("haoline.edges")

    def analyze(self, graph_info: GraphInfo) -> EdgeAnalysisResult:
        """
        Perform complete edge analysis on a graph.

        Args:
            graph_info: Parsed graph information from ONNXGraphLoader.

        Returns:
            EdgeAnalysisResult with all edge information and analysis.
        """
        edges = self._extract_edges(graph_info)
        self._analyze_bottlenecks(edges)
        self._detect_skip_connections(edges, graph_info)
        self._detect_attention_edges(edges, graph_info)

        # Calculate memory profile
        memory_profile = self._calculate_memory_profile(edges, graph_info)

        # Find peak
        peak_bytes = 0
        peak_node = None
        for node_name, mem in memory_profile:
            if mem > peak_bytes:
                peak_bytes = mem
                peak_node = node_name

        # Collect special edges
        bottleneck_edges = [e.tensor_name for e in edges if e.is_bottleneck]
        attention_edges = [e.tensor_name for e in edges if e.is_attention_qk]
        skip_edges = [e.tensor_name for e in edges if e.is_skip_connection]

        total_bytes = sum(e.size_bytes for e in edges if not e.is_weight)

        return EdgeAnalysisResult(
            edges=edges,
            total_activation_bytes=total_bytes,
            peak_activation_bytes=peak_bytes,
            peak_activation_node=peak_node,
            bottleneck_edges=bottleneck_edges,
            attention_edges=attention_edges,
            skip_connection_edges=skip_edges,
            memory_profile=memory_profile,
        )

    def _extract_edges(self, graph_info: GraphInfo) -> list[EdgeInfo]:
        """Extract all edges (tensors) from the graph."""
        edges: list[EdgeInfo] = []
        tensor_to_consumers: dict[str, list[str]] = {}

        # Build consumer map
        for node in graph_info.nodes:
            for inp in node.inputs:
                if inp not in tensor_to_consumers:
                    tensor_to_consumers[inp] = []
                tensor_to_consumers[inp].append(node.name)

        # Process all tensors
        processed: set[str] = set()

        # Graph inputs
        for name, shape in graph_info.input_shapes.items():
            if name in processed:
                continue
            processed.add(name)

            dtype = self._get_tensor_dtype(name, graph_info)
            size_bytes = self._calculate_tensor_bytes(shape, dtype)
            precision = self._dtype_to_precision(dtype)

            edges.append(
                EdgeInfo(
                    tensor_name=name,
                    source_node=None,
                    target_nodes=tensor_to_consumers.get(name, []),
                    shape=shape,
                    dtype=dtype,
                    size_bytes=size_bytes,
                    is_weight=False,
                    precision=precision,
                )
            )

        # Initializers (weights)
        for name, arr in graph_info.initializers.items():
            if name in processed:
                continue
            processed.add(name)

            shape = list(arr.shape)
            dtype = str(arr.dtype)
            size_bytes = arr.nbytes
            precision = self._dtype_to_precision(dtype)

            edges.append(
                EdgeInfo(
                    tensor_name=name,
                    source_node=None,
                    target_nodes=tensor_to_consumers.get(name, []),
                    shape=shape,
                    dtype=dtype,
                    size_bytes=size_bytes,
                    is_weight=True,
                    precision=precision,
                )
            )

        # Node outputs (activations)
        for node in graph_info.nodes:
            for output in node.outputs:
                if output in processed:
                    continue
                processed.add(output)

                shape = graph_info.value_shapes.get(output, [])
                dtype = self._get_tensor_dtype(output, graph_info)
                size_bytes = self._calculate_tensor_bytes(shape, dtype)
                precision = self._dtype_to_precision(dtype)

                edges.append(
                    EdgeInfo(
                        tensor_name=output,
                        source_node=node.name,
                        target_nodes=tensor_to_consumers.get(output, []),
                        shape=shape,
                        dtype=dtype,
                        size_bytes=size_bytes,
                        is_weight=False,
                        precision=precision,
                    )
                )

        return edges

    def _get_tensor_dtype(self, name: str, graph_info: GraphInfo) -> str:
        """Get dtype for a tensor."""
        # Check if it's an initializer
        if name in graph_info.initializers:
            return str(graph_info.initializers[name].dtype)

        # Default to float32 for activations
        return "float32"

    def _calculate_tensor_bytes(self, shape: list, dtype: str) -> int:
        """Calculate tensor size in bytes."""
        # Handle symbolic dimensions
        num_elements = 1
        for dim in shape:
            if isinstance(dim, int) and dim > 0:
                num_elements *= dim
            elif isinstance(dim, str):
                # Symbolic dim - estimate as 1 for now
                # Could use batch_size=1, seq_len=512 defaults
                num_elements *= 1
            else:
                num_elements *= 1

        # Get dtype size
        dtype_lower = dtype.lower().replace("torch.", "").replace("numpy.", "")
        element_bytes = DTYPE_SIZES.get(dtype_lower, 4)  # Default to 4 bytes

        return num_elements * element_bytes

    def _dtype_to_precision(self, dtype: str) -> str:
        """Convert dtype string to precision category."""
        dtype_lower = dtype.lower()
        if "float32" in dtype_lower or dtype_lower == "float":
            return "fp32"
        elif "float16" in dtype_lower or "half" in dtype_lower:
            return "fp16"
        elif "bfloat16" in dtype_lower:
            return "bf16"
        elif "int8" in dtype_lower:
            return "int8"
        elif "uint8" in dtype_lower:
            return "uint8"
        elif "int4" in dtype_lower:
            return "int4"
        elif "int" in dtype_lower:
            return "int32"
        else:
            return "fp32"

    def _analyze_bottlenecks(self, edges: list[EdgeInfo]) -> None:
        """Mark edges that are memory bottlenecks."""
        # Find max activation size (excluding weights)
        activation_sizes = [e.size_bytes for e in edges if not e.is_weight]
        if not activation_sizes:
            return

        max_size = max(activation_sizes)
        max_size * 0.5  # Top 50% are potential bottlenecks

        for edge in edges:
            if edge.is_weight:
                continue

            edge.memory_intensity = edge.size_bytes / max_size if max_size > 0 else 0

            # Mark as bottleneck if in top 20%
            if edge.size_bytes >= max_size * 0.8:
                edge.is_bottleneck = True

    def _detect_skip_connections(self, edges: list[EdgeInfo], graph_info: GraphInfo) -> None:
        """Detect skip connection edges."""
        # Skip connections typically:
        # 1. Go from earlier node to later Add node
        # 2. Bypass multiple nodes

        # Build node position map (topological order)
        node_positions: dict[str, int] = {}
        for i, node in enumerate(graph_info.nodes):
            node_positions[node.name] = i

        for edge in edges:
            if edge.source_node is None:
                continue

            source_pos = node_positions.get(edge.source_node, 0)

            for target_name in edge.target_nodes:
                target_pos = node_positions.get(target_name, 0)

                # Check if target is an Add (residual connection point)
                target_node = None
                for n in graph_info.nodes:
                    if n.name == target_name:
                        target_node = n
                        break

                if target_node and target_node.op_type == "Add":
                    # Check if there's a significant skip distance
                    if target_pos - source_pos >= 3:
                        edge.is_skip_connection = True
                        break

    def _detect_attention_edges(self, edges: list[EdgeInfo], graph_info: GraphInfo) -> None:
        """Detect O(seq^2) attention edges (Q @ K^T output)."""
        # Look for Softmax nodes and mark their input edges
        for node in graph_info.nodes:
            if node.op_type == "Softmax":
                # The input to Softmax in attention is Q @ K^T (shape: [batch, heads, seq, seq])
                for inp in node.inputs:
                    for edge in edges:
                        if edge.tensor_name == inp:
                            # Check if shape suggests attention (last two dims equal)
                            shape = edge.shape
                            if len(shape) >= 2:
                                # Handle symbolic dims
                                last = shape[-1]
                                second_last = shape[-2]
                                if last == second_last:
                                    edge.is_attention_qk = True
                                elif (
                                    isinstance(last, str)
                                    and isinstance(second_last, str)
                                    and last == second_last
                                ):
                                    edge.is_attention_qk = True

    def _calculate_memory_profile(
        self, edges: list[EdgeInfo], graph_info: GraphInfo
    ) -> list[tuple[str, int]]:
        """
        Calculate memory usage at each point in execution.

        Task 5.6.7: Calculate peak memory point in graph.

        Returns list of (node_name, cumulative_memory) tuples.
        """
        profile: list[tuple[str, int]] = []
        live_tensors: dict[str, int] = {}  # tensor_name -> size_bytes

        # Build tensor lifecycle info
        tensor_last_use: dict[str, str] = {}  # tensor -> last consuming node
        for node in graph_info.nodes:
            for inp in node.inputs:
                tensor_last_use[inp] = node.name

        # Simulate execution
        for node in graph_info.nodes:
            # Add outputs of this node
            for edge in edges:
                if edge.source_node == node.name:
                    live_tensors[edge.tensor_name] = edge.size_bytes

            # Calculate current memory
            current_mem = sum(live_tensors.values())
            profile.append((node.name, current_mem))

            # Free tensors whose last use is this node
            to_free = [t for t, last_node in tensor_last_use.items() if last_node == node.name]
            for tensor in to_free:
                live_tensors.pop(tensor, None)

        return profile


# Edge visualization helpers


def compute_edge_thickness(size_bytes: int, min_width: float = 1, max_width: float = 10) -> float:
    """
    Compute edge thickness based on tensor size.

    Task 5.6.2: Map edge thickness to tensor size.

    Uses log scale to handle the huge range of tensor sizes.
    """
    if size_bytes <= 0:
        return min_width

    # Log scale: 1KB = min_width, 10GB = max_width
    log_size = math.log10(max(size_bytes, 1))
    log_min = 3  # 1KB
    log_max = 10  # 10GB

    t = (log_size - log_min) / (log_max - log_min)
    t = max(0, min(1, t))

    return min_width + t * (max_width - min_width)


# Precision colors (Task 5.6.3)
PRECISION_EDGE_COLORS: dict[str, str] = {
    "fp32": "#4A90D9",  # Blue
    "fp16": "#2ECC71",  # Green
    "bf16": "#9B59B6",  # Purple
    "int8": "#F1C40F",  # Yellow
    "int4": "#E67E22",  # Orange
    "uint8": "#F39C12",  # Dark yellow
}


def get_edge_color(edge: EdgeInfo) -> str:
    """
    Get color for an edge based on its properties.

    Task 5.6.3: Color edges by precision.
    Task 5.6.4: Highlight memory bottleneck edges.
    Task 5.6.8: Highlight O(seq^2) attention edges.
    """
    # Priority: bottleneck > attention > skip > precision
    if edge.is_bottleneck:
        return "#E74C3C"  # Red for bottlenecks
    elif edge.is_attention_qk:
        return "#E67E22"  # Orange for O(seq^2) attention
    elif edge.is_skip_connection:
        return "#27AE60"  # Dark green for skip connections (dashed)
    else:
        return PRECISION_EDGE_COLORS.get(edge.precision, "#7F8C8D")


def get_edge_style(edge: EdgeInfo) -> str:
    """Get edge line style."""
    if edge.is_skip_connection:
        return "dashed"
    return "solid"


def format_tensor_shape(shape: list) -> str:
    """
    Format tensor shape for display.

    Task 5.6.5: Show tensor shape on hover.
    """
    if not shape:
        return "[]"

    parts = []
    for dim in shape:
        if isinstance(dim, int):
            parts.append(str(dim))
        else:
            parts.append(str(dim))

    return f"[{', '.join(parts)}]"


def format_tensor_size(size_bytes: int) -> str:
    """Format tensor size for display."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def generate_edge_tooltip(edge: EdgeInfo) -> str:
    """Generate tooltip text for an edge."""
    lines = [
        f"Tensor: {edge.tensor_name}",
        f"Shape: {format_tensor_shape(edge.shape)}",
        f"Size: {format_tensor_size(edge.size_bytes)}",
        f"Precision: {edge.precision}",
    ]

    if edge.is_bottleneck:
        lines.append("⚠️ Memory Bottleneck")
    if edge.is_attention_qk:
        lines.append("🔴 O(seq²) Attention")
    if edge.is_skip_connection:
        lines.append("⤴️ Skip Connection")

    return "\n".join(lines)
