# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
TensorRT vs ONNX Comparison Module.

Compare original ONNX model structure with compiled TensorRT engine to show:
- Layer mapping (which ONNX ops became which TRT layers)
- Fusion analysis (N ONNX ops → 1 TRT kernel)
- Precision changes (FP32 → FP16/INT8)
- Shape changes (dynamic → static)
- Removed/optimized-away layers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ONNXNodeInfo(BaseModel):
    """Information about an ONNX node."""

    model_config = ConfigDict(frozen=True)

    name: str
    op_type: str
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    # Shape info if available
    input_shapes: list[tuple[int, ...]] = Field(default_factory=list)
    output_shapes: list[tuple[int, ...]] = Field(default_factory=list)


class LayerMapping(BaseModel):
    """Mapping between ONNX nodes and TRT layers."""

    model_config = ConfigDict(frozen=True)

    trt_layer_name: str
    trt_layer_type: str
    trt_precision: str
    # ONNX nodes that were fused into this TRT layer
    onnx_nodes: list[str] = Field(default_factory=list)
    onnx_op_types: list[str] = Field(default_factory=list)
    # Fusion info
    is_fusion: bool = False
    fusion_description: str = ""


class ShapeChange(BaseModel):
    """Shape change between ONNX and TRT."""

    model_config = ConfigDict(frozen=True)

    tensor_name: str
    onnx_shape: tuple[int | str, ...] = ()  # May have dynamic dims like 'batch'
    trt_shape: tuple[int, ...] = ()
    is_dynamic_to_static: bool = False


class PrecisionChange(BaseModel):
    """Precision change for a layer."""

    model_config = ConfigDict(frozen=True)

    layer_name: str
    original_precision: str = "FP32"  # ONNX is typically FP32
    trt_precision: str = "FP32"
    reason: str = ""  # Why TRT chose this precision


class MemoryMetrics(BaseModel):
    """Memory comparison metrics between ONNX and TRT."""

    model_config = ConfigDict(frozen=True)

    onnx_file_size_bytes: int = 0
    trt_engine_size_bytes: int = 0
    trt_device_memory_bytes: int = 0
    # Compression metrics
    file_size_ratio: float = 1.0  # TRT/ONNX file size ratio
    # Estimated savings from precision changes
    estimated_precision_savings_bytes: int = 0
    estimated_precision_savings_ratio: float = 0.0


class TRTComparisonReport(BaseModel):
    """Full comparison report between ONNX and TRT engine."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    onnx_path: Path
    trt_path: Path
    # Summary stats
    onnx_node_count: int = 0
    trt_layer_count: int = 0
    fusion_count: int = 0
    removed_node_count: int = 0
    # Memory metrics
    memory_metrics: MemoryMetrics = Field(default_factory=MemoryMetrics)
    # Detailed mappings
    layer_mappings: list[LayerMapping] = Field(default_factory=list)
    shape_changes: list[ShapeChange] = Field(default_factory=list)
    precision_changes: list[PrecisionChange] = Field(default_factory=list)
    # Nodes that were completely removed (optimized away)
    removed_nodes: list[str] = Field(default_factory=list)
    # Nodes that couldn't be mapped
    unmapped_onnx_nodes: list[str] = Field(default_factory=list)
    unmapped_trt_layers: list[str] = Field(default_factory=list)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "ONNX vs TensorRT Comparison",
            "=" * 40,
            f"ONNX nodes: {self.onnx_node_count}",
            f"TRT layers: {self.trt_layer_count}",
            f"Compression ratio: {self.onnx_node_count / max(self.trt_layer_count, 1):.1f}x",
            "",
            f"Fusions: {self.fusion_count}",
            f"Removed nodes: {self.removed_node_count}",
            f"Precision changes: {len(self.precision_changes)}",
            f"Shape changes: {len(self.shape_changes)}",
        ]

        # Memory metrics
        mm = self.memory_metrics
        if mm.onnx_file_size_bytes > 0 and mm.trt_engine_size_bytes > 0:
            lines.extend(
                [
                    "",
                    "Memory:",
                    f"  ONNX file: {mm.onnx_file_size_bytes / 1024 / 1024:.1f} MB",
                    f"  TRT engine: {mm.trt_engine_size_bytes / 1024 / 1024:.1f} MB",
                    f"  File ratio: {mm.file_size_ratio:.2f}x",
                ]
            )
            if mm.trt_device_memory_bytes > 0:
                lines.append(f"  Device memory: {mm.trt_device_memory_bytes / 1024 / 1024:.1f} MB")
            if mm.estimated_precision_savings_ratio > 0:
                lines.append(
                    f"  Precision savings: ~{mm.estimated_precision_savings_ratio * 100:.0f}%"
                )

        if self.unmapped_onnx_nodes:
            lines.append(f"\nUnmapped ONNX nodes: {len(self.unmapped_onnx_nodes)}")

        return "\n".join(lines)


@dataclass
class TRTONNXComparator:
    """Compare ONNX model with TensorRT engine."""

    onnx_path: Path
    trt_path: Path
    # Cached data
    _onnx_nodes: dict[str, ONNXNodeInfo] = field(default_factory=dict)
    _trt_layers: list[Any] = field(default_factory=list)

    def compare(self) -> TRTComparisonReport:
        """
        Perform comparison between ONNX and TRT.

        Returns:
            TRTComparisonReport with detailed comparison data.
        """
        # Load ONNX graph
        onnx_nodes = self._load_onnx_nodes()

        # Load TRT layers
        trt_info = self._load_trt_info()

        # Perform mapping
        layer_mappings = []
        mapped_onnx = set()
        unmapped_trt = []

        for layer in trt_info.layers:
            mapping = self._map_trt_layer_to_onnx(layer, onnx_nodes)
            if mapping:
                layer_mappings.append(mapping)
                mapped_onnx.update(mapping.onnx_nodes)
            else:
                unmapped_trt.append(layer.name)

        # Find unmapped/removed ONNX nodes
        all_onnx_names = set(onnx_nodes.keys())
        unmapped_onnx = list(all_onnx_names - mapped_onnx)

        # Categorize unmapped nodes - some are truly removed optimizations
        removed_nodes = []
        truly_unmapped = []
        for name in unmapped_onnx:
            node = onnx_nodes.get(name)
            if node and self._is_optimization_removal(node):
                removed_nodes.append(name)
            else:
                truly_unmapped.append(name)

        # Detect precision changes
        precision_changes = self._detect_precision_changes(layer_mappings)

        # Detect shape changes
        shape_changes = self._detect_shape_changes(onnx_nodes, trt_info)

        # Count fusions
        fusion_count = sum(1 for m in layer_mappings if m.is_fusion)

        # Compute memory metrics
        memory_metrics = self._compute_memory_metrics(trt_info, precision_changes)

        return TRTComparisonReport(
            onnx_path=self.onnx_path,
            trt_path=self.trt_path,
            onnx_node_count=len(onnx_nodes),
            trt_layer_count=len(trt_info.layers),
            fusion_count=fusion_count,
            removed_node_count=len(removed_nodes),
            memory_metrics=memory_metrics,
            layer_mappings=layer_mappings,
            shape_changes=shape_changes,
            precision_changes=precision_changes,
            removed_nodes=removed_nodes,
            unmapped_onnx_nodes=truly_unmapped,
            unmapped_trt_layers=unmapped_trt,
        )

    def _load_onnx_nodes(self) -> dict[str, ONNXNodeInfo]:
        """Load ONNX graph nodes."""
        try:
            import onnx
        except ImportError:
            return {}

        model = onnx.load(str(self.onnx_path))
        nodes = {}

        for node in model.graph.node:
            nodes[node.name] = ONNXNodeInfo(
                name=node.name,
                op_type=node.op_type,
                inputs=list(node.input),
                outputs=list(node.output),
            )

        return nodes

    def _load_trt_info(self) -> Any:
        """Load TRT engine info."""
        from haoline.formats.tensorrt import TRTEngineReader

        reader = TRTEngineReader(self.trt_path)
        return reader.read()

    def _map_trt_layer_to_onnx(
        self,
        trt_layer: Any,
        onnx_nodes: dict[str, ONNXNodeInfo],
    ) -> LayerMapping | None:
        """Map a TRT layer back to its source ONNX node(s)."""
        matched_nodes = []
        matched_op_types = []

        # Strategy 1: Direct name matching
        # TRT often preserves ONNX node names or uses them as prefixes
        layer_name = trt_layer.name

        for onnx_name, onnx_node in onnx_nodes.items():
            # Check if ONNX name appears in TRT layer name
            if onnx_name in layer_name or layer_name.startswith(onnx_name):
                matched_nodes.append(onnx_name)
                matched_op_types.append(onnx_node.op_type)

        # Strategy 2: Check fused_ops if available
        if hasattr(trt_layer, "fused_ops") and trt_layer.fused_ops:
            for fused_name in trt_layer.fused_ops:
                # Clean up fused name
                clean_name = fused_name.strip()
                for onnx_name, onnx_node in onnx_nodes.items():
                    if clean_name in onnx_name or onnx_name in clean_name:
                        if onnx_name not in matched_nodes:
                            matched_nodes.append(onnx_name)
                            matched_op_types.append(onnx_node.op_type)

        # Strategy 3: Check origin field if available
        if hasattr(trt_layer, "origin") and trt_layer.origin:
            origin = trt_layer.origin
            if origin in onnx_nodes and origin not in matched_nodes:
                matched_nodes.append(origin)
                matched_op_types.append(onnx_nodes[origin].op_type)

        if not matched_nodes:
            return None

        # Determine if this is a fusion
        is_fusion = len(matched_nodes) > 1 or trt_layer.is_fused
        fusion_desc = ""
        if is_fusion:
            unique_ops = list(dict.fromkeys(matched_op_types))  # Preserve order
            fusion_desc = " + ".join(unique_ops)

        return LayerMapping(
            trt_layer_name=trt_layer.name,
            trt_layer_type=trt_layer.type,
            trt_precision=trt_layer.precision,
            onnx_nodes=matched_nodes,
            onnx_op_types=matched_op_types,
            is_fusion=is_fusion,
            fusion_description=fusion_desc,
        )

    def _is_optimization_removal(self, node: ONNXNodeInfo) -> bool:
        """Check if an ONNX node was likely removed as an optimization."""
        # Common ops that TRT removes/folds:
        removable_ops = {
            "Identity",
            "Dropout",  # Removed in inference
            "Shape",
            "Gather",  # Often folded into reshapes
            "Unsqueeze",
            "Squeeze",
            "Flatten",  # Often fused
            "Cast",  # Often absorbed
            "Constant",  # Folded
            "ConstantOfShape",
        }
        return node.op_type in removable_ops

    def _detect_precision_changes(self, mappings: list[LayerMapping]) -> list[PrecisionChange]:
        """Detect layers where precision changed from default FP32."""
        changes = []
        for mapping in mappings:
            if mapping.trt_precision not in ("FP32", "Mixed", "Unknown"):
                changes.append(
                    PrecisionChange(
                        layer_name=mapping.trt_layer_name,
                        original_precision="FP32",
                        trt_precision=mapping.trt_precision,
                        reason="TRT auto-selection or user config",
                    )
                )
        return changes

    def _detect_shape_changes(
        self,
        onnx_nodes: dict[str, ONNXNodeInfo],
        trt_info: Any,
    ) -> list[ShapeChange]:
        """Detect shape changes between ONNX and TRT."""
        changes = []

        # Compare input/output bindings
        for binding in trt_info.bindings:
            # Check if any dimension was -1 (dynamic) in ONNX but is now static
            trt_shape = binding.shape
            if -1 not in trt_shape:  # TRT shape is fully static
                # This suggests dynamic dims were resolved
                changes.append(
                    ShapeChange(
                        tensor_name=binding.name,
                        onnx_shape=(),  # Would need to extract from ONNX
                        trt_shape=trt_shape,
                        is_dynamic_to_static=True,
                    )
                )

        return changes

    def _compute_memory_metrics(
        self,
        trt_info: Any,
        precision_changes: list[PrecisionChange],
    ) -> MemoryMetrics:
        """Compute memory comparison metrics."""
        # Get file sizes
        onnx_size = self.onnx_path.stat().st_size if self.onnx_path.exists() else 0
        trt_size = self.trt_path.stat().st_size if self.trt_path.exists() else 0

        # File size ratio
        file_ratio = trt_size / onnx_size if onnx_size > 0 else 1.0

        # TRT device memory
        device_memory = (
            trt_info.device_memory_bytes if hasattr(trt_info, "device_memory_bytes") else 0
        )

        # Estimate precision savings
        # FP16 = 50% of FP32, INT8 = 25% of FP32
        precision_savings_ratio = 0.0
        total_layers = max(len(precision_changes), 1)
        for change in precision_changes:
            if change.trt_precision == "FP16":
                precision_savings_ratio += 0.5 / total_layers
            elif change.trt_precision == "INT8":
                precision_savings_ratio += 0.75 / total_layers

        # Estimate bytes saved (rough approximation based on model size)
        estimated_savings = int(onnx_size * precision_savings_ratio)

        return MemoryMetrics(
            onnx_file_size_bytes=onnx_size,
            trt_engine_size_bytes=trt_size,
            trt_device_memory_bytes=device_memory,
            file_size_ratio=file_ratio,
            estimated_precision_savings_bytes=estimated_savings,
            estimated_precision_savings_ratio=precision_savings_ratio,
        )


def compare_onnx_trt(onnx_path: str | Path, trt_path: str | Path) -> TRTComparisonReport:
    """
    Compare an ONNX model with its compiled TensorRT engine.

    Args:
        onnx_path: Path to the source ONNX model.
        trt_path: Path to the compiled TensorRT engine.

    Returns:
        TRTComparisonReport with detailed comparison data.
    """
    comparator = TRTONNXComparator(
        onnx_path=Path(onnx_path),
        trt_path=Path(trt_path),
    )
    return comparator.compare()
