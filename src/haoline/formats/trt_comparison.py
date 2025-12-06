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


class LayerRewrite(BaseModel):
    """Represents a layer rewrite optimization by TensorRT.

    TensorRT can replace patterns of operations with optimized implementations:
    - Multi-head attention → Flash Attention kernel
    - LayerNorm → optimized fused kernel
    - GELU → fast GELU approximation
    - Softmax + Scale → ScaledSoftmax kernel
    """

    model_config = ConfigDict(frozen=True)

    rewrite_type: str  # "FlashAttention", "FusedLayerNorm", "FastGELU", etc.
    original_ops: list[str] = Field(default_factory=list)  # ONNX op names
    original_op_types: list[str] = Field(default_factory=list)  # ONNX op types
    trt_layer_name: str = ""  # Resulting TRT layer
    trt_kernel: str = ""  # Kernel/tactic used (if available)
    description: str = ""  # Human-readable description
    speedup_estimate: str = ""  # "2-4x faster", "memory efficient", etc.


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
    # Layer rewrites (attention → Flash Attention, etc.)
    layer_rewrites: list[LayerRewrite] = Field(default_factory=list)
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
            f"Layer rewrites: {len(self.layer_rewrites)}",
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

        # Layer rewrites summary
        if self.layer_rewrites:
            lines.extend(["", "Layer Rewrites:"])
            for rewrite in self.layer_rewrites[:5]:  # Show top 5
                lines.append(f"  {rewrite.rewrite_type}: {rewrite.description}")
            if len(self.layer_rewrites) > 5:
                lines.append(f"  ... and {len(self.layer_rewrites) - 5} more")

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

        # Detect layer rewrites (attention → Flash Attention, etc.)
        layer_rewrites = self._detect_layer_rewrites(onnx_nodes, trt_info, layer_mappings)

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
            layer_rewrites=layer_rewrites,
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

    def _detect_layer_rewrites(
        self,
        onnx_nodes: dict[str, ONNXNodeInfo],
        trt_info: Any,
        layer_mappings: list[LayerMapping],
    ) -> list[LayerRewrite]:
        """
        Detect layer rewrites where TRT replaced ONNX patterns with optimized kernels.

        Known TensorRT optimizations:
        - Multi-head attention → Flash Attention (fMHA kernel)
        - LayerNorm → Fused LayerNorm kernel
        - GELU → Fast GELU approximation
        - Softmax patterns → Optimized softmax
        - Transpose + MatMul → Fused kernel
        """
        rewrites: list[LayerRewrite] = []

        # Pattern definitions: (onnx_op_pattern, rewrite_type, description, speedup)
        REWRITE_PATTERNS: list[tuple[set[str], str, str, str]] = [
            # Multi-head attention pattern
            (
                {"MatMul", "Softmax", "Transpose"},
                "FlashAttention",
                "Multi-head attention fused to Flash Attention kernel",
                "2-4x faster, memory efficient",
            ),
            (
                {"MatMul", "Softmax", "Reshape", "Transpose"},
                "FlashAttention",
                "Multi-head attention with reshape fused to Flash Attention",
                "2-4x faster, memory efficient",
            ),
            # LayerNorm pattern
            (
                {"ReduceMean", "Sub", "Pow", "Mul", "Add"},
                "FusedLayerNorm",
                "Layer normalization fused to optimized kernel",
                "1.5-2x faster",
            ),
            (
                {"InstanceNormalization"},
                "FusedInstanceNorm",
                "Instance normalization using optimized kernel",
                "1.3-1.5x faster",
            ),
            # GELU patterns
            (
                {"Erf", "Mul", "Add"},
                "FastGELU",
                "GELU activation using fast approximation",
                "1.2-1.5x faster",
            ),
            (
                {"Tanh", "Mul", "Add", "Pow"},
                "FastGELU",
                "GELU (tanh approximation) optimized",
                "1.2-1.5x faster",
            ),
            # Softmax optimizations
            (
                {"Softmax", "Mul"},
                "ScaledSoftmax",
                "Scaled softmax fused kernel",
                "1.2x faster",
            ),
            # GroupNorm
            (
                {"Reshape", "InstanceNormalization"},
                "FusedGroupNorm",
                "Group normalization fused to single kernel",
                "1.3-1.5x faster",
            ),
            # QKV projection fusion
            (
                {"MatMul", "Concat"},
                "FusedQKVProjection",
                "Query/Key/Value projections fused",
                "1.5-2x faster",
            ),
        ]

        # TRT kernel name patterns that indicate specific optimizations
        TRT_KERNEL_PATTERNS: dict[str, tuple[str, str, str]] = {
            "fmha": ("FlashAttention", "Flash/Fused Multi-Head Attention kernel", "2-4x faster"),
            "flash": ("FlashAttention", "Flash Attention kernel", "2-4x faster, memory efficient"),
            "mha": ("FusedMHA", "Fused Multi-Head Attention", "1.5-2x faster"),
            "layernorm": ("FusedLayerNorm", "Fused LayerNorm kernel", "1.5-2x faster"),
            "ln": ("FusedLayerNorm", "Fused LayerNorm kernel", "1.5-2x faster"),
            "gelu": ("FastGELU", "Fast GELU kernel", "1.2-1.5x faster"),
            "swish": ("FusedSwish", "Fused Swish/SiLU activation", "1.2x faster"),
            "groupnorm": ("FusedGroupNorm", "Fused GroupNorm kernel", "1.3-1.5x faster"),
            "gn": ("FusedGroupNorm", "Fused GroupNorm kernel", "1.3-1.5x faster"),
        }

        # Check TRT layer names/tactics for known optimization patterns
        for layer in trt_info.layers:
            layer_name_lower = layer.name.lower()
            tactic_lower = (layer.tactic or "").lower()

            for pattern, (rewrite_type, desc, speedup) in TRT_KERNEL_PATTERNS.items():
                if pattern in layer_name_lower or pattern in tactic_lower:
                    # Find corresponding ONNX ops from mappings
                    original_ops: list[str] = []
                    original_types: list[str] = []
                    for mapping in layer_mappings:
                        if mapping.trt_layer_name == layer.name:
                            original_ops = mapping.onnx_nodes
                            original_types = mapping.onnx_op_types
                            break

                    rewrites.append(
                        LayerRewrite(
                            rewrite_type=rewrite_type,
                            original_ops=original_ops,
                            original_op_types=original_types,
                            trt_layer_name=layer.name,
                            trt_kernel=layer.tactic or "",
                            description=desc,
                            speedup_estimate=speedup,
                        )
                    )
                    break

        # Check for pattern-based rewrites in layer mappings
        for mapping in layer_mappings:
            if not mapping.is_fusion or len(mapping.onnx_op_types) < 2:
                continue

            op_types_set = set(mapping.onnx_op_types)

            for pattern_ops, rewrite_type, desc, speedup in REWRITE_PATTERNS:
                # Check if the mapping contains this pattern
                if pattern_ops.issubset(op_types_set):
                    # Avoid duplicates from TRT kernel detection
                    already_added = any(
                        r.trt_layer_name == mapping.trt_layer_name for r in rewrites
                    )
                    if not already_added:
                        rewrites.append(
                            LayerRewrite(
                                rewrite_type=rewrite_type,
                                original_ops=mapping.onnx_nodes,
                                original_op_types=mapping.onnx_op_types,
                                trt_layer_name=mapping.trt_layer_name,
                                trt_kernel="",
                                description=desc,
                                speedup_estimate=speedup,
                            )
                        )
                        break

        return rewrites

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


def generate_comparison_html(report: TRTComparisonReport) -> str:
    """
    Generate an interactive HTML report with side-by-side ONNX vs TRT comparison.

    Task 22.3.6: Side-by-side graph comparison in HTML report.

    The HTML shows:
    - Summary metrics at top
    - Side-by-side view: ONNX ops on left, TRT layers on right
    - Lines connecting related nodes (fusions highlighted)
    - Color coding for precision changes
    - Layer rewrite visualization

    Args:
        report: TRTComparisonReport from compare_onnx_trt()

    Returns:
        HTML string that can be saved to a file.
    """
    import json

    # Prepare data for JavaScript
    onnx_nodes_data = []
    for mapping in report.layer_mappings:
        for onnx_name in mapping.onnx_nodes:
            op_type = ""
            for idx, name in enumerate(mapping.onnx_nodes):
                if name == onnx_name and idx < len(mapping.onnx_op_types):
                    op_type = mapping.onnx_op_types[idx]
                    break
            onnx_nodes_data.append(
                {
                    "name": onnx_name,
                    "op_type": op_type,
                    "mapped_to": mapping.trt_layer_name,
                    "is_fused": mapping.is_fusion,
                }
            )

    # Add unmapped nodes
    for name in report.unmapped_onnx_nodes:
        onnx_nodes_data.append(
            {
                "name": name,
                "op_type": "Unknown",
                "mapped_to": None,
                "is_fused": False,
            }
        )

    # Add removed nodes
    for name in report.removed_nodes:
        onnx_nodes_data.append(
            {
                "name": name,
                "op_type": "Removed",
                "mapped_to": "__removed__",
                "is_fused": False,
            }
        )

    trt_layers_data = []
    for mapping in report.layer_mappings:
        trt_layers_data.append(
            {
                "name": mapping.trt_layer_name,
                "type": mapping.trt_layer_type,
                "precision": mapping.trt_precision,
                "onnx_sources": mapping.onnx_nodes,
                "is_fusion": mapping.is_fusion,
                "fusion_desc": mapping.fusion_description,
            }
        )

    # Add unmapped TRT layers
    for name in report.unmapped_trt_layers:
        trt_layers_data.append(
            {
                "name": name,
                "type": "Unknown",
                "precision": "Unknown",
                "onnx_sources": [],
                "is_fusion": False,
                "fusion_desc": "",
            }
        )

    # Layer rewrites data
    rewrites_data = []
    if report.layer_rewrites:
        for r in report.layer_rewrites:
            rewrites_data.append(
                {
                    "type": r.rewrite_type,
                    "description": r.description,
                    "speedup": r.speedup_estimate,
                    "original_ops": r.original_ops,
                    "trt_layer": r.trt_layer_name,
                }
            )

    mm = report.memory_metrics

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ONNX vs TensorRT Comparison - HaoLine</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-dark: #0d1117;
            --bg-card: #161b22;
            --bg-hover: #21262d;
            --border: #30363d;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --accent-blue: #58a6ff;
            --accent-green: #3fb950;
            --accent-orange: #d29922;
            --accent-red: #f85149;
            --accent-purple: #a371f7;
            --accent-teal: #39d353;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            line-height: 1.5;
        }}

        .header {{
            background: var(--bg-card);
            border-bottom: 1px solid var(--border);
            padding: 24px 32px;
        }}

        .header h1 {{
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 8px;
        }}

        .header .subtitle {{
            color: var(--text-secondary);
            font-size: 14px;
        }}

        .metrics-row {{
            display: flex;
            gap: 16px;
            padding: 24px 32px;
            flex-wrap: wrap;
        }}

        .metric-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px 20px;
            min-width: 160px;
        }}

        .metric-card .label {{
            color: var(--text-secondary);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }}

        .metric-card .value {{
            font-size: 24px;
            font-weight: 600;
        }}

        .metric-card .value.green {{ color: var(--accent-green); }}
        .metric-card .value.blue {{ color: var(--accent-blue); }}
        .metric-card .value.orange {{ color: var(--accent-orange); }}
        .metric-card .value.purple {{ color: var(--accent-purple); }}

        .comparison-container {{
            display: flex;
            padding: 24px 32px;
            gap: 32px;
            min-height: calc(100vh - 250px);
        }}

        .side-panel {{
            flex: 1;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }}

        .panel-header {{
            padding: 16px 20px;
            border-bottom: 1px solid var(--border);
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .panel-header .badge {{
            background: var(--bg-hover);
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
            color: var(--text-secondary);
        }}

        .node-list {{
            flex: 1;
            overflow-y: auto;
            padding: 12px;
        }}

        .node-item {{
            padding: 10px 14px;
            border-radius: 8px;
            margin-bottom: 4px;
            cursor: pointer;
            transition: background 0.15s;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .node-item:hover {{
            background: var(--bg-hover);
        }}

        .node-item.highlight {{
            background: rgba(88, 166, 255, 0.15);
            border: 1px solid var(--accent-blue);
        }}

        .node-item.fusion {{
            border-left: 3px solid var(--accent-purple);
        }}

        .node-item.removed {{
            opacity: 0.5;
            text-decoration: line-through;
        }}

        .node-name {{
            font-size: 13px;
            font-weight: 500;
            word-break: break-all;
        }}

        .node-type {{
            font-size: 11px;
            color: var(--text-secondary);
            margin-top: 2px;
        }}

        .precision-badge {{
            font-size: 10px;
            font-weight: 600;
            padding: 3px 8px;
            border-radius: 4px;
            text-transform: uppercase;
        }}

        .precision-badge.fp32 {{ background: var(--bg-hover); color: var(--text-secondary); }}
        .precision-badge.fp16 {{ background: rgba(88, 166, 255, 0.2); color: var(--accent-blue); }}
        .precision-badge.int8 {{ background: rgba(63, 185, 80, 0.2); color: var(--accent-green); }}
        .precision-badge.mixed {{ background: rgba(210, 153, 34, 0.2); color: var(--accent-orange); }}

        .rewrites-section {{
            padding: 24px 32px;
            border-top: 1px solid var(--border);
        }}

        .rewrites-section h3 {{
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 16px;
        }}

        .rewrite-item {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
        }}

        .rewrite-item .type {{
            font-weight: 600;
            color: var(--accent-purple);
            margin-bottom: 4px;
        }}

        .rewrite-item .desc {{
            color: var(--text-secondary);
            font-size: 13px;
        }}

        .rewrite-item .speedup {{
            margin-top: 8px;
            font-size: 12px;
            color: var(--accent-green);
        }}

        .connection-canvas {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 1000;
        }}

        .legend {{
            padding: 16px 32px;
            display: flex;
            gap: 24px;
            flex-wrap: wrap;
            background: var(--bg-card);
            border-top: 1px solid var(--border);
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
            color: var(--text-secondary);
        }}

        .legend-color {{
            width: 12px;
            height: 12px;
            border-radius: 3px;
        }}

        .search-box {{
            padding: 12px;
            border-bottom: 1px solid var(--border);
        }}

        .search-box input {{
            width: 100%;
            padding: 10px 14px;
            background: var(--bg-dark);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text-primary);
            font-size: 13px;
        }}

        .search-box input:focus {{
            outline: none;
            border-color: var(--accent-blue);
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>ONNX vs TensorRT Comparison</h1>
        <div class="subtitle">
            <strong>ONNX:</strong> {report.onnx_path.name} |
            <strong>TRT:</strong> {report.trt_path.name}
        </div>
    </div>

    <div class="metrics-row">
        <div class="metric-card">
            <div class="label">ONNX Nodes</div>
            <div class="value">{report.onnx_node_count:,}</div>
        </div>
        <div class="metric-card">
            <div class="label">TRT Layers</div>
            <div class="value blue">{report.trt_layer_count:,}</div>
        </div>
        <div class="metric-card">
            <div class="label">Compression</div>
            <div class="value green">{
        report.onnx_node_count / max(report.trt_layer_count, 1):.1f}x</div>
        </div>
        <div class="metric-card">
            <div class="label">Fusions</div>
            <div class="value purple">{report.fusion_count}</div>
        </div>
        <div class="metric-card">
            <div class="label">Layer Rewrites</div>
            <div class="value orange">{len(report.layer_rewrites)}</div>
        </div>
        <div class="metric-card">
            <div class="label">File Size Ratio</div>
            <div class="value">{mm.file_size_ratio:.2f}x</div>
        </div>
    </div>

    <div class="comparison-container">
        <div class="side-panel" id="onnx-panel">
            <div class="panel-header">
                ONNX Graph
                <span class="badge">{report.onnx_node_count} nodes</span>
            </div>
            <div class="search-box">
                <input type="text" id="onnx-search" placeholder="Search ONNX nodes...">
            </div>
            <div class="node-list" id="onnx-nodes"></div>
        </div>

        <div class="side-panel" id="trt-panel">
            <div class="panel-header">
                TensorRT Engine
                <span class="badge">{report.trt_layer_count} layers</span>
            </div>
            <div class="search-box">
                <input type="text" id="trt-search" placeholder="Search TRT layers...">
            </div>
            <div class="node-list" id="trt-nodes"></div>
        </div>
    </div>

    {
        ""
        if not report.layer_rewrites
        else '''
    <div class="rewrites-section">
        <h3>Layer Rewrites (Optimized Kernel Substitutions)</h3>
        <div id="rewrites-list"></div>
    </div>
    '''
    }

    <div class="legend">
        <div class="legend-item">
            <div class="legend-color" style="background: var(--accent-purple);"></div>
            Fused operation
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: var(--accent-green);"></div>
            INT8 precision
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: var(--accent-blue);"></div>
            FP16 precision
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: var(--text-secondary);"></div>
            Removed/Optimized away
        </div>
    </div>

    <script>
        const onnxNodes = {json.dumps(onnx_nodes_data)};
        const trtLayers = {json.dumps(trt_layers_data)};
        const rewrites = {json.dumps(rewrites_data)};

        // Render ONNX nodes
        function renderOnnxNodes(filter = '') {{
            const container = document.getElementById('onnx-nodes');
            container.innerHTML = '';

            const filtered = filter
                ? onnxNodes.filter(n => n.name.toLowerCase().includes(filter.toLowerCase()))
                : onnxNodes;

            filtered.forEach(node => {{
                const div = document.createElement('div');
                div.className = 'node-item';
                if (node.is_fused) div.classList.add('fusion');
                if (node.mapped_to === '__removed__') div.classList.add('removed');
                div.dataset.name = node.name;
                div.dataset.mappedTo = node.mapped_to || '';

                div.innerHTML = `
                    <div>
                        <div class="node-name">${{node.name}}</div>
                        <div class="node-type">${{node.op_type}}</div>
                    </div>
                `;

                div.addEventListener('mouseenter', () => highlightConnection(node.name, node.mapped_to));
                div.addEventListener('mouseleave', clearHighlights);

                container.appendChild(div);
            }});
        }}

        // Render TRT layers
        function renderTrtLayers(filter = '') {{
            const container = document.getElementById('trt-nodes');
            container.innerHTML = '';

            const filtered = filter
                ? trtLayers.filter(l => l.name.toLowerCase().includes(filter.toLowerCase()))
                : trtLayers;

            filtered.forEach(layer => {{
                const div = document.createElement('div');
                div.className = 'node-item';
                if (layer.is_fusion) div.classList.add('fusion');
                div.dataset.name = layer.name;
                div.dataset.sources = JSON.stringify(layer.onnx_sources);

                const precClass = layer.precision.toLowerCase().replace('mixed', 'mixed');

                div.innerHTML = `
                    <div>
                        <div class="node-name">${{layer.name.slice(0, 60)}}${{layer.name.length > 60 ? '...' : ''}}</div>
                        <div class="node-type">${{layer.type}}${{layer.is_fusion ? ' (' + layer.fusion_desc + ')' : ''}}</div>
                    </div>
                    <span class="precision-badge ${{precClass}}">${{layer.precision}}</span>
                `;

                div.addEventListener('mouseenter', () => highlightSources(layer.name, layer.onnx_sources));
                div.addEventListener('mouseleave', clearHighlights);

                container.appendChild(div);
            }});
        }}

        // Render rewrites
        function renderRewrites() {{
            const container = document.getElementById('rewrites-list');
            if (!container) return;

            rewrites.forEach(r => {{
                const div = document.createElement('div');
                div.className = 'rewrite-item';
                div.innerHTML = `
                    <div class="type">${{r.type}}</div>
                    <div class="desc">${{r.description}}</div>
                    <div class="speedup">Estimated: ${{r.speedup}}</div>
                `;
                container.appendChild(div);
            }});
        }}

        // Highlight connections
        function highlightConnection(onnxName, trtName) {{
            clearHighlights();

            // Highlight ONNX node
            document.querySelectorAll('#onnx-nodes .node-item').forEach(el => {{
                if (el.dataset.name === onnxName) el.classList.add('highlight');
            }});

            // Highlight TRT layer
            if (trtName && trtName !== '__removed__') {{
                document.querySelectorAll('#trt-nodes .node-item').forEach(el => {{
                    if (el.dataset.name === trtName) el.classList.add('highlight');
                }});
            }}
        }}

        function highlightSources(trtName, sources) {{
            clearHighlights();

            // Highlight TRT layer
            document.querySelectorAll('#trt-nodes .node-item').forEach(el => {{
                if (el.dataset.name === trtName) el.classList.add('highlight');
            }});

            // Highlight all ONNX sources
            document.querySelectorAll('#onnx-nodes .node-item').forEach(el => {{
                if (sources.includes(el.dataset.name)) el.classList.add('highlight');
            }});
        }}

        function clearHighlights() {{
            document.querySelectorAll('.node-item.highlight').forEach(el => {{
                el.classList.remove('highlight');
            }});
        }}

        // Search handlers
        document.getElementById('onnx-search').addEventListener('input', (e) => {{
            renderOnnxNodes(e.target.value);
        }});

        document.getElementById('trt-search').addEventListener('input', (e) => {{
            renderTrtLayers(e.target.value);
        }});

        // Initial render
        renderOnnxNodes();
        renderTrtLayers();
        renderRewrites();
    </script>
</body>
</html>"""

    return html
