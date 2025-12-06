# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
Report generation and ModelInspector orchestrator for HaoLine.

This module contains:
- InspectionReport: The main data structure holding all analysis results
- ModelInspector: Orchestrator that coordinates all analysis components
"""

from __future__ import annotations

import json
import logging
import pathlib
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from .analyzer import (
        MetricsEngine,
        ONNXGraphLoader,
    )
    from .patterns import PatternAnalyzer
    from .risks import RiskAnalyzer


class ModelMetadata(BaseModel):
    """Basic model metadata extracted from ONNX proto."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: str
    ir_version: int
    producer_name: str
    producer_version: str
    domain: str
    model_version: int
    doc_string: str
    opsets: dict[str, int]  # domain -> version


class GraphSummary(BaseModel):
    """Summary statistics about the ONNX graph."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    num_nodes: int
    num_inputs: int
    num_outputs: int
    num_initializers: int
    input_shapes: dict[str, list[int | str]]  # name -> shape (may have symbolic dims)
    output_shapes: dict[str, list[int | str]]
    op_type_counts: dict[str, int]  # op_type -> count


class DatasetInfo(BaseModel):
    """Dataset and class information extracted from model metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    task: str | None = None  # "detect", "classify", "segment", etc.
    num_classes: int | None = None
    class_names: list[str] = Field(default_factory=list)
    source: str | None = None  # "ultralytics", "output_shape", etc.


def infer_num_classes_from_output(
    output_shapes: dict[str, list[int | str]],
    architecture_type: str = "unknown",
) -> DatasetInfo | None:
    """
    Infer number of classes from model output shapes.

    Analyzes output tensor shapes to detect common patterns:
    - Classification: [batch, num_classes] or [batch, 1, num_classes]
    - Detection (YOLO-style): [batch, num_boxes, 4+num_classes] or [batch, num_boxes, 5+num_classes]
    - Segmentation: [batch, num_classes, height, width]

    Args:
        output_shapes: Dictionary mapping output names to their shapes.
        architecture_type: Detected architecture type (helps disambiguate patterns).

    Returns:
        DatasetInfo with inferred num_classes and task, or None if inference failed.
    """
    if not output_shapes:
        return None

    # Get the primary output (usually the first one, or look for common names)
    primary_output = None
    primary_shape = None

    # Priority: look for outputs named 'output', 'logits', 'predictions', etc.
    # Use exact match or prefix match to avoid matching "some_random_output" with "output"
    priority_names = ["logits", "predictions", "probs", "classes", "output0", "output"]
    for name in priority_names:
        for out_name, shape in output_shapes.items():
            out_lower = out_name.lower()
            # Exact match or starts with the priority name
            if out_lower == name or out_lower.startswith(name + "_"):
                primary_output = out_name
                primary_shape = shape
                break
        if primary_output:
            break

    # Fallback to first output
    if not primary_output:
        primary_output = next(iter(output_shapes.keys()))
        primary_shape = output_shapes[primary_output]

    if not primary_shape:
        return None

    # Convert shape to list of ints where possible (handle symbolic dims)
    def to_int(dim):
        if isinstance(dim, int):
            return dim
        if isinstance(dim, str):
            # Try to parse as int, otherwise return None
            try:
                return int(dim)
            except ValueError:
                return None
        return None

    shape = [to_int(d) for d in primary_shape]

    # Need at least 2 dimensions
    if len(shape) < 2:
        return None

    # Classification pattern: [batch, num_classes] or [batch, 1, num_classes]
    # Typical num_classes: 2-10000 (ImageNet=1000, CIFAR=10/100, etc.)
    if len(shape) == 2:
        # [batch, num_classes]
        _batch, num_classes = shape
        if isinstance(num_classes, int) and 2 <= num_classes <= 10000:
            return DatasetInfo(
                task="classify",
                num_classes=num_classes,
                source="output_shape",
            )

    if len(shape) == 3:
        _batch, dim1, dim2 = shape
        # Could be [batch, 1, num_classes] for classification
        if isinstance(dim1, int) and isinstance(dim2, int):
            if dim1 == 1 and 2 <= dim2 <= 10000:
                return DatasetInfo(
                    task="classify",
                    num_classes=dim2,
                    source="output_shape",
                )
            # Could be [batch, num_boxes, 4+nc] or [batch, num_boxes, 5+nc] for detection
            # YOLO format: boxes * (x, y, w, h, obj_conf, class_probs...)
            # Common box counts: 8400, 25200, etc. (depends on input size)
            # Detection heuristic: many boxes, last dim is 4+nc or 5+nc
            # Minimum: 4 box coords + 1 class = 5
            if dim1 >= 100 and dim2 >= 5:
                # Try to infer num_classes from detection output
                # Format could be: [x, y, w, h, class1, class2, ...] (4 + nc) - YOLOv8 format
                # Or: [x, y, w, h, obj_conf, class1, class2, ...] (5 + nc) - YOLOv5 format
                # Assume YOLOv8 format (no obj_conf) which is more common now
                nc = dim2 - 4
                if nc >= 1:  # At least 1 class
                    return DatasetInfo(
                        task="detect",
                        num_classes=nc,
                        source="output_shape",
                    )

    if len(shape) == 4:
        _batch, dim1, dim2, dim3 = shape
        # Segmentation pattern: [batch, num_classes, height, width]
        # Height/width are typically >= 32 and often equal
        if (
            isinstance(dim1, int)
            and isinstance(dim2, int)
            and isinstance(dim3, int)
            and 2 <= dim1 <= 1000  # num_classes
            and dim2 >= 32  # height
            and dim3 >= 32  # width
        ):
            # Additional check: h/w should be similar (typical segmentation output)
            ratio = max(dim2, dim3) / min(dim2, dim3) if min(dim2, dim3) > 0 else 999
            if ratio <= 4:  # Reasonable aspect ratio
                return DatasetInfo(
                    task="segment",
                    num_classes=dim1,
                    source="output_shape",
                )

    return None


class InspectionReport(BaseModel):
    """
    Complete inspection report for an ONNX model.

    This is the primary output of ModelInspector.inspect() and contains
    all analysis results in a structured format suitable for JSON serialization
    or Markdown rendering.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Metadata
    metadata: ModelMetadata
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    autodoc_version: str = "0.1.0"

    # Graph structure
    graph_summary: GraphSummary | None = None

    # Metrics
    param_counts: Any | None = None  # ParamCounts (not yet Pydantic)
    flop_counts: Any | None = None  # FlopCounts (not yet Pydantic)
    memory_estimates: Any | None = None  # MemoryEstimates (not yet Pydantic)

    # Patterns
    detected_blocks: list[Any] = Field(default_factory=list)  # list[Block]
    architecture_type: str = "unknown"  # "transformer", "cnn", "mlp", "hybrid", "unknown"

    # Risks
    risk_signals: list[Any] = Field(default_factory=list)  # list[RiskSignal]

    # Hardware estimates (optional, set by CLI if --hardware specified)
    hardware_profile: Any | None = None  # HardwareProfile (not yet Pydantic)
    hardware_estimates: Any | None = None  # HardwareEstimates (not yet Pydantic)

    # System Requirements & Scaling (Epic 6C)
    system_requirements: Any | None = None  # SystemRequirements
    batch_size_sweep: Any | None = None  # BatchSizeSweep
    resolution_sweep: Any | None = None  # ResolutionSweep

    # LLM summary (optional, set by CLI if --llm-summary specified)
    llm_summary: dict[str, Any] | None = None

    # Dataset info (optional, extracted from model metadata)
    dataset_info: DatasetInfo | None = None

    # Extra data (profiling results, GPU metrics, etc.)
    extra_data: dict[str, Any] | None = None

    # Universal IR (optional, format-agnostic graph representation)
    # Enables cross-format comparison, structural diff, and advanced visualization
    universal_graph: Any | None = None  # UniversalGraph

    # Quantization analysis (optional, set by CLI if --lint-quantization specified)
    quantization_lint: Any | None = None  # QuantizationLintResult

    def to_dict(self) -> dict[str, Any]:
        """Convert report to a JSON-serializable dictionary."""
        import numpy as np

        def _serialize(obj: Any, depth: int = 0) -> Any:
            """Recursively serialize objects to JSON-compatible types."""
            # Prevent infinite recursion
            if depth > 50:
                return str(obj)

            if obj is None:
                return None
            if isinstance(obj, (str, int, float, bool)):
                return obj
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            # Handle UniversalGraph specially (Pydantic model with to_dict method)
            if hasattr(obj, "to_dict") and hasattr(obj, "num_nodes"):
                # This is a UniversalGraph - serialize without weights
                return obj.to_dict(include_weights=False)
            # Handle Pydantic models (including self and nested models)
            if hasattr(obj, "model_dump"):
                # Use model_dump but recursively serialize for numpy/special types
                dumped = obj.model_dump()
                return _serialize(dumped, depth + 1)
            # Handle dataclasses (for types not yet converted to Pydantic)
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _serialize(getattr(obj, k), depth + 1) for k in obj.__dataclass_fields__}
            if isinstance(obj, list):
                return [_serialize(item, depth + 1) for item in obj]
            if isinstance(obj, dict):
                return {str(k): _serialize(v, depth + 1) for k, v in obj.items()}
            # Fallback: convert to string
            return str(obj)

        # Start serialization from this model's fields
        result: dict[str, Any] = {}
        for field_name in type(self).model_fields:
            value = getattr(self, field_name)
            result[field_name] = _serialize(value)
        return result

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def validate_schema(self) -> tuple[bool, list[str]]:
        """
        Validate this report against the JSON schema.

        Returns:
            Tuple of (is_valid, list of error messages).
            If jsonschema is not installed, returns (True, []) with a warning.

        Example:
            report = inspector.inspect("model.onnx")
            is_valid, errors = report.validate_schema()
            if not is_valid:
                for error in errors:
                    print(f"Validation error: {error}")
        """
        from .schema import validate_report

        return validate_report(self.to_dict())

    def validate_schema_strict(self) -> None:
        """
        Validate this report, raising ValidationError on failure.

        Raises:
            ValidationError: If validation fails with details of errors.

        Example:
            report = inspector.inspect("model.onnx")
            try:
                report.validate_schema_strict()
                print("Report is valid!")
            except ValidationError as e:
                print(f"Invalid report: {e.errors}")
        """
        from .schema import validate_report_strict

        validate_report_strict(self.to_dict())

    def to_markdown(self) -> str:
        """Generate a Markdown model card from this report."""
        lines = []

        # Header
        model_name = pathlib.Path(self.metadata.path).stem
        lines.append(f"# Model Card: {model_name}")
        lines.append("")
        lines.append(f"*Generated by HaoLine v{self.autodoc_version} on {self.generated_at}*")
        lines.append("")

        # Executive Summary (if LLM summary available)
        if self.llm_summary and self.llm_summary.get("success"):
            lines.append("## Executive Summary")
            lines.append("")
            if self.llm_summary.get("short_summary"):
                lines.append(f"{self.llm_summary['short_summary']}")
                lines.append("")
            if self.llm_summary.get("detailed_summary"):
                lines.append("")
                lines.append(self.llm_summary["detailed_summary"])
                lines.append("")
            if self.llm_summary.get("model"):
                lines.append(f"*Generated by {self.llm_summary['model']}*")
                lines.append("")

        # Metadata section
        lines.append("## Metadata")
        lines.append("")
        lines.append("| Property | Value |")
        lines.append("|----------|-------|")
        lines.append(f"| IR Version | {self.metadata.ir_version} |")
        lines.append(
            f"| Producer | {self.metadata.producer_name} {self.metadata.producer_version} |"
        )
        if self.metadata.opsets:
            opset_str = ", ".join(f"{d}:{v}" for d, v in self.metadata.opsets.items())
            lines.append(f"| Opsets | {opset_str} |")
        lines.append("")

        # Graph summary
        if self.graph_summary:
            lines.append("## Graph Summary")
            lines.append("")
            lines.append(f"- **Nodes**: {self.graph_summary.num_nodes}")
            lines.append(f"- **Inputs**: {self.graph_summary.num_inputs}")
            lines.append(f"- **Outputs**: {self.graph_summary.num_outputs}")
            lines.append(f"- **Initializers**: {self.graph_summary.num_initializers}")
            lines.append("")

            # Input/output shapes
            if self.graph_summary.input_shapes:
                lines.append("### Inputs")
                lines.append("")
                for name, shape in self.graph_summary.input_shapes.items():
                    lines.append(f"- `{name}`: {shape}")
                lines.append("")

            if self.graph_summary.output_shapes:
                lines.append("### Outputs")
                lines.append("")
                for name, shape in self.graph_summary.output_shapes.items():
                    lines.append(f"- `{name}`: {shape}")
                lines.append("")

            # Top operators
            if self.graph_summary.op_type_counts:
                lines.append("### Operator Distribution")
                lines.append("")
                lines.append("| Operator | Count |")
                lines.append("|----------|-------|")
                sorted_ops = sorted(self.graph_summary.op_type_counts.items(), key=lambda x: -x[1])
                for op, count in sorted_ops[:15]:  # Top 15
                    lines.append(f"| {op} | {count} |")
                if len(sorted_ops) > 15:
                    lines.append(f"| ... | ({len(sorted_ops) - 15} more) |")
                lines.append("")

        # Metrics
        if self.param_counts or self.flop_counts or self.memory_estimates:
            lines.append("## Complexity Metrics")
            lines.append("")

            if self.param_counts:
                lines.append(
                    f"- **Total Parameters**: {self._format_number(self.param_counts.total)}"
                )
                lines.append(f"  - Trainable: {self._format_number(self.param_counts.trainable)}")
                lines.append(
                    f"  - Non-trainable: {self._format_number(self.param_counts.non_trainable)}"
                )

                # Shared weights info
                if self.param_counts.num_shared_weights > 0:
                    lines.append(
                        f"  - Shared Weights: {self.param_counts.num_shared_weights} "
                        f"(used by multiple nodes)"
                    )

                # Precision breakdown
                if self.param_counts.precision_breakdown:
                    breakdown_parts = [
                        f"{dtype}: {self._format_number(count)}"
                        for dtype, count in sorted(
                            self.param_counts.precision_breakdown.items(),
                            key=lambda x: -x[1],
                        )[:4]
                    ]
                    if breakdown_parts:
                        lines.append(f"  - By Precision: {', '.join(breakdown_parts)}")

                # Quantization info
                if self.param_counts.is_quantized:
                    lines.append("  - **Quantization Detected**")
                    if self.param_counts.quantized_ops:
                        lines.append(
                            f"    - Quantized Ops: {', '.join(self.param_counts.quantized_ops[:5])}"
                        )
                        if len(self.param_counts.quantized_ops) > 5:
                            lines.append(
                                f"    - ... and {len(self.param_counts.quantized_ops) - 5} more"
                            )
                lines.append("")

            if self.flop_counts:
                lines.append(
                    f"- **Estimated FLOPs**: {self._format_number(self.flop_counts.total)}"
                )
                lines.append("")

            if self.memory_estimates:
                lines.append(
                    f"- **Model Size**: {self._format_bytes(self.memory_estimates.model_size_bytes)}"
                )
                lines.append(
                    f"- **Peak Activation Memory** (batch=1): "
                    f"{self._format_bytes(self.memory_estimates.peak_activation_bytes)}"
                )
                # KV cache info for transformers
                if self.memory_estimates.kv_cache_bytes_per_token > 0:
                    lines.append("")
                    lines.append("### KV Cache (Transformer Inference)")
                    lines.append("")
                    config = self.memory_estimates.kv_cache_config
                    lines.append(
                        f"- **Per Token**: {self._format_bytes(self.memory_estimates.kv_cache_bytes_per_token)}"
                    )
                    if config.get("seq_len"):
                        lines.append(
                            f"- **Full Context** (seq={config['seq_len']}): "
                            f"{self._format_bytes(self.memory_estimates.kv_cache_bytes_full_context)}"
                        )
                    if config.get("num_layers"):
                        lines.append(f"- **Layers**: {config['num_layers']}")
                    if config.get("hidden_dim"):
                        lines.append(f"- **Hidden Dim**: {config['hidden_dim']}")

                # Memory breakdown by component
                if self.memory_estimates.breakdown:
                    bd = self.memory_estimates.breakdown
                    if bd.weights_by_op_type:
                        lines.append("")
                        lines.append("### Memory Breakdown by Op Type")
                        lines.append("")
                        lines.append("| Component | Size |")
                        lines.append("|-----------|------|")
                        sorted_weights = sorted(bd.weights_by_op_type.items(), key=lambda x: -x[1])
                        for op_type, size in sorted_weights[:8]:
                            lines.append(f"| {op_type} | {self._format_bytes(size)} |")
                lines.append("")

        # Architecture
        if self.architecture_type != "unknown" or self.detected_blocks:
            lines.append("## Architecture")
            lines.append("")
            lines.append(f"**Detected Type**: {self.architecture_type}")
            lines.append("")

            if self.detected_blocks:
                lines.append("### Detected Blocks")
                lines.append("")
                # Group by block type
                block_types: dict[str, int] = {}
                for block in self.detected_blocks:
                    block_types[block.block_type] = block_types.get(block.block_type, 0) + 1
                for bt, count in sorted(block_types.items(), key=lambda x: -x[1]):
                    lines.append(f"- {bt}: {count}")
                lines.append("")

                # Highlight non-standard residual patterns if present
                nonstandard_residuals = [
                    b
                    for b in self.detected_blocks
                    if b.block_type in ("ResidualConcat", "ResidualGate", "ResidualSub")
                ]
                if nonstandard_residuals:
                    # Group by type for summary
                    by_type: dict[str, int] = {}
                    for block in nonstandard_residuals:
                        by_type[block.block_type] = by_type.get(block.block_type, 0) + 1

                    type_labels = {
                        "ResidualConcat": "Concat-based (DenseNet-style)",
                        "ResidualGate": "Gated skip (Highway/attention)",
                        "ResidualSub": "Subtraction-based",
                    }

                    lines.append("### Non-Standard Skip Connections")
                    lines.append("")
                    lines.append(
                        f"This model uses {len(nonstandard_residuals)} non-standard skip connection(s):"
                    )
                    lines.append("")
                    for block_type, count in by_type.items():
                        label = type_labels.get(block_type, block_type)
                        lines.append(f"- **{label}**: {count}")
                    lines.append("")

        # Dataset info (if available)
        if self.dataset_info:
            lines.append("## Dataset Info")
            lines.append("")
            if self.dataset_info.task:
                lines.append(f"**Task**: {self.dataset_info.task}")
            if self.dataset_info.num_classes:
                lines.append(f"**Number of Classes**: {self.dataset_info.num_classes}")
            if self.dataset_info.class_names:
                lines.append("")
                lines.append("### Class Names")
                lines.append("")
                for idx, name in enumerate(self.dataset_info.class_names):
                    lines.append(f"- `{idx}`: {name}")
            if self.dataset_info.source:
                lines.append("")
                lines.append(f"*Metadata source: {self.dataset_info.source}*")
            lines.append("")

        # Hardware estimates
        if self.hardware_estimates and self.hardware_profile:
            hw = self.hardware_estimates
            lines.append("## Hardware Estimates")
            lines.append("")
            lines.append(f"**Target Device**: {self.hardware_profile.name}")
            lines.append(f"**Precision**: {hw.precision} | **Batch Size**: {hw.batch_size}")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| VRAM Required | {self._format_bytes(hw.vram_required_bytes)} |")
            lines.append(f"| Fits in VRAM | {'Yes' if hw.fits_in_vram else 'No'} |")
            if hw.fits_in_vram:
                lines.append(f"| Theoretical Latency | {hw.theoretical_latency_ms:.2f} ms |")
                lines.append(f"| Bottleneck | {hw.bottleneck} |")
                lines.append(f"| Compute Utilization | {hw.compute_utilization_estimate:.0%} |")
                # GPU Saturation: what % of GPU's 1-second capacity this inference uses
                lines.append(
                    f"| GPU Saturation | {hw.gpu_saturation:.2e} ({hw.gpu_saturation * 100:.4f}%) |"
                )
            lines.append("")

            # Add device specs
            lines.append("### Device Specifications")
            lines.append("")
            lines.append(f"- **VRAM**: {self._format_bytes(self.hardware_profile.vram_bytes)}")
            lines.append(f"- **FP32 Peak**: {self.hardware_profile.peak_fp32_tflops:.1f} TFLOPS")
            lines.append(f"- **FP16 Peak**: {self.hardware_profile.peak_fp16_tflops:.1f} TFLOPS")
            if self.hardware_profile.tdp_watts:
                lines.append(f"- **TDP**: {self.hardware_profile.tdp_watts}W")
            lines.append("")

        # System Requirements (Story 6C.2)
        if self.system_requirements:
            reqs = self.system_requirements
            lines.append("## System Requirements")
            lines.append("")
            lines.append("| Level | Device | VRAM |")
            lines.append("|-------|--------|------|")
            min_name = reqs.minimum_gpu.device if reqs.minimum_gpu else "N/A"
            rec_name = reqs.recommended_gpu.device if reqs.recommended_gpu else "N/A"
            opt_name = reqs.optimal_gpu.device if reqs.optimal_gpu else "N/A"
            min_vram = f"{reqs.minimum_vram_gb} GB" if reqs.minimum_vram_gb is not None else "-"
            rec_vram = (
                f"{reqs.recommended_vram_gb} GB" if reqs.recommended_vram_gb is not None else "-"
            )
            lines.append(f"| Minimum | {min_name} | {min_vram} |")
            lines.append(f"| Recommended | {rec_name} | {rec_vram} |")
            lines.append(f"| Optimal | {opt_name} | - |")
            lines.append("")

        # Batch Size Sweep (Story 6C.1)
        if self.batch_size_sweep:
            batch_sweep = self.batch_size_sweep
            lines.append("## Batch Size Scaling")
            lines.append("")
            lines.append(f"**Optimal Batch Size**: {batch_sweep.optimal_batch_size}")
            lines.append("")
            lines.append("| Batch Size | Latency (ms) | Throughput (inf/s) | VRAM (GB) |")
            lines.append("|------------|--------------|--------------------|-----------|")
            for i, bs in enumerate(batch_sweep.batch_sizes):
                lines.append(
                    f"| {bs} | {batch_sweep.latencies[i]:.2f} | {batch_sweep.throughputs[i]:.1f} | {batch_sweep.vram_usage_gb[i]:.2f} |"
                )
            lines.append("")

        # Resolution Sweep (Story 6.8)
        if self.resolution_sweep:
            res_sweep = self.resolution_sweep
            lines.append("## Resolution Scaling")
            lines.append("")
            lines.append(f"**Max Resolution**: {res_sweep.max_resolution}")
            lines.append(f"**Optimal Resolution**: {res_sweep.optimal_resolution}")
            lines.append("")
            lines.append(
                "| Resolution | FLOPs | Memory (GB) | Latency (ms) | Throughput | VRAM (GB) |"
            )
            lines.append(
                "|------------|-------|-------------|--------------|------------|-----------|"
            )
            for i, res in enumerate(res_sweep.resolutions):
                flops_str = self._format_number(res_sweep.flops[i])
                lat_str = (
                    f"{res_sweep.latencies[i]:.2f}"
                    if res_sweep.latencies[i] != float("inf")
                    else "OOM"
                )
                tput_str = (
                    f"{res_sweep.throughputs[i]:.1f}" if res_sweep.throughputs[i] > 0 else "-"
                )
                lines.append(
                    f"| {res} | {flops_str} | {res_sweep.memory_gb[i]:.2f} | "
                    f"{lat_str} | {tput_str} | {res_sweep.vram_usage_gb[i]:.2f} |"
                )
            lines.append("")

        # Risks
        if self.risk_signals:
            lines.append("## Risk Signals")
            lines.append("")
            for risk in self.risk_signals:
                severity_icon = {"info": "INFO", "warning": "WARN", "high": "HIGH"}
                icon = severity_icon.get(risk.severity, "")
                lines.append(f"### [{icon}] {risk.id}")
                lines.append("")
                lines.append(risk.description)
                lines.append("")
                if risk.recommendation:
                    lines.append(f"**Recommendation**: {risk.recommendation}")
                    lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _format_number(n: int | float) -> str:
        """Format large numbers with K/M/B suffixes."""
        if n >= 1e9:
            return f"{n / 1e9:.2f}B"
        if n >= 1e6:
            return f"{n / 1e6:.2f}M"
        if n >= 1e3:
            return f"{n / 1e3:.2f}K"
        return str(int(n))

    @staticmethod
    def _format_bytes(b: int) -> str:
        """Format bytes with KB/MB/GB suffixes."""
        if b >= 1e9:
            return f"{b / 1e9:.2f} GB"
        if b >= 1e6:
            return f"{b / 1e6:.2f} MB"
        if b >= 1e3:
            return f"{b / 1e3:.2f} KB"
        return f"{b} bytes"

    def to_html(
        self,
        image_paths: dict[str, pathlib.Path] | None = None,
        graph_html: str | None = None,
        layer_table_html: str | None = None,
        eval_metrics_html: str | None = None,
    ) -> str:
        """
        Generate a self-contained HTML report with embedded images.

        Args:
            image_paths: Dictionary mapping image names to file paths.
                         Images will be embedded as base64.
            graph_html: Optional HTML for interactive graph visualization (Task 5.7.8).
                        This should be the inner graph container, not full HTML document.
            layer_table_html: Optional HTML for per-layer summary table (Story 5.8).
            eval_metrics_html: Optional HTML for evaluation metrics section (Task 12.5.4).
                              Generated by haoline.eval.comparison.generate_eval_metrics_html().

        Returns:
            Complete HTML document as a string.
        """
        import base64

        # Embed images as base64
        def embed_image(path: pathlib.Path) -> str:
            if path.exists():
                with open(path, "rb") as f:
                    data = base64.b64encode(f.read()).decode("utf-8")
                return f"data:image/png;base64,{data}"
            return ""

        images = {}
        if image_paths:
            for name, path in image_paths.items():
                images[name] = embed_image(path)

        # Build HTML
        model_name = pathlib.Path(self.metadata.path).stem if self.metadata else "Model"

        html_parts = [self._html_head(model_name)]
        html_parts.append('<body><div class="container">')

        # Header
        html_parts.append(
            f"""
        <header>
            <h1>{model_name}</h1>
            <p class="subtitle">ONNX Model Analysis Report</p>
            <p class="timestamp">Generated by HaoLine v0.1.0 on {datetime.utcnow().isoformat()}Z</p>
        </header>
        """
        )

        # Executive Summary (LLM) - Prominent AI-generated summary
        if self.llm_summary and self.llm_summary.get("success"):
            html_parts.append(
                """
            <section class="executive-summary">
                <h2>AI Executive Summary</h2>
            """
            )
            if self.llm_summary.get("short_summary"):
                html_parts.append(f'<p class="tldr">{self.llm_summary["short_summary"]}</p>')
            if self.llm_summary.get("detailed_summary"):
                html_parts.append(f"<p>{self.llm_summary['detailed_summary']}</p>")
            html_parts.append(
                f'<p class="llm-credit">Generated by {self.llm_summary.get("model", "LLM")}</p>'
            )
            html_parts.append("</section>")

        # Key Metrics Cards
        html_parts.append('<section class="metrics-cards">')
        if self.param_counts:
            html_parts.append(
                f"""
            <div class="card">
                <div class="card-value">{self._format_number(self.param_counts.total)}</div>
                <div class="card-label">Parameters</div>
            </div>
            """
            )
        if self.flop_counts:
            html_parts.append(
                f"""
            <div class="card">
                <div class="card-value">{self._format_number(self.flop_counts.total)}</div>
                <div class="card-label">FLOPs</div>
            </div>
            """
            )
        if self.memory_estimates:
            html_parts.append(
                f"""
            <div class="card">
                <div class="card-value">{self._format_bytes(self.memory_estimates.model_size_bytes)}</div>
                <div class="card-label">Model Size</div>
            </div>
            """
            )
        if self.architecture_type:
            html_parts.append(
                f"""
            <div class="card">
                <div class="card-value">{self.architecture_type.upper()}</div>
                <div class="card-label">Architecture</div>
            </div>
            """
            )
        # Quantization indicator card
        if self.param_counts and self.param_counts.is_quantized:
            html_parts.append(
                """
            <div class="card" style="border-color: #4CAF50;">
                <div class="card-value" style="color: #4CAF50;">Yes</div>
                <div class="card-label">Quantized</div>
            </div>
            """
            )
        html_parts.append("</section>")

        # Interactive Graph Visualization - Placed prominently after metrics (Task 5.7.8)
        if graph_html:
            html_parts.append('<section class="graph-section">')
            html_parts.append("<h2>Neural Network Architecture</h2>")
            html_parts.append(
                '<p class="section-desc">Click nodes to expand/collapse blocks. '
                "Use the search box to find specific operations.</p>"
            )
            html_parts.append('<div class="graph-container">')
            html_parts.append(graph_html)
            html_parts.append("</div></section>")

        # Evaluation Metrics (Task 12.5.4)
        if eval_metrics_html:
            html_parts.append(eval_metrics_html)

        # Complexity Metrics Details (KV Cache + Memory Breakdown)
        if self.memory_estimates:
            # KV Cache section (Task 4.4.2)
            if self.memory_estimates.kv_cache_bytes_per_token > 0:
                html_parts.append('<section class="kv-cache">')
                html_parts.append("<h2>KV Cache (Transformer Inference)</h2>")
                config = self.memory_estimates.kv_cache_config
                html_parts.append("<table>")
                html_parts.append("<tr><th>Metric</th><th>Value</th></tr>")
                html_parts.append(
                    f"<tr><td>Per Token</td><td>{self._format_bytes(self.memory_estimates.kv_cache_bytes_per_token)}</td></tr>"
                )
                if config.get("seq_len"):
                    html_parts.append(
                        f"<tr><td>Full Context (seq={config['seq_len']})</td>"
                        f"<td>{self._format_bytes(self.memory_estimates.kv_cache_bytes_full_context)}</td></tr>"
                    )
                if config.get("num_layers"):
                    html_parts.append(f"<tr><td>Layers</td><td>{config['num_layers']}</td></tr>")
                if config.get("hidden_dim"):
                    html_parts.append(
                        f"<tr><td>Hidden Dim</td><td>{config['hidden_dim']}</td></tr>"
                    )
                html_parts.append("</table></section>")

            # Memory Breakdown by Op Type (Task 4.4.3)
            if self.memory_estimates.breakdown:
                bd = self.memory_estimates.breakdown
                if bd.weights_by_op_type:
                    html_parts.append('<section class="memory-breakdown">')
                    html_parts.append("<h2>Memory Breakdown by Op Type</h2>")
                    html_parts.append("<table>")
                    html_parts.append("<tr><th>Component</th><th>Size</th></tr>")
                    sorted_weights = sorted(bd.weights_by_op_type.items(), key=lambda x: -x[1])
                    for op_type, size in sorted_weights[:8]:
                        html_parts.append(
                            f"<tr><td>{op_type}</td><td>{self._format_bytes(size)}</td></tr>"
                        )
                    if len(sorted_weights) > 8:
                        remaining = sum(s for _, s in sorted_weights[8:])
                        html_parts.append(
                            f"<tr><td>Other ({len(sorted_weights) - 8} types)</td>"
                            f"<td>{self._format_bytes(remaining)}</td></tr>"
                        )
                    html_parts.append("</table></section>")

        # Parameter Details section (shared weights, precision, quantization)
        if self.param_counts:
            has_content = (
                self.param_counts.num_shared_weights > 0
                or self.param_counts.precision_breakdown
                or self.param_counts.is_quantized
            )
            if has_content:
                html_parts.append('<section class="param-details">')
                html_parts.append("<h2>Parameter Details</h2>")

                # Precision breakdown
                if self.param_counts.precision_breakdown:
                    html_parts.append("<h3>Precision Breakdown</h3>")
                    html_parts.append("<table>")
                    html_parts.append("<tr><th>Data Type</th><th>Parameters</th></tr>")
                    for dtype, count in sorted(
                        self.param_counts.precision_breakdown.items(),
                        key=lambda x: -x[1],
                    ):
                        html_parts.append(
                            f"<tr><td>{dtype}</td><td>{self._format_number(count)}</td></tr>"
                        )
                    html_parts.append("</table>")

                # Shared weights info
                if self.param_counts.num_shared_weights > 0:
                    html_parts.append("<h3>Shared Weights</h3>")
                    html_parts.append(
                        f"<p><strong>{self.param_counts.num_shared_weights}</strong> weights are shared across multiple nodes.</p>"
                    )
                    if self.param_counts.shared_weights:
                        html_parts.append("<details><summary>Show shared weight details</summary>")
                        html_parts.append("<table>")
                        html_parts.append("<tr><th>Weight Name</th><th>Used By Nodes</th></tr>")
                        for name, nodes in list(self.param_counts.shared_weights.items())[:10]:
                            nodes_str = ", ".join(nodes[:5])
                            if len(nodes) > 5:
                                nodes_str += f" (+{len(nodes) - 5} more)"
                            html_parts.append(f"<tr><td>{name}</td><td>{nodes_str}</td></tr>")
                        if len(self.param_counts.shared_weights) > 10:
                            html_parts.append(
                                f"<tr><td colspan='2'>... and {len(self.param_counts.shared_weights) - 10} more shared weights</td></tr>"
                            )
                        html_parts.append("</table></details>")

                # Quantization info
                if self.param_counts.is_quantized:
                    html_parts.append("<h3>Quantization</h3>")
                    html_parts.append(
                        '<p style="color: #4CAF50; font-weight: bold;">Model is quantized</p>'
                    )
                    if self.param_counts.quantized_ops:
                        html_parts.append(
                            f"<p>Quantized operations: {', '.join(self.param_counts.quantized_ops)}</p>"
                        )

                html_parts.append("</section>")

        # Visualizations
        if images:
            html_parts.append('<section class="visualizations">')
            html_parts.append("<h2>Visualizations</h2>")
            html_parts.append('<div class="chart-grid">')
            for name, data_uri in images.items():
                if data_uri:
                    label = name.replace("_", " ").title()
                    html_parts.append(
                        f"""
                    <div class="chart-container">
                        <img src="{data_uri}" alt="{label}">
                    </div>
                    """
                    )
            html_parts.append("</div></section>")

        # Per-Layer Summary Table (Story 5.8)
        if layer_table_html:
            html_parts.append('<section class="layer-summary">')
            html_parts.append("<h2>Layer-by-Layer Analysis</h2>")
            html_parts.append(
                '<p class="section-desc">Click column headers to sort. '
                "Use the search box to filter layers.</p>"
            )
            html_parts.append(layer_table_html)
            html_parts.append("</section>")

        # Model Details
        html_parts.append('<section class="details">')
        html_parts.append("<h2>Model Details</h2>")

        # Metadata table
        if self.metadata:
            html_parts.append(
                """
            <h3>Metadata</h3>
            <table>
                <tr><th>Property</th><th>Value</th></tr>
            """
            )
            html_parts.append(f"<tr><td>IR Version</td><td>{self.metadata.ir_version}</td></tr>")
            html_parts.append(
                f"<tr><td>Producer</td><td>{self.metadata.producer_name} {self.metadata.producer_version}</td></tr>"
            )
            opsets = ", ".join(f"{d}:{v}" for d, v in self.metadata.opsets.items())
            html_parts.append(f"<tr><td>Opsets</td><td>{opsets}</td></tr>")
            html_parts.append("</table>")

        # Graph summary
        if self.graph_summary:
            html_parts.append(
                """
            <h3>Graph Summary</h3>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
            """
            )
            html_parts.append(f"<tr><td>Nodes</td><td>{self.graph_summary.num_nodes}</td></tr>")
            html_parts.append(f"<tr><td>Inputs</td><td>{self.graph_summary.num_inputs}</td></tr>")
            html_parts.append(f"<tr><td>Outputs</td><td>{self.graph_summary.num_outputs}</td></tr>")
            html_parts.append(
                f"<tr><td>Initializers</td><td>{self.graph_summary.num_initializers}</td></tr>"
            )
            html_parts.append("</table>")

            # I/O shapes
            if self.graph_summary.input_shapes:
                html_parts.append("<h4>Inputs</h4><ul>")
                for name, shape in self.graph_summary.input_shapes.items():
                    html_parts.append(f"<li><code>{name}</code>: {shape}</li>")
                html_parts.append("</ul>")

            if self.graph_summary.output_shapes:
                html_parts.append("<h4>Outputs</h4><ul>")
                for name, shape in self.graph_summary.output_shapes.items():
                    html_parts.append(f"<li><code>{name}</code>: {shape}</li>")
                html_parts.append("</ul>")

            # Operator Distribution (Task 4.4.1)
            if self.graph_summary.op_type_counts:
                html_parts.append("<h3>Operator Distribution</h3>")
                html_parts.append("<table>")
                html_parts.append("<tr><th>Operator</th><th>Count</th></tr>")
                sorted_ops = sorted(self.graph_summary.op_type_counts.items(), key=lambda x: -x[1])
                for op, count in sorted_ops[:15]:
                    html_parts.append(f"<tr><td>{op}</td><td>{count}</td></tr>")
                if len(sorted_ops) > 15:
                    html_parts.append(
                        f"<tr><td>...</td><td>({len(sorted_ops) - 15} more)</td></tr>"
                    )
                html_parts.append("</table>")

        html_parts.append("</section>")

        # Architecture section
        if self.architecture_type != "unknown" or self.detected_blocks:
            html_parts.append('<section class="architecture">')
            html_parts.append("<h2>Architecture</h2>")
            html_parts.append(
                f'<p><strong>Detected Type:</strong> <span class="arch-type">{self.architecture_type.upper()}</span></p>'
            )

            if self.detected_blocks:
                # Group by block type
                block_types: dict[str, int] = {}
                for block in self.detected_blocks:
                    block_types[block.block_type] = block_types.get(block.block_type, 0) + 1

                html_parts.append("<h3>Detected Blocks</h3>")
                html_parts.append("<table>")
                html_parts.append("<tr><th>Block Type</th><th>Count</th></tr>")
                for bt, count in sorted(block_types.items(), key=lambda x: -x[1]):
                    html_parts.append(f"<tr><td>{bt}</td><td>{count}</td></tr>")
                html_parts.append("</table>")

                # Non-standard residual patterns
                nonstandard_residuals = [
                    b
                    for b in self.detected_blocks
                    if b.block_type in ("ResidualConcat", "ResidualGate", "ResidualSub")
                ]
                if nonstandard_residuals:
                    # Group by type
                    by_type: dict[str, list] = {}
                    for block in nonstandard_residuals:
                        if block.block_type not in by_type:
                            by_type[block.block_type] = []
                        by_type[block.block_type].append(block)

                    type_labels = {
                        "ResidualConcat": "Concat-based (DenseNet-style)",
                        "ResidualGate": "Gated skip (Highway/attention)",
                        "ResidualSub": "Subtraction-based",
                    }

                    html_parts.append('<div class="nonstandard-residuals">')
                    html_parts.append("<h3>Non-Standard Skip Connections</h3>")
                    html_parts.append(
                        f"<p>This model uses {len(nonstandard_residuals)} non-standard skip connection(s):</p>"
                    )

                    # Create collapsible section for each type
                    for block_type, blocks in by_type.items():
                        label = type_labels.get(block_type, block_type)
                        html_parts.append(f"<details><summary>{label} ({len(blocks)})</summary>")
                        html_parts.append('<div class="skip-connections-grid">')
                        for block in blocks:
                            if block_type == "ResidualConcat":
                                depth_diff = block.attributes.get("depth_diff", "?")
                                html_parts.append(
                                    f'<div class="skip-item">{block.name} '
                                    f'<span class="skip-detail">depth: {depth_diff}</span></div>'
                                )
                            else:
                                html_parts.append(f'<div class="skip-item">{block.name}</div>')
                        html_parts.append("</div></details>")
                    html_parts.append("</div>")

            html_parts.append("</section>")

        # Dataset Info
        if self.dataset_info:
            html_parts.append('<section class="dataset-info">')
            html_parts.append("<h2>Dataset Info</h2>")
            if self.dataset_info.task:
                html_parts.append(f"<p><strong>Task:</strong> {self.dataset_info.task}</p>")
            if self.dataset_info.num_classes:
                html_parts.append(
                    f"<p><strong>Number of Classes:</strong> {self.dataset_info.num_classes}</p>"
                )
            if self.dataset_info.class_names:
                num_classes = len(self.dataset_info.class_names)
                html_parts.append(
                    f'<details class="class-names-details">'
                    f"<summary>Class Names ({num_classes} classes) - click to expand</summary>"
                )
                html_parts.append('<div class="class-grid">')
                for idx, name in enumerate(self.dataset_info.class_names):
                    html_parts.append(f'<div class="class-item"><code>{idx}</code> {name}</div>')
                html_parts.append("</div></details>")
            if self.dataset_info.source:
                html_parts.append(
                    f'<p class="metadata-source"><em>Source: {self.dataset_info.source}</em></p>'
                )
            html_parts.append("</section>")

        # Hardware Estimates
        if self.hardware_estimates and self.hardware_profile:
            hw = self.hardware_estimates
            html_parts.append('<section class="hardware">')
            html_parts.append("<h2>Hardware Estimates</h2>")
            html_parts.append(f'<p class="device-name">{hw.device}</p>')
            html_parts.append(
                f'<p class="precision-info">Precision: {hw.precision} | Batch Size: {hw.batch_size}</p>'
            )

            html_parts.append(
                """
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
            """
            )
            html_parts.append(
                f"<tr><td>VRAM Required</td><td>{self._format_bytes(hw.vram_required_bytes)}</td></tr>"
            )
            html_parts.append(
                f"<tr><td>Fits in VRAM</td><td>{'Yes' if hw.fits_in_vram else 'No'}</td></tr>"
            )
            if hw.fits_in_vram:
                html_parts.append(
                    f"<tr><td>Theoretical Latency</td><td>{hw.theoretical_latency_ms:.4f} ms</td></tr>"
                )
                html_parts.append(f"<tr><td>Bottleneck</td><td>{hw.bottleneck}</td></tr>")
                html_parts.append(
                    f"<tr><td>Compute Utilization</td><td>{hw.compute_utilization_estimate:.0%}</td></tr>"
                )
                html_parts.append(
                    f"<tr><td>GPU Saturation</td><td>{hw.gpu_saturation:.2e} ({hw.gpu_saturation * 100:.4f}%)</td></tr>"
                )
            html_parts.append("</table>")

            # Device specs
            html_parts.append("<h3>Device Specifications</h3>")
            html_parts.append("<ul>")
            html_parts.append(
                f"<li><strong>VRAM:</strong> {self._format_bytes(self.hardware_profile.vram_bytes)}</li>"
            )
            html_parts.append(
                f"<li><strong>FP32 Peak:</strong> {self.hardware_profile.peak_fp32_tflops:.1f} TFLOPS</li>"
            )
            html_parts.append(
                f"<li><strong>FP16 Peak:</strong> {self.hardware_profile.peak_fp16_tflops:.1f} TFLOPS</li>"
            )
            if self.hardware_profile.tdp_watts:
                html_parts.append(
                    f"<li><strong>TDP:</strong> {self.hardware_profile.tdp_watts}W</li>"
                )
            html_parts.append("</ul></section>")

        # System Requirements
        if self.system_requirements:
            reqs = self.system_requirements
            html_parts.append('<section class="system-requirements">')
            html_parts.append("<h2>System Requirements</h2>")
            html_parts.append("<table>")
            html_parts.append("<tr><th>Level</th><th>Device</th><th>VRAM</th></tr>")
            min_name = reqs.minimum_gpu.device if reqs.minimum_gpu else "N/A"
            rec_name = reqs.recommended_gpu.device if reqs.recommended_gpu else "N/A"
            opt_name = reqs.optimal_gpu.device if reqs.optimal_gpu else "N/A"
            min_vram = f"{reqs.minimum_vram_gb} GB" if reqs.minimum_vram_gb is not None else "-"
            rec_vram = (
                f"{reqs.recommended_vram_gb} GB" if reqs.recommended_vram_gb is not None else "-"
            )
            html_parts.append(f"<tr><td>Minimum</td><td>{min_name}</td><td>{min_vram}</td></tr>")
            html_parts.append(
                f"<tr><td>Recommended</td><td>{rec_name}</td><td>{rec_vram}</td></tr>"
            )
            html_parts.append(f"<tr><td>Optimal</td><td>{opt_name}</td><td>-</td></tr>")
            html_parts.append("</table></section>")

        # Batch Size Sweep
        if self.batch_size_sweep:
            batch_sweep = self.batch_size_sweep
            html_parts.append('<section class="batch-scaling">')
            html_parts.append("<h2>Batch Size Scaling</h2>")
            html_parts.append(
                f"<p><strong>Optimal Batch Size:</strong> {batch_sweep.optimal_batch_size}</p>"
            )
            html_parts.append("<table>")
            html_parts.append(
                "<tr><th>Batch Size</th><th>Latency (ms)</th><th>Throughput (inf/s)</th><th>VRAM (GB)</th></tr>"
            )
            for i, bs in enumerate(batch_sweep.batch_sizes):
                html_parts.append(
                    f"<tr><td>{bs}</td><td>{batch_sweep.latencies[i]:.2f}</td><td>{batch_sweep.throughputs[i]:.1f}</td><td>{batch_sweep.vram_usage_gb[i]:.2f}</td></tr>"
                )
            html_parts.append("</table></section>")

        # Resolution Sweep (Story 6.8)
        if self.resolution_sweep:
            res_sweep = self.resolution_sweep
            html_parts.append('<section class="resolution-scaling">')
            html_parts.append("<h2>Resolution Scaling</h2>")
            html_parts.append(
                f"<p><strong>Max Resolution:</strong> {res_sweep.max_resolution} | "
                f"<strong>Optimal:</strong> {res_sweep.optimal_resolution}</p>"
            )
            html_parts.append("<table>")
            html_parts.append(
                "<tr><th>Resolution</th><th>FLOPs</th><th>Memory (GB)</th>"
                "<th>Latency (ms)</th><th>Throughput</th><th>VRAM (GB)</th></tr>"
            )
            for i, res in enumerate(res_sweep.resolutions):
                flops_str = self._format_number(res_sweep.flops[i])
                lat_str = (
                    f"{res_sweep.latencies[i]:.2f}"
                    if res_sweep.latencies[i] != float("inf")
                    else "OOM"
                )
                tput_str = (
                    f"{res_sweep.throughputs[i]:.1f}" if res_sweep.throughputs[i] > 0 else "-"
                )
                html_parts.append(
                    f"<tr><td>{res}</td><td>{flops_str}</td><td>{res_sweep.memory_gb[i]:.2f}</td>"
                    f"<td>{lat_str}</td><td>{tput_str}</td><td>{res_sweep.vram_usage_gb[i]:.2f}</td></tr>"
                )
            html_parts.append("</table></section>")

        # Risk Signals
        if self.risk_signals:
            html_parts.append('<section class="risks">')
            html_parts.append("<h2>Risk Signals</h2>")
            for risk in self.risk_signals:
                severity_class = {
                    "info": "info",
                    "warning": "warning",
                    "high": "high",
                }.get(risk.severity, "info")
                html_parts.append(
                    f"""
                <div class="risk-card {severity_class}">
                    <div class="risk-header">
                        <span class="severity">{risk.severity.upper()}</span>
                        <span class="risk-id">{risk.id}</span>
                    </div>
                    <p>{risk.description}</p>
                """
                )
                if risk.recommendation:
                    html_parts.append(
                        f'<p class="recommendation"><strong>Recommendation:</strong> {risk.recommendation}</p>'
                    )
                html_parts.append("</div>")
            html_parts.append("</section>")

        # Quantization Analysis (Task 33.5.4)
        if self.quantization_lint:
            ql = self.quantization_lint
            score = ql.readiness_score
            # Grade color
            if score >= 90:
                grade, color = "A", "#3fb950"
            elif score >= 75:
                grade, color = "B", "#84cc16"
            elif score >= 60:
                grade, color = "C", "#d29922"
            elif score >= 40:
                grade, color = "D", "#ff6b6b"
            else:
                grade, color = "F", "#ef4444"

            html_parts.append('<section class="quant-analysis">')
            html_parts.append("<h2>INT8 Quantization Readiness</h2>")

            # Score card
            html_parts.append(
                f"""
                <div style="display: flex; align-items: center; gap: 2rem; margin-bottom: 1.5rem;">
                    <div style="text-align: center; padding: 1.5rem; background: {color}22;
                         border: 2px solid {color}; border-radius: 12px; min-width: 120px;">
                        <div style="font-size: 2.5rem; font-weight: bold; color: {color};">{grade}</div>
                        <div style="font-size: 1.2rem; color: var(--text-secondary);">{score}/100</div>
                    </div>
                    <div>
                        <div style="margin-bottom: 0.5rem;">
                            <strong>Quant-Friendly Ops:</strong> {ql.quant_friendly_pct:.1f}%
                        </div>
                        <div style="margin-bottom: 0.5rem;">
                            <strong>Total Ops:</strong> {ql.total_ops}
                        </div>
                        <div>
                            <strong>Issues Found:</strong> {len(ql.warnings)}
                            {f" ({ql.critical_count} critical)" if hasattr(ql, "critical_count") and ql.critical_count else ""}
                        </div>
                    </div>
                </div>
                """
            )

            # Warnings
            if ql.warnings:
                html_parts.append("<h3>Issues</h3>")
                severity_colors = {
                    "critical": "#ef4444",
                    "high": "#ff6b6b",
                    "medium": "#d29922",
                    "low": "#84cc16",
                    "info": "#00d4ff",
                }
                for w in sorted(ql.warnings, key=lambda x: x.severity.value)[:10]:
                    sev_color = severity_colors.get(w.severity.value, "#8b949e")
                    html_parts.append(
                        f"""
                        <div style="padding: 0.75rem; margin: 0.5rem 0; border-left: 3px solid {sev_color};
                             background: {sev_color}11;">
                            <span style="color: {sev_color}; font-weight: bold;">[{w.severity.value.upper()}]</span>
                            <span>{w.message}</span>
                            {f'<br><span style="color: var(--text-secondary); margin-left: 2rem;">→ {w.recommendation}</span>' if w.recommendation else ""}
                        </div>
                        """
                    )
                if len(ql.warnings) > 10:
                    html_parts.append(
                        f'<p style="color: var(--text-secondary);">... and {len(ql.warnings) - 10} more issues</p>'
                    )

            # Problem layers
            if ql.problem_layers:
                html_parts.append("<h3>Problem Layers</h3>")
                html_parts.append("<table>")
                html_parts.append(
                    "<tr><th>Layer</th><th>Op Type</th><th>Issue</th><th>Recommendation</th></tr>"
                )
                for layer in ql.problem_layers[:8]:
                    html_parts.append(
                        f"<tr><td>{layer.get('name', 'N/A')}</td>"
                        f"<td>{layer.get('op_type', 'N/A')}</td>"
                        f"<td>{layer.get('reason', 'N/A')}</td>"
                        f"<td>{layer.get('recommendation', 'N/A')}</td></tr>"
                    )
                html_parts.append("</table>")

            html_parts.append("</section>")

        html_parts.append("</div></body></html>")
        return "".join(html_parts)

    def _html_head(self, title: str) -> str:
        """Generate HTML head with embedded CSS."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - HaoLine Report</title>
    <style>
        :root {{
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-card: #21262d;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --accent-cyan: #00d4ff;
            --accent-coral: #ff6b6b;
            --accent-green: #3fb950;
            --accent-yellow: #d29922;
            --border: #30363d;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }}

        header {{
            text-align: center;
            margin-bottom: 3rem;
            padding-bottom: 2rem;
            border-bottom: 1px solid var(--border);
        }}

        header h1 {{
            font-size: 2.5rem;
            color: var(--accent-cyan);
            margin-bottom: 0.5rem;
        }}

        header .subtitle {{
            font-size: 1.2rem;
            color: var(--text-secondary);
        }}

        header .timestamp {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-top: 0.5rem;
        }}

        section {{
            margin-bottom: 3rem;
        }}

        h2 {{
            font-size: 1.5rem;
            color: var(--accent-cyan);
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--accent-cyan);
        }}

        h3 {{
            font-size: 1.2rem;
            color: var(--text-primary);
            margin: 1.5rem 0 1rem;
        }}

        h4 {{
            font-size: 1rem;
            color: var(--text-secondary);
            margin: 1rem 0 0.5rem;
        }}

        /* Executive Summary */
        .executive-summary {{
            background: linear-gradient(135deg, #1a2a3a 0%, #0d1a26 100%);
            padding: 2rem;
            border-radius: 12px;
            border: 2px solid var(--accent-cyan);
            margin: 2rem 0;
            box-shadow: 0 4px 20px rgba(0, 200, 255, 0.15);
        }}

        .executive-summary .tldr {{
            font-size: 1.1rem;
            color: var(--accent-cyan);
            margin-bottom: 1rem;
        }}

        .executive-summary .llm-credit {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-top: 1rem;
            font-style: italic;
        }}

        /* Metrics Cards */
        .metrics-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}

        .card {{
            background: var(--bg-card);
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            border: 1px solid var(--border);
            transition: transform 0.2s, border-color 0.2s;
        }}

        .card:hover {{
            transform: translateY(-2px);
            border-color: var(--accent-cyan);
        }}

        .card-value {{
            font-size: 2rem;
            font-weight: bold;
            color: var(--accent-cyan);
        }}

        .card-label {{
            font-size: 0.9rem;
            color: var(--text-secondary);
            margin-top: 0.5rem;
        }}

        /* Visualizations */
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 1.5rem;
        }}

        .chart-container {{
            background: var(--bg-card);
            padding: 1rem;
            border-radius: 12px;
        }}

        /* Section descriptions */
        .section-desc {{
            font-size: 0.875rem;
            color: var(--text-secondary);
            margin-bottom: 1rem;
        }}

        /* Interactive Graph Section (Task 5.7.8) */
        .graph-section {{
            margin-bottom: 3rem;
        }}

        .graph-container {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            height: 600px;
            overflow: hidden;
            position: relative;
        }}

        .graph-container iframe {{
            width: 100%;
            height: 100%;
            border: none;
        }}

        /* Layer Summary Section (Story 5.8) */
        .layer-summary {{
            margin-bottom: 3rem;
            border: 1px solid var(--border);
        }}

        .chart-container img {{
            width: 100%;
            height: auto;
            border-radius: 8px;
        }}

        /* Tables */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            background: var(--bg-card);
            border-radius: 8px;
            overflow: hidden;
        }}

        th, td {{
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}

        th {{
            background: var(--bg-secondary);
            color: var(--accent-cyan);
            font-weight: 600;
        }}

        tr:last-child td {{
            border-bottom: none;
        }}

        tr:hover {{
            background: var(--bg-secondary);
        }}

        /* Lists */
        ul {{
            list-style: none;
            padding-left: 0;
        }}

        ul li {{
            padding: 0.5rem 0;
            padding-left: 1.5rem;
            position: relative;
        }}

        ul li::before {{
            content: "";
            position: absolute;
            left: 0;
            top: 50%;
            transform: translateY(-50%);
            width: 6px;
            height: 6px;
            background: var(--accent-cyan);
            border-radius: 50%;
        }}

        code {{
            background: var(--bg-secondary);
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-family: 'SF Mono', Monaco, 'Courier New', monospace;
            font-size: 0.9em;
            color: var(--accent-coral);
        }}

        /* Collapsible Details */
        details {{
            background: var(--bg-card);
            border-radius: 8px;
            margin: 1rem 0;
            border: 1px solid var(--border);
        }}

        details summary {{
            padding: 1rem;
            cursor: pointer;
            font-weight: 600;
            color: var(--accent-cyan);
            user-select: none;
            transition: background 0.2s;
        }}

        details summary:hover {{
            background: var(--bg-secondary);
        }}

        details[open] summary {{
            border-bottom: 1px solid var(--border);
        }}

        details summary::marker {{
            color: var(--accent-cyan);
        }}

        /* Class Names Grid */
        .class-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 0.5rem;
            padding: 1rem;
            max-height: 400px;
            overflow-y: auto;
        }}

        .class-item {{
            padding: 0.4rem 0.6rem;
            background: var(--bg-secondary);
            border-radius: 4px;
            font-size: 0.85rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        .class-item code {{
            margin-right: 0.3rem;
            font-size: 0.75rem;
        }}

        /* KV Cache Section */
        .kv-cache {{
            background: var(--bg-secondary);
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 4px solid var(--accent-green);
        }}

        .kv-cache h2 {{
            color: var(--accent-green);
            border-bottom-color: var(--accent-green);
        }}

        /* Memory Breakdown Section */
        .memory-breakdown {{
            background: var(--bg-secondary);
            padding: 1.5rem;
            border-radius: 12px;
        }}

        /* Architecture Section */
        .architecture .arch-type {{
            color: var(--accent-cyan);
            font-weight: bold;
        }}

        .nonstandard-residuals {{
            background: var(--bg-card);
            padding: 1.5rem;
            border-radius: 8px;
            margin-top: 1rem;
            border-left: 4px solid var(--accent-yellow);
        }}

        .nonstandard-residuals h3 {{
            color: var(--accent-yellow);
            margin-top: 0;
        }}

        .nonstandard-residuals details {{
            margin: 0.5rem 0;
            background: var(--bg-primary);
        }}

        .nonstandard-residuals summary {{
            padding: 0.75rem 1rem;
            color: var(--text-primary);
        }}

        .skip-connections-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 0.5rem;
            padding: 1rem;
            max-height: 300px;
            overflow-y: auto;
        }}

        .skip-item {{
            padding: 0.4rem 0.6rem;
            background: var(--bg-secondary);
            border-radius: 4px;
            font-size: 0.8rem;
            font-family: 'SF Mono', Monaco, monospace;
        }}

        .skip-detail {{
            color: var(--text-secondary);
            font-size: 0.7rem;
        }}

        /* Hardware Section */
        .hardware .device-name {{
            font-size: 1.2rem;
            color: var(--accent-cyan);
            margin-bottom: 0.5rem;
        }}

        .hardware .precision-info {{
            color: var(--text-secondary);
            margin-bottom: 1rem;
        }}

        /* Risk Signals */
        .risk-card {{
            background: var(--bg-card);
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            border-left: 4px solid var(--text-secondary);
        }}

        .risk-card.info {{
            border-left-color: var(--accent-cyan);
        }}

        .risk-card.warning {{
            border-left-color: var(--accent-yellow);
        }}

        .risk-card.high {{
            border-left-color: var(--accent-coral);
        }}

        .risk-header {{
            display: flex;
            gap: 1rem;
            align-items: center;
            margin-bottom: 0.75rem;
        }}

        .severity {{
            font-size: 0.75rem;
            font-weight: bold;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            background: var(--bg-secondary);
        }}

        .risk-card.info .severity {{ color: var(--accent-cyan); }}
        .risk-card.warning .severity {{ color: var(--accent-yellow); }}
        .risk-card.high .severity {{ color: var(--accent-coral); }}

        .risk-id {{
            font-weight: 600;
        }}

        .recommendation {{
            margin-top: 0.75rem;
            padding-top: 0.75rem;
            border-top: 1px solid var(--border);
            font-size: 0.9rem;
        }}

        /* Responsive */
        @media (max-width: 768px) {{
            .container {{
                padding: 1rem;
            }}

            header h1 {{
                font-size: 1.8rem;
            }}

            .chart-grid {{
                grid-template-columns: 1fr;
            }}

            .metrics-cards {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
    </style>
</head>
"""


class ModelInspector:
    """
    Main orchestrator for ONNX model analysis.

    Coordinates the loader, metrics engine, pattern analyzer, and risk analyzer
    to produce a comprehensive InspectionReport.

    Example:
        inspector = ModelInspector()
        report = inspector.inspect("model.onnx")
        report.to_json()
    """

    def __init__(
        self,
        loader: ONNXGraphLoader | None = None,
        metrics: MetricsEngine | None = None,
        patterns: PatternAnalyzer | None = None,
        risks: RiskAnalyzer | None = None,
        logger: logging.Logger | None = None,
    ):
        """
        Initialize ModelInspector with optional component overrides.

        Args:
            loader: Custom graph loader. If None, uses default ONNXGraphLoader.
            metrics: Custom metrics engine. If None, uses default MetricsEngine.
            patterns: Custom pattern analyzer. If None, uses default PatternAnalyzer.
            risks: Custom risk analyzer. If None, uses default RiskAnalyzer.
            logger: Logger for diagnostic output.
        """
        # Defer imports to avoid circular dependencies
        from .analyzer import MetricsEngine, ONNXGraphLoader
        from .patterns import PatternAnalyzer
        from .risks import RiskAnalyzer

        self.loader = loader or ONNXGraphLoader()
        self.metrics = metrics or MetricsEngine()
        self.patterns = patterns or PatternAnalyzer()
        self.risks = risks or RiskAnalyzer()
        self.logger = logger or logging.getLogger("haoline")

    def inspect(self, model_path: str | pathlib.Path) -> InspectionReport:
        """
        Run full analysis pipeline on an ONNX model.

        Args:
            model_path: Path to the ONNX model file.

        Returns:
            InspectionReport with all analysis results.
        """
        model_path = pathlib.Path(model_path)
        self.logger.info(f"Inspecting model: {model_path}")

        # Load model
        model, graph_info = self.loader.load(model_path)

        # Extract metadata
        metadata = self._extract_metadata(model, model_path)

        # Build graph summary
        graph_summary = self._build_graph_summary(graph_info)

        # Compute metrics
        self.logger.debug("Computing metrics...")
        param_counts = self.metrics.count_parameters(graph_info)
        flop_counts = self.metrics.estimate_flops(graph_info)
        memory_estimates = self.metrics.estimate_memory(graph_info)

        # Detect patterns
        self.logger.debug("Detecting patterns...")
        detected_blocks = self.patterns.group_into_blocks(graph_info)
        architecture_type = self.patterns.classify_architecture(graph_info, detected_blocks)

        # Analyze risks
        self.logger.debug("Analyzing risks...")
        risk_signals = self.risks.analyze(graph_info, detected_blocks)

        # Try to infer num_classes from output shapes
        dataset_info = infer_num_classes_from_output(graph_info.output_shapes, architecture_type)
        if dataset_info:
            self.logger.debug(
                f"Inferred {dataset_info.task} task with {dataset_info.num_classes} classes from output shape"
            )

        # Load Universal IR representation (optional, for advanced analysis)
        universal_graph = None
        try:
            from .format_adapters import load_model

            self.logger.debug("Loading Universal IR representation...")
            universal_graph = load_model(model_path)
            self.logger.debug(
                f"Universal IR loaded: {universal_graph.num_nodes} nodes, "
                f"{universal_graph.total_parameters:,} params"
            )
        except Exception as e:
            self.logger.debug(f"Universal IR loading skipped: {e}")

        report = InspectionReport(
            metadata=metadata,
            graph_summary=graph_summary,
            param_counts=param_counts,
            flop_counts=flop_counts,
            memory_estimates=memory_estimates,
            detected_blocks=detected_blocks,
            architecture_type=architecture_type,
            risk_signals=risk_signals,
            dataset_info=dataset_info,
            universal_graph=universal_graph,
        )

        self.logger.info(
            f"Inspection complete. Found {len(detected_blocks)} blocks, {len(risk_signals)} risks."
        )
        return report

    def _extract_metadata(self, model, model_path: pathlib.Path) -> ModelMetadata:
        """Extract metadata from ONNX ModelProto."""
        from .analyzer import get_opsets_imported

        opsets = get_opsets_imported(model)

        return ModelMetadata(
            path=str(model_path),
            ir_version=model.ir_version,
            producer_name=model.producer_name or "unknown",
            producer_version=model.producer_version or "",
            domain=model.domain or "",
            model_version=model.model_version,
            doc_string=model.doc_string or "",
            opsets=opsets,
        )

    def _build_graph_summary(self, graph_info) -> GraphSummary:
        """Build summary statistics from GraphInfo."""
        return GraphSummary(
            num_nodes=graph_info.num_nodes,
            num_inputs=len(graph_info.inputs),
            num_outputs=len(graph_info.outputs),
            num_initializers=len(graph_info.initializers),
            input_shapes=graph_info.input_shapes,
            output_shapes=graph_info.output_shapes,
            op_type_counts=graph_info.op_type_counts,
        )
