# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
Quantization Linting - Static analysis for quantization readiness.

Provides "preflight checks" to help users determine if their model is good
quantization material before investing time in QAT or PTQ workflows.

Features:
- Detect quantization-unfriendly ops
- Identify dynamic shapes in problematic positions
- Flag ops with no ONNX quantization support
- Generate severity-ranked warnings with actionable recommendations

Usage:
    from haoline.quantization_linter import QuantizationLinter

    linter = QuantizationLinter()
    result = linter.lint(graph_info)
    print(result.readiness_score)  # 0-100
    for warning in result.warnings:
        print(f"{warning.severity}: {warning.message}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from haoline.analyzer import GraphInfo


class Severity(Enum):
    """Warning severity levels for quantization issues."""

    CRITICAL = "critical"  # Will likely break quantization entirely
    HIGH = "high"  # Significant accuracy loss expected
    MEDIUM = "medium"  # May cause accuracy issues
    LOW = "low"  # Minor concern, usually fine
    INFO = "info"  # Informational, no action needed


class QuantIssueType(Enum):
    """Types of quantization issues."""

    UNSUPPORTED_OP = "unsupported_op"
    ACCURACY_SENSITIVE_OP = "accuracy_sensitive_op"
    DYNAMIC_SHAPE = "dynamic_shape"
    MISSING_FAKE_QUANT = "missing_fake_quant"
    INCONSISTENT_QUANT = "inconsistent_quant"
    WIDE_ACTIVATION_RANGE = "wide_activation_range"
    CUSTOM_OP = "custom_op"
    NO_QUANT_KERNEL = "no_quant_kernel"


@dataclass
class QuantWarning:
    """A single quantization-related warning."""

    severity: Severity
    issue_type: QuantIssueType
    message: str
    node_name: str | None = None
    op_type: str | None = None
    recommendation: str | None = None
    details: dict | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "severity": self.severity.value,
            "issue_type": self.issue_type.value,
            "message": self.message,
            "node_name": self.node_name,
            "op_type": self.op_type,
            "recommendation": self.recommendation,
            "details": self.details,
        }


@dataclass
class OpQuantInfo:
    """Quantization characteristics for an operator type."""

    op_type: str
    has_int8_kernel: bool = True  # Has optimized INT8 implementation
    accuracy_sensitive: bool = False  # Typically causes accuracy loss in INT8
    requires_calibration: bool = True  # Needs proper calibration data
    per_channel_recommended: bool = False  # Per-channel quant recommended
    keep_fp16_recommended: bool = False  # Better to keep at FP16/FP32
    notes: str = ""


# =============================================================================
# QUANTIZATION-UNFRIENDLY OPS DATABASE
# =============================================================================
# Based on real-world experience with TensorRT, ONNX Runtime, and QAT workflows.
# Ops are categorized by their quantization friendliness.

# Ops that have NO INT8 kernel in most runtimes
_NO_INT8_KERNEL_OPS: set[str] = {
    # Normalization (typically kept at FP16/FP32)
    "LayerNormalization",
    "InstanceNormalization",
    "GroupNormalization",
    "LpNormalization",
    "MeanVarianceNormalization",
    # Attention/Transformer components (complex, accuracy-sensitive)
    "Attention",
    "MultiHeadAttention",
    # Reduction ops (accumulation precision matters)
    "ReduceMean",
    "ReduceSum",
    "ReduceL1",
    "ReduceL2",
    "ReduceLogSum",
    "ReduceLogSumExp",
    "ReduceSumSquare",
    # Trigonometric / special functions
    "Sin",
    "Cos",
    "Tan",
    "Asin",
    "Acos",
    "Atan",
    "Sinh",
    "Cosh",
    "Tanh",  # Note: Tanh sometimes has INT8, but often problematic
    "Asinh",
    "Acosh",
    "Atanh",
    # Exponential / logarithmic
    "Exp",
    "Log",
    "Pow",
    "Sqrt",
    "Reciprocal",
    # Softmax family (should stay FP16/FP32 for accuracy)
    "Softmax",
    "LogSoftmax",
    "Hardmax",
    # Complex ops
    "LSTM",
    "GRU",
    "RNN",
    # Misc
    "Einsum",
    "NonMaxSuppression",
    "RoiAlign",
    "GridSample",
}

# Ops that HAVE INT8 kernels but are accuracy-sensitive
_ACCURACY_SENSITIVE_OPS: set[str] = {
    # Final classifier layers (small dynamic range, high impact)
    "Gemm",  # When used as final classifier
    # Residual connections (scale mismatches accumulate)
    "Add",  # When in residual path
    # Concatenation (scale alignment issues)
    "Concat",
    # Pooling (information aggregation)
    "GlobalAveragePool",
    "GlobalMaxPool",
    # Batch normalization (folded into conv, but sensitive)
    "BatchNormalization",
}

# Ops that are typically fine for INT8
_QUANT_FRIENDLY_OPS: set[str] = {
    "Conv",
    "ConvTranspose",
    "MatMul",
    "Relu",
    "LeakyRelu",
    "PRelu",
    "Clip",
    "MaxPool",
    "AveragePool",
    "Flatten",
    "Reshape",
    "Transpose",
    "Squeeze",
    "Unsqueeze",
    "Pad",
    "Slice",
    "Split",
    "Gather",
    "Shape",
    "Constant",
    "Identity",
    "Dropout",  # Usually removed at inference
}

# Ops that indicate already-quantized model
_QUANTIZATION_OPS: set[str] = {
    "QuantizeLinear",
    "DequantizeLinear",
    "DynamicQuantizeLinear",
    "QLinearConv",
    "QLinearMatMul",
    "ConvInteger",
    "MatMulInteger",
    "QLinearAdd",
    "QLinearMul",
    "QLinearSigmoid",
    "QLinearLeakyRelu",
    "QLinearSoftmax",
    "QLinearGlobalAveragePool",
    "QLinearAveragePool",
    "QLinearConcat",
}

# Detailed info for specific ops
_OP_QUANT_INFO: dict[str, OpQuantInfo] = {
    "LayerNormalization": OpQuantInfo(
        op_type="LayerNormalization",
        has_int8_kernel=False,
        accuracy_sensitive=True,
        keep_fp16_recommended=True,
        notes="Keep at FP16. Consider RMSNorm as INT8-friendly alternative.",
    ),
    "Softmax": OpQuantInfo(
        op_type="Softmax",
        has_int8_kernel=False,
        accuracy_sensitive=True,
        keep_fp16_recommended=True,
        notes="Keep at FP16/FP32. Exponential operations lose precision in INT8.",
    ),
    "GELU": OpQuantInfo(
        op_type="Gelu",
        has_int8_kernel=False,
        accuracy_sensitive=True,
        keep_fp16_recommended=True,
        notes="Keep at FP16. Consider ReLU or approximate GELU for INT8.",
    ),
    "Attention": OpQuantInfo(
        op_type="Attention",
        has_int8_kernel=False,
        accuracy_sensitive=True,
        keep_fp16_recommended=True,
        notes="Fused attention ops typically need FP16. Use quantized SDPA if available.",
    ),
    "Conv": OpQuantInfo(
        op_type="Conv",
        has_int8_kernel=True,
        accuracy_sensitive=False,
        per_channel_recommended=True,
        notes="Excellent INT8 support. Use per-channel quantization for best accuracy.",
    ),
    "Gemm": OpQuantInfo(
        op_type="Gemm",
        has_int8_kernel=True,
        accuracy_sensitive=True,  # When used as classifier
        per_channel_recommended=True,
        notes="Good INT8 support, but final classifier layers may need FP16.",
    ),
    "MatMul": OpQuantInfo(
        op_type="MatMul",
        has_int8_kernel=True,
        accuracy_sensitive=False,
        per_channel_recommended=True,
        notes="Good INT8 support with per-channel quantization.",
    ),
    "BatchNormalization": OpQuantInfo(
        op_type="BatchNormalization",
        has_int8_kernel=True,
        accuracy_sensitive=True,
        notes="Typically folded into preceding Conv. If standalone, keep FP16.",
    ),
    "Add": OpQuantInfo(
        op_type="Add",
        has_int8_kernel=True,
        accuracy_sensitive=True,  # In residual paths
        notes="Residual Add ops need matching quantization scales on both inputs.",
    ),
}


@dataclass
class QuantizationLintResult:
    """Result of quantization linting analysis."""

    # Overall assessment
    readiness_score: int = 100  # 0-100, higher is better
    is_already_quantized: bool = False
    has_critical_issues: bool = False

    # Warnings by severity
    warnings: list[QuantWarning] = field(default_factory=list)

    # Op analysis
    unsupported_ops: dict[str, int] = field(default_factory=dict)  # op_type -> count
    accuracy_sensitive_ops: dict[str, int] = field(default_factory=dict)
    quant_friendly_ops: dict[str, int] = field(default_factory=dict)
    custom_ops: list[str] = field(default_factory=list)
    quantization_ops: dict[str, int] = field(default_factory=dict)

    # Shape analysis
    dynamic_shape_nodes: list[str] = field(default_factory=list)

    # Problem layers (for targeted recommendations)
    problem_layers: list[dict] = field(default_factory=list)

    # Summary stats
    total_ops: int = 0
    quant_friendly_pct: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "readiness_score": self.readiness_score,
            "is_already_quantized": self.is_already_quantized,
            "has_critical_issues": self.has_critical_issues,
            "warnings": [w.to_dict() for w in self.warnings],
            "unsupported_ops": self.unsupported_ops,
            "accuracy_sensitive_ops": self.accuracy_sensitive_ops,
            "quant_friendly_ops": self.quant_friendly_ops,
            "custom_ops": self.custom_ops,
            "quantization_ops": self.quantization_ops,
            "dynamic_shape_nodes": self.dynamic_shape_nodes,
            "problem_layers": self.problem_layers,
            "total_ops": self.total_ops,
            "quant_friendly_pct": self.quant_friendly_pct,
        }

    @property
    def warnings_by_severity(self) -> dict[Severity, list[QuantWarning]]:
        """Group warnings by severity."""
        result: dict[Severity, list[QuantWarning]] = {s: [] for s in Severity}
        for w in self.warnings:
            result[w.severity].append(w)
        return result

    @property
    def critical_count(self) -> int:
        """Number of critical issues."""
        return len([w for w in self.warnings if w.severity == Severity.CRITICAL])

    @property
    def high_count(self) -> int:
        """Number of high severity issues."""
        return len([w for w in self.warnings if w.severity == Severity.HIGH])

    def get_summary(self) -> str:
        """Get human-readable summary."""
        if self.readiness_score >= 90:
            grade = "Excellent"
            emoji = "A"
        elif self.readiness_score >= 75:
            grade = "Good"
            emoji = "B"
        elif self.readiness_score >= 60:
            grade = "Fair"
            emoji = "C"
        elif self.readiness_score >= 40:
            grade = "Poor"
            emoji = "D"
        else:
            grade = "Not Recommended"
            emoji = "F"

        return (
            f"Quantization Readiness: {self.readiness_score}/100 ({grade}) [{emoji}]\n"
            f"Critical Issues: {self.critical_count}, High: {self.high_count}\n"
            f"Quant-Friendly Ops: {self.quant_friendly_pct:.1f}%"
        )


class QuantizationLinter:
    """
    Static analyzer for quantization readiness.

    Analyzes ONNX graphs to determine how suitable they are for INT8 quantization,
    identifying problematic ops, dynamic shapes, and providing actionable recommendations.
    """

    def __init__(self, logger: logging.Logger | None = None):
        """Initialize the linter.

        Args:
            logger: Optional logger for debug output.
        """
        self.logger = logger or logging.getLogger("haoline.quant_linter")

    def lint(self, graph_info: GraphInfo) -> QuantizationLintResult:
        """
        Analyze a graph for quantization readiness.

        Args:
            graph_info: Parsed ONNX graph information.

        Returns:
            QuantizationLintResult with score, warnings, and recommendations.
        """
        result = QuantizationLintResult()
        result.total_ops = len(graph_info.nodes)

        # Phase 1: Classify all ops
        self._classify_ops(graph_info, result)

        # Phase 2: Detect dynamic shapes
        self._detect_dynamic_shapes(graph_info, result)

        # Phase 3: Check for custom/unknown ops
        self._detect_custom_ops(graph_info, result)

        # Phase 4: Generate warnings for each issue
        self._generate_warnings(graph_info, result)

        # Phase 5: Calculate readiness score
        self._calculate_score(result)

        # Phase 6: Identify problem layers
        self._identify_problem_layers(graph_info, result)

        return result

    def _classify_ops(self, graph_info: GraphInfo, result: QuantizationLintResult) -> None:
        """Classify all ops by their quantization friendliness."""
        for node in graph_info.nodes:
            op_type = node.op_type

            if op_type in _QUANTIZATION_OPS:
                result.quantization_ops[op_type] = result.quantization_ops.get(op_type, 0) + 1
                result.is_already_quantized = True

            elif op_type in _NO_INT8_KERNEL_OPS:
                result.unsupported_ops[op_type] = result.unsupported_ops.get(op_type, 0) + 1

            elif op_type in _ACCURACY_SENSITIVE_OPS:
                result.accuracy_sensitive_ops[op_type] = (
                    result.accuracy_sensitive_ops.get(op_type, 0) + 1
                )

            elif op_type in _QUANT_FRIENDLY_OPS:
                result.quant_friendly_ops[op_type] = result.quant_friendly_ops.get(op_type, 0) + 1

            # Check for custom ops (ops with non-standard domain)
            if node.domain and node.domain not in ("", "ai.onnx", "ai.onnx.ml"):
                if op_type not in result.custom_ops:
                    result.custom_ops.append(op_type)

        # Calculate quant-friendly percentage
        friendly_count = sum(result.quant_friendly_ops.values())
        if result.total_ops > 0:
            result.quant_friendly_pct = (friendly_count / result.total_ops) * 100

    def _detect_dynamic_shapes(self, graph_info: GraphInfo, result: QuantizationLintResult) -> None:
        """Detect nodes with dynamic shapes that could cause issues."""
        for node in graph_info.nodes:
            # Check if any output has dynamic dimensions
            for output_name in node.outputs:
                if output_name in graph_info.value_shapes:
                    shape = graph_info.value_shapes[output_name]
                    # Dynamic dim indicated by string (symbolic) or -1 or 0
                    has_dynamic = any(
                        isinstance(d, str) or (isinstance(d, int) and d <= 0) for d in shape
                    )
                    if has_dynamic:
                        result.dynamic_shape_nodes.append(node.name)
                        break

    def _detect_custom_ops(self, graph_info: GraphInfo, result: QuantizationLintResult) -> None:
        """Detect custom/unknown ops that may not have quantization support."""
        known_ops = (
            _QUANT_FRIENDLY_OPS | _NO_INT8_KERNEL_OPS | _ACCURACY_SENSITIVE_OPS | _QUANTIZATION_OPS
        )

        for node in graph_info.nodes:
            # If op is not in any known category and not already flagged as custom
            if node.op_type not in known_ops and node.op_type not in result.custom_ops:
                # Check domain - custom domain indicates custom op
                if node.domain and node.domain not in ("", "ai.onnx", "ai.onnx.ml"):
                    result.custom_ops.append(node.op_type)

    def _generate_warnings(self, graph_info: GraphInfo, result: QuantizationLintResult) -> None:
        """Generate warnings for all detected issues."""
        # Already quantized model
        if result.is_already_quantized:
            result.warnings.append(
                QuantWarning(
                    severity=Severity.INFO,
                    issue_type=QuantIssueType.UNSUPPORTED_OP,
                    message="Model already contains quantization ops",
                    recommendation="This model is already quantized. Re-quantization may degrade quality.",
                    details={"quantization_ops": result.quantization_ops},
                )
            )

        # Unsupported ops (no INT8 kernel)
        for op_type, count in result.unsupported_ops.items():
            info = _OP_QUANT_INFO.get(op_type)
            recommendation = info.notes if info else f"Consider keeping {op_type} at FP16/FP32."

            severity = Severity.HIGH
            if op_type in ("Softmax", "LayerNormalization", "Attention"):
                severity = Severity.HIGH  # These are very common in transformers
            elif op_type in ("Exp", "Log", "Pow"):
                severity = Severity.MEDIUM  # Less common

            result.warnings.append(
                QuantWarning(
                    severity=severity,
                    issue_type=QuantIssueType.NO_QUANT_KERNEL,
                    message=f"{op_type} has no efficient INT8 kernel ({count} instances)",
                    op_type=op_type,
                    recommendation=recommendation,
                    details={"count": count},
                )
            )

        # Accuracy-sensitive ops
        for op_type, count in result.accuracy_sensitive_ops.items():
            info = _OP_QUANT_INFO.get(op_type)
            recommendation = info.notes if info else f"Monitor {op_type} accuracy during QAT."

            result.warnings.append(
                QuantWarning(
                    severity=Severity.MEDIUM,
                    issue_type=QuantIssueType.ACCURACY_SENSITIVE_OP,
                    message=f"{op_type} is accuracy-sensitive in INT8 ({count} instances)",
                    op_type=op_type,
                    recommendation=recommendation,
                    details={"count": count},
                )
            )

        # Custom ops
        for op_type in result.custom_ops:
            result.warnings.append(
                QuantWarning(
                    severity=Severity.CRITICAL,
                    issue_type=QuantIssueType.CUSTOM_OP,
                    message=f"Custom op '{op_type}' may not have quantization support",
                    op_type=op_type,
                    recommendation="Check if your runtime supports INT8 for this custom op, "
                    "or replace with standard ONNX ops.",
                )
            )

        # Dynamic shapes
        if result.dynamic_shape_nodes:
            # Only warn if there are many dynamic nodes
            if len(result.dynamic_shape_nodes) > 5:
                result.warnings.append(
                    QuantWarning(
                        severity=Severity.MEDIUM,
                        issue_type=QuantIssueType.DYNAMIC_SHAPE,
                        message=f"{len(result.dynamic_shape_nodes)} nodes have dynamic shapes",
                        recommendation="Dynamic shapes may require dynamic quantization or "
                        "per-batch calibration. Consider using static shapes if possible.",
                        details={"nodes": result.dynamic_shape_nodes[:10]},  # First 10
                    )
                )

    def _calculate_score(self, result: QuantizationLintResult) -> int:
        """Calculate overall readiness score (0-100)."""
        score = 100

        # Deduct for critical issues (custom ops)
        score -= len(result.custom_ops) * 20  # -20 per custom op

        # Deduct for unsupported ops
        unsupported_count = sum(result.unsupported_ops.values())
        if unsupported_count > 0:
            # Scale by percentage of total ops
            unsupported_pct = (unsupported_count / max(result.total_ops, 1)) * 100
            score -= min(unsupported_pct * 0.5, 25)  # Max -25 for unsupported

        # Deduct for accuracy-sensitive ops
        sensitive_count = sum(result.accuracy_sensitive_ops.values())
        if sensitive_count > 0:
            sensitive_pct = (sensitive_count / max(result.total_ops, 1)) * 100
            score -= min(sensitive_pct * 0.2, 15)  # Max -15 for sensitive

        # Deduct for many dynamic shapes
        if len(result.dynamic_shape_nodes) > 10:
            score -= 10

        # Bonus if already quantized (someone already did the work!)
        if result.is_already_quantized:
            score = min(score + 10, 100)

        # Bonus for high quant-friendly percentage
        if result.quant_friendly_pct > 80:
            score = min(score + 5, 100)

        # Clamp to 0-100
        result.readiness_score = max(0, min(100, int(score)))
        result.has_critical_issues = result.critical_count > 0

        return result.readiness_score

    def _identify_problem_layers(
        self, graph_info: GraphInfo, result: QuantizationLintResult
    ) -> None:
        """Identify specific problem layers with recommendations."""
        for node in graph_info.nodes:
            op_type = node.op_type

            # Check if this node is problematic
            is_problem = False
            reason = ""
            recommendation = ""

            if op_type in _NO_INT8_KERNEL_OPS:
                is_problem = True
                reason = "No INT8 kernel"
                info = _OP_QUANT_INFO.get(op_type)
                recommendation = info.notes if info else "Keep at FP16"

            elif op_type in result.custom_ops:
                is_problem = True
                reason = "Custom op"
                recommendation = "Verify INT8 support in target runtime"

            if is_problem:
                result.problem_layers.append(
                    {
                        "name": node.name,
                        "op_type": op_type,
                        "reason": reason,
                        "recommendation": recommendation,
                    }
                )

    def get_recommendations(self, result: QuantizationLintResult) -> list[str]:
        """
        Generate actionable recommendations based on lint result.

        Args:
            result: The lint result to analyze.

        Returns:
            List of recommendation strings.
        """
        recommendations = []

        # Overall recommendation
        if result.readiness_score >= 90:
            recommendations.append(
                "This model is excellent for INT8 quantization. "
                "Proceed with standard PTQ or QAT workflow."
            )
        elif result.readiness_score >= 75:
            recommendations.append(
                "This model is good for quantization with minor adjustments. "
                "Consider keeping a few layers at FP16."
            )
        elif result.readiness_score >= 60:
            recommendations.append(
                "This model can be quantized but may need careful tuning. "
                "Use mixed-precision with sensitive ops at FP16."
            )
        elif result.readiness_score >= 40:
            recommendations.append(
                "Quantization will be challenging. Consider architectural changes "
                "or using FP16 for most of the model."
            )
        else:
            recommendations.append(
                "This model is not well-suited for INT8 quantization. "
                "Consider FP16 inference or architectural changes."
            )

        # Specific recommendations based on ops found
        if "LayerNormalization" in result.unsupported_ops:
            recommendations.append(
                "LayerNorm detected: Keep at FP16, or replace with RMSNorm for INT8."
            )

        if "Softmax" in result.unsupported_ops:
            recommendations.append(
                "Softmax detected: Keep at FP16. Do not quantize attention softmax."
            )

        if "GELU" in result.unsupported_ops or "Gelu" in result.unsupported_ops:
            recommendations.append(
                "GELU detected: Keep at FP16, or use approximate GELU / ReLU for INT8."
            )

        if result.custom_ops:
            recommendations.append(
                f"Custom ops detected ({', '.join(result.custom_ops)}): "
                "Verify INT8 support in your target runtime (TensorRT, ONNX Runtime, etc.)"
            )

        if "Add" in result.accuracy_sensitive_ops:
            recommendations.append(
                "Residual Add ops detected: Ensure matching quantization scales on both inputs "
                "to avoid accuracy loss."
            )

        if "Conv" in result.quant_friendly_ops:
            recommendations.append(
                "Conv layers detected: Use per-channel quantization for best accuracy."
            )

        return recommendations


def lint_model(
    graph_info: GraphInfo, logger: logging.Logger | None = None
) -> QuantizationLintResult:
    """
    Convenience function to lint a model for quantization readiness.

    Args:
        graph_info: Parsed ONNX graph information.
        logger: Optional logger.

    Returns:
        QuantizationLintResult with score and warnings.
    """
    linter = QuantizationLinter(logger=logger)
    return linter.lint(graph_info)
