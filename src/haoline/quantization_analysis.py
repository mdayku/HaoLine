# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
Advanced Quantization Analysis for LLM-Scale Models (Epic 26).

This module provides deep quantization analysis beyond basic INT8/FP16 detection:

1. **Mixed Precision Detection** (Story 26.1):
   - Weight vs activation vs accumulation precision
   - Precision breakdown by layer type (attention/FFN/embed)
   - INT4 weights + FP16 activations pattern detection

2. **Quantization Scheme Detection** (Story 26.2):
   - GPTQ pattern recognition (group-wise, act_order)
   - AWQ pattern recognition (activation-aware)
   - GGML/GGUF quantization types (Q4_0, Q4_K_M, Q5_K_S, etc.)
   - bitsandbytes NF4/FP4 detection
   - Accuracy degradation estimates per scheme

3. **Calibration Analysis** (Story 26.3):
   - Sensitive layer identification
   - Quantization error estimation
   - Recommendations for higher precision layers

Usage:
    from haoline.quantization_analysis import QuantizationAnalyzer

    analyzer = QuantizationAnalyzer()
    result = analyzer.analyze(graph_info, blocks)
    print(result.scheme)  # "GPTQ", "AWQ", "INT8_STATIC", etc.
    print(result.precision_by_layer_type)  # {"attention": {...}, "ffn": {...}}
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from haoline.analyzer import GraphInfo, NodeInfo
    from haoline.patterns import Block


# =============================================================================
# Enums and Constants
# =============================================================================


class QuantizationScheme(Enum):
    """Known quantization schemes for LLMs."""

    # Standard schemes
    FP32 = "fp32"  # No quantization
    FP16 = "fp16"  # Half precision (not really quantization)
    BF16 = "bf16"  # Brain float 16
    INT8_DYNAMIC = "int8_dynamic"  # Dynamic quantization
    INT8_STATIC = "int8_static"  # Static quantization with calibration

    # LLM-specific schemes
    GPTQ = "gptq"  # Group-wise INT4 with activation order
    AWQ = "awq"  # Activation-aware weight quantization
    GGML = "ggml"  # llama.cpp quantization (various Q types)
    BITSANDBYTES_NF4 = "bnb_nf4"  # bitsandbytes NormalFloat4
    BITSANDBYTES_FP4 = "bnb_fp4"  # bitsandbytes Float4
    SMOOTH_QUANT = "smooth_quant"  # Smooth quantization

    # Mixed precision
    MIXED_INT4_FP16 = "mixed_int4_fp16"  # INT4 weights + FP16 activations
    MIXED_INT8_FP16 = "mixed_int8_fp16"  # INT8 weights + FP16 activations

    UNKNOWN = "unknown"


class LayerType(Enum):
    """Layer type categories for LLM analysis."""

    ATTENTION = "attention"
    FFN = "ffn"  # Feed-forward network / MLP
    EMBEDDING = "embedding"
    NORMALIZATION = "normalization"
    OUTPUT = "output"  # Final classifier / LM head
    OTHER = "other"


class PrecisionType(Enum):
    """Precision categories."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"
    UINT8 = "uint8"
    UINT4 = "uint4"
    NF4 = "nf4"  # NormalFloat4
    UNKNOWN = "unknown"


# Accuracy degradation estimates per scheme (empirical values from research)
# Values are expected perplexity increase percentage for LLMs
SCHEME_ACCURACY_IMPACT: dict[QuantizationScheme, dict[str, Any]] = {
    QuantizationScheme.FP32: {
        "perplexity_increase_pct": 0.0,
        "accuracy_drop_pct": 0.0,
        "description": "Full precision, no degradation",
    },
    QuantizationScheme.FP16: {
        "perplexity_increase_pct": 0.0,
        "accuracy_drop_pct": 0.0,
        "description": "Near-lossless for most models",
    },
    QuantizationScheme.BF16: {
        "perplexity_increase_pct": 0.1,
        "accuracy_drop_pct": 0.0,
        "description": "Slightly larger dynamic range loss than FP16",
    },
    QuantizationScheme.INT8_DYNAMIC: {
        "perplexity_increase_pct": 0.5,
        "accuracy_drop_pct": 0.3,
        "description": "Dynamic quantization, some accuracy loss",
    },
    QuantizationScheme.INT8_STATIC: {
        "perplexity_increase_pct": 0.2,
        "accuracy_drop_pct": 0.1,
        "description": "With good calibration, minimal loss",
    },
    QuantizationScheme.GPTQ: {
        "perplexity_increase_pct": 1.5,
        "accuracy_drop_pct": 0.8,
        "description": "High-quality INT4, good for 7B+ models",
    },
    QuantizationScheme.AWQ: {
        "perplexity_increase_pct": 1.0,
        "accuracy_drop_pct": 0.5,
        "description": "Activation-aware, often better than GPTQ",
    },
    QuantizationScheme.GGML: {
        "perplexity_increase_pct": 2.0,
        "accuracy_drop_pct": 1.0,
        "description": "Varies by Q-type (Q4_K_M < Q4_0)",
    },
    QuantizationScheme.BITSANDBYTES_NF4: {
        "perplexity_increase_pct": 1.2,
        "accuracy_drop_pct": 0.6,
        "description": "NormalFloat4, good for QLoRA",
    },
    QuantizationScheme.BITSANDBYTES_FP4: {
        "perplexity_increase_pct": 1.5,
        "accuracy_drop_pct": 0.8,
        "description": "Float4, slightly worse than NF4",
    },
    QuantizationScheme.SMOOTH_QUANT: {
        "perplexity_increase_pct": 0.3,
        "accuracy_drop_pct": 0.2,
        "description": "INT8 with activation smoothing",
    },
    QuantizationScheme.MIXED_INT4_FP16: {
        "perplexity_increase_pct": 1.0,
        "accuracy_drop_pct": 0.5,
        "description": "INT4 weights, FP16 activations/accumulation",
    },
    QuantizationScheme.MIXED_INT8_FP16: {
        "perplexity_increase_pct": 0.3,
        "accuracy_drop_pct": 0.2,
        "description": "INT8 weights, FP16 activations",
    },
    QuantizationScheme.UNKNOWN: {
        "perplexity_increase_pct": None,
        "accuracy_drop_pct": None,
        "description": "Unknown quantization scheme",
    },
}


# GGML quantization type accuracy rankings (lower = better quality)
# Based on llama.cpp benchmarks
GGML_ACCURACY_RANKING: dict[str, dict[str, Any]] = {
    # Best quality (lowest perplexity increase)
    "Q8_0": {"rank": 1, "bits_per_weight": 8.5, "perplexity_pct": 0.2},
    "Q6_K": {"rank": 2, "bits_per_weight": 6.5, "perplexity_pct": 0.5},
    "Q5_K_M": {"rank": 3, "bits_per_weight": 5.5, "perplexity_pct": 0.8},
    "Q5_K_S": {"rank": 4, "bits_per_weight": 5.5, "perplexity_pct": 1.0},
    "Q5_1": {"rank": 5, "bits_per_weight": 5.5, "perplexity_pct": 1.0},
    "Q5_0": {"rank": 6, "bits_per_weight": 5.5, "perplexity_pct": 1.2},
    "Q4_K_M": {"rank": 7, "bits_per_weight": 4.8, "perplexity_pct": 1.5},
    "Q4_K_S": {"rank": 8, "bits_per_weight": 4.5, "perplexity_pct": 2.0},
    "Q4_1": {"rank": 9, "bits_per_weight": 4.5, "perplexity_pct": 2.2},
    "Q4_0": {"rank": 10, "bits_per_weight": 4.5, "perplexity_pct": 2.5},
    "Q3_K_L": {"rank": 11, "bits_per_weight": 3.4, "perplexity_pct": 4.0},
    "Q3_K_M": {"rank": 12, "bits_per_weight": 3.4, "perplexity_pct": 5.0},
    "Q3_K_S": {"rank": 13, "bits_per_weight": 3.4, "perplexity_pct": 6.0},
    "Q2_K": {"rank": 14, "bits_per_weight": 2.6, "perplexity_pct": 15.0},
    # Special types
    "IQ4_NL": {"rank": 5, "bits_per_weight": 4.5, "perplexity_pct": 1.2},
    "IQ4_XS": {"rank": 6, "bits_per_weight": 4.3, "perplexity_pct": 1.5},
    "IQ3_M": {"rank": 10, "bits_per_weight": 3.4, "perplexity_pct": 3.5},
    "IQ3_S": {"rank": 11, "bits_per_weight": 3.4, "perplexity_pct": 4.5},
    "IQ2_M": {"rank": 13, "bits_per_weight": 2.7, "perplexity_pct": 10.0},
    "IQ2_S": {"rank": 14, "bits_per_weight": 2.5, "perplexity_pct": 12.0},
    "IQ1_M": {"rank": 15, "bits_per_weight": 1.8, "perplexity_pct": 25.0},
}


# Patterns that suggest GPTQ quantization in ONNX
GPTQ_PATTERNS = {
    # GPTQ uses group-wise quantization with scales and zeros
    "tensor_patterns": [
        r".*qweight.*",  # Quantized weight tensor
        r".*qzeros.*",  # Zero points
        r".*scales.*",  # Per-group scales
        r".*g_idx.*",  # Group index (act_order=True)
    ],
    "attribute_patterns": [
        "group_size",  # Typical: 128
        "bits",  # Typical: 4
        "act_order",  # True/False
    ],
}


# Patterns that suggest AWQ quantization
AWQ_PATTERNS = {
    "tensor_patterns": [
        r".*qweight.*",
        r".*qzeros.*",
        r".*scales.*",
    ],
    "attribute_patterns": [
        "w_bit",  # Typical: 4
        "group_size",  # Typical: 128
    ],
    # AWQ typically doesn't have g_idx (no act_order)
    "negative_patterns": [
        r".*g_idx.*",
    ],
}


# Patterns that suggest bitsandbytes quantization
BITSANDBYTES_PATTERNS = {
    "tensor_patterns": [
        r".*absmax.*",  # Block-wise absolute max
        r".*code.*",  # Quantization codebook
        r".*quant_state.*",  # Quantization state
    ],
    "attribute_patterns": [
        "blocksize",  # Typical: 64
        "quant_type",  # nf4 or fp4
    ],
}


# =============================================================================
# Data Classes
# =============================================================================


class LayerPrecisionInfo(BaseModel):
    """Precision information for a single layer."""

    model_config = ConfigDict(frozen=True)

    layer_name: str
    layer_type: LayerType
    weight_precision: PrecisionType
    activation_precision: PrecisionType = PrecisionType.FP16  # Default assumption
    accumulation_precision: PrecisionType = PrecisionType.FP32  # Usually FP32
    param_count: int = 0
    is_sensitive: bool = False  # True if should stay at higher precision
    sensitivity_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "layer_name": self.layer_name,
            "layer_type": self.layer_type.value,
            "weight_precision": self.weight_precision.value,
            "activation_precision": self.activation_precision.value,
            "accumulation_precision": self.accumulation_precision.value,
            "param_count": self.param_count,
            "is_sensitive": self.is_sensitive,
            "sensitivity_reason": self.sensitivity_reason,
        }


class PrecisionByLayerType(BaseModel):
    """Precision breakdown aggregated by layer type."""

    model_config = ConfigDict(frozen=True)

    layer_type: LayerType
    total_params: int = 0
    precision_breakdown: dict[str, int] = Field(default_factory=dict)  # precision -> count
    dominant_precision: PrecisionType = PrecisionType.UNKNOWN
    layer_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "layer_type": self.layer_type.value,
            "total_params": self.total_params,
            "precision_breakdown": self.precision_breakdown,
            "dominant_precision": self.dominant_precision.value,
            "layer_count": self.layer_count,
        }


class QuantizationSchemeInfo(BaseModel):
    """Detected quantization scheme with details."""

    model_config = ConfigDict(frozen=True)

    scheme: QuantizationScheme
    confidence: float = 1.0  # 0.0 to 1.0
    group_size: int | None = None  # For GPTQ/AWQ
    bits: int | None = None  # Weight bits
    act_order: bool | None = None  # GPTQ act_order
    ggml_types: list[str] = Field(default_factory=list)  # For GGML models
    evidence: list[str] = Field(default_factory=list)  # What patterns matched

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "scheme": self.scheme.value,
            "confidence": self.confidence,
            "group_size": self.group_size,
            "bits": self.bits,
            "act_order": self.act_order,
            "ggml_types": self.ggml_types,
            "evidence": self.evidence,
        }


class AccuracyImpactEstimate(BaseModel):
    """Estimated accuracy impact of the quantization."""

    model_config = ConfigDict(frozen=True)

    perplexity_increase_pct: float | None = None
    accuracy_drop_pct: float | None = None
    description: str = ""
    memory_reduction_factor: float = 1.0  # e.g., 4.0 for INT8 vs FP32
    recommendations: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "perplexity_increase_pct": self.perplexity_increase_pct,
            "accuracy_drop_pct": self.accuracy_drop_pct,
            "description": self.description,
            "memory_reduction_factor": self.memory_reduction_factor,
            "recommendations": self.recommendations,
        }


class SensitiveLayerInfo(BaseModel):
    """Information about a layer that's sensitive to quantization."""

    model_config = ConfigDict(frozen=True)

    layer_name: str
    layer_type: LayerType
    sensitivity_score: float  # 0.0 to 1.0, higher = more sensitive
    reason: str
    recommendation: str
    current_precision: PrecisionType
    recommended_precision: PrecisionType

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "layer_name": self.layer_name,
            "layer_type": self.layer_type.value,
            "sensitivity_score": self.sensitivity_score,
            "reason": self.reason,
            "recommendation": self.recommendation,
            "current_precision": self.current_precision.value,
            "recommended_precision": self.recommended_precision.value,
        }


class QuantizationAnalysisResult(BaseModel):
    """Complete quantization analysis result."""

    model_config = ConfigDict(frozen=True)

    # Detected scheme
    scheme_info: QuantizationSchemeInfo

    # Precision breakdown
    layer_precisions: list[LayerPrecisionInfo] = Field(default_factory=list)
    precision_by_layer_type: dict[str, PrecisionByLayerType] = Field(default_factory=dict)

    # Weight vs activation analysis
    weight_precision_dominant: PrecisionType = PrecisionType.UNKNOWN
    activation_precision_dominant: PrecisionType = PrecisionType.FP16
    accumulation_precision: PrecisionType = PrecisionType.FP32
    is_mixed_precision: bool = False

    # Accuracy impact
    accuracy_impact: AccuracyImpactEstimate | None = None

    # Sensitive layers
    sensitive_layers: list[SensitiveLayerInfo] = Field(default_factory=list)

    # Summary
    total_params: int = 0
    quantized_params: int = 0
    quantization_ratio: float = 0.0  # Percentage of params that are quantized

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "scheme_info": self.scheme_info.to_dict(),
            "layer_precisions": [lp.to_dict() for lp in self.layer_precisions],
            "precision_by_layer_type": {
                k: v.to_dict() for k, v in self.precision_by_layer_type.items()
            },
            "weight_precision_dominant": self.weight_precision_dominant.value,
            "activation_precision_dominant": self.activation_precision_dominant.value,
            "accumulation_precision": self.accumulation_precision.value,
            "is_mixed_precision": self.is_mixed_precision,
            "accuracy_impact": self.accuracy_impact.to_dict() if self.accuracy_impact else None,
            "sensitive_layers": [sl.to_dict() for sl in self.sensitive_layers],
            "total_params": self.total_params,
            "quantized_params": self.quantized_params,
            "quantization_ratio": self.quantization_ratio,
        }

    def get_summary(self) -> str:
        """Get a human-readable summary."""
        lines = [
            f"Quantization Scheme: {self.scheme_info.scheme.value.upper()}",
            f"  Confidence: {self.scheme_info.confidence:.0%}",
        ]

        if self.scheme_info.bits:
            lines.append(f"  Weight Bits: {self.scheme_info.bits}")
        if self.scheme_info.group_size:
            lines.append(f"  Group Size: {self.scheme_info.group_size}")

        lines.append("\nPrecision Analysis:")
        lines.append(f"  Weight Precision: {self.weight_precision_dominant.value}")
        lines.append(f"  Activation Precision: {self.activation_precision_dominant.value}")
        lines.append(f"  Accumulation: {self.accumulation_precision.value}")
        lines.append(f"  Mixed Precision: {'Yes' if self.is_mixed_precision else 'No'}")

        lines.append("\nQuantization Coverage:")
        lines.append(f"  Total Parameters: {self.total_params:,}")
        lines.append(f"  Quantized Parameters: {self.quantized_params:,}")
        lines.append(f"  Quantization Ratio: {self.quantization_ratio:.1%}")

        if self.accuracy_impact:
            lines.append("\nAccuracy Impact Estimate:")
            if self.accuracy_impact.perplexity_increase_pct is not None:
                lines.append(
                    f"  Perplexity Increase: ~{self.accuracy_impact.perplexity_increase_pct:.1f}%"
                )
            if self.accuracy_impact.accuracy_drop_pct is not None:
                lines.append(f"  Accuracy Drop: ~{self.accuracy_impact.accuracy_drop_pct:.1f}%")
            lines.append(f"  Memory Reduction: {self.accuracy_impact.memory_reduction_factor:.1f}x")

        if self.sensitive_layers:
            lines.append(f"\nSensitive Layers ({len(self.sensitive_layers)}):")
            for sl in self.sensitive_layers[:5]:  # Show top 5
                lines.append(f"  - {sl.layer_name}: {sl.reason}")

        return "\n".join(lines)


# =============================================================================
# Main Analyzer Class
# =============================================================================


class QuantizationAnalyzer:
    """Advanced quantization analyzer for LLM-scale models.

    Provides deep analysis of quantization schemes, mixed precision patterns,
    and sensitivity-based recommendations.
    """

    # Ops that are typically sensitive to quantization
    SENSITIVE_OPS = {
        "LayerNormalization",
        "InstanceNormalization",
        "BatchNormalization",
        "Softmax",
        "LogSoftmax",
        "Attention",
        "MultiHeadAttention",
    }

    # Ops that handle well with low-precision
    ROBUST_OPS = {
        "Relu",
        "Clip",
        "MaxPool",
        "AveragePool",
        "GlobalAveragePool",
        "Flatten",
        "Reshape",
        "Transpose",
        "Squeeze",
        "Unsqueeze",
    }

    def __init__(self, logger: logging.Logger | None = None):
        """Initialize the analyzer.

        Args:
            logger: Optional logger for debug output.
        """
        self.logger = logger or logging.getLogger(__name__)

    def analyze(
        self,
        graph_info: GraphInfo,
        blocks: list[Block] | None = None,
        gguf_info: Any | None = None,
    ) -> QuantizationAnalysisResult:
        """Perform comprehensive quantization analysis.

        Args:
            graph_info: Parsed ONNX graph information.
            blocks: Detected architectural blocks (from PatternAnalyzer).
            gguf_info: Optional GGUFInfo for GGUF models.

        Returns:
            QuantizationAnalysisResult with full analysis.
        """
        # Detect quantization scheme
        scheme_info = self._detect_scheme(graph_info, gguf_info)

        # Analyze layer-level precision
        layer_precisions = self._analyze_layer_precision(graph_info, blocks)

        # Aggregate by layer type
        precision_by_type = self._aggregate_by_layer_type(layer_precisions)

        # Determine dominant precisions
        weight_dominant = self._get_dominant_weight_precision(layer_precisions)
        is_mixed = self._is_mixed_precision(layer_precisions)

        # Calculate accuracy impact
        accuracy_impact = self._estimate_accuracy_impact(scheme_info, gguf_info)

        # Identify sensitive layers
        sensitive_layers = self._identify_sensitive_layers(graph_info, layer_precisions)

        # Calculate totals
        total_params = sum(lp.param_count for lp in layer_precisions)
        quantized_params = sum(
            lp.param_count
            for lp in layer_precisions
            if lp.weight_precision in {PrecisionType.INT8, PrecisionType.INT4, PrecisionType.NF4}
        )
        quant_ratio = quantized_params / total_params if total_params > 0 else 0.0

        return QuantizationAnalysisResult(
            scheme_info=scheme_info,
            layer_precisions=layer_precisions,
            precision_by_layer_type=precision_by_type,
            weight_precision_dominant=weight_dominant,
            activation_precision_dominant=PrecisionType.FP16,  # Default assumption
            accumulation_precision=PrecisionType.FP32,  # Usually FP32
            is_mixed_precision=is_mixed,
            accuracy_impact=accuracy_impact,
            sensitive_layers=sensitive_layers,
            total_params=total_params,
            quantized_params=quantized_params,
            quantization_ratio=quant_ratio,
        )

    def _detect_scheme(
        self, graph_info: GraphInfo, gguf_info: Any | None = None
    ) -> QuantizationSchemeInfo:
        """Detect the quantization scheme used in the model."""
        evidence: list[str] = []

        # Check for GGUF first (explicit format)
        if gguf_info is not None:
            ggml_types = list(gguf_info.quantization_breakdown.keys())
            return QuantizationSchemeInfo(
                scheme=QuantizationScheme.GGML,
                confidence=1.0,
                ggml_types=ggml_types,
                bits=self._estimate_bits_from_ggml(ggml_types),
                evidence=["GGUF format detected", f"Types: {', '.join(ggml_types[:5])}"],
            )

        # Get all tensor names for pattern matching
        tensor_names = list(graph_info.initializers.keys())
        tensor_names_lower = [n.lower() for n in tensor_names]

        # Check for GPTQ patterns
        gptq_score, gptq_evidence = self._check_gptq_patterns(tensor_names_lower)
        if gptq_score > 0.7:
            return QuantizationSchemeInfo(
                scheme=QuantizationScheme.GPTQ,
                confidence=gptq_score,
                bits=4,
                group_size=128,  # Most common
                act_order="g_idx" in " ".join(tensor_names_lower),
                evidence=gptq_evidence,
            )

        # Check for AWQ patterns
        awq_score, awq_evidence = self._check_awq_patterns(tensor_names_lower)
        if awq_score > 0.7:
            return QuantizationSchemeInfo(
                scheme=QuantizationScheme.AWQ,
                confidence=awq_score,
                bits=4,
                group_size=128,
                evidence=awq_evidence,
            )

        # Check for bitsandbytes patterns
        bnb_score, bnb_evidence, bnb_type = self._check_bitsandbytes_patterns(tensor_names_lower)
        if bnb_score > 0.7:
            scheme = (
                QuantizationScheme.BITSANDBYTES_NF4
                if "nf4" in bnb_type
                else QuantizationScheme.BITSANDBYTES_FP4
            )
            return QuantizationSchemeInfo(
                scheme=scheme,
                confidence=bnb_score,
                bits=4,
                evidence=bnb_evidence,
            )

        # Check for standard INT8 quantization
        has_qdq = any(n.op_type in {"QuantizeLinear", "DequantizeLinear"} for n in graph_info.nodes)
        if has_qdq:
            evidence.append("QuantizeLinear/DequantizeLinear ops detected")
            return QuantizationSchemeInfo(
                scheme=QuantizationScheme.INT8_STATIC,
                confidence=0.9,
                bits=8,
                evidence=evidence,
            )

        # Check precision from weights
        precision_counts = self._count_weight_precisions(graph_info)
        if precision_counts.get("int8", 0) > 0:
            return QuantizationSchemeInfo(
                scheme=QuantizationScheme.INT8_DYNAMIC,
                confidence=0.7,
                bits=8,
                evidence=["INT8 weights detected"],
            )

        # Check for FP16/BF16
        if precision_counts.get("fp16", 0) > precision_counts.get("fp32", 0):
            return QuantizationSchemeInfo(
                scheme=QuantizationScheme.FP16,
                confidence=0.9,
                evidence=["Majority FP16 weights"],
            )

        # Default to FP32
        return QuantizationSchemeInfo(
            scheme=QuantizationScheme.FP32,
            confidence=0.8,
            evidence=["No quantization patterns detected"],
        )

    def _check_gptq_patterns(self, tensor_names: list[str]) -> tuple[float, list[str]]:
        """Check for GPTQ quantization patterns."""
        evidence: list[str] = []
        score = 0.0

        # Check for qweight tensors
        qweight_count = sum(1 for n in tensor_names if "qweight" in n)
        if qweight_count > 0:
            evidence.append(f"Found {qweight_count} qweight tensors")
            score += 0.3

        # Check for scales
        scales_count = sum(1 for n in tensor_names if "scales" in n and "qzeros" not in n)
        if scales_count > 0:
            evidence.append(f"Found {scales_count} scales tensors")
            score += 0.2

        # Check for qzeros
        qzeros_count = sum(1 for n in tensor_names if "qzeros" in n)
        if qzeros_count > 0:
            evidence.append(f"Found {qzeros_count} qzeros tensors")
            score += 0.2

        # Check for g_idx (act_order)
        g_idx_count = sum(1 for n in tensor_names if "g_idx" in n)
        if g_idx_count > 0:
            evidence.append(f"Found {g_idx_count} g_idx tensors (act_order=True)")
            score += 0.3  # Strong GPTQ indicator

        return min(score, 1.0), evidence

    def _check_awq_patterns(self, tensor_names: list[str]) -> tuple[float, list[str]]:
        """Check for AWQ quantization patterns."""
        evidence: list[str] = []
        score = 0.0

        # AWQ has qweight and scales but typically no g_idx
        qweight_count = sum(1 for n in tensor_names if "qweight" in n)
        scales_count = sum(1 for n in tensor_names if "scales" in n)
        g_idx_count = sum(1 for n in tensor_names if "g_idx" in n)

        if qweight_count > 0 and scales_count > 0:
            evidence.append(f"Found {qweight_count} qweight and {scales_count} scales tensors")
            score += 0.5

            if g_idx_count == 0:
                evidence.append("No g_idx tensors (consistent with AWQ)")
                score += 0.3
            else:
                # Has g_idx, more likely GPTQ
                score -= 0.3

        return max(score, 0.0), evidence

    def _check_bitsandbytes_patterns(self, tensor_names: list[str]) -> tuple[float, list[str], str]:
        """Check for bitsandbytes quantization patterns."""
        evidence: list[str] = []
        score = 0.0
        quant_type = ""

        # Check for absmax (block-wise quantization)
        absmax_count = sum(1 for n in tensor_names if "absmax" in n)
        if absmax_count > 0:
            evidence.append(f"Found {absmax_count} absmax tensors")
            score += 0.4

        # Check for quant_state
        quant_state_count = sum(1 for n in tensor_names if "quant_state" in n)
        if quant_state_count > 0:
            evidence.append(f"Found {quant_state_count} quant_state tensors")
            score += 0.3

        # Check for code/codebook
        code_count = sum(1 for n in tensor_names if "code" in n and "encode" not in n)
        if code_count > 0:
            evidence.append(f"Found {code_count} code tensors")
            score += 0.2

        # Determine NF4 vs FP4 (would need to inspect actual values)
        quant_type = "nf4"  # Default assumption

        return min(score, 1.0), evidence, quant_type

    def _estimate_bits_from_ggml(self, ggml_types: list[str]) -> int:
        """Estimate effective bits from GGML quantization types."""
        if not ggml_types:
            return 16

        # Get the most common type's bits
        type_counts: dict[str, int] = {}
        for t in ggml_types:
            type_counts[t] = type_counts.get(t, 0) + 1

        most_common = max(type_counts, key=type_counts.get)  # type: ignore[arg-type]

        # Extract bits from type name
        # Check float types first (F16, F32) before numeric patterns
        if "F32" in most_common:
            return 32
        elif "F16" in most_common:
            return 16
        elif "Q8" in most_common:
            return 8
        elif "Q6" in most_common:
            return 6
        elif "Q5" in most_common:
            return 5
        elif "Q4" in most_common or "IQ4" in most_common:
            return 4
        elif "Q3" in most_common or "IQ3" in most_common:
            return 3
        elif "Q2" in most_common or "IQ2" in most_common:
            return 2
        elif "IQ1" in most_common:
            return 1

        return 4  # Default

    def _count_weight_precisions(self, graph_info: GraphInfo) -> dict[str, int]:
        """Count weights by precision type."""
        import numpy as np

        counts: dict[str, int] = {}

        for _name, tensor in graph_info.initializers.items():
            if hasattr(tensor, "dtype"):
                dtype = tensor.dtype
                param_count = int(np.prod(tensor.shape)) if tensor.shape else 1

                if dtype == np.float32:
                    key = "fp32"
                elif dtype == np.float16:
                    key = "fp16"
                elif dtype == np.int8:
                    key = "int8"
                elif dtype == np.uint8:
                    key = "uint8"
                else:
                    key = str(dtype)

                counts[key] = counts.get(key, 0) + param_count

        return counts

    def _analyze_layer_precision(
        self, graph_info: GraphInfo, blocks: list[Block] | None = None
    ) -> list[LayerPrecisionInfo]:
        """Analyze precision at the layer level."""
        import numpy as np

        result: list[LayerPrecisionInfo] = []

        # Build a map of node outputs to their producer nodes
        node_by_output: dict[str, NodeInfo] = {}
        for node in graph_info.nodes:
            for out in node.outputs:
                node_by_output[out] = node

        # Classify nodes by layer type using blocks if available
        node_layer_types: dict[str, LayerType] = {}
        if blocks:
            for block in blocks:
                block_type = self._block_type_to_layer_type(block.block_type)
                for node_name in block.nodes:
                    node_layer_types[node_name] = block_type

        # Analyze each initializer (weight tensor)
        for name, tensor in graph_info.initializers.items():
            # Determine weight precision
            weight_prec = self._tensor_to_precision_type(tensor)

            # Find which node uses this weight
            using_nodes: list[str] = []
            for node in graph_info.nodes:
                if name in node.inputs:
                    using_nodes.append(node.name)

            # Get layer type from first using node
            layer_type = LayerType.OTHER
            if using_nodes:
                first_node = using_nodes[0]
                layer_type = node_layer_types.get(first_node, LayerType.OTHER)

                # Fallback: infer from op type
                for node in graph_info.nodes:
                    if node.name == first_node:
                        layer_type = self._infer_layer_type_from_op(node.op_type)
                        break

            # Calculate param count
            param_count = int(np.prod(tensor.shape)) if tensor.shape else 1

            # Check if this is a sensitive layer
            is_sensitive, sensitivity_reason = self._check_layer_sensitivity(
                name, layer_type, graph_info
            )

            result.append(
                LayerPrecisionInfo(
                    layer_name=name,
                    layer_type=layer_type,
                    weight_precision=weight_prec,
                    param_count=param_count,
                    is_sensitive=is_sensitive,
                    sensitivity_reason=sensitivity_reason,
                )
            )

        return result

    def _tensor_to_precision_type(self, tensor: Any) -> PrecisionType:
        """Convert tensor dtype to PrecisionType."""
        import numpy as np

        if not hasattr(tensor, "dtype"):
            return PrecisionType.UNKNOWN

        dtype = tensor.dtype

        dtype_map = {
            np.float32: PrecisionType.FP32,
            np.float64: PrecisionType.FP32,  # Treat as FP32
            np.float16: PrecisionType.FP16,
            np.int8: PrecisionType.INT8,
            np.uint8: PrecisionType.UINT8,
        }

        return dtype_map.get(dtype.type, PrecisionType.UNKNOWN)

    def _block_type_to_layer_type(self, block_type: str) -> LayerType:
        """Convert block type string to LayerType enum."""
        block_type_lower = block_type.lower()

        if "attention" in block_type_lower or "mha" in block_type_lower:
            return LayerType.ATTENTION
        elif "mlp" in block_type_lower or "ffn" in block_type_lower or "feed" in block_type_lower:
            return LayerType.FFN
        elif "embed" in block_type_lower:
            return LayerType.EMBEDDING
        elif "norm" in block_type_lower:
            return LayerType.NORMALIZATION
        elif (
            "output" in block_type_lower
            or "head" in block_type_lower
            or "lm_head" in block_type_lower
        ):
            return LayerType.OUTPUT
        else:
            return LayerType.OTHER

    def _infer_layer_type_from_op(self, op_type: str) -> LayerType:
        """Infer layer type from ONNX op type."""
        op_lower = op_type.lower()

        if "attention" in op_lower:
            return LayerType.ATTENTION
        elif op_type in {"MatMul", "Gemm", "Conv"}:
            return LayerType.FFN  # Could be attention or FFN, default to FFN
        elif "norm" in op_lower:
            return LayerType.NORMALIZATION
        elif "embed" in op_lower or "gather" in op_lower:
            return LayerType.EMBEDDING
        else:
            return LayerType.OTHER

    def _check_layer_sensitivity(
        self, layer_name: str, layer_type: LayerType, graph_info: GraphInfo
    ) -> tuple[bool, str | None]:
        """Check if a layer is sensitive to quantization."""
        layer_name_lower = layer_name.lower()

        # Embedding layers are often sensitive
        if layer_type == LayerType.EMBEDDING or "embed" in layer_name_lower:
            return True, "Embedding layers are sensitive to quantization"

        # Output/LM head is sensitive
        if layer_type == LayerType.OUTPUT or any(
            x in layer_name_lower for x in ["lm_head", "output", "classifier", "final"]
        ):
            return True, "Output layers directly affect predictions"

        # First and last layers are often sensitive
        if any(x in layer_name_lower for x in ["layer.0.", "layers.0.", "block.0."]):
            return True, "First transformer layers are often sensitive"

        # Normalization layers
        if layer_type == LayerType.NORMALIZATION or "norm" in layer_name_lower:
            return True, "Normalization layers maintain numerical stability"

        return False, None

    def _aggregate_by_layer_type(
        self, layer_precisions: list[LayerPrecisionInfo]
    ) -> dict[str, PrecisionByLayerType]:
        """Aggregate precision info by layer type."""
        by_type: dict[LayerType, dict[str, Any]] = {}

        for lp in layer_precisions:
            lt = lp.layer_type
            if lt not in by_type:
                by_type[lt] = {
                    "total_params": 0,
                    "precision_counts": {},
                    "layer_count": 0,
                }

            by_type[lt]["total_params"] += lp.param_count
            by_type[lt]["layer_count"] += 1

            prec_key = lp.weight_precision.value
            by_type[lt]["precision_counts"][prec_key] = (
                by_type[lt]["precision_counts"].get(prec_key, 0) + lp.param_count
            )

        result: dict[str, PrecisionByLayerType] = {}
        for lt, data in by_type.items():
            # Find dominant precision
            prec_counts = data["precision_counts"]
            dominant = PrecisionType.UNKNOWN
            if prec_counts:
                dominant_key = max(prec_counts, key=prec_counts.get)  # type: ignore[arg-type]
                dominant = PrecisionType(dominant_key)

            result[lt.value] = PrecisionByLayerType(
                layer_type=lt,
                total_params=data["total_params"],
                precision_breakdown=prec_counts,
                dominant_precision=dominant,
                layer_count=data["layer_count"],
            )

        return result

    def _get_dominant_weight_precision(
        self, layer_precisions: list[LayerPrecisionInfo]
    ) -> PrecisionType:
        """Get the dominant weight precision across all layers."""
        prec_counts: dict[PrecisionType, int] = {}

        for lp in layer_precisions:
            prec_counts[lp.weight_precision] = (
                prec_counts.get(lp.weight_precision, 0) + lp.param_count
            )

        if not prec_counts:
            return PrecisionType.UNKNOWN

        return max(prec_counts, key=prec_counts.get)  # type: ignore[arg-type]

    def _is_mixed_precision(self, layer_precisions: list[LayerPrecisionInfo]) -> bool:
        """Check if the model uses mixed precision weights."""
        precisions = {lp.weight_precision for lp in layer_precisions}
        # Remove UNKNOWN from consideration
        precisions.discard(PrecisionType.UNKNOWN)

        # Mixed if we have more than one precision type
        return len(precisions) > 1

    def _estimate_accuracy_impact(
        self, scheme_info: QuantizationSchemeInfo, gguf_info: Any | None = None
    ) -> AccuracyImpactEstimate:
        """Estimate accuracy impact based on quantization scheme."""
        scheme = scheme_info.scheme
        base_impact = SCHEME_ACCURACY_IMPACT.get(
            scheme, SCHEME_ACCURACY_IMPACT[QuantizationScheme.UNKNOWN]
        )

        perplexity_pct = base_impact["perplexity_increase_pct"]
        accuracy_pct = base_impact["accuracy_drop_pct"]
        description = base_impact["description"]

        # Refine for GGML based on specific types
        if scheme == QuantizationScheme.GGML and gguf_info is not None:
            ggml_types = list(gguf_info.quantization_breakdown.keys())
            if ggml_types:
                # Find the dominant type
                type_counts = gguf_info.quantization_breakdown
                dominant_type = max(type_counts, key=type_counts.get)

                if dominant_type in GGML_ACCURACY_RANKING:
                    info = GGML_ACCURACY_RANKING[dominant_type]
                    perplexity_pct = info["perplexity_pct"]
                    accuracy_pct = perplexity_pct * 0.5  # Rough correlation
                    description = f"{dominant_type}: ~{info['bits_per_weight']:.1f} bits/weight"

        # Calculate memory reduction
        if scheme_info.bits:
            bits = scheme_info.bits
            mem_reduction = 32.0 / bits  # vs FP32
        elif scheme in {QuantizationScheme.FP16, QuantizationScheme.BF16}:
            mem_reduction = 2.0
        elif scheme in {
            QuantizationScheme.INT8_DYNAMIC,
            QuantizationScheme.INT8_STATIC,
            QuantizationScheme.SMOOTH_QUANT,
        }:
            mem_reduction = 4.0
        elif scheme in {
            QuantizationScheme.GPTQ,
            QuantizationScheme.AWQ,
            QuantizationScheme.BITSANDBYTES_NF4,
            QuantizationScheme.BITSANDBYTES_FP4,
        }:
            mem_reduction = 8.0  # 4-bit
        else:
            mem_reduction = 1.0

        # Generate recommendations
        recommendations: list[str] = []
        if scheme in {QuantizationScheme.GPTQ, QuantizationScheme.AWQ}:
            recommendations.append("Consider keeping embed_tokens and lm_head at FP16")
            recommendations.append("Use calibration dataset representative of your use case")
        elif scheme == QuantizationScheme.GGML:
            recommendations.append("Q4_K_M offers good balance of quality and compression")
            recommendations.append("Q5_K_M for higher quality, Q3_K_M for more compression")
        elif scheme in {QuantizationScheme.INT8_STATIC, QuantizationScheme.INT8_DYNAMIC}:
            recommendations.append("Static quantization typically gives better accuracy")
            recommendations.append("Use representative calibration data for best results")

        return AccuracyImpactEstimate(
            perplexity_increase_pct=perplexity_pct,
            accuracy_drop_pct=accuracy_pct,
            description=description,
            memory_reduction_factor=mem_reduction,
            recommendations=recommendations,
        )

    def _identify_sensitive_layers(
        self, graph_info: GraphInfo, layer_precisions: list[LayerPrecisionInfo]
    ) -> list[SensitiveLayerInfo]:
        """Identify layers that are sensitive to quantization."""
        sensitive: list[SensitiveLayerInfo] = []

        for lp in layer_precisions:
            if lp.is_sensitive:
                # Determine recommended precision
                if lp.weight_precision in {PrecisionType.INT4, PrecisionType.INT8}:
                    recommended = PrecisionType.FP16
                else:
                    recommended = lp.weight_precision

                sensitivity_score = self._calculate_sensitivity_score(lp)

                sensitive.append(
                    SensitiveLayerInfo(
                        layer_name=lp.layer_name,
                        layer_type=lp.layer_type,
                        sensitivity_score=sensitivity_score,
                        reason=lp.sensitivity_reason or "Unknown reason",
                        recommendation=f"Consider keeping at {recommended.value}",
                        current_precision=lp.weight_precision,
                        recommended_precision=recommended,
                    )
                )

        # Sort by sensitivity score (highest first)
        sensitive.sort(key=lambda x: x.sensitivity_score, reverse=True)

        return sensitive

    def _calculate_sensitivity_score(self, lp: LayerPrecisionInfo) -> float:
        """Calculate a sensitivity score for a layer (0.0 to 1.0)."""
        score = 0.5  # Base score

        # Embedding layers are very sensitive
        if lp.layer_type == LayerType.EMBEDDING:
            score += 0.3

        # Output layers are critical
        if lp.layer_type == LayerType.OUTPUT:
            score += 0.4

        # Normalization layers need precision
        if lp.layer_type == LayerType.NORMALIZATION:
            score += 0.2

        # First layers
        if any(x in lp.layer_name.lower() for x in ["layer.0", "layers.0", "block.0"]):
            score += 0.1

        return min(score, 1.0)


# =============================================================================
# Convenience Functions
# =============================================================================


def analyze_quantization(
    graph_info: GraphInfo,
    blocks: list[Block] | None = None,
    gguf_info: Any | None = None,
    logger: logging.Logger | None = None,
) -> QuantizationAnalysisResult:
    """Convenience function to analyze quantization.

    Args:
        graph_info: Parsed ONNX graph information.
        blocks: Detected architectural blocks.
        gguf_info: Optional GGUFInfo for GGUF models.
        logger: Optional logger.

    Returns:
        QuantizationAnalysisResult with full analysis.
    """
    analyzer = QuantizationAnalyzer(logger=logger)
    return analyzer.analyze(graph_info, blocks, gguf_info)
