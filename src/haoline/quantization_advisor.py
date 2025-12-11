# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
Quantization Advisor module for HaoLine.

Provides LLM-powered quantization recommendations based on lint results.
Generates:
- Architecture-specific quantization strategies (CNN/Transformer/Hybrid)
- Deployment-target-aware recommendations (TensorRT/ORT/TFLite)
- Step-by-step QAT workflow
- Accuracy loss estimates and mitigation strategies

Usage:
    advisor = QuantizationAdvisor()  # Uses OPENAI_API_KEY env var
    advice = advisor.advise(lint_result, graph_info)
    print(advice.strategy)
    print(advice.qat_workflow)
"""

from __future__ import annotations

import json
import logging
import os
from enum import Enum
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from .analyzer import GraphInfo
    from .quantization_linter import QuantizationLintResult

# Check for OpenAI availability
_OPENAI_AVAILABLE = False
try:
    import openai
    from openai import OpenAI

    _OPENAI_AVAILABLE = True
except ImportError:
    openai = None  # type: ignore
    OpenAI = None  # type: ignore

logger = logging.getLogger(__name__)


class ArchitectureType(Enum):
    """Model architecture classification for quantization strategy."""

    CNN = "cnn"
    TRANSFORMER = "transformer"
    RNN = "rnn"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


class DeploymentRuntime(Enum):
    """Target deployment runtime for quantization."""

    TENSORRT = "tensorrt"
    ONNX_RUNTIME = "onnxruntime"
    TFLITE = "tflite"
    OPENVINO = "openvino"
    GENERIC = "generic"


class FakeQuantInsertionPoint(BaseModel):
    """Recommended fake-quantization insertion point for QAT."""

    model_config = ConfigDict(frozen=True)

    node_name: str
    op_type: str
    position: str  # "before_input", "after_output", "weights"
    reason: str
    priority: str  # "required", "recommended", "optional"


class OpSubstitution(BaseModel):
    """Recommended operator substitution for better INT8 performance."""

    model_config = ConfigDict(frozen=True)

    original_op: str
    replacement_op: str
    affected_layers: list[str]
    reason: str
    accuracy_impact: str  # "none", "minimal", "moderate"


class QuantGranularityRec(BaseModel):
    """Per-channel vs per-tensor quantization recommendation."""

    model_config = ConfigDict(frozen=True)

    layer_name: str
    op_type: str
    recommendation: str  # "per_channel", "per_tensor"
    reason: str


class QuantizationAdvice(BaseModel):
    """Container for LLM-generated quantization recommendations."""

    model_config = ConfigDict(frozen=True)

    # Architecture analysis
    architecture_type: ArchitectureType
    architecture_summary: str

    # Quantization strategy
    strategy: str  # Overall strategy recommendation
    sensitive_layers: list[str]  # Layers to keep at FP16
    safe_layers: list[str]  # Layers safe to quantize to INT8

    # QAT workflow
    qat_workflow: list[str]  # Step-by-step QAT instructions
    calibration_tips: str  # Calibration dataset guidance

    # Deployment-specific
    runtime_recommendations: dict[str, str]  # Per-runtime advice

    # Accuracy estimation
    expected_accuracy_impact: str  # e.g., "Low (<1% degradation)"
    mitigation_strategies: list[str]  # Ways to preserve accuracy

    # NEW: Static recommendations (Tasks 33.4.2-4)
    fake_quant_insertions: list[FakeQuantInsertionPoint] = Field(default_factory=list)
    op_substitutions: list[OpSubstitution] = Field(default_factory=list)
    granularity_recommendations: list[QuantGranularityRec] = Field(default_factory=list)

    # Metadata
    model_used: str = ""
    tokens_used: int = 0
    success: bool = True
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "architecture_type": self.architecture_type.value,
            "architecture_summary": self.architecture_summary,
            "strategy": self.strategy,
            "sensitive_layers": self.sensitive_layers,
            "safe_layers": self.safe_layers,
            "qat_workflow": self.qat_workflow,
            "calibration_tips": self.calibration_tips,
            "runtime_recommendations": self.runtime_recommendations,
            "expected_accuracy_impact": self.expected_accuracy_impact,
            "fake_quant_insertions": [
                {
                    "node_name": fq.node_name,
                    "op_type": fq.op_type,
                    "position": fq.position,
                    "reason": fq.reason,
                    "priority": fq.priority,
                }
                for fq in self.fake_quant_insertions
            ],
            "op_substitutions": [
                {
                    "original_op": sub.original_op,
                    "replacement_op": sub.replacement_op,
                    "affected_layers": sub.affected_layers,
                    "reason": sub.reason,
                    "accuracy_impact": sub.accuracy_impact,
                }
                for sub in self.op_substitutions
            ],
            "granularity_recommendations": [
                {
                    "layer_name": gr.layer_name,
                    "op_type": gr.op_type,
                    "recommendation": gr.recommendation,
                    "reason": gr.reason,
                }
                for gr in self.granularity_recommendations
            ],
            "mitigation_strategies": self.mitigation_strategies,
            "model_used": self.model_used,
            "tokens_used": self.tokens_used,
            "success": self.success,
            "error_message": self.error_message,
        }


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

SYSTEM_PROMPT = """You are an expert ML quantization engineer specializing in INT8 deployment.
You analyze ONNX model structures and quantization lint results to provide actionable recommendations.

Your expertise covers:
- Post-Training Quantization (PTQ) vs Quantization-Aware Training (QAT)
- Per-tensor vs per-channel quantization tradeoffs
- Runtime-specific optimizations (TensorRT, ONNX Runtime, TFLite, OpenVINO)
- Architecture-specific strategies (CNNs, Transformers, RNNs, hybrid models)
- Calibration dataset selection and size
- Accuracy preservation techniques

Always provide practical, actionable advice. Be specific about layer names when recommending FP16 fallback."""

ARCHITECTURE_PROMPT = """Analyze this model's architecture for quantization planning:

Model Name: {model_name}
Total Nodes: {total_nodes}
Op Type Distribution: {op_distribution}
Readiness Score: {readiness_score}/100

Determine:
1. Architecture type (CNN, Transformer, RNN, or Hybrid)
2. Key structural patterns (residual connections, attention blocks, etc.)
3. Critical paths that affect quantization strategy

Respond in JSON format:
{{
    "architecture_type": "cnn|transformer|rnn|hybrid",
    "summary": "One paragraph describing the architecture",
    "patterns": ["list", "of", "detected", "patterns"]
}}"""

STRATEGY_PROMPT = """Based on this quantization analysis, provide a comprehensive quantization strategy:

Architecture: {architecture_type}
Readiness Score: {readiness_score}/100
Critical Issues: {critical_count}
High-Risk Layers: {high_risk_layers}

Quantization Lint Results:
- Quant-friendly ops: {quant_friendly_pct}%
- Accuracy-sensitive ops: {accuracy_sensitive_count}
- No INT8 kernel ops: {no_int8_count}
- Dynamic shapes detected: {has_dynamic_shapes}
- Custom ops: {custom_ops}

Top Risk Layers (by score):
{risk_layers_detail}

Provide recommendations in JSON format:
{{
    "strategy": "Overall quantization approach (2-3 sentences)",
    "sensitive_layers": ["layer_names", "to_keep_at_fp16"],
    "safe_layers": ["layer_names", "safe_for_int8"],
    "qat_workflow": [
        "Step 1: ...",
        "Step 2: ...",
        "Step 3: ..."
    ],
    "calibration_tips": "Guidance on calibration dataset",
    "expected_accuracy_impact": "Low/Medium/High with explanation",
    "mitigation_strategies": ["strategy1", "strategy2"]
}}"""

RUNTIME_PROMPT = """Provide runtime-specific quantization recommendations for this model:

Architecture: {architecture_type}
Readiness Score: {readiness_score}/100
Key Issues: {key_issues}

For each runtime, explain:
- Specific settings/flags to use
- Known limitations with this model type
- Optimization tips

Respond in JSON format:
{{
    "tensorrt": "TensorRT-specific recommendation",
    "onnxruntime": "ONNX Runtime-specific recommendation",
    "tflite": "TFLite-specific recommendation",
    "openvino": "OpenVINO-specific recommendation"
}}"""


# =============================================================================
# NON-LLM FALLBACK STRATEGIES
# =============================================================================

_ARCHITECTURE_PATTERNS = {
    "transformer": ["Attention", "MultiHeadAttention", "LayerNormalization", "Softmax"],
    "cnn": ["Conv", "MaxPool", "AveragePool", "BatchNormalization"],
    "rnn": ["LSTM", "GRU", "RNN"],
}


def _detect_architecture(op_counts: dict[str, int]) -> ArchitectureType:
    """Detect architecture type from op counts (non-LLM fallback)."""
    transformer_score = sum(op_counts.get(op, 0) for op in _ARCHITECTURE_PATTERNS["transformer"])
    cnn_score = sum(op_counts.get(op, 0) for op in _ARCHITECTURE_PATTERNS["cnn"])
    rnn_score = sum(op_counts.get(op, 0) for op in _ARCHITECTURE_PATTERNS["rnn"])

    if transformer_score > 5 and cnn_score > 5:
        return ArchitectureType.HYBRID
    if transformer_score > cnn_score and transformer_score > rnn_score:
        return ArchitectureType.TRANSFORMER
    if rnn_score > cnn_score:
        return ArchitectureType.RNN
    if cnn_score > 0:
        return ArchitectureType.CNN
    return ArchitectureType.UNKNOWN


def _get_op_counts(lint_result: QuantizationLintResult) -> dict[str, int]:
    """Combine op counts from lint result for architecture detection."""
    op_counts: dict[str, int] = {}
    # Merge all op type dictionaries
    for ops_dict in [
        lint_result.unsupported_ops,
        lint_result.accuracy_sensitive_ops,
        lint_result.quant_friendly_ops,
        lint_result.quantization_ops,
    ]:
        for op, count in ops_dict.items():
            op_counts[op] = op_counts.get(op, 0) + count
    return op_counts


# =============================================================================
# STATIC RECOMMENDATION GENERATORS (Tasks 33.4.2-4)
# =============================================================================

# Ops that benefit from fake-quant nodes for QAT
_QUANT_TARGET_OPS = {"Conv", "MatMul", "Gemm", "ConvTranspose"}

# Ops with known INT8-friendly alternatives
_OP_SUBSTITUTIONS = {
    "LayerNormalization": ("RMSNorm", "RMSNorm has simpler computation, better INT8 perf"),
    "Softmax": ("ScaledSoftmax", "Pre-scaled softmax avoids large dynamic range"),
    "GELU": ("GELUApprox", "Approximate GELU is more quantization-friendly"),
    "Mish": ("ReLU", "ReLU quantizes perfectly; Mish has numerical sensitivity"),
    "SiLU": ("ReLU", "ReLU quantizes perfectly; SiLU has numerical sensitivity"),
    "HardSwish": ("ReLU6", "ReLU6 is bounded and quantization-friendly"),
}

# Ops that benefit from per-channel quantization
_PER_CHANNEL_OPS = {"Conv", "ConvTranspose", "Gemm", "MatMul"}


def _generate_fake_quant_insertions(
    graph_info: GraphInfo,
    lint_result: QuantizationLintResult,
) -> list[FakeQuantInsertionPoint]:
    """Identify where fake-quant nodes should be inserted for QAT (Task 33.4.2)."""
    insertions = []

    for node in graph_info.nodes:
        if node.op_type in _QUANT_TARGET_OPS:
            # Weight quantization (required for all quant-target ops)
            insertions.append(
                FakeQuantInsertionPoint(
                    node_name=node.name,
                    op_type=node.op_type,
                    position="weights",
                    reason=f"{node.op_type} weights should be quantized for INT8",
                    priority="required",
                )
            )

            # Input activation quantization
            insertions.append(
                FakeQuantInsertionPoint(
                    node_name=node.name,
                    op_type=node.op_type,
                    position="before_input",
                    reason="Input activations need fake-quant for proper range learning",
                    priority="required",
                )
            )

        # Output quantization for accuracy-sensitive ops
        if node.op_type in lint_result.accuracy_sensitive_ops:
            insertions.append(
                FakeQuantInsertionPoint(
                    node_name=node.name,
                    op_type=node.op_type,
                    position="after_output",
                    reason="Accuracy-sensitive op benefits from output fake-quant",
                    priority="recommended",
                )
            )

    return insertions


def _generate_op_substitutions(
    graph_info: GraphInfo,
) -> list[OpSubstitution]:
    """Recommend operator substitutions for better INT8 performance (Task 33.4.3)."""
    substitutions = []
    op_to_layers: dict[str, list[str]] = {}

    # Group layers by op type
    for node in graph_info.nodes:
        if node.op_type in _OP_SUBSTITUTIONS:
            if node.op_type not in op_to_layers:
                op_to_layers[node.op_type] = []
            op_to_layers[node.op_type].append(node.name)

    # Generate substitution recommendations
    for op_type, layers in op_to_layers.items():
        replacement, reason = _OP_SUBSTITUTIONS[op_type]
        accuracy_impact = "minimal" if op_type in ("GELU", "Mish", "SiLU") else "moderate"

        substitutions.append(
            OpSubstitution(
                original_op=op_type,
                replacement_op=replacement,
                affected_layers=layers,
                reason=reason,
                accuracy_impact=accuracy_impact,
            )
        )

    return substitutions


def _generate_granularity_recommendations(
    graph_info: GraphInfo,
    arch_type: ArchitectureType,
) -> list[QuantGranularityRec]:
    """Recommend per-channel vs per-tensor quantization (Task 33.4.4)."""
    recommendations = []

    for node in graph_info.nodes:
        if node.op_type not in _PER_CHANNEL_OPS:
            continue

        # Default to per-channel for weights
        if node.op_type in ("Conv", "ConvTranspose"):
            recommendations.append(
                QuantGranularityRec(
                    layer_name=node.name,
                    op_type=node.op_type,
                    recommendation="per_channel",
                    reason="Conv weights benefit from per-channel (per-output-channel) quantization",
                )
            )
        elif node.op_type in ("Gemm", "MatMul"):
            # Transformers benefit from per-channel; CNNs can use per-tensor for FC
            if arch_type == ArchitectureType.TRANSFORMER:
                recommendations.append(
                    QuantGranularityRec(
                        layer_name=node.name,
                        op_type=node.op_type,
                        recommendation="per_channel",
                        reason="Transformer MatMul weights need per-channel for accuracy",
                    )
                )
            else:
                # Check if this is likely a classifier (late in graph)
                recommendations.append(
                    QuantGranularityRec(
                        layer_name=node.name,
                        op_type=node.op_type,
                        recommendation="per_channel",
                        reason="Per-channel recommended for weight quantization",
                    )
                )

    return recommendations


def _generate_fallback_advice(
    lint_result: QuantizationLintResult,
    graph_info: GraphInfo,
) -> QuantizationAdvice:
    """Generate non-LLM quantization advice based on heuristics."""
    # Detect architecture
    op_counts = _get_op_counts(lint_result)
    arch_type = _detect_architecture(op_counts)

    # Identify sensitive layers (from risk scores)
    sensitive = [
        lr.name for lr in lint_result.layer_risk_scores if lr.risk_level in ("critical", "high")
    ][:10]

    # Identify safe layers
    safe = [lr.name for lr in lint_result.layer_risk_scores if lr.risk_level == "low"][:10]

    # Architecture-specific strategy
    if arch_type == ArchitectureType.TRANSFORMER:
        strategy = (
            "Transformer models require careful attention to LayerNorm and Softmax ops. "
            "Use per-channel quantization for MatMul weights. Keep attention score "
            "computation at FP16 for accuracy. QAT is recommended for <1% accuracy loss."
        )
        calibration = (
            "Use 100-500 representative sequences from your training data. "
            "Include varied sequence lengths for robust calibration."
        )
    elif arch_type == ArchitectureType.CNN:
        strategy = (
            "CNN models typically quantize well with PTQ. Focus on final classifier "
            "layers which are most accuracy-sensitive. Early conv layers are usually safe. "
            "Use per-channel quantization for Conv weights."
        )
        calibration = (
            "Use 500-1000 representative images from your validation set. "
            "Ensure diverse examples covering all classes."
        )
    elif arch_type == ArchitectureType.RNN:
        strategy = (
            "RNN/LSTM models are challenging for INT8. Consider keeping recurrent cells "
            "at FP16 and only quantizing input projections. Gate computations are sensitive. "
            "QAT is strongly recommended."
        )
        calibration = (
            "Use varied-length sequences from your training data. "
            "Include both short and long sequences for calibration."
        )
    else:
        strategy = (
            "Mixed or unknown architecture detected. Start with conservative PTQ using "
            "per-channel quantization. Monitor accuracy closely and fall back to FP16 "
            "for any problematic layers."
        )
        calibration = (
            "Use 500+ representative samples from your validation data. "
            "Ensure coverage of typical input patterns."
        )

    # QAT workflow
    qat_workflow = [
        "Step 1: Train your model to convergence in FP32",
        "Step 2: Insert fake-quantization nodes using your framework's QAT tools",
        "Step 3: Fine-tune for 10-20% of original training epochs with lower LR (0.1x)",
        "Step 4: Export to ONNX with fake-quant nodes preserved",
        "Step 5: Convert to INT8 using target runtime's quantization tools",
        "Step 6: Validate accuracy on held-out test set",
    ]

    # Runtime recommendations
    runtime_recs = {
        "tensorrt": (
            "Use trtexec with --int8 and --fp16 flags. Set --builderOptimizationLevel=5 "
            "for best optimization. Use layer precision overrides for sensitive ops."
        ),
        "onnxruntime": (
            "Use onnxruntime.quantization with CalibrationMethod.MinMax or Entropy. "
            "Enable per_channel=True for Conv/MatMul. Use QDQ format for best compatibility."
        ),
        "tflite": (
            "Use TFLiteConverter with optimizations=[tf.lite.Optimize.DEFAULT]. "
            "Set representative_dataset for full INT8. Consider MLIR-based conversion."
        ),
        "openvino": (
            "Use POT (Post-Training Optimization Tool) with DefaultQuantization. "
            "Set stat_subset_size=300 minimum. Use AccuracyAwareQuantization if needed."
        ),
    }

    # Accuracy impact estimation
    if lint_result.readiness_score >= 80:
        accuracy_impact = "Low (<1% degradation expected with proper calibration)"
    elif lint_result.readiness_score >= 60:
        accuracy_impact = "Medium (1-3% degradation likely, QAT recommended)"
    else:
        accuracy_impact = "High (>3% degradation possible, QAT strongly recommended)"

    # Mitigation strategies
    mitigations = [
        "Use per-channel quantization for weight tensors",
        "Keep final classifier/output layers at FP16",
        "Increase calibration dataset size for better range estimation",
        "Apply QAT fine-tuning if PTQ accuracy is insufficient",
    ]

    critical_count = sum(1 for w in lint_result.warnings if w.severity.value == "critical")
    if critical_count > 0:
        mitigations.insert(0, "Keep critical ops at FP16 (see sensitive_layers)")

    # Generate static recommendations (Tasks 33.4.2-4)
    fake_quant_insertions = _generate_fake_quant_insertions(graph_info, lint_result)
    op_substitutions = _generate_op_substitutions(graph_info)
    granularity_recs = _generate_granularity_recommendations(graph_info, arch_type)

    return QuantizationAdvice(
        architecture_type=arch_type,
        architecture_summary=f"Detected {arch_type.value} architecture with {len(graph_info.nodes)} nodes",
        strategy=strategy,
        sensitive_layers=sensitive,
        safe_layers=safe,
        qat_workflow=qat_workflow,
        calibration_tips=calibration,
        runtime_recommendations=runtime_recs,
        expected_accuracy_impact=accuracy_impact,
        mitigation_strategies=mitigations,
        fake_quant_insertions=fake_quant_insertions,
        op_substitutions=op_substitutions,
        granularity_recommendations=granularity_recs,
        success=True,
    )


def _extract_string_from_nested(value: Any) -> str:
    """
    Recursively extract a readable string from arbitrarily nested LLM output.

    Handles cases like:
    - "simple string" -> "simple string"
    - {"recommendation": "text"} -> "text"
    - {"recommendation": {"description": "text"}} -> "text"
    - {"settings": "a", "notes": "b"} -> "a. b"
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        # Priority keys to look for
        for key in ("recommendation", "description", "text", "value", "summary"):
            if key in value:
                return _extract_string_from_nested(value[key])
        # Fallback: concatenate all string values
        parts = []
        for v in value.values():
            extracted = _extract_string_from_nested(v)
            if extracted:
                parts.append(extracted)
        return ". ".join(parts) if parts else ""
    if isinstance(value, list):
        # Join list items
        return ", ".join(_extract_string_from_nested(item) for item in value if item)
    # Fallback for other types
    return str(value) if value is not None else ""


def _normalize_str_list(value: Any) -> list[str]:
    """
    Coerce LLM/fallback outputs into list[str].

    Handles various LLM response formats:
    - ["a", "b", "c"] -> ["a", "b", "c"]
    - {"layer_names": ["a", "b"]} -> ["a", "b"]
    - [{"name": "a"}, {"name": "b"}] -> ["a", "b"]
    - "single item" -> ["single item"]
    """
    if value is None:
        return []

    # Case 1: Already a list
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                # Try to extract layer name from dict
                if "layer_names" in item and isinstance(item["layer_names"], list):
                    out.extend(str(x) for x in item["layer_names"])
                elif "name" in item:
                    out.append(str(item["name"]))
                elif "layer" in item:
                    out.append(str(item["layer"]))
                else:
                    # Last resort: stringify, but try to extract meaningful content
                    extracted = _extract_string_from_nested(item)
                    if extracted:
                        out.append(extracted)
            else:
                out.append(str(item))
        return out

    # Case 2: Dict with layer_names key
    if isinstance(value, dict):
        if "layer_names" in value and isinstance(value["layer_names"], list):
            return [str(x) for x in value["layer_names"]]
        if "layers" in value and isinstance(value["layers"], list):
            return [str(x) for x in value["layers"]]
        # Try to extract any list values
        for v in value.values():
            if isinstance(v, list):
                return _normalize_str_list(v)

    # Case 3: Single string
    if isinstance(value, str):
        return [value]

    return [str(value)] if value else []


def _normalize_runtime_recs(runtime_data: Any) -> dict[str, str]:
    """
    Coerce runtime recommendations into dict[str, str].

    Handles various LLM response formats:
    - {"tensorrt": "Use INT8..."} -> {"tensorrt": "Use INT8..."}
    - {"tensorrt": {"recommendation": "Use INT8..."}} -> {"tensorrt": "Use INT8..."}
    - {"tensorrt": {"recommendation": {"settings": "...", "notes": "..."}}}
      -> {"tensorrt": ".... ..."}
    """
    if not isinstance(runtime_data, dict):
        return {}

    normalized: dict[str, str] = {}
    for key, val in runtime_data.items():
        if isinstance(val, str):
            normalized[key] = val
        elif isinstance(val, dict):
            # Recursively extract a string from nested structure
            normalized[key] = _extract_string_from_nested(val)
        else:
            normalized[key] = str(val) if val is not None else ""
    return normalized


# =============================================================================
# MAIN ADVISOR CLASS
# =============================================================================


class QuantizationAdvisor:
    """
    LLM-powered quantization advisor.

    Uses OpenAI API to generate contextual quantization recommendations.
    Falls back to heuristic-based advice if LLM is unavailable.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        use_llm: bool = True,
    ) -> None:
        """
        Initialize the advisor.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use for generation
            use_llm: Whether to use LLM (False = heuristic-only mode)
        """
        self.model = model
        self.use_llm = use_llm and _OPENAI_AVAILABLE
        self.client: OpenAI | None = None

        if self.use_llm:
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if key and OpenAI is not None:
                self.client = OpenAI(api_key=key)
            else:
                self.use_llm = False

    def advise(
        self,
        lint_result: QuantizationLintResult,
        graph_info: GraphInfo,
        target_runtime: DeploymentRuntime = DeploymentRuntime.GENERIC,
    ) -> QuantizationAdvice:
        """
        Generate quantization recommendations.

        Args:
            lint_result: Results from QuantizationLinter
            graph_info: Graph structure info
            target_runtime: Target deployment runtime

        Returns:
            QuantizationAdvice with recommendations
        """
        if not self.use_llm or self.client is None:
            return _generate_fallback_advice(lint_result, graph_info)

        try:
            return self._generate_llm_advice(lint_result, graph_info, target_runtime)
        except Exception as e:
            logger.warning(f"LLM advice generation failed: {e}, using fallback")
            advice = _generate_fallback_advice(lint_result, graph_info)
            # Use model_copy since QuantizationAdvice is frozen
            return cast(QuantizationAdvice, advice.model_copy(update={"error_message": str(e)}))

    def _generate_llm_advice(
        self,
        lint_result: QuantizationLintResult,
        graph_info: GraphInfo,
        target_runtime: DeploymentRuntime,
    ) -> QuantizationAdvice:
        """Generate advice using LLM."""
        total_tokens = 0

        # Step 1: Architecture analysis
        op_counts = _get_op_counts(lint_result)
        arch_prompt = ARCHITECTURE_PROMPT.format(
            model_name=graph_info.name,
            total_nodes=len(graph_info.nodes),
            op_distribution=json.dumps(op_counts),
            readiness_score=lint_result.readiness_score,
        )

        arch_response = self._call_llm(arch_prompt)
        total_tokens += arch_response.get("tokens", 0)
        arch_data = self._parse_json_response(arch_response.get("content", "{}"))

        arch_type = ArchitectureType(arch_data.get("architecture_type", "unknown"))
        arch_summary = arch_data.get("summary", "Architecture analysis unavailable")

        # Step 2: Strategy generation
        risk_detail = "\n".join(
            f"- {lr.name} ({lr.op_type}): {lr.risk_score}/100 - {lr.reason}"
            for lr in lint_result.layer_risk_scores[:10]
        )

        # Count issues
        accuracy_sensitive_count = sum(lint_result.accuracy_sensitive_ops.values())
        no_int8_count = sum(lint_result.unsupported_ops.values())
        has_dynamic_shapes = len(lint_result.dynamic_shape_nodes) > 0
        critical_count = sum(1 for w in lint_result.warnings if w.severity.value == "critical")

        strategy_prompt = STRATEGY_PROMPT.format(
            architecture_type=arch_type.value,
            readiness_score=lint_result.readiness_score,
            critical_count=critical_count,
            high_risk_layers=len(
                [lr for lr in lint_result.layer_risk_scores if lr.risk_level == "high"]
            ),
            quant_friendly_pct=lint_result.quant_friendly_pct,
            accuracy_sensitive_count=accuracy_sensitive_count,
            no_int8_count=no_int8_count,
            has_dynamic_shapes=has_dynamic_shapes,
            custom_ops=lint_result.custom_ops,
            risk_layers_detail=risk_detail,
        )

        strategy_response = self._call_llm(strategy_prompt)
        total_tokens += strategy_response.get("tokens", 0)
        strategy_data = self._parse_json_response(strategy_response.get("content", "{}"))

        # Step 3: Runtime-specific recommendations
        runtime_prompt = RUNTIME_PROMPT.format(
            architecture_type=arch_type.value,
            readiness_score=lint_result.readiness_score,
            key_issues=", ".join(lint_result.custom_ops) if lint_result.custom_ops else "None",
        )

        runtime_response = self._call_llm(runtime_prompt)
        total_tokens += runtime_response.get("tokens", 0)
        runtime_data = self._parse_json_response(runtime_response.get("content", "{}"))

        # Generate static recommendations (always available, even with LLM)
        fake_quant_insertions = _generate_fake_quant_insertions(graph_info, lint_result)
        op_substitutions = _generate_op_substitutions(graph_info)
        granularity_recs = _generate_granularity_recommendations(graph_info, arch_type)

        return QuantizationAdvice(
            architecture_type=arch_type,
            architecture_summary=arch_summary,
            strategy=strategy_data.get("strategy", "No strategy available"),
            sensitive_layers=_normalize_str_list(strategy_data.get("sensitive_layers", [])),
            safe_layers=_normalize_str_list(strategy_data.get("safe_layers", [])),
            qat_workflow=_normalize_str_list(strategy_data.get("qat_workflow", [])),
            calibration_tips=strategy_data.get("calibration_tips", ""),
            runtime_recommendations=_normalize_runtime_recs(runtime_data),
            expected_accuracy_impact=strategy_data.get("expected_accuracy_impact", "Unknown"),
            mitigation_strategies=_normalize_str_list(
                strategy_data.get("mitigation_strategies", [])
            ),
            fake_quant_insertions=fake_quant_insertions,
            op_substitutions=op_substitutions,
            granularity_recommendations=granularity_recs,
            model_used=self.model,
            tokens_used=total_tokens,
            success=True,
        )

    def _call_llm(self, prompt: str) -> dict[str, Any]:
        """Make a single LLM API call."""
        if self.client is None:
            return {"content": "{}", "tokens": 0}

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1000,
        )

        content = response.choices[0].message.content or "{}"
        tokens = response.usage.total_tokens if response.usage else 0

        return {"content": content, "tokens": tokens}

    def _parse_json_response(self, content: str) -> dict[str, Any]:
        """Parse JSON from LLM response, handling markdown code blocks."""
        # Strip markdown code fences if present
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first and last lines (```json and ```)
            content = "\n".join(lines[1:-1])

        try:
            result = json.loads(content)
            return result if isinstance(result, dict) else {}
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM JSON response: {content[:100]}")
            return {}


def advise_quantization(
    lint_result: QuantizationLintResult,
    graph_info: GraphInfo,
    api_key: str | None = None,
    use_llm: bool = True,
) -> QuantizationAdvice:
    """
    Convenience function for generating quantization advice.

    Args:
        lint_result: Results from QuantizationLinter
        graph_info: Graph structure info
        api_key: Optional OpenAI API key
        use_llm: Whether to use LLM (False = heuristic-only)

    Returns:
        QuantizationAdvice with recommendations
    """
    advisor = QuantizationAdvisor(api_key=api_key, use_llm=use_llm)
    return advisor.advise(lint_result, graph_info)


# =============================================================================
# REPORT GENERATION (Task 33.4.10)
# =============================================================================


def generate_qat_readiness_report(
    lint_result: QuantizationLintResult,
    advice: QuantizationAdvice,
    model_name: str = "Model",
    format: str = "markdown",
) -> str:
    """
    Generate a comprehensive QAT Readiness Report.

    Args:
        lint_result: Results from QuantizationLinter
        advice: Results from QuantizationAdvisor
        model_name: Name of the model for the report title
        format: "markdown" or "html"

    Returns:
        Formatted report string
    """
    if format == "html":
        return _generate_html_report(lint_result, advice, model_name)
    return _generate_markdown_report(lint_result, advice, model_name)


def _generate_markdown_report(
    lint_result: QuantizationLintResult,
    advice: QuantizationAdvice,
    model_name: str,
) -> str:
    """Generate Markdown QAT readiness report."""
    # Grade based on score
    score = lint_result.readiness_score
    if score >= 90:
        grade, grade_desc = "A", "Excellent - PTQ should work well"
    elif score >= 80:
        grade, grade_desc = "B", "Good - PTQ likely to work with minor issues"
    elif score >= 70:
        grade, grade_desc = "C", "Fair - Consider QAT for best results"
    elif score >= 60:
        grade, grade_desc = "D", "Poor - QAT recommended"
    else:
        grade, grade_desc = "F", "Critical issues - Significant work needed"

    lines = [
        f"# QAT Readiness Report: {model_name}",
        "",
        "## Executive Summary",
        "",
        f"**Readiness Score:** {score}/100 (Grade: {grade})",
        f"**Assessment:** {grade_desc}",
        f"**Architecture:** {advice.architecture_type.value.upper()}",
        "",
        f"> {advice.strategy}",
        "",
        "---",
        "",
        "## Quantization Analysis",
        "",
        "### Overview",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Ops | {lint_result.total_ops} |",
        f"| Quant-Friendly | {lint_result.quant_friendly_pct:.1f}% |",
        f"| Critical Issues | {sum(1 for w in lint_result.warnings if w.severity.value == 'critical')} |",
        f"| High-Risk Layers | {len([lr for lr in lint_result.layer_risk_scores if lr.risk_level == 'high'])} |",
        "",
    ]

    # High-risk layers
    high_risk = [
        lr for lr in lint_result.layer_risk_scores if lr.risk_level in ("critical", "high")
    ]
    if high_risk:
        lines.extend(
            [
                "### High-Risk Layers",
                "",
                "| Layer | Op Type | Risk Score | Reason |",
                "|-------|---------|------------|--------|",
            ]
        )
        for lr in high_risk[:10]:
            lines.append(f"| {lr.name} | {lr.op_type} | {lr.risk_score}/100 | {lr.reason} |")
        lines.append("")

    # Op substitutions
    if advice.op_substitutions:
        lines.extend(
            [
                "### Recommended Op Substitutions",
                "",
                "| Original | Replacement | Impact | Reason |",
                "|----------|-------------|--------|--------|",
            ]
        )
        for sub in advice.op_substitutions:
            lines.append(
                f"| {sub.original_op} | {sub.replacement_op} | {sub.accuracy_impact} | {sub.reason} |"
            )
        lines.append("")

    # Granularity recommendations
    if advice.granularity_recommendations:
        lines.extend(
            [
                "### Quantization Granularity",
                "",
                "| Layer | Recommendation | Reason |",
                "|-------|----------------|--------|",
            ]
        )
        for gr in advice.granularity_recommendations[:10]:
            lines.append(f"| {gr.layer_name} | {gr.recommendation} | {gr.reason[:50]}... |")
        lines.append("")

    # QAT workflow
    lines.extend(
        [
            "---",
            "",
            "## QAT Workflow",
            "",
        ]
    )
    for i, step in enumerate(advice.qat_workflow, 1):
        lines.append(f"{i}. {step}")
    lines.append("")

    # Calibration tips
    lines.extend(
        [
            "### Calibration Guidelines",
            "",
            f"> {advice.calibration_tips}",
            "",
        ]
    )

    # Runtime recommendations
    lines.extend(
        [
            "---",
            "",
            "## Runtime-Specific Recommendations",
            "",
        ]
    )
    for runtime, rec in advice.runtime_recommendations.items():
        lines.extend(
            [
                f"### {runtime.upper()}",
                "",
                rec,
                "",
            ]
        )

    # Fake-quant insertions summary
    if advice.fake_quant_insertions:
        required = [fq for fq in advice.fake_quant_insertions if fq.priority == "required"]
        recommended = [fq for fq in advice.fake_quant_insertions if fq.priority == "recommended"]
        lines.extend(
            [
                "---",
                "",
                "## Fake-Quantization Insertion Points",
                "",
                f"**Required:** {len(required)} insertion points",
                f"**Recommended:** {len(recommended)} insertion points",
                "",
            ]
        )

    # Accuracy impact
    lines.extend(
        [
            "---",
            "",
            "## Expected Accuracy Impact",
            "",
            f"**Estimate:** {advice.expected_accuracy_impact}",
            "",
            "### Mitigation Strategies",
            "",
        ]
    )
    for strategy in advice.mitigation_strategies:
        lines.append(f"- {strategy}")
    lines.append("")

    # Footer
    lines.extend(
        [
            "---",
            "",
            "*Report generated by HaoLine Quantization Advisor*",
        ]
    )
    if advice.model_used:
        lines.append(f"*LLM: {advice.model_used} ({advice.tokens_used} tokens)*")

    return "\n".join(lines)


def _generate_html_report(
    lint_result: QuantizationLintResult,
    advice: QuantizationAdvice,
    model_name: str,
) -> str:
    """Generate HTML QAT readiness report."""
    # Convert markdown to basic HTML
    md_report = _generate_markdown_report(lint_result, advice, model_name)

    # Simple markdown-to-HTML conversion
    html_lines = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        f"<title>QAT Readiness Report: {model_name}</title>",
        "<style>",
        "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; ",
        "       max-width: 900px; margin: 0 auto; padding: 20px; line-height: 1.6; }",
        "h1 { color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: 10px; }",
        "h2 { color: #16213e; margin-top: 30px; }",
        "h3 { color: #0f3460; }",
        "table { border-collapse: collapse; width: 100%; margin: 15px 0; }",
        "th, td { border: 1px solid #ddd; padding: 10px; text-align: left; }",
        "th { background: #16213e; color: white; }",
        "tr:nth-child(even) { background: #f9f9f9; }",
        "blockquote { background: #e8f4f8; border-left: 4px solid #0f3460; ",
        "             padding: 15px; margin: 15px 0; }",
        "hr { border: none; border-top: 1px solid #ddd; margin: 30px 0; }",
        ".grade-A { color: #22c55e; } .grade-B { color: #84cc16; }",
        ".grade-C { color: #eab308; } .grade-D { color: #f97316; }",
        ".grade-F { color: #ef4444; }",
        "code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }",
        "</style>",
        "</head>",
        "<body>",
    ]

    # Convert markdown to HTML (simple conversion)
    in_table = False
    in_list = False

    for line in md_report.split("\n"):
        # Headers
        if line.startswith("# "):
            html_lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("### "):
            html_lines.append(f"<h3>{line[4:]}</h3>")
        # Horizontal rule
        elif line == "---":
            html_lines.append("<hr>")
        # Blockquote
        elif line.startswith("> "):
            html_lines.append(f"<blockquote>{line[2:]}</blockquote>")
        # Table
        elif line.startswith("|"):
            if not in_table:
                html_lines.append("<table>")
                in_table = True
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if all(c.startswith("-") for c in cells):
                continue  # Skip separator row
            tag = "th" if not any("<tr>" in line_item for line_item in html_lines[-5:]) else "td"
            row = "".join(f"<{tag}>{c}</{tag}>" for c in cells)
            html_lines.append(f"<tr>{row}</tr>")
        elif in_table and not line.startswith("|"):
            html_lines.append("</table>")
            in_table = False
        # List
        elif line.startswith("- "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append(f"<li>{line[2:]}</li>")
        elif line.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")):
            if not in_list:
                html_lines.append("<ol>")
                in_list = True
            html_lines.append(f"<li>{line[3:]}</li>")
        elif in_list and not line.startswith(("-", "1.", "2.", "3.")):
            html_lines.append("</ul>" if "<ul>" in "".join(html_lines[-10:]) else "</ol>")
            in_list = False
        # Bold
        elif "**" in line:
            line = line.replace("**", "<strong>", 1).replace("**", "</strong>", 1)
            html_lines.append(f"<p>{line}</p>")
        # Italic (report footer)
        elif line.startswith("*") and line.endswith("*"):
            html_lines.append(f"<p><em>{line[1:-1]}</em></p>")
        # Empty line
        elif line == "":
            pass
        # Plain text
        else:
            html_lines.append(f"<p>{line}</p>")

    # Close any open tags
    if in_table:
        html_lines.append("</table>")
    if in_list:
        html_lines.append("</ul>")

    html_lines.extend(["</body>", "</html>"])

    return "\n".join(html_lines)
