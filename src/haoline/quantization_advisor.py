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
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

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


@dataclass
class QuantizationAdvice:
    """Container for LLM-generated quantization recommendations."""

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
        success=True,
    )


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

        if self.use_llm:
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if key and OpenAI is not None:
                self.client = OpenAI(api_key=key)
            else:
                self.client = None
                self.use_llm = False
        else:
            self.client = None

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
            advice.error_message = str(e)
            return advice

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

        return QuantizationAdvice(
            architecture_type=arch_type,
            architecture_summary=arch_summary,
            strategy=strategy_data.get("strategy", "No strategy available"),
            sensitive_layers=strategy_data.get("sensitive_layers", []),
            safe_layers=strategy_data.get("safe_layers", []),
            qat_workflow=strategy_data.get("qat_workflow", []),
            calibration_tips=strategy_data.get("calibration_tips", ""),
            runtime_recommendations=runtime_data,
            expected_accuracy_impact=strategy_data.get("expected_accuracy_impact", "Unknown"),
            mitigation_strategies=strategy_data.get("mitigation_strategies", []),
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
            return json.loads(content)
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
