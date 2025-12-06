# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
LLM Summarizer module for HaoLine.

Generates human-readable model summaries using LLM APIs (OpenAI, etc.).
Takes the structured JSON report and produces:
- Short summary (1-2 sentences) for quick overview
- Detailed summary (paragraph) for model cards

Usage:
    summarizer = LLMSummarizer()  # Uses OPENAI_API_KEY env var
    result = summarizer.summarize(report)
    print(result.short_summary)
    print(result.detailed_summary)
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from .report import InspectionReport

# Check for OpenAI availability
_OPENAI_AVAILABLE = False
try:
    import openai
    from openai import OpenAI

    _OPENAI_AVAILABLE = True
except ImportError:
    openai = None  # type: ignore
    OpenAI = None  # type: ignore


def is_available() -> bool:
    """Check if LLM summarization is available (openai package installed)."""
    return _OPENAI_AVAILABLE


def has_api_key() -> bool:
    """Check if OpenAI API key is configured."""
    return bool(os.environ.get("OPENAI_API_KEY"))


class LLMSummary(BaseModel):
    """Container for LLM-generated summaries."""

    model_config = ConfigDict(frozen=True)

    short_summary: str  # 1-2 sentences
    detailed_summary: str  # Full paragraph
    model_used: str  # e.g., "gpt-4o-mini"
    tokens_used: int  # Total tokens consumed
    success: bool  # Whether summarization succeeded
    error_message: str | None = None  # Error details if failed


# Prompt templates for model summarization
SYSTEM_PROMPT = """You are an expert ML engineer analyzing ONNX model architectures.
Your task is to provide clear, accurate summaries of model structure and characteristics.
Be concise but informative. Focus on:
- Architecture type and key patterns (CNN, Transformer, RNN, hybrid)
- Model size, computational complexity, and memory requirements
- Hardware deployment considerations (VRAM, latency, bottlenecks)
- Quantization status and precision characteristics
- KV cache requirements for transformer/LLM models
- Potential use cases based on structure
- Any notable characteristics, risks, or optimization opportunities

Respond in plain text without markdown formatting."""

SHORT_SUMMARY_PROMPT = """Based on this ONNX model analysis, write a 1-2 sentence summary.
Focus on: what type of model this is, its size, quantization status, and primary use case.

Model Analysis:
{report_json}

Write only the summary, no preamble or explanation."""

DETAILED_SUMMARY_PROMPT = """Based on this ONNX model analysis, write a detailed paragraph (4-6 sentences).
Include:
1. Architecture type and structure (e.g., CNN, Transformer, hybrid, LLM)
2. Model complexity (parameters, FLOPs, model size, peak memory)
3. Precision and quantization status (FP32, FP16, INT8, mixed precision)
4. Key architectural patterns detected (attention heads, residual blocks, etc.)
5. Hardware deployment analysis:
   - VRAM requirements and whether it fits on target GPU
   - Bottleneck classification (compute-bound vs memory-bound)
   - Theoretical latency and throughput
6. For transformers: KV cache requirements per token and full context
7. Any risk signals, deployment concerns, or optimization recommendations

Model Analysis:
{report_json}

Write only the summary paragraph, no preamble or bullet points."""


class LLMSummarizer:
    """
    Generate human-readable summaries of ONNX models using LLM APIs.

    Supports OpenAI API with graceful fallback when unavailable.

    Example:
        summarizer = LLMSummarizer()
        result = summarizer.summarize(report)
        if result.success:
            print(result.detailed_summary)
    """

    DEFAULT_MODEL: ClassVar[str] = "gpt-4o-mini"  # Cost-effective, fast, good quality
    FALLBACK_MODELS: ClassVar[list[str]] = [
        "gpt-3.5-turbo",
        "gpt-4o",
    ]  # Fallbacks if primary fails

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        logger: logging.Logger | None = None,
    ):
        """
        Initialize the LLM summarizer.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            model: Model to use. If None, uses gpt-4o-mini.
            logger: Logger for diagnostic output.
        """
        self.logger = logger or logging.getLogger("haoline.llm")
        self.model = model or self.DEFAULT_MODEL

        if not _OPENAI_AVAILABLE:
            self.client = None
            self.logger.warning("openai package not installed. LLM summarization disabled.")
            return

        # Get API key from parameter or environment
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            self.client = None
            self.logger.warning("No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
            return

        self.client = OpenAI(api_key=resolved_key)
        self.logger.debug(f"LLM summarizer initialized with model: {self.model}")

    def is_configured(self) -> bool:
        """Check if the summarizer is properly configured and ready to use."""
        return self.client is not None

    def summarize(self, report: InspectionReport) -> LLMSummary:
        """
        Generate both short and detailed summaries for a model report.

        Args:
            report: The inspection report to summarize.

        Returns:
            LLMSummary with both summaries and metadata.
        """
        if not self.is_configured():
            return LLMSummary(
                short_summary="",
                detailed_summary="",
                model_used="",
                tokens_used=0,
                success=False,
                error_message="LLM summarizer not configured. Install openai and set OPENAI_API_KEY.",
            )

        # Prepare a condensed version of the report for the prompt
        report_json = self._prepare_report_for_prompt(report)

        total_tokens = 0
        short_summary = ""
        detailed_summary = ""
        error_message = None

        # Generate short summary
        try:
            short_summary, tokens = self._generate_completion(
                SHORT_SUMMARY_PROMPT.format(report_json=report_json)
            )
            total_tokens += tokens
            self.logger.debug(f"Short summary generated ({tokens} tokens)")
        except Exception as e:
            self.logger.warning(f"Failed to generate short summary: {e}")
            error_message = str(e)

        # Generate detailed summary
        try:
            detailed_summary, tokens = self._generate_completion(
                DETAILED_SUMMARY_PROMPT.format(report_json=report_json)
            )
            total_tokens += tokens
            self.logger.debug(f"Detailed summary generated ({tokens} tokens)")
        except Exception as e:
            self.logger.warning(f"Failed to generate detailed summary: {e}")
            if not error_message:
                error_message = str(e)

        success = bool(short_summary or detailed_summary)

        return LLMSummary(
            short_summary=short_summary,
            detailed_summary=detailed_summary,
            model_used=self.model,
            tokens_used=total_tokens,
            success=success,
            error_message=error_message if not success else None,
        )

    def generate_short_summary(self, report: InspectionReport) -> str:
        """Generate only a short summary (1-2 sentences)."""
        if not self.is_configured():
            return ""

        report_json = self._prepare_report_for_prompt(report)
        try:
            summary, _ = self._generate_completion(
                SHORT_SUMMARY_PROMPT.format(report_json=report_json)
            )
            return summary
        except Exception as e:
            self.logger.error(f"Failed to generate short summary: {e}")
            return ""

    def generate_detailed_summary(self, report: InspectionReport) -> str:
        """Generate only a detailed summary (paragraph)."""
        if not self.is_configured():
            return ""

        report_json = self._prepare_report_for_prompt(report)
        try:
            summary, _ = self._generate_completion(
                DETAILED_SUMMARY_PROMPT.format(report_json=report_json)
            )
            return summary
        except Exception as e:
            self.logger.error(f"Failed to generate detailed summary: {e}")
            return ""

    def _generate_completion(self, user_prompt: str) -> tuple[str, int]:
        """
        Call the OpenAI API to generate a completion.

        Args:
            user_prompt: The user prompt to send.

        Returns:
            Tuple of (response_text, tokens_used)

        Raises:
            Exception: If API call fails after retries.
            RuntimeError: If client is not configured.
        """
        if self.client is None:
            raise RuntimeError("LLM client is not configured")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=500,
                temperature=0.3,  # Lower temperature for more consistent outputs
            )

            content = response.choices[0].message.content or ""
            tokens = response.usage.total_tokens if response.usage else 0

            return content.strip(), tokens

        except openai.RateLimitError as e:
            self.logger.warning(f"Rate limit hit: {e}. Consider adding retry logic.")
            raise

        except openai.APIConnectionError as e:
            self.logger.error(f"API connection error: {e}")
            raise

        except openai.AuthenticationError as e:
            self.logger.error(f"Authentication failed: {e}. Check your API key.")
            raise

        except Exception as e:
            self.logger.error(f"Unexpected error calling OpenAI API: {e}")
            raise

    def _prepare_report_for_prompt(self, report: InspectionReport) -> str:
        """
        Prepare a condensed version of the report for LLM consumption.

        Keeps the most relevant information while staying within token limits.
        Includes all analysis sections: metrics, precision, memory, hardware, KV cache, etc.
        """
        # Build a focused summary dict
        summary: dict[str, Any] = {
            "model_name": report.metadata.path.split("/")[-1].split("\\")[-1],
            "producer": f"{report.metadata.producer_name} {report.metadata.producer_version}".strip(),
            "opsets": report.metadata.opsets,
        }

        if report.graph_summary:
            summary["graph"] = {
                "nodes": report.graph_summary.num_nodes,
                "inputs": report.graph_summary.num_inputs,
                "outputs": report.graph_summary.num_outputs,
                "initializers": report.graph_summary.num_initializers,
                "input_shapes": report.graph_summary.input_shapes,
                "output_shapes": report.graph_summary.output_shapes,
                "top_operators": dict(
                    sorted(report.graph_summary.op_type_counts.items(), key=lambda x: -x[1])[:10]
                ),
            }

        if report.param_counts:
            param_summary: dict[str, Any] = {
                "total": report.param_counts.total,
                "by_op_type": dict(
                    sorted(report.param_counts.by_op_type.items(), key=lambda x: -x[1])[:5]
                ),
            }
            # Precision breakdown (Story 41.5: LLM prompt enhancement)
            if report.param_counts.precision_breakdown:
                param_summary["precision_breakdown"] = report.param_counts.precision_breakdown
            if report.param_counts.is_quantized:
                param_summary["is_quantized"] = True
                if report.param_counts.quantized_ops:
                    param_summary["quantized_ops"] = report.param_counts.quantized_ops[:5]
            # Shared weights
            if report.param_counts.num_shared_weights > 0:
                param_summary["num_shared_weights"] = report.param_counts.num_shared_weights
            summary["parameters"] = param_summary

        if report.flop_counts:
            summary["flops"] = {
                "total": report.flop_counts.total,
                "by_op_type": dict(
                    sorted(report.flop_counts.by_op_type.items(), key=lambda x: -x[1])[:5]
                ),
            }

        if report.memory_estimates:
            mem = report.memory_estimates
            memory_summary: dict[str, Any] = {
                "model_size_bytes": mem.model_size_bytes,
                "peak_activation_bytes": mem.peak_activation_bytes,
            }
            # KV Cache for transformers (Story 41.5)
            if mem.kv_cache_bytes_per_token > 0:
                memory_summary["kv_cache"] = {
                    "bytes_per_token": mem.kv_cache_bytes_per_token,
                    "bytes_full_context": mem.kv_cache_bytes_full_context,
                }
                if mem.kv_cache_config:
                    memory_summary["kv_cache"]["config"] = mem.kv_cache_config
            # Memory breakdown by op type (Story 41.5)
            if mem.breakdown:
                bd = mem.breakdown
                if bd.weights_by_op_type:
                    memory_summary["weights_by_op_type"] = dict(
                        sorted(bd.weights_by_op_type.items(), key=lambda x: -x[1])[:5]
                    )
                if bd.activations_by_op_type:
                    memory_summary["activations_by_op_type"] = dict(
                        sorted(bd.activations_by_op_type.items(), key=lambda x: -x[1])[:5]
                    )
            summary["memory"] = memory_summary

        summary["architecture_type"] = report.architecture_type

        if report.detected_blocks:
            block_types: dict[str, int] = {}
            for block in report.detected_blocks:
                block_types[block.block_type] = block_types.get(block.block_type, 0) + 1
            summary["detected_blocks"] = block_types

        if report.risk_signals:
            summary["risks"] = [
                {"id": r.id, "severity": r.severity, "description": r.description}
                for r in report.risk_signals[:5]  # Top 5 risks
            ]

        if report.hardware_estimates:
            hw = report.hardware_estimates
            hw_summary: dict[str, Any] = {
                "device": hw.device,
                "precision": hw.precision,
                "batch_size": hw.batch_size,
                "vram_required_bytes": hw.vram_required_bytes,
                "fits_in_vram": hw.fits_in_vram,
                "theoretical_latency_ms": round(hw.theoretical_latency_ms, 2),
                "bottleneck": hw.bottleneck,
            }
            # Extended hardware metrics (Story 41.5)
            if hasattr(hw, "compute_utilization_estimate"):
                hw_summary["compute_utilization"] = round(hw.compute_utilization_estimate * 100, 1)
            if hasattr(hw, "gpu_saturation"):
                hw_summary["gpu_saturation_percent"] = round(hw.gpu_saturation * 100, 2)
            if hasattr(hw, "throughput_fps"):
                hw_summary["throughput_fps"] = round(hw.throughput_fps, 1)
            summary["hardware_estimates"] = hw_summary

        # System requirements if available
        if hasattr(report, "system_requirements") and report.system_requirements:
            sr = report.system_requirements
            sr_dict: dict[str, Any] = {}
            if sr.minimum:
                sr_dict["minimum"] = {
                    "gpu": sr.minimum.device,
                    "vram_gb": round(sr.minimum.vram_required_bytes / (1024**3), 1),
                }
            if sr.recommended:
                sr_dict["recommended"] = {
                    "gpu": sr.recommended.device,
                    "vram_gb": round(sr.recommended.vram_required_bytes / (1024**3), 1),
                }
            if sr_dict:
                summary["system_requirements"] = sr_dict

        # Bottleneck analysis with recommendations (Story 41.5.7)
        if hasattr(report, "bottleneck_analysis") and report.bottleneck_analysis:
            ba = report.bottleneck_analysis
            summary["bottleneck_analysis"] = {
                "type": ba.bottleneck_type,
                "compute_ratio": ba.compute_ratio,
                "memory_ratio": ba.memory_ratio,
                "efficiency_percent": ba.efficiency_percent,
                "recommendations": ba.recommendations[:3],  # Top 3 recommendations
            }

        return json.dumps(summary, indent=2)


def summarize_report(
    report: InspectionReport,
    api_key: str | None = None,
    model: str | None = None,
    logger: logging.Logger | None = None,
) -> LLMSummary:
    """
    Convenience function to generate LLM summaries for a report.

    Args:
        report: The inspection report to summarize.
        api_key: OpenAI API key (optional, uses env var if not provided).
        model: Model to use (optional, defaults to gpt-4o-mini).
        logger: Logger for output.

    Returns:
        LLMSummary with results.
    """
    summarizer = LLMSummarizer(api_key=api_key, model=model, logger=logger)
    return summarizer.summarize(report)
