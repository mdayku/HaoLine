# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
LLM Deployment Analysis (Epic 30).

This module analyzes LLM inference characteristics for production deployment:

1. **Prefill vs Decode Analysis** (Story 30.1):
   - Prefill phase: compute-bound, process prompt in parallel
   - Decode phase: memory-bound, generate tokens sequentially
   - TTFT and throughput estimates

2. **Batching Strategy Analysis** (Story 30.2):
   - Static vs continuous batching
   - Throughput vs latency tradeoffs
   - Max concurrent requests for VRAM

3. **Context Length Scaling** (Story 30.3):
   - O(n²) attention scaling
   - O(n) KV cache scaling
   - Context length breakpoints

4. **Serving Framework Compatibility** (Story 30.4):
   - vLLM, TensorRT-LLM, llama.cpp, Triton compatibility
   - Framework recommendations based on model characteristics

Usage:
    from haoline.deployment_analysis import DeploymentAnalyzer

    analyzer = DeploymentAnalyzer()
    result = analyzer.analyze(graph_info, blocks, attention_result, memory_result)
    print(result.prefill_decode)  # Prefill vs decode analysis
    print(result.serving_frameworks)  # Framework compatibility
"""

from __future__ import annotations

import logging
import math
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from haoline.analyzer import GraphInfo
    from haoline.attention_analysis import AttentionAnalysisResult
    from haoline.memory_analysis import MemoryAnalysisResult
    from haoline.patterns import Block


# =============================================================================
# Enums and Constants
# =============================================================================


class ServingFramework(Enum):
    """LLM serving frameworks."""

    VLLM = "vllm"
    TENSORRT_LLM = "tensorrt_llm"
    LLAMA_CPP = "llama_cpp"
    TRITON = "triton"
    HUGGINGFACE_TGI = "huggingface_tgi"
    ONNX_RUNTIME = "onnx_runtime"


class BatchingStrategy(Enum):
    """Batching strategies."""

    STATIC = "static"  # Fixed batch size
    DYNAMIC = "dynamic"  # Variable batch size
    CONTINUOUS = "continuous"  # vLLM-style continuous batching


# GPU compute characteristics (TFLOPs at FP16)
GPU_COMPUTE_TFLOPS: dict[str, float] = {
    "h100": 989.0,
    "a100": 312.0,
    "a10": 125.0,
    "l4": 121.0,
    "rtx4090": 165.0,
    "rtx3090": 71.0,
    "v100": 125.0,
    "t4": 65.0,
}

# GPU memory bandwidth (GB/s)
GPU_MEMORY_BANDWIDTH: dict[str, float] = {
    "h100": 3350.0,
    "a100": 2039.0,
    "a10": 600.0,
    "l4": 300.0,
    "rtx4090": 1008.0,
    "rtx3090": 936.0,
    "v100": 900.0,
    "t4": 300.0,
}


# =============================================================================
# Data Models
# =============================================================================


class PrefillDecodeAnalysis(BaseModel):
    """Prefill vs Decode phase analysis."""

    model_config = ConfigDict(extra="forbid")

    # Prefill characteristics
    prefill_is_compute_bound: bool = True
    prefill_flops_per_token: int = 0  # FLOPs to process one input token
    estimated_ttft_ms: float = 0.0  # Time to first token at 1K context

    # Decode characteristics
    decode_is_memory_bound: bool = True
    decode_bytes_per_token: int = 0  # Memory accessed per output token
    estimated_tokens_per_second: float = 0.0  # At batch=1

    # Optimal settings
    optimal_prefill_batch_size: int = 1
    optimal_decode_batch_size: int = 1

    # Context length projections
    ttft_at_4k_ms: float = 0.0
    ttft_at_8k_ms: float = 0.0
    ttft_at_32k_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prefill_is_compute_bound": self.prefill_is_compute_bound,
            "prefill_flops_per_token": self.prefill_flops_per_token,
            "estimated_ttft_ms": self.estimated_ttft_ms,
            "decode_is_memory_bound": self.decode_is_memory_bound,
            "decode_bytes_per_token": self.decode_bytes_per_token,
            "estimated_tokens_per_second": self.estimated_tokens_per_second,
            "optimal_prefill_batch_size": self.optimal_prefill_batch_size,
            "optimal_decode_batch_size": self.optimal_decode_batch_size,
            "ttft_at_4k_ms": self.ttft_at_4k_ms,
            "ttft_at_8k_ms": self.ttft_at_8k_ms,
            "ttft_at_32k_ms": self.ttft_at_32k_ms,
        }


class BatchingAnalysis(BaseModel):
    """Batching strategy analysis."""

    model_config = ConfigDict(extra="forbid")

    # Current model characteristics
    supports_dynamic_batching: bool = True
    supports_continuous_batching: bool = False
    has_paged_attention: bool = False

    # Throughput analysis
    throughput_at_batch_1: float = 0.0  # tokens/sec
    throughput_at_batch_4: float = 0.0
    throughput_at_batch_8: float = 0.0
    throughput_at_batch_16: float = 0.0
    throughput_at_batch_32: float = 0.0

    # Latency analysis
    latency_at_batch_1_ms: float = 0.0
    latency_at_batch_32_ms: float = 0.0

    # VRAM-based limits
    max_concurrent_requests: int = 1
    max_batch_size_for_vram: int = 1

    # Recommendations
    recommended_strategy: str = "static"
    recommended_batch_size: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "supports_dynamic_batching": self.supports_dynamic_batching,
            "supports_continuous_batching": self.supports_continuous_batching,
            "has_paged_attention": self.has_paged_attention,
            "throughput_at_batch_1": self.throughput_at_batch_1,
            "throughput_at_batch_4": self.throughput_at_batch_4,
            "throughput_at_batch_8": self.throughput_at_batch_8,
            "throughput_at_batch_16": self.throughput_at_batch_16,
            "throughput_at_batch_32": self.throughput_at_batch_32,
            "latency_at_batch_1_ms": self.latency_at_batch_1_ms,
            "latency_at_batch_32_ms": self.latency_at_batch_32_ms,
            "max_concurrent_requests": self.max_concurrent_requests,
            "max_batch_size_for_vram": self.max_batch_size_for_vram,
            "recommended_strategy": self.recommended_strategy,
            "recommended_batch_size": self.recommended_batch_size,
        }


class ContextScalingInfo(BaseModel):
    """Context length scaling analysis."""

    model_config = ConfigDict(extra="forbid")

    # Base metrics
    model_base_context: int = 4096
    max_supported_context: int = 0  # From model config if available

    # Memory scaling (O(n) for KV cache)
    kv_cache_bytes_per_token: int = 0
    memory_at_4k_gb: float = 0.0
    memory_at_8k_gb: float = 0.0
    memory_at_32k_gb: float = 0.0
    memory_at_128k_gb: float = 0.0

    # Compute scaling (O(n²) for attention)
    attention_flops_at_4k: int = 0
    attention_flops_at_8k: int = 0  # ~4x of 4k
    attention_flops_at_32k: int = 0  # ~64x of 4k
    attention_flops_at_128k: int = 0  # ~1024x of 4k

    # Breakpoints
    oom_context_length: int = 0  # Where model runs out of memory
    latency_breakpoint: int = 0  # Where latency becomes problematic
    recommended_max_context: int = 0  # For target hardware

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_base_context": self.model_base_context,
            "max_supported_context": self.max_supported_context,
            "kv_cache_bytes_per_token": self.kv_cache_bytes_per_token,
            "memory_at_4k_gb": self.memory_at_4k_gb,
            "memory_at_8k_gb": self.memory_at_8k_gb,
            "memory_at_32k_gb": self.memory_at_32k_gb,
            "memory_at_128k_gb": self.memory_at_128k_gb,
            "attention_flops_at_4k": self.attention_flops_at_4k,
            "attention_flops_at_8k": self.attention_flops_at_8k,
            "attention_flops_at_32k": self.attention_flops_at_32k,
            "attention_flops_at_128k": self.attention_flops_at_128k,
            "oom_context_length": self.oom_context_length,
            "latency_breakpoint": self.latency_breakpoint,
            "recommended_max_context": self.recommended_max_context,
        }


class FrameworkCompatibility(BaseModel):
    """Serving framework compatibility info."""

    model_config = ConfigDict(extra="forbid")

    framework: str = ""
    compatible: bool = False
    compatibility_score: float = 0.0  # 0-1, how well suited
    notes: list[str] = Field(default_factory=list)
    required_conversions: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "framework": self.framework,
            "compatible": self.compatible,
            "compatibility_score": self.compatibility_score,
            "notes": self.notes,
            "required_conversions": self.required_conversions,
        }


class DeploymentAnalysisResult(BaseModel):
    """Complete deployment analysis result."""

    model_config = ConfigDict(extra="forbid")

    # Prefill vs Decode (Story 30.1)
    prefill_decode: PrefillDecodeAnalysis = Field(default_factory=PrefillDecodeAnalysis)

    # Batching (Story 30.2)
    batching: BatchingAnalysis = Field(default_factory=BatchingAnalysis)

    # Context scaling (Story 30.3)
    context_scaling: ContextScalingInfo = Field(default_factory=ContextScalingInfo)

    # Framework compatibility (Story 30.4)
    serving_frameworks: list[FrameworkCompatibility] = Field(default_factory=list)
    recommended_framework: str = ""

    # Overall recommendations
    recommendations: list[str] = Field(default_factory=list)

    # Hardware used for analysis
    target_gpu: str = "a100"
    target_vram_gb: float = 80.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prefill_decode": self.prefill_decode.to_dict(),
            "batching": self.batching.to_dict(),
            "context_scaling": self.context_scaling.to_dict(),
            "serving_frameworks": [f.to_dict() for f in self.serving_frameworks],
            "recommended_framework": self.recommended_framework,
            "recommendations": self.recommendations,
            "target_gpu": self.target_gpu,
            "target_vram_gb": self.target_vram_gb,
        }

    def get_summary(self) -> str:
        """Generate human-readable summary."""
        lines = ["LLM Deployment Analysis", "=" * 30]

        # Prefill/Decode
        pd = self.prefill_decode
        lines.append("")
        lines.append("Prefill vs Decode:")
        lines.append(f"  TTFT (1K context): {pd.estimated_ttft_ms:.1f} ms")
        lines.append(f"  Decode Speed: {pd.estimated_tokens_per_second:.1f} tokens/sec")
        lines.append(
            f"  Prefill: {'Compute-bound' if pd.prefill_is_compute_bound else 'Memory-bound'}"
        )
        lines.append(
            f"  Decode: {'Memory-bound' if pd.decode_is_memory_bound else 'Compute-bound'}"
        )

        # Batching
        b = self.batching
        lines.append("")
        lines.append("Batching:")
        lines.append(f"  Recommended: {b.recommended_strategy} (batch={b.recommended_batch_size})")
        lines.append(f"  Max Concurrent: {b.max_concurrent_requests} requests")
        if b.has_paged_attention:
            lines.append("  PagedAttention: Supported")

        # Context scaling
        cs = self.context_scaling
        lines.append("")
        lines.append("Context Length:")
        lines.append(f"  OOM at: {cs.oom_context_length:,} tokens")
        lines.append(f"  Recommended max: {cs.recommended_max_context:,} tokens")

        # Framework
        if self.recommended_framework:
            lines.append("")
            lines.append(f"Recommended Framework: {self.recommended_framework}")

        # Recommendations
        if self.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")

        return "\n".join(lines)


# =============================================================================
# Analyzer Class
# =============================================================================


class DeploymentAnalyzer:
    """
    Analyzer for LLM deployment characteristics.

    Analyzes prefill vs decode phases, batching strategies,
    context length scaling, and serving framework compatibility.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        target_gpu: str = "a100",
        vram_gb: float = 80.0,
    ):
        self.logger = logger or logging.getLogger("haoline.deployment_analysis")
        self.target_gpu = target_gpu.lower()
        self.vram_gb = vram_gb

        # Get GPU specs
        self.gpu_tflops = GPU_COMPUTE_TFLOPS.get(self.target_gpu, 312.0)
        self.gpu_bandwidth = GPU_MEMORY_BANDWIDTH.get(self.target_gpu, 2039.0)

    def analyze(
        self,
        graph_info: GraphInfo,
        blocks: list[Block] | None = None,
        attention_result: AttentionAnalysisResult | None = None,
        memory_result: MemoryAnalysisResult | None = None,
        total_params: int = 0,
        total_flops: int = 0,
    ) -> DeploymentAnalysisResult:
        """
        Analyze LLM deployment characteristics.

        Args:
            graph_info: Graph analysis info.
            blocks: Pattern blocks from PatternAnalyzer.
            attention_result: Attention analysis from Epic 27.
            memory_result: Memory analysis from Epic 28.
            total_params: Total parameter count.
            total_flops: Total FLOPs count.

        Returns:
            DeploymentAnalysisResult with all findings.
        """
        result = DeploymentAnalysisResult(
            target_gpu=self.target_gpu,
            target_vram_gb=self.vram_gb,
        )
        recommendations: list[str] = []

        # Extract key metrics from prior analyses
        kv_bytes_per_token = 0
        model_size_gb = 0.0
        has_paged_attention = False
        num_layers = 0
        hidden_dim = 0

        if attention_result:
            if attention_result.kv_cache:
                kv_bytes_per_token = attention_result.kv_cache.bytes_per_token
            num_layers = attention_result.num_attention_layers or self._estimate_layers(graph_info)

        if memory_result:
            model_size_gb = memory_result.model_size_gb
            if memory_result.kv_cache:
                has_paged_attention = memory_result.kv_cache.paged_attention_detected
                if kv_bytes_per_token == 0:
                    kv_bytes_per_token = memory_result.kv_cache.bytes_per_token

        # Estimate hidden dim from params if not available
        if num_layers > 0 and total_params > 0:
            # Rough estimate: params ~= 12 * hidden^2 * num_layers for transformers
            hidden_dim = int(math.sqrt(total_params / (12 * num_layers)))
        if hidden_dim == 0:
            hidden_dim = 4096  # Default

        # Story 30.1: Prefill vs Decode Analysis
        prefill_decode = self._analyze_prefill_decode(
            total_params=total_params,
            total_flops=total_flops,
            kv_bytes_per_token=kv_bytes_per_token,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            model_size_gb=model_size_gb,
        )
        result.prefill_decode = prefill_decode

        if prefill_decode.estimated_ttft_ms > 1000:
            recommendations.append(
                f"High TTFT ({prefill_decode.estimated_ttft_ms:.0f}ms) - "
                "consider chunked prefill or speculative decoding"
            )

        # Story 30.2: Batching Strategy Analysis
        batching = self._analyze_batching(
            model_size_gb=model_size_gb,
            kv_bytes_per_token=kv_bytes_per_token,
            has_paged_attention=has_paged_attention,
            tokens_per_second=prefill_decode.estimated_tokens_per_second,
        )
        result.batching = batching

        if batching.has_paged_attention:
            recommendations.append("PagedAttention detected - vLLM/TensorRT-LLM recommended")
        if batching.max_concurrent_requests < 4:
            recommendations.append(
                f"Low concurrency limit ({batching.max_concurrent_requests}) - "
                "consider model quantization or multi-GPU"
            )

        # Story 30.3: Context Length Scaling
        context_scaling = self._analyze_context_scaling(
            kv_bytes_per_token=kv_bytes_per_token,
            model_size_gb=model_size_gb,
            total_flops=total_flops,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
        )
        result.context_scaling = context_scaling

        if context_scaling.oom_context_length < 32768:
            recommendations.append(
                f"Context limited to {context_scaling.oom_context_length:,} tokens - "
                "use KV cache quantization or multi-GPU for longer contexts"
            )

        # Story 30.4: Serving Framework Compatibility
        frameworks = self._analyze_frameworks(
            graph_info=graph_info,
            model_size_gb=model_size_gb,
            has_paged_attention=has_paged_attention,
            attention_result=attention_result,
        )
        result.serving_frameworks = frameworks

        # Pick recommended framework
        best_framework = max(frameworks, key=lambda f: f.compatibility_score, default=None)
        if best_framework:
            result.recommended_framework = best_framework.framework

        result.recommendations = recommendations
        return result

    def _estimate_layers(self, graph_info: GraphInfo) -> int:
        """Estimate number of transformer layers from graph."""
        # Count LayerNormalization ops - typically 2 per layer
        ln_count = sum(1 for n in graph_info.nodes if "LayerNorm" in n.op_type)
        if ln_count >= 2:
            return ln_count // 2

        # Fallback: count attention-related softmax
        softmax_count = sum(1 for n in graph_info.nodes if n.op_type == "Softmax")
        return max(softmax_count, 1)

    # =========================================================================
    # Story 30.1: Prefill vs Decode Analysis
    # =========================================================================

    def _analyze_prefill_decode(
        self,
        total_params: int,
        total_flops: int,
        kv_bytes_per_token: int,
        num_layers: int,
        hidden_dim: int,
        model_size_gb: float,
    ) -> PrefillDecodeAnalysis:
        """Analyze prefill and decode phase characteristics."""
        result = PrefillDecodeAnalysis()

        # Prefill: compute-bound (process all tokens in parallel)
        result.prefill_is_compute_bound = True

        # FLOPs per token during prefill (forward pass per token)
        # Rough estimate: 2 * params for forward pass, plus attention
        if total_flops > 0:
            result.prefill_flops_per_token = total_flops  # Per forward pass
        else:
            result.prefill_flops_per_token = 2 * total_params

        # TTFT estimate based on compute
        # TTFT = prefill_tokens * flops_per_token / gpu_tflops
        context_tokens = 1024  # Base estimate at 1K
        gpu_flops_per_sec = self.gpu_tflops * 1e12
        prefill_time_sec = (context_tokens * result.prefill_flops_per_token) / gpu_flops_per_sec
        result.estimated_ttft_ms = prefill_time_sec * 1000

        # Projections for different context lengths
        result.ttft_at_4k_ms = result.estimated_ttft_ms * 4
        result.ttft_at_8k_ms = result.estimated_ttft_ms * 8
        result.ttft_at_32k_ms = result.estimated_ttft_ms * 32

        # Decode: memory-bound (sequential token generation)
        result.decode_is_memory_bound = True

        # Memory accessed per decode token:
        # - Model weights (once per token)
        # - KV cache (grows with context)
        model_bytes = model_size_gb * 1e9 if model_size_gb > 0 else total_params * 2
        result.decode_bytes_per_token = int(model_bytes + kv_bytes_per_token * 1024)

        # Tokens per second based on memory bandwidth
        # decode_time = model_bytes / bandwidth + kv_cache_time
        gpu_bandwidth_bytes = self.gpu_bandwidth * 1e9  # Convert GB/s to bytes/s
        if model_bytes > 0:
            decode_time_sec = model_bytes / gpu_bandwidth_bytes
            result.estimated_tokens_per_second = 1.0 / decode_time_sec
        else:
            result.estimated_tokens_per_second = 50.0  # Default estimate

        # Optimal batch sizes
        # Prefill: can use larger batches (compute-bound)
        result.optimal_prefill_batch_size = min(8, max(1, int(self.vram_gb / 10)))

        # Decode: smaller batches often better (memory-bound)
        result.optimal_decode_batch_size = min(32, max(1, int(self.vram_gb / 5)))

        return result

    # =========================================================================
    # Story 30.2: Batching Strategy Analysis
    # =========================================================================

    def _analyze_batching(
        self,
        model_size_gb: float,
        kv_bytes_per_token: int,
        has_paged_attention: bool,
        tokens_per_second: float,
    ) -> BatchingAnalysis:
        """Analyze batching strategies."""
        result = BatchingAnalysis()

        result.supports_dynamic_batching = True
        result.supports_continuous_batching = has_paged_attention
        result.has_paged_attention = has_paged_attention

        # Base throughput at batch=1
        result.throughput_at_batch_1 = tokens_per_second

        # Throughput scaling with batch size
        # Batching improves throughput but not linearly due to memory constraints
        for batch_size in [4, 8, 16, 32]:
            # Sublinear scaling due to memory bottleneck
            scaling_factor = math.sqrt(batch_size)
            throughput = tokens_per_second * scaling_factor

            if batch_size == 4:
                result.throughput_at_batch_4 = throughput
            elif batch_size == 8:
                result.throughput_at_batch_8 = throughput
            elif batch_size == 16:
                result.throughput_at_batch_16 = throughput
            elif batch_size == 32:
                result.throughput_at_batch_32 = throughput

        # Latency increases with batch size
        if tokens_per_second > 0:
            result.latency_at_batch_1_ms = 1000.0 / tokens_per_second
            # Latency scales roughly linearly with batch size
            result.latency_at_batch_32_ms = result.latency_at_batch_1_ms * 32 / math.sqrt(32)

        # VRAM-based limits
        # Available VRAM = total - model - activations overhead
        available_vram_gb = self.vram_gb - model_size_gb - (self.vram_gb * 0.1)  # 10% overhead
        kv_per_request_gb = (kv_bytes_per_token * 4096) / 1e9  # Assume 4K context

        if kv_per_request_gb > 0:
            result.max_concurrent_requests = max(1, int(available_vram_gb / kv_per_request_gb))
            result.max_batch_size_for_vram = result.max_concurrent_requests
        else:
            result.max_concurrent_requests = 16
            result.max_batch_size_for_vram = 16

        # Recommendations
        if has_paged_attention:
            result.recommended_strategy = "continuous"
            result.recommended_batch_size = min(result.max_batch_size_for_vram, 16)
        else:
            result.recommended_strategy = "dynamic"
            result.recommended_batch_size = min(result.max_batch_size_for_vram, 8)

        return result

    # =========================================================================
    # Story 30.3: Context Length Scaling
    # =========================================================================

    def _analyze_context_scaling(
        self,
        kv_bytes_per_token: int,
        model_size_gb: float,
        total_flops: int,
        num_layers: int,
        hidden_dim: int,
    ) -> ContextScalingInfo:
        """Analyze context length scaling characteristics."""
        result = ContextScalingInfo()

        result.model_base_context = 4096  # Default assumption
        result.kv_cache_bytes_per_token = kv_bytes_per_token

        # Memory scaling (O(n) for KV cache)
        for ctx in [4096, 8192, 32768, 131072]:
            memory_gb = (kv_bytes_per_token * ctx) / 1e9
            if ctx == 4096:
                result.memory_at_4k_gb = memory_gb
            elif ctx == 8192:
                result.memory_at_8k_gb = memory_gb
            elif ctx == 32768:
                result.memory_at_32k_gb = memory_gb
            elif ctx == 131072:
                result.memory_at_128k_gb = memory_gb

        # Compute scaling (O(n²) for attention)
        # Attention FLOPs per layer ~= 2 * seq_len^2 * hidden_dim
        base_attention_flops = 2 * 4096 * 4096 * hidden_dim * num_layers
        result.attention_flops_at_4k = base_attention_flops
        result.attention_flops_at_8k = base_attention_flops * 4  # (8K/4K)²
        result.attention_flops_at_32k = base_attention_flops * 64  # (32K/4K)²
        result.attention_flops_at_128k = base_attention_flops * 1024  # (128K/4K)²

        # Find OOM context length
        available_vram_gb = self.vram_gb - model_size_gb - (self.vram_gb * 0.15)
        if kv_bytes_per_token > 0:
            oom_tokens = int((available_vram_gb * 1e9) / kv_bytes_per_token)
            result.oom_context_length = min(oom_tokens, 1_000_000)  # Cap at 1M
        else:
            result.oom_context_length = 128_000  # Default

        # Latency breakpoint: where latency > 100ms per token
        # Rough heuristic: latency scales quadratically with attention
        result.latency_breakpoint = min(result.oom_context_length, 65536)

        # Recommended max context (balance of latency and memory)
        result.recommended_max_context = min(
            result.oom_context_length,
            result.latency_breakpoint,
            32768,  # Practical limit for most use cases
        )

        result.max_supported_context = result.oom_context_length

        return result

    # =========================================================================
    # Story 30.4: Serving Framework Compatibility
    # =========================================================================

    def _analyze_frameworks(
        self,
        graph_info: GraphInfo,
        model_size_gb: float,
        has_paged_attention: bool,
        attention_result: AttentionAnalysisResult | None,
    ) -> list[FrameworkCompatibility]:
        """Analyze serving framework compatibility."""
        frameworks: list[FrameworkCompatibility] = []

        # Check ONNX format (we know it's ONNX if we got here)
        is_onnx = True

        # Check for attention type
        is_gqa = False
        is_mqa = False
        if attention_result:
            attn_type = attention_result.primary_attention_type.value
            is_gqa = "gqa" in attn_type.lower()
            is_mqa = "mqa" in attn_type.lower()

        # 1. vLLM
        vllm = FrameworkCompatibility(framework="vLLM")
        vllm.compatible = True
        vllm.compatibility_score = 0.9 if has_paged_attention else 0.7
        vllm.notes = ["Best for high-throughput serving", "Supports continuous batching"]
        if has_paged_attention:
            vllm.notes.append("PagedAttention detected - excellent fit")
            vllm.compatibility_score = 0.95
        if is_gqa or is_mqa:
            vllm.notes.append("GQA/MQA supported for memory efficiency")
        vllm.required_conversions = ["Convert to HuggingFace format if not already"]
        frameworks.append(vllm)

        # 2. TensorRT-LLM
        trt_llm = FrameworkCompatibility(framework="TensorRT-LLM")
        trt_llm.compatible = self.target_gpu in GPU_COMPUTE_TFLOPS
        trt_llm.compatibility_score = 0.85 if trt_llm.compatible else 0.0
        trt_llm.notes = ["Best for NVIDIA GPUs", "Requires TensorRT compilation"]
        if "h100" in self.target_gpu or "a100" in self.target_gpu:
            trt_llm.notes.append(f"Optimized for {self.target_gpu.upper()}")
            trt_llm.compatibility_score = 0.95
        trt_llm.required_conversions = ["Build TensorRT engine from model"]
        frameworks.append(trt_llm)

        # 3. llama.cpp
        llama_cpp = FrameworkCompatibility(framework="llama.cpp")
        llama_cpp.compatible = model_size_gb < 100  # Practical limit
        llama_cpp.compatibility_score = 0.7 if model_size_gb < 30 else 0.4
        llama_cpp.notes = ["Best for CPU/consumer GPU", "Supports GGUF quantization"]
        if model_size_gb > 30:
            llama_cpp.notes.append("Large model - consider quantization")
        llama_cpp.required_conversions = ["Convert to GGUF format"]
        frameworks.append(llama_cpp)

        # 4. Triton Inference Server
        triton = FrameworkCompatibility(framework="Triton Inference Server")
        triton.compatible = is_onnx
        triton.compatibility_score = 0.75
        triton.notes = ["Enterprise-grade serving", "Supports multiple backends"]
        if is_onnx:
            triton.notes.append("ONNX backend ready")
            triton.compatibility_score = 0.8
        triton.required_conversions = ["Configure Triton model repository"]
        frameworks.append(triton)

        # 5. HuggingFace TGI
        tgi = FrameworkCompatibility(framework="HuggingFace TGI")
        tgi.compatible = True
        tgi.compatibility_score = 0.75
        tgi.notes = ["Easy deployment", "HuggingFace ecosystem integration"]
        tgi.required_conversions = ["Upload to HuggingFace Hub or use local model"]
        frameworks.append(tgi)

        # 6. ONNX Runtime
        ort = FrameworkCompatibility(framework="ONNX Runtime")
        ort.compatible = is_onnx
        ort.compatibility_score = 0.7 if is_onnx else 0.0
        ort.notes = ["Direct ONNX execution", "Cross-platform support"]
        if is_onnx:
            ort.notes.append("Already in ONNX format - no conversion needed")
            ort.required_conversions = []
        else:
            ort.required_conversions = ["Convert to ONNX"]
        frameworks.append(ort)

        return frameworks


# =============================================================================
# Convenience Function
# =============================================================================


def analyze_deployment(
    graph_info: GraphInfo,
    blocks: list[Block] | None = None,
    attention_result: AttentionAnalysisResult | None = None,
    memory_result: MemoryAnalysisResult | None = None,
    total_params: int = 0,
    total_flops: int = 0,
    target_gpu: str = "a100",
    vram_gb: float = 80.0,
    logger: logging.Logger | None = None,
) -> DeploymentAnalysisResult:
    """
    Convenience function to analyze LLM deployment characteristics.

    Args:
        graph_info: Graph analysis info.
        blocks: Pattern blocks from PatternAnalyzer.
        attention_result: Attention analysis from Epic 27.
        memory_result: Memory analysis from Epic 28.
        total_params: Total parameter count.
        total_flops: Total FLOPs count.
        target_gpu: Target GPU type (e.g., "a100", "h100").
        vram_gb: Target VRAM in GB.
        logger: Optional logger instance.

    Returns:
        DeploymentAnalysisResult with all findings.
    """
    analyzer = DeploymentAnalyzer(
        logger=logger,
        target_gpu=target_gpu,
        vram_gb=vram_gb,
    )
    return analyzer.analyze(
        graph_info,
        blocks,
        attention_result,
        memory_result,
        total_params,
        total_flops,
    )
