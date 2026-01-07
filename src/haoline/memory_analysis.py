# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
Memory Pattern Analysis for LLM-Scale Models (Epic 28).

This module provides comprehensive memory analysis for LLM deployment:

1. **KV Cache Analysis** (Story 28.1):
   - INT8 KV cache quantization detection
   - Max context length calculation for given VRAM
   - PagedAttention pattern detection (vLLM-style)
   - KV cache as percentage of total memory

2. **Parallelism Strategy Detection** (Story 28.2):
   - Tensor parallelism (column/row split)
   - Pipeline parallelism (layer sharding)
   - Data parallelism patterns
   - AllReduce/AllGather communication ops

3. **VRAM-Based Recommendations** (Story 28.3):
   - Batch size recommendations for given VRAM
   - Memory per GPU for N-way parallelism

Usage:
    from haoline.memory_analysis import MemoryAnalyzer

    analyzer = MemoryAnalyzer()
    result = analyzer.analyze(graph_info, blocks, attention_result, vram_gb=24.0)
    print(result.max_context_for_vram)  # 32768
    print(result.kv_cache_percent)  # 45.2
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from haoline.analyzer import GraphInfo
    from haoline.attention_analysis import AttentionAnalysisResult
    from haoline.patterns import Block


# =============================================================================
# Enums
# =============================================================================


class ParallelismType(Enum):
    """Types of parallelism strategies."""

    NONE = "none"  # Single GPU
    TENSOR_PARALLEL = "tensor_parallel"  # TP - split layers across GPUs
    PIPELINE_PARALLEL = "pipeline_parallel"  # PP - split model by layers
    DATA_PARALLEL = "data_parallel"  # DP - replicate model, split data
    HYBRID = "hybrid"  # Combination (e.g., TP + PP)
    EXPERT_PARALLEL = "expert_parallel"  # EP - MoE expert distribution
    UNKNOWN = "unknown"


class KVCacheQuantization(Enum):
    """KV cache precision types."""

    FP32 = "fp32"  # Full precision
    FP16 = "fp16"  # Half precision (default)
    BF16 = "bf16"  # Brain float 16
    INT8 = "int8"  # Quantized KV cache
    FP8 = "fp8"  # FP8 KV cache (H100+)
    UNKNOWN = "unknown"


class CommunicationOp(Enum):
    """Distributed communication operations."""

    ALL_REDUCE = "all_reduce"  # Sum across all ranks
    ALL_GATHER = "all_gather"  # Gather from all ranks
    REDUCE_SCATTER = "reduce_scatter"  # Reduce then scatter
    BROADCAST = "broadcast"  # Send from one to all
    SEND_RECV = "send_recv"  # Point-to-point
    ALL_TO_ALL = "all_to_all"  # Full exchange


# =============================================================================
# Data Models
# =============================================================================


class KVCacheAnalysis(BaseModel):
    """Extended KV cache analysis beyond Epic 27."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Quantization
    kv_quantization: str = "fp16"  # Detected KV cache precision
    kv_quantization_detected: bool = False
    int8_kv_supported: bool = False  # Can use INT8 KV cache

    # Memory breakdown
    bytes_per_token: int = 0
    kv_cache_percent_at_4k: float = 0.0  # KV cache as % of total at 4k context
    kv_cache_percent_at_8k: float = 0.0
    kv_cache_percent_at_32k: float = 0.0

    # VRAM-based limits
    max_context_for_vram: int = 0  # Max context length for given VRAM
    vram_gb_used: float = 0.0  # VRAM used for calculation

    # PagedAttention
    paged_attention_detected: bool = False
    paged_attention_indicators: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kv_quantization": self.kv_quantization,
            "kv_quantization_detected": self.kv_quantization_detected,
            "int8_kv_supported": self.int8_kv_supported,
            "bytes_per_token": self.bytes_per_token,
            "kv_cache_percent_at_4k": round(self.kv_cache_percent_at_4k, 1),
            "kv_cache_percent_at_8k": round(self.kv_cache_percent_at_8k, 1),
            "kv_cache_percent_at_32k": round(self.kv_cache_percent_at_32k, 1),
            "max_context_for_vram": self.max_context_for_vram,
            "vram_gb_used": self.vram_gb_used,
            "paged_attention_detected": self.paged_attention_detected,
            "paged_attention_indicators": self.paged_attention_indicators,
        }


class CommunicationOpInfo(BaseModel):
    """Information about a detected communication operation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    op_type: str  # all_reduce, all_gather, etc.
    node_name: str
    tensor_shape: list[int] = Field(default_factory=list)
    tensor_bytes: int = 0
    parallelism_hint: str = ""  # What parallelism this suggests

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_type": self.op_type,
            "node_name": self.node_name,
            "tensor_shape": self.tensor_shape,
            "tensor_bytes": self.tensor_bytes,
            "parallelism_hint": self.parallelism_hint,
        }


class ParallelismAnalysis(BaseModel):
    """Parallelism strategy detection results."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Detected strategy
    detected_type: str = "none"
    confidence: float = 0.0  # 0.0 - 1.0
    indicators: list[str] = Field(default_factory=list)

    # Communication ops found
    communication_ops: list[CommunicationOpInfo] = Field(default_factory=list)
    total_comm_bytes: int = 0

    # Tensor parallelism specifics
    tp_degree_estimate: int = 1  # Estimated TP degree
    column_parallel_layers: int = 0
    row_parallel_layers: int = 0

    # Pipeline parallelism specifics
    pp_degree_estimate: int = 1  # Estimated PP degree
    pipeline_stages_detected: int = 0

    # Memory per GPU estimates
    memory_per_gpu_gb: dict[str, float] = Field(default_factory=dict)  # {1: x, 2: y, 4: z, 8: w}

    def to_dict(self) -> dict[str, Any]:
        return {
            "detected_type": self.detected_type,
            "confidence": round(self.confidence, 2),
            "indicators": self.indicators,
            "communication_ops": [op.to_dict() for op in self.communication_ops],
            "total_comm_bytes": self.total_comm_bytes,
            "tp_degree_estimate": self.tp_degree_estimate,
            "column_parallel_layers": self.column_parallel_layers,
            "row_parallel_layers": self.row_parallel_layers,
            "pp_degree_estimate": self.pp_degree_estimate,
            "pipeline_stages_detected": self.pipeline_stages_detected,
            "memory_per_gpu_gb": {k: round(v, 2) for k, v in self.memory_per_gpu_gb.items()},
        }


class VRAMRecommendation(BaseModel):
    """VRAM-based deployment recommendations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Target VRAM
    target_vram_gb: float = 0.0

    # Batch size recommendations
    recommended_batch_size: int = 1
    max_batch_size: int = 1
    batch_size_breakdown: dict[int, float] = Field(default_factory=dict)  # {batch: vram_gb}

    # Context length recommendations
    recommended_context_length: int = 2048
    max_context_length: int = 0

    # Multi-GPU recommendations
    min_gpus_required: int = 1
    recommended_parallelism: str = "none"
    parallelism_rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_vram_gb": self.target_vram_gb,
            "recommended_batch_size": self.recommended_batch_size,
            "max_batch_size": self.max_batch_size,
            "batch_size_breakdown": {
                str(k): round(v, 2) for k, v in self.batch_size_breakdown.items()
            },
            "recommended_context_length": self.recommended_context_length,
            "max_context_length": self.max_context_length,
            "min_gpus_required": self.min_gpus_required,
            "recommended_parallelism": self.recommended_parallelism,
            "parallelism_rationale": self.parallelism_rationale,
        }


class MemoryAnalysisResult(BaseModel):
    """Complete memory pattern analysis result."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Model basics
    model_size_bytes: int = 0
    model_size_gb: float = 0.0

    # KV cache analysis (Story 28.1)
    kv_cache: KVCacheAnalysis = Field(default_factory=KVCacheAnalysis)

    # Parallelism analysis (Story 28.2)
    parallelism: ParallelismAnalysis = Field(default_factory=ParallelismAnalysis)

    # VRAM recommendations (Story 28.3)
    vram_recommendation: VRAMRecommendation = Field(default_factory=VRAMRecommendation)

    # Summary
    summary: str = ""
    recommendations: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_size_bytes": self.model_size_bytes,
            "model_size_gb": round(self.model_size_gb, 2),
            "kv_cache": self.kv_cache.to_dict(),
            "parallelism": self.parallelism.to_dict(),
            "vram_recommendation": self.vram_recommendation.to_dict(),
            "summary": self.summary,
            "recommendations": self.recommendations,
        }


# =============================================================================
# Communication Op Detection Patterns
# =============================================================================

# ONNX ops that indicate distributed communication
COMM_OP_PATTERNS: dict[str, CommunicationOp] = {
    # NCCL-style ops (common in exported models)
    "AllReduce": CommunicationOp.ALL_REDUCE,
    "NCCLAllReduce": CommunicationOp.ALL_REDUCE,
    "AllGather": CommunicationOp.ALL_GATHER,
    "NCCLAllGather": CommunicationOp.ALL_GATHER,
    "ReduceScatter": CommunicationOp.REDUCE_SCATTER,
    "NCCLReduceScatter": CommunicationOp.REDUCE_SCATTER,
    "Broadcast": CommunicationOp.BROADCAST,
    "NCCLBroadcast": CommunicationOp.BROADCAST,
    "AllToAll": CommunicationOp.ALL_TO_ALL,
    "NCCLAllToAll": CommunicationOp.ALL_TO_ALL,
    # Megatron-style custom ops
    "ColumnParallelLinear": CommunicationOp.ALL_REDUCE,
    "RowParallelLinear": CommunicationOp.ALL_REDUCE,
    # DeepSpeed ops
    "DeepSpeedAllReduce": CommunicationOp.ALL_REDUCE,
    "DeepSpeedAllGather": CommunicationOp.ALL_GATHER,
}

# Name patterns that hint at parallelism
PARALLELISM_NAME_PATTERNS = {
    "column_parallel": ParallelismType.TENSOR_PARALLEL,
    "row_parallel": ParallelismType.TENSOR_PARALLEL,
    "tensor_parallel": ParallelismType.TENSOR_PARALLEL,
    "tp_": ParallelismType.TENSOR_PARALLEL,
    "pipeline": ParallelismType.PIPELINE_PARALLEL,
    "pp_": ParallelismType.PIPELINE_PARALLEL,
    "stage_": ParallelismType.PIPELINE_PARALLEL,
    "expert_": ParallelismType.EXPERT_PARALLEL,
    "moe_": ParallelismType.EXPERT_PARALLEL,
}


# =============================================================================
# Memory Analyzer
# =============================================================================


class MemoryAnalyzer:
    """Analyzer for memory patterns in LLM models."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def analyze(
        self,
        graph_info: GraphInfo,
        blocks: list[Block] | None = None,
        attention_result: AttentionAnalysisResult | None = None,
        vram_gb: float = 24.0,
        batch_size: int = 1,
        context_length: int = 4096,
    ) -> MemoryAnalysisResult:
        """
        Analyze memory patterns in a model.

        Args:
            graph_info: Graph information from analyzer
            blocks: Detected architectural blocks
            attention_result: Results from AttentionAnalyzer (optional)
            vram_gb: Target VRAM in GB for recommendations
            batch_size: Batch size for memory calculations
            context_length: Context length for memory calculations

        Returns:
            MemoryAnalysisResult with complete analysis
        """
        blocks = blocks or []

        # Calculate model size
        model_size_bytes = self._calculate_model_size(graph_info)
        model_size_gb = model_size_bytes / (1024**3)

        # KV cache analysis (Story 28.1)
        kv_cache = self._analyze_kv_cache(
            graph_info,
            attention_result,
            model_size_bytes,
            vram_gb,
            context_length,
        )

        # Parallelism analysis (Story 28.2)
        parallelism = self._analyze_parallelism(graph_info, blocks, model_size_bytes)

        # VRAM recommendations (Story 28.3)
        vram_rec = self._generate_vram_recommendations(
            graph_info,
            model_size_bytes,
            kv_cache,
            parallelism,
            vram_gb,
            batch_size,
            context_length,
        )

        # Generate summary
        summary, recommendations = self._generate_summary(
            model_size_gb, kv_cache, parallelism, vram_rec, vram_gb
        )

        return MemoryAnalysisResult(
            model_size_bytes=model_size_bytes,
            model_size_gb=model_size_gb,
            kv_cache=kv_cache,
            parallelism=parallelism,
            vram_recommendation=vram_rec,
            summary=summary,
            recommendations=recommendations,
        )

    def _calculate_model_size(self, graph_info: GraphInfo) -> int:
        """Calculate total model size in bytes."""
        total = 0
        for tensor in graph_info.initializers.values():
            # Calculate bytes from shape and dtype
            elem_count = 1
            for dim in tensor.shape:
                elem_count *= dim if isinstance(dim, int) else 1

            dtype_bytes = self._dtype_to_bytes(tensor.data_type)
            total += elem_count * dtype_bytes

        return total

    def _dtype_to_bytes(self, dtype: int | str) -> int:
        """Convert dtype to bytes per element."""
        dtype_map = {
            1: 4,  # FLOAT
            2: 1,  # UINT8
            3: 1,  # INT8
            4: 2,  # UINT16
            5: 2,  # INT16
            6: 4,  # INT32
            7: 8,  # INT64
            10: 2,  # FLOAT16
            11: 8,  # DOUBLE
            12: 4,  # UINT32
            13: 8,  # UINT64
            14: 8,  # COMPLEX64
            15: 16,  # COMPLEX128
            16: 2,  # BFLOAT16
        }
        if isinstance(dtype, int):
            return dtype_map.get(dtype, 4)
        # String dtype
        dtype_str_map = {
            "float32": 4,
            "float": 4,
            "float16": 2,
            "half": 2,
            "bfloat16": 2,
            "int8": 1,
            "uint8": 1,
            "int16": 2,
            "uint16": 2,
            "int32": 4,
            "uint32": 4,
            "int64": 8,
            "uint64": 8,
        }
        return dtype_str_map.get(str(dtype).lower(), 4)

    def _analyze_kv_cache(
        self,
        graph_info: GraphInfo,
        attention_result: AttentionAnalysisResult | None,
        model_size_bytes: int,
        vram_gb: float,
        context_length: int,
    ) -> KVCacheAnalysis:
        """Analyze KV cache characteristics (Story 28.1)."""
        result = KVCacheAnalysis()

        # Get bytes_per_token from attention analysis if available
        bytes_per_token = 0
        if attention_result and attention_result.kv_cache_estimate:
            bytes_per_token = attention_result.kv_cache_estimate.bytes_per_token
            result.bytes_per_token = bytes_per_token

        # Detect KV cache quantization
        kv_quant, detected = self._detect_kv_quantization(graph_info)
        result.kv_quantization = kv_quant
        result.kv_quantization_detected = detected
        result.int8_kv_supported = self._check_int8_kv_support(graph_info)

        # Calculate KV cache as percentage of total memory at various context lengths
        if bytes_per_token > 0 and model_size_bytes > 0:
            for ctx_len, attr_name in [
                (4096, "kv_cache_percent_at_4k"),
                (8192, "kv_cache_percent_at_8k"),
                (32768, "kv_cache_percent_at_32k"),
            ]:
                kv_bytes = bytes_per_token * ctx_len
                total_mem = model_size_bytes + kv_bytes
                percent = (kv_bytes / total_mem) * 100
                setattr(result, attr_name, percent)

        # Calculate max context length for given VRAM
        result.vram_gb_used = vram_gb
        if bytes_per_token > 0:
            available_for_kv = (vram_gb * 1024**3) - model_size_bytes
            # Reserve 20% for activations and overhead
            available_for_kv *= 0.8
            if available_for_kv > 0:
                result.max_context_for_vram = int(available_for_kv / bytes_per_token)

        # Detect PagedAttention patterns
        paged, indicators = self._detect_paged_attention(graph_info)
        result.paged_attention_detected = paged
        result.paged_attention_indicators = indicators

        return result

    def _detect_kv_quantization(self, graph_info: GraphInfo) -> tuple[str, bool]:
        """Detect KV cache quantization patterns."""
        # Look for quantized KV cache patterns
        for node in graph_info.nodes:
            name_lower = node.name.lower()

            # Check for INT8 KV cache patterns
            if "kv" in name_lower or "cache" in name_lower:
                if "int8" in name_lower or "quantize" in name_lower:
                    return "int8", True
                if "fp8" in name_lower:
                    return "fp8", True

            # Check op types that indicate quantized KV
            if node.op_type in ("QuantizeLinear", "DequantizeLinear"):
                # If near attention ops, likely KV quantization
                if "kv" in name_lower or "key" in name_lower or "value" in name_lower:
                    return "int8", True

        # Check for FP16/BF16 hints
        fp16_count = graph_info.precision_breakdown.get("FP16", 0)
        bf16_count = graph_info.precision_breakdown.get("BF16", 0)

        if bf16_count > fp16_count:
            return "bf16", False
        elif fp16_count > 0:
            return "fp16", False

        return "fp16", False  # Default assumption

    def _check_int8_kv_support(self, graph_info: GraphInfo) -> bool:
        """Check if model architecture supports INT8 KV cache."""
        # Most modern LLM architectures support INT8 KV cache
        # Look for attention patterns
        attention_ops = {"Attention", "MultiHeadAttention", "Softmax"}
        has_attention = any(op in graph_info.op_type_counts for op in attention_ops)

        # Check for transformer patterns
        has_matmul = graph_info.op_type_counts.get("MatMul", 0) > 0
        has_layernorm = (
            graph_info.op_type_counts.get("LayerNormalization", 0) > 0
            or graph_info.op_type_counts.get("RMSNorm", 0) > 0
        )

        return has_attention or (has_matmul and has_layernorm)

    def _detect_paged_attention(self, graph_info: GraphInfo) -> tuple[bool, list[str]]:
        """Detect PagedAttention patterns (vLLM-style)."""
        indicators = []

        for node in graph_info.nodes:
            name_lower = node.name.lower()
            op_type_lower = node.op_type.lower()

            # Look for paged attention indicators
            if "paged" in name_lower or "paged" in op_type_lower:
                indicators.append(f"PagedAttention op: {node.name}")

            if "block_table" in name_lower:
                indicators.append(f"Block table: {node.name}")

            if "slot_mapping" in name_lower:
                indicators.append(f"Slot mapping: {node.name}")

            # vLLM specific patterns
            if "vllm" in name_lower:
                indicators.append(f"vLLM pattern: {node.name}")

            # TensorRT-LLM patterns
            if "inflight_batching" in name_lower:
                indicators.append(f"Inflight batching: {node.name}")

        return len(indicators) > 0, indicators

    def _analyze_parallelism(
        self,
        graph_info: GraphInfo,
        blocks: list[Block],
        model_size_bytes: int,
    ) -> ParallelismAnalysis:
        """Analyze parallelism patterns (Story 28.2)."""
        result = ParallelismAnalysis()
        indicators = []
        comm_ops = []
        total_comm_bytes = 0

        # Scan for communication ops
        for node in graph_info.nodes:
            # Check op type
            if node.op_type in COMM_OP_PATTERNS:
                comm_type = COMM_OP_PATTERNS[node.op_type]
                tensor_bytes = self._estimate_tensor_bytes(node, graph_info)
                comm_ops.append(
                    CommunicationOpInfo(
                        op_type=comm_type.value,
                        node_name=node.name,
                        tensor_bytes=tensor_bytes,
                        parallelism_hint=self._get_parallelism_hint(comm_type),
                    )
                )
                total_comm_bytes += tensor_bytes
                indicators.append(f"{node.op_type} op found: {node.name}")

            # Check name patterns
            name_lower = node.name.lower()
            for pattern, par_type in PARALLELISM_NAME_PATTERNS.items():
                if pattern in name_lower:
                    indicators.append(f"Name pattern '{pattern}': {node.name}")
                    if par_type == ParallelismType.TENSOR_PARALLEL:
                        if "column" in name_lower:
                            result.column_parallel_layers += 1
                        elif "row" in name_lower:
                            result.row_parallel_layers += 1

        result.communication_ops = comm_ops
        result.total_comm_bytes = total_comm_bytes
        result.indicators = indicators

        # Determine parallelism type and confidence
        if comm_ops:
            # Has explicit communication ops
            result.detected_type = self._infer_parallelism_type(comm_ops, result)
            result.confidence = 0.9
        elif indicators:
            # Has name hints
            result.detected_type = self._infer_from_indicators(indicators)
            result.confidence = 0.5
        else:
            # Check if model is too large for single GPU
            model_gb = model_size_bytes / (1024**3)
            if model_gb > 20:  # Likely needs parallelism
                result.detected_type = "likely_needed"
                result.confidence = 0.3
                indicators.append(f"Model size ({model_gb:.1f}GB) suggests parallelism needed")

        # Estimate memory per GPU for different parallelism degrees
        result.memory_per_gpu_gb = self._estimate_memory_per_gpu(model_size_bytes)

        return result

    def _estimate_tensor_bytes(self, node: Any, graph_info: GraphInfo) -> int:
        """Estimate bytes for a tensor operation."""
        # Try to get output shape
        for output in node.outputs:
            if output in graph_info.value_info:
                shape = graph_info.value_info[output]
                if shape:
                    elem_count = 1
                    for dim in shape:
                        elem_count *= dim if isinstance(dim, int) else 1
                    return elem_count * 4  # Assume FP32
        return 0

    def _get_parallelism_hint(self, comm_type: CommunicationOp) -> str:
        """Get parallelism hint for a communication op."""
        hints = {
            CommunicationOp.ALL_REDUCE: "Tensor Parallelism (gradient sync)",
            CommunicationOp.ALL_GATHER: "Tensor Parallelism (gather outputs)",
            CommunicationOp.REDUCE_SCATTER: "ZeRO / Tensor Parallelism",
            CommunicationOp.BROADCAST: "Pipeline Parallelism (stage communication)",
            CommunicationOp.ALL_TO_ALL: "Expert Parallelism (MoE)",
        }
        return hints.get(comm_type, "Unknown")

    def _infer_parallelism_type(
        self, comm_ops: list[CommunicationOpInfo], result: ParallelismAnalysis
    ) -> str:
        """Infer parallelism type from communication ops."""
        op_types = [op.op_type for op in comm_ops]

        if "all_to_all" in op_types:
            return ParallelismType.EXPERT_PARALLEL.value
        if "all_reduce" in op_types and "all_gather" in op_types:
            return ParallelismType.TENSOR_PARALLEL.value
        if "broadcast" in op_types:
            return ParallelismType.PIPELINE_PARALLEL.value
        if "all_reduce" in op_types:
            return ParallelismType.TENSOR_PARALLEL.value

        return ParallelismType.UNKNOWN.value

    def _infer_from_indicators(self, indicators: list[str]) -> str:
        """Infer parallelism type from name indicators."""
        indicator_text = " ".join(indicators).lower()

        if "tensor_parallel" in indicator_text or "tp_" in indicator_text:
            return ParallelismType.TENSOR_PARALLEL.value
        if "pipeline" in indicator_text or "pp_" in indicator_text:
            return ParallelismType.PIPELINE_PARALLEL.value
        if "expert" in indicator_text or "moe" in indicator_text:
            return ParallelismType.EXPERT_PARALLEL.value

        return ParallelismType.UNKNOWN.value

    def _estimate_memory_per_gpu(self, model_size_bytes: int) -> dict[str, float]:
        """Estimate memory per GPU for different parallelism degrees."""
        model_gb = model_size_bytes / (1024**3)
        return {
            "1": model_gb,
            "2": model_gb / 2 + 0.5,  # + communication overhead
            "4": model_gb / 4 + 0.5,
            "8": model_gb / 8 + 0.5,
        }

    def _generate_vram_recommendations(
        self,
        graph_info: GraphInfo,
        model_size_bytes: int,
        kv_cache: KVCacheAnalysis,
        parallelism: ParallelismAnalysis,
        vram_gb: float,
        batch_size: int,
        context_length: int,
    ) -> VRAMRecommendation:
        """Generate VRAM-based recommendations (Story 28.3)."""
        result = VRAMRecommendation(target_vram_gb=vram_gb)

        model_gb = model_size_bytes / (1024**3)

        # Calculate memory for different batch sizes
        batch_breakdown = {}
        max_batch = 1

        for bs in [1, 2, 4, 8, 16, 32, 64]:
            # Model weights + KV cache * batch + activations
            kv_bytes = (
                kv_cache.bytes_per_token * context_length * bs if kv_cache.bytes_per_token else 0
            )
            activation_bytes = model_size_bytes * 0.1 * bs  # Rough estimate
            total_bytes = model_size_bytes + kv_bytes + activation_bytes
            total_gb = total_bytes / (1024**3)

            batch_breakdown[bs] = total_gb

            if total_gb <= vram_gb * 0.9:  # Leave 10% headroom
                max_batch = bs

        result.batch_size_breakdown = batch_breakdown
        result.max_batch_size = max_batch
        result.recommended_batch_size = min(max_batch, 8)  # Don't go crazy

        # Context length recommendation
        if kv_cache.max_context_for_vram > 0:
            result.max_context_length = kv_cache.max_context_for_vram
            result.recommended_context_length = min(
                kv_cache.max_context_for_vram,
                8192,  # Reasonable default
            )
        else:
            result.recommended_context_length = 2048
            result.max_context_length = 4096

        # Multi-GPU recommendations
        if model_gb > vram_gb * 0.8:
            # Model doesn't fit in single GPU
            result.min_gpus_required = max(1, int(model_gb / (vram_gb * 0.7)) + 1)
            result.recommended_parallelism = "tensor_parallel"
            result.parallelism_rationale = (
                f"Model size ({model_gb:.1f}GB) exceeds {vram_gb}GB VRAM. "
                f"Recommend {result.min_gpus_required}-way tensor parallelism."
            )
        else:
            result.min_gpus_required = 1
            result.recommended_parallelism = "none"
            result.parallelism_rationale = f"Model fits in {vram_gb}GB VRAM."

        return result

    def _generate_summary(
        self,
        model_size_gb: float,
        kv_cache: KVCacheAnalysis,
        parallelism: ParallelismAnalysis,
        vram_rec: VRAMRecommendation,
        vram_gb: float,
    ) -> tuple[str, list[str]]:
        """Generate human-readable summary and recommendations."""
        recommendations = []

        # Model size summary
        if model_size_gb < 1:
            size_desc = "small"
        elif model_size_gb < 10:
            size_desc = "medium"
        elif model_size_gb < 50:
            size_desc = "large"
        else:
            size_desc = "very large"

        summary = f"Model size: {model_size_gb:.1f}GB ({size_desc})"

        # KV cache insights
        if kv_cache.bytes_per_token > 0:
            summary += f" | KV cache: {kv_cache.bytes_per_token:,} bytes/token"

            if kv_cache.kv_cache_percent_at_32k > 50:
                recommendations.append(
                    f"KV cache dominates memory at 32k context ({kv_cache.kv_cache_percent_at_32k:.0f}%). "
                    "Consider INT8 KV cache quantization."
                )

            if kv_cache.int8_kv_supported and not kv_cache.kv_quantization_detected:
                recommendations.append(
                    "INT8 KV cache supported but not detected. "
                    "Enable for 2x context length or 50% VRAM savings."
                )

        if kv_cache.paged_attention_detected:
            summary += " | PagedAttention: detected"
        else:
            if model_size_gb > 5:  # Only for larger models
                recommendations.append(
                    "Consider vLLM or TensorRT-LLM for PagedAttention support "
                    "(better memory efficiency for variable-length sequences)."
                )

        # Parallelism insights
        if parallelism.detected_type != "none":
            summary += f" | Parallelism: {parallelism.detected_type}"

        # VRAM recommendations
        if vram_rec.min_gpus_required > 1:
            recommendations.append(vram_rec.parallelism_rationale)
        elif model_size_gb > vram_gb * 0.6:
            recommendations.append(
                f"Model uses {(model_size_gb / vram_gb) * 100:.0f}% of VRAM. "
                "Limited headroom for long contexts or large batches."
            )

        if vram_rec.max_batch_size < 4 and model_size_gb > 5:
            recommendations.append(
                f"Max batch size limited to {vram_rec.max_batch_size} for {vram_gb}GB VRAM. "
                "Consider quantization or larger GPU for higher throughput."
            )

        return summary, recommendations
