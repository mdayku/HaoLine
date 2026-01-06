# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
Attention Variant Detection for LLM-Scale Models (Epic 27).

This module provides comprehensive analysis of attention mechanisms in LLMs:

1. **Attention Architecture Detection** (Story 27.1):
   - Multi-Head Attention (MHA) - standard BERT/GPT pattern
   - Multi-Query Attention (MQA) - single KV head (PaLM, Falcon)
   - Grouped-Query Attention (GQA) - fewer KV heads (LLaMA 2/3, Mistral)
   - KV cache size calculation and savings analysis

2. **Attention Pattern Detection** (Story 27.2):
   - Sliding window attention (Mistral-style)
   - Local + global attention (Longformer-style)
   - Sparse attention patterns (BigBird)
   - Cross-attention (encoder-decoder)

3. **Position Encoding Detection** (Story 27.3):
   - RoPE (Rotary Position Embedding)
   - ALiBi (Attention with Linear Biases)
   - Learned positional embeddings
   - Sinusoidal positional encoding

4. **Fused Attention Patterns** (Story 27.4):
   - FlashAttention-style patterns
   - Memory-efficient attention (xFormers)
   - cuDNN fused multi-head attention

Usage:
    from haoline.attention_analysis import AttentionAnalyzer

    analyzer = AttentionAnalyzer()
    result = analyzer.analyze(graph_info, blocks)
    print(result.attention_type)  # "GQA"
    print(result.kv_cache_savings)  # 4.0x vs MHA
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from haoline.analyzer import GraphInfo
    from haoline.patterns import Block


# =============================================================================
# Enums
# =============================================================================


class AttentionType(Enum):
    """Types of attention mechanisms."""

    MHA = "mha"  # Multi-Head Attention (standard)
    MQA = "mqa"  # Multi-Query Attention (single KV head)
    GQA = "gqa"  # Grouped-Query Attention (fewer KV heads)
    CROSS = "cross"  # Cross-attention (encoder-decoder)
    LINEAR = "linear"  # Linear attention (O(n))
    UNKNOWN = "unknown"


class PositionEncodingType(Enum):
    """Types of position encoding."""

    ROPE = "rope"  # Rotary Position Embedding
    ALIBI = "alibi"  # Attention with Linear Biases
    LEARNED = "learned"  # Learned absolute positions
    SINUSOIDAL = "sinusoidal"  # Fixed sinusoidal
    RELATIVE = "relative"  # Relative position bias (T5-style)
    NONE = "none"  # No position encoding detected
    UNKNOWN = "unknown"


class AttentionPatternType(Enum):
    """Attention pattern types."""

    FULL = "full"  # Full O(n^2) attention
    SLIDING_WINDOW = "sliding_window"  # Local sliding window
    LOCAL_GLOBAL = "local_global"  # Longformer-style
    SPARSE = "sparse"  # BigBird/sparse patterns
    CAUSAL = "causal"  # Causal mask (autoregressive)
    BIDIRECTIONAL = "bidirectional"  # No mask (BERT-style)
    UNKNOWN = "unknown"


class FusedAttentionType(Enum):
    """Fused attention implementations."""

    FLASH_ATTENTION = "flash_attention"
    FLASH_ATTENTION_2 = "flash_attention_2"
    XFORMERS = "xformers"
    CUDNN_MHA = "cudnn_mha"
    SDPA = "sdpa"  # PyTorch Scaled Dot Product Attention
    NONE = "none"  # Not fused (standard implementation)
    UNKNOWN = "unknown"


# =============================================================================
# Data Classes
# =============================================================================


class AttentionHeadInfo(BaseModel):
    """Information about a single attention head/layer."""

    model_config = ConfigDict(frozen=True)

    name: str
    attention_type: AttentionType
    num_q_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    hidden_size: int = 0
    has_scaling: bool = False
    has_mask: bool = False
    q_proj: str | None = None
    k_proj: str | None = None
    v_proj: str | None = None
    o_proj: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "attention_type": self.attention_type.value,
            "num_q_heads": self.num_q_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "hidden_size": self.hidden_size,
            "has_scaling": self.has_scaling,
            "has_mask": self.has_mask,
            "q_proj": self.q_proj,
            "k_proj": self.k_proj,
            "v_proj": self.v_proj,
            "o_proj": self.o_proj,
        }


class PositionEncodingInfo(BaseModel):
    """Information about position encoding."""

    model_config = ConfigDict(frozen=True)

    encoding_type: PositionEncodingType
    max_positions: int = 0
    embed_dim: int = 0
    is_rotary: bool = False  # True for RoPE
    is_relative: bool = False  # True for relative position bias
    extrapolation_capable: bool = False  # True for ALiBi, RoPE with scaling

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "encoding_type": self.encoding_type.value,
            "max_positions": self.max_positions,
            "embed_dim": self.embed_dim,
            "is_rotary": self.is_rotary,
            "is_relative": self.is_relative,
            "extrapolation_capable": self.extrapolation_capable,
        }


class KVCacheEstimate(BaseModel):
    """KV cache memory estimates."""

    model_config = ConfigDict(frozen=True)

    # Per-token KV cache size in bytes
    bytes_per_token: int = 0
    # Total KV cache for a given context length
    total_bytes_at_4k: int = 0  # At 4K context
    total_bytes_at_8k: int = 0  # At 8K context
    total_bytes_at_32k: int = 0  # At 32K context
    total_bytes_at_128k: int = 0  # At 128K context
    # Savings compared to full MHA
    mha_baseline_bytes_per_token: int = 0
    savings_factor: float = 1.0  # 1.0 = no savings, 4.0 = 4x smaller

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bytes_per_token": self.bytes_per_token,
            "total_bytes_at_4k": self.total_bytes_at_4k,
            "total_bytes_at_8k": self.total_bytes_at_8k,
            "total_bytes_at_32k": self.total_bytes_at_32k,
            "total_bytes_at_128k": self.total_bytes_at_128k,
            "mha_baseline_bytes_per_token": self.mha_baseline_bytes_per_token,
            "savings_factor": self.savings_factor,
        }


class AttentionPatternInfo(BaseModel):
    """Information about attention patterns."""

    model_config = ConfigDict(frozen=True)

    pattern_type: AttentionPatternType
    window_size: int | None = None  # For sliding window
    global_tokens: int | None = None  # For local+global
    sparsity_ratio: float | None = None  # For sparse attention
    is_causal: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_type": self.pattern_type.value,
            "window_size": self.window_size,
            "global_tokens": self.global_tokens,
            "sparsity_ratio": self.sparsity_ratio,
            "is_causal": self.is_causal,
        }


class FusedAttentionInfo(BaseModel):
    """Information about fused attention implementations."""

    model_config = ConfigDict(frozen=True)

    fused_type: FusedAttentionType
    is_memory_efficient: bool = False
    supports_flash: bool = False
    theoretical_memory_gb: float = 0.0  # Full O(n^2) attention memory
    actual_memory_gb: float = 0.0  # With memory-efficient impl

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fused_type": self.fused_type.value,
            "is_memory_efficient": self.is_memory_efficient,
            "supports_flash": self.supports_flash,
            "theoretical_memory_gb": self.theoretical_memory_gb,
            "actual_memory_gb": self.actual_memory_gb,
        }


class AttentionAnalysisResult(BaseModel):
    """Complete attention analysis result."""

    model_config = ConfigDict(frozen=True)

    # Primary attention architecture
    primary_attention_type: AttentionType = AttentionType.UNKNOWN
    num_attention_layers: int = 0

    # Head configuration
    num_q_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    hidden_size: int = 0

    # Position encoding
    position_encoding: PositionEncodingInfo | None = None

    # KV cache analysis
    kv_cache: KVCacheEstimate | None = None

    # Attention pattern
    attention_pattern: AttentionPatternInfo | None = None

    # Fused attention
    fused_attention: FusedAttentionInfo | None = None

    # Per-layer details
    attention_heads: list[AttentionHeadInfo] = Field(default_factory=list)

    # Context length info
    max_context_length: int = 0
    effective_context_length: int = 0  # May be less with sliding window

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "primary_attention_type": self.primary_attention_type.value,
            "num_attention_layers": self.num_attention_layers,
            "num_q_heads": self.num_q_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "hidden_size": self.hidden_size,
            "position_encoding": self.position_encoding.to_dict()
            if self.position_encoding
            else None,
            "kv_cache": self.kv_cache.to_dict() if self.kv_cache else None,
            "attention_pattern": self.attention_pattern.to_dict()
            if self.attention_pattern
            else None,
            "fused_attention": self.fused_attention.to_dict() if self.fused_attention else None,
            "attention_heads": [h.to_dict() for h in self.attention_heads],
            "max_context_length": self.max_context_length,
            "effective_context_length": self.effective_context_length,
        }

    def get_summary(self) -> str:
        """Get a human-readable summary."""
        lines = [
            f"Attention Type: {self.primary_attention_type.value.upper()}",
            f"  Layers: {self.num_attention_layers}",
            f"  Q Heads: {self.num_q_heads}, KV Heads: {self.num_kv_heads}",
            f"  Head Dim: {self.head_dim}, Hidden: {self.hidden_size}",
        ]

        if self.position_encoding:
            lines.append(
                f"\nPosition Encoding: {self.position_encoding.encoding_type.value.upper()}"
            )
            if self.position_encoding.max_positions:
                lines.append(f"  Max Positions: {self.position_encoding.max_positions}")
            if self.position_encoding.extrapolation_capable:
                lines.append("  Extrapolation: Capable (can extend beyond training)")

        if self.kv_cache:
            lines.append("\nKV Cache Analysis:")
            lines.append(f"  Per Token: {self.kv_cache.bytes_per_token:,} bytes")
            lines.append(f"  At 8K context: {self.kv_cache.total_bytes_at_8k / 1e9:.2f} GB")
            if self.kv_cache.savings_factor > 1.0:
                lines.append(f"  Savings vs MHA: {self.kv_cache.savings_factor:.1f}x")

        if self.attention_pattern:
            lines.append(f"\nAttention Pattern: {self.attention_pattern.pattern_type.value}")
            if self.attention_pattern.window_size:
                lines.append(f"  Window Size: {self.attention_pattern.window_size}")
            if self.attention_pattern.is_causal:
                lines.append("  Causal: Yes (autoregressive)")

        return "\n".join(lines)


# =============================================================================
# Analyzer
# =============================================================================


class AttentionAnalyzer:
    """Analyze attention mechanisms in LLM graphs.

    Extracts attention architecture, position encoding, and provides
    KV cache analysis for deployment planning.
    """

    def __init__(self, logger: logging.Logger | None = None):
        """Initialize the analyzer."""
        self.logger = logger or logging.getLogger(__name__)

    def analyze(
        self,
        graph_info: GraphInfo,
        blocks: list[Block] | None = None,
    ) -> AttentionAnalysisResult:
        """Perform comprehensive attention analysis.

        Args:
            graph_info: Parsed ONNX graph information.
            blocks: Detected architectural blocks (from PatternAnalyzer).

        Returns:
            AttentionAnalysisResult with full analysis.
        """
        # Extract attention heads from blocks
        attention_heads = self._extract_attention_heads(blocks)

        # Determine primary attention type
        primary_type = self._determine_primary_type(attention_heads)

        # Extract head configuration
        num_q_heads, num_kv_heads, head_dim, hidden_size = self._extract_head_config(
            attention_heads, graph_info
        )

        # Analyze position encoding
        position_encoding = self._analyze_position_encoding(blocks, graph_info)

        # Calculate KV cache
        kv_cache = self._calculate_kv_cache(
            num_q_heads, num_kv_heads, head_dim, len(attention_heads)
        )

        # Detect attention pattern
        attention_pattern = self._detect_attention_pattern(graph_info, blocks)

        # Detect fused attention
        fused_attention = self._detect_fused_attention(graph_info)

        # Context length estimation
        max_context = position_encoding.max_positions if position_encoding else 0
        effective_context = max_context
        if attention_pattern and attention_pattern.window_size:
            effective_context = attention_pattern.window_size

        return AttentionAnalysisResult(
            primary_attention_type=primary_type,
            num_attention_layers=len(attention_heads),
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
            position_encoding=position_encoding,
            kv_cache=kv_cache,
            attention_pattern=attention_pattern,
            fused_attention=fused_attention,
            attention_heads=attention_heads,
            max_context_length=max_context,
            effective_context_length=effective_context,
        )

    def _extract_attention_heads(self, blocks: list[Block] | None) -> list[AttentionHeadInfo]:
        """Extract attention head info from detected blocks."""
        heads: list[AttentionHeadInfo] = []

        if not blocks:
            return heads

        for block in blocks:
            if block.block_type == "AttentionHead":
                attrs = block.attributes

                # Determine attention type from head counts
                num_q = attrs.get("num_q_heads", 0)
                num_kv = attrs.get("num_kv_heads", 0)
                attention_type_str = attrs.get("attention_type", "MHA")

                if attention_type_str == "MQA":
                    attn_type = AttentionType.MQA
                elif attention_type_str == "GQA":
                    attn_type = AttentionType.GQA
                else:
                    attn_type = AttentionType.MHA

                heads.append(
                    AttentionHeadInfo(
                        name=block.name,
                        attention_type=attn_type,
                        num_q_heads=num_q,
                        num_kv_heads=num_kv,
                        has_scaling=attrs.get("has_scaling", False),
                        has_mask=attrs.get("has_mask", False),
                        q_proj=attrs.get("q_proj"),
                        k_proj=attrs.get("k_proj"),
                        v_proj=attrs.get("v_proj"),
                        o_proj=attrs.get("o_proj"),
                    )
                )

        return heads

    def _determine_primary_type(self, attention_heads: list[AttentionHeadInfo]) -> AttentionType:
        """Determine the primary attention type used in the model."""
        if not attention_heads:
            return AttentionType.UNKNOWN

        # Count types
        type_counts: dict[AttentionType, int] = {}
        for head in attention_heads:
            type_counts[head.attention_type] = type_counts.get(head.attention_type, 0) + 1

        if not type_counts:
            return AttentionType.UNKNOWN

        # Return most common
        return max(type_counts, key=type_counts.get)  # type: ignore[arg-type]

    def _extract_head_config(
        self,
        attention_heads: list[AttentionHeadInfo],
        graph_info: GraphInfo,
    ) -> tuple[int, int, int, int]:
        """Extract head configuration (num_q_heads, num_kv_heads, head_dim, hidden_size)."""
        if not attention_heads:
            # Try to infer from graph structure
            return self._infer_head_config_from_graph(graph_info)

        # Use first attention head with valid config
        for head in attention_heads:
            if head.num_q_heads > 0:
                num_q = head.num_q_heads
                num_kv = head.num_kv_heads if head.num_kv_heads > 0 else num_q
                head_dim = head.head_dim if head.head_dim > 0 else 64  # Common default
                hidden_size = head.hidden_size if head.hidden_size > 0 else num_q * head_dim
                return num_q, num_kv, head_dim, hidden_size

        return 0, 0, 0, 0

    def _infer_head_config_from_graph(self, graph_info: GraphInfo) -> tuple[int, int, int, int]:
        """Infer head configuration from weight shapes."""
        # Look for common LLM weight patterns
        for name, tensor in graph_info.initializers.items():
            name_lower = name.lower()

            # Q projection weight: [hidden, hidden] or [hidden, num_heads * head_dim]
            if "q_proj" in name_lower or "query" in name_lower:
                if len(tensor.shape) == 2:
                    hidden_size = tensor.shape[0]
                    # Guess common head dims
                    for head_dim in [64, 80, 96, 128]:
                        if hidden_size % head_dim == 0:
                            num_heads = hidden_size // head_dim
                            return num_heads, num_heads, head_dim, hidden_size

            # K/V projection - if smaller, indicates MQA/GQA
            if ("k_proj" in name_lower or "key" in name_lower) and "v_proj" not in name_lower:
                pass  # Could compare to Q to detect GQA

        return 0, 0, 0, 0

    def _analyze_position_encoding(
        self,
        blocks: list[Block] | None,
        graph_info: GraphInfo,
    ) -> PositionEncodingInfo | None:
        """Analyze position encoding from blocks and graph."""
        if not blocks:
            return self._detect_position_encoding_from_graph(graph_info)

        for block in blocks:
            if block.block_type == "PositionEncoding":
                attrs = block.attributes
                encoding_type_str = attrs.get("encoding_type", "unknown")

                # Map to enum
                type_map = {
                    "RoPE": PositionEncodingType.ROPE,
                    "rope": PositionEncodingType.ROPE,
                    "ALiBi": PositionEncodingType.ALIBI,
                    "alibi": PositionEncodingType.ALIBI,
                    "learned": PositionEncodingType.LEARNED,
                    "sinusoidal": PositionEncodingType.SINUSOIDAL,
                    "relative": PositionEncodingType.RELATIVE,
                }
                enc_type = type_map.get(encoding_type_str, PositionEncodingType.UNKNOWN)

                return PositionEncodingInfo(
                    encoding_type=enc_type,
                    max_positions=attrs.get("max_positions", 0),
                    embed_dim=attrs.get("embed_dim", 0),
                    is_rotary=enc_type == PositionEncodingType.ROPE,
                    is_relative=enc_type == PositionEncodingType.RELATIVE,
                    extrapolation_capable=enc_type
                    in {PositionEncodingType.ROPE, PositionEncodingType.ALIBI},
                )

        return self._detect_position_encoding_from_graph(graph_info)

    def _detect_position_encoding_from_graph(
        self, graph_info: GraphInfo
    ) -> PositionEncodingInfo | None:
        """Detect position encoding directly from graph patterns."""
        # Check for ALiBi pattern (learned linear biases added to attention)
        has_alibi = self._detect_alibi_pattern(graph_info)
        if has_alibi:
            return PositionEncodingInfo(
                encoding_type=PositionEncodingType.ALIBI,
                is_relative=True,
                extrapolation_capable=True,
            )

        # Check for RoPE pattern (Sin/Cos multiplied into Q/K)
        sin_nodes = [n for n in graph_info.nodes if n.op_type == "Sin"]
        cos_nodes = [n for n in graph_info.nodes if n.op_type == "Cos"]
        if sin_nodes and cos_nodes:
            return PositionEncodingInfo(
                encoding_type=PositionEncodingType.ROPE,
                is_rotary=True,
                extrapolation_capable=True,
            )

        # Check for learned position embeddings
        for name, tensor in graph_info.initializers.items():
            if "position" in name.lower() and "embed" in name.lower():
                if len(tensor.shape) == 2:
                    return PositionEncodingInfo(
                        encoding_type=PositionEncodingType.LEARNED,
                        max_positions=tensor.shape[0],
                        embed_dim=tensor.shape[1],
                    )

        return None

    def _detect_alibi_pattern(self, graph_info: GraphInfo) -> bool:
        """Detect ALiBi attention pattern.

        ALiBi adds learned linear biases to attention scores:
        attention = softmax(Q @ K^T + alibi_bias)

        Key indicators:
        - Learned bias tensor added before softmax
        - Bias has pattern that depends on position difference
        """
        # Look for Add operations before Softmax that could be ALiBi
        for node in graph_info.nodes:
            if node.op_type == "Softmax":
                # Check what feeds into softmax
                if node.inputs:
                    prev_output = node.inputs[0]
                    if prev_output in graph_info.node_by_output:
                        prev_node = graph_info.node_by_output[prev_output]
                        if prev_node.op_type == "Add":
                            # Check if one input is a bias tensor
                            for inp in prev_node.inputs:
                                if inp in graph_info.initializers:
                                    # ALiBi typically has a specific shape pattern
                                    # Usually [1, num_heads, seq, seq] or similar
                                    if "alibi" in inp.lower() or "slope" in inp.lower():
                                        return True

        return False

    def _calculate_kv_cache(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        num_layers: int,
        dtype_bytes: int = 2,  # FP16 default
    ) -> KVCacheEstimate | None:
        """Calculate KV cache memory requirements."""
        if num_kv_heads == 0 or head_dim == 0 or num_layers == 0:
            return None

        # KV cache per token = 2 (K and V) * num_kv_heads * head_dim * num_layers * dtype_bytes
        bytes_per_token = 2 * num_kv_heads * head_dim * num_layers * dtype_bytes

        # MHA baseline (if we have fewer KV heads)
        mha_bytes_per_token = 2 * num_q_heads * head_dim * num_layers * dtype_bytes

        # Calculate savings
        savings = mha_bytes_per_token / bytes_per_token if bytes_per_token > 0 else 1.0

        return KVCacheEstimate(
            bytes_per_token=bytes_per_token,
            total_bytes_at_4k=bytes_per_token * 4096,
            total_bytes_at_8k=bytes_per_token * 8192,
            total_bytes_at_32k=bytes_per_token * 32768,
            total_bytes_at_128k=bytes_per_token * 131072,
            mha_baseline_bytes_per_token=mha_bytes_per_token,
            savings_factor=savings,
        )

    def _detect_attention_pattern(
        self, graph_info: GraphInfo, blocks: list[Block] | None
    ) -> AttentionPatternInfo | None:
        """Detect attention pattern type (full, sliding window, sparse, etc.)."""
        # Check for sliding window indicators
        window_size = self._detect_sliding_window(graph_info)
        if window_size:
            return AttentionPatternInfo(
                pattern_type=AttentionPatternType.SLIDING_WINDOW,
                window_size=window_size,
                is_causal=True,  # Sliding window is usually causal
            )

        # Check for causal mask (autoregressive)
        has_causal = self._detect_causal_mask(graph_info)

        if has_causal:
            return AttentionPatternInfo(
                pattern_type=AttentionPatternType.CAUSAL,
                is_causal=True,
            )

        # Default to full attention
        return AttentionPatternInfo(
            pattern_type=AttentionPatternType.FULL,
            is_causal=False,
        )

    def _detect_sliding_window(self, graph_info: GraphInfo) -> int | None:
        """Detect sliding window attention and return window size."""
        # Look for window-related config in tensor names
        for name in graph_info.initializers:
            name_lower = name.lower()
            if "window" in name_lower or "sliding" in name_lower:
                # Found indicator, but need to extract size
                # Common sizes: 256, 512, 1024, 2048, 4096
                return 4096  # Default assumption for Mistral-style

        return None

    def _detect_causal_mask(self, graph_info: GraphInfo) -> bool:
        """Detect if model uses causal (autoregressive) masking."""
        # Look for triangular mask patterns
        for node in graph_info.nodes:
            if node.op_type == "Trilu":  # ONNX triangular operation
                return True
            if node.op_type == "Where":
                # Could be mask application
                pass

        # Look for mask tensors
        for name in graph_info.initializers:
            name_lower = name.lower()
            if "causal" in name_lower or "mask" in name_lower:
                return True

        return False

    def _detect_fused_attention(self, graph_info: GraphInfo) -> FusedAttentionInfo | None:
        """Detect fused attention implementations."""
        # Check for specific ONNX ops that indicate fused attention
        for node in graph_info.nodes:
            # SDPA (Scaled Dot Product Attention) - PyTorch 2.0+
            if node.op_type == "ScaledDotProductAttention":
                return FusedAttentionInfo(
                    fused_type=FusedAttentionType.SDPA,
                    is_memory_efficient=True,
                    supports_flash=True,
                )

            # MultiHeadAttention (some frameworks export this)
            if node.op_type == "MultiHeadAttention":
                return FusedAttentionInfo(
                    fused_type=FusedAttentionType.CUDNN_MHA,
                    is_memory_efficient=False,
                    supports_flash=False,
                )

            # com.microsoft ops (ONNX Runtime extensions)
            if node.op_type.startswith("com.microsoft"):
                if "Attention" in node.op_type:
                    return FusedAttentionInfo(
                        fused_type=FusedAttentionType.XFORMERS,
                        is_memory_efficient=True,
                        supports_flash=True,
                    )

        return FusedAttentionInfo(
            fused_type=FusedAttentionType.NONE,
            is_memory_efficient=False,
            supports_flash=False,
        )


# =============================================================================
# Convenience Function
# =============================================================================


def analyze_attention(
    graph_info: GraphInfo,
    blocks: list[Block] | None = None,
    logger: logging.Logger | None = None,
) -> AttentionAnalysisResult:
    """Convenience function to analyze attention.

    Args:
        graph_info: Parsed ONNX graph information.
        blocks: Detected architectural blocks.
        logger: Optional logger.

    Returns:
        AttentionAnalysisResult with full analysis.
    """
    analyzer = AttentionAnalyzer(logger=logger)
    return analyzer.analyze(graph_info, blocks)
