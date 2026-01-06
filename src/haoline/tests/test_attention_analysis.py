# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""Tests for Attention Variant Detection (Epic 27)."""

from __future__ import annotations

import pytest

from haoline.attention_analysis import (
    AttentionAnalysisResult,
    AttentionAnalyzer,
    AttentionHeadInfo,
    AttentionPatternInfo,
    AttentionPatternType,
    AttentionType,
    FusedAttentionInfo,
    FusedAttentionType,
    KVCacheEstimate,
    PositionEncodingInfo,
    PositionEncodingType,
)

# =============================================================================
# Test Data Classes
# =============================================================================


class TestAttentionType:
    """Test AttentionType enum."""

    def test_all_types_defined(self) -> None:
        """All attention types should be defined."""
        assert AttentionType.MHA.value == "mha"
        assert AttentionType.MQA.value == "mqa"
        assert AttentionType.GQA.value == "gqa"
        assert AttentionType.CROSS.value == "cross"
        assert AttentionType.UNKNOWN.value == "unknown"


class TestPositionEncodingType:
    """Test PositionEncodingType enum."""

    def test_all_types_defined(self) -> None:
        """All position encoding types should be defined."""
        assert PositionEncodingType.ROPE.value == "rope"
        assert PositionEncodingType.ALIBI.value == "alibi"
        assert PositionEncodingType.LEARNED.value == "learned"
        assert PositionEncodingType.SINUSOIDAL.value == "sinusoidal"


class TestAttentionHeadInfo:
    """Test AttentionHeadInfo dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic creation."""
        head = AttentionHeadInfo(
            name="layer_0_attention",
            attention_type=AttentionType.GQA,
            num_q_heads=32,
            num_kv_heads=8,
            head_dim=128,
            hidden_size=4096,
        )
        assert head.name == "layer_0_attention"
        assert head.attention_type == AttentionType.GQA
        assert head.num_q_heads == 32
        assert head.num_kv_heads == 8

    def test_to_dict(self) -> None:
        """Test serialization."""
        head = AttentionHeadInfo(
            name="test",
            attention_type=AttentionType.MHA,
            num_q_heads=8,
            num_kv_heads=8,
        )
        d = head.to_dict()
        assert d["attention_type"] == "mha"
        assert d["num_q_heads"] == 8


class TestPositionEncodingInfo:
    """Test PositionEncodingInfo dataclass."""

    def test_rope_encoding(self) -> None:
        """Test RoPE encoding info."""
        pe = PositionEncodingInfo(
            encoding_type=PositionEncodingType.ROPE,
            max_positions=4096,
            is_rotary=True,
            extrapolation_capable=True,
        )
        assert pe.encoding_type == PositionEncodingType.ROPE
        assert pe.is_rotary
        assert pe.extrapolation_capable

    def test_learned_encoding(self) -> None:
        """Test learned encoding info."""
        pe = PositionEncodingInfo(
            encoding_type=PositionEncodingType.LEARNED,
            max_positions=512,
            embed_dim=768,
        )
        assert pe.max_positions == 512
        assert not pe.extrapolation_capable


class TestKVCacheEstimate:
    """Test KVCacheEstimate dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic creation."""
        kv = KVCacheEstimate(
            bytes_per_token=2048,
            total_bytes_at_4k=2048 * 4096,
            total_bytes_at_8k=2048 * 8192,
            total_bytes_at_32k=2048 * 32768,
            total_bytes_at_128k=2048 * 131072,
            mha_baseline_bytes_per_token=8192,
            savings_factor=4.0,
        )
        assert kv.bytes_per_token == 2048
        assert kv.savings_factor == 4.0

    def test_to_dict(self) -> None:
        """Test serialization."""
        kv = KVCacheEstimate(
            bytes_per_token=1024,
            total_bytes_at_8k=1024 * 8192,
            savings_factor=2.0,
        )
        d = kv.to_dict()
        assert d["bytes_per_token"] == 1024
        assert d["savings_factor"] == 2.0


class TestAttentionPatternInfo:
    """Test AttentionPatternInfo dataclass."""

    def test_sliding_window(self) -> None:
        """Test sliding window pattern."""
        pattern = AttentionPatternInfo(
            pattern_type=AttentionPatternType.SLIDING_WINDOW,
            window_size=4096,
            is_causal=True,
        )
        assert pattern.pattern_type == AttentionPatternType.SLIDING_WINDOW
        assert pattern.window_size == 4096

    def test_full_attention(self) -> None:
        """Test full attention pattern."""
        pattern = AttentionPatternInfo(
            pattern_type=AttentionPatternType.FULL,
            is_causal=False,
        )
        assert pattern.pattern_type == AttentionPatternType.FULL


class TestFusedAttentionInfo:
    """Test FusedAttentionInfo dataclass."""

    def test_flash_attention(self) -> None:
        """Test FlashAttention info."""
        fused = FusedAttentionInfo(
            fused_type=FusedAttentionType.FLASH_ATTENTION_2,
            is_memory_efficient=True,
            supports_flash=True,
        )
        assert fused.fused_type == FusedAttentionType.FLASH_ATTENTION_2
        assert fused.is_memory_efficient


class TestAttentionAnalysisResult:
    """Test AttentionAnalysisResult dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic creation."""
        result = AttentionAnalysisResult(
            primary_attention_type=AttentionType.GQA,
            num_attention_layers=32,
            num_q_heads=32,
            num_kv_heads=8,
            head_dim=128,
            hidden_size=4096,
        )
        assert result.primary_attention_type == AttentionType.GQA
        assert result.num_attention_layers == 32

    def test_to_dict(self) -> None:
        """Test serialization."""
        result = AttentionAnalysisResult(
            primary_attention_type=AttentionType.MHA,
            num_attention_layers=12,
        )
        d = result.to_dict()
        assert d["primary_attention_type"] == "mha"
        assert d["num_attention_layers"] == 12

    def test_get_summary(self) -> None:
        """Test summary generation."""
        result = AttentionAnalysisResult(
            primary_attention_type=AttentionType.GQA,
            num_attention_layers=32,
            num_q_heads=32,
            num_kv_heads=8,
            head_dim=128,
            hidden_size=4096,
            position_encoding=PositionEncodingInfo(
                encoding_type=PositionEncodingType.ROPE,
                max_positions=8192,
                extrapolation_capable=True,
            ),
            kv_cache=KVCacheEstimate(
                bytes_per_token=1024,
                total_bytes_at_8k=1024 * 8192,
                savings_factor=4.0,
            ),
        )
        summary = result.get_summary()
        assert "GQA" in summary
        assert "32" in summary
        assert "ROPE" in summary
        assert "4.0x" in summary


# =============================================================================
# Test Analyzer
# =============================================================================


class TestAttentionAnalyzer:
    """Test AttentionAnalyzer class."""

    def test_initialization(self) -> None:
        """Test analyzer initialization."""
        analyzer = AttentionAnalyzer()
        assert analyzer is not None

    def test_determine_primary_type_mha(self) -> None:
        """Test MHA type determination."""
        analyzer = AttentionAnalyzer()

        heads = [
            AttentionHeadInfo(name="h1", attention_type=AttentionType.MHA),
            AttentionHeadInfo(name="h2", attention_type=AttentionType.MHA),
            AttentionHeadInfo(name="h3", attention_type=AttentionType.MHA),
        ]
        result = analyzer._determine_primary_type(heads)
        assert result == AttentionType.MHA

    def test_determine_primary_type_gqa(self) -> None:
        """Test GQA type determination."""
        analyzer = AttentionAnalyzer()

        heads = [
            AttentionHeadInfo(name="h1", attention_type=AttentionType.GQA),
            AttentionHeadInfo(name="h2", attention_type=AttentionType.GQA),
        ]
        result = analyzer._determine_primary_type(heads)
        assert result == AttentionType.GQA

    def test_determine_primary_type_empty(self) -> None:
        """Test type determination with no heads."""
        analyzer = AttentionAnalyzer()
        result = analyzer._determine_primary_type([])
        assert result == AttentionType.UNKNOWN

    def test_calculate_kv_cache_mha(self) -> None:
        """Test KV cache calculation for MHA (no savings)."""
        analyzer = AttentionAnalyzer()

        # MHA: num_q_heads == num_kv_heads
        kv = analyzer._calculate_kv_cache(
            num_q_heads=32,
            num_kv_heads=32,
            head_dim=128,
            num_layers=32,
            dtype_bytes=2,
        )
        assert kv is not None
        assert kv.savings_factor == 1.0  # No savings

    def test_calculate_kv_cache_gqa(self) -> None:
        """Test KV cache calculation for GQA (4x savings)."""
        analyzer = AttentionAnalyzer()

        # GQA: 32 Q heads, 8 KV heads -> 4x savings
        kv = analyzer._calculate_kv_cache(
            num_q_heads=32,
            num_kv_heads=8,
            head_dim=128,
            num_layers=32,
            dtype_bytes=2,
        )
        assert kv is not None
        assert kv.savings_factor == 4.0

    def test_calculate_kv_cache_mqa(self) -> None:
        """Test KV cache calculation for MQA (32x savings)."""
        analyzer = AttentionAnalyzer()

        # MQA: 32 Q heads, 1 KV head -> 32x savings
        kv = analyzer._calculate_kv_cache(
            num_q_heads=32,
            num_kv_heads=1,
            head_dim=128,
            num_layers=32,
            dtype_bytes=2,
        )
        assert kv is not None
        assert kv.savings_factor == 32.0

    def test_calculate_kv_cache_zero_values(self) -> None:
        """Test KV cache calculation with zero values."""
        analyzer = AttentionAnalyzer()

        kv = analyzer._calculate_kv_cache(
            num_q_heads=0,
            num_kv_heads=0,
            head_dim=0,
            num_layers=0,
        )
        assert kv is None

    def test_kv_cache_bytes_formula(self) -> None:
        """Test the KV cache bytes per token formula."""
        analyzer = AttentionAnalyzer()

        # Formula: 2 * num_kv_heads * head_dim * num_layers * dtype_bytes
        kv = analyzer._calculate_kv_cache(
            num_q_heads=32,
            num_kv_heads=8,
            head_dim=64,
            num_layers=16,
            dtype_bytes=2,
        )
        assert kv is not None
        expected = 2 * 8 * 64 * 16 * 2  # = 32768
        assert kv.bytes_per_token == expected

    def test_context_length_scaling(self) -> None:
        """Test KV cache scales with context length."""
        analyzer = AttentionAnalyzer()

        kv = analyzer._calculate_kv_cache(
            num_q_heads=8,
            num_kv_heads=8,
            head_dim=64,
            num_layers=12,
            dtype_bytes=2,
        )
        assert kv is not None
        # 8k should be 2x of 4k
        assert kv.total_bytes_at_8k == kv.total_bytes_at_4k * 2


# =============================================================================
# Integration Tests
# =============================================================================


class TestAttentionAnalysisIntegration:
    """Integration tests for attention analysis."""

    def test_full_result_serialization(self) -> None:
        """Test complete result can be serialized to dict."""
        result = AttentionAnalysisResult(
            primary_attention_type=AttentionType.GQA,
            num_attention_layers=32,
            num_q_heads=32,
            num_kv_heads=8,
            head_dim=128,
            hidden_size=4096,
            position_encoding=PositionEncodingInfo(
                encoding_type=PositionEncodingType.ROPE,
                max_positions=8192,
                is_rotary=True,
                extrapolation_capable=True,
            ),
            kv_cache=KVCacheEstimate(
                bytes_per_token=32768,
                total_bytes_at_4k=32768 * 4096,
                total_bytes_at_8k=32768 * 8192,
                total_bytes_at_32k=32768 * 32768,
                total_bytes_at_128k=32768 * 131072,
                mha_baseline_bytes_per_token=131072,
                savings_factor=4.0,
            ),
            attention_pattern=AttentionPatternInfo(
                pattern_type=AttentionPatternType.CAUSAL,
                is_causal=True,
            ),
            fused_attention=FusedAttentionInfo(
                fused_type=FusedAttentionType.SDPA,
                is_memory_efficient=True,
                supports_flash=True,
            ),
            attention_heads=[
                AttentionHeadInfo(
                    name="layer_0",
                    attention_type=AttentionType.GQA,
                    num_q_heads=32,
                    num_kv_heads=8,
                )
            ],
            max_context_length=8192,
            effective_context_length=8192,
        )

        d = result.to_dict()

        # Verify all fields present
        assert d["primary_attention_type"] == "gqa"
        assert d["num_attention_layers"] == 32
        assert d["position_encoding"]["encoding_type"] == "rope"
        assert d["kv_cache"]["savings_factor"] == 4.0
        assert d["attention_pattern"]["pattern_type"] == "causal"
        assert d["fused_attention"]["fused_type"] == "sdpa"
        assert len(d["attention_heads"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
