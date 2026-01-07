# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""Tests for Advanced Quantization Analysis (Epic 26)."""

from __future__ import annotations

import pytest

from haoline.quantization_analysis import (
    GGML_ACCURACY_RANKING,
    SCHEME_ACCURACY_IMPACT,
    AccuracyImpactEstimate,
    LayerPrecisionInfo,
    LayerType,
    PrecisionType,
    QuantizationAnalyzer,
    QuantizationScheme,
    QuantizationSchemeInfo,
    SensitiveLayerInfo,
)

# =============================================================================
# Test Constants and Data
# =============================================================================


class TestSchemeAccuracyImpact:
    """Test accuracy impact constants."""

    def test_all_schemes_have_impact_data(self) -> None:
        """Every QuantizationScheme should have impact data."""
        for scheme in QuantizationScheme:
            assert scheme in SCHEME_ACCURACY_IMPACT, f"Missing impact data for {scheme}"

    def test_fp32_has_zero_impact(self) -> None:
        """FP32 should have zero accuracy impact."""
        impact = SCHEME_ACCURACY_IMPACT[QuantizationScheme.FP32]
        assert impact["perplexity_increase_pct"] == 0.0
        assert impact["accuracy_drop_pct"] == 0.0

    def test_gptq_has_moderate_impact(self) -> None:
        """GPTQ should have moderate accuracy impact."""
        impact = SCHEME_ACCURACY_IMPACT[QuantizationScheme.GPTQ]
        assert 0.5 <= impact["perplexity_increase_pct"] <= 3.0
        assert impact["accuracy_drop_pct"] < 2.0


class TestGGMLAccuracyRanking:
    """Test GGML type ranking data."""

    def test_q8_0_is_highest_quality(self) -> None:
        """Q8_0 should be the highest quality GGML type."""
        assert GGML_ACCURACY_RANKING["Q8_0"]["rank"] == 1

    def test_q2_k_is_low_quality(self) -> None:
        """Q2_K should be low quality (high rank number)."""
        assert GGML_ACCURACY_RANKING["Q2_K"]["rank"] > 10

    def test_all_types_have_perplexity_data(self) -> None:
        """All GGML types should have perplexity data."""
        for name, data in GGML_ACCURACY_RANKING.items():
            assert "perplexity_pct" in data, f"Missing perplexity for {name}"
            assert "bits_per_weight" in data, f"Missing bits for {name}"


# =============================================================================
# Test Data Classes
# =============================================================================


class TestLayerPrecisionInfo:
    """Test LayerPrecisionInfo dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic creation of LayerPrecisionInfo."""
        lp = LayerPrecisionInfo(
            layer_name="layer.0.attn.weight",
            layer_type=LayerType.ATTENTION,
            weight_precision=PrecisionType.FP16,
            param_count=1024,
        )
        assert lp.layer_name == "layer.0.attn.weight"
        assert lp.layer_type == LayerType.ATTENTION
        assert lp.weight_precision == PrecisionType.FP16
        assert lp.param_count == 1024
        assert not lp.is_sensitive

    def test_sensitive_layer(self) -> None:
        """Test sensitive layer flagging."""
        lp = LayerPrecisionInfo(
            layer_name="embed_tokens.weight",
            layer_type=LayerType.EMBEDDING,
            weight_precision=PrecisionType.INT4,
            is_sensitive=True,
            sensitivity_reason="Embedding layers are sensitive",
        )
        assert lp.is_sensitive
        assert lp.sensitivity_reason is not None and "Embedding" in lp.sensitivity_reason

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        lp = LayerPrecisionInfo(
            layer_name="test",
            layer_type=LayerType.FFN,
            weight_precision=PrecisionType.INT8,
        )
        d = lp.to_dict()
        assert d["layer_name"] == "test"
        assert d["layer_type"] == "ffn"
        assert d["weight_precision"] == "int8"


class TestQuantizationSchemeInfo:
    """Test QuantizationSchemeInfo dataclass."""

    def test_gptq_scheme_info(self) -> None:
        """Test GPTQ scheme info."""
        info = QuantizationSchemeInfo(
            scheme=QuantizationScheme.GPTQ,
            confidence=0.95,
            bits=4,
            group_size=128,
            act_order=True,
            evidence=["Found qweight tensors", "Found g_idx"],
        )
        assert info.scheme == QuantizationScheme.GPTQ
        assert info.bits == 4
        assert info.act_order
        assert len(info.evidence) == 2

    def test_ggml_scheme_info(self) -> None:
        """Test GGML scheme info."""
        info = QuantizationSchemeInfo(
            scheme=QuantizationScheme.GGML,
            confidence=1.0,
            ggml_types=["Q4_K_M", "Q6_K"],
            bits=4,
        )
        assert info.scheme == QuantizationScheme.GGML
        assert "Q4_K_M" in info.ggml_types

    def test_to_dict(self) -> None:
        """Test serialization."""
        info = QuantizationSchemeInfo(
            scheme=QuantizationScheme.AWQ,
            confidence=0.8,
        )
        d = info.to_dict()
        assert d["scheme"] == "awq"
        assert d["confidence"] == 0.8


class TestAccuracyImpactEstimate:
    """Test AccuracyImpactEstimate dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic creation."""
        est = AccuracyImpactEstimate(
            perplexity_increase_pct=1.5,
            accuracy_drop_pct=0.8,
            description="GPTQ INT4",
            memory_reduction_factor=8.0,
        )
        assert est.perplexity_increase_pct == 1.5
        assert est.memory_reduction_factor == 8.0

    def test_with_recommendations(self) -> None:
        """Test with recommendations."""
        est = AccuracyImpactEstimate(
            perplexity_increase_pct=1.0,
            recommendations=["Keep embed_tokens at FP16", "Use representative calibration"],
        )
        assert len(est.recommendations) == 2


class TestSensitiveLayerInfo:
    """Test SensitiveLayerInfo dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic creation."""
        sl = SensitiveLayerInfo(
            layer_name="lm_head.weight",
            layer_type=LayerType.OUTPUT,
            sensitivity_score=0.9,
            reason="Output layers directly affect predictions",
            recommendation="Keep at FP16",
            current_precision=PrecisionType.INT4,
            recommended_precision=PrecisionType.FP16,
        )
        assert sl.sensitivity_score == 0.9
        assert sl.recommended_precision == PrecisionType.FP16


# =============================================================================
# Test Analyzer
# =============================================================================


class TestQuantizationAnalyzer:
    """Test QuantizationAnalyzer class."""

    def test_initialization(self) -> None:
        """Test analyzer initialization."""
        analyzer = QuantizationAnalyzer()
        assert analyzer is not None

    def test_check_gptq_patterns(self) -> None:
        """Test GPTQ pattern detection."""
        analyzer = QuantizationAnalyzer()

        # Test with GPTQ-like tensor names
        gptq_tensors = [
            "model.layers.0.self_attn.q_proj.qweight",
            "model.layers.0.self_attn.q_proj.scales",
            "model.layers.0.self_attn.q_proj.qzeros",
            "model.layers.0.self_attn.q_proj.g_idx",
        ]
        score, evidence = analyzer._check_gptq_patterns(gptq_tensors)
        assert score > 0.7, f"GPTQ score too low: {score}"
        assert len(evidence) > 0

    def test_check_awq_patterns(self) -> None:
        """Test AWQ pattern detection."""
        analyzer = QuantizationAnalyzer()

        # AWQ has qweight and scales but no g_idx
        awq_tensors = [
            "model.layers.0.self_attn.q_proj.qweight",
            "model.layers.0.self_attn.q_proj.scales",
        ]
        score, evidence = analyzer._check_awq_patterns(awq_tensors)
        assert score > 0.5, f"AWQ score too low: {score}"

    def test_check_bitsandbytes_patterns(self) -> None:
        """Test bitsandbytes pattern detection."""
        analyzer = QuantizationAnalyzer()

        bnb_tensors = [
            "model.layers.0.self_attn.q_proj.absmax",
            "model.layers.0.self_attn.q_proj.quant_state",
        ]
        score, evidence, quant_type = analyzer._check_bitsandbytes_patterns(bnb_tensors)
        assert score > 0.5, f"BNB score too low: {score}"

    def test_estimate_bits_from_ggml(self) -> None:
        """Test GGML bits estimation."""
        analyzer = QuantizationAnalyzer()

        assert analyzer._estimate_bits_from_ggml(["Q4_K_M", "Q4_K_M"]) == 4
        assert analyzer._estimate_bits_from_ggml(["Q8_0"]) == 8
        assert analyzer._estimate_bits_from_ggml(["Q2_K"]) == 2
        assert analyzer._estimate_bits_from_ggml(["F16"]) == 16

    def test_block_type_to_layer_type(self) -> None:
        """Test block type to layer type conversion."""
        analyzer = QuantizationAnalyzer()

        assert analyzer._block_type_to_layer_type("AttentionHead") == LayerType.ATTENTION
        assert analyzer._block_type_to_layer_type("MHA") == LayerType.ATTENTION
        assert analyzer._block_type_to_layer_type("MLPBlock") == LayerType.FFN
        assert analyzer._block_type_to_layer_type("FFN") == LayerType.FFN
        assert analyzer._block_type_to_layer_type("EmbeddingLayer") == LayerType.EMBEDDING
        assert analyzer._block_type_to_layer_type("LayerNorm") == LayerType.NORMALIZATION
        assert analyzer._block_type_to_layer_type("Unknown") == LayerType.OTHER

    def test_infer_layer_type_from_op(self) -> None:
        """Test layer type inference from op type."""
        analyzer = QuantizationAnalyzer()

        assert analyzer._infer_layer_type_from_op("Attention") == LayerType.ATTENTION
        assert analyzer._infer_layer_type_from_op("MatMul") == LayerType.FFN
        assert analyzer._infer_layer_type_from_op("LayerNormalization") == LayerType.NORMALIZATION
        assert analyzer._infer_layer_type_from_op("Gather") == LayerType.EMBEDDING

    def test_is_mixed_precision(self) -> None:
        """Test mixed precision detection."""
        analyzer = QuantizationAnalyzer()

        # Single precision
        single_prec = [
            LayerPrecisionInfo(
                layer_name="a", layer_type=LayerType.FFN, weight_precision=PrecisionType.FP16
            ),
            LayerPrecisionInfo(
                layer_name="b", layer_type=LayerType.FFN, weight_precision=PrecisionType.FP16
            ),
        ]
        assert not analyzer._is_mixed_precision(single_prec)

        # Mixed precision
        mixed_prec = [
            LayerPrecisionInfo(
                layer_name="a", layer_type=LayerType.FFN, weight_precision=PrecisionType.FP16
            ),
            LayerPrecisionInfo(
                layer_name="b", layer_type=LayerType.FFN, weight_precision=PrecisionType.INT8
            ),
        ]
        assert analyzer._is_mixed_precision(mixed_prec)

    def test_calculate_sensitivity_score(self) -> None:
        """Test sensitivity score calculation."""
        analyzer = QuantizationAnalyzer()

        # Embedding should be sensitive
        embed = LayerPrecisionInfo(
            layer_name="embed", layer_type=LayerType.EMBEDDING, weight_precision=PrecisionType.FP16
        )
        embed_score = analyzer._calculate_sensitivity_score(embed)
        assert embed_score > 0.7

        # Output should be very sensitive
        output = LayerPrecisionInfo(
            layer_name="lm_head", layer_type=LayerType.OUTPUT, weight_precision=PrecisionType.FP16
        )
        output_score = analyzer._calculate_sensitivity_score(output)
        assert output_score > 0.8

        # Regular FFN should be less sensitive
        ffn = LayerPrecisionInfo(
            layer_name="ffn.up", layer_type=LayerType.FFN, weight_precision=PrecisionType.FP16
        )
        ffn_score = analyzer._calculate_sensitivity_score(ffn)
        assert ffn_score < embed_score


# =============================================================================
# Integration Tests
# =============================================================================


class TestQuantizationAnalysisIntegration:
    """Integration tests for quantization analysis."""

    def test_accuracy_impact_for_gptq(self) -> None:
        """Test accuracy impact estimation for GPTQ."""
        analyzer = QuantizationAnalyzer()

        scheme_info = QuantizationSchemeInfo(
            scheme=QuantizationScheme.GPTQ,
            confidence=0.9,
            bits=4,
        )
        impact = analyzer._estimate_accuracy_impact(scheme_info, None)

        assert impact.perplexity_increase_pct is not None
        assert impact.perplexity_increase_pct > 0
        assert impact.memory_reduction_factor >= 4.0  # INT4 = 8x vs FP32
        assert len(impact.recommendations) > 0

    def test_accuracy_impact_for_fp16(self) -> None:
        """Test accuracy impact estimation for FP16."""
        analyzer = QuantizationAnalyzer()

        scheme_info = QuantizationSchemeInfo(
            scheme=QuantizationScheme.FP16,
            confidence=0.9,
        )
        impact = analyzer._estimate_accuracy_impact(scheme_info, None)

        assert impact.perplexity_increase_pct == 0.0
        assert impact.memory_reduction_factor == 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
