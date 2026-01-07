# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
Unit tests for sparse_analysis.py (Epic 29).

Tests cover:
- MoE detection and analysis
- Speculative decoding detection
- Weight sparsity analysis
- Efficient architecture pattern detection
"""

from __future__ import annotations

from unittest.mock import MagicMock

from haoline.sparse_analysis import (
    EfficientPatternInfo,
    LayerSparsityInfo,
    MoEInfo,
    SparseAnalysisResult,
    SparseAnalyzer,
    SparsityInfo,
    SpeculativeDecodingInfo,
    analyze_sparse,
)

# =============================================================================
# Test Fixtures
# =============================================================================


def create_mock_node(
    name: str,
    op_type: str,
    inputs: list[str] | None = None,
    attributes: list[tuple[str, int]] | None = None,
) -> MagicMock:
    """Create a mock NodeInfo."""
    node = MagicMock()
    node.name = name
    node.op_type = op_type
    node.inputs = inputs or []
    node.attributes = attributes or []
    return node


def create_mock_graph_info(nodes: list[MagicMock]) -> MagicMock:
    """Create a mock GraphInfo."""
    graph = MagicMock()
    graph.nodes = nodes
    graph.node_by_output = {}
    return graph


def create_mock_block(
    block_type: str,
    name: str = "block",
    attributes: dict | None = None,
) -> MagicMock:
    """Create a mock Block."""
    block = MagicMock()
    block.block_type = block_type
    block.name = name
    block.attributes = attributes or {}
    return block


# =============================================================================
# Data Model Tests
# =============================================================================


class TestDataModels:
    """Test Pydantic data models."""

    def test_moe_info_default(self) -> None:
        """Test MoEInfo default values."""
        info = MoEInfo()
        assert info.detected is False
        assert info.num_experts == 0
        assert info.routing_type == "unknown"

    def test_moe_info_to_dict(self) -> None:
        """Test MoEInfo serialization."""
        info = MoEInfo(
            detected=True,
            routing_type="top_k",
            num_experts=8,
            active_experts_per_token=2,
        )
        d = info.to_dict()
        assert d["detected"] is True
        assert d["num_experts"] == 8
        assert d["active_experts_per_token"] == 2

    def test_sparsity_info_default(self) -> None:
        """Test SparsityInfo default values."""
        info = SparsityInfo()
        assert info.detected is False
        assert info.overall_sparsity_ratio == 0.0
        assert info.primary_sparsity_type == "dense"

    def test_sparsity_info_to_dict(self) -> None:
        """Test SparsityInfo serialization."""
        info = SparsityInfo(
            detected=True,
            primary_sparsity_type="structured_nm",
            overall_sparsity_ratio=0.5,
            hardware_accelerated=True,
            compatible_hardware=["NVIDIA Ampere"],
        )
        d = info.to_dict()
        assert d["detected"] is True
        assert d["primary_sparsity_type"] == "structured_nm"
        assert d["hardware_accelerated"] is True

    def test_layer_sparsity_info_to_dict(self) -> None:
        """Test LayerSparsityInfo serialization."""
        info = LayerSparsityInfo(
            layer_name="conv1",
            sparsity_type="structured_nm",
            sparsity_ratio=0.5,
            nm_pattern="2:4",
        )
        d = info.to_dict()
        assert d["layer_name"] == "conv1"
        assert d["nm_pattern"] == "2:4"

    def test_speculative_decoding_info_default(self) -> None:
        """Test SpeculativeDecodingInfo default values."""
        info = SpeculativeDecodingInfo()
        assert info.detected is False
        assert info.estimated_speedup == 1.0

    def test_efficient_pattern_info_to_dict(self) -> None:
        """Test EfficientPatternInfo serialization."""
        info = EfficientPatternInfo(
            pattern_type="depthwise_separable",
            count=10,
            flops_saved_ratio=0.85,
        )
        d = info.to_dict()
        assert d["pattern_type"] == "depthwise_separable"
        assert d["count"] == 10

    def test_sparse_analysis_result_to_dict(self) -> None:
        """Test full result serialization."""
        result = SparseAnalysisResult(
            is_moe_model=True,
            is_sparse_model=True,
            recommendations=["Use expert parallelism"],
        )
        d = result.to_dict()
        assert d["is_moe_model"] is True
        assert d["is_sparse_model"] is True
        assert len(d["recommendations"]) == 1

    def test_sparse_analysis_result_get_summary(self) -> None:
        """Test summary generation."""
        result = SparseAnalysisResult()
        summary = result.get_summary()
        assert "Sparse & Efficient Architecture Analysis" in summary
        assert "No sparse/efficient patterns detected" in summary


# =============================================================================
# MoE Detection Tests
# =============================================================================


class TestMoEAnalysis:
    """Test Mixture of Experts detection."""

    def test_detect_moe_from_topk_nodes(self) -> None:
        """Detect MoE from TopK operations."""
        nodes = [
            create_mock_node("router", "MatMul"),
            create_mock_node("topk_expert", "TopK", attributes=[("k", 2)]),
            create_mock_node("expert_0", "MatMul"),
            create_mock_node("expert_1", "MatMul"),
        ]
        graph = create_mock_graph_info(nodes)

        analyzer = SparseAnalyzer()
        result = analyzer.analyze(graph, None, total_params=1_000_000)

        assert result.is_moe_model is True
        assert result.moe_info.detected is True
        assert result.moe_info.active_experts_per_token == 2

    def test_detect_moe_from_blocks(self) -> None:
        """Detect MoE from pattern blocks."""
        nodes = [create_mock_node("layer1", "MatMul")]
        graph = create_mock_graph_info(nodes)

        blocks = [
            create_mock_block(
                "MoERouter",
                attributes={"num_experts": 8, "top_k": 2, "router_type": "top_k"},
            ),
            create_mock_block("MoERouter", attributes={"num_experts": 8, "top_k": 2}),
        ]

        analyzer = SparseAnalyzer()
        result = analyzer.analyze(graph, blocks, total_params=7_000_000_000)

        assert result.moe_info.detected is True
        assert result.moe_info.num_experts == 8
        assert result.moe_info.num_expert_layers == 2
        assert result.moe_info.parameter_efficiency > 0

    def test_detect_moe_from_tensor_names(self) -> None:
        """Detect MoE from tensor naming patterns."""
        nodes = [
            create_mock_node("layer.expert.0.fc1", "MatMul"),
            create_mock_node("layer.expert.1.fc1", "MatMul"),
            create_mock_node("layer.expert.2.fc1", "MatMul"),
            create_mock_node("layer.expert.3.fc1", "MatMul"),
            create_mock_node("layer.moe.gate", "Softmax"),
            create_mock_node("layer.router.linear", "MatMul"),
        ]
        graph = create_mock_graph_info(nodes)

        analyzer = SparseAnalyzer()
        result = analyzer.analyze(graph)

        assert result.moe_info.detected is True
        # Should estimate 4 experts from naming pattern
        assert result.moe_info.num_experts >= 4

    def test_estimate_expert_count(self) -> None:
        """Test expert count estimation from names."""
        nodes = [
            create_mock_node("experts[0].weight", "MatMul"),
            create_mock_node("experts[7].weight", "MatMul"),
        ]
        graph = create_mock_graph_info(nodes)

        analyzer = SparseAnalyzer()
        count = analyzer._estimate_expert_count(graph)
        assert count == 8  # 0-7 = 8 experts

    def test_detect_expert_parallelism(self) -> None:
        """Test expert parallelism detection."""
        nodes = [
            create_mock_node("AllToAll_expert_exchange", "AllToAll"),
        ]
        graph = create_mock_graph_info(nodes)

        analyzer = SparseAnalyzer()
        assert analyzer._detect_expert_parallelism(graph) is True

        # Without AllToAll
        nodes = [create_mock_node("matmul", "MatMul")]
        graph = create_mock_graph_info(nodes)
        assert analyzer._detect_expert_parallelism(graph) is False

    def test_detect_load_balance_loss(self) -> None:
        """Test load balance loss detection."""
        nodes = [create_mock_node("router_aux_loss", "ReduceMean")]
        graph = create_mock_graph_info(nodes)

        analyzer = SparseAnalyzer()
        assert analyzer._detect_load_balance_loss(graph) is True


# =============================================================================
# Speculative Decoding Tests
# =============================================================================


class TestSpeculativeDecoding:
    """Test speculative decoding detection."""

    def test_detect_speculative_decoding_pattern(self) -> None:
        """Detect draft + verify model pattern."""
        nodes = [
            create_mock_node("draft_model.layer1", "MatMul"),
            create_mock_node("draft_model.layer2", "MatMul"),
            create_mock_node("verify_model.layer1", "MatMul"),
            create_mock_node("main_model.layer1", "MatMul"),
        ]
        graph = create_mock_graph_info(nodes)

        analyzer = SparseAnalyzer()
        result = analyzer.analyze(graph)

        assert result.speculative_decoding_info.detected is True
        assert result.speculative_decoding_info.has_draft_model is True
        assert result.speculative_decoding_info.has_verify_model is True
        assert result.speculative_decoding_info.estimated_speedup > 1.0

    def test_no_speculative_decoding(self) -> None:
        """No detection for standard model."""
        nodes = [
            create_mock_node("layer1", "MatMul"),
            create_mock_node("layer2", "MatMul"),
        ]
        graph = create_mock_graph_info(nodes)

        analyzer = SparseAnalyzer()
        result = analyzer.analyze(graph)

        assert result.speculative_decoding_info.detected is False


# =============================================================================
# Sparsity Analysis Tests
# =============================================================================


class TestSparsityAnalysis:
    """Test weight sparsity analysis."""

    def test_detect_structured_sparsity(self) -> None:
        """Detect 2:4 structured sparsity."""
        nodes = [
            create_mock_node("conv1_2:4_sparse", "Conv"),
            create_mock_node("conv2_sparse_semi_structured", "Conv"),
        ]
        graph = create_mock_graph_info(nodes)

        analyzer = SparseAnalyzer()
        result = analyzer.analyze(graph, total_flops=1_000_000)

        assert result.sparsity_info.detected is True
        assert result.sparsity_info.primary_sparsity_type == "structured_nm"
        assert result.sparsity_info.hardware_accelerated is True
        assert "NVIDIA Ampere" in result.sparsity_info.compatible_hardware[0]

    def test_detect_unstructured_sparsity(self) -> None:
        """Detect unstructured pruned weights."""
        nodes = [
            create_mock_node("conv1_sparse", "Conv"),
            create_mock_node("conv2_pruned", "Conv"),
        ]
        graph = create_mock_graph_info(nodes)

        analyzer = SparseAnalyzer()
        result = analyzer.analyze(graph, total_flops=1_000_000)

        assert result.sparsity_info.detected is True
        assert result.sparsity_info.primary_sparsity_type == "unstructured"
        assert result.sparsity_info.hardware_accelerated is False

    def test_detect_sparsity_from_mask(self) -> None:
        """Detect sparsity from mask tensor inputs."""
        node = create_mock_node("conv_with_mask", "Conv", inputs=["input", "weight_mask"])
        graph = create_mock_graph_info([node])

        analyzer = SparseAnalyzer()
        result = analyzer.analyze(graph)

        assert result.sparsity_info.detected is True

    def test_no_sparsity_detection(self) -> None:
        """No detection for dense model."""
        nodes = [
            create_mock_node("conv1", "Conv"),
            create_mock_node("conv2", "Conv"),
        ]
        graph = create_mock_graph_info(nodes)

        analyzer = SparseAnalyzer()
        result = analyzer.analyze(graph)

        assert result.sparsity_info.detected is False
        assert result.is_sparse_model is False


# =============================================================================
# Efficient Architecture Tests
# =============================================================================


class TestEfficientArchitectures:
    """Test efficient architecture pattern detection."""

    def test_detect_depthwise_separable_from_name(self) -> None:
        """Detect depthwise separable convs from naming."""
        nodes = [
            create_mock_node("block1_depthwise_conv", "Conv"),
            create_mock_node("block1_pointwise_conv", "Conv"),
            create_mock_node("block2_dw_conv", "Conv"),
            create_mock_node("block2_pw_conv", "Conv"),
        ]
        graph = create_mock_graph_info(nodes)

        analyzer = SparseAnalyzer()
        result = analyzer.analyze(graph)

        assert result.is_efficient_arch is True
        dw_pattern = next(
            (
                p
                for p in result.efficient_arch_info.patterns_detected
                if p.pattern_type == "depthwise_separable"
            ),
            None,
        )
        assert dw_pattern is not None
        assert dw_pattern.count >= 2

    def test_detect_depthwise_from_group_attr(self) -> None:
        """Detect depthwise convs from group attribute."""
        node = create_mock_node("conv1", "Conv", attributes=[("group", 32)])
        graph = create_mock_graph_info([node])

        analyzer = SparseAnalyzer()
        info = analyzer._detect_depthwise_separable(graph)

        assert info.count == 1
        assert info.flops_saved_ratio > 0

    def test_detect_inverted_residual(self) -> None:
        """Detect inverted residual blocks."""
        nodes = [
            create_mock_node("block_1.expand_conv", "Conv"),
            create_mock_node("block_1.dwconv", "Conv"),
            create_mock_node("block_1.project_conv", "Conv"),
            create_mock_node("mbconv_block_2", "Conv"),
        ]
        graph = create_mock_graph_info(nodes)

        analyzer = SparseAnalyzer()
        info = analyzer._detect_inverted_residual(graph)

        assert info.count >= 1

    def test_detect_squeeze_excitation(self) -> None:
        """Detect squeeze-and-excitation blocks."""
        nodes = [
            create_mock_node("se_block.squeeze", "GlobalAveragePool"),
            create_mock_node("se_block.excite.fc1", "MatMul"),
            create_mock_node("se_block.excite.fc2", "MatMul"),
            create_mock_node("channel_attention", "Sigmoid"),
        ]
        graph = create_mock_graph_info(nodes)

        analyzer = SparseAnalyzer()
        info = analyzer._detect_squeeze_excitation(graph)

        assert info.count >= 1

    def test_detect_nas_patterns(self) -> None:
        """Detect NAS architecture patterns."""
        nodes = [
            create_mock_node("nas_cell_0", "Conv"),
            create_mock_node("choice_block_1", "Conv"),
            create_mock_node("mnasnet_layer", "Conv"),
        ]
        graph = create_mock_graph_info(nodes)

        analyzer = SparseAnalyzer()
        info = analyzer._detect_nas_patterns(graph)

        assert info.count == 3

    def test_classify_mobilenet_architecture(self) -> None:
        """Classify as MobileNet when patterns match."""
        nodes = []
        # Create multiple depthwise + inverted residual patterns
        for i in range(10):
            nodes.append(create_mock_node(f"block_{i}_depthwise", "Conv"))
            nodes.append(create_mock_node(f"block_{i}_expand_conv", "Conv"))

        graph = create_mock_graph_info(nodes)

        analyzer = SparseAnalyzer()
        result = analyzer.analyze(graph)

        assert result.efficient_arch_info.architecture_type == "mobilenet"
        assert result.efficient_arch_info.flops_efficiency_ratio > 1.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for full analysis."""

    def test_analyze_sparse_convenience_function(self) -> None:
        """Test convenience function."""
        nodes = [create_mock_node("layer1", "MatMul")]
        graph = create_mock_graph_info(nodes)

        result = analyze_sparse(graph, total_params=1_000_000, total_flops=1_000_000)

        assert isinstance(result, SparseAnalysisResult)

    def test_combined_moe_and_sparse(self) -> None:
        """Test model with both MoE and sparsity."""
        nodes = [
            create_mock_node("expert_0_sparse", "MatMul"),
            create_mock_node("expert_1_sparse", "MatMul"),
            create_mock_node("moe_router", "TopK", attributes=[("k", 2)]),
        ]
        graph = create_mock_graph_info(nodes)

        result = analyze_sparse(graph, total_params=1_000_000)

        assert result.is_moe_model is True
        assert result.is_sparse_model is True

    def test_recommendations_generated(self) -> None:
        """Test that recommendations are generated for detected patterns."""
        # Large MoE model
        nodes = [create_mock_node(f"expert_{i}", "MatMul") for i in range(16)]
        nodes.append(create_mock_node("topk", "TopK", attributes=[("k", 2)]))
        graph = create_mock_graph_info(nodes)

        blocks = [
            create_mock_block("MoERouter", attributes={"num_experts": 16, "top_k": 2}),
        ]

        result = analyze_sparse(graph, blocks, total_params=100_000_000_000)

        assert len(result.recommendations) > 0
        assert any("expert parallelism" in r.lower() for r in result.recommendations)

    def test_summary_includes_all_sections(self) -> None:
        """Test summary includes all detected sections."""
        nodes = [
            create_mock_node("expert_0", "MatMul"),
            create_mock_node("moe_router", "TopK", attributes=[("k", 2)]),
            create_mock_node("conv_sparse_2:4", "Conv"),
            create_mock_node("depthwise_conv", "Conv"),
        ]
        graph = create_mock_graph_info(nodes)

        result = analyze_sparse(graph, total_params=1_000_000, total_flops=1_000_000)
        summary = result.get_summary()

        # Should have MoE section
        assert "Mixture of Experts" in summary or "MoE" in summary.upper()
        # Should have sparsity section
        assert "Sparsity" in summary
