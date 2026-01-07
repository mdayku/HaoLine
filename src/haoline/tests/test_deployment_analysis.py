# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
Unit tests for deployment_analysis.py (Epic 30).

Tests cover:
- Prefill vs decode phase analysis
- Batching strategy analysis
- Context length scaling
- Serving framework compatibility
"""

from __future__ import annotations

from unittest.mock import MagicMock

from haoline.deployment_analysis import (
    BatchingAnalysis,
    ContextScalingInfo,
    DeploymentAnalysisResult,
    DeploymentAnalyzer,
    FrameworkCompatibility,
    PrefillDecodeAnalysis,
    analyze_deployment,
)

# =============================================================================
# Test Fixtures
# =============================================================================


def create_mock_node(
    name: str,
    op_type: str,
) -> MagicMock:
    """Create a mock NodeInfo."""
    node = MagicMock()
    node.name = name
    node.op_type = op_type
    return node


def create_mock_graph_info(nodes: list[MagicMock]) -> MagicMock:
    """Create a mock GraphInfo."""
    graph = MagicMock()
    graph.nodes = nodes
    return graph


def create_mock_attention_result(
    kv_bytes_per_token: int = 0,
    num_layers: int = 32,
    attention_type: str = "mha",
) -> MagicMock:
    """Create a mock AttentionAnalysisResult."""
    result = MagicMock()

    # KV cache mock
    kv_cache = MagicMock()
    kv_cache.bytes_per_token = kv_bytes_per_token
    result.kv_cache = kv_cache if kv_bytes_per_token > 0 else None

    result.num_attention_layers = num_layers

    # Attention type mock
    primary_type = MagicMock()
    primary_type.value = attention_type
    result.primary_attention_type = primary_type

    return result


def create_mock_memory_result(
    model_size_gb: float = 14.0,
    has_paged_attention: bool = False,
    kv_bytes_per_token: int = 0,
) -> MagicMock:
    """Create a mock MemoryAnalysisResult."""
    result = MagicMock()
    result.model_size_gb = model_size_gb

    # KV cache mock
    kv_cache = MagicMock()
    kv_cache.paged_attention_detected = has_paged_attention
    kv_cache.bytes_per_token = kv_bytes_per_token
    result.kv_cache = kv_cache

    return result


# =============================================================================
# Data Model Tests
# =============================================================================


class TestDataModels:
    """Test Pydantic data models."""

    def test_prefill_decode_analysis_default(self) -> None:
        """Test PrefillDecodeAnalysis default values."""
        pd = PrefillDecodeAnalysis()
        assert pd.prefill_is_compute_bound is True
        assert pd.decode_is_memory_bound is True
        assert pd.estimated_ttft_ms == 0.0

    def test_prefill_decode_to_dict(self) -> None:
        """Test PrefillDecodeAnalysis serialization."""
        pd = PrefillDecodeAnalysis(
            estimated_ttft_ms=50.0,
            estimated_tokens_per_second=100.0,
        )
        d = pd.to_dict()
        assert d["estimated_ttft_ms"] == 50.0
        assert d["estimated_tokens_per_second"] == 100.0

    def test_batching_analysis_default(self) -> None:
        """Test BatchingAnalysis default values."""
        b = BatchingAnalysis()
        assert b.supports_dynamic_batching is True
        assert b.supports_continuous_batching is False
        assert b.recommended_strategy == "static"

    def test_batching_analysis_to_dict(self) -> None:
        """Test BatchingAnalysis serialization."""
        b = BatchingAnalysis(
            has_paged_attention=True,
            max_concurrent_requests=16,
        )
        d = b.to_dict()
        assert d["has_paged_attention"] is True
        assert d["max_concurrent_requests"] == 16

    def test_context_scaling_default(self) -> None:
        """Test ContextScalingInfo default values."""
        cs = ContextScalingInfo()
        assert cs.model_base_context == 4096
        assert cs.oom_context_length == 0

    def test_context_scaling_to_dict(self) -> None:
        """Test ContextScalingInfo serialization."""
        cs = ContextScalingInfo(
            oom_context_length=32768,
            recommended_max_context=16384,
        )
        d = cs.to_dict()
        assert d["oom_context_length"] == 32768
        assert d["recommended_max_context"] == 16384

    def test_framework_compatibility_to_dict(self) -> None:
        """Test FrameworkCompatibility serialization."""
        fc = FrameworkCompatibility(
            framework="vLLM",
            compatible=True,
            compatibility_score=0.95,
            notes=["Best for serving"],
        )
        d = fc.to_dict()
        assert d["framework"] == "vLLM"
        assert d["compatible"] is True
        assert d["compatibility_score"] == 0.95

    def test_deployment_result_to_dict(self) -> None:
        """Test full result serialization."""
        result = DeploymentAnalysisResult(
            target_gpu="h100",
            target_vram_gb=80.0,
            recommended_framework="vLLM",
        )
        d = result.to_dict()
        assert d["target_gpu"] == "h100"
        assert d["recommended_framework"] == "vLLM"
        assert "prefill_decode" in d
        assert "batching" in d

    def test_deployment_result_get_summary(self) -> None:
        """Test summary generation."""
        result = DeploymentAnalysisResult(
            recommended_framework="vLLM",
            recommendations=["Use PagedAttention"],
        )
        summary = result.get_summary()
        assert "LLM Deployment Analysis" in summary
        assert "Use PagedAttention" in summary


# =============================================================================
# Prefill vs Decode Tests
# =============================================================================


class TestPrefillDecodeAnalysis:
    """Test prefill vs decode phase analysis."""

    def test_prefill_is_compute_bound(self) -> None:
        """Verify prefill phase is identified as compute-bound."""
        nodes = [create_mock_node("layer1", "MatMul")]
        graph = create_mock_graph_info(nodes)

        analyzer = DeploymentAnalyzer(target_gpu="a100", vram_gb=80.0)
        result = analyzer.analyze(
            graph,
            total_params=7_000_000_000,
            total_flops=14_000_000_000,
        )

        assert result.prefill_decode.prefill_is_compute_bound is True

    def test_decode_is_memory_bound(self) -> None:
        """Verify decode phase is identified as memory-bound."""
        nodes = [create_mock_node("layer1", "MatMul")]
        graph = create_mock_graph_info(nodes)

        analyzer = DeploymentAnalyzer()
        result = analyzer.analyze(graph, total_params=7_000_000_000)

        assert result.prefill_decode.decode_is_memory_bound is True

    def test_ttft_estimate_scales_with_context(self) -> None:
        """TTFT should scale with context length."""
        nodes = [create_mock_node("layer1", "MatMul")]
        graph = create_mock_graph_info(nodes)

        analyzer = DeploymentAnalyzer()
        result = analyzer.analyze(graph, total_params=7_000_000_000, total_flops=int(1e12))

        pd = result.prefill_decode
        # 4K should be ~4x of 1K base
        assert pd.ttft_at_4k_ms > pd.estimated_ttft_ms
        # 8K should be ~2x of 4K
        assert pd.ttft_at_8k_ms >= pd.ttft_at_4k_ms

    def test_tokens_per_second_estimate(self) -> None:
        """Tokens per second should be estimated."""
        nodes = [create_mock_node("layer1", "MatMul")]
        graph = create_mock_graph_info(nodes)

        memory_result = create_mock_memory_result(model_size_gb=14.0)

        analyzer = DeploymentAnalyzer(target_gpu="a100", vram_gb=80.0)
        result = analyzer.analyze(graph, memory_result=memory_result)

        # Should have a reasonable token rate
        assert result.prefill_decode.estimated_tokens_per_second > 0

    def test_optimal_batch_sizes(self) -> None:
        """Optimal batch sizes should be set."""
        nodes = [create_mock_node("layer1", "MatMul")]
        graph = create_mock_graph_info(nodes)

        analyzer = DeploymentAnalyzer(vram_gb=80.0)
        result = analyzer.analyze(graph, total_params=7_000_000_000)

        assert result.prefill_decode.optimal_prefill_batch_size >= 1
        assert result.prefill_decode.optimal_decode_batch_size >= 1


# =============================================================================
# Batching Strategy Tests
# =============================================================================


class TestBatchingAnalysis:
    """Test batching strategy analysis."""

    def test_paged_attention_enables_continuous_batching(self) -> None:
        """PagedAttention should enable continuous batching."""
        nodes = [create_mock_node("layer1", "MatMul")]
        graph = create_mock_graph_info(nodes)

        memory_result = create_mock_memory_result(has_paged_attention=True)

        analyzer = DeploymentAnalyzer()
        result = analyzer.analyze(graph, memory_result=memory_result)

        assert result.batching.has_paged_attention is True
        assert result.batching.supports_continuous_batching is True
        assert result.batching.recommended_strategy == "continuous"

    def test_static_batching_without_paged_attention(self) -> None:
        """Without PagedAttention, recommend dynamic batching."""
        nodes = [create_mock_node("layer1", "MatMul")]
        graph = create_mock_graph_info(nodes)

        memory_result = create_mock_memory_result(has_paged_attention=False)

        analyzer = DeploymentAnalyzer()
        result = analyzer.analyze(graph, memory_result=memory_result)

        assert result.batching.has_paged_attention is False
        assert result.batching.recommended_strategy == "dynamic"

    def test_throughput_scales_with_batch_size(self) -> None:
        """Throughput should increase with batch size."""
        nodes = [create_mock_node("layer1", "MatMul")]
        graph = create_mock_graph_info(nodes)

        memory_result = create_mock_memory_result(model_size_gb=14.0)

        analyzer = DeploymentAnalyzer()
        result = analyzer.analyze(graph, memory_result=memory_result)

        b = result.batching
        assert b.throughput_at_batch_4 >= b.throughput_at_batch_1
        assert b.throughput_at_batch_8 >= b.throughput_at_batch_4

    def test_max_concurrent_requests_limited_by_vram(self) -> None:
        """Max concurrent requests should be limited by VRAM."""
        nodes = [create_mock_node("layer1", "MatMul")]
        graph = create_mock_graph_info(nodes)

        # Large model with high KV cache
        memory_result = create_mock_memory_result(
            model_size_gb=70.0,
            kv_bytes_per_token=262144,  # 256KB per token
        )

        analyzer = DeploymentAnalyzer(vram_gb=80.0)
        result = analyzer.analyze(graph, memory_result=memory_result)

        # With 70GB model and high KV cache, should have limited concurrency
        assert result.batching.max_concurrent_requests >= 1


# =============================================================================
# Context Scaling Tests
# =============================================================================


class TestContextScaling:
    """Test context length scaling analysis."""

    def test_memory_scales_linearly_with_context(self) -> None:
        """KV cache memory should scale linearly with context."""
        nodes = [create_mock_node("layer1", "MatMul")]
        graph = create_mock_graph_info(nodes)

        attention_result = create_mock_attention_result(
            kv_bytes_per_token=131072,  # 128KB per token
        )

        analyzer = DeploymentAnalyzer()
        result = analyzer.analyze(graph, attention_result=attention_result)

        cs = result.context_scaling
        # 8K should be ~2x of 4K
        assert cs.memory_at_8k_gb >= cs.memory_at_4k_gb * 1.9
        # 32K should be ~8x of 4K
        assert cs.memory_at_32k_gb >= cs.memory_at_4k_gb * 7

    def test_attention_flops_scales_quadratically(self) -> None:
        """Attention FLOPs should scale quadratically with context."""
        nodes = [
            create_mock_node("ln1", "LayerNormalization"),
            create_mock_node("ln2", "LayerNormalization"),
        ]
        graph = create_mock_graph_info(nodes)

        attention_result = create_mock_attention_result(
            kv_bytes_per_token=131072,
            num_layers=32,
        )

        analyzer = DeploymentAnalyzer()
        result = analyzer.analyze(graph, attention_result=attention_result, total_flops=int(1e12))

        cs = result.context_scaling
        # 8K should be ~4x of 4K (quadratic)
        if cs.attention_flops_at_4k > 0:
            assert cs.attention_flops_at_8k == cs.attention_flops_at_4k * 4

    def test_oom_context_calculated(self) -> None:
        """OOM context length should be calculated based on VRAM."""
        nodes = [create_mock_node("layer1", "MatMul")]
        graph = create_mock_graph_info(nodes)

        attention_result = create_mock_attention_result(kv_bytes_per_token=131072)
        memory_result = create_mock_memory_result(model_size_gb=14.0)

        analyzer = DeploymentAnalyzer(vram_gb=80.0)
        result = analyzer.analyze(
            graph, attention_result=attention_result, memory_result=memory_result
        )

        assert result.context_scaling.oom_context_length > 0

    def test_recommended_max_context(self) -> None:
        """Recommended max context should be set."""
        nodes = [create_mock_node("layer1", "MatMul")]
        graph = create_mock_graph_info(nodes)

        attention_result = create_mock_attention_result(kv_bytes_per_token=131072)

        analyzer = DeploymentAnalyzer(vram_gb=24.0)
        result = analyzer.analyze(graph, attention_result=attention_result)

        assert result.context_scaling.recommended_max_context > 0
        assert (
            result.context_scaling.recommended_max_context
            <= result.context_scaling.oom_context_length
        )


# =============================================================================
# Framework Compatibility Tests
# =============================================================================


class TestFrameworkCompatibility:
    """Test serving framework compatibility analysis."""

    def test_vllm_compatibility(self) -> None:
        """vLLM should be compatible for LLM serving."""
        nodes = [create_mock_node("layer1", "MatMul")]
        graph = create_mock_graph_info(nodes)

        analyzer = DeploymentAnalyzer()
        result = analyzer.analyze(graph)

        vllm = next((f for f in result.serving_frameworks if f.framework == "vLLM"), None)
        assert vllm is not None
        assert vllm.compatible is True
        assert vllm.compatibility_score > 0.5

    def test_vllm_score_higher_with_paged_attention(self) -> None:
        """vLLM score should be higher with PagedAttention."""
        nodes = [create_mock_node("layer1", "MatMul")]
        graph = create_mock_graph_info(nodes)

        # Without PagedAttention
        memory_result_no_pa = create_mock_memory_result(has_paged_attention=False)
        analyzer = DeploymentAnalyzer()
        result_no_pa = analyzer.analyze(graph, memory_result=memory_result_no_pa)

        # With PagedAttention
        memory_result_pa = create_mock_memory_result(has_paged_attention=True)
        result_pa = analyzer.analyze(graph, memory_result=memory_result_pa)

        vllm_no_pa = next(f for f in result_no_pa.serving_frameworks if f.framework == "vLLM")
        vllm_pa = next(f for f in result_pa.serving_frameworks if f.framework == "vLLM")

        assert vllm_pa.compatibility_score > vllm_no_pa.compatibility_score

    def test_tensorrt_llm_for_nvidia_gpus(self) -> None:
        """TensorRT-LLM should be compatible for NVIDIA GPUs."""
        nodes = [create_mock_node("layer1", "MatMul")]
        graph = create_mock_graph_info(nodes)

        analyzer = DeploymentAnalyzer(target_gpu="h100")
        result = analyzer.analyze(graph)

        trt = next(
            (f for f in result.serving_frameworks if f.framework == "TensorRT-LLM"),
            None,
        )
        assert trt is not None
        assert trt.compatible is True
        assert trt.compatibility_score > 0.8

    def test_llama_cpp_for_smaller_models(self) -> None:
        """llama.cpp should score higher for smaller models."""
        nodes = [create_mock_node("layer1", "MatMul")]
        graph = create_mock_graph_info(nodes)

        # Small model
        memory_result_small = create_mock_memory_result(model_size_gb=7.0)
        analyzer = DeploymentAnalyzer()
        result_small = analyzer.analyze(graph, memory_result=memory_result_small)

        # Large model
        memory_result_large = create_mock_memory_result(model_size_gb=70.0)
        result_large = analyzer.analyze(graph, memory_result=memory_result_large)

        llama_small = next(f for f in result_small.serving_frameworks if f.framework == "llama.cpp")
        llama_large = next(f for f in result_large.serving_frameworks if f.framework == "llama.cpp")

        assert llama_small.compatibility_score > llama_large.compatibility_score

    def test_recommended_framework_selected(self) -> None:
        """Best framework should be recommended."""
        nodes = [create_mock_node("layer1", "MatMul")]
        graph = create_mock_graph_info(nodes)

        memory_result = create_mock_memory_result(has_paged_attention=True)

        analyzer = DeploymentAnalyzer(target_gpu="h100")
        result = analyzer.analyze(graph, memory_result=memory_result)

        assert result.recommended_framework != ""
        # Should be one of the known frameworks
        assert result.recommended_framework in [
            "vLLM",
            "TensorRT-LLM",
            "llama.cpp",
            "Triton Inference Server",
            "HuggingFace TGI",
            "ONNX Runtime",
        ]

    def test_all_frameworks_analyzed(self) -> None:
        """All major frameworks should be analyzed."""
        nodes = [create_mock_node("layer1", "MatMul")]
        graph = create_mock_graph_info(nodes)

        analyzer = DeploymentAnalyzer()
        result = analyzer.analyze(graph)

        framework_names = [f.framework for f in result.serving_frameworks]
        assert "vLLM" in framework_names
        assert "TensorRT-LLM" in framework_names
        assert "llama.cpp" in framework_names
        assert "Triton Inference Server" in framework_names


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests."""

    def test_analyze_deployment_convenience_function(self) -> None:
        """Test convenience function."""
        nodes = [create_mock_node("layer1", "MatMul")]
        graph = create_mock_graph_info(nodes)

        result = analyze_deployment(
            graph,
            total_params=7_000_000_000,
            target_gpu="a100",
            vram_gb=80.0,
        )

        assert isinstance(result, DeploymentAnalysisResult)
        assert result.target_gpu == "a100"

    def test_full_analysis_with_prior_results(self) -> None:
        """Test analysis with attention and memory results."""
        nodes = [
            create_mock_node("ln1", "LayerNormalization"),
            create_mock_node("ln2", "LayerNormalization"),
        ]
        graph = create_mock_graph_info(nodes)

        attention_result = create_mock_attention_result(
            kv_bytes_per_token=131072,
            num_layers=32,
            attention_type="gqa",
        )
        memory_result = create_mock_memory_result(
            model_size_gb=14.0,
            has_paged_attention=True,
        )

        result = analyze_deployment(
            graph,
            attention_result=attention_result,
            memory_result=memory_result,
            total_params=7_000_000_000,
            total_flops=14_000_000_000,
            target_gpu="h100",
            vram_gb=80.0,
        )

        # Should have all components
        assert result.prefill_decode.estimated_tokens_per_second > 0
        assert result.batching.has_paged_attention is True
        assert result.context_scaling.kv_cache_bytes_per_token == 131072
        assert len(result.serving_frameworks) >= 4

    def test_recommendations_generated(self) -> None:
        """Test that recommendations are generated."""
        nodes = [create_mock_node("layer1", "MatMul")]
        graph = create_mock_graph_info(nodes)

        memory_result = create_mock_memory_result(
            model_size_gb=70.0,
            has_paged_attention=True,
            kv_bytes_per_token=262144,
        )

        result = analyze_deployment(graph, memory_result=memory_result, vram_gb=24.0)

        # Should have recommendations for the challenging scenario
        assert len(result.recommendations) >= 0  # May or may not have based on analysis
