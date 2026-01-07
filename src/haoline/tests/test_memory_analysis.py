# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""Unit tests for Memory Pattern Analysis (Epic 28)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from haoline.memory_analysis import (
    CommunicationOp,
    CommunicationOpInfo,
    KVCacheAnalysis,
    KVCacheQuantization,
    MemoryAnalysisResult,
    MemoryAnalyzer,
    ParallelismAnalysis,
    ParallelismType,
    VRAMRecommendation,
)

# =============================================================================
# Fixtures
# =============================================================================


def create_mock_graph_info(
    *,
    initializers: dict[str, Any] | None = None,
    nodes: list[Any] | None = None,
    op_type_counts: dict[str, int] | None = None,
    precision_breakdown: dict[str, int] | None = None,
    value_info: dict[str, list[int]] | None = None,
) -> MagicMock:
    """Create a mock GraphInfo object."""
    graph_info = MagicMock()

    # Default initializers (simple model weights)
    if initializers is None:
        # Create mock tensors
        mock_tensor = MagicMock()
        mock_tensor.shape = [768, 768]  # ~2.4M params
        mock_tensor.data_type = 1  # FLOAT
        initializers = {"weight1": mock_tensor}

    graph_info.initializers = initializers

    # Default nodes
    if nodes is None:
        mock_node = MagicMock()
        mock_node.name = "layer1"
        mock_node.op_type = "MatMul"
        mock_node.inputs = []
        mock_node.outputs = ["output1"]
        nodes = [mock_node]

    graph_info.nodes = nodes

    # Default op type counts
    graph_info.op_type_counts = op_type_counts or {"MatMul": 10, "LayerNormalization": 5}

    # Default precision breakdown
    graph_info.precision_breakdown = precision_breakdown or {"FP32": 100}

    # Default value info
    graph_info.value_info = value_info or {}

    return graph_info


def create_mock_attention_result(
    *,
    bytes_per_token: int = 2048,
) -> MagicMock:
    """Create a mock AttentionAnalysisResult."""
    result = MagicMock()

    kv_cache = MagicMock()
    kv_cache.bytes_per_token = bytes_per_token
    kv_cache.total_bytes_at_4k = bytes_per_token * 4096
    kv_cache.total_bytes_at_8k = bytes_per_token * 8192
    kv_cache.total_bytes_at_32k = bytes_per_token * 32768
    kv_cache.savings_factor = 4.0

    result.kv_cache_estimate = kv_cache
    return result


# =============================================================================
# Data Model Tests
# =============================================================================


class TestDataModels:
    """Tests for data model classes."""

    def test_kv_cache_analysis_to_dict(self) -> None:
        """Test KVCacheAnalysis serialization."""
        kv = KVCacheAnalysis(
            kv_quantization="int8",
            kv_quantization_detected=True,
            bytes_per_token=1024,
            kv_cache_percent_at_4k=25.5,
            kv_cache_percent_at_8k=35.2,
            kv_cache_percent_at_32k=60.7,
            max_context_for_vram=65536,
            vram_gb_used=24.0,
            paged_attention_detected=True,
            paged_attention_indicators=["vLLM pattern: paged_attn"],
        )

        d = kv.to_dict()
        assert d["kv_quantization"] == "int8"
        assert d["kv_quantization_detected"] is True
        assert d["bytes_per_token"] == 1024
        assert d["kv_cache_percent_at_32k"] == 60.7
        assert d["max_context_for_vram"] == 65536
        assert d["paged_attention_detected"] is True
        assert len(d["paged_attention_indicators"]) == 1

    def test_communication_op_info_to_dict(self) -> None:
        """Test CommunicationOpInfo serialization."""
        op = CommunicationOpInfo(
            op_type="all_reduce",
            node_name="nccl_allreduce_0",
            tensor_shape=[1024, 1024],
            tensor_bytes=4194304,
            parallelism_hint="Tensor Parallelism",
        )

        d = op.to_dict()
        assert d["op_type"] == "all_reduce"
        assert d["node_name"] == "nccl_allreduce_0"
        assert d["tensor_bytes"] == 4194304

    def test_parallelism_analysis_to_dict(self) -> None:
        """Test ParallelismAnalysis serialization."""
        par = ParallelismAnalysis(
            detected_type="tensor_parallel",
            confidence=0.85,
            indicators=["AllReduce found"],
            tp_degree_estimate=4,
            memory_per_gpu_gb={"1": 20.0, "2": 10.5, "4": 5.5, "8": 3.0},
        )

        d = par.to_dict()
        assert d["detected_type"] == "tensor_parallel"
        assert d["confidence"] == 0.85
        assert d["tp_degree_estimate"] == 4
        assert d["memory_per_gpu_gb"]["4"] == 5.5

    def test_vram_recommendation_to_dict(self) -> None:
        """Test VRAMRecommendation serialization."""
        vram = VRAMRecommendation(
            target_vram_gb=24.0,
            recommended_batch_size=4,
            max_batch_size=8,
            batch_size_breakdown={1: 12.5, 2: 15.0, 4: 20.0, 8: 30.0},
            recommended_context_length=8192,
            max_context_length=32768,
            min_gpus_required=2,
            recommended_parallelism="tensor_parallel",
            parallelism_rationale="Model too large for single GPU",
        )

        d = vram.to_dict()
        assert d["target_vram_gb"] == 24.0
        assert d["max_batch_size"] == 8
        assert d["batch_size_breakdown"]["4"] == 20.0
        assert d["min_gpus_required"] == 2

    def test_memory_analysis_result_to_dict(self) -> None:
        """Test MemoryAnalysisResult serialization."""
        result = MemoryAnalysisResult(
            model_size_bytes=10_000_000_000,  # 10GB
            model_size_gb=10.0,
            summary="Large model",
            recommendations=["Use INT8 KV cache", "Consider tensor parallelism"],
        )

        d = result.to_dict()
        assert d["model_size_bytes"] == 10_000_000_000
        assert d["model_size_gb"] == 10.0
        assert len(d["recommendations"]) == 2


# =============================================================================
# Analyzer Tests
# =============================================================================


class TestMemoryAnalyzer:
    """Tests for MemoryAnalyzer."""

    def test_analyze_basic(self) -> None:
        """Test basic analysis without attention result."""
        analyzer = MemoryAnalyzer()
        graph_info = create_mock_graph_info()

        result = analyzer.analyze(graph_info, blocks=[], vram_gb=24.0)

        assert isinstance(result, MemoryAnalysisResult)
        assert result.model_size_bytes > 0
        assert result.model_size_gb > 0

    def test_analyze_with_attention_result(self) -> None:
        """Test analysis with attention result for KV cache info."""
        analyzer = MemoryAnalyzer()
        graph_info = create_mock_graph_info()
        attn_result = create_mock_attention_result(bytes_per_token=4096)

        result = analyzer.analyze(
            graph_info,
            blocks=[],
            attention_result=attn_result,
            vram_gb=24.0,
        )

        assert result.kv_cache.bytes_per_token == 4096
        assert result.kv_cache.max_context_for_vram > 0
        assert result.kv_cache.kv_cache_percent_at_32k > 0

    def test_dtype_to_bytes(self) -> None:
        """Test dtype to bytes conversion."""
        analyzer = MemoryAnalyzer()

        # ONNX numeric types
        assert analyzer._dtype_to_bytes(1) == 4  # FLOAT
        assert analyzer._dtype_to_bytes(10) == 2  # FLOAT16
        assert analyzer._dtype_to_bytes(3) == 1  # INT8
        assert analyzer._dtype_to_bytes(16) == 2  # BFLOAT16

        # String types
        assert analyzer._dtype_to_bytes("float32") == 4
        assert analyzer._dtype_to_bytes("float16") == 2
        assert analyzer._dtype_to_bytes("int8") == 1

    def test_calculate_model_size(self) -> None:
        """Test model size calculation."""
        analyzer = MemoryAnalyzer()

        # Create mock with specific tensor sizes
        mock_tensor1 = MagicMock()
        mock_tensor1.shape = [1000, 1000]  # 1M elements
        mock_tensor1.data_type = 1  # FLOAT = 4 bytes

        mock_tensor2 = MagicMock()
        mock_tensor2.shape = [500, 500]  # 250K elements
        mock_tensor2.data_type = 10  # FLOAT16 = 2 bytes

        graph_info = create_mock_graph_info(initializers={"w1": mock_tensor1, "w2": mock_tensor2})

        size = analyzer._calculate_model_size(graph_info)

        expected = (1000 * 1000 * 4) + (500 * 500 * 2)  # 4M + 500K = 4.5M bytes
        assert size == expected


class TestKVCacheAnalysis:
    """Tests for KV cache analysis."""

    def test_detect_kv_quantization_int8(self) -> None:
        """Test INT8 KV cache detection."""
        analyzer = MemoryAnalyzer()

        # Create node with INT8 KV cache hint
        mock_node = MagicMock()
        mock_node.name = "kv_cache_int8_quantize"
        mock_node.op_type = "QuantizeLinear"

        graph_info = create_mock_graph_info(nodes=[mock_node])

        quant, detected = analyzer._detect_kv_quantization(graph_info)
        assert quant == "int8"
        assert detected is True

    def test_detect_kv_quantization_fp16(self) -> None:
        """Test FP16 detection from precision breakdown."""
        analyzer = MemoryAnalyzer()

        graph_info = create_mock_graph_info(precision_breakdown={"FP16": 80, "FP32": 20})

        quant, detected = analyzer._detect_kv_quantization(graph_info)
        assert quant == "fp16"
        assert detected is False

    def test_check_int8_kv_support(self) -> None:
        """Test INT8 KV cache support detection."""
        analyzer = MemoryAnalyzer()

        # Transformer-like model
        graph_info = create_mock_graph_info(
            op_type_counts={"MatMul": 24, "Softmax": 12, "LayerNormalization": 12}
        )

        assert analyzer._check_int8_kv_support(graph_info) is True

        # Non-transformer model
        graph_info2 = create_mock_graph_info(op_type_counts={"Conv": 20, "Relu": 20})

        assert analyzer._check_int8_kv_support(graph_info2) is False

    def test_detect_paged_attention(self) -> None:
        """Test PagedAttention pattern detection."""
        analyzer = MemoryAnalyzer()

        # Model with paged attention patterns
        mock_node1 = MagicMock()
        mock_node1.name = "paged_attention_layer"
        mock_node1.op_type = "Attention"

        mock_node2 = MagicMock()
        mock_node2.name = "block_table_lookup"
        mock_node2.op_type = "Gather"

        graph_info = create_mock_graph_info(nodes=[mock_node1, mock_node2])

        detected, indicators = analyzer._detect_paged_attention(graph_info)
        assert detected is True
        assert len(indicators) >= 1


class TestParallelismAnalysis:
    """Tests for parallelism detection."""

    def test_detect_allreduce_ops(self) -> None:
        """Test AllReduce communication op detection."""
        analyzer = MemoryAnalyzer()

        mock_node = MagicMock()
        mock_node.name = "nccl_allreduce_0"
        mock_node.op_type = "AllReduce"
        mock_node.outputs = []

        graph_info = create_mock_graph_info(nodes=[mock_node])

        result = analyzer._analyze_parallelism(graph_info, [], 10_000_000_000)

        assert len(result.communication_ops) == 1
        assert result.communication_ops[0].op_type == "all_reduce"
        assert result.detected_type == "tensor_parallel"

    def test_detect_column_parallel_patterns(self) -> None:
        """Test column parallel name pattern detection."""
        analyzer = MemoryAnalyzer()

        mock_node = MagicMock()
        mock_node.name = "layers.0.attention.column_parallel_linear"
        mock_node.op_type = "MatMul"
        mock_node.outputs = []

        graph_info = create_mock_graph_info(nodes=[mock_node])

        result = analyzer._analyze_parallelism(graph_info, [], 10_000_000_000)

        assert result.column_parallel_layers == 1
        assert len(result.indicators) >= 1

    def test_memory_per_gpu_estimates(self) -> None:
        """Test memory per GPU estimation."""
        analyzer = MemoryAnalyzer()

        # 20GB model
        result = analyzer._estimate_memory_per_gpu(20 * 1024**3)

        assert result["1"] == pytest.approx(20.0, rel=0.1)
        assert result["2"] == pytest.approx(10.5, rel=0.1)
        assert result["4"] == pytest.approx(5.5, rel=0.1)


class TestVRAMRecommendations:
    """Tests for VRAM-based recommendations."""

    def test_batch_size_recommendations(self) -> None:
        """Test batch size recommendation generation."""
        analyzer = MemoryAnalyzer()
        graph_info = create_mock_graph_info()
        attn_result = create_mock_attention_result(bytes_per_token=2048)

        result = analyzer.analyze(
            graph_info,
            blocks=[],
            attention_result=attn_result,
            vram_gb=24.0,
            batch_size=1,
            context_length=4096,
        )

        vram = result.vram_recommendation
        assert vram.target_vram_gb == 24.0
        assert vram.max_batch_size >= 1
        assert vram.recommended_batch_size >= 1

    def test_multi_gpu_recommendation(self) -> None:
        """Test multi-GPU recommendation for large models."""
        analyzer = MemoryAnalyzer()

        # Create large model (50GB)
        mock_tensor = MagicMock()
        mock_tensor.shape = [10000, 10000]  # 100M elements
        mock_tensor.data_type = 1  # FLOAT = 400MB per tensor

        initializers = {f"w{i}": mock_tensor for i in range(125)}  # ~50GB total

        graph_info = create_mock_graph_info(initializers=initializers)

        result = analyzer.analyze(graph_info, blocks=[], vram_gb=24.0)

        vram = result.vram_recommendation
        assert vram.min_gpus_required > 1
        assert vram.recommended_parallelism == "tensor_parallel"

    def test_context_length_recommendation(self) -> None:
        """Test context length recommendations."""
        analyzer = MemoryAnalyzer()
        graph_info = create_mock_graph_info()
        attn_result = create_mock_attention_result(bytes_per_token=4096)

        result = analyzer.analyze(
            graph_info,
            blocks=[],
            attention_result=attn_result,
            vram_gb=24.0,
        )

        assert result.kv_cache.max_context_for_vram > 0
        assert result.vram_recommendation.max_context_length > 0


class TestSummaryGeneration:
    """Tests for summary and recommendation generation."""

    def test_generate_summary_small_model(self) -> None:
        """Test summary for small model."""
        analyzer = MemoryAnalyzer()
        graph_info = create_mock_graph_info()

        result = analyzer.analyze(graph_info, blocks=[], vram_gb=24.0)

        assert "Model size" in result.summary
        assert result.model_size_gb < 1  # Small model

    def test_recommendations_include_int8_kv_hint(self) -> None:
        """Test that recommendations suggest INT8 KV when appropriate."""
        analyzer = MemoryAnalyzer()
        graph_info = create_mock_graph_info(
            op_type_counts={"MatMul": 24, "Softmax": 12, "LayerNormalization": 12}
        )
        attn_result = create_mock_attention_result(bytes_per_token=8192)

        result = analyzer.analyze(
            graph_info,
            blocks=[],
            attention_result=attn_result,
            vram_gb=24.0,
        )

        # Should suggest INT8 KV if cache dominates at 32k
        kv_pct = result.kv_cache.kv_cache_percent_at_32k
        if kv_pct > 50:
            assert any("INT8" in rec for rec in result.recommendations)


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Tests for enum values."""

    def test_parallelism_types(self) -> None:
        """Test ParallelismType enum values."""
        assert ParallelismType.TENSOR_PARALLEL.value == "tensor_parallel"
        assert ParallelismType.PIPELINE_PARALLEL.value == "pipeline_parallel"
        assert ParallelismType.DATA_PARALLEL.value == "data_parallel"
        assert ParallelismType.EXPERT_PARALLEL.value == "expert_parallel"

    def test_kv_cache_quantization(self) -> None:
        """Test KVCacheQuantization enum values."""
        assert KVCacheQuantization.FP16.value == "fp16"
        assert KVCacheQuantization.INT8.value == "int8"
        assert KVCacheQuantization.FP8.value == "fp8"

    def test_communication_ops(self) -> None:
        """Test CommunicationOp enum values."""
        assert CommunicationOp.ALL_REDUCE.value == "all_reduce"
        assert CommunicationOp.ALL_GATHER.value == "all_gather"
        assert CommunicationOp.REDUCE_SCATTER.value == "reduce_scatter"
