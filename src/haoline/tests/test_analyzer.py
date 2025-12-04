# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
Unit tests for the analyzer module (parameter counting, FLOP estimation, memory estimates).

These tests use programmatically-created tiny ONNX models to ensure deterministic,
reproducible test results without external dependencies.
"""

from __future__ import annotations

# Import the modules under test
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from ..analyzer import MetricsEngine, ONNXGraphLoader


def create_simple_conv_model() -> onnx.ModelProto:
    """Create a minimal Conv model for testing."""
    # Input: [1, 3, 8, 8]
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])

    # Weight: [16, 3, 3, 3] = 16 * 3 * 3 * 3 = 432 params
    W = helper.make_tensor(
        "W",
        TensorProto.FLOAT,
        [16, 3, 3, 3],
        np.random.randn(16, 3, 3, 3).astype(np.float32).flatten().tolist(),
    )

    # Bias: [16] = 16 params
    B = helper.make_tensor(
        "B",
        TensorProto.FLOAT,
        [16],
        np.zeros(16, dtype=np.float32).tolist(),
    )

    # Output: [1, 16, 6, 6]
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, 6, 6])

    conv_node = helper.make_node(
        "Conv",
        inputs=["X", "W", "B"],
        outputs=["Y"],
        kernel_shape=[3, 3],
        pads=[0, 0, 0, 0],
    )

    graph = helper.make_graph(
        [conv_node],
        "conv_test",
        [X],
        [Y],
        [W, B],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return model


def create_matmul_model() -> onnx.ModelProto:
    """Create a minimal MatMul model for testing."""
    # A: [2, 4, 8]
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 4, 8])

    # B weight: [8, 16] = 128 params
    B = helper.make_tensor(
        "B",
        TensorProto.FLOAT,
        [8, 16],
        np.random.randn(8, 16).astype(np.float32).flatten().tolist(),
    )

    # Output: [2, 4, 16]
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4, 16])

    matmul_node = helper.make_node(
        "MatMul",
        inputs=["A", "B"],
        outputs=["Y"],
    )

    graph = helper.make_graph(
        [matmul_node],
        "matmul_test",
        [A],
        [Y],
        [B],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return model


def create_gemm_model() -> onnx.ModelProto:
    """Create a minimal Gemm model for testing."""
    # A: [4, 8]
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [4, 8])

    # B weight: [8, 16] = 128 params
    B = helper.make_tensor(
        "B",
        TensorProto.FLOAT,
        [8, 16],
        np.random.randn(8, 16).astype(np.float32).flatten().tolist(),
    )

    # C bias: [16] = 16 params
    C = helper.make_tensor(
        "C",
        TensorProto.FLOAT,
        [16],
        np.zeros(16, dtype=np.float32).tolist(),
    )

    # Output: [4, 16]
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 16])

    gemm_node = helper.make_node(
        "Gemm",
        inputs=["A", "B", "C"],
        outputs=["Y"],
    )

    graph = helper.make_graph(
        [gemm_node],
        "gemm_test",
        [A],
        [Y],
        [B, C],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return model


def create_relu_model() -> onnx.ModelProto:
    """Create a minimal ReLU model (no parameters) for testing."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 8, 8])

    relu_node = helper.make_node("Relu", inputs=["X"], outputs=["Y"])

    graph = helper.make_graph([relu_node], "relu_test", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return model


def create_conv_bn_relu_model() -> onnx.ModelProto:
    """Create a Conv-BatchNorm-ReLU sequence for pattern testing."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])

    # Conv weights: [16, 3, 3, 3] = 432 params
    W = helper.make_tensor(
        "W",
        TensorProto.FLOAT,
        [16, 3, 3, 3],
        np.random.randn(16, 3, 3, 3).astype(np.float32).flatten().tolist(),
    )

    # BN params: scale, bias, mean, var each [16] = 64 params total
    scale = helper.make_tensor(
        "scale", TensorProto.FLOAT, [16], np.ones(16, dtype=np.float32).tolist()
    )
    bias = helper.make_tensor(
        "bias", TensorProto.FLOAT, [16], np.zeros(16, dtype=np.float32).tolist()
    )
    mean = helper.make_tensor(
        "mean", TensorProto.FLOAT, [16], np.zeros(16, dtype=np.float32).tolist()
    )
    var = helper.make_tensor("var", TensorProto.FLOAT, [16], np.ones(16, dtype=np.float32).tolist())

    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, 6, 6])

    conv_out = "conv_out"
    bn_out = "bn_out"

    conv_node = helper.make_node("Conv", ["X", "W"], [conv_out], kernel_shape=[3, 3])
    bn_node = helper.make_node(
        "BatchNormalization",
        [conv_out, "scale", "bias", "mean", "var"],
        [bn_out],
    )
    relu_node = helper.make_node("Relu", [bn_out], ["Y"])

    graph = helper.make_graph(
        [conv_node, bn_node, relu_node],
        "conv_bn_relu_test",
        [X],
        [Y],
        [W, scale, bias, mean, var],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return model


class TestONNXGraphLoader:
    """Tests for ONNXGraphLoader class."""

    def test_load_conv_model(self):
        """Test loading a simple Conv model."""
        model = create_simple_conv_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _loaded_model, graph_info = loader.load(model_path)

            assert graph_info.num_nodes == 1
            assert len(graph_info.inputs) == 1
            assert len(graph_info.outputs) == 1
            assert len(graph_info.initializers) == 2  # W and B
            assert "Conv" in graph_info.op_type_counts
        finally:
            model_path.unlink()

    def test_load_extracts_shapes(self):
        """Test that shape information is extracted correctly."""
        model = create_simple_conv_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            assert "X" in graph_info.input_shapes
            assert graph_info.input_shapes["X"] == [1, 3, 8, 8]
            assert "Y" in graph_info.output_shapes
            assert graph_info.output_shapes["Y"] == [1, 16, 6, 6]
        finally:
            model_path.unlink()


class TestMetricsEngine:
    """Tests for MetricsEngine class."""

    def test_count_parameters_conv(self):
        """Test parameter counting for Conv model."""
        model = create_simple_conv_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            engine = MetricsEngine()
            params = engine.count_parameters(graph_info)

            # W: 16*3*3*3 = 432, B: 16 = 448 total
            assert params.total == 448
            assert "Conv" in params.by_op_type
        finally:
            model_path.unlink()

    def test_count_parameters_matmul(self):
        """Test parameter counting for MatMul model."""
        model = create_matmul_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            engine = MetricsEngine()
            params = engine.count_parameters(graph_info)

            # B: 8*16 = 128
            assert params.total == 128
        finally:
            model_path.unlink()

    def test_count_parameters_no_weights(self):
        """Test parameter counting for model without weights."""
        model = create_relu_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            engine = MetricsEngine()
            params = engine.count_parameters(graph_info)

            assert params.total == 0
        finally:
            model_path.unlink()

    def test_estimate_flops_conv(self):
        """Test FLOP estimation for Conv model."""
        model = create_simple_conv_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            engine = MetricsEngine()
            flops = engine.estimate_flops(graph_info)

            # Conv FLOPs: 2 * K_h * K_w * C_in * C_out * H_out * W_out
            # = 2 * 3 * 3 * 3 * 16 * 6 * 6 + bias = 31,104 + 576 = 31,680
            expected_flops = 2 * 3 * 3 * 3 * 16 * 6 * 6 + 16 * 6 * 6
            assert flops.total == expected_flops
        finally:
            model_path.unlink()

    def test_estimate_flops_matmul(self):
        """Test FLOP estimation for MatMul model."""
        model = create_matmul_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            engine = MetricsEngine()
            flops = engine.estimate_flops(graph_info)

            # MatMul FLOPs: 2 * batch * M * N * K
            # = 2 * 2 * 4 * 16 * 8 = 2048
            expected_flops = 2 * 2 * 4 * 16 * 8
            assert flops.total == expected_flops
        finally:
            model_path.unlink()

    def test_estimate_memory(self):
        """Test memory estimation."""
        model = create_simple_conv_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            engine = MetricsEngine()
            memory = engine.estimate_memory(graph_info)

            # Model size: 448 params * 4 bytes = 1792 bytes
            assert memory.model_size_bytes == 448 * 4
            assert memory.peak_activation_bytes >= 0
        finally:
            model_path.unlink()


class TestMetricsEngineEdgeCases:
    """Edge case tests for MetricsEngine."""

    def test_gemm_with_bias(self):
        """Test Gemm with bias adds extra FLOPs."""
        model = create_gemm_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            engine = MetricsEngine()
            params = engine.count_parameters(graph_info)
            flops = engine.estimate_flops(graph_info)

            # B: 8*16=128, C: 16 = 144 total params
            assert params.total == 144

            # Gemm FLOPs: 2*M*N*K + M*N (bias) = 2*4*16*8 + 4*16 = 1024 + 64 = 1088
            expected_flops = 2 * 4 * 16 * 8 + 4 * 16
            assert flops.total == expected_flops
        finally:
            model_path.unlink()


def create_transformer_like_model() -> onnx.ModelProto:
    """Create a minimal transformer-like model with Softmax (for KV cache testing)."""
    # Input: [1, 128, 768] - batch, seq, hidden
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 128, 768])

    # QKV projection weights: [768, 768] = 589,824 params each
    Wq = helper.make_tensor(
        "Wq",
        TensorProto.FLOAT,
        [768, 768],
        np.random.randn(768, 768).astype(np.float32).flatten().tolist(),
    )
    Wk = helper.make_tensor(
        "Wk",
        TensorProto.FLOAT,
        [768, 768],
        np.random.randn(768, 768).astype(np.float32).flatten().tolist(),
    )
    Wv = helper.make_tensor(
        "Wv",
        TensorProto.FLOAT,
        [768, 768],
        np.random.randn(768, 768).astype(np.float32).flatten().tolist(),
    )

    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 128, 768])

    # Create Q, K, V projections
    q_node = helper.make_node("MatMul", ["X", "Wq"], ["Q"])
    k_node = helper.make_node("MatMul", ["X", "Wk"], ["K"])
    v_node = helper.make_node("MatMul", ["X", "Wv"], ["V"])

    # Softmax for attention scores
    softmax_node = helper.make_node("Softmax", ["Q"], ["attn_scores"], axis=-1)

    # Output projection (simplified - just use V as output for testing)
    add_node = helper.make_node("Add", ["attn_scores", "V"], ["Y"])

    graph = helper.make_graph(
        [q_node, k_node, v_node, softmax_node, add_node],
        "transformer_test",
        [X],
        [Y],
        [Wq, Wk, Wv],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return model


class TestKVCacheEstimation:
    """Tests for KV cache estimation in transformer models."""

    def test_kv_cache_detected_for_transformer(self):
        """Test that KV cache is estimated for transformer-like models."""
        model = create_transformer_like_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            engine = MetricsEngine()
            memory = engine.estimate_memory(graph_info)

            # Should detect KV cache for transformer model
            assert memory.kv_cache_bytes_per_token > 0
            assert memory.kv_cache_bytes_full_context > 0
            assert "num_layers" in memory.kv_cache_config
            assert "hidden_dim" in memory.kv_cache_config
        finally:
            model_path.unlink()

    def test_kv_cache_not_detected_for_cnn(self):
        """Test that KV cache is not estimated for CNN models."""
        model = create_simple_conv_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            engine = MetricsEngine()
            memory = engine.estimate_memory(graph_info)

            # Should NOT detect KV cache for CNN model
            assert memory.kv_cache_bytes_per_token == 0
            assert memory.kv_cache_bytes_full_context == 0
        finally:
            model_path.unlink()

    def test_kv_cache_formula(self):
        """Test KV cache per-token formula: 2 * num_layers * hidden_dim * bytes."""
        model = create_transformer_like_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            engine = MetricsEngine()
            memory = engine.estimate_memory(graph_info)

            config = memory.kv_cache_config
            if config:
                # Verify formula: 2 * layers * hidden * bytes_per_elem
                expected_per_token = (
                    2 * config["num_layers"] * config["hidden_dim"] * config["bytes_per_elem"]
                )
                assert memory.kv_cache_bytes_per_token == expected_per_token

                # Full context = per_token * seq_len
                expected_full = expected_per_token * config["seq_len"]
                assert memory.kv_cache_bytes_full_context == expected_full
        finally:
            model_path.unlink()

    def test_memory_estimates_to_dict_includes_kv_cache(self):
        """Test that to_dict includes KV cache info when present."""
        model = create_transformer_like_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            engine = MetricsEngine()
            memory = engine.estimate_memory(graph_info)

            result = memory.to_dict()
            assert "kv_cache_bytes_per_token" in result
            assert "kv_cache_bytes_full_context" in result
            assert "kv_cache_config" in result
        finally:
            model_path.unlink()


def create_shared_weights_model() -> onnx.ModelProto:
    """Create a model with shared weights (same weight used by two nodes)."""
    # Input: [1, 8]
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8])

    # Shared weight: [8, 16] = 128 params
    W_shared = helper.make_tensor(
        "W_shared",
        TensorProto.FLOAT,
        [8, 16],
        np.random.randn(8, 16).astype(np.float32).flatten().tolist(),
    )

    # Output: [1, 16]
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16])

    # Two MatMul nodes using the same weight
    matmul1 = helper.make_node("MatMul", ["X", "W_shared"], ["hidden"], name="MatMul1")
    matmul2 = helper.make_node("MatMul", ["hidden", "W_shared"], ["Y"], name="MatMul2")

    graph = helper.make_graph(
        [matmul1, matmul2],
        "shared_weights_test",
        [X],
        [Y],
        [W_shared],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return model


def create_int8_weights_model() -> onnx.ModelProto:
    """Create a model with INT8 quantized weights."""
    # Input: [1, 8]
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8])

    # INT8 weight: [8, 16] = 128 params
    W_int8 = helper.make_tensor(
        "W_int8",
        TensorProto.INT8,
        [8, 16],
        np.random.randint(-128, 127, (8, 16), dtype=np.int8).flatten().tolist(),
    )

    # Scale and zero point for dequantization
    scale = helper.make_tensor("scale", TensorProto.FLOAT, [], [0.01])
    zero_point = helper.make_tensor("zero_point", TensorProto.INT8, [], [0])

    # Output: [1, 16]
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16])

    # Dequantize the weight, then MatMul
    dequant_node = helper.make_node(
        "DequantizeLinear", ["W_int8", "scale", "zero_point"], ["W_float"]
    )
    matmul_node = helper.make_node("MatMul", ["X", "W_float"], ["Y"])

    graph = helper.make_graph(
        [dequant_node, matmul_node],
        "int8_weights_test",
        [X],
        [Y],
        [W_int8, scale, zero_point],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return model


def create_mixed_precision_model() -> onnx.ModelProto:
    """Create a model with mixed precision weights (fp32, fp16)."""
    # Input: [1, 8]
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8])

    # FP32 weight: [8, 16] = 128 params
    W_fp32 = helper.make_tensor(
        "W_fp32",
        TensorProto.FLOAT,
        [8, 16],
        np.random.randn(8, 16).astype(np.float32).flatten().tolist(),
    )

    # FP16 weight: [16, 8] = 128 params
    W_fp16 = helper.make_tensor(
        "W_fp16",
        TensorProto.FLOAT16,
        [16, 8],
        np.random.randn(16, 8).astype(np.float16).flatten().tolist(),
    )

    # Output: [1, 8]
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8])

    # Two MatMul nodes with different precision weights
    matmul1 = helper.make_node("MatMul", ["X", "W_fp32"], ["hidden"])
    cast_node = helper.make_node("Cast", ["W_fp16"], ["W_fp16_casted"], to=TensorProto.FLOAT)
    matmul2 = helper.make_node("MatMul", ["hidden", "W_fp16_casted"], ["Y"])

    graph = helper.make_graph(
        [matmul1, cast_node, matmul2],
        "mixed_precision_test",
        [X],
        [Y],
        [W_fp32, W_fp16],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return model


class TestSharedWeights:
    """Tests for shared weight handling (Task 2.2.4)."""

    def test_shared_weights_detected(self):
        """Test that shared weights are correctly detected."""
        model = create_shared_weights_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            engine = MetricsEngine()
            params = engine.count_parameters(graph_info)

            # Should detect 1 shared weight
            assert params.num_shared_weights == 1
            assert "W_shared" in params.shared_weights
            assert len(params.shared_weights["W_shared"]) == 2
        finally:
            model_path.unlink()

    def test_shared_weights_fractional_attribution(self):
        """Test that shared weights use fractional attribution."""
        model = create_shared_weights_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            engine = MetricsEngine()
            params = engine.count_parameters(graph_info)

            # Total should still be 128 (8*16)
            assert params.total == 128

            # by_op_type should sum to 128 (fractional attribution)
            op_type_sum = sum(params.by_op_type.values())
            assert abs(op_type_sum - 128) < 0.01  # Allow floating point tolerance

            # MatMul should have the full 128 (64 + 64 from two nodes)
            assert "MatMul" in params.by_op_type
            assert abs(params.by_op_type["MatMul"] - 128) < 0.01
        finally:
            model_path.unlink()

    def test_no_shared_weights_normal_model(self):
        """Test that normal models report 0 shared weights."""
        model = create_simple_conv_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            engine = MetricsEngine()
            params = engine.count_parameters(graph_info)

            # Should have no shared weights
            assert params.num_shared_weights == 0
            assert len(params.shared_weights) == 0
        finally:
            model_path.unlink()


class TestQuantizedParams:
    """Tests for quantized parameter detection (Task 2.2.4)."""

    def test_int8_weights_detected(self):
        """Test that INT8 weights are detected."""
        model = create_int8_weights_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            engine = MetricsEngine()
            params = engine.count_parameters(graph_info)

            # Should detect quantization
            assert params.is_quantized is True
            assert "DequantizeLinear" in params.quantized_ops
        finally:
            model_path.unlink()

    def test_precision_breakdown(self):
        """Test that precision breakdown is computed correctly."""
        model = create_int8_weights_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            engine = MetricsEngine()
            params = engine.count_parameters(graph_info)

            # Should have precision breakdown
            assert len(params.precision_breakdown) > 0
            # INT8 weight: 8*16 = 128 params + zero_point (1) = 129
            assert "int8" in params.precision_breakdown
            assert params.precision_breakdown["int8"] == 129
        finally:
            model_path.unlink()

    def test_mixed_precision_breakdown(self):
        """Test precision breakdown for mixed precision model."""
        model = create_mixed_precision_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            engine = MetricsEngine()
            params = engine.count_parameters(graph_info)

            # Should have precision breakdown with both precisions
            assert "fp32" in params.precision_breakdown
            assert "fp16" in params.precision_breakdown
            assert params.precision_breakdown["fp32"] == 128  # 8*16
            assert params.precision_breakdown["fp16"] == 128  # 16*8

            # Total should be 256
            assert params.total == 256
        finally:
            model_path.unlink()

    def test_non_quantized_model(self):
        """Test that non-quantized models are not marked as quantized."""
        model = create_simple_conv_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            engine = MetricsEngine()
            params = engine.count_parameters(graph_info)

            # Should not be quantized
            assert params.is_quantized is False
            assert len(params.quantized_ops) == 0
        finally:
            model_path.unlink()

    def test_param_counts_to_dict_includes_new_fields(self):
        """Test that to_dict includes shared weights and quantization info."""
        model = create_int8_weights_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            engine = MetricsEngine()
            params = engine.count_parameters(graph_info)

            result = params.to_dict()

            # Check new fields exist
            assert "shared_weights" in result
            assert "count" in result["shared_weights"]
            assert "details" in result["shared_weights"]
            assert "precision_breakdown" in result
            assert "is_quantized" in result
            assert "quantized_ops" in result
        finally:
            model_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
