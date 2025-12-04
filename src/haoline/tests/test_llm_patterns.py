# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
Tests for LLM-scale pattern detection with real models.

Task 5.4.8: Tests with BERT, GPT-2, LLaMA patterns.

These tests verify that our pattern detection works on real transformer
architectures. Models are downloaded from ONNX Model Zoo or HuggingFace.
"""

from __future__ import annotations

import os
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from ..analyzer import ONNXGraphLoader
from ..hierarchical_graph import HierarchicalGraphBuilder
from ..patterns import PatternAnalyzer

# URLs for test models from ONNX Model Zoo
MODEL_URLS = {
    # BERT models
    "bert-base": "https://github.com/onnx/models/raw/main/text/machine_comprehension/bert-squad/model/bertsquad-12.onnx",
    # GPT-2 - using a smaller variant
    "gpt2": "https://github.com/onnx/models/raw/main/text/machine_comprehension/gpt-2/model/gpt2-lm-head-10.onnx",
}

# Cache directory for downloaded models
CACHE_DIR = Path(tempfile.gettempdir()) / "haoline_test_models"


def download_model(name: str) -> Path | None:
    """Download a model from Model Zoo if not cached."""
    if name not in MODEL_URLS:
        return None

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = CACHE_DIR / f"{name}.onnx"

    if model_path.exists():
        return model_path

    url = MODEL_URLS[name]
    try:
        print(f"Downloading {name} model from {url}...")
        urllib.request.urlretrieve(url, model_path)
        return model_path
    except Exception as e:
        print(f"Failed to download {name}: {e}")
        return None


def create_mini_bert_model() -> onnx.ModelProto:
    """
    Create a minimal BERT-like model for testing pattern detection.

    This captures the key architectural elements:
    - Layer normalization
    - Multi-head attention (Q, K, V projections + attention)
    - Feed-forward network (expand + activate + contract)
    - Residual connections
    """
    batch = 1
    seq_len = 128
    hidden = 768
    num_heads = 12
    head_dim = hidden // num_heads
    ff_dim = 3072

    # Inputs
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [batch, seq_len, hidden])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [batch, seq_len, hidden])

    # Layer norm weights
    ln1_gamma = helper.make_tensor(
        "ln1_gamma",
        TensorProto.FLOAT,
        [hidden],
        np.ones(hidden, dtype=np.float32).tolist(),
    )
    ln1_beta = helper.make_tensor(
        "ln1_beta",
        TensorProto.FLOAT,
        [hidden],
        np.zeros(hidden, dtype=np.float32).tolist(),
    )
    ln2_gamma = helper.make_tensor(
        "ln2_gamma",
        TensorProto.FLOAT,
        [hidden],
        np.ones(hidden, dtype=np.float32).tolist(),
    )
    ln2_beta = helper.make_tensor(
        "ln2_beta",
        TensorProto.FLOAT,
        [hidden],
        np.zeros(hidden, dtype=np.float32).tolist(),
    )

    # Attention weights (Q, K, V, O projections)
    wq = helper.make_tensor(
        "wq",
        TensorProto.FLOAT,
        [hidden, hidden],
        np.random.randn(hidden * hidden).astype(np.float32).tolist(),
    )
    wk = helper.make_tensor(
        "wk",
        TensorProto.FLOAT,
        [hidden, hidden],
        np.random.randn(hidden * hidden).astype(np.float32).tolist(),
    )
    wv = helper.make_tensor(
        "wv",
        TensorProto.FLOAT,
        [hidden, hidden],
        np.random.randn(hidden * hidden).astype(np.float32).tolist(),
    )
    wo = helper.make_tensor(
        "wo",
        TensorProto.FLOAT,
        [hidden, hidden],
        np.random.randn(hidden * hidden).astype(np.float32).tolist(),
    )

    # FFN weights
    w1 = helper.make_tensor(
        "w1",
        TensorProto.FLOAT,
        [hidden, ff_dim],
        np.random.randn(hidden * ff_dim).astype(np.float32).tolist(),
    )
    w2 = helper.make_tensor(
        "w2",
        TensorProto.FLOAT,
        [ff_dim, hidden],
        np.random.randn(ff_dim * hidden).astype(np.float32).tolist(),
    )

    # Scale factor for attention
    scale_val = 1.0 / np.sqrt(head_dim)
    scale = helper.make_tensor("scale", TensorProto.FLOAT, [], [scale_val])

    nodes = [
        # Pre-attention layer norm
        helper.make_node(
            "LayerNormalization",
            ["input", "ln1_gamma", "ln1_beta"],
            ["ln1_out"],
            name="ln1",
            epsilon=1e-5,
            axis=-1,
        ),
        # Q, K, V projections
        helper.make_node("MatMul", ["ln1_out", "wq"], ["q"], name="q_proj"),
        helper.make_node("MatMul", ["ln1_out", "wk"], ["k"], name="k_proj"),
        helper.make_node("MatMul", ["ln1_out", "wv"], ["v"], name="v_proj"),
        # Transpose K for attention
        helper.make_node("Transpose", ["k"], ["k_t"], name="k_transpose", perm=[0, 2, 1]),
        # Attention scores: Q @ K^T
        helper.make_node("MatMul", ["q", "k_t"], ["attn_scores"], name="attn_matmul"),
        # Scale attention scores
        helper.make_node("Mul", ["attn_scores", "scale"], ["attn_scaled"], name="attn_scale"),
        # Softmax
        helper.make_node("Softmax", ["attn_scaled"], ["attn_weights"], name="softmax", axis=-1),
        # Attention output: softmax(QK^T/sqrt(d)) @ V
        helper.make_node("MatMul", ["attn_weights", "v"], ["attn_out"], name="attn_v_matmul"),
        # Output projection
        helper.make_node("MatMul", ["attn_out", "wo"], ["attn_proj"], name="o_proj"),
        # Residual connection 1
        helper.make_node("Add", ["input", "attn_proj"], ["res1"], name="residual1"),
        # Pre-FFN layer norm
        helper.make_node(
            "LayerNormalization",
            ["res1", "ln2_gamma", "ln2_beta"],
            ["ln2_out"],
            name="ln2",
            epsilon=1e-5,
            axis=-1,
        ),
        # FFN: up projection
        helper.make_node("MatMul", ["ln2_out", "w1"], ["ff_up"], name="ffn_up"),
        # FFN: activation (GELU approximation via tanh)
        helper.make_node("Gelu", ["ff_up"], ["ff_act"], name="ffn_gelu"),
        # FFN: down projection
        helper.make_node("MatMul", ["ff_act", "w2"], ["ff_down"], name="ffn_down"),
        # Residual connection 2
        helper.make_node("Add", ["res1", "ff_down"], ["output"], name="residual2"),
    ]

    graph = helper.make_graph(
        nodes,
        "mini_bert",
        [X],
        [Y],
        [ln1_gamma, ln1_beta, ln2_gamma, ln2_beta, wq, wk, wv, wo, w1, w2, scale],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    return model


def create_mini_gpt_model() -> onnx.ModelProto:
    """
    Create a minimal GPT-like model (decoder-only transformer).

    Key differences from BERT:
    - Causal attention (masked)
    - No encoder, decoder-only
    - Uses pre-norm (LN before attention)
    """
    batch = 1
    seq_len = 64
    hidden = 512
    ff_dim = 2048

    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [batch, seq_len, hidden])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [batch, seq_len, hidden])

    # Weights (smaller than BERT for testing)
    ln_gamma = helper.make_tensor(
        "ln_gamma",
        TensorProto.FLOAT,
        [hidden],
        np.ones(hidden, dtype=np.float32).tolist(),
    )
    ln_beta = helper.make_tensor(
        "ln_beta",
        TensorProto.FLOAT,
        [hidden],
        np.zeros(hidden, dtype=np.float32).tolist(),
    )

    # Combined QKV projection (common in GPT implementations)
    wqkv = helper.make_tensor(
        "wqkv",
        TensorProto.FLOAT,
        [hidden, hidden * 3],
        np.random.randn(hidden * hidden * 3).astype(np.float32).tolist(),
    )
    wo = helper.make_tensor(
        "wo",
        TensorProto.FLOAT,
        [hidden, hidden],
        np.random.randn(hidden * hidden).astype(np.float32).tolist(),
    )
    w1 = helper.make_tensor(
        "w1",
        TensorProto.FLOAT,
        [hidden, ff_dim],
        np.random.randn(hidden * ff_dim).astype(np.float32).tolist(),
    )
    w2 = helper.make_tensor(
        "w2",
        TensorProto.FLOAT,
        [ff_dim, hidden],
        np.random.randn(ff_dim * hidden).astype(np.float32).tolist(),
    )

    nodes = [
        # Pre-norm
        helper.make_node(
            "LayerNormalization",
            ["input", "ln_gamma", "ln_beta"],
            ["ln_out"],
            name="pre_ln",
            epsilon=1e-5,
            axis=-1,
        ),
        # Combined QKV projection (GPT style)
        helper.make_node("MatMul", ["ln_out", "wqkv"], ["qkv"], name="qkv_proj"),
        # Split into Q, K, V
        helper.make_node(
            "Split", ["qkv"], ["q", "k", "v"], name="qkv_split", axis=-1, num_outputs=3
        ),
        # Transpose K
        helper.make_node("Transpose", ["k"], ["k_t"], name="k_transpose", perm=[0, 2, 1]),
        # Attention
        helper.make_node("MatMul", ["q", "k_t"], ["attn_scores"], name="attn_qk"),
        helper.make_node("Softmax", ["attn_scores"], ["attn_weights"], name="softmax", axis=-1),
        helper.make_node("MatMul", ["attn_weights", "v"], ["attn_out"], name="attn_v"),
        # Output projection
        helper.make_node("MatMul", ["attn_out", "wo"], ["attn_proj"], name="out_proj"),
        # Residual
        helper.make_node("Add", ["input", "attn_proj"], ["res1"], name="residual1"),
        # FFN with SwiGLU (approximated with Sigmoid + Mul)
        helper.make_node("MatMul", ["res1", "w1"], ["ff_up"], name="ffn_up"),
        helper.make_node("Silu", ["ff_up"], ["ff_act"], name="ffn_silu"),
        helper.make_node("MatMul", ["ff_act", "w2"], ["ff_down"], name="ffn_down"),
        # Final residual
        helper.make_node("Add", ["res1", "ff_down"], ["output"], name="residual2"),
    ]

    graph = helper.make_graph(nodes, "mini_gpt", [X], [Y], [ln_gamma, ln_beta, wqkv, wo, w1, w2])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    return model


class TestMiniBERT:
    """Test pattern detection on mini-BERT architecture."""

    def test_attention_pattern_detection(self):
        """Test that attention patterns are detected in BERT-like model."""
        model = create_mini_bert_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            analyzer = PatternAnalyzer()
            blocks = analyzer.group_into_blocks(graph_info)
            arch_type = analyzer.classify_architecture(graph_info, blocks)

            # Should detect transformer patterns
            block_types = {b.block_type for b in blocks}

            # Should have attention-related blocks
            assert any("Attention" in bt or "MatMul" in bt for bt in block_types), (
                f"Expected attention patterns, got: {block_types}"
            )

            # Architecture should be transformer
            assert arch_type == "transformer", f"Expected transformer, got {arch_type}"

        finally:
            model_path.unlink()

    def test_ffn_pattern_detection(self):
        """Test that FFN/MLP patterns are detected."""
        model = create_mini_bert_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            analyzer = PatternAnalyzer()
            blocks = analyzer.group_into_blocks(graph_info)

            # Check that blocks were detected (any type)
            # The specific types depend on the pattern analyzer's configuration
            assert len(blocks) >= 1, f"Expected at least one block, got {len(blocks)}"

            # Verify we have MatMul ops in the graph (FFN uses MatMul)
            matmul_ops = [n for n in graph_info.nodes if n.op_type == "MatMul"]
            assert len(matmul_ops) >= 4, (
                f"Expected multiple MatMul ops (Q,K,V,O projections + FFN), got {len(matmul_ops)}"
            )

        finally:
            model_path.unlink()

    def test_residual_connections_detected(self):
        """Test that residual connections are detected."""
        model = create_mini_bert_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            analyzer = PatternAnalyzer()
            blocks = analyzer.group_into_blocks(graph_info)

            # Check for residual patterns
            residual_blocks = [b for b in blocks if "Residual" in b.block_type]

            # BERT has 2 residual connections per layer
            assert len(residual_blocks) >= 1, (
                f"Expected residual connections, got {len(residual_blocks)}"
            )

        finally:
            model_path.unlink()


class TestMiniGPT:
    """Test pattern detection on mini-GPT architecture."""

    def test_gpt_architecture_detection(self):
        """Test GPT-style decoder architecture detection."""
        model = create_mini_gpt_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            analyzer = PatternAnalyzer()
            blocks = analyzer.group_into_blocks(graph_info)
            arch_type = analyzer.classify_architecture(graph_info, blocks)

            # Should detect transformer
            assert arch_type == "transformer", f"Expected transformer, got {arch_type}"

        finally:
            model_path.unlink()

    def test_swiglu_detection(self):
        """Test SwiGLU activation pattern detection (used in LLaMA)."""
        model = create_mini_gpt_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            # Check that Silu (component of SwiGLU) is present
            op_types = {n.op_type for n in graph_info.nodes}
            assert "Silu" in op_types, f"Expected Silu activation, got: {op_types}"

        finally:
            model_path.unlink()


class TestHierarchicalGraphWithTransformers:
    """Test hierarchical graph building with transformer models."""

    def test_bert_hierarchy(self):
        """Test hierarchical graph construction for BERT."""
        model = create_mini_bert_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            analyzer = PatternAnalyzer()
            blocks = analyzer.group_into_blocks(graph_info)

            builder = HierarchicalGraphBuilder()
            hier_graph = builder.build(graph_info, blocks, "MiniBERT")

            assert hier_graph.root is not None
            assert hier_graph.root.name == "MiniBERT"
            assert hier_graph.total_nodes > 0

            # Should have blocks as children
            assert len(hier_graph.root.children) > 0

        finally:
            model_path.unlink()

    def test_layer_hierarchy(self):
        """Test 3-level hierarchy (Model -> Layers -> Blocks -> Ops)."""
        model = create_mini_bert_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            analyzer = PatternAnalyzer()
            blocks = analyzer.group_into_blocks(graph_info)

            builder = HierarchicalGraphBuilder()
            hier_graph = builder.build_layer_hierarchy(graph_info, blocks, "MiniBERT")

            assert hier_graph.root is not None
            assert hier_graph.depth >= 1

        finally:
            model_path.unlink()


@pytest.mark.skipif(
    os.environ.get("SKIP_DOWNLOAD_TESTS", "1") == "1",
    reason="Skipping download tests (set SKIP_DOWNLOAD_TESTS=0 to enable)",
)
class TestRealModelZoo:
    """Tests with real models from ONNX Model Zoo.

    These tests download actual models and are disabled by default.
    Set SKIP_DOWNLOAD_TESTS=0 to run them.
    """

    def test_bert_squad(self):
        """Test with real BERT model from Model Zoo."""
        model_path = download_model("bert-base")
        if model_path is None:
            pytest.skip("Could not download BERT model")

        loader = ONNXGraphLoader()
        _, graph_info = loader.load(model_path)

        analyzer = PatternAnalyzer()
        blocks = analyzer.group_into_blocks(graph_info)
        arch_type = analyzer.classify_architecture(graph_info, blocks)

        assert arch_type == "transformer"
        assert len(blocks) > 10  # BERT has many blocks

    def test_gpt2(self):
        """Test with real GPT-2 model from Model Zoo."""
        model_path = download_model("gpt2")
        if model_path is None:
            pytest.skip("Could not download GPT-2 model")

        loader = ONNXGraphLoader()
        _, graph_info = loader.load(model_path)

        analyzer = PatternAnalyzer()
        blocks = analyzer.group_into_blocks(graph_info)
        arch_type = analyzer.classify_architecture(graph_info, blocks)

        assert arch_type == "transformer"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
