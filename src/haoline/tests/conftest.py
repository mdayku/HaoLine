# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
Pytest configuration and shared fixtures for HaoLine tests.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper


@pytest.fixture
def logger():
    """Create a test logger."""
    return logging.getLogger("haoline.test")


@pytest.fixture
def simple_conv_model():
    """Create a simple Conv model for testing."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 224, 224])
    W = helper.make_tensor(
        "W",
        TensorProto.FLOAT,
        [64, 3, 7, 7],
        np.random.randn(64, 3, 7, 7).astype(np.float32).flatten().tolist(),
    )
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 64, 218, 218])

    conv = helper.make_node("Conv", ["X", "W"], ["Y"], kernel_shape=[7, 7])

    graph = helper.make_graph([conv], "simple_conv", [X], [Y], [W])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx.save(model, f.name)
        yield Path(f.name)

    # Cleanup
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def conv_bn_relu_model():
    """Create a Conv-BN-ReLU model for testing."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 224, 224])
    W = helper.make_tensor(
        "W",
        TensorProto.FLOAT,
        [64, 3, 7, 7],
        np.random.randn(64, 3, 7, 7).astype(np.float32).flatten().tolist(),
    )
    scale = helper.make_tensor(
        "scale", TensorProto.FLOAT, [64], np.ones(64, dtype=np.float32).tolist()
    )
    bias = helper.make_tensor(
        "bias", TensorProto.FLOAT, [64], np.zeros(64, dtype=np.float32).tolist()
    )
    mean = helper.make_tensor(
        "mean", TensorProto.FLOAT, [64], np.zeros(64, dtype=np.float32).tolist()
    )
    var = helper.make_tensor("var", TensorProto.FLOAT, [64], np.ones(64, dtype=np.float32).tolist())
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 64, 112, 112])

    conv = helper.make_node(
        "Conv",
        ["X", "W"],
        ["conv_out"],
        kernel_shape=[7, 7],
        strides=[2, 2],
        pads=[3, 3, 3, 3],
    )
    bn = helper.make_node(
        "BatchNormalization",
        ["conv_out", "scale", "bias", "mean", "var"],
        ["bn_out"],
    )
    relu = helper.make_node("Relu", ["bn_out"], ["Y"])

    graph = helper.make_graph(
        [conv, bn, relu], "conv_bn_relu", [X], [Y], [W, scale, bias, mean, var]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx.save(model, f.name)
        yield Path(f.name)

    # Cleanup
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def matmul_model():
    """Create a simple MatMul model for testing."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 512])
    W = helper.make_tensor(
        "W",
        TensorProto.FLOAT,
        [512, 1000],
        np.random.randn(512, 1000).astype(np.float32).flatten().tolist(),
    )
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1000])

    matmul = helper.make_node("MatMul", ["X", "W"], ["Y"])

    graph = helper.make_graph([matmul], "matmul", [X], [Y], [W])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx.save(model, f.name)
        yield Path(f.name)

    # Cleanup
    Path(f.name).unlink(missing_ok=True)
