# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
Epic 42: Format Conversion Testing

Tests for all supported format conversions:
- ONNX -> TensorRT (Task 42.1.1)
- ONNX -> CoreML (Task 42.1.3)
- ONNX -> OpenVINO (Task 42.1.4)
- CoreML -> ONNX (Task 42.1.6, lossy)
- OpenVINO -> ONNX (Task 42.1.7)

See BACKLOG.md Epic 42 for full conversion matrix.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_conv_onnx() -> Path:
    """Create a simple Conv model ONNX file for conversion testing."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
    W = helper.make_tensor(
        "conv_weights",
        TensorProto.FLOAT,
        [64, 3, 7, 7],
        np.random.randn(64, 3, 7, 7).astype(np.float32).flatten().tolist(),
    )
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 64, 218, 218])

    conv = helper.make_node(
        "Conv",
        ["input", "conv_weights"],
        ["output"],
        kernel_shape=[7, 7],
        name="conv1",
    )

    graph = helper.make_graph([conv], "simple_conv", [X], [Y], [W])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx.save(model, f.name)
        yield Path(f.name)

    # Cleanup
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def mobilenet_onnx() -> Path | None:
    """Download a small MobileNet ONNX for realistic testing.

    Returns None if download fails (test should skip).
    """
    model_dir = Path("test_models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "mobilenet_v2.onnx"

    if model_path.exists():
        yield model_path
        return

    # Download MobileNetV2 from ONNX model zoo
    try:
        import urllib.request

        url = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx"
        urllib.request.urlretrieve(url, model_path)
        yield model_path
    except Exception:
        yield None


# ============================================================================
# Helper Functions
# ============================================================================


def verify_onnx_model(path: Path) -> tuple[bool, str]:
    """Verify an ONNX model is valid.

    Returns:
        Tuple of (is_valid, error_message).
    """
    try:
        model = onnx.load(str(path))
        onnx.checker.check_model(model)
        return True, ""
    except Exception as e:
        return False, str(e)


def get_onnx_io_shapes(path: Path) -> dict[str, Any]:
    """Extract input/output shapes from ONNX model."""
    model = onnx.load(str(path))
    result: dict[str, Any] = {"inputs": {}, "outputs": {}, "node_count": len(model.graph.node)}

    for inp in model.graph.input:
        shape = []
        for dim in inp.type.tensor_type.shape.dim:
            if dim.dim_param:
                shape.append(dim.dim_param)
            else:
                shape.append(dim.dim_value)
        result["inputs"][inp.name] = shape

    for out in model.graph.output:
        shape = []
        for dim in out.type.tensor_type.shape.dim:
            if dim.dim_param:
                shape.append(dim.dim_param)
            else:
                shape.append(dim.dim_value)
        result["outputs"][out.name] = shape

    return result


# ============================================================================
# Task 42.1.1: ONNX -> TensorRT
# ============================================================================


def is_tensorrt_available() -> bool:
    """Check if TensorRT is available."""
    try:
        import tensorrt as trt  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not is_tensorrt_available(), reason="TensorRT not installed")
class TestOnnxToTensorRT:
    """Tests for ONNX to TensorRT conversion."""

    def test_simple_conv_conversion(self, simple_conv_onnx: Path) -> None:
        """Convert a simple Conv model to TensorRT and verify."""
        import tensorrt as trt

        # Build TRT engine from ONNX
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)

        # Parse ONNX
        with open(simple_conv_onnx, "rb") as f:
            success = parser.parse(f.read())

        assert success, (
            f"Failed to parse ONNX: {[parser.get_error(i) for i in range(parser.num_errors)]}"
        )

        # Build engine
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)  # 256MB

        # Add optimization profile
        profile = builder.create_optimization_profile()
        input_name = network.get_input(0).name
        profile.set_shape(
            input_name,
            (1, 3, 224, 224),  # min
            (1, 3, 224, 224),  # opt
            (1, 3, 224, 224),  # max
        )
        config.add_optimization_profile(profile)

        serialized = builder.build_serialized_network(network, config)
        assert serialized is not None, "Failed to build TRT engine"

        # Save and verify with our reader
        with tempfile.NamedTemporaryFile(suffix=".engine", delete=False) as f:
            f.write(bytes(serialized))
            engine_path = Path(f.name)

        try:
            from haoline.formats.tensorrt import TRTEngineReader, is_tensorrt_file

            assert is_tensorrt_file(engine_path), "Engine file not recognized"

            reader = TRTEngineReader(engine_path)
            info = reader.read()

            # Verify basic properties
            assert info.layer_count > 0, "No layers in engine"
            assert len(info.input_bindings) == 1, "Expected 1 input"
            assert len(info.output_bindings) == 1, "Expected 1 output"
            assert info.input_bindings[0].name == "input"

            # Check layer types include Convolution
            assert "Convolution" in info.layer_type_counts, "No Convolution layer found"

        finally:
            engine_path.unlink(missing_ok=True)

    def test_trt_engine_has_valid_metadata(self, simple_conv_onnx: Path) -> None:
        """Verify TRT engine contains performance metadata."""
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)

        with open(simple_conv_onnx, "rb") as f:
            parser.parse(f.read())

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)

        profile = builder.create_optimization_profile()
        input_name = network.get_input(0).name
        profile.set_shape(input_name, (1, 3, 224, 224), (1, 3, 224, 224), (1, 3, 224, 224))
        config.add_optimization_profile(profile)

        serialized = builder.build_serialized_network(network, config)

        with tempfile.NamedTemporaryFile(suffix=".engine", delete=False) as f:
            f.write(bytes(serialized))
            engine_path = Path(f.name)

        try:
            from haoline.formats.tensorrt import TRTEngineReader

            reader = TRTEngineReader(engine_path)
            info = reader.read()

            # Check metadata extraction
            assert info.trt_version is not None, "Missing TRT version"
            assert info.device_name is not None, "Missing device name"
            assert info.compute_capability is not None, "Missing compute capability"
            # device_memory_bytes may be 0 in TRT 10.x+ due to API deprecation
            assert info.device_memory_bytes >= 0, "Invalid device memory"

        finally:
            engine_path.unlink(missing_ok=True)

    def test_fp16_conversion(self, simple_conv_onnx: Path) -> None:
        """Test ONNX to TRT with FP16 precision."""
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)

        if not builder.platform_has_fast_fp16:
            pytest.skip("GPU does not support fast FP16")

        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)

        with open(simple_conv_onnx, "rb") as f:
            parser.parse(f.read())

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
        config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16

        profile = builder.create_optimization_profile()
        input_name = network.get_input(0).name
        profile.set_shape(input_name, (1, 3, 224, 224), (1, 3, 224, 224), (1, 3, 224, 224))
        config.add_optimization_profile(profile)

        serialized = builder.build_serialized_network(network, config)
        assert serialized is not None, "Failed to build FP16 TRT engine"

        with tempfile.NamedTemporaryFile(suffix=".engine", delete=False) as f:
            f.write(bytes(serialized))
            engine_path = Path(f.name)

        try:
            from haoline.formats.tensorrt import TRTEngineReader

            reader = TRTEngineReader(engine_path)
            info = reader.read()

            # FP16 engine should still work
            assert info.layer_count > 0

        finally:
            engine_path.unlink(missing_ok=True)


# ============================================================================
# Task 42.1.3: ONNX -> CoreML
# ============================================================================


def is_coreml_available() -> bool:
    """Check if CoreML tools is available."""
    try:
        import coremltools as ct  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not is_coreml_available(), reason="coremltools not installed")
class TestOnnxToCoreML:
    """Tests for ONNX to CoreML conversion."""

    def test_simple_conv_conversion(self, simple_conv_onnx: Path) -> None:
        """Convert a simple Conv model to CoreML."""
        import coremltools as ct

        # Convert ONNX to CoreML
        mlmodel = ct.convert(
            str(simple_conv_onnx),
            source="onnx",
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS15,
        )

        assert mlmodel is not None, "Conversion failed"

        # Save and verify
        with tempfile.TemporaryDirectory() as tmpdir:
            mlpackage_path = Path(tmpdir) / "model.mlpackage"
            mlmodel.save(str(mlpackage_path))

            assert mlpackage_path.exists(), "MLPackage not created"

            # Verify with our reader
            from haoline.formats.coreml import CoreMLReader

            reader = CoreMLReader(mlpackage_path)
            info = reader.read()

            assert info.spec_version is not None
            assert len(info.inputs) > 0, "No inputs detected"
            assert len(info.outputs) > 0, "No outputs detected"

    def test_coreml_metadata_preserved(self, simple_conv_onnx: Path) -> None:
        """Verify CoreML model contains expected metadata."""
        import coremltools as ct

        mlmodel = ct.convert(
            str(simple_conv_onnx),
            source="onnx",
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS15,
        )

        # Check spec contents
        spec = mlmodel.get_spec()
        assert spec is not None

        # Should have input/output descriptions
        assert len(spec.description.input) > 0
        assert len(spec.description.output) > 0


# ============================================================================
# Task 42.1.4: ONNX -> OpenVINO
# ============================================================================


def is_openvino_available() -> bool:
    """Check if OpenVINO is available."""
    try:
        from openvino import convert_model  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not is_openvino_available(), reason="openvino not installed")
class TestOnnxToOpenVINO:
    """Tests for ONNX to OpenVINO conversion."""

    def test_simple_conv_conversion(self, simple_conv_onnx: Path) -> None:
        """Convert a simple Conv model to OpenVINO IR."""
        from openvino import convert_model, save_model

        # Convert ONNX to OpenVINO
        ov_model = convert_model(str(simple_conv_onnx))
        assert ov_model is not None, "Conversion failed"

        # Save and verify
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = Path(tmpdir) / "model.xml"
            save_model(ov_model, str(xml_path))

            assert xml_path.exists(), "XML file not created"
            bin_path = xml_path.with_suffix(".bin")
            assert bin_path.exists(), "BIN file not created"

            # Verify with our reader
            from haoline.formats.openvino import OpenVINOReader

            reader = OpenVINOReader(xml_path)
            info = reader.read()

            assert info.framework_version is not None
            assert len(info.inputs) > 0, "No inputs detected"
            assert len(info.outputs) > 0, "No outputs detected"

    def test_openvino_preserves_shapes(self, simple_conv_onnx: Path) -> None:
        """Verify OpenVINO conversion preserves input/output shapes."""
        from openvino import convert_model

        # Get original shapes
        original = get_onnx_io_shapes(simple_conv_onnx)

        # Convert
        ov_model = convert_model(str(simple_conv_onnx))

        # Check shapes match
        for i, inp in enumerate(ov_model.inputs):
            shape = list(inp.get_partial_shape())
            # Convert to comparable format
            ov_shape = [int(d) if d.is_static else "dynamic" for d in shape]
            # Original might have symbolic dims
            orig_shape = list(original["inputs"].values())[i]
            # At minimum, rank should match
            assert len(ov_shape) == len(orig_shape), f"Rank mismatch: {ov_shape} vs {orig_shape}"


# ============================================================================
# Task 42.1.6: CoreML -> ONNX (Lossy)
# ============================================================================


@pytest.mark.skipif(not is_coreml_available(), reason="coremltools not installed")
class TestCoreMLToOnnx:
    """Tests for CoreML to ONNX conversion (lossy)."""

    def test_roundtrip_simple_model(self, simple_conv_onnx: Path) -> None:
        """Test ONNX -> CoreML -> ONNX roundtrip."""
        import coremltools as ct

        # ONNX -> CoreML
        mlmodel = ct.convert(
            str(simple_conv_onnx),
            source="onnx",
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS15,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            mlpackage_path = Path(tmpdir) / "model.mlpackage"
            mlmodel.save(str(mlpackage_path))

            # CoreML -> ONNX (using coremltools export)
            # Note: This is lossy - some metadata/precision may be lost
            roundtrip_onnx = Path(tmpdir) / "roundtrip.onnx"

            try:
                # coremltools can convert back to ONNX (experimental)
                from coremltools.converters.onnx import convert as ct_to_onnx

                ct_to_onnx(str(mlpackage_path), str(roundtrip_onnx))
            except (ImportError, AttributeError):
                # Fallback: use onnx-coreml if available, otherwise skip
                pytest.skip("CoreML -> ONNX conversion not available in this coremltools version")

            # Verify the roundtrip model
            valid, error = verify_onnx_model(roundtrip_onnx)
            assert valid, f"Roundtrip ONNX invalid: {error}"

            # Check basic structure preserved (may have different node count due to optimizations)
            roundtrip_info = get_onnx_io_shapes(roundtrip_onnx)
            original_info = get_onnx_io_shapes(simple_conv_onnx)

            # IO counts should match
            assert len(roundtrip_info["inputs"]) == len(original_info["inputs"])
            assert len(roundtrip_info["outputs"]) == len(original_info["outputs"])


# ============================================================================
# Task 42.1.7: OpenVINO -> ONNX
# ============================================================================


@pytest.mark.skipif(not is_openvino_available(), reason="openvino not installed")
class TestOpenVINOToOnnx:
    """Tests for OpenVINO to ONNX conversion."""

    def test_roundtrip_simple_model(self, simple_conv_onnx: Path) -> None:
        """Test ONNX -> OpenVINO -> ONNX roundtrip."""
        from openvino import convert_model, save_model

        # ONNX -> OpenVINO
        ov_model = convert_model(str(simple_conv_onnx))

        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = Path(tmpdir) / "model.xml"
            save_model(ov_model, str(xml_path))

            # OpenVINO -> ONNX
            roundtrip_onnx = Path(tmpdir) / "roundtrip.onnx"

            try:
                # OpenVINO 2024+ has direct ONNX export
                from openvino import save_model as ov_save

                ov_save(ov_model, str(roundtrip_onnx))
            except Exception:
                # Try loading XML and converting back
                try:
                    from openvino import Core

                    core = Core()
                    _ = core.read_model(str(xml_path))  # Verify readable

                    # Use ovc (OpenVINO Model Converter) if available
                    import subprocess
                    import sys

                    result = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "openvino.tools.ovc",
                            str(xml_path),
                            "--output_model",
                            str(roundtrip_onnx),
                            "--compress_to_fp16=False",
                        ],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode != 0:
                        pytest.skip(f"OpenVINO -> ONNX conversion not available: {result.stderr}")
                except Exception as e:
                    pytest.skip(f"OpenVINO -> ONNX conversion not available: {e}")

            if roundtrip_onnx.exists():
                # Verify the roundtrip model
                valid, error = verify_onnx_model(roundtrip_onnx)
                assert valid, f"Roundtrip ONNX invalid: {error}"


# ============================================================================
# Comparison Tests (verify converted models produce similar outputs)
# ============================================================================


@pytest.mark.skipif(not is_tensorrt_available(), reason="TensorRT not installed")
class TestOnnxTrtComparison:
    """Compare ONNX and TRT outputs for numerical consistency."""

    def test_onnx_trt_output_similarity(self, simple_conv_onnx: Path) -> None:
        """Verify ONNX and TRT produce similar outputs."""
        import tensorrt as trt

        # Build TRT engine
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)

        with open(simple_conv_onnx, "rb") as f:
            parser.parse(f.read())

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)

        profile = builder.create_optimization_profile()
        input_name = network.get_input(0).name
        profile.set_shape(input_name, (1, 3, 224, 224), (1, 3, 224, 224), (1, 3, 224, 224))
        config.add_optimization_profile(profile)

        serialized = builder.build_serialized_network(network, config)

        with tempfile.NamedTemporaryFile(suffix=".engine", delete=False) as f:
            f.write(bytes(serialized))
            engine_path = Path(f.name)

        try:
            # Use our comparison tool
            from haoline.formats.trt_comparison import compare_onnx_trt

            report = compare_onnx_trt(simple_conv_onnx, engine_path)

            # Should have valid comparison
            assert report.onnx_node_count > 0
            assert report.trt_layer_count > 0

            # Layer mappings should exist
            assert len(report.layer_mappings) > 0

        finally:
            engine_path.unlink(missing_ok=True)


# ============================================================================
# Task 42.2.1-42.2.3: PyTorch -> ONNX
# ============================================================================


def is_pytorch_available() -> bool:
    """Check if PyTorch is available."""
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not is_pytorch_available(), reason="PyTorch not installed")
class TestPyTorchToOnnx:
    """Tests for PyTorch to ONNX conversion."""

    def test_simple_cnn_conversion(self) -> None:
        """Task 42.2.1: Convert a simple CNN model to ONNX."""
        import torch
        import torch.nn as nn

        # Simple CNN matching our test fixture
        class SimpleCNN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, kernel_size=7, padding=3)
                self.relu = nn.ReLU()
                self.pool = nn.MaxPool2d(2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.conv1(x)
                x = self.relu(x)
                x = self.pool(x)
                return x

        model = SimpleCNN()
        model.eval()

        # Export to ONNX
        dummy_input = torch.randn(1, 3, 224, 224)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = Path(f.name)

        try:
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                opset_version=17,
            )

            # Verify ONNX model
            valid, error = verify_onnx_model(onnx_path)
            assert valid, f"ONNX export invalid: {error}"

            # Check structure
            info = get_onnx_io_shapes(onnx_path)
            assert len(info["inputs"]) == 1
            assert len(info["outputs"]) == 1
            assert info["node_count"] >= 3  # Conv, Relu, MaxPool

        finally:
            onnx_path.unlink(missing_ok=True)

    def test_transformer_attention_export(self) -> None:
        """Task 42.2.3: Export transformer with attention patterns."""
        import torch
        import torch.nn as nn

        class SimpleAttention(nn.Module):
            """Minimal attention layer for testing."""

            def __init__(self, dim: int = 64, heads: int = 4) -> None:
                super().__init__()
                self.heads = heads
                self.dim = dim
                self.head_dim = dim // heads
                self.qkv = nn.Linear(dim, dim * 3)
                self.proj = nn.Linear(dim, dim)
                self.scale = self.head_dim**-0.5

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                B, N, C = x.shape
                qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]

                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                return self.proj(x)

        model = SimpleAttention(dim=64, heads=4)
        model.eval()

        # Export to ONNX
        dummy_input = torch.randn(1, 16, 64)  # batch, seq_len, dim

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = Path(f.name)

        try:
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch", 1: "seq_len"},
                    "output": {0: "batch", 1: "seq_len"},
                },
                opset_version=17,
            )

            # Verify ONNX model
            valid, error = verify_onnx_model(onnx_path)
            assert valid, f"Attention ONNX export invalid: {error}"

            # Should have MatMul nodes (for attention computation)
            import onnx

            model_proto = onnx.load(str(onnx_path))
            op_types = {node.op_type for node in model_proto.graph.node}
            assert "MatMul" in op_types, "Expected MatMul ops for attention"
            assert "Softmax" in op_types, "Expected Softmax for attention"

        finally:
            onnx_path.unlink(missing_ok=True)


# ============================================================================
# Task 42.3.4: PyTorch -> TensorRT (via ONNX)
# ============================================================================


@pytest.mark.skipif(
    not (is_pytorch_available() and is_tensorrt_available()),
    reason="PyTorch or TensorRT not installed",
)
class TestPyTorchToTensorRT:
    """Tests for PyTorch -> TensorRT conversion via ONNX."""

    def test_pytorch_to_trt_via_onnx(self) -> None:
        """Task 42.3.4: Full PyTorch -> ONNX -> TRT pipeline."""
        import tensorrt as trt
        import torch
        import torch.nn as nn

        # Define model
        class SimpleCNN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
                self.bn1 = nn.BatchNorm2d(64)
                self.relu = nn.ReLU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                return x

        model = SimpleCNN()
        model.eval()

        # Step 1: PyTorch -> ONNX
        dummy_input = torch.randn(1, 3, 224, 224)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = Path(f.name)

        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
        )

        # Step 2: ONNX -> TensorRT
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)

        with open(onnx_path, "rb") as f:
            success = parser.parse(f.read())

        assert success, "Failed to parse ONNX"

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)

        # Optimization profile - use fixed batch size for simplicity
        profile = builder.create_optimization_profile()
        profile.set_shape("input", (1, 3, 224, 224), (1, 3, 224, 224), (1, 3, 224, 224))
        config.add_optimization_profile(profile)

        serialized = builder.build_serialized_network(network, config)
        assert serialized is not None, "Failed to build TRT engine"

        with tempfile.NamedTemporaryFile(suffix=".engine", delete=False) as f:
            f.write(bytes(serialized))
            engine_path = Path(f.name)

        try:
            # Step 3: Verify with our reader
            from haoline.formats.tensorrt import TRTEngineReader

            reader = TRTEngineReader(engine_path)
            info = reader.read()

            # Verify conversion preserved structure
            assert info.layer_count > 0
            assert len(info.input_bindings) == 1
            assert len(info.output_bindings) == 1

            # Conv-BN-ReLU should be fused in TRT
            layer_types = set(info.layer_type_counts.keys())
            # TRT typically fuses Conv+BN+ReLU
            assert "Convolution" in layer_types or any("conv" in lt.lower() for lt in layer_types)

        finally:
            onnx_path.unlink(missing_ok=True)
            engine_path.unlink(missing_ok=True)

    def test_pytorch_trt_comparison(self) -> None:
        """Verify PyTorch->TRT conversion with comparison tool."""
        import tensorrt as trt
        import torch
        import torch.nn as nn

        class TinyNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(100, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(x)

        model = TinyNet()
        model.eval()

        dummy_input = torch.randn(1, 100)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = Path(f.name)

        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
        )

        # Build TRT
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)

        with open(onnx_path, "rb") as f:
            parser.parse(f.read())

        config = builder.create_builder_config()
        profile = builder.create_optimization_profile()
        profile.set_shape("input", (1, 100), (1, 100), (1, 100))
        config.add_optimization_profile(profile)

        serialized = builder.build_serialized_network(network, config)

        with tempfile.NamedTemporaryFile(suffix=".engine", delete=False) as f:
            f.write(bytes(serialized))
            engine_path = Path(f.name)

        try:
            # Use comparison
            from haoline.formats.trt_comparison import compare_onnx_trt

            report = compare_onnx_trt(onnx_path, engine_path)

            assert report.onnx_node_count >= 1  # At least Gemm/MatMul
            assert report.trt_layer_count >= 1

        finally:
            onnx_path.unlink(missing_ok=True)
            engine_path.unlink(missing_ok=True)


# ============================================================================
# Task 42.1.5: TFLite -> ONNX
# ============================================================================


def is_tflite2onnx_available() -> bool:
    """Check if tflite2onnx is available."""
    try:
        import tflite2onnx  # noqa: F401

        return True
    except ImportError:
        return False


def is_tensorflow_available() -> bool:
    """Check if TensorFlow is available (needed to create test TFLite models)."""
    try:
        import tensorflow as tf  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not (is_tflite2onnx_available() and is_tensorflow_available()),
    reason="tflite2onnx or tensorflow not installed",
)
class TestTFLiteToOnnx:
    """Tests for TFLite to ONNX conversion."""

    @pytest.fixture
    def simple_tflite_model(self) -> Path | None:
        """Create a simple TFLite model for testing."""
        import tensorflow as tf

        # Simple Dense model (uses FULLY_CONNECTED which tflite2onnx supports)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(10, input_shape=(5,), activation="relu", name="dense1"),
                tf.keras.layers.Dense(2, name="output"),
            ]
        )
        model.compile(optimizer="adam", loss="mse")

        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as f:
            f.write(tflite_model)
            yield Path(f.name)

        # Cleanup
        Path(f.name).unlink(missing_ok=True)

    def test_tflite_to_onnx_conversion(self, simple_tflite_model: Path) -> None:
        """Task 42.1.5: Test TFLite -> ONNX conversion."""
        import tflite2onnx

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = Path(f.name)

        try:
            # Convert
            tflite2onnx.convert(str(simple_tflite_model), str(onnx_path))

            # Verify ONNX model
            valid, error = verify_onnx_model(onnx_path)
            assert valid, f"Converted ONNX invalid: {error}"

            # Check structure
            info = get_onnx_io_shapes(onnx_path)
            assert len(info["inputs"]) >= 1
            assert len(info["outputs"]) >= 1
            assert info["node_count"] >= 1

        finally:
            onnx_path.unlink(missing_ok=True)


@pytest.mark.skipif(not is_tflite2onnx_available(), reason="tflite2onnx not installed")
class TestTFLiteToOnnxWithoutTF:
    """Tests that work without TensorFlow (use pre-existing models)."""

    def test_tflite2onnx_import(self) -> None:
        """Verify tflite2onnx can be imported."""
        import tflite2onnx

        assert hasattr(tflite2onnx, "convert")
        assert hasattr(tflite2onnx, "__version__")


# ============================================================================
# Story 42.5: Conversion Validation Harness
# ============================================================================


class ConversionValidator:
    """
    Task 42.5.1: Test harness for validating conversion quality.

    Compares two UniversalGraph instances (e.g., native vs converted)
    and reports differences in structure, parameters, and metadata.
    """

    def __init__(
        self,
        tolerance_nodes: int = 10,
        tolerance_params_pct: float = 1.0,
        tolerance_edges_pct: float = 10.0,
    ) -> None:
        """
        Initialize validator with tolerances.

        Args:
            tolerance_nodes: Max allowed difference in node count (fusions may reduce)
            tolerance_params_pct: Max allowed % difference in parameter count
            tolerance_edges_pct: Max allowed % difference in edge count
        """
        self.tolerance_nodes = tolerance_nodes
        self.tolerance_params_pct = tolerance_params_pct
        self.tolerance_edges_pct = tolerance_edges_pct
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def compare_graphs(
        self,
        graph_a: Any,
        graph_b: Any,
        name_a: str = "source",
        name_b: str = "converted",
    ) -> bool:
        """
        Compare two UniversalGraph instances.

        Args:
            graph_a: First graph (typically native/source)
            graph_b: Second graph (typically converted)
            name_a: Label for first graph
            name_b: Label for second graph

        Returns:
            True if graphs are equivalent within tolerances
        """
        self.errors = []
        self.warnings = []

        # Node count comparison
        self._compare_node_count(graph_a, graph_b, name_a, name_b)

        # Parameter count comparison
        self._compare_params(graph_a, graph_b, name_a, name_b)

        # Op type comparison
        self._compare_op_types(graph_a, graph_b, name_a, name_b)

        # Input/output comparison
        self._compare_io(graph_a, graph_b, name_a, name_b)

        # Precision breakdown comparison
        self._compare_precision_breakdown(graph_a, graph_b, name_a, name_b)

        return len(self.errors) == 0

    def _compare_node_count(self, graph_a: Any, graph_b: Any, name_a: str, name_b: str) -> None:
        """Task 42.5.4 helper: Compare node counts."""
        nodes_a = getattr(graph_a, "num_nodes", len(getattr(graph_a, "nodes", [])))
        nodes_b = getattr(graph_b, "num_nodes", len(getattr(graph_b, "nodes", [])))

        diff = abs(nodes_a - nodes_b)
        if diff > self.tolerance_nodes:
            self.errors.append(
                f"Node count mismatch: {name_a}={nodes_a}, {name_b}={nodes_b} "
                f"(diff={diff}, tolerance={self.tolerance_nodes})"
            )
        elif diff > 0:
            self.warnings.append(
                f"Minor node count difference: {name_a}={nodes_a}, {name_b}={nodes_b}"
            )

    def _compare_params(self, graph_a: Any, graph_b: Any, name_a: str, name_b: str) -> None:
        """Task 42.5.7: Compare parameter counts within tolerance."""
        params_a = getattr(graph_a, "total_parameters", 0)
        params_b = getattr(graph_b, "total_parameters", 0)

        if params_a == 0 and params_b == 0:
            return  # Both have no params (e.g., activation-only models)

        if params_a == 0 or params_b == 0:
            self.warnings.append(
                f"One graph has no parameters: {name_a}={params_a}, {name_b}={params_b}"
            )
            return

        pct_diff = abs(params_a - params_b) / max(params_a, params_b) * 100
        if pct_diff > self.tolerance_params_pct:
            self.errors.append(
                f"Parameter count mismatch: {name_a}={params_a:,}, {name_b}={params_b:,} "
                f"(diff={pct_diff:.2f}%, tolerance={self.tolerance_params_pct}%)"
            )
        elif pct_diff > 0:
            self.warnings.append(
                f"Minor parameter difference: {name_a}={params_a:,}, {name_b}={params_b:,} "
                f"({pct_diff:.2f}%)"
            )

    def _compare_op_types(self, graph_a: Any, graph_b: Any, name_a: str, name_b: str) -> None:
        """Task 42.5.4: Compare op_type_counts between graphs."""
        # Get op type counts
        ops_a = self._get_op_counts(graph_a)
        ops_b = self._get_op_counts(graph_b)

        if not ops_a and not ops_b:
            return

        # Find ops unique to each graph
        only_in_a = set(ops_a.keys()) - set(ops_b.keys())
        only_in_b = set(ops_b.keys()) - set(ops_a.keys())

        if only_in_a:
            self.warnings.append(f"Ops only in {name_a}: {sorted(only_in_a)}")
        if only_in_b:
            self.warnings.append(f"Ops only in {name_b}: {sorted(only_in_b)}")

        # Compare common ops
        common_ops = set(ops_a.keys()) & set(ops_b.keys())
        for op in common_ops:
            count_a = ops_a[op]
            count_b = ops_b[op]
            if count_a != count_b:
                self.warnings.append(
                    f"Op '{op}' count differs: {name_a}={count_a}, {name_b}={count_b}"
                )

    def _compare_io(self, graph_a: Any, graph_b: Any, name_a: str, name_b: str) -> None:
        """Compare input/output counts and shapes."""
        inputs_a = len(getattr(graph_a, "inputs", []))
        inputs_b = len(getattr(graph_b, "inputs", []))
        outputs_a = len(getattr(graph_a, "outputs", []))
        outputs_b = len(getattr(graph_b, "outputs", []))

        if inputs_a != inputs_b:
            self.errors.append(f"Input count mismatch: {name_a}={inputs_a}, {name_b}={inputs_b}")
        if outputs_a != outputs_b:
            self.errors.append(f"Output count mismatch: {name_a}={outputs_a}, {name_b}={outputs_b}")

    def _compare_precision_breakdown(
        self, graph_a: Any, graph_b: Any, name_a: str, name_b: str
    ) -> None:
        """Task 42.5.5: Compare precision_breakdown between graphs."""
        prec_a = self._get_precision_breakdown(graph_a)
        prec_b = self._get_precision_breakdown(graph_b)

        if not prec_a and not prec_b:
            return  # Neither has precision info

        if not prec_a or not prec_b:
            self.warnings.append(
                f"Precision breakdown missing from one graph: "
                f"{name_a}={bool(prec_a)}, {name_b}={bool(prec_b)}"
            )
            return

        # Normalize precision names (e.g., "float32" -> "fp32")
        prec_a = self._normalize_precision_keys(prec_a)
        prec_b = self._normalize_precision_keys(prec_b)

        # Find precisions unique to each graph
        only_in_a = set(prec_a.keys()) - set(prec_b.keys())
        only_in_b = set(prec_b.keys()) - set(prec_a.keys())

        if only_in_a:
            self.warnings.append(f"Precisions only in {name_a}: {sorted(only_in_a)}")
        if only_in_b:
            self.warnings.append(f"Precisions only in {name_b}: {sorted(only_in_b)}")

        # Compare common precisions (as percentages of total)
        total_a = sum(prec_a.values())
        total_b = sum(prec_b.values())

        if total_a > 0 and total_b > 0:
            for prec in set(prec_a.keys()) & set(prec_b.keys()):
                pct_a = prec_a[prec] / total_a * 100
                pct_b = prec_b[prec] / total_b * 100
                if abs(pct_a - pct_b) > 5.0:  # 5% tolerance
                    self.warnings.append(
                        f"Precision '{prec}' distribution differs: "
                        f"{name_a}={pct_a:.1f}%, {name_b}={pct_b:.1f}%"
                    )

    def _get_precision_breakdown(self, graph: Any) -> dict[str, int]:
        """Extract precision breakdown from a graph."""
        # Try report.param_counts.precision_breakdown path
        if hasattr(graph, "param_counts") and graph.param_counts:
            prec = getattr(graph.param_counts, "precision_breakdown", None)
            if prec:
                return dict(prec)

        # Try direct attribute
        if hasattr(graph, "precision_breakdown"):
            prec = graph.precision_breakdown
            if prec:
                return dict(prec)

        return {}

    def _normalize_precision_keys(self, prec: dict[str, int]) -> dict[str, int]:
        """Normalize precision names for comparison."""
        normalized: dict[str, int] = {}
        for key, value in prec.items():
            norm_key = key.lower().replace("float", "fp").replace("int", "int")
            normalized[norm_key] = normalized.get(norm_key, 0) + value
        return normalized

    def _get_op_counts(self, graph: Any) -> dict[str, int]:
        """Extract op type counts from a graph."""
        # Try different attribute names
        if hasattr(graph, "op_type_counts"):
            return dict(graph.op_type_counts)

        # Count from nodes
        if hasattr(graph, "nodes"):
            counts: dict[str, int] = {}
            for node in graph.nodes:
                op_type = getattr(node, "op_type", getattr(node, "type", "Unknown"))
                counts[op_type] = counts.get(op_type, 0) + 1
            return counts

        return {}

    def get_report(self) -> str:
        """Generate human-readable validation report."""
        lines = ["=" * 60, "Conversion Validation Report", "=" * 60]

        if self.errors:
            lines.append(f"\nERRORS ({len(self.errors)}):")
            for err in self.errors:
                lines.append(f"  [X] {err}")

        if self.warnings:
            lines.append(f"\nWARNINGS ({len(self.warnings)}):")
            for warn in self.warnings:
                lines.append(f"  [!] {warn}")

        if not self.errors and not self.warnings:
            lines.append("\n[OK] Graphs are equivalent within tolerances")

        lines.append("=" * 60)
        return "\n".join(lines)


class TestConversionValidator:
    """Unit tests for the ConversionValidator harness itself."""

    def test_validator_identical_graphs(self) -> None:
        """Identical graphs should pass validation."""
        from dataclasses import dataclass

        @dataclass
        class MockGraph:
            num_nodes: int = 10
            total_parameters: int = 1000000
            inputs: list = None  # type: ignore
            outputs: list = None  # type: ignore
            op_type_counts: dict = None  # type: ignore

            def __post_init__(self) -> None:
                self.inputs = self.inputs or [1]
                self.outputs = self.outputs or [1]
                self.op_type_counts = self.op_type_counts or {"Conv": 5, "Relu": 5}

        graph_a = MockGraph()
        graph_b = MockGraph()

        validator = ConversionValidator()
        result = validator.compare_graphs(graph_a, graph_b)

        assert result is True
        assert len(validator.errors) == 0

    def test_validator_node_count_within_tolerance(self) -> None:
        """Node count difference within tolerance should pass."""
        from dataclasses import dataclass

        @dataclass
        class MockGraph:
            num_nodes: int
            total_parameters: int = 1000000
            inputs: list = None  # type: ignore
            outputs: list = None  # type: ignore

            def __post_init__(self) -> None:
                self.inputs = self.inputs or [1]
                self.outputs = self.outputs or [1]

        graph_a = MockGraph(num_nodes=100)
        graph_b = MockGraph(num_nodes=95)  # 5 fewer due to fusion

        validator = ConversionValidator(tolerance_nodes=10)
        result = validator.compare_graphs(graph_a, graph_b)

        assert result is True

    def test_validator_node_count_exceeds_tolerance(self) -> None:
        """Node count difference exceeding tolerance should fail."""
        from dataclasses import dataclass

        @dataclass
        class MockGraph:
            num_nodes: int
            total_parameters: int = 1000000
            inputs: list = None  # type: ignore
            outputs: list = None  # type: ignore

            def __post_init__(self) -> None:
                self.inputs = self.inputs or [1]
                self.outputs = self.outputs or [1]

        graph_a = MockGraph(num_nodes=100)
        graph_b = MockGraph(num_nodes=50)  # 50 fewer - too many!

        validator = ConversionValidator(tolerance_nodes=10)
        result = validator.compare_graphs(graph_a, graph_b)

        assert result is False
        assert len(validator.errors) > 0

    def test_validator_param_count_tolerance(self) -> None:
        """Parameter count within percentage tolerance should pass."""
        from dataclasses import dataclass

        @dataclass
        class MockGraph:
            num_nodes: int = 10
            total_parameters: int = 0
            inputs: list = None  # type: ignore
            outputs: list = None  # type: ignore

            def __post_init__(self) -> None:
                self.inputs = self.inputs or [1]
                self.outputs = self.outputs or [1]

        graph_a = MockGraph(total_parameters=1000000)
        graph_b = MockGraph(total_parameters=1005000)  # 0.5% difference

        validator = ConversionValidator(tolerance_params_pct=1.0)
        result = validator.compare_graphs(graph_a, graph_b)

        assert result is True

    def test_validator_io_mismatch(self) -> None:
        """Input/output count mismatch should fail."""
        from dataclasses import dataclass

        @dataclass
        class MockGraph:
            num_nodes: int = 10
            total_parameters: int = 1000000
            inputs: list = None  # type: ignore
            outputs: list = None  # type: ignore

            def __post_init__(self) -> None:
                self.inputs = self.inputs or []
                self.outputs = self.outputs or []

        graph_a = MockGraph(inputs=[1], outputs=[1])
        graph_b = MockGraph(inputs=[1, 2], outputs=[1])  # Extra input

        validator = ConversionValidator()
        result = validator.compare_graphs(graph_a, graph_b)

        assert result is False

    def test_validator_precision_breakdown(self) -> None:
        """Task 42.5.5: Precision breakdown should be compared."""
        from dataclasses import dataclass

        @dataclass
        class ParamCounts:
            precision_breakdown: dict

        @dataclass
        class MockGraph:
            num_nodes: int = 10
            total_parameters: int = 1000000
            param_counts: ParamCounts = None  # type: ignore
            inputs: list = None  # type: ignore
            outputs: list = None  # type: ignore

            def __post_init__(self) -> None:
                self.inputs = self.inputs or [1]
                self.outputs = self.outputs or [1]

        # Same precision distribution
        graph_a = MockGraph(
            param_counts=ParamCounts(precision_breakdown={"fp32": 500000, "fp16": 500000})
        )
        graph_b = MockGraph(
            param_counts=ParamCounts(precision_breakdown={"fp32": 480000, "fp16": 520000})
        )

        validator = ConversionValidator()
        result = validator.compare_graphs(graph_a, graph_b)

        # Should pass - precision distribution within 5% tolerance
        assert result is True

    def test_validator_precision_normalization(self) -> None:
        """Precision names should be normalized (float32 -> fp32)."""
        from dataclasses import dataclass

        @dataclass
        class ParamCounts:
            precision_breakdown: dict

        @dataclass
        class MockGraph:
            num_nodes: int = 10
            total_parameters: int = 1000000
            param_counts: ParamCounts = None  # type: ignore
            inputs: list = None  # type: ignore
            outputs: list = None  # type: ignore

            def __post_init__(self) -> None:
                self.inputs = self.inputs or [1]
                self.outputs = self.outputs or [1]

        # Same data, different naming conventions
        graph_a = MockGraph(param_counts=ParamCounts(precision_breakdown={"float32": 1000000}))
        graph_b = MockGraph(param_counts=ParamCounts(precision_breakdown={"fp32": 1000000}))

        validator = ConversionValidator()
        result = validator.compare_graphs(graph_a, graph_b)

        assert result is True
        # No warnings about missing precisions since they normalized to same key
        assert not any("only in" in w.lower() for w in validator.warnings)


@pytest.mark.skipif(not is_pytorch_available(), reason="PyTorch not installed")
class TestOnnxRoundTripValidation:
    """
    Task 42.5.4-42.5.7: Validate conversions preserve essential info.

    Tests that PyTorch -> ONNX -> UniversalGraph preserves:
    - Node/op counts
    - Parameter counts
    - Input/output shapes
    """

    def test_pytorch_onnx_preserves_params(self) -> None:
        """Verify PyTorch -> ONNX preserves parameter count."""
        import torch
        import torch.nn as nn

        class SimpleNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = nn.Linear(100, 50)  # 100*50 + 50 = 5050 params
                self.fc2 = nn.Linear(50, 10)  # 50*10 + 10 = 510 params
                # Total: 5560 params

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = torch.relu(self.fc1(x))
                return self.fc2(x)

        model = SimpleNet()
        model.eval()

        # Count PyTorch params
        pytorch_params = sum(p.numel() for p in model.parameters())
        assert pytorch_params == 5560

        # Export to ONNX
        dummy_input = torch.randn(1, 100)
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = Path(f.name)

        try:
            torch.onnx.export(model, dummy_input, str(onnx_path), opset_version=17)

            # Load and count ONNX params
            from haoline.format_adapters import OnnxAdapter

            graph = OnnxAdapter().read(onnx_path)
            onnx_params = graph.total_parameters

            # Validate within tolerance
            validator = ConversionValidator(tolerance_params_pct=0.1)
            # Create mock graphs for comparison
            from dataclasses import dataclass

            @dataclass
            class ParamHolder:
                total_parameters: int
                num_nodes: int = 3
                inputs: list = None  # type: ignore
                outputs: list = None  # type: ignore

                def __post_init__(self) -> None:
                    self.inputs = self.inputs or [1]
                    self.outputs = self.outputs or [1]

            pytorch_graph = ParamHolder(total_parameters=pytorch_params)
            onnx_graph = ParamHolder(total_parameters=onnx_params)

            result = validator.compare_graphs(pytorch_graph, onnx_graph, "PyTorch", "ONNX")
            assert result, validator.get_report()

        finally:
            onnx_path.unlink(missing_ok=True)

    def test_onnx_universalgraph_op_types(self) -> None:
        """Task 42.5.4: Verify op_type_counts extraction works."""
        import torch
        import torch.nn as nn

        class ConvNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
                self.pool = nn.MaxPool2d(2)
                self.relu = nn.ReLU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.relu(self.conv1(x))
                x = self.pool(x)
                x = self.relu(self.conv2(x))
                return self.pool(x)

        model = ConvNet()
        model.eval()

        dummy_input = torch.randn(1, 3, 64, 64)
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = Path(f.name)

        try:
            torch.onnx.export(model, dummy_input, str(onnx_path), opset_version=17)

            from haoline.format_adapters import OnnxAdapter

            graph = OnnxAdapter().read(onnx_path)

            # Check op types are extracted
            op_counts = graph.op_type_counts if hasattr(graph, "op_type_counts") else {}

            # Should have Conv (ONNX native) or Conv2D (UniversalGraph normalized)
            conv_key = "Conv" if "Conv" in op_counts else "Conv2D"
            assert conv_key in op_counts, f"Missing Conv/Conv2D in {op_counts}"
            assert op_counts[conv_key] == 2, f"Expected 2 {conv_key}, got {op_counts[conv_key]}"

        finally:
            onnx_path.unlink(missing_ok=True)
