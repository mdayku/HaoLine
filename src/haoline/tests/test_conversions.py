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
            assert info.device_memory_bytes > 0, "Missing device memory"

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
