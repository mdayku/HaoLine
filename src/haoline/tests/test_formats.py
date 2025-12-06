# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""Tests for format readers (SafeTensors, GGUF, TFLite, CoreML, OpenVINO)."""

from pathlib import Path

import pytest

# ============================================================================
# Helper Functions
# ============================================================================


def _safetensors_available() -> bool:
    """Check if safetensors is available."""
    try:
        import safetensors  # noqa: F401

        return True
    except ImportError:
        return False


def _coremltools_available() -> bool:
    """Check if coremltools is available."""
    try:
        import coremltools  # noqa: F401

        return True
    except ImportError:
        return False


def _tflite_available() -> bool:
    """Check if tflite-runtime is available."""
    try:
        from tflite_runtime.interpreter import Interpreter  # noqa: F401

        return True
    except ImportError:
        return False


def _openvino_available() -> bool:
    """Check if openvino is available."""
    try:
        import openvino  # noqa: F401

        return True
    except ImportError:
        return False


# ============================================================================
# SafeTensors Reader Tests
# ============================================================================


class TestSafeTensorsReader:
    """Tests for SafeTensorsReader."""

    def test_is_available(self) -> None:
        """Check if safetensors library availability detection works."""
        from haoline.formats.safetensors import is_available

        # Should return True if safetensors is installed, False otherwise
        result = is_available()
        assert isinstance(result, bool)

    def test_is_safetensors_file_with_valid_extension(self, tmp_path: Path) -> None:
        """is_safetensors_file should return True for valid .safetensors files."""
        from haoline.formats.safetensors import is_safetensors_file

        # Create a mock safetensors file with valid header
        test_file = tmp_path / "test.safetensors"
        # SafeTensors format: 8 bytes header size + JSON header
        header = b'{"test": {"dtype": "F32", "shape": [10], "data_offsets": [0, 40]}}'
        header_size = len(header).to_bytes(8, "little")
        test_file.write_bytes(header_size + header + b"\x00" * 40)

        assert is_safetensors_file(test_file) is True

    def test_is_safetensors_file_wrong_extension(self, tmp_path: Path) -> None:
        """is_safetensors_file should return False for wrong extension."""
        from haoline.formats.safetensors import is_safetensors_file

        test_file = tmp_path / "test.onnx"
        test_file.write_bytes(b"not safetensors")
        assert is_safetensors_file(test_file) is False

    def test_is_safetensors_file_nonexistent(self) -> None:
        """is_safetensors_file should return False for non-existent files."""
        from haoline.formats.safetensors import is_safetensors_file

        assert is_safetensors_file(Path("/nonexistent/file.safetensors")) is False

    @pytest.mark.skipif(not _safetensors_available(), reason="safetensors not installed")
    def test_reader_file_not_found_raises(self) -> None:
        """SafeTensorsReader should raise FileNotFoundError for missing files."""
        from haoline.formats.safetensors import SafeTensorsReader

        with pytest.raises(FileNotFoundError):
            SafeTensorsReader("/nonexistent/model.safetensors")

    @pytest.mark.skipif(not _safetensors_available(), reason="safetensors not installed")
    def test_read_header_only(self, tmp_path: Path) -> None:
        """read_header_only should parse header without loading weights."""
        from haoline.formats.safetensors import SafeTensorsReader

        # Create a valid safetensors file
        test_file = tmp_path / "test.safetensors"
        header = b'{"weight": {"dtype": "F32", "shape": [10, 10], "data_offsets": [0, 400]}}'
        header_size = len(header).to_bytes(8, "little")
        test_file.write_bytes(header_size + header + b"\x00" * 400)

        reader = SafeTensorsReader(test_file)
        info = reader.read_header_only()

        assert len(info.tensors) == 1
        assert info.tensors[0].name == "weight"
        assert info.tensors[0].dtype == "F32"
        assert info.tensors[0].shape == (10, 10)
        assert info.total_params == 100

    def test_safetensor_info_computed_fields(self) -> None:
        """Test computed fields on SafeTensorInfo."""
        from haoline.formats.safetensors import SafeTensorInfo

        tensor = SafeTensorInfo(name="test", dtype="F32", shape=(10, 20))
        assert tensor.n_elements == 200
        assert tensor.size_bytes == 800  # 200 * 4 bytes

    def test_safetensors_info_dtype_breakdown(self) -> None:
        """Test dtype_breakdown computed field."""
        from haoline.formats.safetensors import SafeTensorInfo, SafeTensorsInfo

        info = SafeTensorsInfo(
            path=Path("test.safetensors"),
            tensors=[
                SafeTensorInfo(name="a", dtype="F32", shape=(10,)),
                SafeTensorInfo(name="b", dtype="F32", shape=(20,)),
                SafeTensorInfo(name="c", dtype="F16", shape=(30,)),
            ],
        )
        assert info.dtype_breakdown == {"F32": 2, "F16": 1}
        assert info.total_params == 60


# ============================================================================
# GGUF Reader Tests
# ============================================================================


class TestGGUFReader:
    """Tests for GGUFReader (pure Python, no external deps)."""

    def test_is_gguf_file_with_valid_magic(self, tmp_path: Path) -> None:
        """is_gguf_file should return True for files with GGUF magic."""
        from haoline.formats.gguf import is_gguf_file

        test_file = tmp_path / "test.gguf"
        # GGUF magic: "GGUF" + version (3) + tensor_count (0) + kv_count (0)
        test_file.write_bytes(b"GGUF" + b"\x03\x00\x00\x00" + b"\x00" * 16)
        assert is_gguf_file(test_file) is True

    def test_is_gguf_file_wrong_magic(self, tmp_path: Path) -> None:
        """is_gguf_file should return False for wrong magic."""
        from haoline.formats.gguf import is_gguf_file

        test_file = tmp_path / "test.gguf"
        test_file.write_bytes(b"NOTG" + b"\x00" * 20)
        assert is_gguf_file(test_file) is False

    def test_is_gguf_file_nonexistent(self) -> None:
        """is_gguf_file should return False for non-existent files."""
        from haoline.formats.gguf import is_gguf_file

        assert is_gguf_file(Path("/nonexistent/model.gguf")) is False

    def test_reader_file_not_found_raises(self) -> None:
        """GGUFReader should raise FileNotFoundError for missing files."""
        from haoline.formats.gguf import GGUFReader

        with pytest.raises(FileNotFoundError):
            GGUFReader("/nonexistent/model.gguf")

    def test_reader_invalid_magic_raises(self, tmp_path: Path) -> None:
        """GGUFReader should raise ValueError for invalid magic."""
        from haoline.formats.gguf import GGUFReader

        test_file = tmp_path / "test.gguf"
        test_file.write_bytes(b"NOTG" + b"\x00" * 100)

        reader = GGUFReader(test_file)
        with pytest.raises(ValueError, match="Not a GGUF file"):
            reader.read()

    def test_tensor_info_computed_fields(self) -> None:
        """Test computed fields on TensorInfo."""
        from haoline.formats.gguf import GGMLType, TensorInfo

        tensor = TensorInfo(
            name="test",
            n_dims=2,
            dims=(10, 20),
            type_id=GGMLType.F32,
            offset=0,
        )
        assert tensor.n_elements == 200
        assert tensor.type_name == "F32"
        assert tensor.bits_per_element == 32.0
        assert tensor.size_bytes == 800

    def test_tensor_info_quantized_type(self) -> None:
        """Test TensorInfo with quantized type."""
        from haoline.formats.gguf import GGMLType, TensorInfo

        tensor = TensorInfo(
            name="test",
            n_dims=2,
            dims=(100, 100),
            type_id=GGMLType.Q4_K,
            offset=0,
        )
        assert tensor.type_name == "Q4_K"
        assert tensor.bits_per_element == 4.5
        # 10000 elements * 4.5 bits / 8 = 5625 bytes
        assert tensor.size_bytes == 5625

    def test_gguf_info_vram_estimate(self) -> None:
        """Test VRAM estimation on GGUFInfo."""
        from haoline.formats.gguf import GGMLType, GGUFInfo, TensorInfo

        info = GGUFInfo(
            path=Path("test.gguf"),
            version=3,
            tensor_count=1,
            tensors=[
                TensorInfo(
                    name="weights",
                    n_dims=2,
                    dims=(1000, 1000),
                    type_id=GGMLType.F16,
                    offset=0,
                )
            ],
            metadata={
                "llama.block_count": 32,
                "llama.embedding_length": 4096,
                "llama.attention.head_count": 32,
                "llama.attention.head_count_kv": 8,
            },
        )

        vram = info.estimate_vram(2048)
        assert "weights" in vram
        assert "kv_cache" in vram
        assert "total" in vram
        assert vram["total"] > vram["weights"]

    def test_format_size(self) -> None:
        """Test format_size helper."""
        from haoline.formats.gguf import format_size

        assert format_size(100) == "100.00 B"
        assert format_size(1024) == "1.00 KB"
        assert format_size(1024 * 1024) == "1.00 MB"
        assert format_size(1024 * 1024 * 1024) == "1.00 GB"


# ============================================================================
# CoreML Reader Tests
# ============================================================================


class TestCoreMLReader:
    """Tests for CoreMLReader."""

    def test_is_available(self) -> None:
        """Check if coremltools availability detection works."""
        from haoline.formats.coreml import is_available

        result = is_available()
        assert isinstance(result, bool)

    def test_is_coreml_file_mlmodel(self, tmp_path: Path) -> None:
        """is_coreml_file should return True for .mlmodel files."""
        from haoline.formats.coreml import is_coreml_file

        test_file = tmp_path / "test.mlmodel"
        test_file.touch()
        assert is_coreml_file(test_file) is True

    def test_is_coreml_file_mlpackage(self, tmp_path: Path) -> None:
        """is_coreml_file should return True for .mlpackage directories."""
        from haoline.formats.coreml import is_coreml_file

        test_dir = tmp_path / "test.mlpackage"
        test_dir.mkdir()
        assert is_coreml_file(test_dir) is True

    def test_is_coreml_file_wrong_extension(self, tmp_path: Path) -> None:
        """is_coreml_file should return False for wrong extension."""
        from haoline.formats.coreml import is_coreml_file

        test_file = tmp_path / "test.onnx"
        test_file.touch()
        assert is_coreml_file(test_file) is False

    def test_is_coreml_file_nonexistent(self) -> None:
        """is_coreml_file should return False for non-existent files."""
        from haoline.formats.coreml import is_coreml_file

        assert is_coreml_file(Path("/nonexistent/model.mlmodel")) is False

    def test_coreml_info_layer_type_counts(self) -> None:
        """Test layer_type_counts computed field."""
        from haoline.formats.coreml import CoreMLInfo, CoreMLLayerInfo

        info = CoreMLInfo(
            path=Path("test.mlmodel"),
            spec_version=5,
            description="Test model",
            author="Test",
            license="MIT",
            layers=[
                CoreMLLayerInfo(name="conv1", type="convolution"),
                CoreMLLayerInfo(name="conv2", type="convolution"),
                CoreMLLayerInfo(name="relu1", type="activation"),
            ],
        )
        assert info.layer_type_counts == {"convolution": 2, "activation": 1}
        assert info.layer_count == 3


# ============================================================================
# TFLite Reader Tests
# ============================================================================


class TestTFLiteReader:
    """Tests for TFLiteReader."""

    def test_is_available(self) -> None:
        """Check if tflite-runtime availability detection works."""
        from haoline.formats.tflite import is_available

        result = is_available()
        assert isinstance(result, bool)

    def test_is_tflite_file_with_valid_identifier(self, tmp_path: Path) -> None:
        """is_tflite_file should return True for files with TFL3 identifier."""
        from haoline.formats.tflite import is_tflite_file

        test_file = tmp_path / "test.tflite"
        # TFLite format: 4 bytes root offset + "TFL3" identifier
        test_file.write_bytes(b"\x00\x00\x00\x00TFL3" + b"\x00" * 100)
        assert is_tflite_file(test_file) is True

    def test_is_tflite_file_wrong_identifier(self, tmp_path: Path) -> None:
        """is_tflite_file should return False for wrong identifier."""
        from haoline.formats.tflite import is_tflite_file

        test_file = tmp_path / "test.tflite"
        test_file.write_bytes(b"\x00\x00\x00\x00NOTF" + b"\x00" * 100)
        assert is_tflite_file(test_file) is False

    def test_is_tflite_file_nonexistent(self) -> None:
        """is_tflite_file should return False for non-existent files."""
        from haoline.formats.tflite import is_tflite_file

        assert is_tflite_file(Path("/nonexistent/model.tflite")) is False

    def test_reader_file_not_found_raises(self) -> None:
        """TFLiteReader should raise FileNotFoundError for missing files."""
        from haoline.formats.tflite import TFLiteReader

        with pytest.raises(FileNotFoundError):
            TFLiteReader("/nonexistent/model.tflite")


# ============================================================================
# OpenVINO Reader Tests
# ============================================================================


class TestOpenVINOReader:
    """Tests for OpenVINOReader."""

    def test_is_available(self) -> None:
        """Check if openvino availability detection works."""
        from haoline.formats.openvino import is_available

        result = is_available()
        assert isinstance(result, bool)

    def test_is_openvino_file_with_valid_xml(self, tmp_path: Path) -> None:
        """is_openvino_file should return True for valid OpenVINO XML."""
        from haoline.formats.openvino import is_openvino_file

        test_file = tmp_path / "test.xml"
        test_file.write_text('<net name="test" version="11">')
        assert is_openvino_file(test_file) is True

    def test_is_openvino_file_wrong_xml(self, tmp_path: Path) -> None:
        """is_openvino_file should return False for non-OpenVINO XML."""
        from haoline.formats.openvino import is_openvino_file

        test_file = tmp_path / "test.xml"
        test_file.write_text("<html><body>Not OpenVINO</body></html>")
        assert is_openvino_file(test_file) is False

    def test_is_openvino_file_nonexistent(self) -> None:
        """is_openvino_file should return False for non-existent files."""
        from haoline.formats.openvino import is_openvino_file

        assert is_openvino_file(Path("/nonexistent/model.xml")) is False

    def test_openvino_info_layer_type_counts(self) -> None:
        """Test layer_type_counts computed field."""
        from haoline.formats.openvino import OpenVINOInfo, OpenVINOLayerInfo

        info = OpenVINOInfo(
            path=Path("test.xml"),
            name="test_model",
            framework="pytorch",
            layers=[
                OpenVINOLayerInfo(name="conv1", type="Convolution"),
                OpenVINOLayerInfo(name="conv2", type="Convolution"),
                OpenVINOLayerInfo(name="relu1", type="ReLU"),
            ],
        )
        assert info.layer_type_counts == {"Convolution": 2, "ReLU": 1}
        assert info.layer_count == 3


# ============================================================================
# TensorRT Reader Tests
# ============================================================================


def _tensorrt_available() -> bool:
    """Check if tensorrt is available."""
    try:
        import tensorrt  # noqa: F401

        return True
    except ImportError:
        return False


class TestTRTEngineReader:
    """Tests for TRTEngineReader."""

    def test_is_available(self) -> None:
        """Check if tensorrt availability detection works."""
        from haoline.formats.tensorrt import is_available

        result = is_available()
        assert isinstance(result, bool)

    def test_is_tensorrt_file_engine_extension(self, tmp_path: Path) -> None:
        """is_tensorrt_file should return True for .engine files."""
        from haoline.formats.tensorrt import is_tensorrt_file

        test_file = tmp_path / "model.engine"
        test_file.write_bytes(b"\x00" * 100)
        assert is_tensorrt_file(test_file) is True

    def test_is_tensorrt_file_plan_extension(self, tmp_path: Path) -> None:
        """is_tensorrt_file should return True for .plan files."""
        from haoline.formats.tensorrt import is_tensorrt_file

        test_file = tmp_path / "model.plan"
        test_file.write_bytes(b"\x00" * 100)
        assert is_tensorrt_file(test_file) is True

    def test_is_tensorrt_file_wrong_extension(self, tmp_path: Path) -> None:
        """is_tensorrt_file should return False for wrong extension."""
        from haoline.formats.tensorrt import is_tensorrt_file

        test_file = tmp_path / "model.onnx"
        test_file.write_bytes(b"\x00" * 100)
        assert is_tensorrt_file(test_file) is False

    def test_is_tensorrt_file_nonexistent(self) -> None:
        """is_tensorrt_file should return False for non-existent files."""
        from haoline.formats.tensorrt import is_tensorrt_file

        assert is_tensorrt_file(Path("/nonexistent/model.engine")) is False

    @pytest.mark.skipif(not _tensorrt_available(), reason="tensorrt not installed")
    def test_reader_file_not_found_raises(self) -> None:
        """TRTEngineReader should raise FileNotFoundError for missing files."""
        from haoline.formats.tensorrt import TRTEngineReader

        with pytest.raises(FileNotFoundError):
            TRTEngineReader("/nonexistent/model.engine")

    def test_trt_layer_info_computed_fields(self) -> None:
        """Test TRTLayerInfo fields."""
        from haoline.formats.tensorrt import TRTLayerInfo

        layer = TRTLayerInfo(
            name="conv1_fwd + bn1_fwd + relu1_fwd",
            type="Fused:BatchNorm+Convolution+ReLU",
            precision="FP16",
            is_fused=True,
            fused_ops=["conv1_fwd", "bn1_fwd", "relu1_fwd"],
        )
        assert layer.name == "conv1_fwd + bn1_fwd + relu1_fwd"
        assert layer.type == "Fused:BatchNorm+Convolution+ReLU"
        assert layer.precision == "FP16"
        assert layer.is_fused is True
        assert len(layer.fused_ops) == 3

    def test_trt_builder_config_fields(self) -> None:
        """Test TRTBuilderConfig fields."""
        from haoline.formats.tensorrt import TRTBuilderConfig

        config = TRTBuilderConfig(
            num_io_tensors=2,
            num_layers=10,
            max_batch_size=8,
            has_implicit_batch=False,
            device_memory_size=1024 * 1024 * 256,  # 256 MB workspace
            dla_core=-1,
            num_optimization_profiles=1,
            hardware_compatibility_level="None",
            engine_capability="Standard",
        )
        assert config.max_batch_size == 8
        assert config.device_memory_size == 1024 * 1024 * 256
        assert config.dla_core == -1
        assert config.num_optimization_profiles == 1

    def test_trt_engine_info_computed_fields(self) -> None:
        """Test computed fields on TRTEngineInfo."""
        from haoline.formats.tensorrt import (
            TRTBindingInfo,
            TRTBuilderConfig,
            TRTEngineInfo,
            TRTLayerInfo,
        )

        config = TRTBuilderConfig(
            num_io_tensors=2,
            num_layers=3,
            max_batch_size=4,
            device_memory_size=1024 * 1024 * 100,
        )
        info = TRTEngineInfo(
            path=Path("test.engine"),
            trt_version="10.14.0",
            builder_config=config,
            device_name="RTX 4050",
            compute_capability=(8, 9),
            device_memory_bytes=1024 * 1024 * 50,
            layers=[
                TRTLayerInfo(name="conv1", type="Convolution", precision="FP16"),
                TRTLayerInfo(name="conv2", type="Convolution", precision="FP16"),
                TRTLayerInfo(name="pool1", type="Pooling", precision="FP32"),
            ],
            bindings=[
                TRTBindingInfo(name="input", shape=(1, 3, 224, 224), dtype="FLOAT", is_input=True),
                TRTBindingInfo(name="output", shape=(1, 1000), dtype="FLOAT", is_input=False),
            ],
        )
        assert info.layer_count == 3
        assert info.layer_type_counts == {"Convolution": 2, "Pooling": 1}
        assert info.precision_breakdown == {"FP16": 2, "FP32": 1}
        assert len(info.input_bindings) == 1
        assert len(info.output_bindings) == 1
        assert info.builder_config.max_batch_size == 4

    def test_trt_engine_info_fusion_computed_fields(self) -> None:
        """Test fusion-related computed fields on TRTEngineInfo."""
        from haoline.formats.tensorrt import TRTEngineInfo, TRTLayerInfo

        info = TRTEngineInfo(
            path=Path("test.engine"),
            trt_version="10.14.0",
            layers=[
                TRTLayerInfo(
                    name="conv1 + bn1 + relu1",
                    type="Fused",
                    is_fused=True,
                    fused_ops=["conv1", "bn1", "relu1"],
                ),
                TRTLayerInfo(
                    name="conv2 + bn2",
                    type="Fused",
                    is_fused=True,
                    fused_ops=["conv2", "bn2"],
                ),
                TRTLayerInfo(name="pool1", type="Pooling", is_fused=False),
                TRTLayerInfo(name="fc1", type="FullyConnected", is_fused=False),
            ],
        )
        assert info.fused_layer_count == 2
        assert info.fusion_ratio == 0.5  # 2 out of 4
        assert info.original_ops_fused == 5  # 3 + 2

    def test_format_bytes(self) -> None:
        """Test format_bytes helper."""
        from haoline.formats.tensorrt import format_bytes

        assert format_bytes(100) == "100.00 B"
        assert format_bytes(1024) == "1.00 KB"
        assert format_bytes(1024 * 1024) == "1.00 MB"
        assert format_bytes(1024 * 1024 * 1024) == "1.00 GB"
