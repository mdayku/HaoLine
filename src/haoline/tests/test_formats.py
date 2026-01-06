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

    # Story 49.3: Config detection tests
    def test_detect_hf_config_with_config_json(self, tmp_path: Path) -> None:
        """detect_hf_config should parse config.json if present."""
        import json

        from haoline.formats.safetensors import detect_hf_config

        # Create config.json
        config = {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        # Create dummy safetensors file
        st_path = tmp_path / "model.safetensors"
        st_path.write_bytes(b"dummy")

        result = detect_hf_config(st_path)
        assert result is not None
        assert result.model_type == "llama"
        assert result.architecture_name == "LlamaForCausalLM"
        assert result.hidden_size == 4096
        assert result.layers == 32
        assert result.heads == 32

    def test_detect_hf_config_no_config_json(self, tmp_path: Path) -> None:
        """detect_hf_config should return None if no config.json."""
        from haoline.formats.safetensors import detect_hf_config

        st_path = tmp_path / "model.safetensors"
        st_path.write_bytes(b"dummy")

        result = detect_hf_config(st_path)
        assert result is None

    def test_detect_hf_repo_id_from_cache_structure(self, tmp_path: Path) -> None:
        """detect_hf_repo_id should detect repo from HF cache directory structure."""
        from haoline.formats.safetensors import detect_hf_repo_id

        # Simulate HF cache structure: models--org--model-name/snapshots/hash/model.safetensors
        cache_dir = tmp_path / "models--meta-llama--Llama-2-7b" / "snapshots" / "abc123"
        cache_dir.mkdir(parents=True)
        st_path = cache_dir / "model.safetensors"
        st_path.write_bytes(b"dummy")

        result = detect_hf_repo_id(st_path)
        assert result == "meta-llama/Llama-2-7b"

    def test_get_safetensors_upgrade_hint_with_config(self, tmp_path: Path) -> None:
        """get_safetensors_upgrade_hint should return hint when config found."""
        import json

        from haoline.formats.safetensors import get_safetensors_upgrade_hint

        # Create config.json
        config = {"model_type": "bert", "architectures": ["BertModel"]}
        (tmp_path / "config.json").write_text(json.dumps(config))

        st_path = tmp_path / "model.safetensors"
        st_path.write_bytes(b"dummy")

        hint, cfg = get_safetensors_upgrade_hint(st_path)
        assert hint is not None
        assert "BertModel" in hint
        assert "--from-huggingface" in hint
        assert cfg is not None
        assert cfg.architecture_name == "BertModel"


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

    # Story 49.5.2: CoreML FLOP estimation tests
    def test_coreml_flop_formula_mapping_exists(self) -> None:
        """COREML_FLOP_FORMULAS should map common layer types."""
        from haoline.formats.coreml import COREML_FLOP_FORMULAS

        assert COREML_FLOP_FORMULAS["convolution"] == "conv"
        assert COREML_FLOP_FORMULAS["innerProduct"] == "matmul"
        assert COREML_FLOP_FORMULAS["softmax"] == "softmax"
        assert COREML_FLOP_FORMULAS["relu"] == "elementwise"
        assert COREML_FLOP_FORMULAS["reshape"] == "none"

    def test_coreml_info_total_flops(self) -> None:
        """Test CoreMLInfo.total_flops computed field."""
        from haoline.formats.coreml import CoreMLInfo, CoreMLLayerInfo

        info = CoreMLInfo(
            path=Path("test.mlmodel"),
            spec_version=5,
            description="Test",
            author="Test",
            license="MIT",
            layers=[
                CoreMLLayerInfo(name="conv1", type="convolution"),
                CoreMLLayerInfo(name="relu1", type="relu"),
            ],
            inputs=[{"name": "input", "type": "MultiArray(FLOAT32, [1, 3, 224, 224])"}],
            outputs=[{"name": "output", "type": "MultiArray(FLOAT32, [1, 1000])"}],
        )

        # Should have non-zero FLOPs
        assert info.total_flops > 0
        assert "convolution" in info.flops_by_layer_type
        assert "relu" in info.flops_by_layer_type


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

    # Story 49.5.1: TFLite FLOP estimation tests
    def test_flop_formula_mapping_exists(self) -> None:
        """TFLITE_FLOP_FORMULAS should map common ops to formula types."""
        from haoline.formats.tflite import TFLITE_FLOP_FORMULAS

        # Key ops should have formulas
        assert TFLITE_FLOP_FORMULAS["CONV_2D"] == "conv"
        assert TFLITE_FLOP_FORMULAS["DEPTHWISE_CONV_2D"] == "depthwise_conv"
        assert TFLITE_FLOP_FORMULAS["FULLY_CONNECTED"] == "matmul"
        assert TFLITE_FLOP_FORMULAS["SOFTMAX"] == "softmax"
        assert TFLITE_FLOP_FORMULAS["RELU"] == "elementwise"
        assert TFLITE_FLOP_FORMULAS["RESHAPE"] == "none"

    def test_tflite_op_flops_estimation(self) -> None:
        """Test FLOP estimation for TFLite operators."""
        from haoline.formats.tflite import (
            TFLiteOperatorInfo,
            TFLiteTensorInfo,
            _estimate_tflite_op_flops,
        )

        # Create mock tensors: input [1, 224, 224, 3], weight [32, 3, 3, 3], output [1, 112, 112, 32]
        tensors = [
            TFLiteTensorInfo(name="input", shape=(1, 224, 224, 3), dtype="float32", buffer_idx=0),
            TFLiteTensorInfo(name="weight", shape=(32, 3, 3, 3), dtype="float32", buffer_idx=1),
            TFLiteTensorInfo(name="output", shape=(1, 112, 112, 32), dtype="float32", buffer_idx=2),
        ]

        # Conv2D op: inputs=[0, 1], outputs=[2]
        conv_op = TFLiteOperatorInfo(
            opcode_index=0,
            builtin_code=3,
            inputs=[0, 1],
            outputs=[2],  # 3 = CONV_2D
        )
        flops = _estimate_tflite_op_flops(conv_op, tensors)

        # Expected: 2 * 3 * 3 * 3 * 32 * 112 * 112 = 21,676,032
        expected = 2 * 3 * 3 * 3 * 32 * 112 * 112
        assert flops == expected

    def test_tflite_info_total_flops(self) -> None:
        """Test TFLiteInfo.total_flops computed field."""
        from haoline.formats.tflite import TFLiteInfo, TFLiteOperatorInfo, TFLiteTensorInfo

        tensors = [
            TFLiteTensorInfo(name="input", shape=(1, 28, 28, 1), dtype="float32", buffer_idx=0),
            TFLiteTensorInfo(name="output", shape=(1, 10), dtype="float32", buffer_idx=1),
        ]
        operators = [
            TFLiteOperatorInfo(
                opcode_index=0,
                builtin_code=25,
                inputs=[0],
                outputs=[1],  # 25 = SOFTMAX
            ),
        ]

        info = TFLiteInfo(
            path=Path("test.tflite"),
            version=3,
            description="Test",
            tensors=tensors,
            operators=operators,
            inputs=[0],
            outputs=[1],
        )

        # Softmax: 5 * output_elements = 5 * 10 = 50
        assert info.total_flops == 50
        assert info.flops_by_op == {"SOFTMAX": 50}


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

    # Story 49.5.3: OpenVINO FLOP estimation tests
    def test_openvino_flop_formula_mapping_exists(self) -> None:
        """OPENVINO_FLOP_FORMULAS should map common op types."""
        from haoline.formats.openvino import OPENVINO_FLOP_FORMULAS

        assert OPENVINO_FLOP_FORMULAS["Convolution"] == "conv"
        assert OPENVINO_FLOP_FORMULAS["MatMul"] == "matmul"
        assert OPENVINO_FLOP_FORMULAS["SoftMax"] == "softmax"
        assert OPENVINO_FLOP_FORMULAS["Relu"] == "elementwise"
        assert OPENVINO_FLOP_FORMULAS["Reshape"] == "none"

    def test_openvino_info_total_flops(self) -> None:
        """Test OpenVINOInfo.total_flops computed field."""
        from haoline.formats.openvino import OpenVINOInfo, OpenVINOLayerInfo

        info = OpenVINOInfo(
            path=Path("test.xml"),
            name="test_model",
            framework="pytorch",
            layers=[
                OpenVINOLayerInfo(
                    name="conv1",
                    type="Convolution",
                    output_shapes=[(1, 32, 112, 112)],
                ),
                OpenVINOLayerInfo(
                    name="relu1",
                    type="Relu",
                    output_shapes=[(1, 32, 112, 112)],
                ),
            ],
        )

        # Should have non-zero FLOPs
        assert info.total_flops > 0
        assert "Convolution" in info.flops_by_layer_type
        assert "Relu" in info.flops_by_layer_type


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

    def test_graceful_degradation_when_trt_unavailable(self) -> None:
        """Task 52.3.2: Graceful degradation when TensorRT not installed.

        This test verifies that the TensorRT module can be imported and
        queried for availability without crashing, even when TRT is not installed.
        """
        from haoline.formats.tensorrt import is_available, is_tensorrt_file

        # These should never raise, even without TensorRT
        availability = is_available()
        assert isinstance(availability, bool)

        # File detection should work without TRT installed
        result = is_tensorrt_file(Path("test.engine"))
        assert isinstance(result, bool)

    def test_reader_import_without_tensorrt(self) -> None:
        """Task 52.3.2: Reader should be importable without TensorRT.

        The TRTEngineReader class should be importable even when TensorRT
        is not installed. Actual reading would require TRT, but import should work.
        """
        # This import should not raise, even without TensorRT
        from haoline.formats.tensorrt import TRTEngineInfo, TRTEngineReader

        # Classes should be defined
        assert TRTEngineReader is not None
        assert TRTEngineInfo is not None


class TestTRTComparison:
    """Tests for TRT comparison module."""

    def test_layer_mapping_fields(self) -> None:
        """Test LayerMapping model."""
        from haoline.formats.trt_comparison import LayerMapping

        mapping = LayerMapping(
            trt_layer_name="conv1_bn1_relu1",
            trt_layer_type="Fused",
            trt_precision="FP16",
            onnx_nodes=["conv1", "bn1", "relu1"],
            onnx_op_types=["Conv", "BatchNorm", "Relu"],
            is_fusion=True,
            fusion_description="Conv + BatchNorm + Relu",
        )
        assert mapping.is_fusion is True
        assert len(mapping.onnx_nodes) == 3
        assert mapping.trt_precision == "FP16"

    def test_precision_change_fields(self) -> None:
        """Test PrecisionChange model."""
        from haoline.formats.trt_comparison import PrecisionChange

        change = PrecisionChange(
            layer_name="conv1",
            original_precision="FP32",
            trt_precision="FP16",
            reason="TRT auto-selection",
        )
        assert change.original_precision == "FP32"
        assert change.trt_precision == "FP16"

    def test_shape_change_fields(self) -> None:
        """Test ShapeChange model."""
        from haoline.formats.trt_comparison import ShapeChange

        change = ShapeChange(
            tensor_name="input",
            onnx_shape=(-1, 3, 224, 224),
            trt_shape=(1, 3, 224, 224),
            is_dynamic_to_static=True,
        )
        assert change.is_dynamic_to_static is True
        assert change.trt_shape == (1, 3, 224, 224)

    def test_comparison_report_summary(self) -> None:
        """Test TRTComparisonReport summary generation."""
        from haoline.formats.trt_comparison import TRTComparisonReport

        report = TRTComparisonReport(
            onnx_path=Path("model.onnx"),
            trt_path=Path("model.engine"),
            onnx_node_count=100,
            trt_layer_count=25,
            fusion_count=15,
            removed_node_count=10,
        )
        summary = report.summary()
        assert "100" in summary  # ONNX nodes
        assert "25" in summary  # TRT layers
        assert "4.0x" in summary  # Compression ratio
        assert "15" in summary  # Fusions

    def test_onnx_node_info_fields(self) -> None:
        """Test ONNXNodeInfo model."""
        from haoline.formats.trt_comparison import ONNXNodeInfo

        node = ONNXNodeInfo(
            name="Conv_0",
            op_type="Conv",
            inputs=["input", "weight", "bias"],
            outputs=["conv_output"],
        )
        assert node.op_type == "Conv"
        assert len(node.inputs) == 3

    def test_memory_metrics_fields(self) -> None:
        """Test MemoryMetrics model."""
        from haoline.formats.trt_comparison import MemoryMetrics

        metrics = MemoryMetrics(
            onnx_file_size_bytes=100 * 1024 * 1024,  # 100 MB
            trt_engine_size_bytes=50 * 1024 * 1024,  # 50 MB
            trt_device_memory_bytes=80 * 1024 * 1024,  # 80 MB
            file_size_ratio=0.5,
            estimated_precision_savings_bytes=25 * 1024 * 1024,
            estimated_precision_savings_ratio=0.25,
        )
        assert metrics.file_size_ratio == 0.5
        assert metrics.estimated_precision_savings_ratio == 0.25


class TestTRTQuantBottleneckAnalysis:
    """Tests for TensorRT quantization bottleneck analysis (Story 22.8)."""

    def test_bottleneck_zone_model(self) -> None:
        """Test BottleneckZone Pydantic model."""
        from haoline.formats.tensorrt import BottleneckZone

        zone = BottleneckZone(
            start_idx=5,
            end_idx=10,
            layer_count=6,
            layer_names=["layer_5", "layer_6", "layer_7"],
            layer_types=["Conv", "BN", "ReLU"],
            severity="High",
        )
        assert zone.layer_count == 6
        assert zone.severity == "High"

    def test_failed_fusion_pattern_model(self) -> None:
        """Test FailedFusionPattern Pydantic model."""
        from haoline.formats.tensorrt import FailedFusionPattern

        pattern = FailedFusionPattern(
            pattern_type="Conv+BN+ReLU",
            layer_names=["conv1", "bn1", "relu1"],
            layer_indices=[0, 1, 2],
            expected_fused_name="Conv+BN+ReLU_0",
            reason="Sequential pattern not fused",
            speed_impact="High",
        )
        assert pattern.pattern_type == "Conv+BN+ReLU"
        assert len(pattern.layer_names) == 3

    def test_quant_bottleneck_analysis_model(self) -> None:
        """Test QuantBottleneckAnalysis Pydantic model."""
        from haoline.formats.tensorrt import BottleneckZone, QuantBottleneckAnalysis

        zone = BottleneckZone(
            start_idx=5,
            end_idx=10,
            layer_count=6,
            layer_names=["layer_5", "layer_6"],
            layer_types=["Conv", "BN"],
            severity="High",
        )

        analysis = QuantBottleneckAnalysis(
            int8_layer_count=80,
            fp16_layer_count=10,
            fp32_layer_count=10,
            total_layer_count=100,
            quantization_ratio=0.8,
            fp32_fallback_ratio=0.1,
            bottleneck_zones=[zone],
            estimated_speedup_potential=1.2,
            recommendations=["Quantization looks good!"],
        )
        assert analysis.int8_layer_count == 80
        assert analysis.quantization_ratio == 0.8
        assert analysis.largest_bottleneck is not None
        assert analysis.largest_bottleneck.layer_count == 6

    def test_analyze_quant_bottlenecks_mock_engine(self) -> None:
        """Test analyze_quant_bottlenecks with mock engine data."""
        from pathlib import Path

        from haoline.formats.tensorrt import (
            TRTEngineInfo,
            TRTLayerInfo,
            analyze_quant_bottlenecks,
        )

        # Create mock layers with mixed precision
        layers = [
            TRTLayerInfo(name="conv1", type="Convolution", precision="INT8"),
            TRTLayerInfo(name="conv2", type="Convolution", precision="INT8"),
            # FP32 bottleneck zone
            TRTLayerInfo(name="layernorm1", type="Normalization", precision="FP32"),
            TRTLayerInfo(name="add1", type="ElementWise", precision="FP32"),
            TRTLayerInfo(name="matmul1", type="MatrixMultiply", precision="FP32"),
            # Back to INT8
            TRTLayerInfo(name="conv3", type="Convolution", precision="INT8"),
            TRTLayerInfo(name="conv4", type="Convolution", precision="INT8"),
        ]

        engine_info = TRTEngineInfo(
            path=Path("test.engine"),
            trt_version="10.0.0",
            layers=layers,
        )

        analysis = analyze_quant_bottlenecks(engine_info)

        assert analysis.int8_layer_count == 4
        assert analysis.fp32_layer_count == 3
        assert analysis.total_layer_count == 7
        assert len(analysis.bottleneck_zones) >= 1
        assert analysis.estimated_speedup_potential > 1.0

    def test_bottleneck_zone_severity(self) -> None:
        """Test zone severity based on layer count."""
        from haoline.formats.tensorrt import _zone_severity

        assert _zone_severity(2) == "Low"
        assert _zone_severity(3) == "Medium"
        assert _zone_severity(5) == "High"
        assert _zone_severity(10) == "Critical"
