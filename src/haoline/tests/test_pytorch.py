# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
Unit tests for PyTorch to ONNX conversion functionality.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add parent path for test imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from .._cli_legacy import (
    _convert_pytorch_to_onnx,
    _extract_ultralytics_metadata,
)
from ..report import DatasetInfo, infer_num_classes_from_output

# Check if torch is available
try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    nn = None  # type: ignore[assignment]  # Placeholder when torch unavailable


# Only define the model class if torch is available
if _TORCH_AVAILABLE:

    class SimpleTestModel(nn.Module):
        """Simple model for testing conversion."""

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, padding=1)
            self.relu = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(16, 10)

        def forward(self, x):
            x = self.relu(self.conv(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
class TestPyTorchConversion:
    """Tests for PyTorch to ONNX conversion."""

    def test_torchscript_model_conversion(self, tmp_path):
        """TorchScript models should convert successfully."""
        # Create and save a TorchScript model
        model = SimpleTestModel()
        model.eval()
        dummy_input = torch.randn(1, 3, 32, 32)
        traced = torch.jit.trace(model, dummy_input)

        pt_path = tmp_path / "model.pt"
        torch.jit.save(traced, str(pt_path))

        logger = logging.getLogger("test")

        # Convert
        onnx_path, _temp_file = _convert_pytorch_to_onnx(
            pt_path,
            input_shape_str="1,3,32,32",
            output_path=tmp_path / "output.onnx",
            opset_version=17,
            logger=logger,
        )

        assert onnx_path is not None
        assert onnx_path.exists()
        assert onnx_path.suffix == ".onnx"

    def test_conversion_requires_input_shape(self, tmp_path):
        """Conversion should fail without input shape."""
        model = SimpleTestModel()
        model.eval()
        dummy_input = torch.randn(1, 3, 32, 32)
        traced = torch.jit.trace(model, dummy_input)

        pt_path = tmp_path / "model.pt"
        torch.jit.save(traced, str(pt_path))

        logger = logging.getLogger("test")

        # Convert without input shape
        onnx_path, _ = _convert_pytorch_to_onnx(
            pt_path,
            input_shape_str=None,  # No input shape
            output_path=None,
            opset_version=17,
            logger=logger,
        )

        assert onnx_path is None

    def test_conversion_invalid_input_shape(self, tmp_path):
        """Conversion should fail with invalid input shape format."""
        model = SimpleTestModel()
        model.eval()
        dummy_input = torch.randn(1, 3, 32, 32)
        traced = torch.jit.trace(model, dummy_input)

        pt_path = tmp_path / "model.pt"
        torch.jit.save(traced, str(pt_path))

        logger = logging.getLogger("test")

        # Convert with invalid input shape
        onnx_path, _ = _convert_pytorch_to_onnx(
            pt_path,
            input_shape_str="invalid,shape",
            output_path=None,
            opset_version=17,
            logger=logger,
        )

        assert onnx_path is None

    def test_conversion_nonexistent_file(self, tmp_path):
        """Conversion should fail gracefully for nonexistent file."""
        logger = logging.getLogger("test")

        onnx_path, _ = _convert_pytorch_to_onnx(
            tmp_path / "nonexistent.pt",
            input_shape_str="1,3,32,32",
            output_path=None,
            opset_version=17,
            logger=logger,
        )

        assert onnx_path is None

    def test_conversion_temp_file_cleanup(self, tmp_path):
        """Temp file should be created when no output path specified."""
        model = SimpleTestModel()
        model.eval()
        dummy_input = torch.randn(1, 3, 32, 32)
        traced = torch.jit.trace(model, dummy_input)

        pt_path = tmp_path / "model.pt"
        torch.jit.save(traced, str(pt_path))

        logger = logging.getLogger("test")

        # Convert without output path (should create temp file)
        onnx_path, temp_file = _convert_pytorch_to_onnx(
            pt_path,
            input_shape_str="1,3,32,32",
            output_path=None,  # No output path
            opset_version=17,
            logger=logger,
        )

        assert onnx_path is not None
        assert temp_file is not None
        assert onnx_path.exists()

        # Clean up
        onnx_path.unlink()

    def test_state_dict_not_supported(self, tmp_path):
        """State dict models should fail with helpful error."""
        model = SimpleTestModel()

        # Save as state_dict (not full model)
        pt_path = tmp_path / "weights.pth"
        torch.save(model.state_dict(), pt_path)

        logger = logging.getLogger("test")

        onnx_path, _ = _convert_pytorch_to_onnx(
            pt_path,
            input_shape_str="1,3,32,32",
            output_path=None,
            opset_version=17,
            logger=logger,
        )

        assert onnx_path is None


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
class TestUltralyticsMetadataExtraction:
    """Tests for Ultralytics metadata extraction."""

    def test_extraction_without_ultralytics(self, tmp_path):
        """Should return None gracefully when ultralytics not available."""
        logger = logging.getLogger("test")

        # Mock ultralytics not being available
        with patch.dict("sys.modules", {"ultralytics": None}):
            result = _extract_ultralytics_metadata(tmp_path / "fake.pt", logger)
            # Should return None, not crash
            assert result is None or isinstance(result, dict)


class TestDatasetInfo:
    """Tests for DatasetInfo dataclass."""

    def test_dataset_info_creation(self):
        """DatasetInfo should be created with expected fields."""
        info = DatasetInfo(
            task="detect",
            num_classes=5,
            class_names=["cat", "dog", "bird", "fish", "car"],
            source="ultralytics",
        )

        assert info.task == "detect"
        assert info.num_classes == 5
        assert len(info.class_names) == 5
        assert info.source == "ultralytics"

    def test_dataset_info_defaults(self):
        """DatasetInfo should have sensible defaults."""
        info = DatasetInfo()

        assert info.task is None
        assert info.num_classes is None
        assert info.class_names == []
        assert info.source is None


class TestInferNumClassesFromOutput:
    """Tests for infer_num_classes_from_output function (Task 4B.2.2)."""

    def test_classification_2d_output(self):
        """Should detect classification from [batch, num_classes] shape."""
        output_shapes = {"output": [1, 1000]}  # ImageNet-style
        result = infer_num_classes_from_output(output_shapes)

        assert result is not None
        assert result.task == "classify"
        assert result.num_classes == 1000
        assert result.source == "output_shape"

    def test_classification_3d_output(self):
        """Should detect classification from [batch, 1, num_classes] shape."""
        output_shapes = {"logits": [1, 1, 100]}  # CIFAR-100 style
        result = infer_num_classes_from_output(output_shapes)

        assert result is not None
        assert result.task == "classify"
        assert result.num_classes == 100
        assert result.source == "output_shape"

    def test_detection_yolo_output(self):
        """Should detect detection from YOLO-style [batch, boxes, 4+nc] shape."""
        # YOLOv8 output: [1, 8400, 84] for 80 COCO classes + 4 box coords
        output_shapes = {"output0": [1, 8400, 84]}
        result = infer_num_classes_from_output(output_shapes)

        assert result is not None
        assert result.task == "detect"
        assert result.num_classes == 80  # 84 - 4 = 80
        assert result.source == "output_shape"

    def test_segmentation_output(self):
        """Should detect segmentation from [batch, num_classes, h, w] shape."""
        output_shapes = {"output": [1, 21, 512, 512]}  # Pascal VOC style
        result = infer_num_classes_from_output(output_shapes)

        assert result is not None
        assert result.task == "segment"
        assert result.num_classes == 21
        assert result.source == "output_shape"

    def test_empty_output_shapes(self):
        """Should return None for empty output shapes."""
        result = infer_num_classes_from_output({})
        assert result is None

    def test_single_output_dimension(self):
        """Should return None for single-dimension outputs."""
        output_shapes = {"output": [10]}
        result = infer_num_classes_from_output(output_shapes)
        assert result is None

    def test_symbolic_dimensions(self):
        """Should handle symbolic dimensions gracefully."""
        output_shapes = {"output": ["batch", 1000]}
        result = infer_num_classes_from_output(output_shapes)

        assert result is not None
        assert result.task == "classify"
        assert result.num_classes == 1000

    def test_priority_output_names(self):
        """Should prioritize outputs with known names like 'logits'."""
        output_shapes = {
            "some_random_output": [1, 5],  # Would infer 5 classes
            "logits": [1, 100],  # Should prefer this
        }
        result = infer_num_classes_from_output(output_shapes)

        assert result is not None
        assert result.num_classes == 100

    def test_num_classes_too_small(self):
        """Should not infer if num_classes is too small (< 2)."""
        output_shapes = {"output": [1, 1]}  # Only 1 class - not valid
        result = infer_num_classes_from_output(output_shapes)
        assert result is None

    def test_num_classes_too_large(self):
        """Should not infer if num_classes is too large (> 10000)."""
        output_shapes = {"output": [1, 50000]}  # Unlikely to be classes
        result = infer_num_classes_from_output(output_shapes)
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
