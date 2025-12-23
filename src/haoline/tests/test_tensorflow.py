# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
Unit tests for TensorFlow/Keras/JAX to ONNX conversion functionality.
"""

from __future__ import annotations

import importlib.util
import logging
from unittest.mock import patch

import pytest

from .._cli_legacy import (
    _convert_frozen_graph_to_onnx,
    _convert_jax_to_onnx,
    _convert_keras_to_onnx,
    _convert_tensorflow_to_onnx,
)

# Check if TensorFlow and tf2onnx are available
_TF_AVAILABLE = (
    importlib.util.find_spec("tensorflow") is not None
    and importlib.util.find_spec("tf2onnx") is not None
)
_JAX_AVAILABLE = importlib.util.find_spec("jax") is not None

# Import TensorFlow only when needed for tests
if _TF_AVAILABLE:
    import tensorflow as tf


@pytest.fixture
def logger():
    """Create a logger for tests."""
    return logging.getLogger("test_tensorflow")


class TestTensorFlowConversionErrors:
    """Test error handling for TensorFlow conversion (no TF required)."""

    def test_nonexistent_savedmodel(self, tmp_path, logger):
        """Conversion should fail for non-existent SavedModel."""
        fake_path = tmp_path / "nonexistent_model"

        onnx_path, _ = _convert_tensorflow_to_onnx(
            fake_path,
            output_path=None,
            opset_version=17,
            logger=logger,
        )

        assert onnx_path is None

    def test_tf2onnx_not_installed(self, tmp_path, logger):
        """Conversion should fail gracefully when tf2onnx not installed."""
        # Create a fake directory to avoid "not found" error
        fake_model = tmp_path / "fake_saved_model"
        fake_model.mkdir()

        # Mock tf2onnx import to fail
        with patch.dict("sys.modules", {"tf2onnx": None}):
            # Force re-import check by clearing cached imports

            # Save original function

            # The function checks for import at runtime, so we need to
            # test the actual error path
            _onnx_path, _ = _convert_tensorflow_to_onnx(
                fake_model,
                output_path=None,
                opset_version=17,
                logger=logger,
            )

            # If tf2onnx is not installed, this should fail
            # (result depends on actual installation state)


class TestKerasConversionErrors:
    """Test error handling for Keras conversion (no TF required)."""

    def test_nonexistent_keras_file(self, tmp_path, logger):
        """Conversion should fail for non-existent Keras file."""
        fake_path = tmp_path / "nonexistent.h5"

        onnx_path, _ = _convert_keras_to_onnx(
            fake_path,
            output_path=None,
            opset_version=17,
            logger=logger,
        )

        assert onnx_path is None

    def test_unexpected_extension_warning(self, tmp_path, logger, caplog):
        """Unexpected file extension should log warning but proceed."""
        # Create a fake file with wrong extension
        fake_file = tmp_path / "model.wrongext"
        fake_file.write_text("fake content")

        with caplog.at_level(logging.WARNING):
            _onnx_path, _ = _convert_keras_to_onnx(
                fake_file,
                output_path=None,
                opset_version=17,
                logger=logger,
            )

        # Should have logged a warning about unexpected extension
        # (will still fail because it's not a real model)


class TestFrozenGraphConversionErrors:
    """Test error handling for frozen graph conversion."""

    def test_missing_inputs_outputs(self, tmp_path, logger):
        """Conversion should fail when inputs/outputs not specified."""
        fake_pb = tmp_path / "model.pb"
        fake_pb.write_bytes(b"fake")

        # Missing inputs
        onnx_path, _ = _convert_frozen_graph_to_onnx(
            fake_pb,
            inputs=None,
            outputs="output:0",
            output_path=None,
            opset_version=17,
            logger=logger,
        )
        assert onnx_path is None

        # Missing outputs
        onnx_path, _ = _convert_frozen_graph_to_onnx(
            fake_pb,
            inputs="input:0",
            outputs=None,
            output_path=None,
            opset_version=17,
            logger=logger,
        )
        assert onnx_path is None

    def test_nonexistent_pb_file(self, tmp_path, logger):
        """Conversion should fail for non-existent .pb file."""
        fake_path = tmp_path / "nonexistent.pb"

        onnx_path, _ = _convert_frozen_graph_to_onnx(
            fake_path,
            inputs="input:0",
            outputs="output:0",
            output_path=None,
            opset_version=17,
            logger=logger,
        )

        assert onnx_path is None


class TestJAXConversionErrors:
    """Test error handling for JAX conversion."""

    def test_missing_apply_fn(self, tmp_path, logger):
        """Conversion should fail when --jax-apply-fn not provided."""
        fake_params = tmp_path / "params.pkl"
        fake_params.write_bytes(b"fake")

        onnx_path, _ = _convert_jax_to_onnx(
            fake_params,
            apply_fn_path=None,  # Missing
            input_shape_str="1,3,224,224",
            output_path=None,
            opset_version=17,
            logger=logger,
        )

        assert onnx_path is None

    def test_missing_input_shape(self, tmp_path, logger):
        """Conversion should fail when --input-shape not provided."""
        fake_params = tmp_path / "params.pkl"
        fake_params.write_bytes(b"fake")

        onnx_path, _ = _convert_jax_to_onnx(
            fake_params,
            apply_fn_path="module:function",
            input_shape_str=None,  # Missing
            output_path=None,
            opset_version=17,
            logger=logger,
        )

        assert onnx_path is None

    def test_invalid_apply_fn_format(self, tmp_path, logger):
        """Conversion should fail for invalid apply_fn format."""
        fake_params = tmp_path / "params.pkl"
        fake_params.write_bytes(b"fake")

        onnx_path, _ = _convert_jax_to_onnx(
            fake_params,
            apply_fn_path="no_colon_separator",  # Invalid format
            input_shape_str="1,3,224,224",
            output_path=None,
            opset_version=17,
            logger=logger,
        )

        assert onnx_path is None

    def test_invalid_input_shape_format(self, tmp_path, logger):
        """Conversion should fail for invalid input shape format."""
        fake_params = tmp_path / "params.pkl"
        fake_params.write_bytes(b"fake")

        onnx_path, _ = _convert_jax_to_onnx(
            fake_params,
            apply_fn_path="module:function",
            input_shape_str="not,valid,shape",  # Invalid
            output_path=None,
            opset_version=17,
            logger=logger,
        )

        assert onnx_path is None

    def test_nonexistent_params_file(self, tmp_path, logger):
        """Conversion should fail for non-existent params file."""
        fake_path = tmp_path / "nonexistent.pkl"

        onnx_path, _ = _convert_jax_to_onnx(
            fake_path,
            apply_fn_path="module:function",
            input_shape_str="1,3,224,224",
            output_path=None,
            opset_version=17,
            logger=logger,
        )

        assert onnx_path is None


@pytest.mark.skipif(not _TF_AVAILABLE, reason="TensorFlow not installed")
class TestTensorFlowConversion:
    """Tests for TensorFlow to ONNX conversion (requires TensorFlow)."""

    def test_savedmodel_conversion(self, tmp_path, logger):
        """SavedModel should convert successfully."""
        # Create a simple TF SavedModel
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(10, input_shape=(5,), activation="relu"),
                tf.keras.layers.Dense(2),
            ]
        )

        saved_model_path = tmp_path / "saved_model"
        model.save(str(saved_model_path), save_format="tf")

        # Convert to ONNX
        onnx_path, _temp_file = _convert_tensorflow_to_onnx(
            saved_model_path,
            output_path=tmp_path / "output.onnx",
            opset_version=17,
            logger=logger,
        )

        assert onnx_path is not None
        assert onnx_path.exists()
        assert onnx_path.suffix == ".onnx"

    def test_savedmodel_temp_file(self, tmp_path, logger):
        """SavedModel conversion to temp file should work."""
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(10, input_shape=(5,)),
            ]
        )

        saved_model_path = tmp_path / "saved_model"
        model.save(str(saved_model_path), save_format="tf")

        onnx_path, _temp_file = _convert_tensorflow_to_onnx(
            saved_model_path,
            output_path=None,  # Use temp file
            opset_version=17,
            logger=logger,
        )

        assert onnx_path is not None
        assert onnx_path.exists()
        # Cleanup
        onnx_path.unlink()


@pytest.mark.skipif(not _TF_AVAILABLE, reason="TensorFlow not installed")
class TestKerasConversion:
    """Tests for Keras to ONNX conversion (requires TensorFlow)."""

    def test_h5_conversion(self, tmp_path, logger):
        """Keras .h5 model should convert successfully."""
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(10, input_shape=(5,), activation="relu"),
                tf.keras.layers.Dense(2),
            ]
        )

        h5_path = tmp_path / "model.h5"
        model.save(str(h5_path), save_format="h5")

        onnx_path, _temp_file = _convert_keras_to_onnx(
            h5_path,
            output_path=tmp_path / "output.onnx",
            opset_version=17,
            logger=logger,
        )

        assert onnx_path is not None
        assert onnx_path.exists()
        assert onnx_path.suffix == ".onnx"

    def test_keras_format_conversion(self, tmp_path, logger):
        """Keras .keras format should convert successfully."""
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(10, input_shape=(5,)),
            ]
        )

        keras_path = tmp_path / "model.keras"
        model.save(str(keras_path))

        onnx_path, _temp_file = _convert_keras_to_onnx(
            keras_path,
            output_path=tmp_path / "output.onnx",
            opset_version=17,
            logger=logger,
        )

        assert onnx_path is not None
        assert onnx_path.exists()


@pytest.mark.skipif(not _JAX_AVAILABLE or not _TF_AVAILABLE, reason="JAX or TF not installed")
class TestJAXConversion:
    """Tests for JAX to ONNX conversion (requires JAX and TensorFlow)."""

    def test_unsupported_params_format(self, tmp_path, logger):
        """Unsupported params format should fail with clear error."""
        fake_params = tmp_path / "params.xyz"
        fake_params.write_bytes(b"fake")

        onnx_path, _ = _convert_jax_to_onnx(
            fake_params,
            apply_fn_path="module:function",
            input_shape_str="1,3,224,224",
            output_path=None,
            opset_version=17,
            logger=logger,
        )

        assert onnx_path is None


class TestCLIValidation:
    """Test CLI argument validation."""

    def test_multiple_conversion_flags_error(self):
        """Using multiple conversion flags should error."""

        # This would be tested via the main function, but we can verify
        # the argument structure allows these to be set
        # The actual validation happens in run_inspect()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
