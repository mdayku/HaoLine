"""
Tests for quantization_advisor.py normalization functions and advisor logic.

These tests specifically cover the LLM response normalization to prevent
Pydantic validation errors when LLM returns nested/malformed structures.
"""

from haoline.quantization_advisor import (
    _extract_string_from_nested,
    _normalize_runtime_recs,
    _normalize_str_list,
)

# =============================================================================
# _extract_string_from_nested tests
# =============================================================================


class TestExtractStringFromNested:
    """Tests for _extract_string_from_nested helper."""

    def test_simple_string(self) -> None:
        """Direct string passthrough."""
        assert _extract_string_from_nested("hello world") == "hello world"

    def test_empty_string(self) -> None:
        """Empty string returns empty."""
        assert _extract_string_from_nested("") == ""

    def test_none_value(self) -> None:
        """None returns empty string."""
        assert _extract_string_from_nested(None) == ""

    def test_dict_with_recommendation_key(self) -> None:
        """Dict with 'recommendation' key extracts that value."""
        data = {"recommendation": "Use INT8 quantization"}
        assert _extract_string_from_nested(data) == "Use INT8 quantization"

    def test_dict_with_nested_recommendation(self) -> None:
        """Nested recommendation dict is recursively extracted."""
        data = {
            "recommendation": {"description": "Enable per-channel quantization for best accuracy"}
        }
        assert (
            _extract_string_from_nested(data) == "Enable per-channel quantization for best accuracy"
        )

    def test_dict_with_deeply_nested_structure(self) -> None:
        """Deeply nested dicts are flattened to string."""
        data = {
            "recommendation": {
                "settings": "Use --int8 --fp16 flags",
                "notes": "Calibrate with 500 images",
            }
        }
        result = _extract_string_from_nested(data)
        assert "Use --int8 --fp16 flags" in result
        assert "Calibrate with 500 images" in result

    def test_dict_with_description_key(self) -> None:
        """Dict with 'description' key (no 'recommendation')."""
        data = {"description": "This is a description"}
        assert _extract_string_from_nested(data) == "This is a description"

    def test_dict_with_text_key(self) -> None:
        """Dict with 'text' key."""
        data = {"text": "Some text content"}
        assert _extract_string_from_nested(data) == "Some text content"

    def test_dict_fallback_concatenation(self) -> None:
        """Dict without priority keys concatenates all values."""
        data = {"a": "first", "b": "second"}
        result = _extract_string_from_nested(data)
        assert "first" in result
        assert "second" in result

    def test_list_of_strings(self) -> None:
        """List of strings joined with comma."""
        data = ["item1", "item2", "item3"]
        assert _extract_string_from_nested(data) == "item1, item2, item3"

    def test_integer_value(self) -> None:
        """Integer converted to string."""
        assert _extract_string_from_nested(42) == "42"


# =============================================================================
# _normalize_str_list tests
# =============================================================================


class TestNormalizeStrList:
    """Tests for _normalize_str_list function."""

    def test_none_returns_empty_list(self) -> None:
        """None input returns empty list."""
        assert _normalize_str_list(None) == []

    def test_simple_string_list(self) -> None:
        """Simple list of strings passes through."""
        data = ["layer1", "layer2", "layer3"]
        assert _normalize_str_list(data) == ["layer1", "layer2", "layer3"]

    def test_dict_with_layer_names_key(self) -> None:
        """Dict with 'layer_names' key extracts the list."""
        data = {"layer_names": ["/model/conv1", "/model/conv2"]}
        assert _normalize_str_list(data) == ["/model/conv1", "/model/conv2"]

    def test_dict_with_layers_key(self) -> None:
        """Dict with 'layers' key extracts the list."""
        data = {"layers": ["layer_a", "layer_b"]}
        assert _normalize_str_list(data) == ["layer_a", "layer_b"]

    def test_list_of_dicts_with_name(self) -> None:
        """List of dicts with 'name' key extracts names."""
        data = [{"name": "layer1"}, {"name": "layer2"}]
        assert _normalize_str_list(data) == ["layer1", "layer2"]

    def test_list_of_dicts_with_layer(self) -> None:
        """List of dicts with 'layer' key extracts layer names."""
        data = [{"layer": "conv1"}, {"layer": "conv2"}]
        assert _normalize_str_list(data) == ["conv1", "conv2"]

    def test_mixed_list(self) -> None:
        """Mixed list of strings and dicts."""
        data = ["layer1", {"name": "layer2"}, "layer3"]
        assert _normalize_str_list(data) == ["layer1", "layer2", "layer3"]

    def test_single_string(self) -> None:
        """Single string wrapped in list."""
        assert _normalize_str_list("single_layer") == ["single_layer"]

    def test_nested_layer_names_in_list(self) -> None:
        """List item with nested layer_names."""
        data = [{"layer_names": ["a", "b"]}, "c"]
        assert _normalize_str_list(data) == ["a", "b", "c"]

    def test_empty_list(self) -> None:
        """Empty list returns empty list."""
        assert _normalize_str_list([]) == []

    def test_real_llm_response_sensitive_layers(self) -> None:
        """
        Actual failing case from production:
        LLM returned {'layer_names': [...]} instead of [...]
        """
        data = {
            "layer_names": [
                "/model/0/conv1",
                "/model/1/bn1",
                "/model/22/dfl/Softmax",
            ]
        }
        result = _normalize_str_list(data)
        assert result == [
            "/model/0/conv1",
            "/model/1/bn1",
            "/model/22/dfl/Softmax",
        ]


# =============================================================================
# _normalize_runtime_recs tests
# =============================================================================


class TestNormalizeRuntimeRecs:
    """Tests for _normalize_runtime_recs function."""

    def test_none_returns_empty_dict(self) -> None:
        """None input returns empty dict."""
        assert _normalize_runtime_recs(None) == {}

    def test_non_dict_returns_empty_dict(self) -> None:
        """Non-dict input returns empty dict."""
        assert _normalize_runtime_recs("not a dict") == {}
        assert _normalize_runtime_recs(["list"]) == {}

    def test_simple_string_values(self) -> None:
        """Simple dict with string values passes through."""
        data = {
            "tensorrt": "Use INT8 with FP16 fallback",
            "onnxruntime": "Enable per-channel quantization",
        }
        assert _normalize_runtime_recs(data) == data

    def test_nested_recommendation_dict(self) -> None:
        """Dict with nested 'recommendation' key is flattened."""
        data = {
            "tensorrt": {"recommendation": "Use trtexec with --int8"},
            "onnxruntime": {"recommendation": "Use CalibrationMethod.MinMax"},
        }
        result = _normalize_runtime_recs(data)
        assert result["tensorrt"] == "Use trtexec with --int8"
        assert result["onnxruntime"] == "Use CalibrationMethod.MinMax"

    def test_deeply_nested_recommendation(self) -> None:
        """
        Actual failing case from production:
        LLM returned {'recommendation': {'settings': '...', 'notes': '...'}}
        """
        data = {
            "tensorrt": {
                "recommendation": {
                    "settings": "Use --int8 --fp16 flags",
                    "description": "to maintain accuracy",
                }
            },
            "onnxruntime": {
                "recommendation": {
                    "settings": "per_channel=True",
                    "notes": "for better accuracy",
                }
            },
        }
        result = _normalize_runtime_recs(data)

        # Should extract meaningful strings, not raw dict repr
        assert "tensorrt" in result
        assert "onnxruntime" in result
        assert isinstance(result["tensorrt"], str)
        assert isinstance(result["onnxruntime"], str)
        # Content should be present
        assert "int8" in result["tensorrt"].lower() or "accuracy" in result["tensorrt"]

    def test_mixed_nesting_levels(self) -> None:
        """Mix of simple strings and nested dicts."""
        data = {
            "tensorrt": "Simple string recommendation",
            "onnxruntime": {"recommendation": "Nested recommendation"},
            "tflite": {"recommendation": {"description": "Deeply nested"}},
        }
        result = _normalize_runtime_recs(data)
        assert result["tensorrt"] == "Simple string recommendation"
        assert result["onnxruntime"] == "Nested recommendation"
        assert result["tflite"] == "Deeply nested"

    def test_empty_dict(self) -> None:
        """Empty dict returns empty dict."""
        assert _normalize_runtime_recs({}) == {}

    def test_null_values_handled(self) -> None:
        """None values in dict are converted to empty strings."""
        data = {"tensorrt": None, "onnxruntime": "Valid recommendation"}
        result = _normalize_runtime_recs(data)
        assert result["tensorrt"] == ""
        assert result["onnxruntime"] == "Valid recommendation"


# =============================================================================
# Integration test with actual LLM failure patterns
# =============================================================================


class TestLLMResponsePatterns:
    """
    Tests reproducing actual LLM response patterns that caused Pydantic errors.

    These are based on the production error:
    ```
    LLM advice generation failed: 6 validation errors for QuantizationAdvice
    sensitive_layers: Input should be a valid list, input_value={'layer_names': [...]}
    runtime_recommendations.tensorrt: Input should be a valid string,
        input_value={'recommendation': {'sett...}}
    ```
    """

    def test_sensitive_layers_dict_pattern(self) -> None:
        """LLM returned dict instead of list for sensitive_layers."""
        llm_response = {
            "layer_names": [
                "/model/0/conv/Conv",
                "/model/1/bn/BatchNormalization",
                "/model/22/dfl/Softmax",
            ]
        }
        result = _normalize_str_list(llm_response)
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(item, str) for item in result)

    def test_safe_layers_dict_pattern(self) -> None:
        """LLM returned dict instead of list for safe_layers."""
        llm_response = {"layer_names": ["All other convolution layers", "Activation layers"]}
        result = _normalize_str_list(llm_response)
        assert result == ["All other convolution layers", "Activation layers"]

    def test_runtime_recommendations_nested_dict(self) -> None:
        """LLM returned nested dicts for runtime_recommendations."""
        llm_response = {
            "tensorrt": {
                "recommendation": {
                    "settings": "Use --int8 --fp16",
                    "description": "to maintain accuracy.",
                }
            },
            "onnxruntime": {
                "recommendation": {
                    "settings": "per_channel=True",
                    "notes": "achieve better accuracy.",
                }
            },
            "tflite": {
                "recommendation": {
                    "settings": "optimizations=[tf.lite.Optimize.DEFAULT]",
                    "description": "distribution of the model.",
                }
            },
            "openvino": {
                "recommendation": {
                    "settings": "stat_subset_size=300",
                    "notes": "accuracy is preserved.",
                }
            },
        }
        result = _normalize_runtime_recs(llm_response)

        # All keys present
        assert set(result.keys()) == {"tensorrt", "onnxruntime", "tflite", "openvino"}

        # All values are strings (not dicts)
        for key, val in result.items():
            assert isinstance(val, str), f"{key} should be str, got {type(val)}"
            assert len(val) > 0, f"{key} should not be empty"

    def test_qat_workflow_as_dict(self) -> None:
        """LLM might return qat_workflow as dict with 'steps' key."""
        llm_response = {
            "steps": [
                "Step 1: Train model to convergence",
                "Step 2: Insert fake-quant nodes",
                "Step 3: Fine-tune with lower LR",
            ]
        }
        # Our normalization should handle this via fallback
        result = _normalize_str_list(llm_response)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_mitigation_strategies_mixed_format(self) -> None:
        """LLM returns mix of strings and dicts in mitigation list."""
        llm_response = [
            "Use per-channel quantization",
            {"strategy": "Keep output layers at FP16"},
            "Increase calibration dataset size",
        ]
        result = _normalize_str_list(llm_response)
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(item, str) for item in result)
