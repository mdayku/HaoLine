# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
Unit tests for the LLM summarizer module.

Tests API client, prompt templates, and graceful error handling.
Note: Most tests mock the OpenAI API to avoid actual API calls.
"""

from __future__ import annotations

import os

import pytest

from ..analyzer import FlopCounts, MemoryEstimates, ParamCounts
from ..llm_summarizer import (
    DETAILED_SUMMARY_PROMPT,
    SHORT_SUMMARY_PROMPT,
    SYSTEM_PROMPT,
    LLMSummarizer,
    LLMSummary,
    has_api_key,
    is_available,
    summarize_report,
)
from ..report import GraphSummary, InspectionReport, ModelMetadata


class TestLLMAvailability:
    """Tests for availability checks."""

    def test_is_available_returns_bool(self):
        """is_available() should return a boolean."""
        result = is_available()
        assert isinstance(result, bool)

    def test_has_api_key_returns_bool(self):
        """has_api_key() should return a boolean."""
        result = has_api_key()
        assert isinstance(result, bool)

    def test_has_api_key_checks_env_var(self):
        """has_api_key() should check OPENAI_API_KEY env var."""
        # Save original value
        original = os.environ.get("OPENAI_API_KEY")

        try:
            # Test with key set
            os.environ["OPENAI_API_KEY"] = "test-key"
            assert has_api_key() is True

            # Test with key unset
            del os.environ["OPENAI_API_KEY"]
            assert has_api_key() is False
        finally:
            # Restore original
            if original:
                os.environ["OPENAI_API_KEY"] = original


class TestLLMSummaryDataclass:
    """Tests for the LLMSummary dataclass."""

    def test_summary_creation(self):
        """LLMSummary should be created with all fields."""
        summary = LLMSummary(
            short_summary="Test short",
            detailed_summary="Test detailed",
            model_used="gpt-4o-mini",
            tokens_used=100,
            success=True,
        )
        assert summary.short_summary == "Test short"
        assert summary.detailed_summary == "Test detailed"
        assert summary.model_used == "gpt-4o-mini"
        assert summary.tokens_used == 100
        assert summary.success is True
        assert summary.error_message is None

    def test_summary_with_error(self):
        """LLMSummary should handle error state."""
        summary = LLMSummary(
            short_summary="",
            detailed_summary="",
            model_used="",
            tokens_used=0,
            success=False,
            error_message="API error",
        )
        assert summary.success is False
        assert summary.error_message == "API error"


class TestPromptTemplates:
    """Tests for prompt templates."""

    def test_system_prompt_exists(self):
        """System prompt should be defined."""
        assert SYSTEM_PROMPT
        assert "ML engineer" in SYSTEM_PROMPT

    def test_short_summary_prompt_has_placeholder(self):
        """Short summary prompt should have report_json placeholder."""
        assert "{report_json}" in SHORT_SUMMARY_PROMPT
        assert "1-2 sentence" in SHORT_SUMMARY_PROMPT

    def test_detailed_summary_prompt_has_placeholder(self):
        """Detailed summary prompt should have report_json placeholder."""
        assert "{report_json}" in DETAILED_SUMMARY_PROMPT
        assert "paragraph" in DETAILED_SUMMARY_PROMPT


class TestLLMSummarizer:
    """Tests for the LLMSummarizer class."""

    def test_summarizer_initialization(self):
        """Summarizer should initialize without errors."""
        summarizer = LLMSummarizer()
        assert summarizer is not None
        assert summarizer.logger is not None

    def test_summarizer_default_model(self):
        """Summarizer should use default model."""
        summarizer = LLMSummarizer()
        assert summarizer.model == "gpt-4o-mini"

    def test_summarizer_custom_model(self):
        """Summarizer should accept custom model."""
        summarizer = LLMSummarizer(model="gpt-4o")
        assert summarizer.model == "gpt-4o"

    def test_is_configured_without_key(self):
        """is_configured should return False without API key."""
        # Save and clear env var
        original = os.environ.get("OPENAI_API_KEY")
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

        try:
            summarizer = LLMSummarizer()
            assert summarizer.is_configured() is False
        finally:
            if original:
                os.environ["OPENAI_API_KEY"] = original

    def test_summarize_returns_error_when_not_configured(self):
        """summarize() should return error LLMSummary when not configured."""
        # Save and clear env var
        original = os.environ.get("OPENAI_API_KEY")
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

        try:
            metadata = ModelMetadata(
                path="test.onnx",
                ir_version=8,
                producer_name="test",
                producer_version="1.0",
                domain="",
                model_version=1,
                doc_string="",
                opsets={"ai.onnx": 17},
            )

            report = InspectionReport(metadata=metadata)

            summarizer = LLMSummarizer()
            result = summarizer.summarize(report)

            assert result.success is False
            assert "not configured" in result.error_message.lower()
        finally:
            if original:
                os.environ["OPENAI_API_KEY"] = original


class TestReportPreparation:
    """Tests for report preparation for LLM consumption."""

    def test_prepare_report_for_prompt(self):
        """_prepare_report_for_prompt should create valid JSON."""
        metadata = ModelMetadata(
            path="models/test.onnx",
            ir_version=8,
            producer_name="pytorch",
            producer_version="2.0",
            domain="",
            model_version=1,
            doc_string="",
            opsets={"ai.onnx": 17},
        )

        graph_summary = GraphSummary(
            num_nodes=100,
            num_inputs=1,
            num_outputs=1,
            num_initializers=50,
            input_shapes={"input": [1, 3, 224, 224]},
            output_shapes={"output": [1, 1000]},
            op_type_counts={"Conv": 50, "Relu": 48},
        )

        param_counts = ParamCounts(
            total=25000000,
            trainable=25000000,
            by_op_type={"Conv": 20000000},
        )

        flop_counts = FlopCounts(
            total=4000000000,
            by_op_type={"Conv": 3500000000},
        )

        memory_estimates = MemoryEstimates(
            model_size_bytes=100000000,
            peak_activation_bytes=50000000,
        )

        report = InspectionReport(
            metadata=metadata,
            graph_summary=graph_summary,
            param_counts=param_counts,
            flop_counts=flop_counts,
            memory_estimates=memory_estimates,
            architecture_type="cnn",
        )

        summarizer = LLMSummarizer()
        json_str = summarizer._prepare_report_for_prompt(report)

        import json

        parsed = json.loads(json_str)

        assert "model_name" in parsed
        assert parsed["model_name"] == "test.onnx"
        assert "graph" in parsed
        assert parsed["graph"]["nodes"] == 100
        assert "parameters" in parsed
        assert parsed["parameters"]["total"] == 25000000


class TestConvenienceFunction:
    """Tests for the summarize_report convenience function."""

    def test_summarize_report_function(self):
        """summarize_report should work as convenience function."""
        # Save and clear env var
        original = os.environ.get("OPENAI_API_KEY")
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

        try:
            metadata = ModelMetadata(
                path="test.onnx",
                ir_version=8,
                producer_name="test",
                producer_version="1.0",
                domain="",
                model_version=1,
                doc_string="",
                opsets={},
            )

            report = InspectionReport(metadata=metadata)
            result = summarize_report(report)

            assert isinstance(result, LLMSummary)
            assert result.success is False  # No API key
        finally:
            if original:
                os.environ["OPENAI_API_KEY"] = original


@pytest.mark.skipif(
    not is_available() or not has_api_key(),
    reason="OpenAI not installed or API key not set",
)
class TestLLMIntegration:
    """Integration tests that make real API calls (skipped without API key)."""

    def test_real_summarization(self):
        """Test actual LLM summarization with real API."""
        metadata = ModelMetadata(
            path="resnet50.onnx",
            ir_version=8,
            producer_name="pytorch",
            producer_version="2.0",
            domain="",
            model_version=1,
            doc_string="",
            opsets={"ai.onnx": 17},
        )

        graph_summary = GraphSummary(
            num_nodes=150,
            num_inputs=1,
            num_outputs=1,
            num_initializers=100,
            input_shapes={"input": [1, 3, 224, 224]},
            output_shapes={"output": [1, 1000]},
            op_type_counts={"Conv": 53, "Relu": 49, "Add": 16, "MaxPool": 1},
        )

        param_counts = ParamCounts(
            total=25557032,
            trainable=25557032,
            by_op_type={"Conv": 23000000, "Gemm": 2500000},
        )

        flop_counts = FlopCounts(
            total=4100000000,
            by_op_type={"Conv": 4000000000},
        )

        memory_estimates = MemoryEstimates(
            model_size_bytes=100000000,
            peak_activation_bytes=50000000,
        )

        report = InspectionReport(
            metadata=metadata,
            graph_summary=graph_summary,
            param_counts=param_counts,
            flop_counts=flop_counts,
            memory_estimates=memory_estimates,
            architecture_type="cnn",
        )

        summarizer = LLMSummarizer()
        result = summarizer.summarize(report)

        assert result.success is True
        assert len(result.short_summary) > 0
        assert len(result.detailed_summary) > 0
        assert result.tokens_used > 0
        assert result.model_used == "gpt-4o-mini"
