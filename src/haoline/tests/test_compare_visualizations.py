#!/usr/bin/env python
# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""Unit tests for compare_visualizations module."""

import tempfile
from pathlib import Path
from unittest import TestCase

from ..compare_visualizations import (
    CalibrationRecommendation,
    TradeoffPoint,
    analyze_tradeoffs,
    build_enhanced_markdown,
    compute_tradeoff_points,
    generate_calibration_recommendations,
    generate_memory_savings_chart,
    generate_tradeoff_chart,
    is_available,
)


def create_sample_compare_json() -> dict:
    """Create a sample comparison JSON for testing."""
    return {
        "model_family_id": "resnet50_test",
        "baseline_precision": "fp32",
        "architecture_compatible": True,
        "compatibility_warnings": [],
        "variants": [
            {
                "precision": "fp32",
                "quantization_scheme": "none",
                "model_path": "resnet_fp32.onnx",
                "size_bytes": 102400000,
                "total_params": 25000000,
                "total_flops": 4100000000,
                "memory_bytes": 102400000,
                "metrics": {
                    "f1_macro": 0.931,
                    "latency_ms_p50": 14.5,
                    "throughput_qps": 680,
                },
                "hardware_estimates": None,
                "deltas_vs_baseline": None,
            },
            {
                "precision": "fp16",
                "quantization_scheme": "fp16",
                "model_path": "resnet_fp16.onnx",
                "size_bytes": 51200000,
                "total_params": 25000000,
                "total_flops": 4100000000,
                "memory_bytes": 51200000,
                "metrics": {
                    "f1_macro": 0.929,
                    "latency_ms_p50": 9.1,
                    "throughput_qps": 1080,
                },
                "hardware_estimates": None,
                "deltas_vs_baseline": {
                    "size_bytes": -51200000,
                    "f1_macro": -0.002,
                    "latency_ms_p50": -5.4,
                },
            },
            {
                "precision": "int8",
                "quantization_scheme": "int8",
                "model_path": "resnet_int8.onnx",
                "size_bytes": 25600000,
                "total_params": 25000000,
                "total_flops": 4100000000,
                "memory_bytes": 25600000,
                "metrics": {
                    "f1_macro": 0.915,
                    "latency_ms_p50": 5.2,
                    "throughput_qps": 1850,
                },
                "hardware_estimates": None,
                "deltas_vs_baseline": {
                    "size_bytes": -76800000,
                    "f1_macro": -0.016,
                    "latency_ms_p50": -9.3,
                },
            },
        ],
    }


class TestIsAvailable(TestCase):
    """Test visualization availability check."""

    def test_is_available_returns_bool(self):
        """is_available should return a boolean."""
        result = is_available()
        self.assertIsInstance(result, bool)


class TestComputeTradeoffPoints(TestCase):
    """Tests for compute_tradeoff_points function."""

    def test_basic_computation(self):
        """Should compute tradeoff points for all variants."""
        compare_json = create_sample_compare_json()
        points = compute_tradeoff_points(compare_json)

        self.assertEqual(len(points), 3)
        self.assertIsInstance(points[0], TradeoffPoint)

    def test_baseline_speedup_is_one(self):
        """Baseline variant should have speedup of 1.0."""
        compare_json = create_sample_compare_json()
        points = compute_tradeoff_points(compare_json)

        # Find fp32 (baseline)
        fp32_point = next(p for p in points if p.precision == "fp32")
        self.assertAlmostEqual(fp32_point.speedup, 1.0, places=2)
        self.assertAlmostEqual(fp32_point.accuracy_delta, 0.0, places=5)

    def test_faster_variant_has_higher_speedup(self):
        """Faster variant should have speedup > 1.0."""
        compare_json = create_sample_compare_json()
        points = compute_tradeoff_points(compare_json)

        int8_point = next(p for p in points if p.precision == "int8")
        # int8 is faster: 14.5 / 5.2 ≈ 2.79x speedup
        self.assertGreater(int8_point.speedup, 2.0)

    def test_accuracy_delta_computed(self):
        """Should compute accuracy delta relative to baseline."""
        compare_json = create_sample_compare_json()
        points = compute_tradeoff_points(compare_json)

        fp16_point = next(p for p in points if p.precision == "fp16")
        # fp16: 0.929 - 0.931 = -0.002
        self.assertAlmostEqual(fp16_point.accuracy_delta, -0.002, places=5)

    def test_empty_variants(self):
        """Should handle empty variants list."""
        compare_json = {"variants": []}
        points = compute_tradeoff_points(compare_json)
        self.assertEqual(len(points), 0)


class TestAnalyzeTradeoffs(TestCase):
    """Tests for analyze_tradeoffs function."""

    def test_returns_recommendations(self):
        """Should return analysis with recommendations."""
        compare_json = create_sample_compare_json()
        analysis = analyze_tradeoffs(compare_json)

        self.assertIn("recommendations", analysis)
        self.assertIsInstance(analysis["recommendations"], list)

    def test_identifies_best_variants(self):
        """Should identify best variants for different criteria."""
        compare_json = create_sample_compare_json()
        analysis = analyze_tradeoffs(compare_json)

        self.assertIn("best_speed", analysis)
        self.assertIn("smallest", analysis)

    def test_tradeoff_points_included(self):
        """Should include tradeoff points in analysis."""
        compare_json = create_sample_compare_json()
        analysis = analyze_tradeoffs(compare_json)

        self.assertIn("tradeoff_points", analysis)
        self.assertEqual(len(analysis["tradeoff_points"]), 3)


class TestCalibrationRecommendations(TestCase):
    """Tests for generate_calibration_recommendations function."""

    def test_returns_recommendations(self):
        """Should return list of CalibrationRecommendation objects."""
        compare_json = create_sample_compare_json()
        recs = generate_calibration_recommendations(compare_json)

        self.assertIsInstance(recs, list)
        for rec in recs:
            self.assertIsInstance(rec, CalibrationRecommendation)

    def test_int8_accuracy_warning(self):
        """Should warn about INT8 accuracy drop > 2%."""
        # Modify to have significant INT8 accuracy drop
        compare_json = create_sample_compare_json()
        # INT8 has 1.6% drop, which is below threshold, so let's make it worse
        compare_json["variants"][2]["metrics"]["f1_macro"] = 0.90  # 3.1% drop

        recs = generate_calibration_recommendations(compare_json)

        # Should have warning about calibration
        warnings = [r for r in recs if r.severity == "warning"]
        self.assertGreater(len(warnings), 0)


class TestBuildEnhancedMarkdown(TestCase):
    """Tests for build_enhanced_markdown function."""

    def test_basic_markdown_generation(self):
        """Should generate valid Markdown."""
        compare_json = create_sample_compare_json()
        md = build_enhanced_markdown(compare_json, include_charts=False)

        self.assertIn("# Quantization Impact Report", md)
        self.assertIn("resnet50_test", md)
        self.assertIn("FP32", md)
        self.assertIn("FP16", md)
        self.assertIn("INT8", md)

    def test_includes_tradeoff_analysis(self):
        """Should include trade-off analysis section."""
        compare_json = create_sample_compare_json()
        md = build_enhanced_markdown(compare_json, include_charts=False)

        self.assertIn("## Trade-off Analysis", md)
        self.assertIn("### Recommendations", md)

    def test_includes_variant_table(self):
        """Should include variant comparison table."""
        compare_json = create_sample_compare_json()
        md = build_enhanced_markdown(compare_json, include_charts=False)

        self.assertIn("## Variant Comparison", md)
        self.assertIn("| Precision |", md)

    def test_compatibility_warnings_shown(self):
        """Should show compatibility warnings if present."""
        compare_json = create_sample_compare_json()
        compare_json["architecture_compatible"] = False
        compare_json["compatibility_warnings"] = ["Test warning"]

        md = build_enhanced_markdown(compare_json, include_charts=False)
        self.assertIn("⚠️ Compatibility Warnings", md)
        self.assertIn("Test warning", md)


class TestChartGeneration(TestCase):
    """Tests for chart generation functions."""

    def test_tradeoff_chart_no_matplotlib(self):
        """Should return None if matplotlib not available."""
        # This test will pass if matplotlib IS available, as it will
        # generate the chart. We just verify it doesn't crash.
        compare_json = create_sample_compare_json()
        points = compute_tradeoff_points(compare_json)

        result = generate_tradeoff_chart(points)
        # Result is either bytes (matplotlib available) or None
        if is_available():
            self.assertIsInstance(result, bytes)
            self.assertGreater(len(result), 0)
        else:
            self.assertIsNone(result)

    def test_tradeoff_chart_to_file(self):
        """Should save chart to file if path provided."""
        if not is_available():
            self.skipTest("matplotlib not available")

        compare_json = create_sample_compare_json()
        points = compute_tradeoff_points(compare_json)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "tradeoff.png"
            result = generate_tradeoff_chart(points, output_path)

            self.assertIsNotNone(result)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_memory_savings_chart(self):
        """Should generate memory savings chart."""
        if not is_available():
            self.skipTest("matplotlib not available")

        compare_json = create_sample_compare_json()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "memory.png"
            result = generate_memory_savings_chart(compare_json, output_path)

            self.assertIsNotNone(result)
            self.assertTrue(output_path.exists())

    def test_empty_points_returns_none(self):
        """Should return None for empty points list."""
        result = generate_tradeoff_chart([])
        self.assertIsNone(result)
