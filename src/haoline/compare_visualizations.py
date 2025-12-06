#!/usr/bin/env python
# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
Compare Mode Visualizations for Quantization Impact Reports.

Story 6.4: Quantization Impact Report (TRT EngineXplorer-inspired)
------------------------------------------------------------------

This module provides visualization and analysis functions for multi-model
comparison reports. It generates:

- Accuracy vs Speedup tradeoff charts
- Memory savings analysis
- Layer-wise precision breakdown
- Trade-off analysis summaries
- Calibration recommendations

Requires matplotlib for chart generation (optional graceful fallback).
"""

from __future__ import annotations

import base64
import logging
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

LOGGER = logging.getLogger("haoline.compare_viz")

# Try to import matplotlib
try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None  # type: ignore
    Figure = None  # type: ignore


class TradeoffPoint(BaseModel):
    """A single point on the accuracy vs speedup tradeoff curve."""

    model_config = ConfigDict(frozen=True)

    precision: str
    speedup: float  # Relative to baseline (1.0 = same speed)
    accuracy_delta: float  # Relative to baseline (negative = worse)
    size_ratio: float  # Relative to baseline (< 1.0 = smaller)
    memory_ratio: float  # Relative to baseline (< 1.0 = smaller)


class LayerPrecisionBreakdown(BaseModel):
    """Precision breakdown for a single layer."""

    model_config = ConfigDict(frozen=True)

    layer_name: str
    op_type: str
    precision: str
    param_count: int
    flops: int
    memory_bytes: int


class CalibrationRecommendation(BaseModel):
    """Recommendation for quantization calibration."""

    model_config = ConfigDict(frozen=True)

    recommendation: str
    reason: str
    severity: str  # "info", "warning", "critical"
    affected_layers: list[str]


def is_available() -> bool:
    """Check if visualization is available (matplotlib installed)."""
    return MATPLOTLIB_AVAILABLE


def extract_layer_precision_breakdown(
    variant_report: Any,
    precision: str,
) -> list[LayerPrecisionBreakdown]:
    """
    Extract per-layer precision breakdown from an inspection report.

    Task 6.4.4: Layer-wise precision breakdown

    Returns a list of LayerPrecisionBreakdown for each layer/op in the model.
    """
    breakdown: list[LayerPrecisionBreakdown] = []

    # Get layer summary if available
    layer_summary = getattr(variant_report, "layer_summary", None)
    if layer_summary is None:
        return breakdown

    layers = getattr(layer_summary, "layers", [])
    for layer in layers:
        layer_name = getattr(layer, "name", "unknown")
        op_type = getattr(layer, "op_type", "unknown")
        params = getattr(layer, "param_count", 0)
        flops = getattr(layer, "flops", 0)
        memory = getattr(layer, "memory_bytes", 0)

        breakdown.append(
            LayerPrecisionBreakdown(
                layer_name=layer_name,
                op_type=op_type,
                precision=precision,
                param_count=params,
                flops=flops,
                memory_bytes=memory,
            )
        )

    return breakdown


def generate_layer_precision_chart(
    breakdowns: dict[str, list[LayerPrecisionBreakdown]],
    output_path: Path | None = None,
    title: str = "Per-Layer Precision Comparison",
    top_n: int = 20,
) -> bytes | None:
    """
    Generate a chart showing per-layer precision breakdown.

    Task 6.4.4: Layer-wise precision breakdown visualization

    Args:
        breakdowns: Dict mapping precision to list of layer breakdowns
        output_path: Optional path to save the chart
        title: Chart title
        top_n: Number of top layers to show

    Returns:
        PNG bytes if successful, None otherwise
    """
    if not MATPLOTLIB_AVAILABLE or not breakdowns:
        return None

    plt.style.use("dark_background")

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#0a0a0a")
    ax.set_facecolor("#1a1a1a")

    # Get all precisions and sort layers by FLOPs (from first precision)
    precisions = list(breakdowns.keys())
    if not precisions:
        return None

    first_breakdown = breakdowns[precisions[0]]
    # Sort by FLOPs descending and take top N
    sorted_layers = sorted(first_breakdown, key=lambda x: x.flops, reverse=True)[:top_n]
    layer_names = [layer.layer_name[:30] for layer in sorted_layers]  # Truncate long names

    # Colors for different precisions
    precision_colors = {
        "fp32": "#4A90D9",
        "fp16": "#30D158",
        "bf16": "#64D2FF",
        "int8": "#FFD60A",
        "int4": "#FF9F0A",
    }

    x = range(len(layer_names))
    width = 0.8 / len(precisions)

    for idx, precision in enumerate(precisions):
        layers = breakdowns[precision]
        layer_map = {layer.layer_name: layer for layer in layers}

        flops_values = []
        for layer in sorted_layers:
            if layer.layer_name in layer_map:
                flops_values.append(layer_map[layer.layer_name].flops / 1e9)  # Convert to GFLOPs
            else:
                flops_values.append(0)

        offset = (idx - len(precisions) / 2 + 0.5) * width
        color = precision_colors.get(precision.lower(), "#BF5AF2")
        ax.barh(
            [i + offset for i in x],
            flops_values,
            width,
            label=precision.upper(),
            color=color,
            alpha=0.8,
        )

    ax.set_xlabel("FLOPs (G)", fontsize=12, color="white")
    ax.set_ylabel("Layer", fontsize=12, color="white")
    ax.set_title(title, fontsize=14, fontweight="bold", color="white", pad=15)
    ax.set_yticks(list(x))
    ax.set_yticklabels(layer_names, fontsize=8)
    ax.tick_params(colors="white")

    ax.spines["bottom"].set_color("#636366")
    ax.spines["left"].set_color("#636366")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(loc="lower right", framealpha=0.3, facecolor="#1a1a1a")
    ax.invert_yaxis()  # Highest FLOPs at top

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor="#0a0a0a")
    buf.seek(0)
    png_bytes = buf.read()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(png_bytes)

    plt.close(fig)
    return png_bytes


def compute_tradeoff_points(
    compare_json: dict[str, Any],
) -> list[TradeoffPoint]:
    """
    Compute tradeoff points from comparison JSON.

    Task 6.4.3: Trade-off analysis

    Returns a list of TradeoffPoint objects for each variant,
    with speedup and accuracy delta relative to baseline.
    """
    variants = compare_json.get("variants", [])
    baseline_precision = compare_json.get("baseline_precision", "fp32")

    # Find baseline variant
    baseline = None
    for v in variants:
        if v.get("precision") == baseline_precision:
            baseline = v
            break

    if baseline is None and variants:
        baseline = variants[0]

    if baseline is None:
        return []

    # Extract baseline metrics
    baseline_metrics = baseline.get("metrics", {})
    baseline_latency = (
        baseline_metrics.get("latency_ms_p50") or baseline_metrics.get("latency_ms") or 1.0
    )
    baseline_accuracy = baseline_metrics.get("f1_macro") or baseline_metrics.get("accuracy") or 1.0
    baseline_size = baseline.get("size_bytes", 1)
    baseline_memory = baseline.get("memory_bytes") or baseline_size

    points: list[TradeoffPoint] = []
    for v in variants:
        precision = v.get("precision", "unknown")
        metrics = v.get("metrics", {})

        latency = metrics.get("latency_ms_p50") or metrics.get("latency_ms") or baseline_latency
        accuracy = metrics.get("f1_macro") or metrics.get("accuracy") or baseline_accuracy
        size = v.get("size_bytes", baseline_size)
        memory = v.get("memory_bytes") or size

        # Compute ratios
        speedup = baseline_latency / max(latency, 0.001)
        accuracy_delta = accuracy - baseline_accuracy
        size_ratio = size / max(baseline_size, 1)
        memory_ratio = memory / max(baseline_memory, 1)

        points.append(
            TradeoffPoint(
                precision=precision,
                speedup=speedup,
                accuracy_delta=accuracy_delta,
                size_ratio=size_ratio,
                memory_ratio=memory_ratio,
            )
        )

    return points


def generate_tradeoff_chart(
    points: Sequence[TradeoffPoint],
    output_path: Path | None = None,
    title: str = "Accuracy vs Speedup Tradeoff",
) -> bytes | None:
    """
    Generate accuracy vs speedup tradeoff chart.

    Task 6.4.5: Show accuracy vs speedup tradeoff chart

    Args:
        points: List of TradeoffPoint objects
        output_path: Optional path to save the chart
        title: Chart title

    Returns:
        PNG bytes if successful, None otherwise
    """
    if not MATPLOTLIB_AVAILABLE or not points:
        return None

    # Chart styling (dark theme matching existing visualizations)
    plt.style.use("dark_background")

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0a0a0a")
    ax.set_facecolor("#1a1a1a")

    # Colors for different precisions
    precision_colors = {
        "fp32": "#4A90D9",
        "fp16": "#30D158",
        "bf16": "#64D2FF",
        "int8": "#FFD60A",
        "int4": "#FF9F0A",
        "unknown": "#636366",
    }

    # Plot each point
    for p in points:
        color = precision_colors.get(p.precision.lower(), "#BF5AF2")

        # Size based on memory reduction (smaller = bigger marker)
        marker_size = max(100, 400 * (1 - p.memory_ratio + 0.5))

        ax.scatter(
            p.speedup,
            p.accuracy_delta * 100,  # Convert to percentage
            s=marker_size,
            c=color,
            label=p.precision.upper(),
            alpha=0.8,
            edgecolors="white",
            linewidths=1.5,
            zorder=5,
        )

        # Annotate with precision label
        ax.annotate(
            p.precision.upper(),
            (p.speedup, p.accuracy_delta * 100),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="white",
        )

    # Reference lines
    ax.axhline(y=0, color="#636366", linestyle="--", alpha=0.5, label="Baseline accuracy")
    ax.axvline(x=1.0, color="#636366", linestyle="--", alpha=0.5, label="Baseline speed")

    # Styling
    ax.set_xlabel("Speedup (×)", fontsize=12, color="white")  # noqa: RUF001
    ax.set_ylabel("Accuracy Change (%)", fontsize=12, color="white")
    ax.set_title(title, fontsize=14, fontweight="bold", color="white", pad=15)

    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#636366")
    ax.spines["left"].set_color("#636366")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(True, alpha=0.2, color="#636366")

    # Add quadrant labels
    ax.text(
        0.98,
        0.98,
        "Slower + Better",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#636366",
        alpha=0.7,
    )
    ax.text(
        0.02,
        0.98,
        "Faster + Better ✓",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#30D158",
        alpha=0.9,
    )
    ax.text(
        0.98,
        0.02,
        "Slower + Worse ✗",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#FF453A",
        alpha=0.9,
    )
    ax.text(
        0.02,
        0.02,
        "Faster + Worse",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#FFD60A",
        alpha=0.7,
    )

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicate labels
    by_label = dict(zip(labels, handles, strict=False))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper right",
        framealpha=0.3,
        facecolor="#1a1a1a",
        edgecolor="#636366",
    )

    plt.tight_layout()

    # Save to bytes
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor="#0a0a0a")
    buf.seek(0)
    png_bytes = buf.read()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(png_bytes)

    plt.close(fig)
    return png_bytes


def generate_memory_savings_chart(
    compare_json: dict[str, Any],
    output_path: Path | None = None,
    title: str = "Memory & Size Reduction",
) -> bytes | None:
    """
    Generate memory savings comparison chart.

    Task 6.4.6: Display memory savings per layer analysis

    Shows size and memory reduction for each variant relative to baseline.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    variants = compare_json.get("variants", [])
    if not variants:
        return None

    plt.style.use("dark_background")

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0a0a0a")
    ax.set_facecolor("#1a1a1a")

    # Find baseline
    baseline_precision = compare_json.get("baseline_precision", "fp32")
    baseline_size = 1
    baseline_memory = 1
    for v in variants:
        if v.get("precision") == baseline_precision:
            baseline_size = v.get("size_bytes", 1)
            baseline_memory = v.get("memory_bytes") or baseline_size
            break

    # Prepare data - skip baseline since it shows 0% reduction
    precisions = []
    size_reductions = []
    memory_reductions = []

    for v in variants:
        precision = v.get("precision", "unknown")
        # Skip baseline - it always shows 0% reduction
        if precision == baseline_precision:
            continue

        size = v.get("size_bytes", baseline_size)
        memory = v.get("memory_bytes") or size

        precisions.append(f"{precision.upper()} vs {baseline_precision.upper()}")
        size_reductions.append((1 - size / baseline_size) * 100)
        memory_reductions.append((1 - memory / baseline_memory) * 100)

    if not precisions:
        return None  # Nothing to compare if only baseline

    x = range(len(precisions))
    width = 0.35

    bars1 = ax.bar(
        [i - width / 2 for i in x],
        size_reductions,
        width,
        label="File Size",
        color="#4A90D9",
        alpha=0.8,
    )
    bars2 = ax.bar(
        [i + width / 2 for i in x],
        memory_reductions,
        width,
        label="Memory",
        color="#30D158",
        alpha=0.8,
    )

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color="white",
        )
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color="white",
        )

    ax.set_ylabel("Reduction (%)", fontsize=12, color="white")
    ax.set_title(title, fontsize=14, fontweight="bold", color="white", pad=15)
    ax.set_xticks(list(x))
    ax.set_xticklabels(precisions)
    ax.tick_params(colors="white")

    ax.axhline(y=0, color="#636366", linestyle="-", alpha=0.5)

    ax.spines["bottom"].set_color("#636366")
    ax.spines["left"].set_color("#636366")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(loc="upper right", framealpha=0.3, facecolor="#1a1a1a")

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor="#0a0a0a")
    buf.seek(0)
    png_bytes = buf.read()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(png_bytes)

    plt.close(fig)
    return png_bytes


def analyze_tradeoffs(
    compare_json: dict[str, Any],
) -> dict[str, Any]:
    """
    Analyze tradeoffs between variants and generate recommendations.

    Task 6.4.3: Add trade-off analysis section

    Returns a dict with:
    - best_balanced: variant with best speedup-accuracy balance
    - best_speed: variant with best speedup
    - best_accuracy: variant with best accuracy
    - recommendations: list of textual recommendations
    """
    points = compute_tradeoff_points(compare_json)
    if not points:
        return {"recommendations": ["No variants to analyze"]}

    analysis: dict[str, Any] = {}

    # Find best variants
    best_balanced_score = float("-inf")
    best_balanced = None

    for p in points:
        # Score: speedup bonus minus accuracy penalty (weighted)
        # Higher speedup is good, negative accuracy delta is bad
        score = p.speedup - 1.0 + (p.accuracy_delta * 10)  # 10x weight on accuracy
        if score > best_balanced_score:
            best_balanced_score = score
            best_balanced = p

    best_speed = max(points, key=lambda p: p.speedup)
    best_accuracy = max(points, key=lambda p: p.accuracy_delta)
    smallest = min(points, key=lambda p: p.size_ratio)

    analysis["best_balanced"] = best_balanced.precision if best_balanced else None
    analysis["best_speed"] = best_speed.precision
    analysis["best_accuracy"] = best_accuracy.precision
    analysis["smallest"] = smallest.precision

    # Generate recommendations
    recommendations: list[str] = []

    # Check for sweet spot
    for p in points:
        if p.speedup > 1.3 and p.accuracy_delta > -0.01:
            recommendations.append(
                f"**{p.precision.upper()}** offers {p.speedup:.1f}x speedup with "
                f"minimal accuracy loss ({p.accuracy_delta * 100:.2f}%) - recommended."
            )

    # Warn about significant accuracy drops
    for p in points:
        if p.accuracy_delta < -0.05:
            recommendations.append(
                f"**{p.precision.upper()}** has significant accuracy drop "
                f"({p.accuracy_delta * 100:.2f}%) — validate on your data before use."
            )

    # Memory savings
    for p in points:
        if p.memory_ratio < 0.6:
            savings = (1 - p.memory_ratio) * 100
            recommendations.append(
                f"**{p.precision.upper()}** saves {savings:.0f}% memory — "
                "ideal for edge/mobile deployment."
            )

    if not recommendations:
        recommendations.append(
            "All variants show similar trade-offs. Consider your "
            "deployment constraints (latency, memory, accuracy) to choose."
        )

    analysis["recommendations"] = recommendations
    analysis["tradeoff_points"] = [
        {
            "precision": p.precision,
            "speedup": round(p.speedup, 3),
            "accuracy_delta": round(p.accuracy_delta, 5),
            "size_ratio": round(p.size_ratio, 3),
            "memory_ratio": round(p.memory_ratio, 3),
        }
        for p in points
    ]

    return analysis


class NormalizedMetrics(BaseModel):
    """Normalized efficiency metrics for a model variant."""

    model_config = ConfigDict(frozen=True)

    precision: str
    flops_per_param: float  # Compute intensity
    memory_per_param: float  # Memory efficiency
    params_per_mb: float  # Storage efficiency
    efficiency_score: float  # Composite efficiency (0-100)


def compute_normalized_metrics(
    compare_json: dict[str, Any],
) -> list[NormalizedMetrics]:
    """
    Compute normalized metrics for comparison.

    Task 6.10.2: Implement normalized metrics (FLOPs/param, memory/param, etc.)

    Returns normalized efficiency metrics for each variant.
    """
    variants = compare_json.get("variants", [])
    metrics_list: list[NormalizedMetrics] = []

    for v in variants:
        precision = v.get("precision", "unknown")
        size_bytes = v.get("size_bytes", 0)
        report = v.get("report", {})

        # Extract raw metrics
        param_counts = report.get("param_counts", {})
        flop_counts = report.get("flop_counts", {})
        memory_est = report.get("memory_estimates", {})

        total_params = param_counts.get("total", 0) or 1  # Avoid div by zero
        total_flops = flop_counts.get("total", 0)
        peak_memory = memory_est.get("peak_activation_bytes", 0)

        # Compute normalized metrics
        flops_per_param = total_flops / total_params if total_params > 0 else 0
        memory_per_param = peak_memory / total_params if total_params > 0 else 0
        size_mb = size_bytes / (1024 * 1024) if size_bytes > 0 else 1
        params_per_mb = total_params / size_mb

        # Composite efficiency score (higher = more efficient)
        # Favors: higher FLOPs/param (more compute per weight), lower memory/param
        efficiency_score = min(100, max(0, (flops_per_param / 100) - (memory_per_param / 1000)))

        metrics_list.append(
            NormalizedMetrics(
                precision=precision,
                flops_per_param=flops_per_param,
                memory_per_param=memory_per_param,
                params_per_mb=params_per_mb,
                efficiency_score=efficiency_score,
            )
        )

    return metrics_list


def generate_radar_chart(
    compare_json: dict[str, Any],
    output_path: Path | None = None,
) -> bytes | None:
    """
    Generate radar chart comparing key metrics across model variants.

    Task 6.10.3: Add radar chart comparing key metrics across models

    Compares: Size, Params, FLOPs, Memory, Latency (if available)
    Each axis is normalized 0-1 where lower is better.
    """
    if not MATPLOTLIB_AVAILABLE:
        LOGGER.warning("matplotlib not available, skipping radar chart")
        return None

    import numpy as np

    variants = compare_json.get("variants", [])
    if not variants:
        return None

    baseline_precision = compare_json.get("baseline_precision", "fp32")

    # Find baseline for normalization
    baseline = None
    for v in variants:
        if v.get("precision") == baseline_precision:
            baseline = v
            break
    if baseline is None:
        baseline = variants[0]

    # Extract baseline values for normalization
    baseline_size = baseline.get("size_bytes", 1) or 1
    baseline_report = baseline.get("report", {})
    baseline_params = baseline_report.get("param_counts", {}).get("total", 1) or 1
    baseline_flops = baseline_report.get("flop_counts", {}).get("total", 1) or 1
    baseline_memory = (
        baseline_report.get("memory_estimates", {}).get("peak_activation_bytes", 1) or 1
    )
    baseline_metrics = baseline.get("metrics", {})
    baseline_latency = (
        baseline_metrics.get("latency_ms_p50") or baseline_metrics.get("latency_ms") or 1
    )

    # Define radar categories (lower is better for all)
    categories = ["Size", "Params", "FLOPs", "Memory", "Latency"]
    num_vars = len(categories)

    # Compute angles for radar
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the loop

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    # Colors for each variant
    colors = ["#4361ee", "#f72585", "#4cc9f0", "#7209b7", "#3a0ca3"]

    for idx, v in enumerate(variants):
        precision = v.get("precision", "unknown")
        size = v.get("size_bytes", baseline_size)
        report = v.get("report", {})
        params = report.get("param_counts", {}).get("total", baseline_params) or baseline_params
        flops = report.get("flop_counts", {}).get("total", baseline_flops) or baseline_flops
        memory = (
            report.get("memory_estimates", {}).get("peak_activation_bytes", baseline_memory)
            or baseline_memory
        )
        metrics = v.get("metrics", {})
        latency = metrics.get("latency_ms_p50") or metrics.get("latency_ms") or baseline_latency

        # Normalize (0-1 scale, relative to baseline, capped at 2x)
        values = [
            min(2.0, size / baseline_size),
            min(2.0, params / baseline_params),
            min(2.0, flops / baseline_flops),
            min(2.0, memory / baseline_memory),
            min(2.0, latency / baseline_latency),
        ]
        values += values[:1]  # Complete the loop

        color = colors[idx % len(colors)]
        ax.plot(angles, values, "o-", linewidth=2, label=precision, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)

    # Styling
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color="#e0e0e0", size=10)
    ax.tick_params(colors="#e0e0e0")

    # Set radial limits
    ax.set_ylim(0, 2.0)
    ax.set_yticks([0.5, 1.0, 1.5, 2.0])
    ax.set_yticklabels(["0.5x", "1x", "1.5x", "2x"], color="#a0a0a0", size=8)

    # Grid styling
    ax.grid(True, color="#404060", linestyle="-", linewidth=0.5)

    # Legend
    legend = ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    legend.get_frame().set_facecolor("#16213e")
    legend.get_frame().set_edgecolor("#404060")
    for text in legend.get_texts():
        text.set_color("#e0e0e0")

    ax.set_title(
        "Model Comparison (lower = better)",
        color="#e0e0e0",
        size=14,
        fontweight="bold",
        pad=20,
    )

    # Save to bytes
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor="#1a1a2e", bbox_inches="tight")
    buf.seek(0)
    chart_bytes = buf.getvalue()
    plt.close(fig)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(chart_bytes)
        LOGGER.info(f"Radar chart saved to {output_path}")

    return chart_bytes


def generate_compare_pdf(
    compare_json: dict[str, Any],
    output_path: Path,
    include_charts: bool = True,
) -> Path | None:
    """
    Generate PDF report for model comparison.

    Task 6.10.9: Generate comparison PDF report

    Uses the existing PDF generator infrastructure if available.
    """
    try:
        from .pdf_generator import PDFGenerator
        from .pdf_generator import is_available as is_pdf_available
    except ImportError:
        LOGGER.warning("PDF generator not available")
        return None

    if not is_pdf_available():
        LOGGER.warning("PDF generation not available (Playwright not installed)")
        return None

    # First generate HTML content
    html_content = generate_compare_html(
        compare_json,
        include_charts=include_charts,
    )

    # Use PDF generator to convert HTML to PDF
    generator = PDFGenerator()
    try:
        generator.generate_from_html(
            html_content=html_content,
            output_path=output_path,
        )
        LOGGER.info(f"Comparison PDF saved to {output_path}")
        return output_path
    except Exception as e:
        LOGGER.error(f"Failed to generate PDF: {e}")
        return None


def generate_calibration_recommendations(
    compare_json: dict[str, Any],
) -> list[CalibrationRecommendation]:
    """
    Generate quantization calibration recommendations.

    Task 6.4.8: Show quantization calibration recommendations

    Provides guidance on improving quantization quality based on
    observed accuracy/performance gaps.
    """
    recommendations: list[CalibrationRecommendation] = []
    points = compute_tradeoff_points(compare_json)

    # Check for INT8 models - warn about GPU inference limitations
    has_int8 = any("int8" in p.precision.lower() for p in points)
    if has_int8:
        recommendations.append(
            CalibrationRecommendation(
                recommendation="Use TensorRT EP for INT8 GPU inference",
                reason="ONNX Runtime's CUDA EP lacks optimized INT8 kernels, falling back to CPU",
                severity="warning",
                affected_layers=["ConvInteger", "MatMulInteger"],
            )
        )
        recommendations.append(
            CalibrationRecommendation(
                recommendation="Alternative: Export to TensorRT engine for native INT8 GPU",
                reason="TensorRT has full INT8 GPU kernel support with tensor cores",
                severity="info",
                affected_layers=["all"],
            )
        )

    # Check for significant int8 accuracy drop
    for p in points:
        if "int8" in p.precision.lower() and p.accuracy_delta < -0.02:
            recommendations.append(
                CalibrationRecommendation(
                    recommendation="Consider increasing calibration dataset size",
                    reason=f"INT8 shows {abs(p.accuracy_delta) * 100:.1f}% accuracy drop",
                    severity="warning",
                    affected_layers=["all"],
                )
            )
            recommendations.append(
                CalibrationRecommendation(
                    recommendation="Try percentile calibration (99.99%) instead of minmax",
                    reason="Percentile calibration is more robust to outliers",
                    severity="info",
                    affected_layers=["all"],
                )
            )

    # Check for int4 issues
    for p in points:
        if "int4" in p.precision.lower() and p.accuracy_delta < -0.05:
            recommendations.append(
                CalibrationRecommendation(
                    recommendation="Consider mixed-precision: keep attention in INT8/FP16",
                    reason=f"INT4 shows significant accuracy drop ({abs(p.accuracy_delta) * 100:.1f}%)",
                    severity="critical",
                    affected_layers=["attention", "qkv_proj"],
                )
            )
            recommendations.append(
                CalibrationRecommendation(
                    recommendation="Use GPTQ or AWQ for better INT4 accuracy",
                    reason="Advanced quantization methods preserve accuracy better",
                    severity="info",
                    affected_layers=["all"],
                )
            )

    # General recommendations
    if not recommendations:
        recommendations.append(
            CalibrationRecommendation(
                recommendation="Quantization quality looks good",
                reason="No significant accuracy drops detected",
                severity="info",
                affected_layers=[],
            )
        )

    return recommendations


def build_enhanced_markdown(
    compare_json: dict[str, Any],
    include_charts: bool = True,
    assets_dir: Path | None = None,
) -> str:
    """
    Build enhanced Markdown report with trade-off analysis.

    Task 6.4.2: Create comparison Markdown table (enhanced)
    Task 6.4.3: Add trade-off analysis section

    If include_charts is True and matplotlib is available, generates
    charts and embeds them in the Markdown.
    """
    lines: list[str] = []

    model_family_id = compare_json.get("model_family_id", "unknown_model")
    baseline_precision = compare_json.get("baseline_precision", "unknown")

    lines.append(f"# Quantization Impact Report: {model_family_id}")
    lines.append("")
    lines.append(f"**Baseline**: {baseline_precision.upper()}")
    lines.append("")

    # Architecture compatibility
    if not compare_json.get("architecture_compatible", True):
        lines.append("## ⚠️ Compatibility Warnings")
        lines.append("")
        for warn in compare_json.get("compatibility_warnings", []):
            lines.append(f"- {warn}")
        lines.append("")

    # Trade-off analysis
    analysis = analyze_tradeoffs(compare_json)
    lines.append("## Trade-off Analysis")
    lines.append("")

    if analysis.get("best_balanced"):
        lines.append(f"- **Best Balance**: {analysis['best_balanced'].upper()}")
    if analysis.get("best_speed"):
        lines.append(f"- **Fastest**: {analysis['best_speed'].upper()}")
    if analysis.get("smallest"):
        lines.append(f"- **Smallest**: {analysis['smallest'].upper()}")
    lines.append("")

    # Recommendations
    lines.append("### Recommendations")
    lines.append("")
    for rec in analysis.get("recommendations", []):
        lines.append(f"- {rec}")
    lines.append("")

    # Tradeoff chart
    if include_charts and MATPLOTLIB_AVAILABLE:
        points = compute_tradeoff_points(compare_json)
        if points and assets_dir:
            chart_path = assets_dir / "tradeoff_chart.png"
            png_bytes = generate_tradeoff_chart(points, chart_path)
            if png_bytes:
                lines.append("### Accuracy vs Speedup")
                lines.append("")
                lines.append(f"![Tradeoff Chart]({chart_path.name})")
                lines.append("")

            # Memory chart
            mem_chart_path = assets_dir / "memory_savings.png"
            mem_bytes = generate_memory_savings_chart(compare_json, mem_chart_path)
            if mem_bytes:
                lines.append("### Memory Savings")
                lines.append("")
                lines.append(f"![Memory Savings]({mem_chart_path.name})")
                lines.append("")

    # Variant comparison table
    lines.append("## Variant Comparison")
    lines.append("")
    lines.append("| Precision | Size | Params | FLOPs | Speedup | Δ Accuracy |")
    lines.append("|-----------|------|--------|-------|---------|------------|")

    tradeoff_map = {p["precision"]: p for p in analysis.get("tradeoff_points", [])}

    for v in compare_json.get("variants", []):
        precision = v.get("precision", "unknown")
        size_bytes = v.get("size_bytes", 0)
        total_params = v.get("total_params")
        total_flops = v.get("total_flops")

        tp = tradeoff_map.get(precision, {})
        speedup = tp.get("speedup", 1.0)
        acc_delta = tp.get("accuracy_delta", 0.0)

        size_str = _format_bytes(size_bytes)
        params_str = _format_number(total_params) if total_params else "-"
        flops_str = _format_number(total_flops) if total_flops else "-"
        speedup_str = f"{speedup:.2f}x"
        acc_str = f"{acc_delta * 100:+.2f}%" if acc_delta != 0 else "baseline"

        lines.append(
            f"| {precision.upper()} | {size_str} | {params_str} | {flops_str} | "
            f"{speedup_str} | {acc_str} |"
        )

    lines.append("")

    # Calibration recommendations
    calib_recs = generate_calibration_recommendations(compare_json)
    if calib_recs and any(
        r.severity != "info" or "good" not in r.recommendation for r in calib_recs
    ):
        lines.append("## Calibration Recommendations")
        lines.append("")
        for rec in calib_recs:
            icon = {"info": "i", "warning": "!", "critical": "!!!"}.get(rec.severity, "*")
            lines.append(f"- {icon} **{rec.recommendation}**")
            lines.append(f"  - {rec.reason}")
        lines.append("")

    lines.append("---")
    lines.append("*Generated by HaoLine Compare Mode*")
    lines.append("")

    return "\n".join(lines)


def _format_number(n: float | None) -> str:
    """Format large numbers with K/M/G suffixes."""
    if n is None:
        return "-"
    if abs(n) >= 1e9:
        return f"{n / 1e9:.2f}G"
    if abs(n) >= 1e6:
        return f"{n / 1e6:.2f}M"
    if abs(n) >= 1e3:
        return f"{n / 1e3:.2f}K"
    return f"{n:.0f}"


def _format_bytes(n: int) -> str:
    """Format bytes with KB/MB/GB suffixes."""
    if n >= 1e9:
        return f"{n / 1e9:.2f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.2f} MB"
    if n >= 1e3:
        return f"{n / 1e3:.2f} KB"
    return f"{n} B"


def generate_compare_html(
    compare_json: dict[str, Any],
    output_path: Path | None = None,
    include_charts: bool = True,
) -> str:
    """
    Generate an HTML comparison report with engine summary panel.

    Task 6.4.7: Add engine summary panel (HTML compare report)

    This creates a TRT EngineXplorer-inspired HTML report with:
    - Engine summary panel (model family, baseline, variants)
    - Tradeoff visualization
    - Memory savings analysis
    - Calibration recommendations
    """
    model_family_id = compare_json.get("model_family_id", "unknown_model")
    baseline_precision = compare_json.get("baseline_precision", "fp32")
    variants = compare_json.get("variants", [])
    arch_compatible = compare_json.get("architecture_compatible", True)
    warnings = compare_json.get("compatibility_warnings", [])

    # Compute analysis
    analysis = analyze_tradeoffs(compare_json)
    calib_recs = generate_calibration_recommendations(compare_json)
    tradeoff_points = compute_tradeoff_points(compare_json)

    # Generate charts as base64 if matplotlib available
    tradeoff_chart_b64 = ""
    memory_chart_b64 = ""
    if include_charts and MATPLOTLIB_AVAILABLE and tradeoff_points:
        chart_bytes = generate_tradeoff_chart(tradeoff_points)
        if chart_bytes:
            tradeoff_chart_b64 = base64.b64encode(chart_bytes).decode("utf-8")

        mem_bytes = generate_memory_savings_chart(compare_json)
        if mem_bytes:
            memory_chart_b64 = base64.b64encode(mem_bytes).decode("utf-8")

    # Build HTML
    html_parts: list[str] = []

    html_parts.append(
        f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{model_family_id} - Quantization Impact</title>
    <style>
        :root {{
            --bg-deep: #000000;
            --bg-primary: #0a0a0a;
            --bg-elevated: #1a1a1a;
            --bg-card: #252525;
            --text-primary: rgba(255, 255, 255, 0.92);
            --text-secondary: rgba(255, 255, 255, 0.55);
            --accent: #0A84FF;
            --success: #30D158;
            --warning: #FFD60A;
            --error: #FF453A;
            --border: rgba(255, 255, 255, 0.08);
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', system-ui, sans-serif;
            background: var(--bg-deep);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ font-size: 2rem; font-weight: 600; margin-bottom: 0.5rem; }}
        h2 {{ font-size: 1.3rem; font-weight: 500; margin: 2rem 0 1rem; color: var(--text-primary); }}
        h3 {{ font-size: 1rem; font-weight: 500; margin: 1.5rem 0 0.75rem; color: var(--text-secondary); }}
        .subtitle {{ color: var(--text-secondary); margin-bottom: 2rem; }}

        /* Engine Summary Panel */
        .engine-summary {{
            background: var(--bg-elevated);
            border-radius: 12px;
            padding: 1.5rem;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            border: 1px solid var(--border);
        }}
        .summary-item {{ }}
        .summary-label {{ font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; }}
        .summary-value {{ font-size: 1.5rem; font-weight: 600; margin-top: 0.25rem; }}
        .summary-value.accent {{ color: var(--accent); }}
        .summary-value.success {{ color: var(--success); }}

        /* Variant Cards */
        .variants-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }}
        .variant-card {{
            background: var(--bg-card);
            border-radius: 10px;
            padding: 1.25rem;
            border: 1px solid var(--border);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .variant-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.3);
        }}
        .variant-card.baseline {{ border-color: var(--accent); }}
        .variant-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }}
        .variant-precision {{
            font-size: 1.25rem;
            font-weight: 600;
        }}
        .variant-badge {{
            font-size: 0.7rem;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            background: var(--accent);
            color: white;
        }}
        .variant-stats {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; }}
        .stat {{ }}
        .stat-label {{ font-size: 0.7rem; color: var(--text-secondary); }}
        .stat-value {{ font-size: 1rem; font-weight: 500; }}
        .stat-delta {{ font-size: 0.8rem; }}
        .stat-delta.positive {{ color: var(--success); }}
        .stat-delta.negative {{ color: var(--error); }}

        /* Charts */
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 1.5rem;
            margin-top: 1rem;
        }}
        .chart-container {{
            background: var(--bg-elevated);
            border-radius: 10px;
            padding: 1rem;
            border: 1px solid var(--border);
        }}
        .chart-container img {{
            width: 100%;
            height: auto;
            border-radius: 6px;
        }}

        /* Recommendations */
        .recommendations {{
            background: var(--bg-elevated);
            border-radius: 10px;
            padding: 1.25rem;
            border: 1px solid var(--border);
        }}
        .rec-item {{
            display: flex;
            gap: 0.75rem;
            padding: 0.75rem 0;
            border-bottom: 1px solid var(--border);
        }}
        .rec-item:last-child {{ border-bottom: none; }}
        .rec-icon {{ font-size: 1.2rem; }}
        .rec-text {{ flex: 1; }}
        .rec-reason {{ font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.25rem; }}

        /* Warnings */
        .warning-banner {{
            background: rgba(255, 214, 10, 0.1);
            border: 1px solid var(--warning);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1.5rem;
        }}
        .warning-banner h3 {{ color: var(--warning); margin: 0 0 0.5rem; }}
        .warning-list {{ list-style: none; }}
        .warning-list li {{ padding: 0.25rem 0; color: var(--text-secondary); }}

        /* Table */
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}
        .data-table th, .data-table td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        .data-table th {{
            font-size: 0.75rem;
            text-transform: uppercase;
            color: var(--text-secondary);
            font-weight: 500;
        }}
        .data-table tr:hover td {{ background: rgba(255,255,255,0.02); }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{model_family_id}</h1>
        <p class="subtitle">Quantization Impact Analysis</p>
"""
    )

    # Warning banner if architecture incompatible
    if not arch_compatible:
        html_parts.append('<div class="warning-banner">')
        html_parts.append("<h3>⚠️ Compatibility Warnings</h3>")
        html_parts.append('<ul class="warning-list">')
        for warn in warnings:
            html_parts.append(f"<li>{warn}</li>")
        html_parts.append("</ul></div>")

    # Engine Summary Panel
    html_parts.append(
        """
        <h2>Engine Summary</h2>
        <div class="engine-summary">
            <div class="summary-item">
                <div class="summary-label">Model Family</div>
                <div class="summary-value">{model_family}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Baseline</div>
                <div class="summary-value accent">{baseline}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Variants</div>
                <div class="summary-value">{num_variants}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Best Balance</div>
                <div class="summary-value success">{best_balanced}</div>
            </div>
        </div>
    """.format(
            model_family=model_family_id,
            baseline=baseline_precision.upper(),
            num_variants=len(variants),
            best_balanced=(
                analysis.get("best_balanced", "N/A").upper()
                if analysis.get("best_balanced")
                else "N/A"
            ),
        )
    )

    # Variant Cards
    html_parts.append("<h2>Variant Comparison</h2>")
    html_parts.append('<div class="variants-grid">')

    for v in variants:
        precision = v.get("precision", "unknown")
        is_baseline = precision == baseline_precision
        size_bytes = v.get("size_bytes", 0)
        total_params = v.get("total_params")
        deltas = v.get("deltas_vs_baseline")

        # Find tradeoff point for this variant
        tp = next((p for p in tradeoff_points if p.precision == precision), None)
        speedup = tp.speedup if tp else 1.0
        acc_delta = tp.accuracy_delta if tp else 0.0

        card_class = "variant-card baseline" if is_baseline else "variant-card"
        badge = '<span class="variant-badge">BASELINE</span>' if is_baseline else ""

        # Format deltas
        size_delta_str = ""
        if deltas and deltas.get("size_bytes"):
            d = deltas["size_bytes"]
            pct = (d / size_bytes) * 100 if size_bytes else 0
            delta_class = "positive" if d < 0 else "negative"
            size_delta_str = f'<span class="stat-delta {delta_class}">{pct:+.0f}%</span>'

        html_parts.append(
            f"""
            <div class="{card_class}">
                <div class="variant-header">
                    <span class="variant-precision">{precision.upper()}</span>
                    {badge}
                </div>
                <div class="variant-stats">
                    <div class="stat">
                        <div class="stat-label">Size</div>
                        <div class="stat-value">{_format_bytes(size_bytes)} {size_delta_str}</div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">Parameters</div>
                        <div class="stat-value">{_format_number(total_params) if total_params else "-"}</div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">Speedup</div>
                        <div class="stat-value">{speedup:.2f}x</div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">Accuracy Δ</div>
                        <div class="stat-value">{acc_delta * 100:+.2f}%</div>
                    </div>
                </div>
            </div>
        """
        )

    html_parts.append("</div>")  # variants-grid

    # Charts
    if tradeoff_chart_b64 or memory_chart_b64:
        html_parts.append("<h2>Visualizations</h2>")
        html_parts.append('<div class="charts-grid">')

        if tradeoff_chart_b64:
            html_parts.append(
                f"""
                <div class="chart-container">
                    <h3>Accuracy vs Speedup Tradeoff</h3>
                    <img src="data:image/png;base64,{tradeoff_chart_b64}" alt="Tradeoff Chart">
                </div>
            """
            )

        if memory_chart_b64:
            html_parts.append(
                f"""
                <div class="chart-container">
                    <h3>Memory & Size Savings</h3>
                    <img src="data:image/png;base64,{memory_chart_b64}" alt="Memory Savings Chart">
                </div>
            """
            )

        html_parts.append("</div>")  # charts-grid

    # INT8 Kernel Warning (if INT8 variant present)
    has_int8 = any(v.get("precision", "").lower() == "int8" for v in variants)
    if has_int8:
        html_parts.append(
            """
            <div class="callout warning" style="background: #3d2c00; border-left: 4px solid #ffc107; padding: 1rem; margin: 1.5rem 0; border-radius: 4px;">
                <strong style="color: #ffc107;">INT8 Performance Note</strong>
                <p style="margin: 0.5rem 0 0 0; color: #e0e0e0;">
                    ONNX Runtime does not have optimized INT8 kernels for GPU execution. The INT8 metrics shown here
                    are based on <strong>CPU inference</strong>, which is significantly slower than GPU.
                    For production INT8 GPU inference, consider converting to <strong>TensorRT</strong> or
                    <strong>OpenVINO</strong> format which have native INT8 GPU support.
                </p>
            </div>
        """
        )

    # Recommendations
    html_parts.append("<h2>Recommendations</h2>")
    html_parts.append('<div class="recommendations">')

    for rec in analysis.get("recommendations", []):
        html_parts.append(
            f"""
            <div class="rec-item">
                <span class="rec-icon">💡</span>
                <div class="rec-text">{rec}</div>
            </div>
        """
        )

    html_parts.append("</div>")

    # Calibration Recommendations
    if calib_recs and any(r.severity != "info" for r in calib_recs):
        html_parts.append("<h2>Calibration Recommendations</h2>")
        html_parts.append('<div class="recommendations">')

        for rec in calib_recs:
            icon = {"info": "i", "warning": "!", "critical": "!!!"}.get(rec.severity, "*")
            html_parts.append(
                f"""
                <div class="rec-item">
                    <span class="rec-icon">{icon}</span>
                    <div class="rec-text">
                        <strong>{rec.recommendation}</strong>
                        <div class="rec-reason">{rec.reason}</div>
                    </div>
                </div>
            """
            )

        html_parts.append("</div>")

    # Footer
    html_parts.append(
        """
        <p style="margin-top: 3rem; color: var(--text-secondary); font-size: 0.85rem; text-align: center;">
            Generated by HaoLine Compare Mode
        </p>
    </div>
</body>
</html>
    """
    )

    html_content = "\n".join(html_parts)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_content, encoding="utf-8")

    return html_content
