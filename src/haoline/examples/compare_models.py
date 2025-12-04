#!/usr/bin/env python
# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
Example: Compare Multiple Models

This script demonstrates how to compare multiple model variants
(e.g., FP32 vs FP16 vs INT8) programmatically.

Usage:
    python compare_models.py model_fp32.onnx model_fp16.onnx model_int8.onnx
"""

import sys
from pathlib import Path

from haoline import ModelInspector


def format_size(bytes_val: int) -> str:
    """Format bytes as human-readable string."""
    if bytes_val >= 1_000_000_000:
        return f"{bytes_val / 1_000_000_000:.1f} GB"
    elif bytes_val >= 1_000_000:
        return f"{bytes_val / 1_000_000:.1f} MB"
    elif bytes_val >= 1_000:
        return f"{bytes_val / 1_000:.1f} KB"
    return f"{bytes_val} B"


def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_models.py <model1.onnx> <model2.onnx> [model3.onnx ...]")
        sys.exit(1)

    model_paths = [Path(p) for p in sys.argv[1:]]

    # Validate all models exist
    for path in model_paths:
        if not path.exists():
            print(f"Error: Model not found: {path}")
            sys.exit(1)

    # Inspect all models
    inspector = ModelInspector()
    reports = []

    print("Analyzing models...")
    for path in model_paths:
        print(f"  - {path.name}")
        report = inspector.inspect(path)
        reports.append((path.name, report))

    # Print comparison table
    print(f"\n{'=' * 80}")
    print("MODEL COMPARISON")
    print(f"{'=' * 80}")

    # Header
    print(f"\n{'Metric':<25}", end="")
    for name, _ in reports:
        print(f"{name:<20}", end="")
    print()
    print("-" * (25 + 20 * len(reports)))

    # File size
    print(f"{'File Size':<25}", end="")
    for path, _report in zip(model_paths, reports, strict=True):
        size = path.stat().st_size
        print(f"{format_size(size):<20}", end="")
    print()

    # Parameters
    print(f"{'Parameters':<25}", end="")
    for _, report in reports:
        params = report[1].param_counts.total if report[1].param_counts else 0
        print(f"{params:,}".ljust(20), end="")
    print()

    # FLOPs
    print(f"{'FLOPs':<25}", end="")
    for _, report in reports:
        flops = report[1].flop_counts.total if report[1].flop_counts else 0
        print(f"{flops:,}".ljust(20), end="")
    print()

    # Memory
    print(f"{'Peak Activation (MB)':<25}", end="")
    for _, report in reports:
        mem = report[1].memory_estimates.peak_activation_mb if report[1].memory_estimates else 0
        print(f"{mem:.1f}".ljust(20), end="")
    print()

    # Compute deltas vs first model (baseline)
    baseline_name, baseline_report = reports[0]
    baseline_size = model_paths[0].stat().st_size

    print(f"\n{'=' * 80}")
    print(f"DELTAS vs {baseline_name} (baseline)")
    print(f"{'=' * 80}")

    for _i, (path, (name, report)) in enumerate(zip(model_paths[1:], reports[1:], strict=True)):
        size = path.stat().st_size
        size_delta = (size - baseline_size) / baseline_size * 100

        params = report.param_counts.total if report.param_counts else 0
        baseline_params = baseline_report.param_counts.total if baseline_report.param_counts else 0
        params_delta = (params - baseline_params) / baseline_params * 100 if baseline_params else 0

        print(f"\n{name}:")
        print(f"  Size: {size_delta:+.1f}%")
        print(f"  Parameters: {params_delta:+.1f}%")


if __name__ == "__main__":
    main()
