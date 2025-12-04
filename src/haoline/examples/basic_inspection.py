#!/usr/bin/env python
# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
Example: Basic Model Inspection

This script demonstrates how to use HaoLine to inspect a model
and generate reports programmatically (without CLI).

Usage:
    python basic_inspection.py model.onnx
"""

import sys
from pathlib import Path

from haoline import ModelInspector


def main():
    if len(sys.argv) < 2:
        print("Usage: python basic_inspection.py <model.onnx>")
        sys.exit(1)

    model_path = Path(sys.argv[1])
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)

    # Create inspector and analyze model
    inspector = ModelInspector()
    report = inspector.inspect(model_path)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Model: {model_path.name}")
    print(f"{'=' * 60}")

    # Basic stats
    if report.graph_summary:
        print("\nGraph Structure:")
        print(f"  Nodes: {report.graph_summary.num_nodes}")
        print(f"  Inputs: {report.graph_summary.num_inputs}")
        print(f"  Outputs: {report.graph_summary.num_outputs}")

    # Parameters
    if report.param_counts:
        total_params = report.param_counts.total
        print(f"\nParameters: {total_params:,}")
        if total_params > 1_000_000:
            print(f"  ({total_params / 1_000_000:.1f}M)")

    # FLOPs
    if report.flop_counts:
        total_flops = report.flop_counts.total
        print(f"\nFLOPs: {total_flops:,}")
        if total_flops > 1_000_000_000:
            print(f"  ({total_flops / 1_000_000_000:.1f}G)")

    # Memory
    if report.memory_estimates:
        print("\nMemory:")
        print(f"  Model size: {report.memory_estimates.model_size_mb:.1f} MB")
        print(f"  Peak activation: {report.memory_estimates.peak_activation_mb:.1f} MB")

    # Export to JSON
    json_path = model_path.with_suffix(".json")
    json_path.write_text(report.to_json(), encoding="utf-8")
    print(f"\nJSON report saved to: {json_path}")


if __name__ == "__main__":
    main()
