#!/usr/bin/env python
# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
Example: Hardware Performance Estimation

This script demonstrates how to estimate model performance
on different hardware targets.

Usage:
    python hardware_estimation.py model.onnx
"""

import sys
from pathlib import Path

from haoline import (
    HardwareEstimator,
    ModelInspector,
    get_profile,
    list_available_profiles,
)


def main():
    if len(sys.argv) < 2:
        print("Usage: python hardware_estimation.py <model.onnx>")
        print("\nAvailable hardware profiles:")
        for name in list_available_profiles()[:10]:
            print(f"  - {name}")
        print("  ... and more")
        sys.exit(1)

    model_path = Path(sys.argv[1])
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)

    # Inspect model
    inspector = ModelInspector()
    report = inspector.inspect(model_path)

    print(f"\n{'=' * 60}")
    print(f"Hardware Performance Estimates: {model_path.name}")
    print(f"{'=' * 60}")

    # Test on a few hardware profiles
    test_profiles = ["rtx4090", "rtx3080", "a100_40gb", "t4", "jetson_orin"]

    estimator = HardwareEstimator()

    for profile_name in test_profiles:
        profile = get_profile(profile_name)
        if profile is None:
            continue

        estimates = estimator.estimate(
            flops=report.flop_counts.total if report.flop_counts else 0,
            params=report.param_counts.total if report.param_counts else 0,
            activation_bytes=(
                report.memory_estimates.peak_activation_bytes if report.memory_estimates else 0
            ),
            hardware=profile,
            precision="fp16",
            batch_size=1,
        )

        print(f"\n{profile.name}:")
        print(f"  VRAM Required: {estimates.vram_required_gb:.1f} GB")
        print(f"  Fits in VRAM: {'Yes' if estimates.fits_in_vram else 'No'}")
        print(f"  Estimated Latency: {estimates.latency_ms:.2f} ms")
        print(f"  Estimated Throughput: {estimates.throughput_fps:.0f} fps")
        print(f"  Bottleneck: {estimates.bottleneck}")


if __name__ == "__main__":
    main()
