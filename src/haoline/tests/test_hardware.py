# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
Unit tests for the hardware module (profiles, detection, estimation).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from ..hardware import (
    HARDWARE_PROFILES,
    NVIDIA_A100_80GB,
    NVIDIA_JETSON_NANO,
    NVIDIA_RTX_4090,
    HardwareEstimates,
    HardwareEstimator,
    get_profile,
    list_available_profiles,
)


class TestHardwareProfiles:
    """Tests for hardware profile definitions."""

    def test_profile_registry_not_empty(self):
        """Verify hardware profile registry has entries."""
        assert len(HARDWARE_PROFILES) > 0

    def test_get_profile_by_name(self):
        """Test retrieving profiles by name."""
        profile = get_profile("a100")
        assert profile is not None
        assert "A100" in profile.name

        profile = get_profile("rtx4090")
        assert profile is not None
        assert "4090" in profile.name

    def test_get_profile_case_insensitive(self):
        """Profile lookup should be case-insensitive."""
        assert get_profile("A100") is not None
        assert get_profile("a100") is not None
        assert get_profile("RTX4090") is not None

    def test_get_unknown_profile_returns_none(self):
        """Unknown profile names should return None."""
        assert get_profile("nonexistent_gpu") is None

    def test_list_available_profiles(self):
        """Test listing available profiles."""
        profiles = list_available_profiles()
        assert len(profiles) > 0
        assert any("4090" in p for p in profiles)
        assert any("A100" in p for p in profiles)

    def test_profile_has_required_fields(self):
        """All profiles should have required fields."""
        for name, profile in HARDWARE_PROFILES.items():
            assert profile.name, f"{name} missing name"
            assert profile.vendor, f"{name} missing vendor"
            assert profile.device_type, f"{name} missing device_type"
            assert profile.vram_bytes > 0, f"{name} missing vram"
            assert profile.memory_bandwidth_bytes_per_s > 0, f"{name} missing bandwidth"

    def test_jetson_profiles_exist(self):
        """Verify Jetson edge profiles are available."""
        assert get_profile("jetson-nano") is not None
        assert get_profile("jetson-orin-nano-4gb") is not None
        assert get_profile("jetson-agx-orin") is not None


class TestHardwareProfileDataclass:
    """Tests for HardwareProfile dataclass."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        profile = NVIDIA_A100_80GB
        data = profile.to_dict()

        assert data["name"] == "NVIDIA A100 80GB SXM"
        assert data["vendor"] == "nvidia"
        assert data["device_type"] == "gpu"
        assert data["vram_gb"] == 80.0
        assert data["peak_fp32_tflops"] == 19.5

    def test_compute_capability(self):
        """Verify compute capability is set for NVIDIA GPUs."""
        assert NVIDIA_A100_80GB.compute_capability == "8.0"
        assert NVIDIA_RTX_4090.compute_capability == "8.9"


class TestHardwareEstimator:
    """Tests for HardwareEstimator class."""

    def test_estimate_fits_in_vram(self):
        """Test model that fits in VRAM."""
        estimator = HardwareEstimator()

        # Small model: 1M params, 1B FLOPs
        estimates = estimator.estimate(
            model_params=1_000_000,
            model_flops=1_000_000_000,
            peak_activation_bytes=10_000_000,  # 10 MB
            hardware=NVIDIA_A100_80GB,
            batch_size=1,
            precision="fp32",
        )

        assert estimates.fits_in_vram
        assert estimates.vram_required_bytes > 0
        assert estimates.theoretical_latency_ms > 0
        assert estimates.bottleneck in ("compute", "memory_bandwidth")

    def test_estimate_does_not_fit_in_vram(self):
        """Test model that doesn't fit in VRAM."""
        estimator = HardwareEstimator()

        # Huge model: 100B params (400GB at fp32)
        estimates = estimator.estimate(
            model_params=100_000_000_000,
            model_flops=1_000_000_000_000,
            peak_activation_bytes=10_000_000_000,
            hardware=NVIDIA_A100_80GB,
            batch_size=1,
            precision="fp32",
        )

        assert not estimates.fits_in_vram
        assert estimates.bottleneck == "vram"

    def test_precision_affects_vram(self):
        """Test that precision affects VRAM requirements."""
        estimator = HardwareEstimator()

        params = 10_000_000_000  # 10B params

        fp32_est = estimator.estimate(
            model_params=params,
            model_flops=1_000_000_000,
            peak_activation_bytes=100_000_000,
            hardware=NVIDIA_A100_80GB,
            precision="fp32",
        )

        fp16_est = estimator.estimate(
            model_params=params,
            model_flops=1_000_000_000,
            peak_activation_bytes=100_000_000,
            hardware=NVIDIA_A100_80GB,
            precision="fp16",
        )

        # FP16 should require less VRAM
        assert fp16_est.vram_required_bytes < fp32_est.vram_required_bytes

    def test_batch_size_affects_vram(self):
        """Test that batch size affects VRAM requirements."""
        estimator = HardwareEstimator()

        batch1 = estimator.estimate(
            model_params=1_000_000,
            model_flops=1_000_000_000,
            peak_activation_bytes=100_000_000,
            hardware=NVIDIA_A100_80GB,
            batch_size=1,
        )

        batch8 = estimator.estimate(
            model_params=1_000_000,
            model_flops=1_000_000_000,
            peak_activation_bytes=100_000_000,
            hardware=NVIDIA_A100_80GB,
            batch_size=8,
        )

        # Larger batch should require more VRAM
        assert batch8.vram_required_bytes > batch1.vram_required_bytes

    def test_jetson_nano_constraints(self):
        """Test estimation on resource-constrained Jetson Nano."""
        estimator = HardwareEstimator()

        # Medium model that might not fit on Jetson Nano (4GB)
        estimates = estimator.estimate(
            model_params=100_000_000,  # 100M params = 400MB at fp32
            model_flops=1_000_000_000,
            peak_activation_bytes=500_000_000,  # 500MB activations
            hardware=NVIDIA_JETSON_NANO,
            batch_size=1,
            precision="fp32",
        )

        # Should still fit at fp32 but be tight
        assert estimates.fits_in_vram
        # Jetson Nano is likely memory bandwidth limited
        assert estimates.bottleneck in ("memory_bandwidth", "compute")


class TestHardwareEstimatesDataclass:
    """Tests for HardwareEstimates dataclass."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        estimates = HardwareEstimates(
            device="NVIDIA A100 80GB",
            precision="fp16",
            batch_size=8,
            vram_required_bytes=1024 * 1024 * 1024,  # 1 GB
            fits_in_vram=True,
            theoretical_latency_ms=5.5,
            compute_utilization_estimate=0.75,
            gpu_saturation=0.000001,  # Tiny model on big GPU
            bottleneck="compute",
            model_flops=1_000_000_000,
            hardware_peak_tflops=312.0,
        )

        data = estimates.to_dict()

        assert data["device"] == "NVIDIA A100 80GB"
        assert data["precision"] == "fp16"
        assert data["batch_size"] == 8
        assert data["vram_required_gb"] == 1.0
        assert data["fits_in_vram"] is True
        assert data["theoretical_latency_ms"] == 5.5
        assert data["bottleneck"] == "compute"
        assert data["gpu_saturation"] == 0.000001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
