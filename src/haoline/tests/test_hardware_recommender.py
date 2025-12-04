# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

import unittest

from ..hardware import (
    NVIDIA_JETSON_NANO,
    NVIDIA_RTX_3060_12GB,
    BatchSizeSweeper,
    HardwareEstimator,
    SystemRequirementsRecommender,
)


class TestSystemRequirements(unittest.TestCase):
    def setUp(self):
        self.estimator = HardwareEstimator()
        self.recommender = SystemRequirementsRecommender(self.estimator)

    def test_recommend_tiny_model(self):
        # Tiny model (10MB weights) should fit on Nano
        reqs = self.recommender.recommend(
            model_params=2_500_000,  # ~10MB FP32
            model_flops=1_000_000_000,  # 1 GFLOP
            peak_activation_bytes=10_000_000,  # 10MB
            target_batch_size=1,
        )

        self.assertEqual(reqs.minimum_gpu.name, NVIDIA_JETSON_NANO.name)
        self.assertLess(reqs.minimum_vram_gb, 4.0)

    def test_recommend_large_model(self):
        # Large model (10GB weights) needs bigger GPU
        reqs = self.recommender.recommend(
            model_params=5_000_000_000,  # ~10GB FP16
            model_flops=10_000_000_000_000,  # 10 TFLOPs
            peak_activation_bytes=1_000_000_000,  # 1GB
            target_batch_size=1,
            precision="fp16",
        )

        # Should definitely NOT be Nano
        self.assertNotEqual(reqs.minimum_gpu.name, NVIDIA_JETSON_NANO.name)
        # Should be at least a 12GB card
        self.assertGreaterEqual(reqs.minimum_gpu.vram_bytes, 10 * 1024**3)


class TestBatchSizeSweep(unittest.TestCase):
    def setUp(self):
        self.estimator = HardwareEstimator()
        self.sweeper = BatchSizeSweeper(self.estimator)
        self.gpu = NVIDIA_RTX_3060_12GB

    def test_sweep_basic(self):
        # Medium model
        sweep = self.sweeper.sweep(
            model_params=100_000_000,  # 400MB
            model_flops=10_000_000_000,  # 10 GFLOPs
            peak_activation_bytes=50_000_000,  # 50MB
            hardware=self.gpu,
            max_batch_size=8,
        )

        self.assertEqual(len(sweep.batch_sizes), 4)  # 1, 2, 4, 8
        self.assertEqual(sweep.batch_sizes[-1], 8)
        self.assertTrue(all(lat > 0 for lat in sweep.latencies))

        # Throughput should generally increase (or saturate)
        self.assertGreater(sweep.throughputs[-1], sweep.throughputs[0])

    def test_sweep_oom(self):
        # Huge activation model that OOMs quickly
        sweep = self.sweeper.sweep(
            model_params=100_000_000,
            model_flops=10_000_000_000,
            peak_activation_bytes=4 * 1024**3,  # 4GB activations per sample
            hardware=self.gpu,  # 12GB VRAM
            max_batch_size=8,
        )

        # Should fit batch 1 (4GB < 12GB)
        # batch 2 (8GB < 12GB)
        # batch 4 (16GB > 12GB) -> OOM
        # So we expect 2 results
        self.assertLess(len(sweep.batch_sizes), 4)
        self.assertTrue(len(sweep.batch_sizes) >= 1)


if __name__ == "__main__":
    unittest.main()
