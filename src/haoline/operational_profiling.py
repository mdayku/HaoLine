# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
Operational profiling and system requirements analysis.

This module implements:
- Batch size scalability analysis (sweeps)
- System requirements generation (Steam-style min/rec/optimal)
- Resolution impact analysis (future)
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict

from .hardware import (
    HARDWARE_PROFILES,
    HardwareEstimates,
    HardwareEstimator,
    HardwareProfile,
)


class BatchSweepPoint(BaseModel):
    """Metrics for a single batch size point."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    batch_size: int
    vram_required_bytes: int
    estimated_latency_ms: float
    throughput_fps: float
    compute_utilization: float
    bottleneck: str
    fits_in_vram: bool


class GPUMetrics(BaseModel):
    """Real-time GPU metrics from pynvml."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vram_used_bytes: int
    vram_total_bytes: int
    gpu_utilization_percent: float
    memory_utilization_percent: float
    temperature_c: int
    power_draw_w: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "vram_used_gb": round(self.vram_used_bytes / (1024**3), 3),
            "vram_total_gb": round(self.vram_total_bytes / (1024**3), 1),
            "gpu_utilization_percent": self.gpu_utilization_percent,
            "memory_utilization_percent": self.memory_utilization_percent,
            "temperature_c": self.temperature_c,
            "power_draw_w": self.power_draw_w,
        }


class LayerProfile(BaseModel):
    """Profiling data for a single layer/operator."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    op_type: str
    duration_us: float  # Microseconds
    provider: str  # e.g., "CUDAExecutionProvider"
    input_shapes: list[list[int]]
    output_shapes: list[list[int]]

    @property
    def duration_ms(self) -> float:
        return self.duration_us / 1000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "op_type": self.op_type,
            "duration_ms": round(self.duration_ms, 3),
            "provider": self.provider,
            "input_shapes": self.input_shapes,
            "output_shapes": self.output_shapes,
        }


class ProfilingResult(BaseModel):
    """Complete profiling results from ONNX Runtime."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    total_time_ms: float
    layer_profiles: list[LayerProfile]
    gpu_metrics: GPUMetrics | None
    session_options: dict[str, Any]

    def get_slowest_layers(self, top_n: int = 10) -> list[LayerProfile]:
        """Get the N slowest layers by execution time."""
        return sorted(self.layer_profiles, key=lambda x: -x.duration_us)[:top_n]

    def get_time_by_op_type(self) -> dict[str, float]:
        """Aggregate execution time by operator type."""
        time_by_op: dict[str, float] = {}
        for layer in self.layer_profiles:
            time_by_op[layer.op_type] = time_by_op.get(layer.op_type, 0) + layer.duration_ms
        return dict(sorted(time_by_op.items(), key=lambda x: -x[1]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_time_ms": round(self.total_time_ms, 3),
            "layer_count": len(self.layer_profiles),
            "slowest_layers": [lp.to_dict() for lp in self.get_slowest_layers()],
            "time_by_op_type": {k: round(v, 3) for k, v in self.get_time_by_op_type().items()},
            "gpu_metrics": self.gpu_metrics.to_dict() if self.gpu_metrics else None,
        }


class BottleneckAnalysis(BaseModel):
    """Analysis of model performance bottlenecks."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bottleneck_type: str  # "compute-bound", "memory-bound", "balanced"
    compute_time_ms: float
    memory_time_ms: float  # Estimated memory transfer time
    compute_ratio: float  # Fraction of time spent in compute
    memory_ratio: float  # Fraction of time spent in memory ops
    theoretical_peak_tflops: float
    achieved_tflops: float
    efficiency_percent: float
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "bottleneck_type": self.bottleneck_type,
            "compute_time_ms": round(self.compute_time_ms, 3),
            "memory_time_ms": round(self.memory_time_ms, 3),
            "compute_ratio": round(self.compute_ratio, 2),
            "memory_ratio": round(self.memory_ratio, 2),
            "theoretical_peak_tflops": round(self.theoretical_peak_tflops, 2),
            "achieved_tflops": round(self.achieved_tflops, 4),
            "efficiency_percent": round(self.efficiency_percent, 1),
            "recommendations": self.recommendations,
        }


class ResolutionPoint(BaseModel):
    """Metrics for a single resolution point."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    resolution: tuple[int, int]
    resolution_str: str  # e.g., "224x224"
    flops: int
    memory_bytes: int
    vram_required_bytes: int
    estimated_latency_ms: float
    throughput_fps: float
    fits_in_vram: bool


class ResolutionSweep(BaseModel):
    """Results of a resolution sweep analysis."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    resolutions: list[str]  # ["224x224", "384x384", ...]
    flops: list[int]
    memory_gb: list[float]
    latencies: list[float]
    throughputs: list[float]
    vram_usage_gb: list[float]
    optimal_resolution: str
    max_resolution: str  # Largest resolution that fits in VRAM

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolutions": self.resolutions,
            "flops": self.flops,
            "memory_gb": self.memory_gb,
            "latencies": self.latencies,
            "throughputs": self.throughputs,
            "vram_usage_gb": self.vram_usage_gb,
            "optimal_resolution": self.optimal_resolution,
            "max_resolution": self.max_resolution,
        }


class BatchSizeSweep(BaseModel):
    """Results of a batch size sweep analysis."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    batch_sizes: list[int]
    latencies: list[float]
    throughputs: list[float]
    vram_usage_gb: list[float]
    optimal_batch_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_sizes": self.batch_sizes,
            "latencies": self.latencies,
            "throughputs": self.throughputs,
            "vram_usage_gb": self.vram_usage_gb,
            "optimal_batch_size": self.optimal_batch_size,
        }


class SystemRequirements(BaseModel):
    """Recommended hardware tiers for deployment.

    This is a lightweight, report-friendly wrapper around :class:`HardwareEstimates`.
    It deliberately mirrors the older `SystemRequirements` helper in `hardware.py`,
    exposing `minimum_gpu`, `recommended_gpu`, and `optimal_gpu` style attributes so
    existing report/HTML code (and mental model) continue to work.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Core estimates for each tier
    minimum: HardwareEstimates | None = None  # The lowest spec that runs it
    recommended: HardwareEstimates | None = None  # Good balance of cost/perf
    optimal: HardwareEstimates | None = None  # Maximum performance

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum": self.minimum.to_dict() if self.minimum else None,
            "recommended": self.recommended.to_dict() if self.recommended else None,
            "optimal": self.optimal.to_dict() if self.optimal else None,
        }

    # Backwards/HTML-friendly convenience properties ---------------------
    #
    # These keep the `reqs.minimum_gpu.name` / `reqs.minimum_vram_gb` style
    # access patterns working in `report.py` and HTML templates without
    # duplicating all the shape logic here.

    @property
    def minimum_gpu(self) -> HardwareEstimates | None:
        return self.minimum

    @property
    def recommended_gpu(self) -> HardwareEstimates | None:
        return self.recommended

    @property
    def optimal_gpu(self) -> HardwareEstimates | None:
        return self.optimal

    @staticmethod
    def _vram_gb(est: HardwareEstimates | None) -> float | None:
        if not est:
            return None
        return round(est.vram_required_bytes / (1024**3), 2)

    @property
    def minimum_vram_gb(self) -> float | None:
        return self._vram_gb(self.minimum)

    @property
    def recommended_vram_gb(self) -> float | None:
        return self._vram_gb(self.recommended)

    @property
    def optimal_vram_gb(self) -> float | None:
        return self._vram_gb(self.optimal)


class OperationalProfiler:
    """
    Analyzes model operational characteristics.
    """

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger("haoline.profiler")
        self.hw_estimator = HardwareEstimator(logger=self.logger)

    def _create_input_feed(
        self,
        sess: Any,
        batch_size: int = 1,
        seq_len: int = 128,
    ) -> dict[str, Any]:
        """
        Create input feed dict for all model inputs (Story 9.6).

        Handles multi-input models like BERT, LLMs, and multimodal models.

        Args:
            sess: ONNX Runtime InferenceSession
            batch_size: Batch size for inputs
            seq_len: Sequence length for text inputs (default: 128)

        Returns:
            Dict mapping input names to numpy arrays
        """
        import numpy as np

        input_feed = {}

        for inp in sess.get_inputs():
            name = inp.name
            shape = list(inp.shape)
            dtype_str = inp.type  # e.g., "tensor(float)", "tensor(int64)"

            # Determine numpy dtype from ONNX type
            np_dtype: type[np.generic]
            if "int64" in dtype_str:
                np_dtype = np.int64
                is_text = True
            elif "int32" in dtype_str:
                np_dtype = np.int32
                is_text = True
            elif "float16" in dtype_str:
                np_dtype = np.float16
                is_text = False
            elif "bool" in dtype_str:
                np_dtype = np.bool_
                is_text = False
            else:
                np_dtype = np.float32
                is_text = False

            # Resolve dynamic dimensions
            resolved_shape = []
            for i, dim in enumerate(shape):
                if isinstance(dim, int) and dim > 0:
                    resolved_shape.append(dim)
                elif i == 0:
                    # Batch dimension
                    resolved_shape.append(batch_size)
                elif is_text:
                    # Text models: sequence length
                    resolved_shape.append(seq_len)
                elif len(shape) == 4 and i == 1:
                    # Vision models: channels
                    resolved_shape.append(3)
                else:
                    # Vision models: spatial dims
                    resolved_shape.append(224)

            # Generate appropriate dummy data
            if is_text:
                # Token IDs: random integers in typical vocab range
                # numpy stubs are overly strict about randint dtype
                dummy: np.ndarray = np.random.randint(0, 30000, size=resolved_shape, dtype=np_dtype)  # type: ignore[arg-type]
            elif np_dtype == np.bool_:
                # Boolean masks
                dummy = np.ones(resolved_shape, dtype=np_dtype)
            else:
                # Continuous values (vision, etc.)
                dummy = np.random.randn(*resolved_shape).astype(np_dtype)

            input_feed[name] = dummy

        return input_feed

    def run_batch_sweep(
        self,
        model_params: int,
        model_flops: int,
        peak_activation_bytes: int,
        hardware: HardwareProfile,
        batch_sizes: list[int] | None = None,
        precision: str = "fp16",
    ) -> BatchSizeSweep:
        """
        Analyze performance scaling across batch sizes.

        Args:
            model_params: Total parameters
            model_flops: FLOPs per inference (batch=1)
            peak_activation_bytes: Peak activation memory (batch=1)
            hardware: Target hardware profile
            batch_sizes: List of batch sizes to test (default: powers of 2)
            precision: Precision to simulate ("fp32", "fp16", "int8")

        Returns:
            BatchSizeSweep results
        """
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

        latencies = []
        throughputs = []
        vram_usage = []
        optimal_bs = 1
        max_throughput = 0.0

        for bs in batch_sizes:
            est = self.hw_estimator.estimate(
                model_params=model_params,
                model_flops=model_flops,
                peak_activation_bytes=peak_activation_bytes,
                hardware=hardware,
                batch_size=bs,
                precision=precision,
            )

            # Calculate throughput (inferences per second)
            # If latency is infinite (OOM), throughput is 0
            throughput = 0.0
            latency = float("inf")
            vram_gb = est.vram_required_bytes / (1024**3)

            if est.theoretical_latency_ms > 0 and est.fits_in_vram:
                latency = est.theoretical_latency_ms
                throughput = (1000.0 / latency) * bs

                if throughput > max_throughput:
                    max_throughput = throughput
                    optimal_bs = bs

            latencies.append(latency)
            throughputs.append(throughput)
            vram_usage.append(vram_gb)

        return BatchSizeSweep(
            batch_sizes=batch_sizes,
            latencies=latencies,
            throughputs=throughputs,
            vram_usage_gb=vram_usage,
            optimal_batch_size=optimal_bs,
        )

    def run_batch_sweep_benchmark(
        self,
        model_path: str,
        batch_sizes: list[int] | None = None,
        num_warmup: int = 5,
        num_runs: int = 20,
    ) -> BatchSizeSweep | None:
        """
        Benchmark actual inference performance across batch sizes.

        Uses ONNX Runtime to measure real latency and throughput.
        Requires onnxruntime to be installed.

        Args:
            model_path: Path to ONNX model file
            batch_sizes: List of batch sizes to test (default: powers of 2)
            num_warmup: Number of warmup runs before timing
            num_runs: Number of timed runs per batch size

        Returns:
            BatchSizeSweep with measured (not estimated) metrics
        """
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError:
            self.logger.warning("onnxruntime not available, falling back to estimates")
            return None

        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

        # Create session
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            sess = ort.InferenceSession(model_path, providers=providers)
        except Exception as e:
            self.logger.error(f"Failed to load model for benchmarking: {e}")
            return None

        active_provider = sess.get_providers()[0]
        self.logger.info(f"Benchmarking with {active_provider}")

        # Get ALL input info (Story 9.6: Multi-input model support)
        all_inputs = sess.get_inputs()
        input_specs = []  # List of (name, shape_template, dtype, is_text)

        for inp in all_inputs:
            name = inp.name
            shape = list(inp.shape)
            dtype_str = inp.type  # e.g., "tensor(float)", "tensor(int64)"

            # Determine numpy dtype
            np_dtype: type[np.generic]
            if "int64" in dtype_str:
                np_dtype = np.int64
                is_text = True  # Likely token IDs
            elif "int32" in dtype_str:
                np_dtype = np.int32
                is_text = True
            elif "float16" in dtype_str:
                np_dtype = np.float16
                is_text = False
            else:
                np_dtype = np.float32
                is_text = False

            # Resolve dynamic dimensions with sensible defaults
            resolved_shape = []
            for i, dim in enumerate(shape):
                if isinstance(dim, int) and dim > 0:
                    resolved_shape.append(dim)
                elif i == 0:
                    resolved_shape.append(1)  # Batch dim, replaced per iteration
                elif is_text:
                    # Text models: sequence length
                    resolved_shape.append(128)  # Default seq_len
                elif len(shape) == 4 and i == 1:
                    resolved_shape.append(3)  # Channels for vision
                else:
                    resolved_shape.append(224)  # Spatial dims for vision

            input_specs.append((name, resolved_shape, np_dtype, is_text))
            self.logger.debug(
                f"  Input '{name}': shape={resolved_shape}, dtype={np_dtype.__name__}"
            )

        self.logger.info(f"Model has {len(input_specs)} input(s)")

        latencies = []
        throughputs = []
        vram_usage = []
        optimal_bs = 1
        max_throughput = 0.0

        for bs in batch_sizes:
            # Create input feed for ALL inputs
            input_feed = {}
            total_bytes = 0

            try:
                for name, shape_template, np_dtype, is_text in input_specs:
                    # Set batch size
                    shape = shape_template.copy()
                    shape[0] = bs

                    # Generate appropriate dummy data
                    if is_text:
                        # Token IDs: random integers in vocab range
                        dummy: np.ndarray = np.random.randint(0, 30000, size=shape, dtype=np_dtype)  # type: ignore[arg-type]
                    else:
                        # Vision/continuous: random floats
                        dummy = np.random.randn(*shape).astype(np_dtype)

                    input_feed[name] = dummy
                    total_bytes += dummy.nbytes

            except Exception as e:
                self.logger.warning(f"Failed to create inputs for batch {bs}: {e}")
                latencies.append(float("inf"))
                throughputs.append(0.0)
                vram_usage.append(0.0)
                continue

            # Warmup
            try:
                for _ in range(num_warmup):
                    sess.run(None, input_feed)
            except Exception as e:
                self.logger.warning(f"Batch {bs} failed (OOM?): {e}")
                latencies.append(float("inf"))
                throughputs.append(0.0)
                vram_usage.append(0.0)
                continue

            # Benchmark
            import time

            run_latencies = []
            for _ in range(num_runs):
                start = time.perf_counter()
                sess.run(None, input_feed)
                end = time.perf_counter()
                run_latencies.append((end - start) * 1000)  # ms

            # Use median latency (more stable than mean)
            run_latencies.sort()
            p50_latency = run_latencies[len(run_latencies) // 2]
            throughput = (bs * 1000.0) / p50_latency

            latencies.append(round(p50_latency, 2))
            throughputs.append(round(throughput, 1))

            # VRAM: try to measure with pynvml, fall back to estimate
            gpu_metrics = self.get_gpu_metrics()
            if gpu_metrics:
                vram_gb = gpu_metrics.vram_used_bytes / (1024**3)
            else:
                # Estimate: total input bytes * 10 for activations
                vram_gb = (total_bytes * 10) / (1024**3)
            vram_usage.append(round(vram_gb, 3))

            if throughput > max_throughput:
                max_throughput = throughput
                optimal_bs = bs

            self.logger.info(
                f"  Batch {bs}: latency={p50_latency:.2f}ms, throughput={throughput:.1f} inf/s"
            )

        return BatchSizeSweep(
            batch_sizes=batch_sizes,
            latencies=latencies,
            throughputs=throughputs,
            vram_usage_gb=vram_usage,
            optimal_batch_size=optimal_bs,
        )

    def run_resolution_sweep(
        self,
        base_flops: int,
        base_activation_bytes: int,
        base_resolution: tuple[int, int],
        model_params: int,
        hardware: HardwareProfile,
        resolutions: list[tuple[int, int]] | None = None,
        batch_size: int = 1,
        precision: str = "fp16",
    ) -> ResolutionSweep:
        """
        Analyze performance scaling across input resolutions.

        For vision models, FLOPs and memory scale approximately quadratically
        with resolution (for most architectures like ResNet, ViT, YOLO).

        Args:
            base_flops: FLOPs at base_resolution
            base_activation_bytes: Activation memory at base_resolution
            base_resolution: The resolution used for base measurements (H, W)
            model_params: Total parameters (doesn't change with resolution)
            hardware: Target hardware profile
            resolutions: List of (H, W) resolutions to test
            batch_size: Batch size for estimates
            precision: Precision ("fp32", "fp16", "int8")

        Returns:
            ResolutionSweep results
        """
        base_h, base_w = base_resolution
        base_pixels = base_h * base_w
        base_aspect = base_w / base_h if base_h > 0 else 1.0

        if resolutions is None:
            # Generate resolutions that:
            # 1. Match the aspect ratio of training data
            # 2. Only go UP TO (not above) the training resolution
            # Running above training resolution typically produces poor results
            resolutions = []

            # Common scale factors (smaller than or equal to 1.0)
            if base_aspect == 1.0:
                # Square aspect ratio
                candidates = [
                    128,
                    160,
                    192,
                    224,
                    256,
                    320,
                    384,
                    416,
                    448,
                    512,
                    640,
                    768,
                    1024,
                ]
                for size in candidates:
                    if size <= base_h:
                        resolutions.append((size, size))
            else:
                # Non-square: generate resolutions matching aspect ratio
                scale_factors = [0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
                for scale in scale_factors:
                    h = int(base_h * scale)
                    w = int(base_w * scale)
                    # Round to nearest 32 for GPU efficiency
                    h = max(32, (h // 32) * 32)
                    w = max(32, (w // 32) * 32)
                    if h <= base_h and w <= base_w and (h, w) not in resolutions:
                        resolutions.append((h, w))

            # Always include the base resolution
            if base_resolution not in resolutions:
                resolutions.append(base_resolution)

            # Sort by pixel count
            resolutions.sort(key=lambda r: r[0] * r[1])

        resolution_strs = []
        flops_list = []
        memory_gb_list = []
        latencies = []
        throughputs = []
        vram_usage = []
        optimal_res = f"{base_h}x{base_w}"
        max_res = f"{base_h}x{base_w}"
        max_throughput = 0.0
        max_fitting_pixels = 0

        for h, w in resolutions:
            res_str = f"{h}x{w}"
            resolution_strs.append(res_str)

            # Scale FLOPs and memory quadratically with resolution
            pixels = h * w
            scale_factor = pixels / base_pixels

            scaled_flops = int(base_flops * scale_factor)
            scaled_activation = int(base_activation_bytes * scale_factor)

            flops_list.append(scaled_flops)
            memory_gb_list.append(scaled_activation / (1024**3))

            # Get hardware estimates for this resolution
            est = self.hw_estimator.estimate(
                model_params=model_params,
                model_flops=scaled_flops,
                peak_activation_bytes=scaled_activation,
                hardware=hardware,
                batch_size=batch_size,
                precision=precision,
            )

            vram_gb = est.vram_required_bytes / (1024**3)
            vram_usage.append(vram_gb)

            if est.fits_in_vram and est.theoretical_latency_ms > 0:
                latency = est.theoretical_latency_ms
                throughput = (1000.0 / latency) * batch_size

                latencies.append(latency)
                throughputs.append(throughput)

                # Track max resolution that fits
                if pixels > max_fitting_pixels:
                    max_fitting_pixels = pixels
                    max_res = res_str

                # Track optimal (highest throughput)
                if throughput > max_throughput:
                    max_throughput = throughput
                    optimal_res = res_str
            else:
                latencies.append(float("inf"))
                throughputs.append(0.0)

        return ResolutionSweep(
            resolutions=resolution_strs,
            flops=flops_list,
            memory_gb=memory_gb_list,
            latencies=latencies,
            throughputs=throughputs,
            vram_usage_gb=vram_usage,
            optimal_resolution=optimal_res,
            max_resolution=max_res,
        )

    def recommend_resolution(
        self,
        base_flops: int,
        base_activation_bytes: int,
        base_resolution: tuple[int, int],
        model_params: int,
        hardware: HardwareProfile,
        target_fps: float = 30.0,
        batch_size: int = 1,
        precision: str = "fp16",
    ) -> dict[str, Any]:
        """
        Recommend optimal resolution for target hardware and latency requirements.

        Task 6.8.5: Resolution recommendations for target hardware

        Args:
            base_flops: FLOPs at base_resolution
            base_activation_bytes: Activation memory at base_resolution
            base_resolution: The resolution used for base measurements (H, W)
            model_params: Total parameters
            hardware: Target hardware profile
            target_fps: Desired frames per second (default: 30 fps)
            batch_size: Batch size
            precision: Precision for estimates

        Returns:
            Dict with recommended_resolution, max_resolution, and rationale
        """
        target_latency_ms = 1000.0 / target_fps

        # Run sweep with common resolutions
        sweep = self.run_resolution_sweep(
            base_flops=base_flops,
            base_activation_bytes=base_activation_bytes,
            base_resolution=base_resolution,
            model_params=model_params,
            hardware=hardware,
            batch_size=batch_size,
            precision=precision,
        )

        # Find resolution that meets target FPS
        recommended = None
        recommended_idx = -1
        for i, (res, lat) in enumerate(zip(sweep.resolutions, sweep.latencies, strict=False)):
            if lat != float("inf") and lat <= target_latency_ms:
                recommended = res
                recommended_idx = i

        # Build recommendation rationale
        rationale_parts = []

        if recommended:
            rationale_parts.append(
                f"Resolution **{recommended}** meets {target_fps} FPS target "
                f"({sweep.latencies[recommended_idx]:.1f}ms latency)."
            )
        else:
            # Find closest resolution that fits
            for i, (res, lat) in enumerate(zip(sweep.resolutions, sweep.latencies, strict=False)):
                if lat != float("inf"):
                    recommended = res
                    recommended_idx = i
                    break

            if recommended:
                actual_fps = 1000.0 / sweep.latencies[recommended_idx]
                rationale_parts.append(
                    f"Cannot meet {target_fps} FPS. Best achievable: "
                    f"**{recommended}** at {actual_fps:.1f} FPS."
                )
            else:
                rationale_parts.append("No resolution fits in available VRAM.")

        if sweep.max_resolution and sweep.max_resolution != recommended:
            rationale_parts.append(
                f"Maximum resolution that fits in VRAM: **{sweep.max_resolution}**."
            )

        return {
            "recommended_resolution": recommended,
            "max_resolution": sweep.max_resolution,
            "optimal_resolution": sweep.optimal_resolution,
            "target_fps": target_fps,
            "achievable_fps": (
                1000.0 / sweep.latencies[recommended_idx]
                if recommended and recommended_idx >= 0
                else 0.0
            ),
            "rationale": " ".join(rationale_parts),
            "sweep_results": sweep.to_dict(),
        }

    def determine_system_requirements(
        self,
        model_params: int,
        model_flops: int,
        peak_activation_bytes: int,
        precision: str = "fp16",
        target_fps: float = 30.0,  # For "Recommended" tier
    ) -> SystemRequirements:
        """
        Find suitable hardware tiers ("Steam-style" requirements).

        Strategy:
        - Minimum: Cheapest hardware that fits the model in VRAM (Batch=1)
        - Recommended: Cheapest hardware that hits target_fps (Batch=1) OR fits with good utilization
        - Optimal: Hardware providing highest throughput/lowest latency
        """
        candidates = []

        # Evaluate against all known profiles
        # Filter out mobile/multi-gpu for cleaner list, or keep them?
        # Let's keep single-GPU desktops/servers for simplicity of recommendation
        for name, profile in HARDWARE_PROFILES.items():
            # Skip generic CPU for this analysis unless it's the only option
            if profile.device_type == "cpu":
                continue

            # Skip mobile variants to keep list clean (optional)
            if "mobile" in name:
                continue

            est = self.hw_estimator.estimate(
                model_params=model_params,
                model_flops=model_flops,
                peak_activation_bytes=peak_activation_bytes,
                hardware=profile,
                batch_size=1,
                precision=precision,
            )
            candidates.append((profile, est))

        if not candidates:
            return SystemRequirements(minimum=None, recommended=None, optimal=None)

        # --- Find Minimum ---
        # Sort by VRAM (ascending), then FLOPs (ascending)
        candidates.sort(key=lambda x: (x[0].vram_bytes, x[0].peak_fp16_tflops))

        minimum = None
        for _, est in candidates:
            if est.fits_in_vram:
                minimum = est
                break

        # --- Find Optimal ---
        # Sort by Latency (ascending)
        candidates.sort(key=lambda x: x[1].theoretical_latency_ms)

        optimal = None
        # Filter for ones that fit
        valid_candidates = [x for x in candidates if x[1].fits_in_vram]
        if valid_candidates:
            optimal = valid_candidates[0][1]  # Fastest

        # --- Find Recommended ---
        # Heuristic: Fits VRAM AND (Latency <= 1000/target_fps OR Utilization > 0.5)
        # We want something reasonable, not necessarily the fastest (which is often H100)
        # Let's look for the "cheapest" card that meets a performance bar.

        recommended = None

        # Re-sort by cost proxy (we don't have prices in HardwareProfile, but TFLOPS is a rough proxy)
        valid_candidates.sort(key=lambda x: x[0].peak_fp16_tflops)

        target_latency_ms = 1000.0 / target_fps

        for _, est in valid_candidates:
            if est.theoretical_latency_ms <= target_latency_ms:
                recommended = est
                break

        # If nothing meets strict FPS target, pick the one with decent utilization
        if recommended is None and valid_candidates:
            # Pick median performer? Or just fallback to Minimum if nothing is fast enough?
            # Let's pick the one that is ~4x faster than minimum if possible, or just minimum
            minimum_latency = minimum.theoretical_latency_ms if minimum else float("inf")
            for _, est in valid_candidates:
                if est.theoretical_latency_ms <= minimum_latency / 4.0:
                    recommended = est
                    break

        if recommended is None:
            recommended = minimum  # Fallback

        return SystemRequirements(minimum=minimum, recommended=recommended, optimal=optimal)

    # =========================================================================
    # Story 9.2: GPU Memory Profiling
    # =========================================================================

    def get_gpu_metrics(self, device_index: int = 0) -> GPUMetrics | None:
        """
        Get real-time GPU metrics using pynvml.

        Args:
            device_index: GPU device index (default: 0)

        Returns:
            GPUMetrics with VRAM usage, utilization, temperature, power
            None if pynvml is not available or fails
        """
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            # Utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)

            # Temperature
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

            # Power
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                power_w = power_mw / 1000.0
            except pynvml.NVMLError:
                power_w = 0.0

            pynvml.nvmlShutdown()

            return GPUMetrics(
                vram_used_bytes=mem_info.used,
                vram_total_bytes=mem_info.total,
                gpu_utilization_percent=float(util.gpu),
                memory_utilization_percent=float(util.memory),
                temperature_c=temp,
                power_draw_w=power_w,
            )
        except ImportError:
            self.logger.debug("pynvml not available for GPU metrics")
            return None
        except Exception as e:
            self.logger.warning(f"Failed to get GPU metrics: {e}")
            return None

    def measure_peak_vram(
        self,
        model_path: str,
        batch_size: int = 1,
        num_runs: int = 5,
        device_index: int = 0,
    ) -> dict[str, Any]:
        """
        Measure actual peak VRAM usage during inference.

        Args:
            model_path: Path to ONNX model
            batch_size: Batch size for inference
            num_runs: Number of inference runs
            device_index: GPU device index

        Returns:
            Dict with baseline, peak, and delta VRAM usage
        """
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError:
            return {"error": "onnxruntime not available"}

        # Get baseline GPU metrics
        baseline_metrics = self.get_gpu_metrics(device_index)
        if baseline_metrics is None:
            return {"error": "pynvml not available for VRAM measurement"}

        baseline_vram = baseline_metrics.vram_used_bytes

        # Create session with CUDA
        try:
            sess = ort.InferenceSession(
                model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        except Exception as e:
            return {"error": f"Failed to create session: {e}"}

        # Get input info and create dummy input
        input_info = sess.get_inputs()[0]
        input_shape = list(input_info.shape)
        for i, dim in enumerate(input_shape):
            if not isinstance(dim, int) or dim <= 0:
                if i == 0:
                    input_shape[i] = batch_size
                elif i == 1:
                    input_shape[i] = 3
                else:
                    input_shape[i] = 224
        input_shape[0] = batch_size

        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        # Run inference and measure peak VRAM
        peak_vram = baseline_vram
        vram_samples = []

        for _ in range(num_runs):
            sess.run(None, {input_info.name: dummy_input})
            metrics = self.get_gpu_metrics(device_index)
            if metrics:
                vram_samples.append(metrics.vram_used_bytes)
                if metrics.vram_used_bytes > peak_vram:
                    peak_vram = metrics.vram_used_bytes

        delta_vram = peak_vram - baseline_vram

        return {
            "baseline_vram_gb": round(baseline_vram / (1024**3), 3),
            "peak_vram_gb": round(peak_vram / (1024**3), 3),
            "delta_vram_gb": round(delta_vram / (1024**3), 3),
            "model_vram_estimate_gb": round(delta_vram / (1024**3), 3),
            "batch_size": batch_size,
            "samples": len(vram_samples),
        }

    # =========================================================================
    # Story 9.3: Per-Layer Profiling
    # =========================================================================

    def profile_model(
        self,
        model_path: str,
        batch_size: int = 1,
        num_runs: int = 10,
        device_index: int = 0,
    ) -> ProfilingResult | None:
        """
        Profile model execution with ONNX Runtime's built-in profiler.

        Args:
            model_path: Path to ONNX model
            batch_size: Batch size for profiling
            num_runs: Number of profiling runs
            device_index: GPU device index

        Returns:
            ProfilingResult with per-layer timing data
        """
        try:
            import json
            import os
            import tempfile
            import time

            import onnxruntime as ort
        except ImportError:
            self.logger.warning("onnxruntime not available for profiling")
            return None

        # Create session with profiling enabled
        sess_options = ort.SessionOptions()
        sess_options.enable_profiling = True

        # Use temp directory for profile output
        with tempfile.TemporaryDirectory() as tmpdir:
            sess_options.profile_file_prefix = os.path.join(tmpdir, "ort_profile")

            try:
                sess = ort.InferenceSession(
                    model_path,
                    sess_options=sess_options,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
            except Exception as e:
                self.logger.error(f"Failed to create profiling session: {e}")
                return None

            # Get ALL inputs (Story 9.6: Multi-input model support)
            input_feed = self._create_input_feed(sess, batch_size)

            # Warmup
            for _ in range(3):
                sess.run(None, input_feed)

            # Profile runs
            start = time.perf_counter()
            for _ in range(num_runs):
                sess.run(None, input_feed)
            total_time_ms = ((time.perf_counter() - start) / num_runs) * 1000

            # End profiling and get the file
            profile_file = sess.end_profiling()

            # Parse profile JSON
            layer_profiles = []
            try:
                with open(profile_file, encoding="utf-8") as f:
                    profile_data = json.load(f)

                for event in profile_data:
                    if event.get("cat") == "Node":
                        name = event.get("name", "")
                        args = event.get("args", {})
                        op_type = args.get("op_name", "Unknown")
                        provider = args.get("provider", "Unknown")
                        dur = event.get("dur", 0)  # Duration in microseconds

                        # Parse shapes from args
                        input_shapes: list[list[int]] = []
                        output_shapes: list[list[int]] = []
                        for key, value in args.items():
                            if key.startswith("input_") and "shape" not in key:
                                continue
                            if "shape" in key.lower():
                                try:
                                    if isinstance(value, str):
                                        # Parse shape string like "[1,3,224,224]"
                                        shape = [
                                            int(x)
                                            for x in value.strip("[]").split(",")
                                            if x.strip()
                                        ]
                                        if "output" in key.lower():
                                            output_shapes.append(shape)
                                        else:
                                            input_shapes.append(shape)
                                except (ValueError, AttributeError):
                                    pass

                        layer_profiles.append(
                            LayerProfile(
                                name=name,
                                op_type=op_type,
                                duration_us=dur,
                                provider=provider,
                                input_shapes=input_shapes,
                                output_shapes=output_shapes,
                            )
                        )
            except Exception as e:
                self.logger.warning(f"Failed to parse profile: {e}")

            # Get GPU metrics
            gpu_metrics = self.get_gpu_metrics(device_index)

            return ProfilingResult(
                total_time_ms=total_time_ms,
                layer_profiles=layer_profiles,
                gpu_metrics=gpu_metrics,
                session_options={"batch_size": batch_size, "num_runs": num_runs},
            )

    # =========================================================================
    # Story 9.4: Bottleneck Detection
    # =========================================================================

    def analyze_bottleneck(
        self,
        model_flops: int,
        profiling_result: ProfilingResult | None,
        hardware: HardwareProfile,
        precision: str = "fp16",
    ) -> BottleneckAnalysis:
        """
        Analyze whether model is compute-bound or memory-bound.

        Uses roofline model principles:
        - Compute-bound: Time dominated by FLOP execution
        - Memory-bound: Time dominated by memory bandwidth

        Args:
            model_flops: Total FLOPs per inference
            profiling_result: Results from profile_model()
            hardware: Target hardware profile
            precision: Precision used ("fp32", "fp16", "int8")

        Returns:
            BottleneckAnalysis with classification and recommendations
        """
        # Get peak theoretical compute
        if precision == "fp32":
            peak_tflops = hardware.peak_fp32_tflops or hardware.peak_fp16_tflops / 2
        elif precision == "int8":
            peak_tflops = hardware.peak_int8_tops or hardware.peak_fp16_tflops * 2
        else:
            peak_tflops = hardware.peak_fp16_tflops

        # Actual latency
        if profiling_result:
            actual_latency_ms = profiling_result.total_time_ms
        else:
            # Estimate from theoretical
            actual_latency_ms = (model_flops / (peak_tflops * 1e12)) * 1000

        # Calculate achieved TFLOPs
        achieved_tflops = (model_flops / actual_latency_ms) / 1e9  # TFLOPS

        # Efficiency
        efficiency = (achieved_tflops / peak_tflops) * 100 if peak_tflops > 0 else 0

        # Estimate memory transfer time
        # Rough estimate: assume model params + activations need to be read
        # Memory bandwidth in bytes/s -> convert to bytes/ms
        mem_bandwidth_bytes_per_ms = hardware.memory_bandwidth_bytes_per_s / 1000  # B/s -> B/ms

        # Estimate memory footprint accessed per inference
        # This is a rough estimate - actual depends on caching, batch size, etc.
        bytes_per_param = 2 if precision == "fp16" else 4 if precision == "fp32" else 1
        # Assume we read all params once + some activation memory
        estimated_memory_bytes = model_flops * bytes_per_param / 1000  # Rough

        memory_time_ms = estimated_memory_bytes / mem_bandwidth_bytes_per_ms

        # Compute time (from achieved throughput)
        compute_time_ms = actual_latency_ms - memory_time_ms
        if compute_time_ms < 0:
            compute_time_ms = actual_latency_ms * 0.5  # Fallback

        # Ratios
        total_time = compute_time_ms + memory_time_ms
        compute_ratio = compute_time_ms / total_time if total_time > 0 else 0.5
        memory_ratio = 1.0 - compute_ratio

        # Classification
        if compute_ratio > 0.7:
            bottleneck_type = "compute-bound"
        elif memory_ratio > 0.7:
            bottleneck_type = "memory-bound"
        else:
            bottleneck_type = "balanced"

        # Recommendations based on bottleneck
        recommendations = []

        if bottleneck_type == "compute-bound":
            recommendations.extend(
                [
                    "Use INT8/FP16 quantization to reduce compute requirements",
                    "Consider model pruning to reduce FLOP count",
                    "Use Tensor Cores (if available) for matrix operations",
                    "Increase batch size to improve GPU utilization",
                ]
            )
            if efficiency < 50:
                recommendations.append(
                    f"GPU utilization is low ({efficiency:.0f}%). "
                    "Check for CPU bottlenecks or data loading issues."
                )
        elif bottleneck_type == "memory-bound":
            recommendations.extend(
                [
                    "Use lower precision (FP16/INT8) to reduce memory bandwidth",
                    "Enable operator fusion to reduce memory round-trips",
                    "Consider tensor compression or activation checkpointing",
                    "Use hardware with higher memory bandwidth",
                ]
            )
        else:  # balanced
            recommendations.extend(
                [
                    "Model has balanced compute/memory characteristics",
                    "Both quantization and bandwidth optimization may help",
                    "Profile individual layers to find specific bottlenecks",
                ]
            )

        # Add efficiency-specific recommendations
        if efficiency < 30:
            recommendations.append(
                "Very low GPU efficiency. Consider using TensorRT or "
                "ONNX Runtime optimization passes."
            )

        return BottleneckAnalysis(
            bottleneck_type=bottleneck_type,
            compute_time_ms=compute_time_ms,
            memory_time_ms=memory_time_ms,
            compute_ratio=compute_ratio,
            memory_ratio=memory_ratio,
            theoretical_peak_tflops=peak_tflops,
            achieved_tflops=achieved_tflops,
            efficiency_percent=efficiency,
            recommendations=recommendations,
        )

    # =========================================================================
    # Story 9.5: Resolution Benchmarking
    # =========================================================================

    def benchmark_resolutions(
        self,
        model_path: str,
        resolutions: list[tuple[int, int]] | None = None,
        batch_size: int = 1,
        num_warmup: int = 5,
        num_runs: int = 20,
    ) -> ResolutionSweep | None:
        """
        Benchmark actual inference performance across resolutions.

        Args:
            model_path: Path to ONNX model
            resolutions: List of (H, W) resolutions to test
            batch_size: Batch size for benchmarking
            num_warmup: Warmup runs before timing
            num_runs: Timed runs per resolution

        Returns:
            ResolutionSweep with measured (not estimated) metrics
        """
        try:
            import time

            import numpy as np
            import onnxruntime as ort
        except ImportError:
            self.logger.warning("onnxruntime not available for benchmarking")
            return None

        if resolutions is None:
            # Default resolutions for vision models
            resolutions = [
                (128, 128),
                (224, 224),
                (256, 256),
                (384, 384),
                (512, 512),
                (640, 640),
            ]

        # Create session
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            sess = ort.InferenceSession(model_path, providers=providers)
        except Exception as e:
            self.logger.error(f"Failed to create session: {e}")
            return None

        input_info = sess.get_inputs()[0]
        input_name = input_info.name

        resolution_strs: list[str] = []
        flops_list: list[int] = []
        memory_gb_list: list[float] = []
        latencies: list[float] = []
        throughputs: list[float] = []
        vram_usage: list[float] = []

        max_throughput = 0.0
        optimal_res = ""
        max_res = ""
        max_pixels = 0

        for h, w in resolutions:
            res_str = f"{h}x{w}"
            resolution_strs.append(res_str)

            # Create input with this resolution
            # Assume [N, C, H, W] format
            input_shape = list(input_info.shape)
            for i, dim in enumerate(input_shape):
                if not isinstance(dim, int) or dim <= 0:
                    if i == 0:
                        input_shape[i] = batch_size
                    elif i == 1:
                        input_shape[i] = 3
                    elif i == 2:
                        input_shape[i] = h
                    elif i == 3:
                        input_shape[i] = w

            try:
                dummy_input = np.random.randn(*input_shape).astype(np.float32)
            except Exception as e:
                self.logger.warning(f"Failed to create input for {res_str}: {e}")
                flops_list.append(0)
                memory_gb_list.append(0.0)
                latencies.append(float("inf"))
                throughputs.append(0.0)
                vram_usage.append(0.0)
                continue

            # Estimate FLOPs (scales quadratically with resolution)
            base_flops = 4_000_000_000  # Rough estimate for 224x224
            scale = (h * w) / (224 * 224)
            flops = int(base_flops * scale)
            flops_list.append(flops)

            # Memory estimate
            memory_gb = dummy_input.nbytes / (1024**3)
            memory_gb_list.append(round(memory_gb, 4))

            # Warmup
            try:
                for _ in range(num_warmup):
                    sess.run(None, {input_name: dummy_input})
            except Exception as e:
                self.logger.warning(f"Resolution {res_str} failed (OOM?): {e}")
                latencies.append(float("inf"))
                throughputs.append(0.0)
                vram_usage.append(0.0)
                continue

            # Benchmark
            run_latencies = []
            for _ in range(num_runs):
                start = time.perf_counter()
                sess.run(None, {input_name: dummy_input})
                end = time.perf_counter()
                run_latencies.append((end - start) * 1000)

            run_latencies.sort()
            p50_latency = run_latencies[len(run_latencies) // 2]
            throughput = (batch_size * 1000.0) / p50_latency

            latencies.append(round(p50_latency, 2))
            throughputs.append(round(throughput, 1))

            # VRAM estimate (or measure with pynvml)
            gpu_metrics = self.get_gpu_metrics()
            if gpu_metrics:
                vram_usage.append(round(gpu_metrics.vram_used_bytes / (1024**3), 3))
            else:
                vram_usage.append(round(dummy_input.nbytes * 2 / (1024**3), 3))

            # Track optimal and max
            pixels = h * w
            if pixels > max_pixels:
                max_pixels = pixels
                max_res = res_str

            if throughput > max_throughput:
                max_throughput = throughput
                optimal_res = res_str

            self.logger.info(
                f"  Resolution {res_str}: latency={p50_latency:.2f}ms, "
                f"throughput={throughput:.1f} inf/s"
            )

        return ResolutionSweep(
            resolutions=resolution_strs,
            flops=flops_list,
            memory_gb=memory_gb_list,
            latencies=latencies,
            throughputs=throughputs,
            vram_usage_gb=vram_usage,
            optimal_resolution=optimal_res or resolution_strs[0],
            max_resolution=max_res or resolution_strs[-1],
        )
