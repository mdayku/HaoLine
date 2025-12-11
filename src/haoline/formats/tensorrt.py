# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
TensorRT engine reader for NVIDIA optimized models.

TensorRT engines (.engine, .plan) are compiled, optimized models for NVIDIA GPUs.
This reader extracts:
- Engine metadata (TRT version, build configuration)
- Layer information (names, types, shapes, precision)
- Memory footprint and optimization info
- Hardware binding (GPU architecture, compute capability)

Requires: tensorrt>=10.0.0 (pip install haoline[tensorrt])
Requires: NVIDIA GPU with compatible CUDA driver

Reference: https://docs.nvidia.com/deeplearning/tensorrt/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, computed_field


class TRTLayerInfo(BaseModel):
    """Information about a single layer in the TensorRT engine."""

    model_config = ConfigDict(frozen=True)

    name: str
    type: str
    precision: str = "FP32"
    input_shapes: list[tuple[int, ...]] = Field(default_factory=list)
    output_shapes: list[tuple[int, ...]] = Field(default_factory=list)
    # Tactic/kernel info if available (Task 22.4.3)
    tactic: str | None = None
    tactic_name: str | None = None  # Human-readable tactic name
    # Fusion info
    is_fused: bool = False
    fused_ops: list[str] = Field(default_factory=list)  # Original ops that were fused
    # Timing info (if profiling enabled) - Task 22.4.1
    avg_time_ms: float | None = None
    min_time_ms: float | None = None
    max_time_ms: float | None = None
    # Workspace/memory info - Task 22.4.2
    workspace_size_bytes: int = 0
    output_memory_bytes: int = 0  # Total output tensor memory
    # Compute vs memory bound classification - Task 22.4.4
    bound_type: str = "Unknown"  # "compute", "memory", "balanced", "unknown"
    arithmetic_intensity: float | None = None  # FLOPs / bytes accessed
    # Origin info for ONNX mapping
    origin: str | None = None  # Original ONNX node name


class TRTBindingInfo(BaseModel):
    """Information about an input/output binding."""

    model_config = ConfigDict(frozen=True)

    name: str
    shape: tuple[int, ...]
    dtype: str
    is_input: bool


class TRTBuilderConfig(BaseModel):
    """Builder configuration extracted from TensorRT engine."""

    model_config = ConfigDict(frozen=True)

    # Basic counts
    num_io_tensors: int = 0
    num_layers: int = 0
    # Batch configuration
    max_batch_size: int = 1
    has_implicit_batch: bool = False
    # Memory
    device_memory_size: int = 0  # Workspace size in bytes
    # DLA (Deep Learning Accelerator) - for Jetson devices
    dla_core: int = -1  # -1 means GPU only, 0/1 for DLA core selection
    # Optimization profiles (for dynamic shapes)
    num_optimization_profiles: int = 0
    # Hardware mode
    hardware_compatibility_level: str = "None"
    # Sparsity
    engine_capability: str = "Standard"


class TRTPerformanceMetadata(BaseModel):
    """
    Performance metadata for a TensorRT engine.

    Task 22.4: TRT Performance Metadata Panel
    Extracted from engine inspector and optional profiling data.
    """

    model_config = ConfigDict(frozen=True)

    # Total inference time estimate (if profiled)
    total_time_ms: float | None = None
    # Per-layer timing availability
    has_layer_timing: bool = False
    # Slowest layers (top 10)
    slowest_layers: list[tuple[str, float]] = Field(default_factory=list)  # (name, ms)
    # Total workspace used
    total_workspace_bytes: int = 0
    # Memory bandwidth utilization estimate
    memory_bandwidth_gbps: float | None = None
    # Compute utilization estimate
    compute_utilization_pct: float | None = None
    # Bound type distribution
    compute_bound_layers: int = 0
    memory_bound_layers: int = 0
    balanced_layers: int = 0
    unknown_bound_layers: int = 0


class TRTEngineInfo(BaseModel):
    """Parsed TensorRT engine information."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: Path
    # Engine metadata
    trt_version: str
    builder_config: TRTBuilderConfig = Field(default_factory=TRTBuilderConfig)
    # Hardware binding
    device_name: str = "Unknown"
    compute_capability: tuple[int, int] = (0, 0)
    # Layers and bindings
    layers: list[TRTLayerInfo] = Field(default_factory=list)
    bindings: list[TRTBindingInfo] = Field(default_factory=list)
    # Memory info
    device_memory_bytes: int = 0
    # Performance metadata (Task 22.4)
    performance: TRTPerformanceMetadata = Field(default_factory=TRTPerformanceMetadata)
    # Optimization info (if available from build)
    has_implicit_batch: bool = False
    max_batch_size: int = 1

    @computed_field  # type: ignore[prop-decorator]
    @property
    def layer_count(self) -> int:
        """Number of layers in the engine."""
        return len(self.layers)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def layer_type_counts(self) -> dict[str, int]:
        """Count of layers by type."""
        counts: dict[str, int] = {}
        for layer in self.layers:
            counts[layer.type] = counts.get(layer.type, 0) + 1
        return counts

    @computed_field  # type: ignore[prop-decorator]
    @property
    def precision_breakdown(self) -> dict[str, int]:
        """Count of layers by precision."""
        breakdown: dict[str, int] = {}
        for layer in self.layers:
            breakdown[layer.precision] = breakdown.get(layer.precision, 0) + 1
        return breakdown

    @computed_field  # type: ignore[prop-decorator]
    @property
    def fused_layer_count(self) -> int:
        """Count of fused layers."""
        return sum(1 for layer in self.layers if layer.is_fused)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def fusion_ratio(self) -> float:
        """Ratio of fused layers (0.0-1.0)."""
        if not self.layers:
            return 0.0
        return float(self.fused_layer_count) / float(len(self.layers))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def original_ops_fused(self) -> int:
        """Estimated count of original ops that were fused into single kernels."""
        count = 0
        for layer in self.layers:
            if layer.is_fused:
                count += len(layer.fused_ops) if layer.fused_ops else 2  # Minimum 2 ops per fusion
        return count

    @computed_field  # type: ignore[prop-decorator]
    @property
    def input_bindings(self) -> list[TRTBindingInfo]:
        """Input bindings only."""
        return [b for b in self.bindings if b.is_input]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def output_bindings(self) -> list[TRTBindingInfo]:
        """Output bindings only."""
        return [b for b in self.bindings if not b.is_input]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return cast(dict[str, Any], self.model_dump(mode="json"))


# ============================================================================
# Quantization Bottleneck Analysis (Story 22.8)
# ============================================================================


class FailedFusionPattern(BaseModel):
    """A pattern that should have fused but appears as separate layers."""

    model_config = ConfigDict(frozen=True)

    pattern_type: str  # "Conv+BN+ReLU", "MatMul+Add", "LayerNorm+Add", etc.
    layer_names: list[str]  # Names of the separate layers
    layer_indices: list[int]  # Indices in the layer list
    expected_fused_name: str  # What it should be called if fused
    reason: str  # Why this is a problem
    speed_impact: str  # "High", "Medium", "Low"


class BottleneckZone(BaseModel):
    """A contiguous region of non-quantized (FP32) layers."""

    model_config = ConfigDict(frozen=True)

    start_idx: int
    end_idx: int
    layer_count: int
    layer_names: list[str]
    layer_types: list[str]
    estimated_time_pct: float = 0.0  # % of inference time (if timing available)
    severity: str = "Medium"  # "Critical", "High", "Medium", "Low"


class QuantBottleneckAnalysis(BaseModel):
    """Complete quantization bottleneck analysis for a TRT engine."""

    model_config = ConfigDict(frozen=True)

    # Summary metrics
    int8_layer_count: int = 0
    fp16_layer_count: int = 0
    fp32_layer_count: int = 0
    total_layer_count: int = 0

    # Derived metrics
    quantization_ratio: float = 0.0  # % of layers that are INT8
    fp32_fallback_ratio: float = 0.0  # % of layers that fell back to FP32

    # Bottleneck details
    failed_fusions: list[FailedFusionPattern] = Field(default_factory=list)
    bottleneck_zones: list[BottleneckZone] = Field(default_factory=list)

    # Estimated impact
    estimated_speedup_potential: float = 1.0  # e.g., 1.7 = could be 1.7x faster
    recommendations: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def largest_bottleneck(self) -> BottleneckZone | None:
        """The largest bottleneck zone by layer count."""
        if not self.bottleneck_zones:
            return None
        return max(self.bottleneck_zones, key=lambda z: z.layer_count)


def analyze_quant_bottlenecks(engine_info: TRTEngineInfo) -> QuantBottleneckAnalysis:
    """
    Analyze a TensorRT engine for quantization bottlenecks.

    Args:
        engine_info: Parsed TRT engine information.

    Returns:
        QuantBottleneckAnalysis with failed fusions, bottleneck zones, and recommendations.
    """
    layers = engine_info.layers

    # Count by precision
    int8_count = sum(1 for lyr in layers if lyr.precision == "INT8")
    fp16_count = sum(1 for lyr in layers if lyr.precision == "FP16")
    fp32_count = sum(1 for lyr in layers if lyr.precision in ("FP32", "Mixed", "Unknown"))
    total = len(layers)

    # Detect failed fusions
    failed_fusions = _detect_failed_fusions(layers)

    # Detect bottleneck zones (consecutive FP32 layers)
    bottleneck_zones = _detect_bottleneck_zones(layers)

    # Calculate ratios
    quant_ratio = int8_count / total if total > 0 else 0.0
    fp32_ratio = fp32_count / total if total > 0 else 0.0

    # Estimate speedup potential based on FP32 ratio
    # Rough heuristic: INT8 is ~2-4x faster than FP32
    # If 30% of layers are FP32, potential speedup = 1 + (0.3 * 2.0) = 1.6x
    speedup_potential = 1.0 + (fp32_ratio * 2.0) if fp32_ratio > 0.1 else 1.0

    # Generate recommendations
    recommendations = _generate_recommendations(
        failed_fusions, bottleneck_zones, fp32_ratio, int8_count
    )

    return QuantBottleneckAnalysis(
        int8_layer_count=int8_count,
        fp16_layer_count=fp16_count,
        fp32_layer_count=fp32_count,
        total_layer_count=total,
        quantization_ratio=quant_ratio,
        fp32_fallback_ratio=fp32_ratio,
        failed_fusions=failed_fusions,
        bottleneck_zones=bottleneck_zones,
        estimated_speedup_potential=round(speedup_potential, 2),
        recommendations=recommendations,
    )


def _detect_failed_fusions(layers: list[TRTLayerInfo]) -> list[FailedFusionPattern]:
    """Detect ops that should have fused but appear separately."""
    failed = []

    # Common fusion patterns that should appear as single layer
    # Pattern: (sequence of layer types, expected fused name, speed impact)
    FUSION_PATTERNS = [
        (["Convolution", "Scale", "Activation"], "Conv+BN+ReLU", "High"),
        (["Convolution", "Scale"], "Conv+BN", "Medium"),
        (["Convolution", "Activation"], "Conv+ReLU", "Medium"),
        (["MatrixMultiply", "ElementWise"], "MatMul+Add", "Medium"),
        (["Shuffle", "Scale", "Shuffle"], "LayerNorm", "High"),
        (["SoftMax", "Scale"], "ScaledSoftmax", "Low"),
    ]

    for i in range(len(layers)):
        for pattern_types, expected_name, impact in FUSION_PATTERNS:
            pattern_len = len(pattern_types)
            if i + pattern_len > len(layers):
                continue

            # Check if this sequence matches the pattern
            window = layers[i : i + pattern_len]

            # Skip if already fused
            if any(lyr.is_fused for lyr in window):
                continue

            # Check type match (flexible matching)
            matches = True
            for j, expected_type in enumerate(pattern_types):
                actual_type = window[j].type
                if expected_type.lower() not in actual_type.lower():
                    matches = False
                    break

            if matches:
                # Check if these layers are all FP32 (missed quantization opportunity)
                all_fp32 = all(lyr.precision in ("FP32", "Mixed") for lyr in window)
                if all_fp32:
                    failed.append(
                        FailedFusionPattern(
                            pattern_type=expected_name,
                            layer_names=[lyr.name for lyr in window],
                            layer_indices=list(range(i, i + pattern_len)),
                            expected_fused_name=f"{expected_name}_{i}",
                            reason=f"Sequential {expected_name} pattern not fused, all layers FP32",
                            speed_impact=impact,
                        )
                    )

    return failed


def _detect_bottleneck_zones(layers: list[TRTLayerInfo]) -> list[BottleneckZone]:
    """Find contiguous regions of FP32 layers."""
    zones = []
    current_zone_start = None
    current_zone_layers: list[TRTLayerInfo] = []

    for i, layer in enumerate(layers):
        is_fp32 = layer.precision in ("FP32", "Mixed", "Unknown")

        if is_fp32:
            if current_zone_start is None:
                current_zone_start = i
            current_zone_layers.append(layer)
        else:
            # End of FP32 zone
            if current_zone_start is not None and len(current_zone_layers) >= 2:
                # Only report zones with 2+ layers as bottlenecks
                severity = _zone_severity(len(current_zone_layers))
                zones.append(
                    BottleneckZone(
                        start_idx=current_zone_start,
                        end_idx=i - 1,
                        layer_count=len(current_zone_layers),
                        layer_names=[lyr.name for lyr in current_zone_layers],
                        layer_types=[lyr.type for lyr in current_zone_layers],
                        severity=severity,
                    )
                )
            current_zone_start = None
            current_zone_layers = []

    # Handle zone at end
    if current_zone_start is not None and len(current_zone_layers) >= 2:
        severity = _zone_severity(len(current_zone_layers))
        zones.append(
            BottleneckZone(
                start_idx=current_zone_start,
                end_idx=len(layers) - 1,
                layer_count=len(current_zone_layers),
                layer_names=[lyr.name for lyr in current_zone_layers],
                layer_types=[lyr.type for lyr in current_zone_layers],
                severity=severity,
            )
        )

    return zones


def _zone_severity(layer_count: int) -> str:
    """Determine severity based on zone size."""
    if layer_count >= 10:
        return "Critical"
    elif layer_count >= 5:
        return "High"
    elif layer_count >= 3:
        return "Medium"
    return "Low"


def _generate_recommendations(
    failed_fusions: list[FailedFusionPattern],
    bottleneck_zones: list[BottleneckZone],
    fp32_ratio: float,
    int8_count: int,
) -> list[str]:
    """Generate actionable recommendations based on analysis."""
    recs = []

    if fp32_ratio > 0.5:
        recs.append(
            "High FP32 fallback (>50%) - Consider re-calibrating with more representative data"
        )
    elif fp32_ratio > 0.2:
        recs.append("Moderate FP32 fallback - Check calibration dataset covers edge cases")

    if int8_count == 0:
        recs.append("No INT8 layers detected - Ensure TensorRT builder has INT8 mode enabled")

    high_impact_fusions = [f for f in failed_fusions if f.speed_impact == "High"]
    if high_impact_fusions:
        recs.append(
            f"{len(high_impact_fusions)} high-impact fusion(s) failed - "
            "Consider using TensorRT plugins or restructuring model"
        )

    critical_zones = [z for z in bottleneck_zones if z.severity == "Critical"]
    if critical_zones:
        largest = max(critical_zones, key=lambda z: z.layer_count)
        recs.append(
            f"Critical bottleneck: {largest.layer_count} consecutive FP32 layers - "
            "Focus calibration on these layers"
        )

    if not recs:
        recs.append("Quantization looks good! Most layers are using INT8/FP16.")

    return recs


class TRTEngineReader:
    """Reader for TensorRT engine files (.engine, .plan)."""

    def __init__(self, path: str | Path):
        """
        Initialize reader with file path.

        Args:
            path: Path to the TensorRT engine file.

        Raises:
            ImportError: If tensorrt is not installed.
            FileNotFoundError: If the file doesn't exist.
        """
        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {self.path}")

        # Check TensorRT availability
        try:
            import tensorrt  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "tensorrt required. Install with: pip install haoline[tensorrt]\n"
                "Note: Requires NVIDIA GPU and CUDA 12.x"
            ) from e

    def read(self) -> TRTEngineInfo:
        """
        Read and parse the TensorRT engine.

        Returns:
            TRTEngineInfo with engine metadata and layer information.

        Raises:
            RuntimeError: If the engine cannot be deserialized (e.g., GPU mismatch).
        """
        import tensorrt as trt

        # Create logger and runtime
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        # Read engine file
        with open(self.path, "rb") as f:
            engine_data = f.read()

        # Deserialize engine
        engine = runtime.deserialize_cuda_engine(engine_data)
        if engine is None:
            raise RuntimeError(
                f"Failed to deserialize TensorRT engine: {self.path}\n"
                "This may be due to GPU architecture mismatch or TRT version incompatibility."
            )

        # Extract TensorRT version
        trt_version = trt.__version__

        # Get device info
        device_name, compute_cap = self._get_device_info()

        # Extract bindings (inputs/outputs)
        bindings = self._extract_bindings(engine)

        # Extract layers using inspector if available
        layers = self._extract_layers(engine)

        # Get memory info
        device_memory = engine.device_memory_size if hasattr(engine, "device_memory_size") else 0

        # Check for implicit batch dimension (legacy)
        has_implicit_batch = False
        max_batch_size = 1
        if hasattr(engine, "has_implicit_batch_dimension"):
            has_implicit_batch = engine.has_implicit_batch_dimension
        if hasattr(engine, "max_batch_size"):
            max_batch_size = engine.max_batch_size

        # Extract builder configuration
        builder_config = self._extract_builder_config(engine, device_memory)

        # Compute performance metadata (Task 22.4)
        performance = self._compute_performance_metadata(layers)

        return TRTEngineInfo(
            path=self.path,
            trt_version=trt_version,
            builder_config=builder_config,
            device_name=device_name,
            compute_capability=compute_cap,
            layers=layers,
            bindings=bindings,
            device_memory_bytes=device_memory,
            performance=performance,
            has_implicit_batch=has_implicit_batch,
            max_batch_size=max_batch_size,
        )

    def _compute_performance_metadata(self, layers: list[TRTLayerInfo]) -> TRTPerformanceMetadata:
        """
        Compute performance metadata summary from layer information.

        Task 22.4: TRT Performance Metadata Panel
        """
        # Task 22.4.1: Check if we have timing data
        layers_with_timing = [lyr for lyr in layers if lyr.avg_time_ms is not None]
        has_timing = len(layers_with_timing) > 0

        # Total time if timing available
        total_time: float | None = None
        slowest: list[tuple[str, float]] = []
        if has_timing:
            total_time = sum(lyr.avg_time_ms or 0.0 for lyr in layers_with_timing)
            # Sort by time descending
            sorted_by_time = sorted(
                [(lyr.name, lyr.avg_time_ms or 0.0) for lyr in layers_with_timing],
                key=lambda x: x[1],
                reverse=True,
            )
            slowest = sorted_by_time[:10]

        # Task 22.4.2: Total workspace
        total_workspace = sum(lyr.workspace_size_bytes for lyr in layers)

        # Task 22.4.4: Bound type distribution
        compute_bound = sum(1 for lyr in layers if lyr.bound_type == "compute")
        memory_bound = sum(1 for lyr in layers if lyr.bound_type == "memory")
        balanced = sum(1 for lyr in layers if lyr.bound_type == "balanced")
        unknown_bound = sum(1 for lyr in layers if lyr.bound_type == "unknown")

        return TRTPerformanceMetadata(
            total_time_ms=total_time,
            has_layer_timing=has_timing,
            slowest_layers=slowest,
            total_workspace_bytes=total_workspace,
            compute_bound_layers=compute_bound,
            memory_bound_layers=memory_bound,
            balanced_layers=balanced,
            unknown_bound_layers=unknown_bound,
        )

    def _get_device_info(self) -> tuple[str, tuple[int, int]]:
        """Get GPU device information."""
        try:
            import torch

            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                return props.name, (props.major, props.minor)
        except ImportError:
            pass

        # Fallback: try pynvml
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            # pynvml doesn't easily give compute capability
            return name, (0, 0)
        except Exception:
            pass

        return "Unknown GPU", (0, 0)

    def _extract_builder_config(self, engine: Any, device_memory: int) -> TRTBuilderConfig:
        """Extract builder configuration from engine."""
        # Basic counts
        num_io_tensors = engine.num_io_tensors
        num_layers = engine.num_layers

        # Batch configuration
        max_batch_size = 1
        has_implicit_batch = False
        if hasattr(engine, "has_implicit_batch_dimension"):
            has_implicit_batch = engine.has_implicit_batch_dimension
        if hasattr(engine, "max_batch_size"):
            max_batch_size = engine.max_batch_size

        # Optimization profiles (for dynamic shapes)
        num_profiles = 0
        if hasattr(engine, "num_optimization_profiles"):
            num_profiles = engine.num_optimization_profiles

        # DLA core (-1 = GPU only)
        dla_core = -1
        # TRT doesn't expose DLA config directly from engine after serialization

        # Hardware compatibility level
        hw_compat = "None"
        if hasattr(engine, "hardware_compatibility_level"):
            hw_compat = str(engine.hardware_compatibility_level)

        # Engine capability (Standard, Safety, DLA_Standalone)
        engine_cap = "Standard"
        if hasattr(engine, "engine_capability"):
            engine_cap = str(engine.engine_capability).replace("EngineCapability.", "")

        return TRTBuilderConfig(
            num_io_tensors=num_io_tensors,
            num_layers=num_layers,
            max_batch_size=max_batch_size,
            has_implicit_batch=has_implicit_batch,
            device_memory_size=device_memory,
            dla_core=dla_core,
            num_optimization_profiles=num_profiles,
            hardware_compatibility_level=hw_compat,
            engine_capability=engine_cap,
        )

    def _extract_bindings(self, engine: Any) -> list[TRTBindingInfo]:
        """Extract input/output binding information."""
        import tensorrt as trt

        bindings = []

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = tuple(engine.get_tensor_shape(name))
            dtype = str(engine.get_tensor_dtype(name)).replace("DataType.", "")
            mode = engine.get_tensor_mode(name)
            is_input = mode == trt.TensorIOMode.INPUT

            bindings.append(
                TRTBindingInfo(
                    name=name,
                    shape=shape,
                    dtype=dtype,
                    is_input=is_input,
                )
            )

        return bindings

    def _extract_layers(self, engine: Any) -> list[TRTLayerInfo]:
        """Extract layer information using engine inspector."""
        import tensorrt as trt

        layers = []

        # Try to use inspector API (TRT 8.5+)
        try:
            inspector = engine.create_engine_inspector()
            if inspector is not None:
                import json

                # TRT 10 returns JSON with layer names (may include fusions with '+')
                layer_json = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
                if layer_json:
                    layer_data = json.loads(layer_json)
                    layer_list = layer_data.get("Layers", [])

                    for idx, layer_entry in enumerate(layer_list):
                        # TRT 10: layer_entry is a string (layer name)
                        # TRT 8.x: layer_entry might be a dict with more info
                        if isinstance(layer_entry, str):
                            name = layer_entry
                            layer_type = self._infer_layer_type(name)
                            precision = self._infer_precision(name)
                            tactic = None
                            tactic_name = None
                            origin = None
                            workspace_size = 0
                            output_memory = 0
                        else:
                            # Dict format (older TRT versions) with more details
                            name = layer_entry.get("Name", "Unknown")
                            layer_type = layer_entry.get("LayerType", self._infer_layer_type(name))
                            precision = layer_entry.get("Precision", "FP32")
                            tactic = layer_entry.get("TacticName") or layer_entry.get("Tactic")
                            tactic_name = layer_entry.get("TacticName")
                            origin = layer_entry.get("Origin")
                            # Task 22.4.2: Extract workspace size
                            workspace_size = layer_entry.get("WorkspaceSize", 0)
                            output_memory = layer_entry.get("OutputSize", 0)

                        # Check if this is a fused layer
                        is_fused = "+" in name
                        fused_ops = self._extract_fused_ops(name) if is_fused else []

                        # Try to get per-layer detailed info (TRT 10+)
                        layer_detail = self._get_layer_detail(inspector, idx, trt)
                        if layer_detail:
                            # Override with detailed info if available
                            if "Precision" in layer_detail:
                                precision = layer_detail["Precision"]
                            if "TacticName" in layer_detail:
                                tactic = layer_detail["TacticName"]
                                tactic_name = layer_detail["TacticName"]
                            if "Origin" in layer_detail:
                                origin = layer_detail["Origin"]
                            # Task 22.4.2/22.4.6: Workspace and memory info
                            if "WorkspaceSize" in layer_detail:
                                workspace_size = layer_detail["WorkspaceSize"]
                            if "TotalFootprintBytes" in layer_detail:
                                output_memory = layer_detail["TotalFootprintBytes"]
                            elif "OutputSize" in layer_detail:
                                output_memory = layer_detail["OutputSize"]

                        # Task 22.4.4: Infer compute vs memory bound
                        bound_type = self._classify_layer_bound(layer_type, precision)

                        layers.append(
                            TRTLayerInfo(
                                name=name,
                                type=layer_type,
                                precision=precision,
                                tactic=tactic,
                                tactic_name=tactic_name,
                                is_fused=is_fused,
                                fused_ops=fused_ops,
                                origin=origin,
                                workspace_size_bytes=workspace_size,
                                output_memory_bytes=output_memory,
                                bound_type=bound_type,
                            )
                        )
                    return layers
        except Exception:
            pass

        # Fallback: basic layer count without details
        for i in range(engine.num_layers):
            layers.append(
                TRTLayerInfo(
                    name=f"layer_{i}",
                    type="Unknown",
                    precision="Unknown",
                )
            )

        return layers

    def _classify_layer_bound(self, layer_type: str, precision: str) -> str:
        """
        Classify a layer as compute-bound, memory-bound, or balanced.

        Task 22.4.4: Identify memory-bound vs compute-bound layers.

        Heuristics based on layer type:
        - Convolutions/MatMul: Typically compute-bound for larger kernels
        - Elementwise/Activation: Memory-bound (low arithmetic intensity)
        - Normalization: Memory-bound (multiple passes over data)
        - Pooling: Memory-bound (low compute, data movement)
        """
        layer_lower = layer_type.lower()

        # Compute-bound operations
        compute_bound_ops = {"convolution", "matrixmultiply", "matmul", "fullyconnected", "gemm"}
        for op in compute_bound_ops:
            if op in layer_lower:
                # FP16/INT8 may shift towards memory bound
                if precision in ("FP16", "INT8"):
                    return "balanced"
                return "compute"

        # Memory-bound operations
        memory_bound_ops = {
            "elementwise",
            "activation",
            "relu",
            "sigmoid",
            "tanh",
            "softmax",
            "pooling",
            "pool",
            "normalization",
            "norm",
            "shuffle",
            "reformat",
            "slice",
            "concatenation",
            "concat",
            "reshape",
            "transpose",
            "copy",
        }
        for op in memory_bound_ops:
            if op in layer_lower:
                return "memory"

        # Fused operations are typically balanced (optimized)
        if "fused" in layer_lower or "+" in layer_type:
            return "balanced"

        return "unknown"

    def _get_layer_detail(self, inspector: Any, layer_idx: int, trt: Any) -> dict[str, Any] | None:
        """Get detailed info for a specific layer if available."""
        try:
            # TRT 10+ has get_layer_information
            if hasattr(inspector, "get_layer_information"):
                import json
                from typing import cast

                detail_json = inspector.get_layer_information(
                    layer_idx, trt.LayerInformationFormat.JSON
                )
                if detail_json:
                    result = json.loads(detail_json)
                    return cast(dict[str, Any], result)
        except Exception:
            pass
        return None

    def _extract_fused_ops(self, layer_name: str) -> list[str]:
        """Extract the list of ops that were fused from layer name."""
        # TRT uses '+' to indicate fused ops in layer names
        # e.g., "conv1 + bn1 + relu1" or "PWN(Conv_0 + Relu_1)"
        ops = []
        # Remove common TRT prefixes
        clean_name = layer_name
        for prefix in ["PWN(", "CudnnConvolution(", "Reformatter("]:
            if clean_name.startswith(prefix):
                clean_name = clean_name[len(prefix) :]
                if clean_name.endswith(")"):
                    clean_name = clean_name[:-1]

        # Split by '+'
        parts = [p.strip() for p in clean_name.split("+")]
        for part in parts:
            if part:
                ops.append(part)
        return ops

    def _infer_layer_type(self, name: str) -> str:
        """Infer layer type from layer name."""
        # Check for fused operations (indicated by '+')
        if "+" in name:
            parts = [p.strip() for p in name.split("+")]
            types = set()
            for part in parts:
                t = self._infer_single_layer_type(part)
                if t != "Unknown":
                    types.add(t)
            if types:
                return "Fused:" + "+".join(sorted(types))
            return "Fused"

        return self._infer_single_layer_type(name)

    def _infer_single_layer_type(self, name: str) -> str:
        """Infer type for a single (non-fused) layer name."""
        name_lower = name.lower()

        # Common patterns in layer names
        if "conv" in name_lower:
            return "Convolution"
        if "batchnorm" in name_lower or "bn" in name_lower:
            return "BatchNorm"
        if "relu" in name_lower:
            return "ReLU"
        if "pool" in name_lower:
            return "Pooling"
        if "dense" in name_lower or "fc" in name_lower or "linear" in name_lower:
            return "FullyConnected"
        if "softmax" in name_lower:
            return "Softmax"
        if "activation" in name_lower:
            return "Activation"
        if "add" in name_lower or "plus" in name_lower:
            return "ElementWise"
        if "concat" in name_lower:
            return "Concatenation"
        if "reshape" in name_lower:
            return "Reshape"
        if "copy" in name_lower or "reformat" in name_lower:
            return "Reformat"

        return "Unknown"

    def _infer_precision(self, name: str) -> str:
        """Infer precision from layer name (limited info available)."""
        # TRT doesn't expose per-layer precision easily in the inspector output
        # This would need profiling/timing data to determine
        return "Mixed"  # Assume mixed precision when FP16 is enabled


def parse_timing_cache(cache_path: str | Path) -> dict[str, float] | None:
    """
    Parse a TensorRT timing cache file for layer timing information.

    TensorRT timing caches store kernel selection decisions and can include
    timing data when built with profiling verbosity enabled.

    Args:
        cache_path: Path to the timing cache file (.cache or .timing)

    Returns:
        Dictionary mapping layer names to average time in milliseconds,
        or None if timing data is not available.

    Note:
        Timing caches are builder-side artifacts. For runtime profiling,
        use run_inference_profile() instead.
    """
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None

    try:
        import tensorrt as trt

        # Read timing cache
        with open(cache_path, "rb") as f:
            cache_data = f.read()

        # Create timing cache from serialized data
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)

        timing_cache = builder.create_timing_cache(cache_data)
        if timing_cache is None:
            return None

        # Note: TensorRT's timing cache doesn't expose per-layer timings directly
        # It stores kernel selection decisions for reproducible builds
        # For actual runtime timings, we need to profile during inference

        return None  # Timing cache doesn't contain runtime measurements

    except Exception:
        return None


def run_inference_profile(
    engine_path: str | Path,
    input_data: dict[str, Any] | None = None,
    num_iterations: int = 10,
    warmup_iterations: int = 3,
) -> dict[str, float]:
    """
    Run inference with profiling to get actual per-layer timing.

    This requires running actual inference with the TensorRT engine,
    which needs appropriate input data and GPU resources.

    Args:
        engine_path: Path to the TensorRT engine file.
        input_data: Dictionary of input name -> numpy array. If None, uses random data.
        num_iterations: Number of profiling iterations (default 10).
        warmup_iterations: Warmup iterations before timing (default 3).

    Returns:
        Dictionary mapping layer names to average time in milliseconds.

    Raises:
        ImportError: If TensorRT or CUDA is not available.
        RuntimeError: If profiling fails.
    """
    import numpy as np
    import tensorrt as trt

    engine_path = Path(engine_path)

    # Load engine
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

    if engine is None:
        raise RuntimeError(f"Failed to load engine: {engine_path}")

    # Create execution context with profiler
    context = engine.create_execution_context()

    class SimpleProfiler(trt.IProfiler):
        """Collect per-layer timing data."""

        def __init__(self) -> None:
            super().__init__()
            self.timings: dict[str, list[float]] = {}

        def report_layer_time(self, layer_name: str, time_ms: float) -> None:
            if layer_name not in self.timings:
                self.timings[layer_name] = []
            self.timings[layer_name].append(time_ms)

        def get_average_timings(self) -> dict[str, float]:
            return {name: sum(times) / len(times) for name, times in self.timings.items() if times}

    profiler = SimpleProfiler()
    context.profiler = profiler

    # Allocate buffers
    try:
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
    except ImportError as e:
        raise ImportError("pycuda required for profiling. Install with: pip install pycuda") from e

    # Get binding shapes and allocate
    bindings = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = context.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        size = int(np.prod(shape) * np.dtype(dtype).itemsize)

        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            if input_data and name in input_data:
                data = input_data[name].astype(dtype)
            else:
                data = np.random.randn(*shape).astype(dtype)
            mem = cuda.mem_alloc(data.nbytes)
            cuda.memcpy_htod(mem, data)
        else:
            mem = cuda.mem_alloc(size)

        bindings.append(int(mem))
        context.set_tensor_address(name, int(mem))

    # Warmup
    for _ in range(warmup_iterations):
        context.execute_async_v3(cuda.Stream().handle)

    # Profile
    cuda.Context.synchronize()
    for _ in range(num_iterations):
        context.execute_async_v3(cuda.Stream().handle)
        cuda.Context.synchronize()

    return profiler.get_average_timings()


def estimate_layer_times(
    engine_info: TRTEngineInfo,
    total_inference_time_ms: float | None = None,
) -> dict[str, float]:
    """
    Estimate per-layer times based on layer characteristics.

    This provides rough estimates when actual profiling is not available.
    Estimates are based on layer type and input/output shapes.

    Args:
        engine_info: Parsed TRT engine information.
        total_inference_time_ms: Total inference time to distribute. If None,
            returns relative weights (summing to 1.0).

    Returns:
        Dictionary mapping layer names to estimated time (ms or relative weight).
    """
    # Weight factors by layer type (relative compute intensity)
    TYPE_WEIGHTS = {
        "Convolution": 1.0,
        "MatrixMultiply": 1.0,
        "FullyConnected": 0.8,
        "Normalization": 0.3,
        "Activation": 0.1,
        "ElementWise": 0.1,
        "Pooling": 0.2,
        "SoftMax": 0.2,
        "Shuffle": 0.05,
        "Constant": 0.01,
        "Slice": 0.05,
        "Concatenation": 0.1,
    }

    # Calculate weights
    weights: dict[str, float] = {}
    for layer in engine_info.layers:
        base_weight = TYPE_WEIGHTS.get(layer.type, 0.5)

        # Adjust for precision
        if layer.precision == "INT8":
            base_weight *= 0.5  # INT8 is ~2x faster
        elif layer.precision == "FP16":
            base_weight *= 0.7  # FP16 is ~1.5x faster

        # Adjust for fusion (fused layers are more efficient)
        if layer.is_fused:
            base_weight *= 0.8

        weights[layer.name] = base_weight

    # Normalize
    total_weight = sum(weights.values()) or 1.0

    if total_inference_time_ms is not None:
        return {name: (w / total_weight) * total_inference_time_ms for name, w in weights.items()}
    else:
        return {name: w / total_weight for name, w in weights.items()}


def is_tensorrt_file(path: str | Path) -> bool:
    """
    Check if a file is a TensorRT engine.

    Args:
        path: Path to check.

    Returns:
        True if the file has a TensorRT engine extension.
    """
    path = Path(path)
    if not path.exists() or not path.is_file():
        return False

    # Check extension
    suffix = path.suffix.lower()
    if suffix in (".engine", ".plan"):
        return True

    # Could add magic byte checking here, but TRT engines don't have a standard magic

    return False


def is_available() -> bool:
    """Check if tensorrt is available."""
    try:
        import tensorrt  # noqa: F401

        return True
    except ImportError:
        return False


def format_bytes(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    size_float = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_float < 1024:
            return f"{size_float:.2f} {unit}"
        size_float /= 1024
    return f"{size_float:.2f} PB"


def generate_timing_chart(
    engine_info: TRTEngineInfo,
    output_path: str | Path | None = None,
    top_n: int = 20,
) -> bytes | None:
    """
    Generate a horizontal bar chart showing per-layer timing breakdown.

    Task 22.4.5: Layer timing breakdown chart (HTML/Streamlit).

    Args:
        engine_info: TRT engine information with layer timing.
        output_path: Optional path to save PNG. If None, returns bytes.
        top_n: Number of top layers to show (default 20).

    Returns:
        PNG bytes if output_path is None, otherwise None (saves to file).
        Returns None if no timing data available.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    # Extract layers with timing data
    layers_with_timing = [
        (lyr.name, lyr.avg_time_ms or 0.0, lyr.precision, lyr.bound_type)
        for lyr in engine_info.layers
        if lyr.avg_time_ms is not None and lyr.avg_time_ms > 0
    ]

    if not layers_with_timing:
        # Fall back to estimated times if no actual timing
        estimated = estimate_layer_times(engine_info, 100.0)  # Normalize to 100ms
        if not estimated:
            return None
        layers_with_timing = [
            (
                name,
                time_ms,
                next((lyr.precision for lyr in engine_info.layers if lyr.name == name), "FP32"),
                next((lyr.bound_type for lyr in engine_info.layers if lyr.name == name), "unknown"),
            )
            for name, time_ms in estimated.items()
        ]

    # Sort by time and take top N
    layers_with_timing.sort(key=lambda x: x[1], reverse=True)
    top_layers = layers_with_timing[:top_n]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(6, len(top_layers) * 0.35)))

    # Colors based on precision
    precision_colors = {
        "INT8": "#3fb950",  # Green
        "FP16": "#58a6ff",  # Blue
        "FP32": "#f0883e",  # Orange
        "Mixed": "#a371f7",  # Purple
        "Unknown": "#8b949e",  # Gray
    }

    # Extract data
    names = [t[0][:40] + ("..." if len(t[0]) > 40 else "") for t in top_layers]
    times = [t[1] for t in top_layers]
    colors = [precision_colors.get(t[2], "#8b949e") for t in top_layers]

    # Reverse for bottom-to-top display
    names = names[::-1]
    times = times[::-1]
    colors = colors[::-1]

    # Create horizontal bar chart
    y_pos = range(len(names))
    bars = ax.barh(y_pos, times, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)

    # Labels and styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Time (ms)", fontsize=10)
    ax.set_title(f"Layer Timing Breakdown (Top {len(top_layers)} Layers)", fontsize=12)

    # Add time labels on bars
    for bar, time_val in zip(bars, times, strict=True):
        width = bar.get_width()
        if width > max(times) * 0.1:  # Only show label if bar is wide enough
            ax.text(
                width * 0.95,
                bar.get_y() + bar.get_height() / 2,
                f"{time_val:.2f}",
                ha="right",
                va="center",
                fontsize=7,
                color="white",
                fontweight="bold",
            )

    # Legend for precision colors
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=c, label=p)
        for p, c in precision_colors.items()
        if any(t[2] == p for t in top_layers)
    ]
    if legend_elements:
        ax.legend(handles=legend_elements, loc="lower right", fontsize=8, title="Precision")

    # Style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()

    # Save or return bytes
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return None
    else:
        from io import BytesIO

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()


def generate_bound_type_chart(
    engine_info: TRTEngineInfo,
    output_path: str | Path | None = None,
) -> bytes | None:
    """
    Generate a pie/donut chart showing compute vs memory bound distribution.

    Task 22.4.4: Identify memory-bound vs compute-bound layers.

    Args:
        engine_info: TRT engine information.
        output_path: Optional path to save PNG.

    Returns:
        PNG bytes if output_path is None, otherwise None.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    perf = engine_info.performance

    # Data
    labels = []
    sizes = []
    colors = []

    if perf.compute_bound_layers > 0:
        labels.append(f"Compute-bound\n({perf.compute_bound_layers})")
        sizes.append(perf.compute_bound_layers)
        colors.append("#f0883e")  # Orange

    if perf.memory_bound_layers > 0:
        labels.append(f"Memory-bound\n({perf.memory_bound_layers})")
        sizes.append(perf.memory_bound_layers)
        colors.append("#58a6ff")  # Blue

    if perf.balanced_layers > 0:
        labels.append(f"Balanced\n({perf.balanced_layers})")
        sizes.append(perf.balanced_layers)
        colors.append("#3fb950")  # Green

    if perf.unknown_bound_layers > 0:
        labels.append(f"Unknown\n({perf.unknown_bound_layers})")
        sizes.append(perf.unknown_bound_layers)
        colors.append("#8b949e")  # Gray

    if not sizes:
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Donut chart
    wedges, texts, autotexts = ax.pie(  # type: ignore[misc]
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.75,
        wedgeprops={"width": 0.5, "edgecolor": "white", "linewidth": 2},
    )

    # Style autotexts
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight("bold")

    ax.set_title("Layer Classification\n(Compute vs Memory Bound)", fontsize=12)

    fig.tight_layout()

    # Save or return bytes
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return None
    else:
        from io import BytesIO

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
