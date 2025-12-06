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
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field


class TRTLayerInfo(BaseModel):
    """Information about a single layer in the TensorRT engine."""

    model_config = ConfigDict(frozen=True)

    name: str
    type: str
    precision: str = "FP32"
    input_shapes: list[tuple[int, ...]] = Field(default_factory=list)
    output_shapes: list[tuple[int, ...]] = Field(default_factory=list)
    # Tactic/kernel info if available
    tactic: str | None = None
    # Fusion info
    is_fused: bool = False
    fused_ops: list[str] = Field(default_factory=list)  # Original ops that were fused
    # Timing info (if profiling enabled)
    avg_time_ms: float | None = None
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
        return self.fused_layer_count / len(self.layers)

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
        return self.model_dump(mode="json")


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

        return TRTEngineInfo(
            path=self.path,
            trt_version=trt_version,
            builder_config=builder_config,
            device_name=device_name,
            compute_capability=compute_cap,
            layers=layers,
            bindings=bindings,
            device_memory_bytes=device_memory,
            has_implicit_batch=has_implicit_batch,
            max_batch_size=max_batch_size,
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
                            origin = None
                        else:
                            # Dict format (older TRT versions) with more details
                            name = layer_entry.get("Name", "Unknown")
                            layer_type = layer_entry.get("LayerType", self._infer_layer_type(name))
                            precision = layer_entry.get("Precision", "FP32")
                            tactic = layer_entry.get("TacticName") or layer_entry.get("Tactic")
                            origin = layer_entry.get("Origin")

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
                            if "Origin" in layer_detail:
                                origin = layer_detail["Origin"]

                        layers.append(
                            TRTLayerInfo(
                                name=name,
                                type=layer_type,
                                precision=precision,
                                tactic=tactic,
                                is_fused=is_fused,
                                fused_ops=fused_ops,
                                origin=origin,
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

    def _get_layer_detail(self, inspector: Any, layer_idx: int, trt: Any) -> dict[str, Any] | None:
        """Get detailed info for a specific layer if available."""
        try:
            # TRT 10+ has get_layer_information
            if hasattr(inspector, "get_layer_information"):
                import json

                detail_json = inspector.get_layer_information(
                    layer_idx, trt.LayerInformationFormat.JSON
                )
                if detail_json:
                    return json.loads(detail_json)
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
