# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
Format Adapter system for Universal IR.

This module provides the plugin interface for model format readers/writers.
Each adapter converts format-specific models to/from UniversalGraph.

Usage:
    from haoline.format_adapters import get_adapter, list_adapters

    # Auto-detect and load
    adapter = get_adapter("model.onnx")
    graph = adapter.read("model.onnx")

    # Explicit adapter selection
    from haoline.format_adapters import OnnxAdapter
    graph = OnnxAdapter().read("model.onnx")
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import onnx
    import torch

from .universal_ir import (
    DataType,
    GraphMetadata,
    SourceFormat,
    TensorOrigin,
    UniversalGraph,
    UniversalNode,
    UniversalTensor,
)

logger = logging.getLogger(__name__)


@dataclass
class FormatCapabilities:
    """Capabilities and feature availability for a specific model format.

    Used to determine what analysis features are available for each format
    and provide appropriate user guidance.
    """

    # Core analysis capabilities
    has_graph: bool = True  # Does this format have a computational graph?
    has_flops: bool = True  # Can FLOPs be calculated?
    has_interactive_viz: bool = True  # Supports interactive graph visualization?

    # Metadata capabilities
    has_param_counts: bool = True  # Can parameter counts be extracted?
    has_memory_estimates: bool = True  # Can memory usage be estimated?
    has_quantization_info: bool = True  # Contains quantization metadata?

    # Hardware analysis
    supports_hardware_estimation: bool = True  # Can be used with hardware profiles?

    # Conversion capabilities
    can_convert_to_onnx: bool = False  # Can be converted to ONNX for full analysis?

    # UI/UX hints
    tier: str = "Full"  # Full/Graph/Metadata/Weights
    description: str = ""  # User-friendly description of limitations


# Predefined capabilities for each format
FORMAT_CAPABILITIES: dict[SourceFormat, FormatCapabilities] = {
    SourceFormat.ONNX: FormatCapabilities(
        has_graph=True,
        has_flops=True,
        has_interactive_viz=True,
        has_param_counts=True,
        has_memory_estimates=True,
        has_quantization_info=True,
        supports_hardware_estimation=True,
        can_convert_to_onnx=False,
        tier="Full",
        description="Complete analysis including graph visualization and FLOP estimation",
    ),
    SourceFormat.PYTORCH: FormatCapabilities(
        has_graph=True,
        has_flops=True,
        has_interactive_viz=True,
        has_param_counts=True,
        has_memory_estimates=True,
        has_quantization_info=True,
        supports_hardware_estimation=True,
        can_convert_to_onnx=True,
        tier="Full",
        description="Converted to ONNX for complete analysis",
    ),
    SourceFormat.TENSORFLOW: FormatCapabilities(
        has_graph=True,
        has_flops=True,
        has_interactive_viz=True,
        has_param_counts=True,
        has_memory_estimates=True,
        has_quantization_info=True,
        supports_hardware_estimation=True,
        can_convert_to_onnx=True,
        tier="Full",
        description="Converted to ONNX for complete analysis",
    ),
    SourceFormat.TFLITE: FormatCapabilities(
        has_graph=True,
        has_flops=False,  # FLOP formulas not yet implemented for TFLite ops
        has_interactive_viz=True,
        has_param_counts=True,
        has_memory_estimates=True,
        has_quantization_info=True,
        supports_hardware_estimation=True,
        can_convert_to_onnx=True,
        tier="Graph",
        description="Graph structure available, convert to ONNX for FLOP analysis",
    ),
    SourceFormat.COREML: FormatCapabilities(
        has_graph=True,
        has_flops=False,  # FLOP formulas not yet implemented for CoreML ops
        has_interactive_viz=True,
        has_param_counts=True,
        has_memory_estimates=False,  # Memory estimation may be incomplete
        has_quantization_info=False,  # Quantization info extraction unclear
        supports_hardware_estimation=True,
        can_convert_to_onnx=True,
        tier="Graph",
        description="Graph structure available, convert to ONNX for complete analysis",
    ),
    SourceFormat.OPENVINO: FormatCapabilities(
        has_graph=True,
        has_flops=False,  # FLOP formulas not yet implemented for OpenVINO ops
        has_interactive_viz=True,
        has_param_counts=True,
        has_memory_estimates=False,  # Memory estimation may be incomplete
        has_quantization_info=True,
        supports_hardware_estimation=True,
        can_convert_to_onnx=True,
        tier="Graph",
        description="Graph structure available, convert to ONNX for complete analysis",
    ),
    SourceFormat.TENSORRT: FormatCapabilities(
        has_graph=False,  # TRT engines are compiled/fused, no original graph
        has_flops=False,  # FLOP estimation not available for fused operations
        has_interactive_viz=False,  # No graph to visualize
        has_param_counts=False,  # Parameter counts not available in compiled engines
        has_memory_estimates=True,  # Memory usage can be estimated from engine
        has_quantization_info=True,  # Precision information available
        supports_hardware_estimation=True,
        can_convert_to_onnx=False,
        tier="Metadata",
        description="Compiled engine metadata, compare with source ONNX for fusion analysis",
    ),
    SourceFormat.GGUF: FormatCapabilities(
        has_graph=False,  # GGUF is weights-only, no computational graph
        has_flops=False,  # No operations to count
        has_interactive_viz=False,  # No graph to visualize
        has_param_counts=True,  # Parameter counts available from metadata
        has_memory_estimates=True,  # VRAM estimation available
        has_quantization_info=True,  # Quantization type information available
        supports_hardware_estimation=True,  # Hardware estimation works for LLM inference
        can_convert_to_onnx=False,  # No way to reconstruct full model from weights
        tier="Metadata",
        description="LLM architecture metadata and quantization info, no computational graph",
    ),
    SourceFormat.SAFETENSORS: FormatCapabilities(
        has_graph=False,  # SafeTensors is weights-only
        has_flops=False,  # No operations to count
        has_interactive_viz=False,  # No graph to visualize
        has_param_counts=True,  # Parameter counts available
        has_memory_estimates=True,  # Memory estimation available
        has_quantization_info=False,  # No quantization metadata
        supports_hardware_estimation=False,  # No architecture info for inference estimation
        can_convert_to_onnx=False,  # Needs external architecture (config.json)
        tier="Weights",
        description="Weights only, requires external architecture config for analysis",
    ),
}


def get_format_capabilities(format: SourceFormat) -> FormatCapabilities:
    """Get capabilities for a specific format."""
    return FORMAT_CAPABILITIES.get(format, FormatCapabilities())


# =============================================================================
# Format Adapter Protocol
# =============================================================================


class FormatAdapter(ABC):
    """Abstract base class for model format adapters.

    Implement this interface to add support for a new model format.
    Register the adapter using `register_adapter()`.

    Example:
        class MyFormatAdapter(FormatAdapter):
            name = "myformat"
            extensions = [".myf", ".myformat"]
            source_format = SourceFormat.UNKNOWN

            def can_read(self, path: Path) -> bool:
                return path.suffix.lower() in self.extensions

            def read(self, path: Path) -> UniversalGraph:
                # Parse format-specific file
                # Build and return UniversalGraph
                ...
    """

    # Adapter metadata (override in subclasses)
    name: str = "unknown"
    extensions: list[str] = []
    source_format: SourceFormat = SourceFormat.UNKNOWN

    @abstractmethod
    def can_read(self, path: Path) -> bool:
        """Check if this adapter can read the given file.

        Args:
            path: Path to the model file

        Returns:
            True if this adapter can read the file
        """
        pass

    @abstractmethod
    def read(self, path: Path) -> UniversalGraph:
        """Read a model file and convert to UniversalGraph.

        Args:
            path: Path to the model file

        Returns:
            UniversalGraph representation of the model

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        pass

    def can_write(self) -> bool:
        """Check if this adapter supports writing.

        Override this method if your adapter supports exporting
        UniversalGraph back to the format-specific file.

        Returns:
            True if write() is supported
        """
        return False

    def write(self, graph: UniversalGraph, path: Path) -> None:
        """Write UniversalGraph to a format-specific file.

        Args:
            graph: The graph to export
            path: Output file path

        Raises:
            NotImplementedError: If writing is not supported
        """
        raise NotImplementedError(f"{self.name} adapter does not support writing")


# =============================================================================
# Adapter Registry
# =============================================================================

# Global registry mapping extensions to adapters
_ADAPTER_REGISTRY: dict[str, type[FormatAdapter]] = {}


def register_adapter(adapter_class: type[FormatAdapter]) -> type[FormatAdapter]:
    """Register a format adapter.

    Can be used as a decorator:

        @register_adapter
        class MyFormatAdapter(FormatAdapter):
            ...

    Args:
        adapter_class: The adapter class to register

    Returns:
        The adapter class (for decorator use)
    """
    for ext in adapter_class.extensions:
        ext_lower = ext.lower()
        if ext_lower in _ADAPTER_REGISTRY:
            logger.warning(
                f"Overwriting adapter for {ext_lower}: "
                f"{_ADAPTER_REGISTRY[ext_lower].name} -> {adapter_class.name}"
            )
        _ADAPTER_REGISTRY[ext_lower] = adapter_class
    logger.debug(f"Registered adapter: {adapter_class.name} for {adapter_class.extensions}")
    return adapter_class


def get_adapter(path: str | Path) -> FormatAdapter:
    """Get an adapter for the given file.

    Auto-detects the format based on file extension.

    Args:
        path: Path to the model file

    Returns:
        An instance of the appropriate FormatAdapter

    Raises:
        ValueError: If no adapter is registered for the file extension
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext not in _ADAPTER_REGISTRY:
        available = ", ".join(sorted(_ADAPTER_REGISTRY.keys()))
        raise ValueError(f"No adapter registered for extension '{ext}'. Available: {available}")

    adapter_class = _ADAPTER_REGISTRY[ext]
    return adapter_class()


def list_adapters() -> list[dict[str, str | list[str] | bool]]:
    """List all registered adapters.

    Returns:
        List of adapter info dicts with keys: name, extensions, can_write
    """
    seen: set[str] = set()
    result: list[dict[str, str | list[str] | bool]] = []

    for adapter_class in _ADAPTER_REGISTRY.values():
        if adapter_class.name not in seen:
            seen.add(adapter_class.name)
            instance = adapter_class()
            result.append(
                {
                    "name": adapter_class.name,
                    "extensions": adapter_class.extensions,
                    "source_format": adapter_class.source_format.value,
                    "can_write": instance.can_write(),
                }
            )

    return sorted(result, key=lambda x: str(x["name"]))


# =============================================================================
# Op Type Mapping (ONNX -> Universal)
# =============================================================================

# Map ONNX op types to universal op types
ONNX_TO_UNIVERSAL_OP: dict[str, str] = {
    # Convolution
    "Conv": "Conv2D",
    "ConvTranspose": "ConvTranspose2D",
    # Linear/Dense
    "Gemm": "MatMul",
    "MatMul": "MatMul",
    "MatMulInteger": "MatMul",
    # Normalization
    "BatchNormalization": "BatchNorm",
    "LayerNormalization": "LayerNorm",
    "InstanceNormalization": "InstanceNorm",
    "GroupNormalization": "GroupNorm",
    # Activations
    "Relu": "Relu",
    "LeakyRelu": "LeakyRelu",
    "Sigmoid": "Sigmoid",
    "Tanh": "Tanh",
    "Softmax": "Softmax",
    "Gelu": "Gelu",
    "Silu": "Silu",
    "Mish": "Mish",
    # Pooling
    "MaxPool": "MaxPool2D",
    "AveragePool": "AvgPool2D",
    "GlobalAveragePool": "GlobalAvgPool",
    "GlobalMaxPool": "GlobalMaxPool",
    # Element-wise
    "Add": "Add",
    "Sub": "Sub",
    "Mul": "Mul",
    "Div": "Div",
    # Reshape/View
    "Reshape": "Reshape",
    "Flatten": "Flatten",
    "Squeeze": "Squeeze",
    "Unsqueeze": "Unsqueeze",
    "Transpose": "Transpose",
    # Attention (custom/subgraph)
    "Attention": "Attention",
    "MultiHeadAttention": "MultiHeadAttention",
    # Misc
    "Concat": "Concat",
    "Split": "Split",
    "Slice": "Slice",
    "Gather": "Gather",
    "Dropout": "Dropout",
    "Constant": "Constant",
    "Identity": "Identity",
    "Cast": "Cast",
    "ReduceMean": "ReduceMean",
    "ReduceSum": "ReduceSum",
    "Clip": "Clip",
    "Pad": "Pad",
    "Resize": "Resize",
    "Upsample": "Upsample",
}


def map_onnx_op_to_universal(onnx_op: str) -> str:
    """Map ONNX op type to universal op type.

    Args:
        onnx_op: ONNX operator name (e.g., "Conv", "Gemm")

    Returns:
        Universal op type (e.g., "Conv2D", "MatMul")
    """
    return ONNX_TO_UNIVERSAL_OP.get(onnx_op, onnx_op)


# =============================================================================
# ONNX Adapter
# =============================================================================


@register_adapter
class OnnxAdapter(FormatAdapter):
    """Adapter for ONNX models (.onnx files).

    This is the primary adapter since ONNX is HaoLine's native format.
    Supports both reading and writing.
    """

    name = "onnx"
    extensions = [".onnx"]
    source_format = SourceFormat.ONNX

    def can_read(self, path: Path) -> bool:
        """Check if file is an ONNX model."""
        return path.suffix.lower() == ".onnx"

    def read(self, path: Path) -> UniversalGraph:
        """Read ONNX model and convert to UniversalGraph."""
        import onnx
        from onnx import numpy_helper

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"ONNX model not found: {path}")

        # Load model
        model = onnx.load(str(path))

        # Run shape inference for better metadata
        try:
            model = onnx.shape_inference.infer_shapes(model)
        except Exception as e:
            logger.warning(f"Shape inference failed: {e}")

        graph = model.graph

        # Build metadata
        metadata = GraphMetadata(
            name=graph.name or path.stem,
            source_format=SourceFormat.ONNX,
            source_path=str(path),
            ir_version=model.ir_version,
            producer_name=model.producer_name or None,
            producer_version=model.producer_version or None,
            opset_version=model.opset_import[0].version if model.opset_import else None,
            input_names=[inp.name for inp in graph.input],
            output_names=[out.name for out in graph.output],
        )

        # Build tensors dict
        tensors: dict[str, UniversalTensor] = {}

        # Add initializers (weights)
        for init in graph.initializer:
            tensor_data = numpy_helper.to_array(init)
            tensors[init.name] = UniversalTensor(
                name=init.name,
                shape=list(tensor_data.shape),
                dtype=DataType.from_numpy_dtype(tensor_data.dtype),
                origin=TensorOrigin.WEIGHT,
                data=tensor_data,
                source_name=init.name,
            )

        # Add inputs (non-initializer)
        initializer_names = {init.name for init in graph.initializer}
        for inp in graph.input:
            if inp.name not in initializer_names:
                shape = self._extract_shape(inp)
                dtype = self._extract_dtype(inp)
                tensors[inp.name] = UniversalTensor(
                    name=inp.name,
                    shape=shape,
                    dtype=dtype,
                    origin=TensorOrigin.INPUT,
                    source_name=inp.name,
                )

        # Add outputs
        for out in graph.output:
            shape = self._extract_shape(out)
            dtype = self._extract_dtype(out)
            tensors[out.name] = UniversalTensor(
                name=out.name,
                shape=shape,
                dtype=dtype,
                origin=TensorOrigin.OUTPUT,
                source_name=out.name,
            )

        # Add value_info (intermediate tensors)
        for vi in graph.value_info:
            if vi.name not in tensors:
                shape = self._extract_shape(vi)
                dtype = self._extract_dtype(vi)
                tensors[vi.name] = UniversalTensor(
                    name=vi.name,
                    shape=shape,
                    dtype=dtype,
                    origin=TensorOrigin.ACTIVATION,
                    source_name=vi.name,
                )

        # Build nodes
        nodes: list[UniversalNode] = []
        for node in graph.node:
            # Extract output shapes from value_info or tensors
            output_shapes: list[list[int]] = []
            output_dtypes: list[DataType] = []
            for out_name in node.output:
                if out_name in tensors:
                    output_shapes.append(tensors[out_name].shape)
                    output_dtypes.append(tensors[out_name].dtype)
                else:
                    output_shapes.append([])
                    output_dtypes.append(DataType.UNKNOWN)

            # Extract attributes
            attrs = self._extract_attributes(node)

            nodes.append(
                UniversalNode(
                    id=node.name or f"{node.op_type}_{len(nodes)}",
                    op_type=map_onnx_op_to_universal(node.op_type),
                    inputs=list(node.input),
                    outputs=list(node.output),
                    attributes=attrs,
                    output_shapes=output_shapes,
                    output_dtypes=output_dtypes,
                    source_op=node.op_type,
                    source_domain=node.domain or "ai.onnx",
                )
            )

        return UniversalGraph(
            nodes=nodes,
            tensors=tensors,
            metadata=metadata,
        )

    def can_write(self) -> bool:
        """ONNX adapter supports writing."""
        return True

    def write(self, graph: UniversalGraph, path: Path) -> None:
        """Write UniversalGraph to ONNX format."""
        import onnx
        from onnx import helper, numpy_helper

        # Create initializers (weights)
        initializers = []
        for tensor in graph.tensors.values():
            if tensor.origin == TensorOrigin.WEIGHT and tensor.data is not None:
                init = numpy_helper.from_array(tensor.data, name=tensor.name)
                initializers.append(init)

        # Create inputs
        inputs = []
        for tensor in graph.tensors.values():
            if tensor.origin == TensorOrigin.INPUT:
                elem_type = self._dtype_to_onnx(tensor.dtype)
                shape = tensor.shape if tensor.shape else None
                inp = helper.make_tensor_value_info(tensor.name, elem_type, shape)
                inputs.append(inp)

        # Also add weight tensors as inputs (ONNX convention)
        for tensor in graph.tensors.values():
            if tensor.origin == TensorOrigin.WEIGHT:
                elem_type = self._dtype_to_onnx(tensor.dtype)
                inp = helper.make_tensor_value_info(tensor.name, elem_type, tensor.shape)
                inputs.append(inp)

        # Create outputs
        outputs = []
        for tensor in graph.tensors.values():
            if tensor.origin == TensorOrigin.OUTPUT:
                elem_type = self._dtype_to_onnx(tensor.dtype)
                shape = tensor.shape if tensor.shape else None
                out = helper.make_tensor_value_info(tensor.name, elem_type, shape)
                outputs.append(out)

        # Create nodes
        onnx_nodes = []
        for node in graph.nodes:
            # Map universal op back to ONNX op
            onnx_op = node.source_op or self._universal_to_onnx_op(node.op_type)

            onnx_node = helper.make_node(
                onnx_op,
                inputs=node.inputs,
                outputs=node.outputs,
                name=node.id,
                domain=node.source_domain or "",
                **node.attributes,
            )
            onnx_nodes.append(onnx_node)

        # Create graph
        onnx_graph = helper.make_graph(
            onnx_nodes,
            name=graph.metadata.name or "haoline_export",
            inputs=inputs,
            outputs=outputs,
            initializer=initializers,
        )

        # Create model
        opset_version = graph.metadata.opset_version or 17
        model = helper.make_model(
            onnx_graph,
            opset_imports=[helper.make_opsetid("", opset_version)],
            producer_name="haoline",
        )

        # Save
        onnx.save(model, str(path))

    def _extract_shape(self, value_info: onnx.ValueInfoProto) -> list[int]:
        """Extract shape from ONNX ValueInfoProto."""
        shape: list[int] = []
        try:
            tensor_type = value_info.type.tensor_type
            for dim in tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                else:
                    # Dynamic dimension (dim_param)
                    shape.append(-1)
        except Exception:
            pass
        return shape

    def _extract_dtype(self, value_info: onnx.ValueInfoProto) -> DataType:
        """Extract dtype from ONNX ValueInfoProto."""
        try:
            elem_type = value_info.type.tensor_type.elem_type
            return DataType.from_onnx_dtype(elem_type)
        except Exception:
            return DataType.UNKNOWN

    def _extract_attributes(self, node: onnx.NodeProto) -> dict[str, object]:
        """Extract attributes from ONNX NodeProto."""
        attrs: dict[str, object] = {}
        for attr in node.attribute:
            if attr.type == 1:  # FLOAT
                attrs[attr.name] = attr.f
            elif attr.type == 2:  # INT
                attrs[attr.name] = attr.i
            elif attr.type == 3:  # STRING
                attrs[attr.name] = attr.s.decode("utf-8") if attr.s else ""
            elif attr.type == 6:  # FLOATS
                attrs[attr.name] = list(attr.floats)
            elif attr.type == 7:  # INTS
                attrs[attr.name] = list(attr.ints)
            elif attr.type == 8:  # STRINGS
                attrs[attr.name] = [s.decode("utf-8") for s in attr.strings]
            # Skip TENSOR and GRAPH types for now
        return attrs

    def _dtype_to_onnx(self, dtype: DataType) -> int:
        """Convert DataType to ONNX TensorProto dtype."""
        from onnx import TensorProto

        mapping = {
            DataType.FLOAT32: TensorProto.FLOAT,
            DataType.FLOAT64: TensorProto.DOUBLE,
            DataType.FLOAT16: TensorProto.FLOAT16,
            DataType.BFLOAT16: TensorProto.BFLOAT16,
            DataType.INT64: TensorProto.INT64,
            DataType.INT32: TensorProto.INT32,
            DataType.INT16: TensorProto.INT16,
            DataType.INT8: TensorProto.INT8,
            DataType.UINT8: TensorProto.UINT8,
            DataType.BOOL: TensorProto.BOOL,
            DataType.STRING: TensorProto.STRING,
        }
        return mapping.get(dtype, TensorProto.FLOAT)

    def _universal_to_onnx_op(self, universal_op: str) -> str:
        """Map universal op type back to ONNX op."""
        # Reverse mapping
        reverse_map = {v: k for k, v in ONNX_TO_UNIVERSAL_OP.items()}
        return reverse_map.get(universal_op, universal_op)


# =============================================================================
# PyTorch Adapter
# =============================================================================


@register_adapter
class PyTorchAdapter(FormatAdapter):
    """Adapter for PyTorch models (.pt, .pth files).

    Converts PyTorch models to UniversalGraph by first exporting to ONNX,
    then using the OnnxAdapter. This ensures consistent representation.

    For full models (nn.Module), uses torch.onnx.export.
    For state_dicts, extracts weights without graph structure.
    """

    name = "pytorch"
    extensions = [".pt", ".pth"]
    source_format = SourceFormat.PYTORCH

    def can_read(self, path: Path) -> bool:
        """Check if file is a PyTorch model."""
        return path.suffix.lower() in [".pt", ".pth"]

    def read(self, path: Path) -> UniversalGraph:
        """Read PyTorch model and convert to UniversalGraph.

        Note: Requires sample input for tracing. Will attempt to
        auto-detect input shape from the model structure.
        """
        import torch

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PyTorch model not found: {path}")

        loaded = torch.load(str(path), map_location="cpu", weights_only=False)

        # Check if it's a full model or state_dict
        if isinstance(loaded, torch.nn.Module):
            return self._convert_module(loaded, path)
        elif isinstance(loaded, dict):
            # Could be state_dict or Ultralytics model
            if "model" in loaded:
                # Ultralytics YOLO model
                return self._convert_ultralytics(loaded, path)
            else:
                # Pure state_dict - weights only
                return self._convert_state_dict(loaded, path)
        else:
            raise ValueError(f"Unknown PyTorch file format: {type(loaded)}")

    def _convert_module(self, model: torch.nn.Module, path: Path) -> UniversalGraph:
        """Convert torch.nn.Module to UniversalGraph via ONNX."""
        import tempfile

        import torch

        model.eval()

        # Try to detect input shape
        dummy_input = self._create_dummy_input(model)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = Path(f.name)

        try:
            torch.onnx.export(
                model,
                (dummy_input,),
                str(onnx_path),
                opset_version=17,
                do_constant_folding=True,
            )

            # Use ONNX adapter to read
            graph = OnnxAdapter().read(onnx_path)

            # Update metadata to reflect PyTorch origin
            graph.metadata.source_format = SourceFormat.PYTORCH
            graph.metadata.source_path = str(path)

            return graph

        finally:
            if onnx_path.exists():
                onnx_path.unlink()

    def _convert_ultralytics(self, loaded: dict[str, object], path: Path) -> UniversalGraph:
        """Convert Ultralytics YOLO model to UniversalGraph."""
        import tempfile

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "Ultralytics YOLO model detected. Install ultralytics: pip install ultralytics"
            ) from e

        # Use Ultralytics export
        yolo = YOLO(str(path))

        with tempfile.TemporaryDirectory() as _tmpdir:
            onnx_path_str: str = yolo.export(format="onnx")
            onnx_path = Path(onnx_path_str)

            # Use ONNX adapter to read
            graph = OnnxAdapter().read(onnx_path)

            # Update metadata
            graph.metadata.source_format = SourceFormat.PYTORCH
            graph.metadata.source_path = str(path)
            graph.metadata.extra["ultralytics"] = True

            return graph

    def _convert_state_dict(self, state_dict: dict[str, object], path: Path) -> UniversalGraph:
        """Convert state_dict to UniversalGraph (weights only, no graph)."""
        import torch

        tensors: dict[str, UniversalTensor] = {}

        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                np_data = param.detach().cpu().numpy()
                tensors[name] = UniversalTensor(
                    name=name,
                    shape=list(np_data.shape),
                    dtype=DataType.from_numpy_dtype(np_data.dtype),
                    origin=TensorOrigin.WEIGHT,
                    data=np_data,
                    source_name=name,
                )

        return UniversalGraph(
            nodes=[],  # No graph structure for state_dict
            tensors=tensors,
            metadata=GraphMetadata(
                name=path.stem,
                source_format=SourceFormat.PYTORCH,
                source_path=str(path),
                extra={"type": "state_dict"},
            ),
        )

    def _create_dummy_input(self, model: torch.nn.Module) -> torch.Tensor:
        """Create dummy input for ONNX export.

        Attempts to auto-detect input shape from the model's first layer.
        """
        import torch

        # Try to find first conv or linear layer
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                # Assume image input
                in_channels = module.in_channels
                return torch.randn(1, in_channels, 224, 224)
            elif isinstance(module, torch.nn.Linear):
                in_features = module.in_features
                return torch.randn(1, in_features)

        # Default: batch of 224x224 RGB images
        return torch.randn(1, 3, 224, 224)


# =============================================================================
# Utility Functions
# =============================================================================


def load_model(path: str | Path) -> UniversalGraph:
    """Load a model file and convert to UniversalGraph.

    Auto-detects format based on file extension.

    Args:
        path: Path to the model file

    Returns:
        UniversalGraph representation

    Example:
        graph = load_model("model.onnx")
        graph = load_model("model.pt")
    """
    adapter = get_adapter(path)
    return adapter.read(Path(path))


def save_model(graph: UniversalGraph, path: str | Path) -> None:
    """Save UniversalGraph to a model file.

    Format is determined by file extension.

    Args:
        graph: The graph to save
        path: Output file path

    Raises:
        ValueError: If adapter doesn't support writing
    """
    path = Path(path)
    adapter = get_adapter(path)
    if not adapter.can_write():
        raise ValueError(f"{adapter.name} adapter does not support writing")
    adapter.write(graph, path)


# =============================================================================
# Conversion Matrix (Task 18.3)
# =============================================================================


class ConversionLevel(str, Enum):
    """Conversion capability between formats.

    Describes how well a conversion preserves information:
    - FULL: Lossless conversion, all info preserved
    - PARTIAL: Some limitations or requires multi-step
    - LOSSY: Some information is lost
    - NONE: No conversion path available
    """

    FULL = "full"  # Lossless, complete conversion
    PARTIAL = "partial"  # Some limitations or multi-step required
    LOSSY = "lossy"  # Information loss during conversion
    NONE = "none"  # No conversion path


# Conversion matrix: (source, target) -> ConversionLevel
# Format: CONVERSION_MATRIX[(source_format, target_format)] = level
_CONVERSION_MATRIX: dict[tuple[SourceFormat, SourceFormat], ConversionLevel] = {
    # ONNX conversions (primary interchange format)
    (SourceFormat.ONNX, SourceFormat.TENSORRT): ConversionLevel.PARTIAL,  # TensorRT-specific ops
    (SourceFormat.ONNX, SourceFormat.TFLITE): ConversionLevel.PARTIAL,  # Some ops unsupported
    (SourceFormat.ONNX, SourceFormat.COREML): ConversionLevel.PARTIAL,  # iOS-specific limits
    (SourceFormat.ONNX, SourceFormat.OPENVINO): ConversionLevel.FULL,  # Good ONNX support
    # PyTorch conversions (via ONNX)
    (SourceFormat.PYTORCH, SourceFormat.ONNX): ConversionLevel.FULL,  # torch.onnx.export
    (SourceFormat.PYTORCH, SourceFormat.TENSORRT): ConversionLevel.PARTIAL,  # Via ONNX
    (SourceFormat.PYTORCH, SourceFormat.TFLITE): ConversionLevel.PARTIAL,  # Via ONNX
    (SourceFormat.PYTORCH, SourceFormat.COREML): ConversionLevel.PARTIAL,  # coremltools
    # TensorFlow conversions
    (SourceFormat.TENSORFLOW, SourceFormat.ONNX): ConversionLevel.PARTIAL,  # tf2onnx
    (SourceFormat.TENSORFLOW, SourceFormat.TFLITE): ConversionLevel.FULL,  # TFLite converter
    (SourceFormat.TENSORFLOW, SourceFormat.COREML): ConversionLevel.PARTIAL,  # coremltools
    # TensorRT (inference-only, limited export)
    (SourceFormat.TENSORRT, SourceFormat.ONNX): ConversionLevel.NONE,  # Cannot export
    # CoreML (Apple ecosystem)
    (SourceFormat.COREML, SourceFormat.ONNX): ConversionLevel.LOSSY,  # Some info lost
    # TFLite (mobile)
    (SourceFormat.TFLITE, SourceFormat.ONNX): ConversionLevel.PARTIAL,  # tflite2onnx
    # Weights-only formats (no graph structure)
    (SourceFormat.SAFETENSORS, SourceFormat.ONNX): ConversionLevel.NONE,  # Weights only
    (SourceFormat.GGUF, SourceFormat.ONNX): ConversionLevel.NONE,  # Weights only
}


def get_conversion_level(source: SourceFormat | str, target: SourceFormat | str) -> ConversionLevel:
    """Get the conversion capability between two formats.

    Args:
        source: Source model format
        target: Target model format

    Returns:
        ConversionLevel indicating conversion capability
    """
    # Normalize to SourceFormat
    if isinstance(source, str):
        try:
            source = SourceFormat(source.lower())
        except ValueError:
            return ConversionLevel.NONE
    if isinstance(target, str):
        try:
            target = SourceFormat(target.lower())
        except ValueError:
            return ConversionLevel.NONE

    # Identity conversion
    if source == target:
        return ConversionLevel.FULL

    return _CONVERSION_MATRIX.get((source, target), ConversionLevel.NONE)


def list_conversion_paths(
    source: SourceFormat | str | None = None,
    target: SourceFormat | str | None = None,
) -> list[dict[str, str]]:
    """List available conversion paths.

    Args:
        source: Filter by source format (optional)
        target: Filter by target format (optional)

    Returns:
        List of dicts with source, target, and level
    """
    result: list[dict[str, str]] = []

    for (src, tgt), level in _CONVERSION_MATRIX.items():
        # Apply filters
        if source is not None:
            source_fmt = (
                source if isinstance(source, SourceFormat) else SourceFormat(source.lower())
            )
            if src != source_fmt:
                continue
        if target is not None:
            target_fmt = (
                target if isinstance(target, SourceFormat) else SourceFormat(target.lower())
            )
            if tgt != target_fmt:
                continue

        result.append(
            {
                "source": src.value,
                "target": tgt.value,
                "level": level.value,
            }
        )

    return sorted(result, key=lambda x: (x["source"], x["target"]))


def can_convert(source: SourceFormat | str, target: SourceFormat | str) -> bool:
    """Check if conversion is possible between two formats.

    Returns True for FULL, PARTIAL, or LOSSY conversions.

    Args:
        source: Source model format
        target: Target model format

    Returns:
        True if any conversion path exists
    """
    level = get_conversion_level(source, target)
    return level != ConversionLevel.NONE


def convert_model(
    graph: UniversalGraph,
    target_format: SourceFormat | str,
    output_path: Path | str,
) -> Path:
    """Convert a model to a different format.

    Args:
        graph: UniversalGraph to convert
        target_format: Target format (e.g., "onnx", "tflite")
        output_path: Output file path

    Returns:
        Path to the converted model

    Raises:
        ValueError: If conversion is not supported
    """
    output_path = Path(output_path)

    # Get conversion level
    source = graph.metadata.source_format
    if isinstance(target_format, str):
        target_format = SourceFormat(target_format.lower())

    level = get_conversion_level(source, target_format)
    if level == ConversionLevel.NONE:
        raise ValueError(
            f"Cannot convert from {source.value} to {target_format.value}. "
            f"No conversion path available."
        )

    # Log warning for lossy conversions
    if level == ConversionLevel.LOSSY:
        logger.warning(f"Converting {source.value} to {target_format.value} may lose information")

    # Get target adapter
    # For now, only ONNX writing is supported
    if target_format != SourceFormat.ONNX:
        raise NotImplementedError(
            f"Direct conversion to {target_format.value} not yet implemented. "
            f"Export to ONNX first, then use format-specific tools."
        )

    adapter = OnnxAdapter()
    adapter.write(graph, output_path)

    return output_path


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Protocol
    "FormatAdapter",
    # Registry
    "register_adapter",
    "get_adapter",
    "list_adapters",
    # Adapters
    "OnnxAdapter",
    "PyTorchAdapter",
    # Utilities
    "load_model",
    "save_model",
    "map_onnx_op_to_universal",
    "ONNX_TO_UNIVERSAL_OP",
    # Conversion Matrix
    "ConversionLevel",
    "get_conversion_level",
    "list_conversion_paths",
    "can_convert",
    "convert_model",
]
