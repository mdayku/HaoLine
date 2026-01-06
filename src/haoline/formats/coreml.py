# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
CoreML format reader.

CoreML is Apple's machine learning framework for iOS, macOS, watchOS,
and tvOS. This reader extracts model metadata and layer information.

Requires: coremltools (pip install coremltools)

Reference: https://coremltools.readme.io/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, computed_field

# CoreML layer type to FLOP formula mapping (Story 49.5.2)
# Note: Without full shape info, these are rough estimates
COREML_FLOP_FORMULAS: dict[str, str] = {
    # Convolutions
    "convolution": "conv",
    "convolution2d": "conv",
    "depthwiseConv2d": "depthwise_conv",
    "transposedConv2d": "conv",
    # Fully connected
    "innerProduct": "matmul",
    "linear": "matmul",
    # Pooling
    "pooling": "pooling",
    "globalPooling": "pooling",
    "maxPool2d": "pooling",
    "avgPool2d": "pooling",
    # Normalization
    "batchnorm": "batchnorm",
    "layerNorm": "norm",
    "instanceNorm": "norm",
    # Activations (elementwise)
    "activation": "elementwise",
    "relu": "elementwise",
    "leakyRelu": "elementwise",
    "sigmoid": "elementwise",
    "tanh": "elementwise",
    "softmax": "softmax",
    # Arithmetic
    "add": "elementwise",
    "multiply": "elementwise",
    "subtract": "elementwise",
    "divide": "elementwise",
    # Recurrent
    "uniDirectionalLSTM": "lstm",
    "biDirectionalLSTM": "lstm",
    "gru": "rnn",
    "simpleRecurrent": "rnn",
    # Reshape/data movement (no FLOPs)
    "reshape": "none",
    "flatten": "none",
    "permute": "none",
    "slice": "none",
    "concat": "none",
    "split": "none",
    "squeeze": "none",
    "expandDims": "none",
}


class CoreMLLayerInfo(BaseModel):
    """Information about a CoreML layer."""

    model_config = ConfigDict(frozen=True)

    name: str
    type: str
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)


class CoreMLInfo(BaseModel):
    """Parsed CoreML model information."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: Path
    spec_version: int
    description: str
    author: str
    license: str
    layers: list[CoreMLLayerInfo] = Field(default_factory=list)
    inputs: list[dict[str, Any]] = Field(default_factory=list)
    outputs: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def layer_count(self) -> int:
        """Number of layers."""
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
    def model_type(self) -> str:
        """Detected model type."""
        return str(self.metadata.get("model_type", "unknown"))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_flops(self) -> int:
        """
        Estimated total FLOPs for the model (Story 49.5.2).

        Note: CoreML doesn't expose intermediate tensor shapes, so this is
        a rough estimate based on layer types and input shapes.
        """
        total = 0
        # Try to get input shape for estimates
        input_elements = 1
        for inp in self.inputs:
            type_str = inp.get("type", "")
            # Parse shape from type string like "MultiArray(FLOAT32, [1, 3, 224, 224])"
            if "MultiArray" in type_str and "[" in type_str:
                try:
                    shape_str = type_str.split("[")[1].split("]")[0]
                    shape = [int(x.strip()) for x in shape_str.split(",") if x.strip()]
                    for dim in shape:
                        input_elements *= dim
                except (ValueError, IndexError):
                    pass

        for layer in self.layers:
            flops = _estimate_coreml_layer_flops(layer, input_elements)
            total += flops

        return total

    @computed_field  # type: ignore[prop-decorator]
    @property
    def flops_by_layer_type(self) -> dict[str, int]:
        """FLOPs breakdown by layer type."""
        breakdown: dict[str, int] = {}
        input_elements = 1
        for inp in self.inputs:
            type_str = inp.get("type", "")
            if "MultiArray" in type_str and "[" in type_str:
                try:
                    shape_str = type_str.split("[")[1].split("]")[0]
                    shape = [int(x.strip()) for x in shape_str.split(",") if x.strip()]
                    for dim in shape:
                        input_elements *= dim
                except (ValueError, IndexError):
                    pass

        for layer in self.layers:
            flops = _estimate_coreml_layer_flops(layer, input_elements)
            breakdown[layer.type] = breakdown.get(layer.type, 0) + flops

        return breakdown

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return cast(dict[str, Any], self.model_dump(mode="json"))


def _estimate_coreml_layer_flops(layer: CoreMLLayerInfo, input_elements: int) -> int:
    """
    Estimate FLOPs for a CoreML layer.

    Args:
        layer: The layer to estimate.
        input_elements: Number of elements in model input (for scaling).

    Returns:
        Estimated FLOPs for this layer.
    """
    formula = COREML_FLOP_FORMULAS.get(layer.type, "elementwise")

    if formula == "none":
        return 0

    # Without intermediate shapes, use input size as a rough baseline
    # and scale based on typical layer behavior
    baseline = input_elements

    if formula == "conv":
        # Conv typically has 9-27x FLOPs vs input size (3x3 or 5x5 kernel)
        return baseline * 9

    elif formula == "depthwise_conv":
        # Depthwise is cheaper than full conv
        return baseline * 3

    elif formula == "matmul":
        # FC layers typically reduce spatial dims, estimate as 2x input
        return baseline * 2

    elif formula == "lstm":
        # LSTM has 4 gates, roughly 8x a matmul
        return baseline * 16

    elif formula == "rnn":
        return baseline * 4

    elif formula == "softmax":
        return baseline * 5

    elif formula == "norm":
        return baseline * 5

    elif formula == "batchnorm":
        return baseline * 2

    elif formula == "pooling":
        # Pooling reduces size, comparisons per output
        return baseline // 4

    else:  # elementwise
        return baseline


class CoreMLReader:
    """Reader for CoreML format files (.mlmodel, .mlpackage)."""

    def __init__(self, path: str | Path):
        """
        Initialize reader with file path.

        Args:
            path: Path to the CoreML model (.mlmodel or .mlpackage).

        Raises:
            ImportError: If coremltools is not installed.
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"CoreML model not found: {self.path}")

        try:
            import coremltools  # noqa: F401
        except ImportError as e:
            raise ImportError("coremltools required. Install with: pip install coremltools") from e

    def read(self) -> CoreMLInfo:
        """
        Read and parse the CoreML model.

        Returns:
            CoreMLInfo with model metadata.
        """
        import coremltools as ct

        # Load model
        model = ct.models.MLModel(str(self.path))
        spec = model.get_spec()

        # Extract description
        desc = spec.description
        description = desc.metadata.shortDescription or ""
        author = desc.metadata.author or ""
        license_info = desc.metadata.license or ""

        # Extract inputs
        inputs = []
        for inp in desc.input:
            input_info = {"name": inp.name, "type": self._get_feature_type(inp.type)}
            inputs.append(input_info)

        # Extract outputs
        outputs = []
        for out in desc.output:
            output_info = {"name": out.name, "type": self._get_feature_type(out.type)}
            outputs.append(output_info)

        # Extract layers based on model type
        layers = []
        metadata = {}

        if spec.HasField("neuralNetwork"):
            nn = spec.neuralNetwork
            metadata["model_type"] = "neuralNetwork"
            for layer in nn.layers:
                layers.append(
                    CoreMLLayerInfo(
                        name=layer.name,
                        type=layer.WhichOneof("layer"),
                        inputs=list(layer.input),
                        outputs=list(layer.output),
                    )
                )
        elif spec.HasField("neuralNetworkClassifier"):
            nn = spec.neuralNetworkClassifier
            metadata["model_type"] = "neuralNetworkClassifier"
            for layer in nn.layers:
                layers.append(
                    CoreMLLayerInfo(
                        name=layer.name,
                        type=layer.WhichOneof("layer"),
                        inputs=list(layer.input),
                        outputs=list(layer.output),
                    )
                )
        elif spec.HasField("neuralNetworkRegressor"):
            nn = spec.neuralNetworkRegressor
            metadata["model_type"] = "neuralNetworkRegressor"
            for layer in nn.layers:
                layers.append(
                    CoreMLLayerInfo(
                        name=layer.name,
                        type=layer.WhichOneof("layer"),
                        inputs=list(layer.input),
                        outputs=list(layer.output),
                    )
                )
        elif spec.HasField("mlProgram"):
            metadata["model_type"] = "mlProgram"
            # ML Programs have a different structure
            # Basic info only for now
        elif spec.HasField("pipeline"):
            metadata["model_type"] = "pipeline"
        else:
            metadata["model_type"] = "other"

        return CoreMLInfo(
            path=self.path,
            spec_version=spec.specificationVersion,
            description=description,
            author=author,
            license=license_info,
            layers=layers,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
        )

    def _get_feature_type(self, feature_type) -> str:
        """Get human-readable feature type."""
        type_name = feature_type.WhichOneof("Type")
        if type_name == "multiArrayType":
            shape = list(feature_type.multiArrayType.shape)
            dtype = feature_type.multiArrayType.dataType
            dtype_names = {0: "INVALID", 65568: "FLOAT32", 65600: "FLOAT64", 131104: "INT32"}
            return f"MultiArray({dtype_names.get(dtype, dtype)}, {shape})"
        elif type_name == "imageType":
            w = feature_type.imageType.width
            h = feature_type.imageType.height
            return f"Image({w}x{h})"
        elif type_name == "dictionaryType":
            return "Dictionary"
        elif type_name == "stringType":
            return "String"
        elif type_name == "int64Type":
            return "Int64"
        elif type_name == "doubleType":
            return "Double"
        else:
            return type_name or "unknown"


def is_coreml_file(path: str | Path) -> bool:
    """
    Check if a file is a CoreML model.

    Args:
        path: Path to check.

    Returns:
        True if the file is a CoreML model.
    """
    path = Path(path)
    if not path.exists():
        return False

    # Check extension
    suffix = path.suffix.lower()
    if suffix == ".mlmodel":
        return True
    if suffix == ".mlpackage" or (path.is_dir() and path.suffix == ".mlpackage"):
        return True

    return False


def is_available() -> bool:
    """Check if coremltools is available."""
    try:
        import coremltools  # noqa: F401

        return True
    except ImportError:
        return False
