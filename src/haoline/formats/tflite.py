# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
TFLite format reader.

TensorFlow Lite models use FlatBuffer format. This reader extracts
basic metadata without requiring TensorFlow dependencies.

For full analysis, use tflite-runtime or convert to ONNX first.

Reference: https://www.tensorflow.org/lite/guide
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, computed_field

# TFLite FlatBuffer identifier
TFLITE_IDENTIFIER = b"TFL3"

# TFLite tensor types (by ID for pure Python parsing)
TFLITE_TYPES: dict[int, tuple[str, int]] = {
    0: ("FLOAT32", 4),
    1: ("FLOAT16", 2),
    2: ("INT32", 4),
    3: ("UINT8", 1),
    4: ("INT64", 8),
    5: ("STRING", 0),  # Variable
    6: ("BOOL", 1),
    7: ("INT16", 2),
    8: ("COMPLEX64", 8),
    9: ("INT8", 1),
    10: ("FLOAT64", 8),
    11: ("COMPLEX128", 16),
    12: ("UINT64", 8),
    13: ("RESOURCE", 0),
    14: ("VARIANT", 0),
    15: ("UINT32", 4),
    16: ("UINT16", 2),
    17: ("INT4", 0),  # Packed
}

# Bytes per element for dtype strings (from numpy dtype names)
DTYPE_BYTES: dict[str, int] = {
    "float32": 4,
    "float16": 2,
    "int32": 4,
    "uint8": 1,
    "int64": 8,
    "bool": 1,
    "int16": 2,
    "complex64": 8,
    "int8": 1,
    "float64": 8,
    "complex128": 16,
    "uint64": 8,
    "uint16": 2,
    "uint32": 4,
}

# TFLite builtin operators
TFLITE_BUILTINS: dict[int, str] = {
    0: "ADD",
    1: "AVERAGE_POOL_2D",
    2: "CONCATENATION",
    3: "CONV_2D",
    4: "DEPTHWISE_CONV_2D",
    5: "DEPTH_TO_SPACE",
    6: "DEQUANTIZE",
    7: "EMBEDDING_LOOKUP",
    8: "FLOOR",
    9: "FULLY_CONNECTED",
    10: "HASHTABLE_LOOKUP",
    11: "L2_NORMALIZATION",
    12: "L2_POOL_2D",
    13: "LOCAL_RESPONSE_NORMALIZATION",
    14: "LOGISTIC",
    15: "LSH_PROJECTION",
    16: "LSTM",
    17: "MAX_POOL_2D",
    18: "MUL",
    19: "RELU",
    20: "RELU_N1_TO_1",
    21: "RELU6",
    22: "RESHAPE",
    23: "RESIZE_BILINEAR",
    24: "RNN",
    25: "SOFTMAX",
    26: "SPACE_TO_DEPTH",
    27: "SVDF",
    28: "TANH",
    29: "CONCAT_EMBEDDINGS",
    30: "SKIP_GRAM",
    31: "CALL",
    32: "CUSTOM",
    33: "EMBEDDING_LOOKUP_SPARSE",
    34: "PAD",
    35: "UNIDIRECTIONAL_SEQUENCE_RNN",
    36: "GATHER",
    37: "BATCH_TO_SPACE_ND",
    38: "SPACE_TO_BATCH_ND",
    39: "TRANSPOSE",
    40: "MEAN",
    41: "SUB",
    42: "DIV",
    43: "SQUEEZE",
    44: "UNIDIRECTIONAL_SEQUENCE_LSTM",
    45: "STRIDED_SLICE",
    46: "BIDIRECTIONAL_SEQUENCE_RNN",
    47: "EXP",
    48: "TOPK_V2",
    49: "SPLIT",
    50: "LOG_SOFTMAX",
    # ... many more, but these are the common ones
}

# TFLite op to FLOP formula type mapping (Story 49.5.1)
# Formula types: conv, depthwise_conv, matmul, elementwise, softmax, pooling, none
TFLITE_FLOP_FORMULAS: dict[str, str] = {
    # Convolutions (highest FLOPs)
    "CONV_2D": "conv",
    "DEPTHWISE_CONV_2D": "depthwise_conv",
    "TRANSPOSE_CONV": "conv",
    # Fully connected / matrix multiply
    "FULLY_CONNECTED": "matmul",
    # Pooling (comparisons, not multiply-adds)
    "AVERAGE_POOL_2D": "pooling",
    "MAX_POOL_2D": "pooling",
    "L2_POOL_2D": "pooling",
    # Normalization
    "L2_NORMALIZATION": "norm",
    "LOCAL_RESPONSE_NORMALIZATION": "norm",
    "BATCH_NORMALIZATION": "batchnorm",
    # Activations (elementwise)
    "RELU": "elementwise",
    "RELU6": "elementwise",
    "RELU_N1_TO_1": "elementwise",
    "LOGISTIC": "elementwise",  # sigmoid
    "TANH": "elementwise",
    "SOFTMAX": "softmax",
    "LOG_SOFTMAX": "softmax",
    # Arithmetic (elementwise)
    "ADD": "elementwise",
    "SUB": "elementwise",
    "MUL": "elementwise",
    "DIV": "elementwise",
    "EXP": "elementwise",
    "MEAN": "reduce",
    # Recurrent
    "LSTM": "lstm",
    "RNN": "rnn",
    "UNIDIRECTIONAL_SEQUENCE_LSTM": "lstm",
    "UNIDIRECTIONAL_SEQUENCE_RNN": "rnn",
    "BIDIRECTIONAL_SEQUENCE_RNN": "rnn",
    # No FLOPs (reshape, data movement)
    "RESHAPE": "none",
    "SQUEEZE": "none",
    "TRANSPOSE": "none",
    "CONCATENATION": "none",
    "SPLIT": "none",
    "GATHER": "none",
    "PAD": "none",
    "STRIDED_SLICE": "none",
    "DEQUANTIZE": "none",
    "QUANTIZE": "none",
    "EMBEDDING_LOOKUP": "none",
}


def _estimate_tflite_op_flops(
    op: TFLiteOperatorInfo,
    tensors: list[TFLiteTensorInfo],
) -> int:
    """
    Estimate FLOPs for a single TFLite operator.

    Args:
        op: The operator to estimate.
        tensors: List of all tensors in the model (for shape lookup).

    Returns:
        Estimated FLOPs for this operator.
    """
    formula = TFLITE_FLOP_FORMULAS.get(op.op_name, "elementwise")

    if formula == "none":
        return 0

    # Get output shape for elementwise estimates
    output_elements = 1
    if op.outputs:
        out_idx = op.outputs[0]
        if 0 <= out_idx < len(tensors):
            output_elements = tensors[out_idx].n_elements

    if formula == "conv":
        # Conv2D: 2 * K_h * K_w * C_in * C_out * H_out * W_out
        # Need weight tensor shape: [C_out, K_h, K_w, C_in] (TFLite format)
        if len(op.inputs) >= 2:
            weight_idx = op.inputs[1]
            if 0 <= weight_idx < len(tensors):
                w = tensors[weight_idx]
                if len(w.shape) == 4:
                    # TFLite: [out_channels, kernel_h, kernel_w, in_channels]
                    c_out, k_h, k_w, c_in = w.shape
                    # Output spatial size from output tensor
                    if op.outputs and 0 <= op.outputs[0] < len(tensors):
                        out = tensors[op.outputs[0]]
                        if len(out.shape) == 4:
                            _, h_out, w_out, _ = out.shape
                            return 2 * k_h * k_w * c_in * c_out * h_out * w_out
        return output_elements * 9  # Fallback: assume 3x3 kernel

    elif formula == "depthwise_conv":
        # DepthwiseConv2D: 2 * K_h * K_w * C * H_out * W_out (no C_in * C_out, just C)
        if len(op.inputs) >= 2:
            weight_idx = op.inputs[1]
            if 0 <= weight_idx < len(tensors):
                w = tensors[weight_idx]
                if len(w.shape) == 4:
                    # TFLite depthwise: [1, kernel_h, kernel_w, channels * multiplier]
                    _, k_h, k_w, c = w.shape
                    if op.outputs and 0 <= op.outputs[0] < len(tensors):
                        out = tensors[op.outputs[0]]
                        if len(out.shape) == 4:
                            _, h_out, w_out, _ = out.shape
                            return 2 * k_h * k_w * c * h_out * w_out
        return output_elements * 9  # Fallback

    elif formula == "matmul":
        # FullyConnected: 2 * M * N * K (M=batch, K=input_features, N=output_features)
        if len(op.inputs) >= 2:
            weight_idx = op.inputs[1]
            if 0 <= weight_idx < len(tensors):
                w = tensors[weight_idx]
                if len(w.shape) >= 2:
                    # TFLite FC weight: [output_features, input_features]
                    n, k = w.shape[-2], w.shape[-1]
                    # Batch size from input
                    if op.inputs and 0 <= op.inputs[0] < len(tensors):
                        inp = tensors[op.inputs[0]]
                        m = inp.n_elements // k if k > 0 else 1
                        return 2 * m * n * k
        return output_elements * 2  # Fallback

    elif formula == "lstm":
        # LSTM: 4 gates × (2 × hidden × input + 2 × hidden × hidden) per timestep
        # Simplified: 8 * hidden^2 * seq_len
        if op.outputs and 0 <= op.outputs[0] < len(tensors):
            out = tensors[op.outputs[0]]
            if len(out.shape) >= 2:
                # Assume [batch, seq_len, hidden] or similar
                hidden = out.shape[-1] if out.shape else 256
                seq_len = out.shape[-2] if len(out.shape) >= 2 else 1
                return 8 * hidden * hidden * seq_len
        return output_elements * 8

    elif formula == "rnn":
        # RNN: 2 * hidden * (input + hidden) per timestep
        if op.outputs and 0 <= op.outputs[0] < len(tensors):
            out = tensors[op.outputs[0]]
            if len(out.shape) >= 2:
                hidden = out.shape[-1] if out.shape else 256
                seq_len = out.shape[-2] if len(out.shape) >= 2 else 1
                return 4 * hidden * hidden * seq_len
        return output_elements * 4

    elif formula == "softmax":
        # Softmax: ~5 ops per element (exp, sum, div, etc.)
        return output_elements * 5

    elif formula == "norm":
        # Normalization: ~5 ops per element
        return output_elements * 5

    elif formula == "batchnorm":
        # BatchNorm: ~2 ops per element (scale + shift)
        return output_elements * 2

    elif formula == "pooling":
        # Pooling: comparisons, roughly 1 op per output element per kernel element
        # Assume 2x2 or 3x3 kernel
        return output_elements * 4

    elif formula == "reduce":
        # Mean/Sum: 1 op per input element
        if op.inputs and 0 <= op.inputs[0] < len(tensors):
            return tensors[op.inputs[0]].n_elements
        return output_elements

    else:  # elementwise
        return output_elements


class TFLiteTensorInfo(BaseModel):
    """Information about a TFLite tensor."""

    model_config = ConfigDict(frozen=True)

    name: str
    shape: tuple[int, ...]
    dtype: str  # e.g., "float32", "int8", "uint8"
    buffer_idx: int
    quantization: dict[str, Any] | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def bytes_per_element(self) -> int:
        """Bytes per element."""
        return int(DTYPE_BYTES.get(self.dtype.lower(), 4))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def n_elements(self) -> int:
        """Total number of elements."""
        result = 1
        for d in self.shape:
            result *= d
        return result

    @computed_field  # type: ignore[prop-decorator]
    @property
    def size_bytes(self) -> int:
        """Estimated size in bytes."""
        bpe = self.bytes_per_element
        if bpe == 0:
            return 0  # Variable size types
        return int(self.n_elements * bpe)


class TFLiteOperatorInfo(BaseModel):
    """Information about a TFLite operator."""

    model_config = ConfigDict(frozen=True)

    opcode_index: int
    builtin_code: int
    inputs: list[int]
    outputs: list[int]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def op_name(self) -> str:
        """Human-readable operator name."""
        return TFLITE_BUILTINS.get(self.builtin_code, f"CUSTOM_{self.builtin_code}")


class TFLiteInfo(BaseModel):
    """Parsed TFLite file information."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: Path
    version: int
    description: str
    tensors: list[TFLiteTensorInfo] = Field(default_factory=list)
    operators: list[TFLiteOperatorInfo] = Field(default_factory=list)
    inputs: list[int] = Field(default_factory=list)
    outputs: list[int] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_params(self) -> int:
        """Total parameter count (non-input tensors)."""
        input_set = set(self.inputs)
        return sum(t.n_elements for i, t in enumerate(self.tensors) if i not in input_set)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_size_bytes(self) -> int:
        """Total model size in bytes."""
        return sum(t.size_bytes for t in self.tensors)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def op_counts(self) -> dict[str, int]:
        """Count of operators by type."""
        counts: dict[str, int] = {}
        for op in self.operators:
            name = op.op_name
            counts[name] = counts.get(name, 0) + 1
        return counts

    @computed_field  # type: ignore[prop-decorator]
    @property
    def type_breakdown(self) -> dict[str, int]:
        """Count of tensors by type."""
        breakdown: dict[str, int] = {}
        for t in self.tensors:
            dtype = t.dtype
            breakdown[dtype] = breakdown.get(dtype, 0) + 1
        return breakdown

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_quantized(self) -> bool:
        """Check if model uses quantized types."""
        quant_types = {"int8", "uint8", "int16", "int4"}
        return any(t.dtype.lower() in quant_types for t in self.tensors)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_flops(self) -> int:
        """Total estimated FLOPs for the model (Story 49.5.1)."""
        return sum(_estimate_tflite_op_flops(op, self.tensors) for op in self.operators)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def flops_by_op(self) -> dict[str, int]:
        """FLOPs breakdown by operator type."""
        breakdown: dict[str, int] = {}
        for op in self.operators:
            flops = _estimate_tflite_op_flops(op, self.tensors)
            breakdown[op.op_name] = breakdown.get(op.op_name, 0) + flops
        return breakdown

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return cast(dict[str, Any], self.model_dump(mode="json"))


class TFLiteReader:
    """
    Reader for TFLite format files.

    This provides basic metadata extraction using pure Python.
    For full tensor access, use tflite-runtime.
    """

    def __init__(self, path: str | Path):
        """
        Initialize reader with file path.

        Args:
            path: Path to the TFLite file.
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"TFLite file not found: {self.path}")

    def read(self) -> TFLiteInfo:
        """
        Read and parse the TFLite file.

        This uses tflite-runtime if available, otherwise falls back
        to basic FlatBuffer parsing.

        Returns:
            TFLiteInfo with parsed metadata.
        """
        # Try tflite-runtime first
        try:
            return self._read_with_interpreter()
        except ImportError:
            pass

        # Fall back to basic parsing
        return self._read_basic()

    def _read_with_interpreter(self) -> TFLiteInfo:
        """Read using TFLite Interpreter."""
        from tflite_runtime.interpreter import Interpreter

        interpreter = Interpreter(model_path=str(self.path))
        interpreter.allocate_tensors()

        # Get tensor details
        tensor_details = interpreter.get_tensor_details()
        tensors = []
        for td in tensor_details:
            quant = None
            if "quantization" in td:
                quant = {
                    "scale": td["quantization"][0],
                    "zero_point": td["quantization"][1],
                }
            # Convert numpy dtype to string name
            dtype_val = td["dtype"]
            if hasattr(dtype_val, "__name__"):
                dtype_str = dtype_val.__name__  # numpy.float32 -> "float32"
            elif hasattr(dtype_val, "name"):
                dtype_str = dtype_val.name
            else:
                dtype_str = str(dtype_val)

            tensors.append(
                TFLiteTensorInfo(
                    name=td["name"],
                    shape=tuple(td["shape"]),
                    dtype=dtype_str,
                    buffer_idx=td.get("buffer_idx", 0),
                    quantization=quant,
                )
            )

        # Get input/output indices
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        inputs = [d["index"] for d in input_details]
        outputs = [d["index"] for d in output_details]

        return TFLiteInfo(
            path=self.path,
            version=3,  # TFL3
            description="",
            tensors=tensors,
            operators=[],  # Not easily accessible via interpreter
            inputs=inputs,
            outputs=outputs,
        )

    def _read_basic(self) -> TFLiteInfo:
        """Basic FlatBuffer parsing without tflite-runtime."""
        with open(self.path, "rb") as f:
            data = f.read()

        # Verify file identifier (at offset 4-8)
        if len(data) < 8:
            raise ValueError("File too small to be a valid TFLite model")

        identifier = data[4:8]
        if identifier != TFLITE_IDENTIFIER:
            raise ValueError(
                f"Not a TFLite file: expected {TFLITE_IDENTIFIER!r}, got {identifier!r}"
            )

        # FlatBuffer root table offset
        root_offset = struct.unpack("<I", data[0:4])[0]

        # This is a simplified parser - full FlatBuffer parsing is complex
        # We extract what we can from the structure

        return TFLiteInfo(
            path=self.path,
            version=3,
            description="TFLite model (basic parsing - install tflite-runtime for full details)",
            tensors=[],
            operators=[],
            inputs=[],
            outputs=[],
            metadata={"file_size": len(data), "root_offset": root_offset},
        )


def is_tflite_file(path: str | Path) -> bool:
    """
    Check if a file is a valid TFLite file.

    Args:
        path: Path to check.

    Returns:
        True if the file is a TFLite model.
    """
    path = Path(path)
    if not path.exists() or not path.is_file():
        return False

    try:
        with open(path, "rb") as f:
            # Read enough for identifier check
            data = f.read(8)
            if len(data) < 8:
                return False
            identifier = data[4:8]
            return identifier == TFLITE_IDENTIFIER
    except Exception:
        return False


def is_available() -> bool:
    """Check if tflite-runtime is available."""
    try:
        from tflite_runtime.interpreter import Interpreter  # noqa: F401

        return True
    except ImportError:
        return False
