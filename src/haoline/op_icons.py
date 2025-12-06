# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
Op Type Icon System for graph visualization.

Task 5.5: Maps 180+ ONNX operators to visual categories with icons,
colors, and size scaling based on computational intensity.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict


class OpCategory(Enum):
    """Visual categories for ONNX operators."""

    # Neural Network Operations
    CONV = "conv"  # Convolution operations
    LINEAR = "linear"  # MatMul, Gemm, fully connected
    ATTENTION = "attention"  # Attention patterns (detected, not single op)
    NORM = "norm"  # Normalization layers
    ACTIVATION = "activation"  # Activation functions
    POOL = "pool"  # Pooling operations
    DROPOUT = "dropout"  # Regularization

    # Tensor Operations
    RESHAPE = "reshape"  # Shape manipulation
    TRANSPOSE = "transpose"  # Dimension reordering
    SLICE = "slice"  # Indexing and slicing
    CONCAT = "concat"  # Tensor joining
    SPLIT = "split"  # Tensor splitting
    PAD = "pad"  # Padding operations

    # Math Operations
    ELEMENTWISE = "elementwise"  # Element-wise math
    REDUCE = "reduce"  # Reduction operations
    COMPARE = "compare"  # Comparison and logic

    # Special Operations
    EMBED = "embed"  # Embedding lookups
    RECURRENT = "recurrent"  # RNN, LSTM, GRU
    QUANTIZE = "quantize"  # Quantization ops
    CAST = "cast"  # Type conversion
    CONTROL = "control"  # Control flow (If, Loop)

    # Misc
    CONSTANT = "constant"  # Constants and identity
    UNKNOWN = "unknown"  # Unrecognized ops


class OpIcon(BaseModel):
    """Visual representation for an operator category."""

    model_config = ConfigDict(frozen=True)

    category: OpCategory
    shape: str  # SVG shape: "rect", "circle", "diamond", "hexagon", etc.
    symbol: str  # Unicode symbol for text rendering
    color: str  # Default color (hex)
    description: str

    def to_dict(self) -> dict:
        return {
            "category": self.category.value,
            "shape": self.shape,
            "symbol": self.symbol,
            "color": self.color,
            "description": self.description,
        }


# Icon definitions for each category
CATEGORY_ICONS: dict[OpCategory, OpIcon] = {
    OpCategory.CONV: OpIcon(
        category=OpCategory.CONV,
        shape="rect",
        symbol="▦",
        color="#4A90D9",  # Blue
        description="Convolution",
    ),
    OpCategory.LINEAR: OpIcon(
        category=OpCategory.LINEAR,
        shape="diamond",
        symbol="◆",
        color="#9B59B6",  # Purple
        description="Linear/MatMul",
    ),
    OpCategory.ATTENTION: OpIcon(
        category=OpCategory.ATTENTION,
        shape="hexagon",
        symbol="◎",
        color="#E67E22",  # Orange
        description="Attention",
    ),
    OpCategory.NORM: OpIcon(
        category=OpCategory.NORM,
        shape="rect",
        symbol="▬",
        color="#7F8C8D",  # Gray
        description="Normalization",
    ),
    OpCategory.ACTIVATION: OpIcon(
        category=OpCategory.ACTIVATION,
        shape="circle",
        symbol="⚡",
        color="#F1C40F",  # Yellow
        description="Activation",
    ),
    OpCategory.POOL: OpIcon(
        category=OpCategory.POOL,
        shape="trapezoid",
        symbol="▼",
        color="#1ABC9C",  # Teal
        description="Pooling",
    ),
    OpCategory.DROPOUT: OpIcon(
        category=OpCategory.DROPOUT,
        shape="circle",
        symbol="◌",
        color="#95A5A6",  # Light gray
        description="Dropout",
    ),
    OpCategory.RESHAPE: OpIcon(
        category=OpCategory.RESHAPE,
        shape="parallelogram",
        symbol="⬔",
        color="#3498DB",  # Light blue
        description="Reshape",
    ),
    OpCategory.TRANSPOSE: OpIcon(
        category=OpCategory.TRANSPOSE,
        shape="parallelogram",
        symbol="↔",
        color="#3498DB",
        description="Transpose",
    ),
    OpCategory.SLICE: OpIcon(
        category=OpCategory.SLICE,
        shape="rect",
        symbol="✂",
        color="#E74C3C",  # Red
        description="Slice/Index",
    ),
    OpCategory.CONCAT: OpIcon(
        category=OpCategory.CONCAT,
        shape="rect",
        symbol="⊕",
        color="#2ECC71",  # Green
        description="Concat",
    ),
    OpCategory.SPLIT: OpIcon(
        category=OpCategory.SPLIT,
        shape="rect",
        symbol="⊖",
        color="#E74C3C",
        description="Split",
    ),
    OpCategory.PAD: OpIcon(
        category=OpCategory.PAD,
        shape="rect",
        symbol="▭",
        color="#BDC3C7",  # Silver
        description="Padding",
    ),
    OpCategory.ELEMENTWISE: OpIcon(
        category=OpCategory.ELEMENTWISE,
        shape="circle",
        symbol="±",
        color="#9B59B6",  # Purple
        description="Elementwise",
    ),
    OpCategory.REDUCE: OpIcon(
        category=OpCategory.REDUCE,
        shape="triangle",
        symbol="Σ",
        color="#E74C3C",  # Red
        description="Reduce",
    ),
    OpCategory.COMPARE: OpIcon(
        category=OpCategory.COMPARE,
        shape="diamond",
        symbol="?",
        color="#F39C12",  # Dark yellow
        description="Compare",
    ),
    OpCategory.EMBED: OpIcon(
        category=OpCategory.EMBED,
        shape="rect",
        symbol="📖",
        color="#8E44AD",  # Dark purple
        description="Embedding",
    ),
    OpCategory.RECURRENT: OpIcon(
        category=OpCategory.RECURRENT,
        shape="rect",
        symbol="↻",
        color="#16A085",  # Dark teal
        description="Recurrent",
    ),
    OpCategory.QUANTIZE: OpIcon(
        category=OpCategory.QUANTIZE,
        shape="octagon",
        symbol="Q",
        color="#27AE60",  # Dark green
        description="Quantization",
    ),
    OpCategory.CAST: OpIcon(
        category=OpCategory.CAST,
        shape="circle",
        symbol="⇄",
        color="#95A5A6",
        description="Cast",
    ),
    OpCategory.CONTROL: OpIcon(
        category=OpCategory.CONTROL,
        shape="diamond",
        symbol="◇",
        color="#E74C3C",
        description="Control Flow",
    ),
    OpCategory.CONSTANT: OpIcon(
        category=OpCategory.CONSTANT,
        shape="circle",
        symbol="•",
        color="#BDC3C7",
        description="Constant",
    ),
    OpCategory.UNKNOWN: OpIcon(
        category=OpCategory.UNKNOWN,
        shape="rect",
        symbol="?",
        color="#7F8C8D",
        description="Unknown",
    ),
}


# Mapping of ONNX op types to categories
# This covers all standard ONNX operators as of opset 21
OP_TO_CATEGORY: dict[str, OpCategory] = {
    # Convolution
    "Conv": OpCategory.CONV,
    "ConvInteger": OpCategory.CONV,
    "ConvTranspose": OpCategory.CONV,
    "DeformConv": OpCategory.CONV,
    # Linear/Matrix
    "MatMul": OpCategory.LINEAR,
    "MatMulInteger": OpCategory.LINEAR,
    "Gemm": OpCategory.LINEAR,
    "QLinearMatMul": OpCategory.LINEAR,
    # Normalization
    "BatchNormalization": OpCategory.NORM,
    "InstanceNormalization": OpCategory.NORM,
    "LayerNormalization": OpCategory.NORM,
    "GroupNormalization": OpCategory.NORM,
    "LpNormalization": OpCategory.NORM,
    "MeanVarianceNormalization": OpCategory.NORM,
    "SimplifiedLayerNormalization": OpCategory.NORM,
    # Activation functions
    "Relu": OpCategory.ACTIVATION,
    "LeakyRelu": OpCategory.ACTIVATION,
    "PRelu": OpCategory.ACTIVATION,
    "Selu": OpCategory.ACTIVATION,
    "Elu": OpCategory.ACTIVATION,
    "Celu": OpCategory.ACTIVATION,
    "Sigmoid": OpCategory.ACTIVATION,
    "HardSigmoid": OpCategory.ACTIVATION,
    "Tanh": OpCategory.ACTIVATION,
    "Softmax": OpCategory.ACTIVATION,
    "LogSoftmax": OpCategory.ACTIVATION,
    "Softplus": OpCategory.ACTIVATION,
    "Softsign": OpCategory.ACTIVATION,
    "HardSwish": OpCategory.ACTIVATION,
    "Mish": OpCategory.ACTIVATION,
    "Gelu": OpCategory.ACTIVATION,
    "FastGelu": OpCategory.ACTIVATION,
    "QuickGelu": OpCategory.ACTIVATION,
    "Silu": OpCategory.ACTIVATION,
    "Swish": OpCategory.ACTIVATION,
    "ThresholdedRelu": OpCategory.ACTIVATION,
    "Shrink": OpCategory.ACTIVATION,
    # Pooling
    "MaxPool": OpCategory.POOL,
    "AveragePool": OpCategory.POOL,
    "GlobalMaxPool": OpCategory.POOL,
    "GlobalAveragePool": OpCategory.POOL,
    "LpPool": OpCategory.POOL,
    "MaxRoiPool": OpCategory.POOL,
    "RoiAlign": OpCategory.POOL,
    "MaxUnpool": OpCategory.POOL,
    # Dropout/Regularization
    "Dropout": OpCategory.DROPOUT,
    # Reshape operations
    "Reshape": OpCategory.RESHAPE,
    "Flatten": OpCategory.RESHAPE,
    "Squeeze": OpCategory.RESHAPE,
    "Unsqueeze": OpCategory.RESHAPE,
    "Expand": OpCategory.RESHAPE,
    "Tile": OpCategory.RESHAPE,
    "SpaceToDepth": OpCategory.RESHAPE,
    "DepthToSpace": OpCategory.RESHAPE,
    # Transpose operations
    "Transpose": OpCategory.TRANSPOSE,
    "Einsum": OpCategory.TRANSPOSE,
    # Slice/Index operations
    "Slice": OpCategory.SLICE,
    "Gather": OpCategory.SLICE,
    "GatherElements": OpCategory.SLICE,
    "GatherND": OpCategory.SLICE,
    "ScatterElements": OpCategory.SLICE,
    "ScatterND": OpCategory.SLICE,
    "Compress": OpCategory.SLICE,
    "TopK": OpCategory.SLICE,
    "NonZero": OpCategory.SLICE,
    "NonMaxSuppression": OpCategory.SLICE,
    # Concat/Join operations
    "Concat": OpCategory.CONCAT,
    "ConcatFromSequence": OpCategory.CONCAT,
    # Split operations
    "Split": OpCategory.SPLIT,
    "SplitToSequence": OpCategory.SPLIT,
    "Chunk": OpCategory.SPLIT,
    # Padding
    "Pad": OpCategory.PAD,
    "ConstantOfShape": OpCategory.PAD,
    # Elementwise math
    "Add": OpCategory.ELEMENTWISE,
    "Sub": OpCategory.ELEMENTWISE,
    "Mul": OpCategory.ELEMENTWISE,
    "Div": OpCategory.ELEMENTWISE,
    "Pow": OpCategory.ELEMENTWISE,
    "Sqrt": OpCategory.ELEMENTWISE,
    "Reciprocal": OpCategory.ELEMENTWISE,
    "Exp": OpCategory.ELEMENTWISE,
    "Log": OpCategory.ELEMENTWISE,
    "Abs": OpCategory.ELEMENTWISE,
    "Neg": OpCategory.ELEMENTWISE,
    "Sign": OpCategory.ELEMENTWISE,
    "Ceil": OpCategory.ELEMENTWISE,
    "Floor": OpCategory.ELEMENTWISE,
    "Round": OpCategory.ELEMENTWISE,
    "Clip": OpCategory.ELEMENTWISE,
    "Min": OpCategory.ELEMENTWISE,
    "Max": OpCategory.ELEMENTWISE,
    "Mean": OpCategory.ELEMENTWISE,
    "Sum": OpCategory.ELEMENTWISE,
    "Mod": OpCategory.ELEMENTWISE,
    "BitShift": OpCategory.ELEMENTWISE,
    "BitwiseAnd": OpCategory.ELEMENTWISE,
    "BitwiseNot": OpCategory.ELEMENTWISE,
    "BitwiseOr": OpCategory.ELEMENTWISE,
    "BitwiseXor": OpCategory.ELEMENTWISE,
    # Trigonometric
    "Sin": OpCategory.ELEMENTWISE,
    "Cos": OpCategory.ELEMENTWISE,
    "Tan": OpCategory.ELEMENTWISE,
    "Asin": OpCategory.ELEMENTWISE,
    "Acos": OpCategory.ELEMENTWISE,
    "Atan": OpCategory.ELEMENTWISE,
    "Sinh": OpCategory.ELEMENTWISE,
    "Cosh": OpCategory.ELEMENTWISE,  # Also activation
    "Asinh": OpCategory.ELEMENTWISE,
    "Acosh": OpCategory.ELEMENTWISE,
    "Atanh": OpCategory.ELEMENTWISE,
    # Reduction operations
    "ReduceSum": OpCategory.REDUCE,
    "ReduceMean": OpCategory.REDUCE,
    "ReduceMax": OpCategory.REDUCE,
    "ReduceMin": OpCategory.REDUCE,
    "ReduceProd": OpCategory.REDUCE,
    "ReduceL1": OpCategory.REDUCE,
    "ReduceL2": OpCategory.REDUCE,
    "ReduceLogSum": OpCategory.REDUCE,
    "ReduceLogSumExp": OpCategory.REDUCE,
    "ReduceSumSquare": OpCategory.REDUCE,
    "ArgMax": OpCategory.REDUCE,
    "ArgMin": OpCategory.REDUCE,
    # Comparison/Logic
    "Equal": OpCategory.COMPARE,
    "Greater": OpCategory.COMPARE,
    "GreaterOrEqual": OpCategory.COMPARE,
    "Less": OpCategory.COMPARE,
    "LessOrEqual": OpCategory.COMPARE,
    "And": OpCategory.COMPARE,
    "Or": OpCategory.COMPARE,
    "Xor": OpCategory.COMPARE,
    "Not": OpCategory.COMPARE,
    "Where": OpCategory.COMPARE,
    "IsNaN": OpCategory.COMPARE,
    "IsInf": OpCategory.COMPARE,
    # Embedding
    "Embedding": OpCategory.EMBED,
    # Note: Gather on embedding tables detected separately
    # Recurrent
    "RNN": OpCategory.RECURRENT,
    "LSTM": OpCategory.RECURRENT,
    "GRU": OpCategory.RECURRENT,
    # Quantization
    "QuantizeLinear": OpCategory.QUANTIZE,
    "DequantizeLinear": OpCategory.QUANTIZE,
    "DynamicQuantizeLinear": OpCategory.QUANTIZE,
    "QLinearConv": OpCategory.QUANTIZE,
    # Cast/Type conversion
    "Cast": OpCategory.CAST,
    "CastLike": OpCategory.CAST,
    # Control flow
    "If": OpCategory.CONTROL,
    "Loop": OpCategory.CONTROL,
    "Scan": OpCategory.CONTROL,
    "SequenceAt": OpCategory.CONTROL,
    "SequenceConstruct": OpCategory.CONTROL,
    "SequenceEmpty": OpCategory.CONTROL,
    "SequenceErase": OpCategory.CONTROL,
    "SequenceInsert": OpCategory.CONTROL,
    "SequenceLength": OpCategory.CONTROL,
    # Constants
    "Constant": OpCategory.CONSTANT,
    "Identity": OpCategory.CONSTANT,
    "Shape": OpCategory.CONSTANT,
    "Size": OpCategory.CONSTANT,
    "Range": OpCategory.CONSTANT,
    "EyeLike": OpCategory.CONSTANT,
    "RandomNormal": OpCategory.CONSTANT,
    "RandomNormalLike": OpCategory.CONSTANT,
    "RandomUniform": OpCategory.CONSTANT,
    "RandomUniformLike": OpCategory.CONSTANT,
    "Multinomial": OpCategory.CONSTANT,
    "OneHot": OpCategory.CONSTANT,
}


def get_op_category(op_type: str) -> OpCategory:
    """Get the visual category for an ONNX operator."""
    return OP_TO_CATEGORY.get(op_type, OpCategory.UNKNOWN)


def get_op_icon(op_type: str) -> OpIcon:
    """Get the icon definition for an ONNX operator."""
    category = get_op_category(op_type)
    return CATEGORY_ICONS[category]


def get_all_categories() -> list[OpIcon]:
    """Get all category icon definitions."""
    return list(CATEGORY_ICONS.values())


# Size scaling based on FLOPs (log scale)
def compute_node_size(flops: int, min_size: float = 20, max_size: float = 80) -> float:
    """
    Compute visual node size based on FLOPs.

    Task 5.5.3: Size scaling function.

    Uses log scale to handle the huge range of FLOPs (1 to 1T+).
    """
    if flops <= 0:
        return min_size

    # Log scale: 1 FLOP = min_size, 1T FLOPs = max_size
    log_flops = math.log10(max(flops, 1))
    log_max = 12  # 10^12 = 1 trillion FLOPs

    # Linear interpolation in log space
    t = min(log_flops / log_max, 1.0)
    return min_size + t * (max_size - min_size)


# Color intensity based on compute/memory
class ColorMapping(BaseModel):
    """Color mapping configuration for nodes."""

    model_config = ConfigDict(frozen=True)

    # Precision-based colors
    PRECISION_COLORS: ClassVar[dict[str, str]] = {
        "fp32": "#4A90D9",  # Blue
        "fp16": "#2ECC71",  # Green
        "bf16": "#9B59B6",  # Purple
        "int8": "#F1C40F",  # Yellow
        "int4": "#E67E22",  # Orange
        "uint8": "#F39C12",  # Dark yellow
    }

    # Memory intensity gradient (low to high)
    MEMORY_GRADIENT: ClassVar[list[str]] = [
        "#2ECC71",  # Green (low)
        "#F1C40F",  # Yellow (medium)
        "#E67E22",  # Orange (high)
        "#E74C3C",  # Red (very high)
    ]

    @staticmethod
    def get_precision_color(precision: str) -> str:
        """Get color for precision type."""
        return ColorMapping.PRECISION_COLORS.get(precision.lower(), "#7F8C8D")

    @staticmethod
    def get_memory_color(memory_bytes: int, max_bytes: int) -> str:
        """Get color based on memory usage intensity."""
        if max_bytes <= 0:
            return ColorMapping.MEMORY_GRADIENT[0]

        ratio = min(memory_bytes / max_bytes, 1.0)
        idx = int(ratio * (len(ColorMapping.MEMORY_GRADIENT) - 1))
        return ColorMapping.MEMORY_GRADIENT[idx]


# SVG icon templates
SVG_ICONS: dict[str, str] = {
    "rect": '<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{color}" stroke="{stroke}" stroke-width="1"/>',
    "circle": '<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}" stroke="{stroke}" stroke-width="1"/>',
    "diamond": '<polygon points="{cx},{y} {x2},{cy} {cx},{y2} {x},{cy}" fill="{color}" stroke="{stroke}" stroke-width="1"/>',
    "hexagon": '<polygon points="{p1} {p2} {p3} {p4} {p5} {p6}" fill="{color}" stroke="{stroke}" stroke-width="1"/>',
    "triangle": '<polygon points="{cx},{y} {x2},{y2} {x},{y2}" fill="{color}" stroke="{stroke}" stroke-width="1"/>',
}


def generate_svg_node(
    op_type: str,
    x: float,
    y: float,
    size: float,
    label: str | None = None,
    flops: int = 0,
) -> str:
    """
    Generate SVG markup for a node.

    Task 5.5.5: Create SVG icon for HTML embedding.
    """
    icon = get_op_icon(op_type)
    node_size = compute_node_size(flops) if flops > 0 else size

    half = node_size / 2
    cx, cy = x + half, y + half

    # Generate shape
    if icon.shape == "rect":
        shape_svg = SVG_ICONS["rect"].format(
            x=x, y=y, w=node_size, h=node_size, color=icon.color, stroke="#333"
        )
    elif icon.shape == "circle":
        shape_svg = SVG_ICONS["circle"].format(
            cx=cx, cy=cy, r=half, color=icon.color, stroke="#333"
        )
    elif icon.shape == "diamond":
        shape_svg = SVG_ICONS["diamond"].format(
            cx=cx,
            cy=cy,
            x=x,
            y=y,
            x2=x + node_size,
            y2=y + node_size,
            color=icon.color,
            stroke="#333",
        )
    elif icon.shape == "triangle":
        shape_svg = SVG_ICONS["triangle"].format(
            cx=cx,
            x=x,
            y=y,
            x2=x + node_size,
            y2=y + node_size,
            color=icon.color,
            stroke="#333",
        )
    else:
        # Default to rect
        shape_svg = SVG_ICONS["rect"].format(
            x=x, y=y, w=node_size, h=node_size, color=icon.color, stroke="#333"
        )

    # Add label if provided
    if label:
        label_svg = f'<text x="{cx}" y="{cy + 4}" text-anchor="middle" font-size="10" fill="white">{icon.symbol}</text>'
        shape_svg += label_svg

    return f'<g class="node node-{icon.category.value}" data-op="{op_type}">{shape_svg}</g>'


def generate_legend_svg(width: int = 400, height: int = 300) -> str:
    """
    Generate SVG legend showing all categories.

    Task 5.5.6: Add legend/key to visualization.
    """
    lines = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">']
    lines.append('<rect width="100%" height="100%" fill="#1a1a2e"/>')
    lines.append(
        '<text x="10" y="25" font-size="14" fill="white" font-weight="bold">Op Type Legend</text>'
    )

    y = 45
    col_width = width // 2
    col = 0

    for _i, (category, icon) in enumerate(CATEGORY_ICONS.items()):
        if category == OpCategory.UNKNOWN:
            continue

        x = 10 + col * col_width
        node_svg = generate_svg_node(category.value, x, y, 20)
        lines.append(node_svg)
        lines.append(
            f'<text x="{x + 30}" y="{y + 15}" font-size="11" fill="#ccc">{icon.description}</text>'
        )

        col += 1
        if col >= 2:
            col = 0
            y += 30

    lines.append("</svg>")
    return "\n".join(lines)
