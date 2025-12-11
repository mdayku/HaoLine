# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
GGUF format reader for llama.cpp models.

GGUF (GGML Universal Format) is the standard format for llama.cpp
and other GGML-based inference engines. This reader extracts:
- Model metadata (architecture, context length, etc.)
- Tensor information (names, shapes, quantization types)
- Memory footprint estimates

Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
"""

from __future__ import annotations

import struct
from enum import IntEnum
from pathlib import Path
from typing import Any, BinaryIO, cast

from pydantic import BaseModel, ConfigDict, Field, computed_field

# GGUF magic number
GGUF_MAGIC = b"GGUF"

# Supported GGUF versions
GGUF_VERSION_MIN = 2
GGUF_VERSION_MAX = 3


class GGMLType(IntEnum):
    """GGML tensor data types with their properties."""

    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    BF16 = 29


# Bits per weight for each quantization type
GGML_TYPE_BITS: dict[int, float] = {
    GGMLType.F32: 32.0,
    GGMLType.F16: 16.0,
    GGMLType.BF16: 16.0,
    GGMLType.Q4_0: 4.5,  # 4 bits + 0.5 for scales
    GGMLType.Q4_1: 5.0,
    GGMLType.Q5_0: 5.5,
    GGMLType.Q5_1: 6.0,
    GGMLType.Q8_0: 8.5,
    GGMLType.Q8_1: 9.0,
    GGMLType.Q2_K: 2.5625,
    GGMLType.Q3_K: 3.4375,
    GGMLType.Q4_K: 4.5,
    GGMLType.Q5_K: 5.5,
    GGMLType.Q6_K: 6.5625,
    GGMLType.Q8_K: 8.5,
    GGMLType.IQ2_XXS: 2.0625,
    GGMLType.IQ2_XS: 2.3125,
    GGMLType.IQ3_XXS: 3.0625,
    GGMLType.IQ1_S: 1.5,
    GGMLType.IQ4_NL: 4.5,
    GGMLType.IQ3_S: 3.4375,
    GGMLType.IQ2_S: 2.5,
    GGMLType.IQ4_XS: 4.25,
    GGMLType.I8: 8.0,
    GGMLType.I16: 16.0,
    GGMLType.I32: 32.0,
    GGMLType.I64: 64.0,
    GGMLType.F64: 64.0,
}


def ggml_type_name(type_id: int) -> str:
    """Get human-readable name for a GGML type."""
    try:
        return GGMLType(type_id).name
    except ValueError:
        return f"UNKNOWN_{type_id}"


class GGUFValueType(IntEnum):
    """GGUF metadata value types."""

    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class TensorInfo(BaseModel):
    """Information about a single tensor in the GGUF file."""

    model_config = ConfigDict(frozen=True)

    name: str
    n_dims: int
    dims: tuple[int, ...]
    type_id: int
    offset: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def type_name(self) -> str:
        """Human-readable type name."""
        return ggml_type_name(self.type_id)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def n_elements(self) -> int:
        """Total number of elements."""
        result = 1
        for d in self.dims:
            result *= d
        return result

    @computed_field  # type: ignore[prop-decorator]
    @property
    def bits_per_element(self) -> float:
        """Bits per element for this tensor's type."""
        return GGML_TYPE_BITS.get(self.type_id, 32.0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def size_bytes(self) -> int:
        """Estimated size in bytes."""
        return int(self.n_elements * self.bits_per_element / 8)


class GGUFInfo(BaseModel):
    """Parsed GGUF file information."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: Path
    version: int
    tensor_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    tensors: list[TensorInfo] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def architecture(self) -> str:
        """Model architecture (e.g., 'llama', 'mistral')."""
        return str(self.metadata.get("general.architecture", "unknown"))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def model_name(self) -> str:
        """Model name from metadata."""
        return str(self.metadata.get("general.name", self.path.stem))

    @property
    def context_length(self) -> int | None:
        """Maximum context length."""
        arch = self.architecture
        return self.metadata.get(f"{arch}.context_length")

    @property
    def embedding_length(self) -> int | None:
        """Hidden size / embedding dimension."""
        arch = self.architecture
        return self.metadata.get(f"{arch}.embedding_length")

    @property
    def block_count(self) -> int | None:
        """Number of transformer blocks/layers."""
        arch = self.architecture
        return self.metadata.get(f"{arch}.block_count")

    @property
    def head_count(self) -> int | None:
        """Number of attention heads."""
        arch = self.architecture
        return self.metadata.get(f"{arch}.attention.head_count")

    @property
    def head_count_kv(self) -> int | None:
        """Number of KV heads (for GQA/MQA)."""
        arch = self.architecture
        return self.metadata.get(f"{arch}.attention.head_count_kv")

    @property
    def vocab_size(self) -> int | None:
        """Vocabulary size."""
        arch = self.architecture
        return self.metadata.get(f"{arch}.vocab_size")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_params(self) -> int:
        """Total parameter count."""
        return sum(t.n_elements for t in self.tensors)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_size_bytes(self) -> int:
        """Total model size in bytes."""
        return sum(t.size_bytes for t in self.tensors)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def quantization_breakdown(self) -> dict[str, int]:
        """Count of tensors by quantization type."""
        breakdown: dict[str, int] = {}
        for t in self.tensors:
            type_name = t.type_name
            breakdown[type_name] = breakdown.get(type_name, 0) + 1
        return breakdown

    @computed_field  # type: ignore[prop-decorator]
    @property
    def size_breakdown(self) -> dict[str, int]:
        """Size in bytes by quantization type."""
        breakdown: dict[str, int] = {}
        for t in self.tensors:
            type_name = t.type_name
            breakdown[type_name] = breakdown.get(type_name, 0) + t.size_bytes
        return breakdown

    def estimate_vram(self, context_length: int | None = None) -> dict[str, int]:
        """
        Estimate VRAM requirements.

        Args:
            context_length: Context length to use for KV cache estimation.
                           Defaults to model's context_length metadata.

        Returns:
            Dict with 'weights', 'kv_cache', and 'total' in bytes.
        """
        ctx = context_length or self.context_length or 2048
        weights = self.total_size_bytes

        # Estimate KV cache
        kv_cache = 0
        n_layers = self.block_count or 32
        hidden = self.embedding_length or 4096
        n_kv_heads = self.head_count_kv or self.head_count or 32
        head_dim = hidden // (self.head_count or 32)

        # KV cache: 2 (K+V) * layers * ctx * kv_heads * head_dim * 2 bytes (fp16)
        kv_cache = 2 * n_layers * ctx * n_kv_heads * head_dim * 2

        return {
            "weights": weights,
            "kv_cache": kv_cache,
            "total": weights + kv_cache,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return cast(dict[str, Any], self.model_dump(mode="json"))


class GGUFReader:
    """Reader for GGUF format files."""

    def __init__(self, path: str | Path):
        """
        Initialize reader with file path.

        Args:
            path: Path to the GGUF file.
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"GGUF file not found: {self.path}")

    def read(self) -> GGUFInfo:
        """
        Read and parse the GGUF file.

        Returns:
            GGUFInfo with parsed metadata and tensor information.

        Raises:
            ValueError: If the file is not a valid GGUF file.
        """
        with open(self.path, "rb") as f:
            return self._parse(f)

    def _parse(self, f: BinaryIO) -> GGUFInfo:
        """Parse the GGUF file structure."""
        # Read magic
        magic = f.read(4)
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file: expected {GGUF_MAGIC!r}, got {magic!r}")

        # Read version
        version = struct.unpack("<I", f.read(4))[0]
        if version < GGUF_VERSION_MIN or version > GGUF_VERSION_MAX:
            raise ValueError(
                f"Unsupported GGUF version {version} "
                f"(supported: {GGUF_VERSION_MIN}-{GGUF_VERSION_MAX})"
            )

        # Read counts
        tensor_count = struct.unpack("<Q", f.read(8))[0]
        metadata_kv_count = struct.unpack("<Q", f.read(8))[0]

        # Read metadata
        metadata = {}
        for _ in range(metadata_kv_count):
            key = self._read_string(f)
            value = self._read_value(f)
            metadata[key] = value

        # Read tensor info
        tensors = []
        for _ in range(tensor_count):
            tensor = self._read_tensor_info(f)
            tensors.append(tensor)

        return GGUFInfo(
            path=self.path,
            version=version,
            tensor_count=tensor_count,
            metadata=metadata,
            tensors=tensors,
        )

    def _read_string(self, f: BinaryIO) -> str:
        """Read a length-prefixed string."""
        length = struct.unpack("<Q", f.read(8))[0]
        return f.read(length).decode("utf-8")

    def _read_value(self, f: BinaryIO) -> Any:
        """Read a typed metadata value."""
        value_type = struct.unpack("<I", f.read(4))[0]

        if value_type == GGUFValueType.UINT8:
            return struct.unpack("<B", f.read(1))[0]
        elif value_type == GGUFValueType.INT8:
            return struct.unpack("<b", f.read(1))[0]
        elif value_type == GGUFValueType.UINT16:
            return struct.unpack("<H", f.read(2))[0]
        elif value_type == GGUFValueType.INT16:
            return struct.unpack("<h", f.read(2))[0]
        elif value_type == GGUFValueType.UINT32:
            return struct.unpack("<I", f.read(4))[0]
        elif value_type == GGUFValueType.INT32:
            return struct.unpack("<i", f.read(4))[0]
        elif value_type == GGUFValueType.FLOAT32:
            return struct.unpack("<f", f.read(4))[0]
        elif value_type == GGUFValueType.BOOL:
            return struct.unpack("<B", f.read(1))[0] != 0
        elif value_type == GGUFValueType.STRING:
            return self._read_string(f)
        elif value_type == GGUFValueType.ARRAY:
            return self._read_array(f)
        elif value_type == GGUFValueType.UINT64:
            return struct.unpack("<Q", f.read(8))[0]
        elif value_type == GGUFValueType.INT64:
            return struct.unpack("<q", f.read(8))[0]
        elif value_type == GGUFValueType.FLOAT64:
            return struct.unpack("<d", f.read(8))[0]
        else:
            raise ValueError(f"Unknown value type: {value_type}")

    def _read_array(self, f: BinaryIO) -> list[Any]:
        """Read an array value."""
        element_type = struct.unpack("<I", f.read(4))[0]
        length = struct.unpack("<Q", f.read(8))[0]

        # Read elements based on type
        if element_type == GGUFValueType.UINT8:
            return list(struct.unpack(f"<{length}B", f.read(length)))
        elif element_type == GGUFValueType.INT8:
            return list(struct.unpack(f"<{length}b", f.read(length)))
        elif element_type == GGUFValueType.UINT16:
            return list(struct.unpack(f"<{length}H", f.read(length * 2)))
        elif element_type == GGUFValueType.INT16:
            return list(struct.unpack(f"<{length}h", f.read(length * 2)))
        elif element_type == GGUFValueType.UINT32:
            return list(struct.unpack(f"<{length}I", f.read(length * 4)))
        elif element_type == GGUFValueType.INT32:
            return list(struct.unpack(f"<{length}i", f.read(length * 4)))
        elif element_type == GGUFValueType.FLOAT32:
            return list(struct.unpack(f"<{length}f", f.read(length * 4)))
        elif element_type == GGUFValueType.BOOL:
            return [b != 0 for b in struct.unpack(f"<{length}B", f.read(length))]
        elif element_type == GGUFValueType.STRING:
            return [self._read_string(f) for _ in range(length)]
        elif element_type == GGUFValueType.UINT64:
            return list(struct.unpack(f"<{length}Q", f.read(length * 8)))
        elif element_type == GGUFValueType.INT64:
            return list(struct.unpack(f"<{length}q", f.read(length * 8)))
        elif element_type == GGUFValueType.FLOAT64:
            return list(struct.unpack(f"<{length}d", f.read(length * 8)))
        else:
            # Skip unknown array types
            return []

    def _read_tensor_info(self, f: BinaryIO) -> TensorInfo:
        """Read tensor metadata."""
        name = self._read_string(f)
        n_dims = struct.unpack("<I", f.read(4))[0]
        dims = tuple(struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims))
        type_id = struct.unpack("<I", f.read(4))[0]
        offset = struct.unpack("<Q", f.read(8))[0]

        return TensorInfo(
            name=name,
            n_dims=n_dims,
            dims=dims,
            type_id=type_id,
            offset=offset,
        )


def is_gguf_file(path: str | Path) -> bool:
    """
    Check if a file is a valid GGUF file.

    Args:
        path: Path to check.

    Returns:
        True if the file starts with GGUF magic bytes.
    """
    path = Path(path)
    if not path.exists() or not path.is_file():
        return False

    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            return magic == GGUF_MAGIC
    except Exception:
        return False


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    size_float = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_float < 1024:
            return f"{size_float:.2f} {unit}"
        size_float /= 1024
    return f"{size_float:.2f} PB"
