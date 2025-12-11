# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
SafeTensors format reader.

SafeTensors is a simple, safe format for storing tensors, widely used
in the HuggingFace ecosystem. Unlike pickle-based formats, SafeTensors
cannot execute arbitrary code.

This reader extracts:
- Tensor names, shapes, and dtypes
- Parameter counts and memory estimates
- Metadata (if present)

Reference: https://github.com/huggingface/safetensors
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, computed_field

# Bytes per dtype
DTYPE_SIZES: dict[str, int] = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}


class SafeTensorInfo(BaseModel):
    """Information about a single tensor."""

    model_config = ConfigDict(frozen=True)

    name: str
    dtype: str
    shape: tuple[int, ...]

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
        """Size in bytes."""
        return int(self.n_elements * DTYPE_SIZES.get(self.dtype, 4))


class SafeTensorsInfo(BaseModel):
    """Parsed SafeTensors file information."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: Path
    tensors: list[SafeTensorInfo] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_params(self) -> int:
        """Total parameter count."""
        return sum(t.n_elements for t in self.tensors)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_size_bytes(self) -> int:
        """Total size in bytes."""
        return sum(t.size_bytes for t in self.tensors)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dtype_breakdown(self) -> dict[str, int]:
        """Count of tensors by dtype."""
        breakdown: dict[str, int] = {}
        for t in self.tensors:
            breakdown[t.dtype] = breakdown.get(t.dtype, 0) + 1
        return breakdown

    @computed_field  # type: ignore[prop-decorator]
    @property
    def size_breakdown(self) -> dict[str, int]:
        """Size in bytes by dtype."""
        breakdown: dict[str, int] = {}
        for t in self.tensors:
            breakdown[t.dtype] = breakdown.get(t.dtype, 0) + t.size_bytes
        return breakdown

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return cast(dict[str, Any], self.model_dump(mode="json"))


class SafeTensorsReader:
    """Reader for SafeTensors format files."""

    def __init__(self, path: str | Path):
        """
        Initialize reader with file path.

        Args:
            path: Path to the SafeTensors file.

        Raises:
            ImportError: If safetensors library is not installed.
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"SafeTensors file not found: {self.path}")

        try:
            import safetensors  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "safetensors library required. Install with: pip install safetensors"
            ) from e

    def read(self) -> SafeTensorsInfo:
        """
        Read and parse the SafeTensors file.

        Returns:
            SafeTensorsInfo with tensor information.
        """
        from safetensors import safe_open

        tensors = []
        metadata = {}

        with safe_open(self.path, framework="np") as f:
            # Get metadata if present
            meta = f.metadata()
            if meta:
                metadata = dict(meta)

            # Get tensor info
            for name in f.keys():
                tensor = f.get_tensor(name)
                info = SafeTensorInfo(
                    name=name,
                    dtype=self._numpy_dtype_to_safetensors(tensor.dtype),
                    shape=tuple(tensor.shape),
                )
                tensors.append(info)

        return SafeTensorsInfo(
            path=self.path,
            tensors=tensors,
            metadata=metadata,
        )

    def read_header_only(self) -> SafeTensorsInfo:
        """
        Read only the header without loading tensor data.

        This is faster for large files when you only need metadata.

        Returns:
            SafeTensorsInfo with tensor information.
        """
        import json

        tensors = []
        metadata = {}

        # SafeTensors header is at the start of the file
        # Format: 8 bytes (header size as u64) + header JSON
        with open(self.path, "rb") as f:
            # Read header size
            header_size = int.from_bytes(f.read(8), "little")

            # Read header JSON
            header_bytes = f.read(header_size)
            header = json.loads(header_bytes)

            # Extract metadata
            if "__metadata__" in header:
                metadata = header.pop("__metadata__")

            # Extract tensor info
            for name, info in header.items():
                dtype = info["dtype"]
                shape = tuple(info["shape"])
                tensors.append(SafeTensorInfo(name=name, dtype=dtype, shape=shape))

        return SafeTensorsInfo(
            path=self.path,
            tensors=tensors,
            metadata=metadata,
        )

    def _numpy_dtype_to_safetensors(self, dtype) -> str:
        """Convert numpy dtype to safetensors dtype string."""
        import numpy as np

        dtype_map = {
            np.float64: "F64",
            np.float32: "F32",
            np.float16: "F16",
            np.int64: "I64",
            np.int32: "I32",
            np.int16: "I16",
            np.int8: "I8",
            np.uint8: "U8",
            np.bool_: "BOOL",
        }

        # Handle bfloat16 specially (might be stored as uint16)
        dtype_name = str(dtype)
        if "bfloat16" in dtype_name:
            return "BF16"

        return dtype_map.get(dtype.type, "F32")


def is_safetensors_file(path: str | Path) -> bool:
    """
    Check if a file is a valid SafeTensors file.

    Args:
        path: Path to check.

    Returns:
        True if the file appears to be a valid SafeTensors file.
    """
    path = Path(path)
    if not path.exists() or not path.is_file():
        return False

    # Check extension
    if path.suffix.lower() != ".safetensors":
        return False

    # Try to read header
    try:
        with open(path, "rb") as f:
            header_size = int.from_bytes(f.read(8), "little")
            # Sanity check: header shouldn't be larger than 100MB
            if header_size > 100 * 1024 * 1024:
                return False
            # Try to parse as JSON

            header_bytes = f.read(min(header_size, 1024))  # Just read start
            # Should start with '{'
            return header_bytes.startswith(b"{")
    except Exception:
        return False


def is_available() -> bool:
    """Check if safetensors library is available."""
    try:
        import safetensors  # noqa: F401

        return True
    except ImportError:
        return False
