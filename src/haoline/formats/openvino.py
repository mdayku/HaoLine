# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
OpenVINO format reader.

OpenVINO is Intel's toolkit for optimizing and deploying AI inference.
This reader extracts model metadata from OpenVINO IR format (.xml/.bin).

Requires: openvino (pip install openvino)

Reference: https://docs.openvino.ai/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, computed_field


class OpenVINOLayerInfo(BaseModel):
    """Information about an OpenVINO layer."""

    model_config = ConfigDict(frozen=True)

    name: str
    type: str
    input_shapes: list[tuple[int, ...]] = Field(default_factory=list)
    output_shapes: list[tuple[int, ...]] = Field(default_factory=list)
    precision: str = "FP32"


class OpenVINOInfo(BaseModel):
    """Parsed OpenVINO model information."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: Path
    name: str
    framework: str
    layers: list[OpenVINOLayerInfo] = Field(default_factory=list)
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
    def precision_breakdown(self) -> dict[str, int]:
        """Count of layers by precision."""
        breakdown: dict[str, int] = {}
        for layer in self.layers:
            breakdown[layer.precision] = breakdown.get(layer.precision, 0) + 1
        return breakdown

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return cast(dict[str, Any], self.model_dump(mode="json"))


class OpenVINOReader:
    """Reader for OpenVINO IR format (.xml/.bin)."""

    def __init__(self, path: str | Path):
        """
        Initialize reader with file path.

        Args:
            path: Path to the OpenVINO model (.xml or directory).

        Raises:
            ImportError: If openvino is not installed.
        """
        self.path = Path(path)

        # Handle both .xml path and directory containing model
        if self.path.is_dir():
            xml_files = list(self.path.glob("*.xml"))
            if xml_files:
                self.path = xml_files[0]
            else:
                raise FileNotFoundError(f"No .xml file found in: {self.path}")
        elif self.path.suffix.lower() != ".xml":
            # Try adding .xml
            xml_path = self.path.with_suffix(".xml")
            if xml_path.exists():
                self.path = xml_path

        if not self.path.exists():
            raise FileNotFoundError(f"OpenVINO model not found: {self.path}")

        try:
            import openvino  # noqa: F401
        except ImportError as e:
            raise ImportError("openvino required. Install with: pip install openvino") from e

    def read(self) -> OpenVINOInfo:
        """
        Read and parse the OpenVINO model.

        Returns:
            OpenVINOInfo with model metadata.
        """
        from openvino.runtime import Core

        core = Core()
        model = core.read_model(str(self.path))

        # Extract basic info
        name = model.get_friendly_name()

        # Get RT info for framework
        rt_info = model.get_rt_info()
        framework = "unknown"
        if "framework" in rt_info:
            framework = str(rt_info["framework"].value)

        # Extract inputs
        inputs = []
        for inp in model.inputs:
            input_info = {
                "name": inp.get_any_name(),
                "shape": list(inp.get_partial_shape()),
                "element_type": str(inp.get_element_type()),
            }
            inputs.append(input_info)

        # Extract outputs
        outputs = []
        for out in model.outputs:
            output_info = {
                "name": out.get_any_name(),
                "shape": list(out.get_partial_shape()),
                "element_type": str(out.get_element_type()),
            }
            outputs.append(output_info)

        # Extract layers (operations)
        layers = []
        for op in model.get_ordered_ops():
            # Get input shapes
            input_shapes = []
            for inp in op.inputs():
                shape = inp.get_partial_shape()
                if shape.is_static:
                    input_shapes.append(tuple(shape.to_shape()))
                else:
                    input_shapes.append(tuple(d.get_length() if d.is_static else -1 for d in shape))

            # Get output shapes
            output_shapes = []
            for out in op.outputs():
                shape = out.get_partial_shape()
                if shape.is_static:
                    output_shapes.append(tuple(shape.to_shape()))
                else:
                    output_shapes.append(
                        tuple(d.get_length() if d.is_static else -1 for d in shape)
                    )

            # Get precision from output
            precision = "FP32"
            if op.outputs():
                element_type = str(op.outputs()[0].get_element_type())
                precision = element_type.upper().replace("F", "FP").replace("I", "INT")

            layers.append(
                OpenVINOLayerInfo(
                    name=op.get_friendly_name(),
                    type=op.get_type_name(),
                    input_shapes=input_shapes,
                    output_shapes=output_shapes,
                    precision=precision,
                )
            )

        # Additional metadata
        metadata = {}
        for key in rt_info:
            try:
                metadata[key] = str(rt_info[key].value)
            except Exception:
                pass

        return OpenVINOInfo(
            path=self.path,
            name=name,
            framework=framework,
            layers=layers,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
        )


def is_openvino_file(path: str | Path) -> bool:
    """
    Check if a file is an OpenVINO model.

    Args:
        path: Path to check.

    Returns:
        True if the file is an OpenVINO IR model.
    """
    path = Path(path)
    if not path.exists():
        return False

    # Check for .xml file
    if path.suffix.lower() == ".xml":
        # Quick check for OpenVINO XML structure
        try:
            with open(path, encoding="utf-8") as f:
                header = f.read(500)
                return "<net" in header and ("name=" in header or "version=" in header)
        except Exception:
            return False

    # Check if .bin file has corresponding .xml
    if path.suffix.lower() == ".bin":
        xml_path = path.with_suffix(".xml")
        return xml_path.exists() and is_openvino_file(xml_path)

    return False


def is_available() -> bool:
    """Check if openvino is available."""
    try:
        import openvino  # noqa: F401

        return True
    except ImportError:
        return False
