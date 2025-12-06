# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
Format adapters for various model file formats.

Each adapter provides read (and optionally write) capabilities
for a specific model format, extracting metadata and tensor info.

Supported formats:
- GGUF: llama.cpp models (pure Python, no deps)
- SafeTensors: HuggingFace models (requires safetensors)
- TFLite: TensorFlow Lite models (pure Python header parsing)
- CoreML: Apple ML models (requires coremltools)
- OpenVINO: Intel models (requires openvino)
- TensorRT: NVIDIA optimized engines (requires tensorrt + NVIDIA GPU)
"""

from .coreml import (
    CoreMLInfo,
    CoreMLReader,
    is_coreml_file,
)
from .coreml import (
    is_available as coreml_available,
)
from .gguf import GGUFInfo, GGUFReader, is_gguf_file
from .openvino import (
    OpenVINOInfo,
    OpenVINOReader,
    is_openvino_file,
)
from .openvino import (
    is_available as openvino_available,
)
from .safetensors import (
    SafeTensorsInfo,
    SafeTensorsReader,
    is_safetensors_file,
)
from .safetensors import (
    is_available as safetensors_available,
)
from .tensorrt import (
    TRTEngineInfo,
    TRTEngineReader,
    is_tensorrt_file,
)
from .tensorrt import (
    is_available as tensorrt_available,
)
from .tflite import (
    TFLiteInfo,
    TFLiteReader,
    is_tflite_file,
)
from .tflite import (
    is_available as tflite_available,
)

__all__ = [
    # GGUF (Epic 24)
    "GGUFReader",
    "GGUFInfo",
    "is_gguf_file",
    # SafeTensors (Epic 19)
    "SafeTensorsReader",
    "SafeTensorsInfo",
    "is_safetensors_file",
    "safetensors_available",
    # TFLite (Epic 21)
    "TFLiteReader",
    "TFLiteInfo",
    "is_tflite_file",
    "tflite_available",
    # CoreML (Epic 20)
    "CoreMLReader",
    "CoreMLInfo",
    "is_coreml_file",
    "coreml_available",
    # OpenVINO (Epic 23)
    "OpenVINOReader",
    "OpenVINOInfo",
    "is_openvino_file",
    "openvino_available",
    # TensorRT (Epic 22)
    "TRTEngineReader",
    "TRTEngineInfo",
    "is_tensorrt_file",
    "tensorrt_available",
]


def detect_format(path: str) -> str | None:
    """
    Auto-detect model format from file.

    Args:
        path: Path to model file.

    Returns:
        Format name ('gguf', 'safetensors', 'tflite', 'coreml', 'openvino', 'tensorrt', 'onnx')
        or None if unknown.
    """
    from pathlib import Path as P

    p = P(path)

    # Check by extension first
    suffix = p.suffix.lower()
    if suffix == ".onnx":
        return "onnx"
    if suffix == ".gguf":
        return "gguf"
    if suffix == ".safetensors":
        return "safetensors"
    if suffix == ".tflite":
        return "tflite"
    if suffix in (".mlmodel", ".mlpackage"):
        return "coreml"
    if suffix in (".engine", ".plan"):
        return "tensorrt"
    if suffix == ".xml":
        if is_openvino_file(path):
            return "openvino"

    # Check by magic bytes / validation
    if is_gguf_file(path):
        return "gguf"
    if is_safetensors_file(path):
        return "safetensors"
    if is_tflite_file(path):
        return "tflite"
    if is_coreml_file(path):
        return "coreml"
    if is_openvino_file(path):
        return "openvino"
    if is_tensorrt_file(path):
        return "tensorrt"

    return None
