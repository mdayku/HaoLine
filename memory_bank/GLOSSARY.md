# HaoLine Glossary

> Project-specific terms, concepts, and shorthand used in HaoLine.

---

## Core Concepts

### Universal IR / UniversalGraph
The internal format-agnostic intermediate representation that HaoLine uses to normalize different model formats. All format readers (ONNX, PyTorch, TFLite, etc.) produce `UniversalGraph` objects that can be analyzed, compared, and visualized uniformly.

### InspectionReport
The primary output of HaoLine analysis. A structured report containing:
- Model metadata (name, format, framework)
- Graph summary (nodes, edges, parameters, FLOPs)
- Layer breakdown
- Hardware estimates
- Risk signals
- Universal IR graph

### Risk Signal
A heuristic flag computed from the model graph indicating potential deployment issues:
- Extreme depth (vanishing gradients)
- Unusual operations (unsupported on target hardware)
- Memory spikes (layers requiring large intermediate tensors)
- Quantization sensitivity

### Hardware Profile
A configuration describing target deployment hardware:
- GPU specs (memory, compute capability, tensor cores)
- Precision support (FP32, FP16, INT8, BF16)
- Estimated throughput and latency

---

## Formats

### ONNX (Open Neural Network Exchange)
Primary interchange format. Full support in HaoLine including graph visualization, FLOPs counting, and interactive maps.

### TensorRT
NVIDIA's high-performance inference engine. HaoLine can read `.engine`/`.plan` files and analyze quantization patterns.

### TFLite (TensorFlow Lite)
TensorFlow's mobile/edge format. HaoLine can read `.tflite` files and convert to ONNX via `tflite2onnx`.

### CoreML
Apple's ML framework format. HaoLine provides basic analysis on macOS with `coremltools`.

### OpenVINO
Intel's inference toolkit format. HaoLine reads `.xml`/`.bin` IR files.

### SafeTensors
Secure tensor storage format (HuggingFace). Weights-only, no graph - HaoLine extracts parameter counts and dtypes.

### GGUF
LLM weight format used by `llama.cpp`. HaoLine extracts metadata and parameter info.

---

## Metrics

### FLOPs (Floating Point Operations)
Total number of floating-point operations required for one forward pass. Key metric for computational cost.

### MACs (Multiply-Accumulate Operations)
Similar to FLOPs but counts multiply-add as one operation. FLOPs ≈ 2 × MACs.

### VRAM Footprint
Estimated GPU memory required for inference, including:
- Model weights
- Activations (intermediate tensors)
- Workspace memory

### Throughput
Estimated inferences per second on target hardware.

### Latency
Estimated time for one inference on target hardware.

---

## CLI Flags

### `--hardware`
Specify target hardware profile for estimates (e.g., `rtx4090`, `a100-80gb`, `t4`).

### `--precision`
Analysis precision: `fp32`, `fp16`, `int8`, `bf16`.

### `--compare`
Compare two models side-by-side.

### `--convert-to`
Convert model to another format (e.g., `onnx`, `tensorrt`).

### `--from-*`
Convert from a format to ONNX for analysis (e.g., `--from-pytorch`, `--from-tflite`).

### `--privacy`
Privacy level: `full` (default), `redact-names`, `summary-only`.

---

## Architecture

### Format Adapter
A module that reads a specific format and produces a `UniversalGraph`. Located in `src/haoline/formats/`.

### Analyzer
Core analysis engine that processes models and generates `InspectionReport`. Located in `src/haoline/analyzer.py`.

### Compare Mode
Feature that diffs two `InspectionReport` objects and highlights differences. Useful for quantization analysis.

### Quantization Advisor
Module that analyzes quantization patterns and provides recommendations. Uses TensorRT engine metadata when available.

---

## Project Structure

### Epic
A major feature area in the backlog (e.g., "Epic 21: Format Conversion").

### Story
A user-facing capability within an epic (e.g., "Story 21.3: TFLite Bidirectional Conversion").

### Task
An atomic work item within a story (e.g., "Task 21.3.1: Implement tflite2onnx integration").

### Delta Log
Daily progress notes appended to PRD.md documenting what was accomplished.

---

## External Tools

### `tflite2onnx`
Python package for converting TFLite models to ONNX. Lightweight, no TF dependency.

### `onnx2tf` / `onnx-tf`
Python packages for converting ONNX to TensorFlow. Currently broken with TF 2.16+/Keras 3.x.

### `coremltools`
Apple's toolkit for CoreML model manipulation.

### `openvino`
Intel's inference toolkit, includes model reading capabilities.

### `tensorrt`
NVIDIA's inference optimizer. Requires NVIDIA GPU.

---

*Add new terms alphabetically within their section.*

