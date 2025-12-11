# HaoLine Decision Log

> Chronological log of important architectural, product, and technical decisions.
> Format: Date - Title, Decision, Alternatives, Reasoning, Impacts.

---

## 2025-12-11 - ONNX to TFLite Conversion Removed

- **Decision:** Remove `--convert-to tflite` CLI option and mark as BLOCKED in backlog.
- **Alternatives considered:**
  - `onnx2tf` package (tried, failed with Keras 3.x compatibility)
  - `onnx-tf` + `tf.lite.TFLiteConverter` (tried, same Keras 3.x issues)
  - Waiting for upstream fixes
- **Reasoning:** Both `onnx2tf` and `onnx-tf` are fundamentally broken with TensorFlow 2.16+ / Keras 3.x. The error is `ValueError: A KerasTensor cannot be used as input to a TensorFlow function`. No workaround exists without pinning to older TF versions.
- **Impacts:**
  - CLI shows error with workarounds instead of broken conversion
  - Task 42.1.2 and 42.5.2 marked BLOCKED in BACKLOG.md
  - Removed `onnx2tf`, `onnx-tf`, `tensorflow` from `[tflite]` extras in pyproject.toml
  - Users directed to use TF-native workflows or `ai-edge-litert`

---

## 2025-12-11 - TFLite to ONNX via tflite2onnx

- **Decision:** Use `tflite2onnx` package for TFLite -> ONNX conversion.
- **Alternatives considered:**
  - `tf2onnx` (requires full TensorFlow install)
  - Manual conversion (too complex)
- **Reasoning:** `tflite2onnx` is lightweight, doesn't require TensorFlow, and handles most common ops. Some ops unsupported (e.g., SQUEEZE) but good enough for basic models.
- **Impacts:**
  - Added `--from-tflite` CLI flag
  - Added `tflite2onnx>=0.4.0` to `[tflite]` extras

---

## 2025-12-10 - Universal IR as Primary Abstraction

- **Decision:** HaoLine uses a "Universal IR" (intermediate representation) to normalize all model formats.
- **Alternatives considered:**
  - Format-specific analyzers with no common representation
  - Using ONNX as the canonical format for everything
- **Reasoning:** Universal IR allows format-agnostic analysis while preserving format-specific metadata. ONNX is too heavyweight for some formats (SafeTensors, GGUF).
- **Impacts:**
  - `universal_ir.py` is the core abstraction
  - All format readers produce UniversalGraph objects
  - Comparison and visualization work on UniversalGraph

---

## 2025-12-09 - FLOPs + VRAM as Primary Metrics

- **Decision:** HaoLine compares models primarily on FLOPs, VRAM footprint, and latency estimates.
- **Alternatives considered:**
  - Raw parameter count only
  - Synthetic benchmark scores
  - Inference time measurements
- **Reasoning:** FLOPs + VRAM are closer to actual deployment cost and easier to explain to infra + execs. Parameter count doesn't capture computational cost (e.g., depthwise conv vs regular conv).
- **Impacts:**
  - Report schema highlights these metrics
  - Streamlit UI shows FLOPs/VRAM prominently
  - Hardware profiler estimates based on these

---

## 2025-12-08 - Streamlit for Web UI

- **Decision:** Use Streamlit for the web interface, deployed on HuggingFace Spaces.
- **Alternatives considered:**
  - Gradio (less flexible for complex layouts)
  - FastAPI + React (too much overhead for MVP)
  - CLI only (not accessible enough)
- **Reasoning:** Streamlit is fast to build, Python-native, and HF Spaces offers free hosting with GPU support.
- **Impacts:**
  - `streamlit_app.py` at repo root for HF compatibility
  - Some features CLI-only due to dependency constraints

---

## 2025-12-07 - ONNX as Primary Format

- **Decision:** ONNX is the "first-class citizen" format with full feature support.
- **Alternatives considered:**
  - PyTorch-first approach
  - Format-neutral from day one
- **Reasoning:** ONNX is the de facto interchange format, has good tooling, and most frameworks can export to it. Starting with ONNX lets us build core features fast, then add format adapters.
- **Impacts:**
  - Full analysis (graph, params, FLOPs, visualization) for ONNX
  - Other formats get varying levels of support
  - PyTorch/TF users encouraged to export to ONNX for full features

---

*Add new decisions at the top of this file.*

