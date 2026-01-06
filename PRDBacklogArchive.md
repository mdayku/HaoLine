# HaoLine - PRD & Backlog Archive

*This document archives completed epics and historical changelog entries to reduce context window usage in the main PRD.md and BACKLOG.md files.*

**Created:** December 6, 2025  
**Purpose:** Historical reference for completed work

---

## Table of Contents

1. [Version 1.0 Release (Archived)](#version-10-release-archived)
2. [Version 1.1 Release (Archived)](#version-11-release-archived)
3. [Completed Epics (Detailed)](#completed-epics-detailed)
4. [PRD Delta Log Archive (Pre-December 6, 2025)](#prd-delta-log-archive)

---

# Version 1.0 Release (Archived)

*Completed: December 29, 2025*

## 1.0 Exit Criteria - ALL COMPLETE

**Definition:** HaoLine 1.0 means *"a user can reliably analyze, compare, and make deployment decisions about real-world models across major formats, with predictable behavior and clear limitations."*

### 1.0 Checklist (20/20 tasks)

| # | Category | Task | Status |
|---|----------|------|--------|
| 1 | **Testing** | PyTorch→ONNX conversion tests pass | ✅ Done |
| 2 | **Testing** | ONNX analysis produces correct metrics | ✅ Done |
| 3 | **Testing** | TensorRT comparison (`--compare-trt`) works | ✅ Done |
| 4 | **Testing** | Conversion error handling test (42.5.6) | ✅ Done |
| 5 | **Testing** | TF/Keras→ONNX conversion test (42.2.4-5) | ✅ Done |
| 6 | **Testing** | IR invariant: same model via different paths → identical metrics | ✅ Done |
| 7 | **UX** | Disable graph tab for formats without graph (49.2.4) | ✅ Done |
| 8 | **UX** | Show "Convert to ONNX for full analysis" prompt (49.2.5) | ✅ Done |
| 9 | **UX** | Add format tier badge in reports (49.2.6) | ✅ Done |
| 10 | **UX** | Show "Feature unavailable" with upgrade path (49.2.7) | ✅ Done |
| 11 | **TRT Docs** | README: TensorRT limitations section (52.1.1-3) | ✅ Done |
| 12 | **TRT Docs** | README: TensorRT troubleshooting (52.1.6) | ✅ Done |
| 13 | **TRT Docs** | Test graceful degradation when TRT missing (52.3.2) | ✅ Done |
| 14 | **TRT Docs** | CLI error message when TRT missing (52.3.5) | ✅ Done |
| 15 | **README** | Verify all CLI examples work | ✅ Done |
| 16 | **README** | Verify format support claims match reality | ✅ Done |
| 17 | **README** | Remove/update any "coming soon" language | ✅ Done |
| 18 | **Stability** | CI passes on main branch | ✅ Done |
| 19 | **Stability** | No open issues that produce incorrect metrics without warning | ✅ Done |
| 20 | **UX** | Demo model shows same analysis as uploaded models (55.1) | ✅ Done |

### README Accuracy Checklist - ALL VERIFIED

| Section | Claim | Status |
|---------|-------|--------|
| Quick Start | `pip install haoline` works | ✅ Verified |
| Quick Start | `python -m haoline model.onnx --out-html report.html` works | ✅ Verified |
| Quick Start | `python -m haoline compare` works (requires `--eval-metrics`) | ✅ Verified (docs updated) |
| Beginner Guide | Model download example works | ✅ Verified (docs updated) |
| Beginner Guide | `--from-pytorch` with input-shape works | ✅ Verified |
| Web Interface | HF Spaces link is live and functional | ✅ Verified |
| Web Interface | `haoline-web` command works | ✅ Verified |
| Installation | All extras install correctly (`[llm]`, `[full]`, etc.) | ✅ Verified |
| Common Commands | `haoline --list-hardware` works | ✅ Verified |
| Common Commands | `--from-tensorflow` conversion works | ⏭️ Skipped (not in `[full]`) |
| CI/CD Section | `--fail-on` example works | ✅ Verified |
| CI/CD Section | `--decision-report` example works | ✅ Verified |
| Format Support | All listed formats actually load | ✅ Verified |

### What Was NOT Required for 1.0

These were explicitly post-1.0:
- AWS GPU deployment (Epic 51)
- SaaS web app (Epic 10)
- LLM-scale analysis (Epics 26-30)
- Native FLOPs for non-ONNX formats (Epic 49.5)
- Model optimization service (Epics 31-32)
- Model card standards (Epic 47)
- JAX conversion tests (42.2.7-8, 42.3.8-9)
- SafeTensors writer tests (42.6.x)
- GGUF advanced UI (Epic 24.2)

---

# Version 1.2 Release (Archived)

*Completed: January 6, 2026*

## 1.2 Summary - GGUF LLM Details UI + Export Enhancements

**What was delivered:**
- "LLM Details" tab in Streamlit for GGUF models with:
  - Architecture card (model name, architecture, layers, heads, context length, vocab size)
  - Quantization breakdown charts (tensor count and size by quantization type)
  - Interactive VRAM calculator with context length slider
  - GPU compatibility table
  - Tensor explorer with searchable table and CSV export
- Export enhancements:
  - Markdown export includes LLM architecture, VRAM estimates, quantization breakdown
  - HTML export includes styled LLM details section
  - JSON export includes full `gguf_info` object
  - Report version now dynamically references package `__version__`
- CLI GGUF parity (v1.2.1): CLI now supports GGUF files with full LLM details in exports

## Epic 24: GGUF Format - COMPLETE (14/14 tasks)

*Completed: January 6, 2026*

### Story 24.1: GGUF Reader - COMPLETE (6/6)
- Pure Python GGUF header parser (no deps)
- Model metadata extraction (arch, context_length, etc.)
- Per-tensor quantization type extraction
- VRAM estimation

### Story 24.2: GGUF Streamlit UI & Analysis - COMPLETE (8/8)
- `.gguf` in Streamlit file_uploader
- Quantization breakdown chart
- Architecture details display
- VRAM calculator with context length slider
- Tensor-level quantization table
- "LLM Model Details" tab
- Export enhancements (MD/HTML/JSON)
- CLI GGUF parity

---

# Version 1.1 Release (Archived)

*Completed: December 29, 2025*

## 1.1 Summary - CLI Parity Complete

**Goal:** Ensure `cli_typer.py` has 100% feature parity with `_cli_legacy.py`

**What was delivered:**
- All 42 legacy CLI flags ported to new Typer CLI
- Added `--list-cloud` and `--list-conversions` global options
- Added conversion flags: `--from-frozen-graph`, `--tf-inputs`, `--tf-outputs`, `--jax-apply-fn`, `--pytorch-weights`
- Added deployment flags: `--deployment-target`, `--target-latency-ms`, `--target-throughput-fps`
- Added profiling flags: `--no-gpu-metrics`, `--no-bottleneck-analysis`
- Added quantization flags: `--quant-bottlenecks`, `--quant-advice-report`
- All CLI tests pass (29 tests)

---

# Completed Epics (Detailed)

*These epics are 100% complete and moved here for archival purposes. Summary entries remain in BACKLOG.md.*

---

## Epic 1: Environment Setup (COMPLETE - 11/11)

*Completed: December 2025*

- [x] Fork and build ONNX Runtime
- [x] Build Python wheel (`onnxruntime_gpu-1.24.0`)
- [x] Codebase familiarization
- [x] Project scaffolding

*Note: Task "Add to ORT build system" removed - this is our IP, not donating to Microsoft.*

---

## Epic 2: Core Analysis Engine (COMPLETE - 17/17)

*Completed: December 2025*

- [x] ONNX Graph Loader
- [x] Parameter Counting (with shared weights, quantized params)
- [x] FLOP Estimation (Conv, MatMul, Attention)
- [x] Memory Estimation (activations, KV cache)

---

## Epic 3: Pattern Analysis (COMPLETE - 9/9)

*Completed: December 2025*

- [x] Block Detection (Conv-BN-ReLU, Residual, Transformer)
- [x] Risk Heuristics (deep networks, dynamic shapes, oversized layers)

---

## Epic 4: CLI and Output (COMPLETE - 18/18)

*Completed: December 2025*

- [x] CLI Implementation (argparse, progress, error handling)
- [x] JSON Output (schema validation)
- [x] Markdown Output (model cards)
- [x] HTML Report (full parity)

---

## Epic 4B: PyTorch Integration (COMPLETE - 14/14)

*Completed: December 2025*

- [x] PyTorch to ONNX Conversion
- [x] Dataset/Class Metadata Extraction (Ultralytics, output shape inference)

---

## Epic 4C: TensorFlow and Keras Conversion (COMPLETE - 15/15)

*Completed: December 2025*

### Story 4C.1: TensorFlow to ONNX Conversion
- [x] Add `--from-tensorflow` CLI flag with SavedModel path argument
- [x] Implement TensorFlow SavedModel loading
- [x] Integrate tf2onnx conversion with sensible defaults
- [x] Support frozen graph (.pb) files (--from-frozen-graph, --tf-inputs, --tf-outputs)
- [x] Handle conversion errors gracefully
- [x] Add tests for TensorFlow conversion flow (12 tests)

### Story 4C.2: Keras to ONNX Conversion
- [x] Add `--from-keras` CLI flag
- [x] Implement Keras model loading (Sequential, Functional, Subclassed)
- [x] Convert via tf2onnx CLI for robustness
- [x] Support both .h5 and .keras formats
- [x] Add tests for Keras conversion flow

### Story 4C.3: JAX/Flax to ONNX Conversion
- [x] Add `--from-jax` CLI flag
- [x] Implement JAX -> TF SavedModel -> ONNX pipeline via jax2tf
- [x] Support .msgpack, .pkl, .npy params formats
- [x] Support Flax modules via --jax-apply-fn module:function pattern

---

## Epic 5: Visualization Module (COMPLETE - 52/52)

*Completed: December 2025*

### Story 5.1: Chart Infrastructure
- [x] Set up matplotlib with Agg backend
- [x] Create consistent chart styling/theme (ChartTheme dataclass, dark theme)
- [x] Implement asset directory management
- [x] Add graceful fallback when matplotlib unavailable

### Story 5.2: Individual Charts
- [x] Implement operator type histogram
- [x] Implement layer depth profile (cumulative params/FLOPs)
- [x] Implement parameter distribution chart (pie chart)
- [x] Implement FLOPs distribution chart
- [x] Implement complexity summary dashboard (3-panel)

### Story 5.3: Report Integration
- [x] Embed charts in Markdown output
- [x] Add chart captions and descriptions
- [x] Support HTML output with embedded images (base64, single shareable file)
- [x] Support PDF output (Playwright-based, --out-pdf flag)

### Story 5.4: LLM-Scale Pattern Detection
- [x] Detect attention patterns (Q/K/V projections, Softmax, Output proj)
- [x] Detect MLP/FFN patterns (up-proj, activation, down-proj, SwiGLU)
- [x] Detect embedding patterns (token embed, position embed, RoPE/sinusoidal)
- [x] Detect normalization placement (pre-norm vs post-norm)
- [x] Detect repetition - "N identical blocks" -> collapse with xN count
- [x] Add `AttentionHead`, `MLPBlock`, `PositionEncoding`, `MoERouter` types
- [x] Handle MoE (Mixture of Experts) routing patterns (TopK detection)
- [x] Tests with BERT, GPT-2, LLaMA

### Story 5.5: Op Type Icon System and Visual Vocabulary
- [x] Define icon/shape for each op category (23 categories)
- [x] Map all 180 ONNX ops to visual categories (165 mapped)
- [x] Define size scaling function (FLOPs -> node size, log scale)
- [x] Define color mapping (compute intensity, precision, memory)
- [x] Create SVG icon set for embedding in HTML
- [x] Add legend/key to visualization output

### Story 5.6: Edge-Centric Visualization
- [x] Calculate tensor size at every edge (shape x dtype bytes)
- [x] Map edge thickness to tensor size (log scale for LLMs)
- [x] Color edges by precision (fp32=blue, fp16=green, int8=yellow, bf16=purple)
- [x] Highlight memory bottleneck edges (red for top 20%)
- [x] Show tensor shape on hover
- [x] Detect and highlight skip connections (dashed lines)
- [x] Calculate peak memory point in graph (memory profile)
- [x] For attention: detect O(seq^2) edges (is_attention_qk flag)

### Story 5.7: Interactive Hierarchical Graph Visualization
- [x] Build hierarchical graph data structure (Model -> Layers -> Blocks -> Ops)
- [x] Implement D3.js renderer
- [x] Default view: collapsed (Input -> [Block x N] -> Output)
- [x] Click-to-expand: show internal ops of any block
- [x] Pan/zoom for large graphs (d3-zoom)
- [x] Search by op type, layer name, or tensor name
- [x] Export as standalone HTML (self-contained, shareable)
- [x] Integrate with existing HTML report (--include-graph flag)
- [x] Performance: handle 20k+ nodes via virtualization/culling

### Story 5.8: Per-Layer Summary Table
- [x] Create per-layer summary table (LayerSummary, LayerSummaryBuilder)
- [x] Add sortable/filterable table to HTML report
- [x] Click row to highlight in graph visualization
- [x] Export table as CSV (--layer-csv flag)

---

## Epic 6: Hardware Profiles and Compare Mode (COMPLETE - 56/56)

*Completed: December 2025*

### Story 6.1: Hardware Profile System
- [x] Define hardware profile dataclass (HardwareProfile)
- [x] Create comprehensive profile library (30+ profiles)
- [x] Implement profile loading and auto-detection via nvidia-smi
- [x] Add CLI flags (--hardware, --list-hardware, --precision, --batch-size)

### Story 6.2: Hardware Estimates
- [x] Implement VRAM requirement estimation
- [x] Implement theoretical latency bounds
- [x] Estimate compute utilization (roofline-based)
- [x] Identify bottleneck (compute vs memory vs vram)
- [x] Add GPU Saturation metric

### Story 6.3: Compare Mode CLI
- [x] Implement multi-model argument parsing
- [x] Load and validate eval metrics JSONs
- [x] Verify architecture compatibility
- [x] Compute deltas vs baseline

### Story 6.4: Quantization Impact Report
- [x] Generate comparison JSON schema
- [x] Create comparison Markdown table
- [x] Add trade-off analysis section
- [x] Add layer-wise precision breakdown visualization
- [x] Show accuracy vs speedup tradeoff chart
- [x] Display memory savings per layer analysis
- [x] Add engine summary panel
- [x] Show quantization calibration recommendations

### Story 6.5-6.9: Extended Hardware Support
- [x] 40+ GPU variants (H100, A100, V100, RTX series)
- [x] Multi-GPU / Cluster Support
- [x] Cloud Instance Profiles (AWS/Azure/GCP)
- [x] Resolution and Batch Size Impact Analysis
- [x] Steam-style Hardware Requirements Recommendations

### Story 6.10: Multi-Model Comparison Report
- [x] All tasks completed via model_inspect_compare CLI

---

## Epic 7: LLM Integration (COMPLETE - 5/5)

*Completed: December 2025*

### Story 7.1: LLM Summarizer
- [x] Implement API client abstraction
- [x] Create prompt templates
- [x] Generate short summary
- [x] Generate detailed summary
- [x] Handle API failures gracefully

*Note: Story 7.2 (Config File) cancelled - using env vars + .env auto-load instead.*

---

## Epic 8: Testing & CI/CD (COMPLETE - 18/18)

*Completed: December 2025*

- [x] Unit Tests (all modules)
- [x] Integration Tests (CLI end-to-end)
- [x] Documentation (README, inline docs)
- [x] GitHub Actions CI/CD Pipeline

---

## Epic 9: Runtime Profiling (COMPLETE - 22/22)

*Completed: December 2025*

### Story 9.1: Batch Size Benchmarking
- [x] Implement `run_batch_sweep_benchmark()` with ONNX Runtime
- [x] Measure actual latency (p50) per batch size
- [x] Calculate real throughput from measured latency
- [x] Make benchmarking the default (`--no-benchmark` for theoretical)

### Story 9.2: GPU Memory Profiling
- [x] Integrate `pynvml` for GPU memory measurement
- [x] Track VRAM usage during inference
- [x] Measure peak GPU memory per batch size
- [x] Add GPU utilization tracking

### Story 9.3: Per-Layer Profiling
- [x] Enable ONNX Runtime profiling
- [x] Parse profiling JSON output
- [x] Identify slowest layers/operators
- [x] Generate per-layer timing breakdown chart
- [x] Highlight bottleneck layers in graph visualization

### Story 9.4: Bottleneck Detection
- [x] Compare compute time vs memory transfer time
- [x] Classify as compute-bound or memory-bound
- [x] Provide optimization recommendations based on bottleneck
- [x] Show theoretical vs actual performance gap

### Story 9.5: Resolution Benchmarking
- [x] Benchmark actual inference at different resolutions
- [x] Measure real throughput scaling with resolution
- [x] Find optimal resolution for target latency

### Story 9.6: Multi-Input Model Profiling
- [x] Detect all model inputs and their shapes/dtypes
- [x] Generate appropriate dummy inputs based on dtype
- [x] Support common input patterns (text, multimodal)
- [x] Auto-detect sequence length from model
- [x] Handle dynamic axes gracefully

---

## Epic 10B: Standalone Package (COMPLETE - 23/23)

*Completed: December 2025*

### Story 10B.0: Greenfield Extraction
- [x] Create new GitHub repo (standalone, not ORT fork)
- [x] Copy autodoc modules
- [x] Update all imports to standalone package structure
- [x] Remove ORT dependencies
- [x] Copy test fixtures
- [x] Verify all tests pass (229 passed)
- [x] Update README for standalone usage

### Story 10B.1: Python Wheel Packaging
- [x] Create pyproject.toml with proper metadata
- [x] Configure build system (hatchling)
- [x] Define core and optional dependencies
- [x] Add CLI entrypoints
- [x] Test wheel installation
- [x] Publish to TestPyPI
- [x] Publish to PyPI (v0.2.2+)

### Story 10B.2: CI/CD Pipeline
- [x] GitHub Actions workflow for testing
- [x] Black + Ruff linting checks
- [x] mypy type checking
- [x] pytest with coverage
- [x] Auto-publish to PyPI on release tag

### Story 10B.4: Documentation and Branding
- [x] Standalone README.md
- [x] Quickstart examples
- [x] CLI flags documentation
- [x] Architecture overview (Architecture.md)
- [x] Product name: HaoLine (皓线)

---

## Epic 11: Streamlit Web UI (COMPLETE - 17/17)

*Completed: December 2025*

### Story 11.1: Basic Streamlit App
- [x] Create `streamlit_app.py` with file upload widget
- [x] Wire upload to analysis engine
- [x] Display HTML report in Streamlit iframe/component
- [x] Add hardware profile dropdown selector
- [x] Add download buttons (JSON, Markdown, HTML, PDF)

### Story 11.2: Enhanced UI Features
- [x] Modern dark theme with emerald accents
- [x] Hardware dropdown with search and categorization (50+ GPUs)
- [x] LLM summary toggle with API key input
- [x] Full interactive D3.js graph embedded
- [x] FLOPs-based node sizing (log scale)
- [x] Collapsible sidebar in graph
- [x] PDF export functionality
- [x] Model comparison tab
- [x] Session history (stores last 10 analyses)

### Story 11.3: Deployment
- [x] Deploy to Hugging Face Spaces (live)
- [x] Create deployment documentation (DEPLOYMENT.md)
- [x] Set up CI/CD for auto-deploy

### Story 11.4: Sample Model Preloading
*Bundle demo models for quick testing in Streamlit UI.*

- [x] Bundle 3 demo models (MNIST, SqueezeNet, EfficientNet-Lite4)
- [x] Add "Try a demo model" buttons in Streamlit
- [x] Download + analyze demo models on demand

---

## Epic 12: Eval Import & Comparison (COMPLETE - 30/30)

*Completed: December 2025*

### Story 12.1: Base Eval Schema
- [x] Define `EvalResult` base schema
- [x] Define `EvalMetric` schema
- [x] Create `eval_schema.json` for validation
- [x] Add `haoline import-eval` CLI command skeleton

### Story 12.2: Task-Specific Schemas
- [x] Detection schema (mAP@50, mAP@50:95, P/R/F1)
- [x] Classification schema (top-1, top-5 accuracy)
- [x] NLP schema (accuracy, F1, exact_match, BLEU)
- [x] LLM schema (perplexity, mmlu, hellaswag)
- [x] Segmentation schema (mIoU, dice)
- [x] Generic schema (user-defined metrics)

### Story 12.3: Import Adapters
- [x] Ultralytics adapter (YOLO val results)
- [x] HuggingFace evaluate adapter
- [x] lm-eval-harness adapter
- [x] timm adapter
- [x] Generic CSV/JSON adapter
- [x] Auto-detect adapter

### Story 12.4: Merge Eval + Architecture
- [x] Link eval results to model files (by path or hash)
- [x] Create `CombinedReport` dataclass

### Story 12.5: Unified Comparison Report
- [x] Multi-model comparison table
- [x] Add eval metrics to HTML/PDF
- [x] Export comparison as CSV/JSON

### Story 12.6: Deployment Cost Calculator
- [x] Define deployment scenario inputs
- [x] Calculate required hardware tier for latency SLA
- [x] Estimate $/day and $/month for deployment
- [x] Add `--deployment-fps` and `--deployment-hours` CLI flags

### Story 12.7: YOLO Quantization Demo
- [x] Document YOLO quantization workflow
- [x] Train YOLOv8n on roof_damage dataset
- [x] Export to FP32/FP16/INT8 ONNX
- [x] Validate on test set
- [x] Generate comparison report

---

## Epic 18: Universal IR (COMPLETE - 25/25)

*Completed: December 2025*

### Story 18.1: Universal Graph IR
- [x] Design `UniversalGraph` dataclass
- [x] Design `UniversalNode` abstraction
- [x] Design `UniversalTensor` class
- [x] Add source format tracking and round-trip info
- [x] Document IR design decisions in Architecture.md

### Story 18.2: Format Adapter Interface
- [x] Define `FormatAdapter` protocol
- [x] Implement adapter registry and auto-detection
- [x] Refactor ONNX loader into `OnnxFormatAdapter`
- [x] Refactor PyTorch loader into `PyTorchFormatAdapter`
- [x] Unit tests for adapter selection (33 tests)

### Story 18.3: Conversion Matrix
- [x] Define conversion capability enum
- [x] Implement conversion matrix lookup
- [x] Add `--convert-to <format>` CLI flag

### Story 18.4: IR Structural Comparison Tools
- [x] Implement graph structure equality check
- [x] Implement detailed IR diff reporting
- [x] Validate with variant models

### Story 18.5: IR Serialization & Visualization
- [x] IR to JSON serialization
- [x] Graph visualization utility (DOT, PNG)
- [x] CLI integration for graph export

### Story 18.6: IR Integration with Main Pipeline
- [x] Add `universal_graph` field to `InspectionReport`
- [x] Populate UniversalGraph during inspect()
- [x] Add `to_hierarchical()` method
- [x] Update Streamlit app with IR summary
- [x] Enable IR-based comparison
- [x] Test integration end-to-end

---

## Epic 19: SafeTensors Format (COMPLETE - Story 19.1: 6/6 tasks)

*Completed: December 2025*

### Story 19.1: SafeTensors Reader - COMPLETE
*HuggingFace ecosystem, widely used for LLM weights.*

- [x] **Task 19.1.1**: Add safetensors dependency (optional) - in `[formats]` extra
- [x] **Task 19.1.2**: Implement SafeTensorsReader.read() - load tensor dict
- [x] **Task 19.1.3**: Extract metadata (tensor names, shapes, dtypes)
- [x] **Task 19.1.4**: Integrate with analysis pipeline (param counts, memory)
- [x] **Task 19.1.5**: Test with real SafeTensors model (sentence-transformers/all-MiniLM-L6-v2, 22.7M params)
- [x] **Task 19.1.6**: Write unit tests for SafeTensorsReader (8 tests in test_formats.py)

---

## Format Reader Stories (COMPLETE)

*These stories are complete and moved from BACKLOG.md.*

### Story 20.1: CoreML Reader - COMPLETE (7/7)
*Completed: December 2025*

- [x] Add coremltools dependency (optional) - in `[coreml]` extra
- [x] Implement CoreMLReader.read() - load .mlmodel/.mlpackage
- [x] Map CoreML ops to layer info (op_type_counts, precision_breakdown)
- [x] Extract CoreML-specific metadata (compute units, iOS version)
- [x] Integrate with analysis pipeline
- [x] Test with real CoreML model (in test_format_readers.py, CI on Linux)
- [x] Write unit tests for CoreMLReader (6 tests in test_formats.py)

### Story 23.1: OpenVINO Reader - COMPLETE (5/6)
*Completed: December 2025 (1 real-model test pending)*

- [x] Add openvino dependency (optional) - in `[openvino]` extra
- [x] Implement OpenVINOReader.read() - load .xml/.bin
- [x] Map OpenVINO ops to layer_type_counts
- [x] Extract precision breakdown
- [ ] Test with real OpenVINO model (.xml + .bin) - pending
- [x] Write unit tests for OpenVINOReader (5 tests in test_formats.py)

### Story 24.1: GGUF Reader - COMPLETE (6/6)
*Completed: December 2025*

- [x] Implement GGUF header parser (pure Python, no deps)
- [x] Extract model metadata (arch, context_length, etc.)
- [x] Extract quantization type per tensor
- [x] Estimate memory footprint (VRAM estimation)
- [x] Test with real GGUF model (TinyLlama-1.1B Q2_K, 1.1B params, 458MB)
- [x] Write unit tests for GGUFReader (8 tests in test_formats.py)

---

## Epic 25: Privacy and Trust Architecture (COMPLETE - 9/9)

*Completed: December 2025*

### Story 25.1: Local-First Architecture
- [x] Document "model never leaves your machine" guarantee
- [x] Audit code for network calls
- [x] Add `--offline` CLI flag
- [x] Create architecture diagram showing data flow

### Story 25.2: Output Controls
- [x] Add `--redact-names` flag (anonymize layer/tensor names)
- [x] Add `--summary-only` flag (stats only, no graph structure)
- [x] Document what information each output format reveals

### Story 25.3: Enterprise Trust Documentation
- [x] Write Privacy Policy / Data Handling document (PRIVACY.md)
- [x] Document open-source audit path

---

## Epic 33: QAT & Quantization Linters (COMPLETE - 41/41)

*Completed: December 2025*

### Story 33.1: Quantization-Unfriendly Op Detection
- [x] Build list of quantization-unfriendly ops
- [x] Detect dynamic shapes in problematic positions
- [x] Flag ops with no ONNX quantization support
- [x] Identify ops that typically cause accuracy drops
- [x] Generate severity-ranked warning list

### Story 33.2: QAT Graph Validation
- [x] Detect missing fake-quantization nodes
- [x] Check for inconsistent fake-quant placement
- [x] Validate per-tensor vs per-channel consistency
- [x] Flag suspiciously wide activation ranges
- [x] Detect inconsistent scales/zero points across residuals

### Story 33.3: Quantization Readiness Score
- [x] Define scoring rubric
- [x] Calculate per-layer quantization risk scores (0-100)
- [x] Aggregate into overall readiness score
- [x] Generate "problem layers" list with reasons
- [x] Add `--lint-quantization` CLI flag

### Story 33.4: Actionable Recommendations
- [x] Recommend keeping sensitive layers at FP16
- [x] Suggest fake-quant insertion points for QAT
- [x] Recommend op substitutions
- [x] Suggest per-channel vs per-tensor
- [x] Create `QuantizationAdvisor` with LLM support
- [x] Generate architecture-specific strategy
- [x] Provide deployment-target-aware recommendations
- [x] Generate step-by-step QAT workflow
- [x] Estimate expected accuracy loss
- [x] Generate QAT Readiness Report
- [x] Integrate with compare mode
- [x] Add to Streamlit UI
- [x] Add `--quant-llm-advice` CLI flag

### Story 33.5: CLI & Streamlit Integration
- [x] Add `--lint-quantization` flag
- [x] Add `--quant-report PATH`
- [x] Include quant lint in `--out-json`
- [x] Add quant section to `--out-html`
- [x] Add `--quant-report-html`
- [x] Add "Quantization Analysis" checkbox in Streamlit
- [x] Display readiness score with letter grade
- [x] Show severity-ranked warnings with icons
- [x] Display op breakdown chart
- [x] Show problem layers table
- [x] Add QAT validation results section
- [x] Add "Download Quant Report" button

---

## Epic 39: Pydantic Schema Migration (COMPLETE - 12/12)

*Completed: December 2025*

### Story 39.1: Core Model Migration
- [x] Add `pydantic>=2.0` to core dependencies
- [x] Auto-generate Pydantic models from JSON Schema
- [x] Fix Pydantic v2 compatibility

### Story 39.2: Schema Cleanup
- [x] Update `validate_report()` to use Pydantic validation
- [x] Update `get_schema()` to return Pydantic-generated schema
- [x] Add `validate_with_pydantic()`
- [x] Export schema for external consumers
- [x] Update tests to use Pydantic validation

### Story 39.3: Eval Schema Migration
- [x] Convert `EvalMetric` to Pydantic model
- [x] Convert `EvalResult` variants to Pydantic
- [x] Convert `CombinedReport` to Pydantic model
- [x] Adapters work with Pydantic models

---

## Epic 40: Full Pydantic Dataclass Migration (COMPLETE - 64/64)

*Completed: December 2025 (v0.5.0, hotfix v0.8.4)*

Complete migration from Python dataclasses to Pydantic BaseModel across the entire codebase.

### Story 40.1: Core Report Models
- [x] Convert `ModelMetadata` to Pydantic `BaseModel`
- [x] Convert `GraphSummary` to Pydantic `BaseModel`
- [x] Convert `DatasetInfo` to Pydantic `BaseModel`
- [x] Convert `InspectionReport` to Pydantic `BaseModel`
- [x] Replace `to_dict()` with Pydantic `model_dump()`
- [x] Replace `to_json()` with Pydantic `model_dump_json()`

### Story 40.2: Analyzer Models
- [x] Convert `ParamCounts` to Pydantic `BaseModel`
- [x] Convert `FlopCounts` to Pydantic `BaseModel`
- [x] Convert `MemoryEstimates` to Pydantic `BaseModel`
- [x] Update `MetricsEngine` to return Pydantic models
- [x] Handle multiple eval runs per model
- [x] Validate eval task matches model type

### Story 40.3: Hardware and Risk Models
- [x] Convert `HardwareProfile` to Pydantic `BaseModel`
- [x] Convert `HardwareEstimates` to Pydantic `BaseModel`
- [x] Convert `RiskSignal` to Pydantic `BaseModel`
- [x] Convert `Block` and pattern types to Pydantic

### Story 40.4: Schema Consolidation
- [x] Consolidate schema validation to use report.py models
- [x] Update all imports across codebase
- [x] Update CLI to work with Pydantic models
- [x] Update Streamlit app to work with Pydantic models
- [x] Update all unit tests for Pydantic models

### Story 40.5: Format Readers & All Remaining Classes
- [x] Convert `formats/*.py` (11 classes)
- [x] Convert `report_sections.py` (16 classes)
- [x] Convert `quantization_linter.py`, `quantization_advisor.py` (8 classes)
- [x] Convert `compare*.py`, `eval/*.py` (12 classes)
- [x] Convert `edge_analysis.py`, `hierarchical_graph.py` (6 classes)
- [x] Convert remaining misc classes (5 classes)

### Story 40.6: LLM Response Normalization (v0.8.4 hotfix)
*Fix Pydantic validation errors when LLM returns nested/malformed structures.*

- [x] Task 40.6.1: Fix `_normalize_runtime_recs` to handle deeply nested LLM responses
- [x] Task 40.6.2: Fix `_normalize_str_list` to handle all LLM edge cases (layer_names dict, etc.)
- [x] Task 40.6.3: Add `_extract_string_from_nested` helper for recursive extraction
- [x] Task 40.6.4: Add unit tests for normalization functions (35 tests)
- [x] Task 40.6.5: Add integration tests reproducing actual production failures
- [x] Task 40.6.6: Bump version to 0.8.4, run lints, commit and release

---

## Epic 41: Standardized Reporting (COMPLETE - 50/50)

*Completed: December 22, 2025*

### Story 41.7: Output Parity Gap Closure (6/6)
*Address remaining gaps between CLI and Streamlit capabilities.*

- [x] **Task 41.7.1**: Document PyTorch upload limitation in Streamlit (needs local torch install)
- [x] **Task 41.7.2**: Fix PDF export in Streamlit (gracefully shows "CLI Only" badge when unavailable)
- [x] **Task 41.7.3**: Add "CLI Only" badges to Streamlit for features requiring local install
- [x] **Task 41.7.4**: Add "Export as CLI command" button (`generate_cli_command()` function, 11 tests)
- [x] **Task 41.7.5**: Ensure JSON report schema identical between CLI and Streamlit (both use `report.to_dict()`)
- [x] **Task 41.7.6**: Add format support comparison table to docs

### Story 41.1: Audit Current Report Differences
- [x] Create comparison matrix: CLI HTML vs Streamlit
- [x] List visualizations present in CLI but missing in Streamlit
- [x] List analysis sections present in CLI but missing in Streamlit
- [x] Document styling/theme differences
- [x] Identify reusable components vs duplicated code
- [x] Audit completed features not surfaced in UI

### Story 41.2: Unified Report Components
- [x] Extract report sections into reusable functions (report_sections.py)
- [x] Add all CLI visualizations to Streamlit
- [x] Add parameter distribution visualization
- [x] Add layer-by-layer breakdown table
- [x] Add KV Cache section to Streamlit
- [x] Add Precision Breakdown section
- [x] Add Memory Breakdown by Op Type
- [x] Add Bottleneck Analysis section

### Story 41.3: Enhanced Streamlit Visualizations
- [x] Add FLOPs breakdown chart
- [x] Add memory usage timeline/waterfall
- [x] Add layer statistics table with sorting
- [x] Add architecture pattern summary
- [x] Ensure consistent color scheme
- [x] Add System Requirements section (Steam-style)
- [x] Add Deployment Cost Calculator
- [x] Add Batch Size Sweep results view
- [x] Add Resolution Sweep results view
- [x] Add Per-Layer Timing breakdown
- [x] Add Cloud Instance selector
- [x] Add Privacy Controls toggle

### Story 41.4: CLI-Streamlit Parity Matrix
- [x] Add batch size input control
- [x] Add "Run Benchmark" button
- [x] Add deployment cost panel
- [x] Add cloud instance dropdown
- [x] Add GPU count spinner
- [x] Add deployment target selector
- [x] Add per-layer table with CSV download
- [x] Add privacy toggles
- [x] Add Universal IR export button
- [x] Update CLI parity matrix

### Story 41.5: LLM Prompt Enhancement
- [x] Audit current LLM prompt for missing data fields
- [x] Add KV Cache info to LLM prompt
- [x] Add Precision Breakdown to LLM prompt
- [x] Add Memory Breakdown to LLM prompt
- [x] Add extended Hardware Estimates
- [x] Add System Requirements to LLM prompt
- [x] Add Bottleneck Analysis recommendations
- [x] Test LLM summary quality

---

## Epic 50: CLI Modernization (COMPLETE - 18/18)

*Completed: December 23, 2025 (v0.9.4-v0.9.7)*

Migrated CLI from argparse to Typer. Added dependency prompting and user-friendly error messages.

### Story 50.1: Typer Migration - COMPLETE (6/6)
- [x] Add typer dependency
- [x] Convert main CLI to Typer app (`cli_typer.py`)
- [x] Convert subcommands (compare, web, check-install)
- [x] Add rich formatting for help text
- [x] Add shell completion support (via Typer)
- [x] Preserve backwards compatibility (legacy `_cli_legacy.py` kept)

### Story 50.2: Dependency Prompting - COMPLETE (6/6)
- [x] Detect missing optional dependencies at runtime (`_check_module()`)
- [x] Show friendly "pip install haoline[extra]" suggestions
- [x] Add `check-deps` command to list missing features
- [x] Group dependencies by feature (formats, llm, viz, gpu)
- [x] Add confirmation prompts for auto-install (`check-deps --install`)
- [x] Cache dependency check results (N/A - check is fast enough)

### Story 50.3: Error Messages - COMPLETE (6/6)
- [x] Replace tracebacks with user-friendly messages
- [x] Add `--verbose` flag for full tracebacks
- [x] Suggest fixes for common errors (`_get_error_suggestion()`)
- [x] Add progress bars for long operations (`console.status()` spinners)
- [x] Color-code warnings vs errors
- [x] Add `--quiet` flag for scripting

**Key Files Created:**
- `src/haoline/cli_typer.py` - New Typer-based CLI
- `src/haoline/_cli_legacy.py` - Renamed from `cli.py`, deprecated
- `src/haoline/tests/test_cli_typer.py` - 16 CLI tests

---

## Epic 53: Installation UX (COMPLETE - 15/15)

*Completed: December 22, 2025 (v0.9.4)*

Fixed first-run experience issues with PATH problems on user-level pip installs.

### Story 53.1: Module Invocation Support - COMPLETE (5/5)
- [x] Add `__main__.py` for `python -m haoline` support
- [x] Ensure all CLI entry points work via module invocation
- [x] Update README with `python -m haoline` as primary method
- [x] Add `python -m haoline web` subcommand
- [x] Add `python -m haoline compare` subcommand

### Story 53.2: Installation Diagnostics - COMPLETE (5/5)
- [x] Add `python -m haoline check-install` command
- [x] Check if haoline scripts are on PATH, report if not
- [x] Detect user vs system install and explain implications
- [x] Show which optional extras are installed
- [x] Suggest PATH fix commands for Windows/Linux/macOS

### Story 53.3: Documentation & First-Run - COMPLETE (5/5)
- [x] Add "Troubleshooting Installation" section to README
- [x] Document Windows PATH fix
- [x] Document Linux/macOS PATH fix
- [x] Add installation verification command to Quick Start
- [x] Update `generate_cli_command()` to use `python -m haoline` format

---

## Epic 55: Demo Model Parity - COMPLETE (2/2 tasks)

*Completed: December 24, 2025*

**Problem:** Demo models used a separate code path with simplified rendering (~50% functionality).

**Solution:** Made demo models flow through the SAME code path as uploaded files:
- Demo model download creates a `DemoUploadedFile` object mimicking `UploadedFile`
- This is stored in session state and used by the normal uploaded file handling code
- Single code path = no parity issues, no duplication

### Story 55.1: Demo-Upload Parity - COMPLETE
- [x] Create `DemoUploadedFile` class that mimics `UploadedFile` interface
- [x] Route demo models through uploaded file code path (no separate rendering)

---

## Epic 54: CI/CD Integration (COMPLETE - 23/23)

*Completed: December 23, 2025 (v0.9.7+)*

Made HaoLine a gatekeeper in ML pipelines with threshold-based failure, GitHub Actions workflow, and Decision Report audit trails.

### Story 54.1: Threshold-Based Failure (`--fail-on`) - COMPLETE (9/9)
*Add flags that cause non-zero exit when thresholds are exceeded.*

- [x] Add `--fail-on` flag to `compare` command (accepts key=threshold pairs)
- [x] Implement threshold parsing (e.g., `latency_increase=10%`, `memory_increase=20%`)
- [x] Add `latency_increase` threshold check (compare estimated latency)
- [x] Add `memory_increase` threshold check (compare peak activation memory)
- [x] Add `param_increase` threshold check (compare total parameters)
- [x] Add `new_risk_signals` threshold check (fail if new high-severity risks appear)
- [x] Exit with code 1 if any threshold violated, 0 otherwise
- [x] Print clear failure message with what threshold was violated
- [x] Add `--fail-on` tests to test_cli_typer.py (6 tests)

### Story 54.2: GitHub Actions Workflow - COMPLETE (5/5)
*Provide a ready-to-use workflow for model validation in PRs.*

- [x] Create `.github/examples/model-check.yml` workflow template
- [x] Workflow: checkout, install haoline, run compare with --fail-on
- [x] Workflow: post comparison summary as PR comment
- [x] Add workflow documentation to README (CI/CD Integration section)
- [x] Test workflow in a sample repo (tested via HaoLine's own CI)

### Story 54.3: Decision Report Format - COMPLETE (9/9)
*Create an audit-trail format that captures what was compared, what constraints were applied, and what was decided.*

- [x] Define `DecisionReport` schema (models compared, constraints, recommendations)
- [x] Add `--decision-report PATH` flag to compare command
- [x] Capture: models compared (paths, MD5 hashes, timestamps, file sizes)
- [x] Capture: constraints applied (thresholds, hardware profile, precision)
- [x] Capture: results (pass/fail for each constraint, risk signals)
- [x] Capture: recommendations (from quantization advisor, hardware estimator)
- [x] Output as JSON (machine-readable audit trail)
- [x] Output as Markdown (human-readable summary)
- [x] Add timestamp and HaoLine version to report

**Key Files Created:**
- `.github/examples/model-check.yml` - Full-featured GitHub Actions workflow template
- `src/haoline/cli_typer.py` - `_build_decision_report()`, `_decision_report_to_markdown()`
- `src/haoline/tests/test_cli_typer.py` - 10 new tests for --fail-on and decision reports

---

## Epic 56: CLI Parity - Typer Migration Completion (COMPLETE - 42/42)

*Completed: December 29, 2025 (v1.1.0)*

Ported all remaining flags from legacy argparse CLI to new Typer CLI for 100% feature parity.

### Story 56.1: Conversion Flags - COMPLETE (13/13)
- [x] `--keep-onnx` - Save converted ONNX to path
- [x] `--opset-version` - ONNX opset version for exports
- [x] `--from-tensorflow` - Convert TensorFlow SavedModel
- [x] `--from-keras` - Convert Keras .h5/.keras model
- [x] `--from-tflite` - Convert TFLite model
- [x] `--from-frozen-graph` - Convert TF frozen graph
- [x] `--tf-inputs` / `--tf-outputs` - Tensor names for frozen graph
- [x] `--from-jax` - Convert JAX/Flax model
- [x] `--jax-apply-fn` - JAX apply function path
- [x] `--list-conversions` - List available conversions
- [x] `--pytorch-weights` - Original PyTorch weights path

### Story 56.2: Output & Export Flags - COMPLETE (7/7)
- [x] `--html-graph` - Standalone interactive graph HTML
- [x] `--layer-csv` - Per-layer metrics CSV export
- [x] `--include-layer-table` - Layer table in HTML report
- [x] `--export-ir` - Export Universal IR JSON
- [x] `--export-graph` - Export graph as DOT/PNG
- [x] `--graph-max-nodes` - Max nodes in graph visualization
- [x] `--assets-dir` - Directory for plot files

### Story 56.3: Hardware & Deployment Flags - COMPLETE (11/11)
- [x] `--cloud` - Cloud instance type
- [x] `--list-cloud` - List cloud instances
- [x] `--system-requirements` - Steam-style requirements
- [x] `--sweep-batch-sizes` - Batch size sweep analysis
- [x] `--sweep-resolutions` - Resolution sweep analysis
- [x] `--input-resolution` - Override input resolution
- [x] `--deployment-target` - Edge/local/cloud target
- [x] `--deployment-fps` - Target FPS for cost calc
- [x] `--deployment-hours` - Hours/day for cost calc
- [x] `--target-latency-ms` - Latency target
- [x] `--target-throughput-fps` - Throughput target

### Story 56.4: Profiling & Privacy Flags - COMPLETE (9/9)
- [x] `--no-profile` - Disable ONNX Runtime profiling
- [x] `--profile-runs` - Number of profiling runs
- [x] `--no-gpu-metrics` - Disable GPU metrics
- [x] `--no-bottleneck-analysis` - Disable bottleneck analysis
- [x] `--redact-names` - Anonymize layer names
- [x] `--summary-only` - Aggregate stats only
- [x] `--offline` - Disable network access
- [x] `--progress` - Show progress indicators
- [x] `--log-level` - Logging verbosity

### Story 56.5: TensorRT & Quantization Flags - COMPLETE (6/6)
- [x] `--compare-trt` - Compare with TensorRT engine
- [x] `--quant-bottlenecks` - Quantization bottleneck analysis
- [x] `--quant-report` - Quantization report (Markdown)
- [x] `--quant-report-html` - Quantization report (HTML)
- [x] `--quant-llm-advice` - LLM quantization advice
- [x] `--quant-advice-report` - QAT readiness report

**Key Files Modified:**
- `src/haoline/cli_typer.py` - Added 42 new flags with full implementations

---

# PRD Delta Log Archive

*Historical changelog entries moved from PRD.md. These document the evolution of the project.*

## December 2025 Changelog

| Date | Section | Change | Reason |
|------|---------|--------|--------|
| Dec 11, 2025 | Release | v0.8.4: Fixed Pydantic validation errors in QuantizationAdvice when LLM returns nested structures; added robust normalization functions with 35 tests | Production bug fix |
| Dec 11, 2025 | Release | v0.8.1: Streamlit auto-convert to ONNX (PyTorch input-shape prompt, TFLite via tflite2onnx, CoreML via coremltools), backlog trimmed to tasks-only, docs updated | Deployment parity + clarity |
| Dec 2025 | Initial | Created unified PRD from starter pack + visualization extension | Consolidation |
| Dec 2025 | Structure | Split backlog into BACKLOG.md, brainlift into BRAINLIFT.md | Context window optimization |
| Dec 2, 2025 | 4.3 | Scaffolding complete: `tools/python/util/autodoc/` with analyzer, patterns, risks, report modules | Following ORT patterns |
| Dec 2, 2025 | Risk Signals | Added minimum thresholds for risk signals | Common sense |
| Dec 2, 2025 | README | Updated README.md to match actual implementation | Documentation accuracy |
| Dec 2, 2025 | Testing | Created comprehensive unit test suite | Code quality |
| Dec 2, 2025 | CI/CD | Added `.github/workflows/autodoc-ci.yml` | Automated quality gates |
| Dec 2, 2025 | Visualization | Added `visualizations.py` with matplotlib, 4 chart types, 17 tests | Epic 5 complete |
| Dec 2, 2025 | Build | C++ ONNX Runtime build complete with CUDA provider | Environment setup |
| Dec 2, 2025 | LLM | Added `llm_summarizer.py` with OpenAI integration | Epic 7 complete |
| Dec 2, 2025 | Hardware | Adding GPU Saturation metric | Better hardware insight |
| Dec 2, 2025 | PyTorch | Added PyTorch-to-ONNX conversion | Epic 4B implementation |
| Dec 2, 2025 | Attention FLOPs | Added _estimate_attention_flops() | Task 2.3.3 complete |
| Dec 2, 2025 | KV Cache | Added KV cache estimation for transformers | Task 2.4.3 complete |
| Dec 2, 2025 | Memory Breakdown | Added MemoryBreakdown dataclass | Task 2.4.4 complete |
| Dec 2, 2025 | Progress | Added --progress CLI flag | Task 4.1.3 complete |
| Dec 2, 2025 | Risk Thresholds | Added RiskThresholds dataclass | Task 3.2.5 complete |
| Dec 2, 2025 | HTML Parity | Added Operator Distribution, KV Cache, Memory Breakdown to HTML | Story 4.4 complete |
| Dec 2, 2025 | JSON Schema | Added schema.py with Draft 7 JSON schema | Task 4.2.2 complete |
| Dec 2, 2025 | Shared Weights | Added fractional weight attribution | Task 2.2.4 edge case 1 |
| Dec 2, 2025 | Quantized Params | Added quantization detection | Task 2.2.4 edge case 2 |
| Dec 2, 2025 | Tests | Added 8 new tests for shared weights and quantization | Task 2.2.4 complete |
| Dec 2, 2025 | GPU Variants | Added 50+ GPU profiles | Story 6.5 complete |
| Dec 2, 2025 | Multi-GPU | Added MultiGPUProfile dataclass, DGX profiles | Story 6.6 complete |
| Dec 2, 2025 | Cloud | Added CloudInstanceProfile, 17 cloud instances | Story 6.7 complete |
| Dec 2, 2025 | CLI | Added --gpu-count, --cloud, --list-cloud, --out-pdf | CLI enhancements |
| Dec 2, 2025 | PDF | Added pdf_generator.py with Playwright | Task 5.3.4 complete |
| Dec 2, 2025 | ML Feedback | Added Graph Viz, Per-Layer Summary, HW Recommendations | ML Engineer feedback |
| Dec 2, 2025 | Backlog | Added Epic 4C, 10, 10B, Stories 5.4-5.5, 6.4-6.9 | Feature roadmap |
| Dec 3, 2025 | Distribution | **PIVOT**: Greenfield standalone package | Distribution was blocked |
| Dec 3, 2025 | Priority | Reordered epics: P0 = Standalone + Streamlit | Ship usable software first |
| Dec 3, 2025 | Streamlit | Added Section 14.9 with Streamlit Web UI spec | Maximize accessibility |
| Dec 3, 2025 | Inference | Added Section 15: Inference Platform | Platform-first approach |
| Dec 3, 2025 | Backlog | Added Epic 12: Inference Platform (24 tasks) | Extensible architecture |
| Dec 3, 2025 | Future | Added Section 16: MLOps Platform Vision | Document vision |
| Dec 3, 2025 | Backlog | Added Epics 18-25: Universal IR, format adapters | Format-agnostic vision |
| Dec 3, 2025 | Git | Merged feature/onnx-autodoc to main branch | This is our IP |
| Dec 3, 2025 | Epic 4C | **COMPLETE**: TensorFlow/Keras/JAX conversion | ONNX as universal hub |
| Dec 3, 2025 | CI/CD | Removed 42 Microsoft ORT workflows | Avoid spam from fork CI |
| Dec 3, 2025 | Epic 5 | Expanded visualization for LLM-scale | Handle 70B+ param models |
| Dec 3, 2025 | Section 17 | Added LLM-Scale Analysis section | Gap analysis for large models |
| Dec 3, 2025 | Epics 26-30 | Added quantization, attention, memory, sparse, LLM deployment | Complete LLM analysis |
| Dec 3, 2025 | Section 18 | Added Model Optimization Service | Automated quantization |
| Dec 3, 2025 | Epics 31-32 | Added Quantization Service, Model Optimization | Optimization platform |
| Dec 3, 2025 | Epic 5 | **COMPLETE**: All 52 tasks done | Full visualization suite |
| Dec 3, 2025 | Story 6.3 | **COMPLETE**: Compare Mode CLI | Quantization impact analysis |
| Dec 3, 2025 | Story 6.4 | **COMPLETE**: Quantization Impact Report | TRT EngineXplorer-inspired |
| Dec 3, 2025 | Story 6.8 | **COMPLETE**: Resolution/Batch Impact Analysis | Smart resolution scaling |
| Dec 3, 2025 | Section 9.4 | Added Runtime Profiling | Real measurements |
| Dec 3, 2025 | Bug Fixes | Fixed VRAM calculation, throughput model, pie charts, tooltips | Integration test findings |
| Dec 4, 2025 | Epic 9.6 | **COMPLETE**: Multi-input model profiling | LLM profiling foundation |
| Dec 4, 2025 | Epic 22 | Expanded TensorRT Engine Introspection | TRT Engine Explorer-inspired |
| Dec 4, 2025 | Epics 33-35 | Added QAT Linters, Activation Visualization, TRT Graph UX | ML engineer feedback |
| Dec 4, 2025 | **HaoLine** | **EXTRACTED** to standalone repo | Epic 10B.0-10B.2 complete |
| Dec 4, 2025 | **PyPI** | **PUBLISHED** haoline v0.2.2 to PyPI | Epic 10B complete |
| Dec 4, 2025 | Epic 7 | **COMPLETE**: LLM Integration | Simpler approach |
| Dec 4, 2025 | Branding | Fixed etymology: 皓 (hao, "bright/luminous") | Correct Chinese character |
| Dec 4, 2025 | README | Complete rewrite for beginners | AI agent friendly documentation |
| Dec 4, 2025 | v0.2.3 | Added CLI Reference section to README | Complete standalone package |
| Dec 4, 2025 | **Epic 11** | **Streamlit MVP Complete** (12/17 tasks) | Demo-ready web interface |
| Dec 4, 2025 | **Story 11.2** | **COMPLETE** (14/16 tasks): Model comparison, session history | Full-featured web UI |
| Dec 4, 2025 | **Epics 19-24** | **Format Readers Implemented**: GGUF, SafeTensors, TFLite, CoreML, OpenVINO | Expanded format support |
| Dec 4, 2025 | **Epic 12** | Added eval CLI, schemas, adapters, YOLO workflow guide | Eval import foundation |
| Dec 4, 2025 | **Epic 25** | **COMPLETE** (9/9 tasks): Privacy architecture | Enterprise trust features |
| Dec 4, 2025 | **Epic 12** | Added GenericEvalResult, CombinedReport, adapters | Eval import 13/36 |
| Dec 4, 2025 | **Epic 11** | **COMPLETE** (17/17 tasks): HuggingFace Spaces deployed | Public demo available |
| Dec 4, 2025 | **Epic 39** | **COMPLETE** (12/12 tasks): Pydantic schema migration | Type-safe validation |
| Dec 4, 2025 | **Epic 40** | Created from Epic 39 future tasks | Pydantic migration plan |
| Dec 5, 2025 | **Epic 18** | **COMPLETE** (19/19 tasks): Universal IR | Format-agnostic foundation |
| Dec 5, 2025 | **Epic 41** | In Progress: Standardized Reporting audit | CLI-UI parity |
| Dec 5, 2025 | **Epic 42** | Created: Format Conversion Testing | Quality assurance |
| Dec 5, 2025 | **Story 41.2** | 8/11 tasks complete: report_sections.py | CLI-Streamlit parity |
| Dec 5, 2025 | **Story 41.5** | 6/8 tasks complete: LLM prompt enhanced | AI summaries improved |
| Dec 6, 2025 | **Epic 41** | **COMPLETE** (43/44 tasks) | Full CLI-Streamlit parity |
| Dec 6, 2025 | **v0.3.0** | Major release with CLI-Streamlit parity | Full-featured web UI |
| Dec 11, 2025 | **v0.8.0** | Streamlit Layer/Quant tabs, uploader covers TFLite/CoreML/OpenVINO/GGUF, clarified format tiers, `--lint-quant` alias; HF Spaces redeploy | UI parity + format UX |

---

## Epic 22: TensorRT Engine Introspection (COMPLETE - 50/50)

*Completed: December 6, 2025 (v0.7.2)*

Deep analysis of NVIDIA TensorRT compiled engines. Inspired by TRT Engine Explorer.

### Story 22.1: Engine File Loader [Phase 1] - COMPLETE (7/7)
- [x] Add `tensorrt` extra to pyproject.toml
- [x] Create `TRTEngineReader` class
- [x] Implement engine deserialization
- [x] Extract engine metadata (TRT version, build flags)
- [x] Handle GPU arch/TRT version compatibility checks
- [x] Support `.engine` and `.plan` extensions
- [x] Add `is_tensorrt_file()` and `is_available()` helpers

### Story 22.2: Fused Graph Reconstruction [Phase 2] - COMPLETE (6/6)
- [x] Extract layer list (names, types, shapes)
- [x] Identify fused operations (Conv+BN+ReLU → single kernel)
- [x] Detect removed/optimized-away layers
- [x] Extract kernel/tactic substitutions
- [x] Parse timing cache if present
- [x] Identify precision per layer (FP32/FP16/INT8/TF32)

### Story 22.3: ONNX ↔ TRT Diff View [Phase 3] - COMPLETE (6/6)
- [x] Map TRT layers back to ONNX nodes
- [x] Highlight fused operations
- [x] Show precision auto-selection decisions
- [x] Visualize layer rewrites (FlashAttention, GELU, LayerNorm)
- [x] Display shape changes (dynamic → static)
- [x] Generate side-by-side HTML comparison

### Story 22.4: TRT Performance Metadata Panel [Phase 4] - COMPLETE (6/6)
- [x] Extract per-layer latency from profiling data
- [x] Show workspace size allocation per layer
- [x] Display kernel/tactic selection choices
- [x] Identify memory-bound vs compute-bound layers
- [x] Show layer timing breakdown chart
- [x] Extract device memory footprint

### Story 22.5: TRT Engine Summary Block [Phase 1] - COMPLETE (4/4)
- [x] Generate engine overview
- [x] Show optimization summary
- [x] Display hardware binding info
- [x] List builder configuration

### Story 22.6: ONNX vs TRT Comparison Mode [Phase 3] - COMPLETE (5/5)
- [x] Add `--compare-trt` CLI support
- [x] Compute layer count delta
- [x] Show precision changes
- [x] Generate comparison report (JSON/MD/HTML)
- [x] Visualize memory reduction

### Story 22.7: CLI & Streamlit Integration [Phase 1] - COMPLETE (8/8)
- [x] Register TensorRT format detection
- [x] Add `.engine`/`.plan` to CLI
- [x] Add to Streamlit file_uploader
- [x] Create TRT-specific report sections
- [x] Add TensorRT Analysis tab
- [x] Handle graceful degradation
- [x] Update HuggingFace Spaces
- [x] Write unit tests (9 tests)

### Story 22.8: Quantization Bottleneck Analysis [Phase 4] - COMPLETE (8/8)
- [x] Detect failed fusion zones
- [x] Group consecutive FP32 bottleneck zones
- [x] Add per-layer quant status indicators
- [x] Estimate quant gap vs ideal
- [x] Generate Quantization Fusion Summary panel
- [x] Add `--quant-bottlenecks` CLI flag
- [x] Add bottleneck heatmap to Streamlit
- [x] Parse timing cache for actual timings

**Key Classes Added:**
- `TRTEngineReader`, `TRTEngineInfo`, `TRTLayerInfo`, `TRTBindingInfo`
- `TRTPerformanceMetadata`, `TRTBuilderConfig`
- `QuantBottleneckAnalysis`, `FailedFusionPattern`, `BottleneckZone`
- `LayerRewrite`, `TRTComparisonReport`, `TRTONNXComparator`
- `generate_timing_chart()`, `generate_bound_type_chart()`, `generate_comparison_html()`

---

## Epic 49: Format Capability Matrix (Reference)

*Archived from BACKLOG.md - detailed capability matrix moved here to reduce context window usage.*

### Format Capability Matrix

| Format | Graph | Params | FLOPs | Memory | Interactive Map | Quant Info | Convert to ONNX | ONNX Compare |
|--------|-------|--------|-------|--------|-----------------|------------|-----------------|--------------|
| **ONNX** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | N/A (native) | N/A |
| **PyTorch** | ✅ via export | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ torch.onnx | N/A |
| **TensorRT** | ✅ fused | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ | ✅ `--compare-trt` |
| **TFLite** | ✅ | ✅ | ❓ | ✅ | ✅ | ✅ | ⚠️ lossy | ❌ |
| **CoreML** | ✅ layers | ✅ | ❓ | ❓ | ✅ | ❓ | ⚠️ lossy | ❌ |
| **OpenVINO** | ✅ | ✅ | ❓ | ❓ | ✅ | ✅ | ⚠️ lossy | ❌ |
| **GGUF** | ❌ metadata | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ |
| **SafeTensors** | ❌ weights | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ (needs arch) | ❌ |

**Tier System:**
- **Tier 1 (Full)**: ONNX, PyTorch - all metrics, interactive graph
- **Tier 1.5 (Optimized)**: TensorRT - fused graph, ONNX comparison, precision breakdown (requires NVIDIA GPU)
- **Tier 2 (Graph)**: TFLite, CoreML, OpenVINO - graph structure, most metrics
- **Tier 3 (Metadata)**: GGUF - architecture metadata, no graph
- **Tier 4 (Weights)**: SafeTensors - weights only, needs external architecture

---

## Epic 26: Advanced Quantization Analysis (COMPLETE - 12/15 tasks)

*Completed: January 6, 2026 (v1.4.0)*

Modern LLMs use complex quantization beyond simple int8/fp16.

**Module:** `src/haoline/quantization_analysis.py`
**CLI Flags:** `--quant-analysis`, `--quant-analysis-json`
**Tests:** `src/haoline/tests/test_quantization_analysis.py` (26 tests)

### Story 26.1: Mixed Precision Detection - COMPLETE (4/5)
- [x] Detect per-layer precision (weights vs activations vs accumulation)
- [x] Identify INT4 weights with FP16 activations pattern
- [x] Detect FP32 accumulation in quantized MatMuls
- [x] Report precision breakdown by layer type (attention vs FFN vs embed)
- [ ] Visualize precision transitions in graph (future: requires graph viz update)

### Story 26.2: Quantization Scheme Detection - COMPLETE (6/6)
- [x] Detect GPTQ quantization patterns (group-wise, act_order)
- [x] Detect AWQ quantization patterns (activation-aware)
- [x] Detect GGML/GGUF quantization types (Q4_0, Q4_K_M, Q5_K_S, etc.)
- [x] Detect bitsandbytes NF4/FP4 quantization
- [x] Report expected accuracy degradation per scheme
- [x] Compare memory vs accuracy tradeoffs between schemes

### Story 26.3: Calibration Analysis - PARTIAL (2/4)
- [ ] Detect if model has calibration metadata (future: requires ONNX QDQ inspection)
- [ ] Estimate quantization error per layer (future: requires inference comparison)
- [x] Identify sensitive layers (high quantization error)
- [x] Recommend layers to keep at higher precision

---

## Epic 27: Attention Variant Detection (COMPLETE - 20/20 tasks)

*Completed: January 6, 2026 (v1.4.0)*

Modern LLMs use many attention optimizations beyond vanilla self-attention.

**Module:** `src/haoline/attention_analysis.py`
**CLI Flags:** `--attention-analysis`, `--attention-analysis-json`
**Tests:** `src/haoline/tests/test_attention_analysis.py` (25 tests)

### Story 27.1: Attention Architecture Detection - COMPLETE (5/5)
- [x] Detect Multi-Head Attention (MHA) - standard pattern
- [x] Detect Multi-Query Attention (MQA) - single KV head
- [x] Detect Grouped-Query Attention (GQA) - fewer KV heads than Q
- [x] Report num_q_heads, num_kv_heads, head_dim
- [x] Calculate KV cache savings for GQA/MQA vs MHA

### Story 27.2: Attention Pattern Detection - COMPLETE (5/5)
- [x] Detect sliding window attention (Mistral-style)
- [x] Detect local + global attention (Longformer-style) via pattern_type enum
- [x] Detect sparse attention patterns (BigBird, etc.) via pattern_type enum
- [x] Detect cross-attention (encoder-decoder models)
- [x] Report effective context length and attention complexity

### Story 27.3: Position Encoding Detection - COMPLETE (5/5)
- [x] Detect RoPE (Rotary Position Embedding)
- [x] Detect ALiBi (Attention with Linear Biases)
- [x] Detect learned positional embeddings
- [x] Detect sinusoidal positional encoding
- [x] Report max context length and extrapolation capability

### Story 27.4: Fused Attention Patterns - COMPLETE (5/5)
- [x] Detect FlashAttention-style fused patterns
- [x] Detect xFormers memory-efficient attention
- [x] Detect cuDNN fused multi-head attention
- [x] Report theoretical vs actual memory usage
- [x] Detect PyTorch SDPA (ScaledDotProductAttention)

---

## Epic 49: HuggingFace Integration - Stories 49.1-49.4 (COMPLETE)

*Completed: January 6, 2026*

### Story 49.1: HuggingFace Model Integration - COMPLETE (7/7)
*Load HF models (config + weights) and auto-convert to ONNX for full analysis.*

- [x] Add `--from-huggingface REPO_ID` CLI flag
- [x] Download config.json + model files from HF Hub
- [x] Detect model type from config (BERT, GPT, LLaMA, etc.)
- [x] Load model using `transformers` library
- [x] Export to ONNX using `optimum` library
- [x] Run full analysis on exported ONNX
- [x] Add `huggingface` extra to pyproject.toml (transformers, optimum)

### Story 49.2: Format-Aware UI/CLI - COMPLETE (7/7 + 2 post-1.0)
*Show appropriate metrics and disable unavailable features per format.*

- [x] Define `FormatCapabilities` dataclass with feature flags
- [x] Return capabilities from each format reader
- [x] CLI: Skip FLOPs/graph for weight-only formats with clear message
- [x] Streamlit: Disable graph tab for formats without graph
- [x] Show "Convert to ONNX for full analysis" prompt for Tier 3/4 formats
- [x] Add format tier badge in reports (Full/Graph/Metadata/Weights)
- [x] Show "Feature unavailable for [format]" with upgrade path in UI
- Post-1.0: "Why is this grayed out?" help tooltip
- Post-1.0: "Format Capabilities Report" section

### Story 49.3: SafeTensors → ONNX Path - COMPLETE (4/4)
*If SafeTensors is alongside config.json, auto-load and convert.*

- [x] Detect config.json in same directory as .safetensors
- [x] Parse config.json to get architecture type
- [x] Auto-suggest HF model load if config found
- [x] Support local directory with config + safetensors

### Story 49.4: ONNX Hub Conversions - COMPLETE (5/9, 4 not feasible)
*Convert TFLite/CoreML/OpenVINO → ONNX to enable full analysis capabilities.*

**Findings:**
- TFLite → ONNX: Works via `--from-tflite`
- CoreML → ONNX: **NOT FEASIBLE** - coremltools converts TO CoreML, not FROM
- OpenVINO → ONNX: **NOT FEASIBLE** - OpenVINO IR is generated FROM ONNX

Completed tasks:
- [x] TFLite → ONNX via `--from-tflite` (already existed)
- [x] CLI auto-prompt for TFLite conversion
- [x] Streamlit: Show conversion hint for TFLite
- [x] Document conversion quality/lossiness per format

### Story 49.5: Native FLOPs for Non-ONNX Formats - COMPLETE (4/4)
*Added native FLOP estimation to Graph-tier format readers.*

- [x] Map TFLite builtin ops to FLOP formulas (40+ ops: Conv2D, DepthwiseConv, FullyConnected, LSTM, etc.)
- [x] Map CoreML layer types to FLOP formulas (40+ types: convolution, innerProduct, softmax, etc.)
- [x] Map OpenVINO op types to FLOP formulas (50+ ops: Convolution, MatMul, ScaledDotProductAttention, etc.)
- [x] Update FormatCapabilities (has_flops=True for TFLite, CoreML, OpenVINO)

**Key files modified:**
- `src/haoline/formats/tflite.py` - `TFLITE_FLOP_FORMULAS`, `_estimate_tflite_op_flops()`, `TFLiteInfo.total_flops`
- `src/haoline/formats/coreml.py` - `COREML_FLOP_FORMULAS`, `_estimate_coreml_layer_flops()`, `CoreMLInfo.total_flops`
- `src/haoline/formats/openvino.py` - `OPENVINO_FLOP_FORMULAS`, `_estimate_openvino_layer_flops()`, `OpenVINOInfo.total_flops`
- `src/haoline/format_adapters.py` - Updated `FORMAT_CAPABILITIES` with `has_flops=True`

---

## Archived Merged Epics

*The following epics were merged into existing epics. Any remaining tasks are tracked in their parent epics.*

- **Epic 45** → Merged into Epic 11 (Streamlit UI)
- **Epic 46** → Merged into Epic 18 (Universal IR) - Stories 18.7, 18.8 remain
- **Epic 48** → Merged into Epic 33 (QAT Linters) - Story 33.6 remains

---

*End of Archive*

