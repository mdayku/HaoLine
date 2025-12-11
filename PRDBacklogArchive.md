# HaoLine - PRD & Backlog Archive

*This document archives completed epics and historical changelog entries to reduce context window usage in the main PRD.md and BACKLOG.md files.*

**Created:** December 6, 2025  
**Purpose:** Historical reference for completed work

---

## Table of Contents

1. [Completed Epics (Detailed)](#completed-epics-detailed)
2. [PRD Delta Log Archive (Pre-December 6, 2025)](#prd-delta-log-archive)

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

## Epic 41: Standardized Reporting (COMPLETE - 44/44)

*Completed: December 2025*

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

*End of Archive*

