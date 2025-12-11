# HaoLine (皓线) - Project Backlog

*Universal model analysis and inspection platform. See what's really inside your models.*

**Related Documents:**
- [PRD.md](PRD.md) - Product requirements and specifications
- [BRAINLIFT.md](BRAINLIFT.md) - Daily learning logs
- [Architecture.md](Architecture.md) - System design details

---

## Current Priority Focus (Dec 2025)

**Goal:** Close UI parity gaps identified by deep research audit.

| Priority | Focus Area | Key Work |
|----------|------------|----------|
| **P1** | UI Parity Gaps | Stories 11.5 (layer table), 11.6 (quant UI), format uploaders |
| **P1** | Format UX | Epic 49 (tier hints, disabled feature messaging) |
| **P1** | Honest Docs | Story 18.8 (Universal IR limitations documented) |
| **P2** | LLM Support | Epic 24 (GGUF UI: context slider, quant charts) |
| **P2** | CLI UX | Epic 50 (dependency prompting, --check-deps) |

**Quick Wins (< 1 hour each):** ✅ **ALL COMPLETE (Dec 11, 2025)**
- ~~Add `.gguf` to Streamlit uploader (Task 24.2.1)~~ ✅
- ~~Add `.tflite` to Streamlit uploader (Task 21.2.5)~~ ✅
- ~~Add `.mlmodel` to Streamlit uploader (Task 20.2.5)~~ ✅
- ~~Add `.xml` to Streamlit uploader (Task 23.2.5)~~ ✅
- Updated format capabilities matrix with tier system

---

## Progress Summary

| Epic | Status | Stories | Tasks Complete | Priority |
|------|--------|---------|----------------|----------|
| Epic 1: Environment Setup | **Complete** | 3 | 11/11 | Done |
| Epic 2: Core Analysis Engine | **Complete** | 4 | 17/17 | Done |
| Epic 3: Pattern Analysis | **Complete** | 2 | 9/9 | Done |
| Epic 4: CLI and Output | **Complete** | 4 | 18/18 | Done |
| Epic 4B: PyTorch Integration | **Complete** | 2 | 14/14 | Done |
| Epic 4C: TensorFlow/Keras/JAX | **Complete** | 3 | 15/15 | Done |
| Epic 5: Visualization | **Complete** | 8 | 52/52 | Done |
| Epic 6: Hardware/Compare | **COMPLETE** | 10 | 56/56 | P3 |
| Epic 7: LLM Integration | **COMPLETE** | 2 | 5/5 | P3 |
| Epic 8: Testing & CI/CD | **COMPLETE** | 4 | 18/18 | P3 |
| Epic 9: Runtime Profiling | **COMPLETE** | 6 | 22/22 | P2 |
| Epic 10: SaaS Web App | Not Started | 5 | 0/27 | P4 |
| Epic 10B: Standalone Package | **COMPLETE** | 4 | 23/23 | Done |
| Epic 11: Streamlit Web UI | **COMPLETE** (+11.5-11.6 in progress) | 5 | 27/29 | **P1** ← parity gaps |
| Epic 12: Eval Import & Comparison | **COMPLETE** | 7 | 30/30 | Done |
| Epic 13-17: MLOps Platform | Future | 5 | 0/? | P5 |
| Epic 18: Universal IR | **COMPLETE** (+18.8 pending) | 7 | 25/31 | **P1** ← honest docs |
| Epic 19: SafeTensors | In Progress | 2 | 6/10 | P2 |
| Epic 20: CoreML | In Progress | 3 | 8/19 | P2 |
| Epic 21: TFLite | In Progress | 3 | 5/18 | P2 (ONNX→TFLite blocked) |
| Epic 22: TensorRT Engine Introspection | **COMPLETE** | 8 | 50/50 | Done |
| Epic 23: OpenVINO | In Progress | 3 | 7/16 | P3 |
| Epic 24: GGUF | In Progress | 2 | 7/13 | **P2** ← LLM parity |
| Epic 25: Privacy/Trust | **COMPLETE** | 3 | 9/9 | P1 |
| **LLM-SCALE ANALYSIS** ||||
| Epic 26: Advanced Quantization | Not Started | 3 | 0/16 | P3 |
| Epic 27: Attention Variants | Not Started | 4 | 0/19 | P3 |
| Epic 28: Memory Patterns | Not Started | 4 | 0/18 | P3 |
| Epic 29: Sparse/Efficient | Not Started | 4 | 0/16 | P3 |
| Epic 30: LLM Deployment | Not Started | 4 | 0/19 | P3 |
| **OPTIMIZATION** ||||
| Epic 31: Quantization Service | Not Started | 6 | 0/32 | **P2** |
| Epic 32: Model Optimization | Not Started | 3 | 0/14 | P3 |
| Epic 33: QAT Linters | **COMPLETE** | 5 | 41/41 | **P1** |
| Epic 34: Activation Visualization | Not Started | 5 | 0/25 | P2/P3 |
| Epic 35: TRT-Aware Graph UX | Not Started | 3 | 0/16 | **P2** |
| Epic 36: Layer Visualization | In Progress | 5 | 4/25 | **P2** *(4 tasks via Epic 22)* |
| Epic 37: Hardware Recommender | Not Started | 2 | 0/10 | P3 |
| Epic 38: Docker Distribution | In Progress | 1 | 1/5 | P3 |
| Epic 39: Pydantic Schema Migration | **COMPLETE** | 3 | 12/12 | Done |
| Epic 40: Full Pydantic Dataclass Migration | **COMPLETE** | 6 | 58/58 | Done ✓ v0.5.0 |
| Epic 41: Standardized Reporting | **COMPLETE** (+41.7 pending) | 6 | 44/50 | **P1** ← parity |
| Epic 42: Format Conversion Testing | In Progress | 6 | 15/38 | P1 (ONNX→TFLite blocked) |
| Epic 49: Format Tiers & HuggingFace | Not Started | 5 | 0/30 | **P1** ← format UX |
| Epic 50: CLI Modernization (Typer) | Not Started | 3 | 0/18 | **P2** ← dep prompts |
| Epic 51: AWS GPU Deployment | Not Started | 5 | 0/26 | P3 |
| **DEEP RESEARCH SUGGESTIONS** | | | | *Dec 2025* |
| Epic 43: Performance & Scalability | Not Started | 3 | 0/14 | P3 |
| Epic 44: Expanded Op Type Support | Not Started | 3 | 0/14 | P3 |
| ~~Epic 45: UI Demo Polish~~ | *Merged* | - | - | *→ Epic 11* |
| ~~Epic 46: Enhanced Model Diff~~ | *Merged* | - | - | *→ Epic 18* |
| Epic 47: Model Card & Standards | Not Started | 2 | 0/10 | P4 |
| ~~Epic 48: Quantization Efficiency~~ | *Merged* | - | - | *→ Epic 33* |

---

## Completed Epics (Core Foundation)

*Detailed task lists archived in [PRDBacklogArchive.md](PRDBacklogArchive.md)*

- **Epic 1: Environment Setup** (11/11) - Fork, build, scaffolding
- **Epic 2: Core Analysis Engine** (17/17) - Graph loader, params, FLOPs, memory
- **Epic 3: Pattern Analysis** (9/9) - Block detection, risk heuristics
- **Epic 4: CLI and Output** (18/18) - CLI, JSON, Markdown, HTML

- **Epic 4B: PyTorch Integration** (14/14) - PyTorch→ONNX conversion
- **Epic 4C: TensorFlow/Keras/JAX** (15/15) - TF/Keras/JAX→ONNX conversion
- **Epic 5: Visualization** (52/52) - Charts, patterns, icons, edges, interactive graph, layer table
- **Epic 6: Hardware/Compare** (56/56) - Hardware profiles, estimates, compare mode, quantization impact
- **Epic 7: LLM Integration** (5/5) - OpenAI API, summaries, env var config
- **Epic 8: Testing & CI/CD** (18/18) - Unit tests, integration tests, GitHub Actions
- **Epic 9: Runtime Profiling** (22/22) - Batch benchmarking, GPU memory, per-layer timing, bottleneck detection

---

## Epic 10: SaaS Web Application (P4)

*W&B-style hosted service. Requires Epic 25 (Privacy/Trust) first.*

### Story 10.1: Web Backend API
- [ ] **Task 10.1.1-10.1.6**: FastAPI backend (6 tasks)

### Story 10.2: Frontend MVP
- [ ] **Task 10.2.1-10.2.6**: React/Next.js frontend (6 tasks)

### Story 10.3: Authentication and Users
- [ ] **Task 10.3.1-10.3.5**: Auth integration (5 tasks)

### Story 10.4: Cloud Infrastructure
- [ ] **Task 10.4.1-10.4.6**: Deployment setup (6 tasks)

### Story 10.5: Model History and Comparison
- [ ] **Task 10.5.1-10.5.5**: Versioning features (5 tasks)

- **Epic 10B: Standalone Package** (23/23) - Greenfield extraction, PyPI publishing, CI/CD
- **Epic 11: Streamlit Web UI** (17/17 + 11.5-11.6 pending) - Web interface, HuggingFace Spaces deployment

**New Story 11.5 (in Epic 11):** Layer Summary In-App Table **← DEEP RESEARCH PRIORITY**
*Gap: Per-layer breakdowns exist internally but only downloadable via CSV, no in-app table.*

- [x] **Task 11.5.1**: Add "Layer Details" tab in Streamlit analysis view
- [x] **Task 11.5.2**: Use LayerSummaryBuilder to generate pandas DataFrame
- [x] **Task 11.5.3**: Display layer table via st.dataframe (sortable, filterable)
- [x] **Task 11.5.4**: Show layer name, op type, param count, FLOPs, output shape
- [x] **Task 11.5.5**: Add download buttons for CSV/JSON export (mirror CLI --layer-csv)
- [x] **Task 11.5.6**: Add layer search/filter input

**New Story 11.6 (in Epic 11):** Quantization Lint/Advice Display **← DEEP RESEARCH PRIORITY**
*Gap: Quantization lint/advice computed in CLI but no UI display.*

- [x] **Task 11.6.1**: Add "Quantization" panel/tab in Streamlit
- [x] **Task 11.6.2**: Display readiness_score from QuantizationLinter
- [x] **Task 11.6.3**: List quantization warnings with severity colors
- [x] **Task 11.6.4**: Show QuantizationAdvisor strategy recommendations
- [ ] **Task 11.6.5**: Display per-layer quantization sensitivity if available
- [ ] **Task 11.6.6**: Add --lint-quant CLI flag to print lint warnings
- **Epic 12: Eval Import** (30/30) - Eval schemas, adapters, cost calculator, YOLO demo
- **Epic 18: Universal IR** (25/25) - Universal graph, format adapters, conversion matrix
- **Epic 25: Privacy/Trust** (9/9) - Local-first, output controls, enterprise documentation
- **Epic 33: QAT Linters** (41/41) - Quantization readiness, QAT validation, recommendations
- **Epic 39: Pydantic Migration** (12/12) - Schema validation, Pydantic models
- **Epic 41: Standardized Reporting** (44/44 + 41.7 in progress) - CLI-Streamlit parity, enhanced LLM prompts

**New Story 41.7 (in Epic 41):** Output Parity Gap Closure
*Address remaining gaps between CLI and Streamlit capabilities.*

- [ ] **Task 41.7.1**: Document PyTorch upload limitation in Streamlit (needs local torch install)
- [ ] **Task 41.7.2**: Fix or remove PDF export in Streamlit (playwright issues on HF Spaces)
- [ ] **Task 41.7.3**: Add "CLI Only" badges to Streamlit for features requiring local install
- [ ] **Task 41.7.4**: Add "Export as CLI command" button (generate equivalent haoline CLI command)
- [ ] **Task 41.7.5**: Ensure JSON report schema identical between CLI and Streamlit
- [ ] **Task 41.7.6**: Add format support comparison table to docs (what works where)

---

## Future Epics (Not Started)

### MLOps Platform Vision (P5) - Epics 13-17
*High-level placeholder. Do not implement until there's demand.*
- Epic 13: Cloud Provider Integration
- Epic 14: GPU Orchestration
- Epic 15: Training Estimation
- Epic 16: Model Inventory Management
- Epic 17: Billing and Usage

---

## Epic 19: SafeTensors Format (P2)

*HuggingFace ecosystem, widely used for LLM weights. Easy win.*

**Note:** Story 19.2 (Writer) exports *to* SafeTensors. Epic 49 imports *from* SafeTensors/HuggingFace *to* ONNX. They are independent - Story 19.2 is NOT a prerequisite for Epic 49.

### Story 19.1: SafeTensors Reader - **COMPLETE**
- [x] **Task 19.1.1**: Add safetensors dependency (optional) - in `[formats]` extra
- [x] **Task 19.1.2**: Implement SafeTensorsReader.read() - load tensor dict
- [x] **Task 19.1.3**: Extract metadata (tensor names, shapes, dtypes)
- [x] **Task 19.1.4**: Integrate with analysis pipeline (param counts, memory)
- [x] **Task 19.1.5**: Test with real SafeTensors model (sentence-transformers/all-MiniLM-L6-v2, 22.7M params)
- [x] **Task 19.1.6**: Write unit tests for SafeTensorsReader (8 tests in test_formats.py)

### Story 19.2: SafeTensors Writer
- [ ] **Task 19.2.1**: Implement SafeTensorsAdapter.write() - export weights
- [ ] **Task 19.2.2**: Support conversion from ONNX initializers to SafeTensors
- [ ] **Task 19.2.3**: Support conversion from PyTorch state_dict to SafeTensors
- [ ] **Task 19.2.4**: Add `--export-weights safetensors` CLI flag

---

## Epic 20: CoreML Format (P2)

*Apple's ML framework for iOS/macOS deployment.*

### Story 20.1: CoreML Reader - **COMPLETE**
- [x] **Task 20.1.1**: Add coremltools dependency (optional) - in `[coreml]` extra
- [x] **Task 20.1.2**: Implement CoreMLReader.read() - load .mlmodel/.mlpackage
- [x] **Task 20.1.3**: Map CoreML ops to layer info (op_type_counts, precision_breakdown)
- [x] **Task 20.1.4**: Extract CoreML-specific metadata (compute units, iOS version)
- [x] **Task 20.1.5**: Integrate with analysis pipeline
- [x] **Task 20.1.6**: Test with real CoreML model (in test_format_readers.py, CI on Linux)
- [x] **Task 20.1.7**: Write unit tests for CoreMLReader (6 tests in test_formats.py)

### Story 20.2: CoreML → UniversalGraph Adapter (Native Path) **← DEEP RESEARCH PRIORITY**
*Enable interactive graph visualization and layer-by-layer analysis for CoreML models.*

**Gap Identified:** CoreML can be analyzed via conversion pipelines but `.mlmodel`/`.mlpackage` not in Streamlit file picker.

**Alternative:** Epic 49.4 implements CoreML → ONNX hub conversion (simpler, may be lossy). This story is the native approach if hub conversion proves inadequate.

- [ ] **Task 20.2.1**: Implement `CoreMLAdapter.read()` - convert CoreML to UniversalGraph
- [ ] **Task 20.2.2**: Map CoreML layer types to universal ops
- [ ] **Task 20.2.3**: Extract layer shapes and connections from CoreML spec
- [ ] **Task 20.2.4**: Register adapter in format_adapters.py
- [x] **Task 20.2.5**: Add `.mlmodel`/`.mlpackage` to Streamlit file_uploader accepted types ✅ **DONE**
- [ ] **Task 20.2.6**: Test interactive graph generation with CoreML model
- [ ] **Task 20.2.7**: Display CoreML-specific metadata in UI (compute units, iOS version)

### Story 20.3: CoreML Writer
- [ ] **Task 20.3.1**: Implement CoreMLAdapter.write() via coremltools conversion
- [ ] **Task 20.3.2**: Support ONNX → CoreML conversion path
- [ ] **Task 20.3.3**: Support PyTorch → CoreML conversion path
- [ ] **Task 20.3.4**: Add iOS/macOS deployment target options
- [ ] **Task 20.3.5**: Add `--convert-to coreml` CLI flag

---

## Epic 21: TFLite Format (P2)

*TensorFlow Lite for mobile and edge deployment.*

### Story 21.1: TFLite Reader - **PARTIAL** (requires tflite-runtime for full parsing)
- [x] **Task 21.1.1**: Add tflite-runtime dependency (optional) - in `[formats]` extra
- [x] **Task 21.1.2**: Implement TFLiteReader.read() - load .tflite (full impl requires tflite-runtime)
- [ ] **Task 21.1.3**: Parse FlatBuffer schema for ops and tensors - **STUB** (pure Python fallback returns empty)
- [ ] **Task 21.1.4**: Map TFLite ops to op_type_counts - **STUB** (only works with tflite-runtime)
- [ ] **Task 21.1.5**: Extract quantization info (int8, float16) - **STUB** (only works with tflite-runtime)
- [ ] **Task 21.1.6**: Test with real TFLite model (Linux CI with tflite-runtime)
- [ ] **Task 21.1.7**: Write unit tests for TFLiteReader

### Story 21.2: TFLite → UniversalGraph Adapter (Native Path) **← DEEP RESEARCH PRIORITY**
*Enable interactive graph visualization and layer-by-layer analysis for TFLite models.*

**Gap Identified:** TFLite can be analyzed via conversion but `.tflite` not in Streamlit file picker.

**Alternative:** Epic 49.4 implements TFLite → ONNX hub conversion (simpler, may be lossy). This story is the native approach if hub conversion proves inadequate.

- [ ] **Task 21.2.1**: Implement `TFLiteAdapter.read()` - convert TFLite to UniversalGraph
- [ ] **Task 21.2.2**: Map TFLite op codes to universal ops
- [ ] **Task 21.2.3**: Extract tensor shapes and operator connections from FlatBuffer
- [ ] **Task 21.2.4**: Register adapter in format_adapters.py
- [x] **Task 21.2.5**: Add `.tflite` to Streamlit file_uploader accepted types ✅ **DONE**
- [ ] **Task 21.2.6**: Test interactive graph generation with TFLite model

### Story 21.3: TFLite Writer
- [ ] **Task 21.3.1**: Implement TFLiteAdapter.write() via tf.lite.TFLiteConverter
- [ ] **Task 21.3.2**: Support ONNX → TFLite conversion ⛔ **BLOCKED** (onnx2tf/onnx-tf broken with TF 2.16+)
- [ ] **Task 21.3.3**: Add quantization options (dynamic, full int8)
- [ ] **Task 21.3.4**: Add representative dataset hook for calibration
- [x] **Task 21.3.5**: Add `--convert-to tflite` CLI flag ✅ **COMPLETE** (Epic 42)

---

## Epic 22: TensorRT Engine Introspection - **COMPLETE**

*Archived to [PRDBacklogArchive.md](PRDBacklogArchive.md) - 50/50 tasks (v0.7.2)*

**Summary:** Deep analysis of NVIDIA TensorRT compiled engines including layer rewrite detection (FlashAttention, GELU, LayerNorm), ONNX↔TRT side-by-side HTML comparison, performance metadata panel, quantization bottleneck analysis, and timing charts.

---

## Epic 23: OpenVINO Format (P3)

*Intel's inference toolkit for CPU/GPU/VPU deployment.*

### Story 23.1: OpenVINO Reader - **COMPLETE**
- [x] **Task 23.1.1**: Add openvino dependency (optional) - in `[openvino]` extra
- [x] **Task 23.1.2**: Implement OpenVINOReader.read() - load .xml/.bin
- [x] **Task 23.1.3**: Map OpenVINO ops to layer_type_counts
- [x] **Task 23.1.4**: Extract precision breakdown
- [ ] **Task 23.1.5**: Test with real OpenVINO model (.xml + .bin)
- [x] **Task 23.1.6**: Write unit tests for OpenVINOReader (5 tests in test_formats.py)

### Story 23.2: OpenVINO → UniversalGraph Adapter (Native Path) **← DEEP RESEARCH PRIORITY**
*Enable interactive graph visualization and layer-by-layer analysis for OpenVINO models.*

**Gap Identified:** OpenVINO can be analyzed but `.xml`/`.bin` not in Streamlit file picker.

**Alternative:** Epic 49.4 implements OpenVINO → ONNX hub conversion (simpler, may be lossy). This story is the native approach if hub conversion proves inadequate.

- [ ] **Task 23.2.1**: Implement `OpenVINOAdapter.read()` - convert OpenVINO IR to UniversalGraph
- [ ] **Task 23.2.2**: Map OpenVINO op types to universal ops
- [ ] **Task 23.2.3**: Extract layer shapes and connections from IR
- [ ] **Task 23.2.4**: Register adapter in format_adapters.py
- [x] **Task 23.2.5**: Add `.xml` to Streamlit file_uploader accepted types ✅ **DONE**
- [ ] **Task 23.2.6**: Test interactive graph generation with OpenVINO model

### Story 23.3: OpenVINO Writer
- [ ] **Task 23.3.1**: Implement OpenVINOAdapter.write()
- [ ] **Task 23.3.2**: Support ONNX → OpenVINO conversion
- [ ] **Task 23.3.3**: Add precision options
- [ ] **Task 23.3.4**: Add `--convert-to openvino` CLI flag

---

## Epic 24: GGUF Format (P3 - Read-Only)

*llama.cpp format for running LLMs locally.*

**Note:** GGUF is a weights-only format with architecture metadata. It does NOT contain a computational graph, so interactive graph visualization is not possible. However, we can display quantization breakdown, VRAM estimates, and architecture details.

### Story 24.1: GGUF Reader - **COMPLETE**
- [x] **Task 24.1.1**: Implement GGUF header parser (pure Python, no deps)
- [x] **Task 24.1.2**: Extract model metadata (arch, context_length, etc.)
- [x] **Task 24.1.3**: Extract quantization type per tensor
- [x] **Task 24.1.4**: Estimate memory footprint (VRAM estimation)
- [x] **Task 24.1.5**: Test with real GGUF model (TinyLlama-1.1B Q2_K, 1.1B params, 458MB)
- [x] **Task 24.1.6**: Write unit tests for GGUFReader (8 tests in test_formats.py)

### Story 24.2: GGUF Streamlit UI & Analysis Features **← DEEP RESEARCH PRIORITY**
**Gap Identified:** GGUF can be analyzed but `.gguf` not in Streamlit file picker. LLM-specific charts (quant-breakdown, context-slider) not implemented.

- [x] **Task 24.2.1**: Add `.gguf` to Streamlit file_uploader accepted types ✅ **DONE**
- [ ] **Task 24.2.2**: Display quantization breakdown chart (bar chart of tensor counts by bit-width)
- [ ] **Task 24.2.3**: Display architecture details (layers, hidden_size, num_heads, context_length)
- [ ] **Task 24.2.4**: Add VRAM calculator with context length slider (recompute estimates dynamically)
- [ ] **Task 24.2.5**: Show tensor-level quantization table
- [ ] **Task 24.2.6**: Create "LLM Model Details" tab in Streamlit for GGUF models
- [ ] **Task 24.2.7**: Show unsupported_ops warnings from quantization_lint

---

## Epic 49: Format Capability Tiers & HuggingFace Integration (P2)

*Rationalize what metrics are available per format, and add auto-conversion for weight-only formats.*

**Relationship to Epic 42:** Epic 42 tests existing conversion paths work correctly. Epic 49 adds NEW features (HuggingFace CLI, format-aware UI). Testing for Epic 49 features should be added to Epic 42 after implementation.

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

### Story 49.1: HuggingFace Model Integration
*Load HF models (config + weights) and auto-convert to ONNX for full analysis.*

- [ ] **Task 49.1.1**: Add `--from-huggingface REPO_ID` CLI flag
- [ ] **Task 49.1.2**: Download config.json + model files from HF Hub
- [ ] **Task 49.1.3**: Detect model type from config (BERT, GPT, LLaMA, etc.)
- [ ] **Task 49.1.4**: Load model using `transformers` library
- [ ] **Task 49.1.5**: Export to ONNX using `optimum` library
- [ ] **Task 49.1.6**: Run full analysis on exported ONNX
- [ ] **Task 49.1.7**: Add `huggingface` extra to pyproject.toml (transformers, optimum)

### Story 49.2: Format-Aware UI/CLI
*Show appropriate metrics and disable unavailable features per format.*

- [ ] **Task 49.2.1**: Define `FormatCapabilities` dataclass with feature flags
- [ ] **Task 49.2.2**: Return capabilities from each format reader
- [ ] **Task 49.2.3**: CLI: Skip FLOPs/graph for weight-only formats with clear message
- [ ] **Task 49.2.4**: Streamlit: Disable graph tab for formats without graph
- [ ] **Task 49.2.5**: Show "Convert to ONNX for full analysis" prompt for Tier 3/4 formats
- [ ] **Task 49.2.6**: Add format tier badge in reports (Full/Graph/Metadata/Weights)
- [ ] **Task 49.2.7**: Show "Feature unavailable for [format]" with upgrade path in UI
- [ ] **Task 49.2.8**: Add "Why is this grayed out?" help tooltip explaining format limitations
- [ ] **Task 49.2.9**: Generate "Format Capabilities Report" section showing what was/wasn't analyzed

### Story 49.3: SafeTensors → ONNX Path
*If SafeTensors is alongside config.json, auto-load and convert.*

- [ ] **Task 49.3.1**: Detect config.json in same directory as .safetensors
- [ ] **Task 49.3.2**: Parse config.json to get architecture type
- [ ] **Task 49.3.3**: Auto-suggest HF model load if config found
- [ ] **Task 49.3.4**: Support local directory with config + safetensors

### Story 49.4: ONNX Hub Conversions (Full Analysis Path)
*Convert TFLite/CoreML/OpenVINO → ONNX to enable full analysis capabilities.*

**Why this approach:** Instead of building native UniversalGraph adapters for each format, convert to ONNX first and reuse all existing analysis code. Trade-off: some conversions may be lossy.

- [ ] **Task 49.4.1**: Implement TFLite → ONNX via `tflite2onnx` or `tf2onnx`
- [ ] **Task 49.4.2**: Add `--convert-to-onnx` flag for TFLite files
- [ ] **Task 49.4.3**: Implement CoreML → ONNX via `coremltools.converters.onnx`
- [ ] **Task 49.4.4**: Add `--convert-to-onnx` flag for CoreML files
- [ ] **Task 49.4.5**: Implement OpenVINO → ONNX conversion path
- [ ] **Task 49.4.6**: Add `--convert-to-onnx` flag for OpenVINO files
- [ ] **Task 49.4.7**: CLI auto-prompt: "Convert to ONNX for full analysis? (y/n)"
- [ ] **Task 49.4.8**: Streamlit: Add "Convert to ONNX" button for Tier 2/3 formats
- [ ] **Task 49.4.9**: Document conversion quality/lossiness per format

### Story 49.5: Native FLOPs for Non-ONNX Formats (Optional)
*Alternative to hub conversion: Add FLOPs directly to format readers.*

**Note:** This is lower priority than 49.4. Only implement if conversion proves too lossy for specific use cases.

- [ ] **Task 49.5.1**: Map TFLite builtin ops to FLOP formulas
- [ ] **Task 49.5.2**: Map CoreML layer types to FLOP formulas
- [ ] **Task 49.5.3**: Map OpenVINO op types to FLOP formulas
- [ ] **Task 49.5.4**: Add FLOPs to format reader return types

---

## Epic 50: CLI Modernization with Typer (P3)

*Modernize CLI experience with better UX, dependency prompting, and shell completion.*

**Why Typer:**
- Built on Click, modern Pythonic API
- Automatic help generation from type hints
- Shell completion out of the box
- Better error messages
- Cleaner code than argparse

### Story 50.1: Migrate to Typer
*Replace argparse with Typer for main CLI.*

- [ ] **Task 50.1.1**: Add `typer` dependency to pyproject.toml
- [ ] **Task 50.1.2**: Refactor `cli.py` main parser to Typer app
- [ ] **Task 50.1.3**: Migrate all CLI arguments to Typer options/arguments
- [ ] **Task 50.1.4**: Add shell completion support (bash, zsh, fish, PowerShell)
- [ ] **Task 50.1.5**: Update CLI help strings for Typer format

### Story 50.2: Graceful Dependency Prompting
*When a feature requires missing dependencies, show helpful install commands.*

- [ ] **Task 50.2.1**: Create `check_dependency()` helper that returns install command
- [ ] **Task 50.2.2**: Add dependency checks for TensorRT features (`pip install haoline[tensorrt]`)
- [ ] **Task 50.2.3**: Add dependency checks for runtime profiling (`pip install haoline[runtime]`)
- [ ] **Task 50.2.4**: Add dependency checks for LLM features (`pip install haoline[llm]`)
- [ ] **Task 50.2.5**: Add dependency checks for format readers (safetensors, coreml, openvino)
- [ ] **Task 50.2.6**: Show clear error with install command, not cryptic ImportError
- [x] **Task 50.2.7**: Add Streamlit UI feature availability matrix (show which features need GPU/deps)
- [x] **Task 50.2.8**: Add "Requires: TensorRT" / "Requires: GPU" badges to blocked features in UI
- [ ] **Task 50.2.9**: Add `--check-deps` CLI flag to audit installed vs available features
- [ ] **Task 50.2.10**: Update `--list-formats` to show "Available" vs "Needs: pip install haoline[X]"
- [ ] **Task 50.2.11**: Add PDF export dependency check (playwright + chromium install)

### Story 50.3: CLI UX Improvements
*Better progress, feedback, and interactive features.*

- [ ] **Task 50.3.1**: Add progress bars for long operations (analysis, conversion)
- [ ] **Task 50.3.2**: Add `--quiet` and `--verbose` flags consistently
- [ ] **Task 50.3.3**: Add colored output for warnings/errors (with `--no-color` flag)
- [ ] **Task 50.3.4**: Add `haoline doctor` command to check system setup (GPU, deps, versions)

---

## Epic 51: AWS GPU Deployment (P3)

*Deploy HaoLine on AWS with GPU support to unlock all features not available on HF Spaces free tier.*

**Motivation:**
- HuggingFace Spaces free tier = CPU only (no TensorRT, no runtime benchmarking)
- AWS g4dn/g5 instances provide NVIDIA GPUs for full feature support
- Professional deployment with custom domain, scaling

### Story 51.1: Infrastructure Setup
*Set up AWS infrastructure for HaoLine deployment.*

- [ ] **Task 51.1.1**: Create Dockerfile with CUDA, TensorRT, and all dependencies
- [ ] **Task 51.1.2**: Set up ECR repository for container images
- [ ] **Task 51.1.3**: Configure g4dn.xlarge instance (T4 GPU, cost-effective)
- [ ] **Task 51.1.4**: Set up ALB with SSL certificate (ACM)
- [ ] **Task 51.1.5**: Configure auto-start/stop to minimize costs (scheduled scaling)

### Story 51.2: Streamlit Deployment
*Deploy Streamlit app on AWS with GPU support.*

- [ ] **Task 51.2.1**: Create ECS task definition with GPU resource allocation
- [ ] **Task 51.2.2**: Deploy Streamlit container to ECS Fargate (GPU)
- [ ] **Task 51.2.3**: Configure CloudWatch logging and monitoring
- [ ] **Task 51.2.4**: Add health checks and auto-recovery

### Story 51.3: Feature Unlocking
*Enable GPU-dependent features in AWS deployment.*

- [ ] **Task 51.3.1**: Enable TensorRT engine analysis (full support)
- [ ] **Task 51.3.2**: Enable runtime inference benchmarking (actual GPU timings)
- [ ] **Task 51.3.3**: Add UI indicator showing "GPU-Accelerated" status

### Story 51.4: Runtime Profiling Streamlit UI
*Epic 9 (Runtime Profiling) is CLI-only. Add Streamlit UI for GPU deployments.*

- [ ] **Task 51.4.1**: Add "Benchmark" tab in Streamlit analysis view
- [ ] **Task 51.4.2**: Implement batch size sweep UI with live progress
- [ ] **Task 51.4.3**: Show per-layer timing breakdown chart
- [ ] **Task 51.4.4**: Display GPU memory utilization during inference
- [ ] **Task 51.4.5**: Add bottleneck detection summary card
- [ ] **Task 51.4.6**: Show "GPU Required" message on HF Spaces free tier

### Story 51.5: Eval Import Streamlit UI
*Epic 12 (Eval Import & Comparison) is CLI-only. Add Streamlit UI.*

- [ ] **Task 51.5.1**: Add "Import Eval Results" expander in analysis view
- [ ] **Task 51.5.2**: Support file upload for eval JSON (Ultralytics, HF, lm-eval)
- [ ] **Task 51.5.3**: Auto-detect eval format and parse metrics
- [ ] **Task 51.5.4**: Display eval metrics cards (mAP, accuracy, F1, etc.)
- [ ] **Task 51.5.5**: Create combined report view (architecture + eval)
- [ ] **Task 51.5.6**: Add deployment cost calculator UI ($/day, $/month estimates)
- [ ] **Task 51.5.7**: Show accuracy vs size/speed tradeoff visualization
- [ ] **Task 51.5.8**: Export combined report as JSON/PDF

---

# LLM-SCALE ANALYSIS (P3+)

*Epics 26-30: Handle models like Opus 4.5, GPT-4, LLaMA-70B, Mixtral*

---

## Epic 26: Advanced Quantization Analysis (P3)

*Modern LLMs use complex quantization beyond simple int8/fp16.*

### Story 26.1: Mixed Precision Detection
- [ ] **Task 26.1.1**: Detect per-layer precision (weights vs activations vs accumulation)
- [ ] **Task 26.1.2**: Identify INT4 weights with FP16 activations pattern
- [ ] **Task 26.1.3**: Detect FP32 accumulation in quantized MatMuls
- [ ] **Task 26.1.4**: Report precision breakdown by layer type (attention vs FFN vs embed)
- [ ] **Task 26.1.5**: Visualize precision transitions in graph (where fp16→int8 happens)

### Story 26.2: Quantization Scheme Detection
- [ ] **Task 26.2.1**: Detect GPTQ quantization patterns (group-wise, act_order)
- [ ] **Task 26.2.2**: Detect AWQ quantization patterns (activation-aware)
- [ ] **Task 26.2.3**: Detect GGML/GGUF quantization types (Q4_0, Q4_K_M, Q5_K_S, etc.)
- [ ] **Task 26.2.4**: Detect bitsandbytes NF4/FP4 quantization
- [ ] **Task 26.2.5**: Report expected accuracy degradation per scheme
- [ ] **Task 26.2.6**: Compare memory vs accuracy tradeoffs between schemes

### Story 26.3: Calibration Analysis
- [ ] **Task 26.3.1**: Detect if model has calibration metadata
- [ ] **Task 26.3.2**: Estimate quantization error per layer
- [ ] **Task 26.3.3**: Identify sensitive layers (high quantization error)
- [ ] **Task 26.3.4**: Recommend layers to keep at higher precision

---

## Epic 27: Attention Variant Detection (P3)

*Modern LLMs use many attention optimizations beyond vanilla self-attention.*

### Story 27.1: Attention Architecture Detection
- [ ] **Task 27.1.1**: Detect Multi-Head Attention (MHA) - standard pattern
- [ ] **Task 27.1.2**: Detect Multi-Query Attention (MQA) - single KV head
- [ ] **Task 27.1.3**: Detect Grouped-Query Attention (GQA) - fewer KV heads than Q
- [ ] **Task 27.1.4**: Report num_q_heads, num_kv_heads, head_dim
- [ ] **Task 27.1.5**: Calculate KV cache savings for GQA/MQA vs MHA

### Story 27.2: Attention Pattern Detection
- [ ] **Task 27.2.1**: Detect sliding window attention (Mistral-style)
- [ ] **Task 27.2.2**: Detect local + global attention (Longformer-style)
- [ ] **Task 27.2.3**: Detect sparse attention patterns (BigBird, etc.)
- [ ] **Task 27.2.4**: Detect cross-attention (encoder-decoder models)
- [ ] **Task 27.2.5**: Report effective context length and attention complexity

### Story 27.3: Position Encoding Detection
- [ ] **Task 27.3.1**: Detect RoPE (Rotary Position Embedding)
- [ ] **Task 27.3.2**: Detect ALiBi (Attention with Linear Biases)
- [ ] **Task 27.3.3**: Detect learned positional embeddings
- [ ] **Task 27.3.4**: Detect sinusoidal positional encoding
- [ ] **Task 27.3.5**: Report max context length and extrapolation capability

### Story 27.4: Fused Attention Patterns
- [ ] **Task 27.4.1**: Detect FlashAttention-style fused patterns
- [ ] **Task 27.4.2**: Detect xFormers memory-efficient attention
- [ ] **Task 27.4.3**: Detect cuDNN fused multi-head attention
- [ ] **Task 27.4.4**: Report theoretical vs actual memory usage

---

## Epic 28: Memory Pattern Analysis (P3)

*LLM deployment is memory-bound. Understand where memory goes.*

### Story 28.1: Activation Checkpointing Detection
- [ ] **Task 28.1.1**: Detect activation checkpointing patterns (recompute on backward)
- [ ] **Task 28.1.2**: Identify checkpoint boundaries
- [ ] **Task 28.1.3**: Calculate memory savings vs compute overhead
- [ ] **Task 28.1.4**: Recommend optimal checkpoint granularity

### Story 28.2: KV Cache Analysis
- [ ] **Task 28.2.1**: Calculate KV cache size per layer per token
- [ ] **Task 28.2.2**: Project KV cache for variable context lengths (1k, 4k, 8k, 32k, 128k)
- [ ] **Task 28.2.3**: Detect KV cache quantization (INT8 KV cache)
- [ ] **Task 28.2.4**: Calculate max context length for given VRAM
- [ ] **Task 28.2.5**: Detect PagedAttention patterns (vLLM-style)
- [ ] **Task 28.2.6**: Report KV cache as % of total memory

### Story 28.3: Parallelism Strategy Detection
- [ ] **Task 28.3.1**: Detect tensor parallelism patterns (column/row split)
- [ ] **Task 28.3.2**: Detect pipeline parallelism patterns (layer sharding)
- [ ] **Task 28.3.3**: Detect data parallelism patterns
- [ ] **Task 28.3.4**: Identify all-reduce / all-gather communication ops
- [ ] **Task 28.3.5**: Report memory per GPU for N-way parallelism
- [ ] **Task 28.3.6**: Recommend parallelism strategy for target hardware

### Story 28.4: Memory Waterfall Analysis
- [ ] **Task 28.4.1**: Calculate peak memory at each point in forward pass
- [ ] **Task 28.4.2**: Generate memory waterfall chart (memory over time)
- [ ] **Task 28.4.3**: Identify memory spike locations
- [ ] **Task 28.4.4**: Recommend batch size for given VRAM constraint

---

## Epic 29: Sparse and Efficient Architecture Analysis (P3)

*Mixture of Experts, speculative decoding, and sparsity patterns.*

### Story 29.1: Mixture of Experts (MoE) Analysis
- [ ] **Task 29.1.1**: Detect MoE routing patterns (top-k gating)
- [ ] **Task 29.1.2**: Count total experts and active experts per token
- [ ] **Task 29.1.3**: Calculate effective vs total parameters
- [ ] **Task 29.1.4**: Analyze expert utilization/load balancing
- [ ] **Task 29.1.5**: Report memory for all experts vs active subset
- [ ] **Task 29.1.6**: Detect expert parallelism patterns

### Story 29.2: Speculative Decoding Detection
- [ ] **Task 29.2.1**: Detect draft model + verify model pattern
- [ ] **Task 29.2.2**: Identify draft model architecture
- [ ] **Task 29.2.3**: Calculate speculative decoding speedup potential
- [ ] **Task 29.2.4**: Report token acceptance rate requirements

### Story 29.3: Weight Sparsity Analysis
- [ ] **Task 29.3.1**: Detect structured sparsity (N:M sparsity)
- [ ] **Task 29.3.2**: Detect unstructured sparsity (pruned weights)
- [ ] **Task 29.3.3**: Calculate actual vs theoretical FLOPs with sparsity
- [ ] **Task 29.3.4**: Identify sparse-compatible hardware requirements
- [ ] **Task 29.3.5**: Report sparsity % per layer

### Story 29.4: Efficient Architecture Patterns
- [ ] **Task 29.4.1**: Detect depth-wise separable convolutions
- [ ] **Task 29.4.2**: Detect inverted residual blocks (MobileNet-style)
- [ ] **Task 29.4.3**: Detect squeeze-and-excitation patterns
- [ ] **Task 29.4.4**: Detect neural architecture search (NAS) patterns
- [ ] **Task 29.4.5**: Compare efficiency vs baseline architectures

---

## Epic 30: LLM Deployment Analysis (P3)

*Inference patterns differ from training. Understand production characteristics.*

### Story 30.1: Prefill vs Decode Analysis
- [ ] **Task 30.1.1**: Identify prefill phase (process prompt, compute-bound)
- [ ] **Task 30.1.2**: Identify decode phase (generate tokens, memory-bound)
- [ ] **Task 30.1.3**: Calculate time-to-first-token (TTFT) estimate
- [ ] **Task 30.1.4**: Calculate tokens-per-second decode rate
- [ ] **Task 30.1.5**: Report optimal batch size for each phase

### Story 30.2: Batching Strategy Analysis
- [ ] **Task 30.2.1**: Analyze static batching characteristics
- [ ] **Task 30.2.2**: Detect continuous batching compatibility
- [ ] **Task 30.2.3**: Calculate throughput vs latency tradeoffs
- [ ] **Task 30.2.4**: Report max concurrent requests for given VRAM
- [ ] **Task 30.2.5**: Model request queuing and scheduling impact

### Story 30.3: Context Length Scaling
- [ ] **Task 30.3.1**: Calculate O(n²) attention scaling impact
- [ ] **Task 30.3.2**: Calculate O(n) KV cache scaling impact
- [ ] **Task 30.3.3**: Generate context length vs memory/latency curves
- [ ] **Task 30.3.4**: Identify context length breakpoints (where OOM occurs)
- [ ] **Task 30.3.5**: Recommend context length for target hardware

### Story 30.4: Serving Framework Compatibility
- [ ] **Task 30.4.1**: Check vLLM compatibility (PagedAttention, continuous batching)
- [ ] **Task 30.4.2**: Check TensorRT-LLM compatibility
- [ ] **Task 30.4.3**: Check llama.cpp compatibility
- [ ] **Task 30.4.4**: Check Triton Inference Server compatibility
- [ ] **Task 30.4.5**: Report recommended serving framework for model characteristics

---

## Epic 31: Automated Quantization Service (P2)

*Don't just analyze quantization - DO the quantization. Users upload model + test data, we handle the rest.*

**Value Proposition:**
- Users don't need to learn quantization tooling
- Automatic before/after comparison
- Recommend best scheme for their accuracy/speed tradeoff
- Download optimized model ready for deployment

### Story 31.1: Calibration Dataset Interface
- [ ] **Task 31.1.1**: Define calibration dataset format (images, tensors, text)
- [ ] **Task 31.1.2**: Implement image folder loader for CV models
- [ ] **Task 31.1.3**: Implement JSON/CSV loader for tabular data
- [ ] **Task 31.1.4**: Implement text file loader for LLMs (prompts)
- [ ] **Task 31.1.5**: Add `--calibration-data` CLI flag
- [ ] **Task 31.1.6**: Validate calibration data matches model input spec

### Story 31.2: ONNX Runtime Quantization
- [ ] **Task 31.2.1**: Integrate onnxruntime.quantization module
- [ ] **Task 31.2.2**: Implement dynamic quantization (no calibration needed)
- [ ] **Task 31.2.3**: Implement static INT8 quantization with calibration
- [ ] **Task 31.2.4**: Implement QDQ (Quantize-Dequantize) format for TensorRT
- [ ] **Task 31.2.5**: Add `--quantize int8|int8-dynamic|qdq` CLI flag
- [ ] **Task 31.2.6**: Save quantized model to user-specified path

### Story 31.3: Advanced Quantization Backends
- [ ] **Task 31.3.0**: Compare deployment cost across precision variants (fp32 vs fp16 vs int8) — *from Epic 12.6.4*
- [ ] **Task 31.3.1**: Integrate Intel Neural Compressor (INC) for advanced PTQ
- [ ] **Task 31.3.2**: Integrate ONNX GPTQ quantization (for LLMs)
- [ ] **Task 31.3.3**: Integrate AWQ quantization support
- [ ] **Task 31.3.4**: Add INT4 quantization option
- [ ] **Task 31.3.5**: Add mixed-precision quantization (sensitive layers stay fp16)
- [ ] **Task 31.3.6**: Add `--quantize-scheme gptq|awq|int4` CLI flag

### Story 31.4: Accuracy Validation
- [ ] **Task 31.4.1**: Run inference on calibration set with original model
- [ ] **Task 31.4.2**: Run inference on calibration set with quantized model
- [ ] **Task 31.4.3**: Calculate output difference (MSE, cosine similarity)
- [ ] **Task 31.4.4**: Report per-layer quantization error
- [ ] **Task 31.4.5**: Flag layers with high error (candidates for mixed precision)
- [ ] **Task 31.4.6**: Generate accuracy comparison report

### Story 31.5: Multi-Variant Generation
- [ ] **Task 31.5.1**: Generate multiple quantized variants (int8, int4, mixed)
- [ ] **Task 31.5.2**: Benchmark all variants on calibration set
- [ ] **Task 31.5.3**: Generate Pareto frontier chart (accuracy vs size vs speed)
- [ ] **Task 31.5.4**: Recommend best variant for user's constraints
- [ ] **Task 31.5.5**: Package all variants in downloadable archive

### Story 31.6: Quantization Report
- [ ] **Task 31.6.1**: Add quantization section to HTML report
- [ ] **Task 31.6.2**: Show size comparison (original vs quantized)
- [ ] **Task 31.6.3**: Show accuracy comparison
- [ ] **Task 31.6.4**: Show per-layer precision breakdown
- [ ] **Task 31.6.5**: Include download links for quantized models
- [ ] **Task 31.6.6**: Add quantization recommendations

---

## Epic 32: Model Optimization Suite (P3)

*Beyond quantization - graph optimizations, layer fusion, pruning.*

### Story 32.1: ONNX Graph Optimizations
- [ ] **Task 32.1.1**: Integrate onnxruntime graph optimizers
- [ ] **Task 32.1.2**: Apply constant folding
- [ ] **Task 32.1.3**: Apply node fusion (Conv+BN, MatMul+Add)
- [ ] **Task 32.1.4**: Eliminate redundant ops (Identity, Dropout in inference)
- [ ] **Task 32.1.5**: Add `--optimize` CLI flag
- [ ] **Task 32.1.6**: Report optimizations applied and impact

### Story 32.2: Shape Optimization
- [ ] **Task 32.2.1**: Fix dynamic shapes to static for deployment
- [ ] **Task 32.2.2**: Add `--fix-batch-size N` CLI flag
- [ ] **Task 32.2.3**: Add `--fix-sequence-length N` CLI flag
- [ ] **Task 32.2.4**: Warn about shape changes that affect flexibility

### Story 32.3: Weight Pruning (Experimental)
- [ ] **Task 32.3.1**: Research structured pruning integration
- [ ] **Task 32.3.2**: Implement magnitude-based pruning
- [ ] **Task 32.3.3**: Report sparsity achieved
- [ ] **Task 32.3.4**: Validate accuracy after pruning

---

## Epic 33: QAT & Quantization Linters (P1 - High Leverage)

*Archived to PRDBacklogArchive.md - 41/41 tasks complete*

---

## Epic 34: Feature Map & Activation Visualization (P2/P3)

*Let users inspect what each layer is "doing" on real data, and compare activations across FP32 vs quantized models.*

**Requires:** Runtime execution, user-provided sample inputs.

### Story 34.1: Activation Capture Pipeline
*Run model and capture intermediate activations.*
- [ ] **Task 34.1.1**: Implement ONNX Runtime activation hook mechanism
- [ ] **Task 34.1.2**: Capture outputs of all intermediate nodes
- [ ] **Task 34.1.3**: Store activations efficiently (memory-mapped for large models)
- [ ] **Task 34.1.4**: Add `--capture-activations` CLI flag
- [ ] **Task 34.1.5**: Accept user sample input (image path, tensor file, etc.)

### Story 34.2: Conv Feature Map Visualization
*Visualize CNN feature maps as heatmaps/grids.*
- [ ] **Task 34.2.1**: Extract per-channel feature maps from Conv outputs
- [ ] **Task 34.2.2**: Generate grid visualization (N channels × spatial)
- [ ] **Task 34.2.3**: Apply colormap (viridis, jet, etc.)
- [ ] **Task 34.2.4**: Add channel-wise statistics (mean, std, sparsity)
- [ ] **Task 34.2.5**: Highlight "dead" channels (all zeros)

### Story 34.3: Activation Distribution Analysis
*Show histograms and statistics per layer.*
- [ ] **Task 34.3.1**: Compute activation histogram per layer
- [ ] **Task 34.3.2**: Detect saturation/clipping (values at extremes)
- [ ] **Task 34.3.3**: Identify potential quantization issues (very wide or skewed distributions)
- [ ] **Task 34.3.4**: Generate distribution comparison chart (FP32 vs INT8)
- [ ] **Task 34.3.5**: Flag layers with high activation divergence after quantization

### Story 34.4: FP32 vs Quantized Comparison
*Side-by-side activation comparison.*
- [ ] **Task 34.4.1**: Run both FP32 and quantized model on same input
- [ ] **Task 34.4.2**: Compute per-layer activation difference (MSE, cosine)
- [ ] **Task 34.4.3**: Highlight layers with largest divergence
- [ ] **Task 34.4.4**: Visualize divergence heatmap on graph
- [ ] **Task 34.4.5**: Generate "Quantization Impact by Layer" report

### Story 34.5: Interactive UI Integration
*Click-to-inspect in Streamlit/HTML.*
- [ ] **Task 34.5.1**: Add "Inspect Activations" button in graph UI
- [ ] **Task 34.5.2**: Click node → show feature maps/histogram popup
- [ ] **Task 34.5.3**: Layer comparison slider (FP32 ↔ INT8)
- [ ] **Task 34.5.4**: Export activation visualizations as images
- [ ] **Task 34.5.5**: Add activation inspection to Streamlit UI

### Story 34.6: Comparison Visualizations
*Charts for multi-model comparison.*
- [ ] **Task 34.6.1**: Accuracy vs Speed scatter plot (Pareto frontier) — *from Epic 12.5.2*
- [ ] **Task 34.6.2**: Per-class metric comparison (radar chart or grouped bars) — *from Epic 12.5.3*
- [ ] **Task 34.6.3**: Size vs Accuracy bubble chart
- [ ] **Task 34.6.4**: Interactive comparison dashboard in Streamlit

---

## Epic 35: TensorRT-Aware Graph Visualization (P2)

*Enhance graph visualization with TRT fusion hints and inspector-quality UX.*

### Story 35.1: TRT Fusion Hints on ONNX Graph
*Show "would be fused" annotations for common TRT fusion patterns.*
- [ ] **Task 35.1.1**: Define common TRT fusion patterns (Conv+BN+ReLU, etc.)
- [ ] **Task 35.1.2**: Detect fusion candidates in ONNX graph
- [ ] **Task 35.1.3**: Annotate graph nodes with "fusible with: [list]"
- [ ] **Task 35.1.4**: Color-code fusible groups in visualization
- [ ] **Task 35.1.5**: Show estimated layer count after TRT optimization

### Story 35.2: TRT Engine Overlay
*Accept TRT engine metadata and overlay on ONNX graph.*
- [ ] **Task 35.2.1**: Parse TRT Engine Explorer JSON exports
- [ ] **Task 35.2.2**: Map TRT layers back to ONNX nodes
- [ ] **Task 35.2.3**: Highlight fused regions (N ONNX → 1 TRT)
- [ ] **Task 35.2.4**: Show precision per fused block
- [ ] **Task 35.2.5**: Display kernel/tactic info on hover

### Story 35.3: Graph UX Improvements (TRT Explorer-Inspired)
*Make the graph visualization best-in-class.*
- [ ] **Task 35.3.1**: Add node search/filter (by type, precision, name)
- [ ] **Task 35.3.2**: Add "show only bottleneck ops" toggle
- [ ] **Task 35.3.3**: Add "show only fused chains" toggle
- [ ] **Task 35.3.4**: Improve zoom/pan smoothness

---

## Epic 37: Hardware Recommendation Engine (P3)

*Help users choose optimal deployment hardware based on model requirements.*

**Current Design**: Assume user runs HaoLine on target deployment hardware. `--hardware auto` detects local GPU.

**Future (with Streamlit UI)**:
- Allow user to specify target hardware from profile list
- Recommend cloud instances (AWS, Azure, GCP) based on model size/latency requirements
- Recommend on-prem hardware based on budget and throughput needs
- Show cost/performance tradeoffs

### Story 37.1: Hardware Recommendation API
- [ ] **Task 37.1.1**: Define user requirements input (latency SLA, throughput, budget)
- [ ] **Task 37.1.2**: Score hardware profiles against requirements
- [ ] **Task 37.1.3**: Rank and return top 3-5 recommendations with rationale
- [ ] **Task 37.1.4**: Include cloud cost estimates ($/hour, $/1M inferences)
- [ ] **Task 37.1.5**: Show "fits in VRAM" / "requires multi-GPU" warnings
- [ ] **Task 37.1.6**: Generate "Deployment Recommendation" section in report — *from Epic 12.6.5*

### Story 37.2: Streamlit Hardware Selector
- [ ] **Task 37.2.1**: Add hardware profile dropdown in UI
- [ ] **Task 37.2.2**: Add "Recommend for me" button with requirements form
- [ ] **Task 37.2.3**: Display recommendation cards with key metrics
- [ ] **Task 37.2.4**: Allow side-by-side hardware comparison
- [ ] **Task 37.2.5**: Show "Your Hardware" vs "Recommended" comparison

---

## Epic 36: Layer-Level Visualization Expansion (P2)

*Leverage per-layer metrics (layers.csv) for rich analysis visualizations in PDF/HTML reports.*

**Note:** Epic 22 (TensorRT) implemented `generate_timing_chart()` and `generate_bound_type_chart()` - these can be generalized for all formats.

### Story 36.1: Layer Waterfall Chart
*Show per-layer latency as stacked horizontal bars (flame graph style).*
- [x] **Task 36.1.1**: Create horizontal waterfall chart from layer timing data *(implemented for TRT in Epic 22)*
- [x] **Task 36.1.2**: Color-code by op type (Conv=blue, MatMul=green, etc.) *(implemented for TRT - color by precision)*
- [ ] **Task 36.1.3**: Show cumulative latency on x-axis
- [ ] **Task 36.1.4**: Add interactive hover for layer details
- [ ] **Task 36.1.5**: Integrate into PDF report (new page)

### Story 36.2: Memory Timeline Chart
*Show activation memory building through network layers.*
- [ ] **Task 36.2.1**: Plot activation memory vs layer index
- [ ] **Task 36.2.2**: Mark peak memory location with annotation
- [ ] **Task 36.2.3**: Show memory "released" after each layer (stacked area)
- [ ] **Task 36.2.4**: Color-code by op type
- [ ] **Task 36.2.5**: Add VRAM limit line for target hardware

### Story 36.3: Layer Efficiency Analysis
*Scatter plots and heatmaps for layer-level efficiency.*
- [ ] **Task 36.3.1**: FLOPs vs Latency scatter plot (identify inefficient layers)
- [ ] **Task 36.3.2**: Params vs Memory scatter plot
- [x] **Task 36.3.3**: Compute/Memory ratio heatmap by layer *(implemented for TRT in Epic 22 - bound_type donut chart)*
- [x] **Task 36.3.4**: Per-layer roofline positioning *(Epic 22 TRT: compute vs memory bound classification)*
- [ ] **Task 36.3.5**: Highlight outlier layers (>2σ from mean)

### Story 36.4: Critical Path Analysis
*Identify and visualize the longest execution path.*
- [ ] **Task 36.4.1**: Compute critical path through graph (DAG longest path)
- [ ] **Task 36.4.2**: Highlight critical path in graph visualization
- [ ] **Task 36.4.3**: Show % of total latency on critical path
- [ ] **Task 36.4.4**: Suggest parallelization opportunities
- [ ] **Task 36.4.5**: Compare critical path: FP32 vs INT8

### Story 36.5: Compute Distribution Sankey
*Flow diagram showing FLOPs distribution through network.*
- [ ] **Task 36.5.1**: Create Sankey diagram: Input → Op Types → Output
- [ ] **Task 36.5.2**: Width proportional to FLOPs
- [ ] **Task 36.5.3**: Color by op type
- [ ] **Task 36.5.4**: Interactive hover for layer names
- [ ] **Task 36.5.5**: Export as SVG for high-quality embedding
- [ ] **Task 35.3.5**: Add minimap for large graphs
- [ ] **Task 35.3.6**: Add keyboard shortcuts (/ to search, h for heatmap)

---

## Epic 38: Docker Distribution (P3)

*Containerized deployment for CI/CD pipelines and non-Python users.*

### Story 38.1: Pre-built Docker Image
- [x] **Task 38.1.1**: Create Dockerfile with all dependencies (`Dockerfile.test` for format reader testing)
- [ ] **Task 38.1.2**: Optimize image size (multi-stage build)
- [ ] **Task 38.1.3**: Add GPU support variant (CUDA base image)
- [ ] **Task 38.1.4**: Publish to Docker Hub / GitHub Container Registry
- [ ] **Task 38.1.5**: Create docker-compose.yml for easy local setup

---

## Epic 40: Full Pydantic Dataclass Migration - **COMPLETE**

*Archived to PRDBacklogArchive.md - 58 classes migrated to Pydantic (v0.5.0)*

---

## Epic 42: Format Conversion Testing (P1)

*Comprehensive test suite for all format conversions in HaoLine.*

**Goal:** Ensure every `to` and `from` conversion path works correctly, preserves metadata, and handles edge cases gracefully.

**Relationship to Epic 49:** This epic tests EXISTING conversions. When Epic 49 adds new paths (HuggingFace → ONNX), add corresponding tests here.

**Conversion Matrix:**
```
              TO →
FROM ↓     | ONNX | TRT | TFLite | CoreML | OpenVINO | SafeTensors
-----------+------+-----+--------+--------+----------+------------
ONNX       |  -   | ✅  | ✅     | ✅     | ✅       | 🔨
TensorRT   | ⛔   |  -  | ⛔     | ⛔     | ⛔       | ⛔
TFLite     | ✅   | ⛔  |  -     | ⛔     | ⛔       | ⛔
CoreML     | ⚠️   | ⛔  | ⛔     |  -     | ⛔       | ⛔
OpenVINO   | ⚠️   | ⛔  | ⛔     | ⛔     |  -       | ⛔
PyTorch    | ✅   | →   | →      | ✅     | →        | 🔨
TensorFlow | ✅   | →   | ✅     | ✅     | →        | ⛔
JAX        | ✅   | →   | →      | →      | →        | ⛔
```

**Legend:**
| Symbol | Meaning | Test Status |
|--------|---------|-------------|
| ✅ | **Implemented & working** | Needs test |
| ⚠️ | **Implemented but lossy** (some data lost) | Needs test |
| 🔨 | **Planned** - Epic exists, not yet built | Blocked |
| → | **Via ONNX** - convert to ONNX first | Needs test |
| ⛔ | **Not feasible** - compiled/proprietary format, no export path | N/A |
| - | Same format (no conversion needed) | N/A |

**Notes:**
- TensorRT engines are compiled binaries - cannot be converted TO other formats
- TFLite→ONNX now supported via `tflite2onnx` (some ops may be unsupported)
- ONNX→TFLite is **BLOCKED** - both onnx2tf and onnx-tf are broken with TF 2.16+/Keras 3.x
- SafeTensors is weights-only, needs architecture info for full model export

### Story 42.1: ONNX Hub Conversions
*Test ONNX as the interchange format (most common path).*

**ONNX → Other Formats:**
- [x] **Task 42.1.1**: Test ONNX → TensorRT conversion ✅ **COMPLETE** (4 tests pass)
- [ ] **Task 42.1.2**: Test ONNX → TFLite conversion ⛔ **BLOCKED** (onnx2tf/onnx-tf broken with TF 2.16+)
- [x] **Task 42.1.3**: Test ONNX → CoreML conversion ✅ **COMPLETE** (test written, skips if no coremltools)
- [x] **Task 42.1.4**: Test ONNX → OpenVINO conversion ✅ **COMPLETE** (test written, skips if no openvino)

**Other Formats → ONNX:**
- [x] **Task 42.1.5**: Test TFLite → ONNX conversion ✅ **COMPLETE** (tflite2onnx added)
- [x] **Task 42.1.6**: Test CoreML → ONNX conversion ✅ **COMPLETE** (test written, skips if no coremltools)
- [x] **Task 42.1.7**: Test OpenVINO → ONNX conversion ✅ **COMPLETE** (test written, skips if no openvino)

### Story 42.2: Framework-to-ONNX Conversions ✅ **FULLY UNBLOCKED**
*Test native framework exports to ONNX.*

**PyTorch → ONNX:**
- [x] **Task 42.2.1**: Test PyTorch → ONNX with simple CNN model ✅ **COMPLETE**
- [ ] **Task 42.2.2**: Test PyTorch → ONNX with Ultralytics YOLO model
- [x] **Task 42.2.3**: Test PyTorch → ONNX with transformer model (attention patterns) ✅ **COMPLETE**

**TensorFlow/Keras → ONNX:**
- [ ] **Task 42.2.4**: Test TensorFlow SavedModel → ONNX conversion
- [ ] **Task 42.2.5**: Test Keras .h5 → ONNX conversion
- [ ] **Task 42.2.6**: Test TensorFlow frozen graph → ONNX conversion

**JAX → ONNX:**
- [ ] **Task 42.2.7**: Test JAX/Flax → ONNX with simple MLP
- [ ] **Task 42.2.8**: Test JAX → ONNX with custom apply function

### Story 42.3: Multi-Hop & Direct Conversions
*Test conversions that go through ONNX as intermediary or direct paths.*

**PyTorch Multi-Hop:**
- [ ] **Task 42.3.1**: Test PyTorch → TFLite (via ONNX) ✅ **UNBLOCKED**
- [ ] **Task 42.3.2**: Test PyTorch → CoreML (via coremltools direct) ✅ **UNBLOCKED**
- [ ] **Task 42.3.3**: Test PyTorch → OpenVINO (via ONNX) ✅ **UNBLOCKED**
- [x] **Task 42.3.4**: Test PyTorch → TensorRT (via ONNX) ✅ **COMPLETE** (2 tests pass)

**TensorFlow Multi-Hop:**
- [ ] **Task 42.3.5**: Test TensorFlow → TFLite (direct tf.lite.TFLiteConverter) ✅ **UNBLOCKED**
- [ ] **Task 42.3.6**: Test TensorFlow → CoreML (via coremltools) ✅ **UNBLOCKED**
- [ ] **Task 42.3.7**: Test TensorFlow → OpenVINO (via ONNX) ✅ **UNBLOCKED**

**JAX Multi-Hop:**
- [ ] **Task 42.3.8**: Test JAX → TFLite (via ONNX) ✅ **UNBLOCKED**
- [ ] **Task 42.3.9**: Test JAX → CoreML (via ONNX) ✅ **UNBLOCKED**

### Story 42.4: ONNX↔TRT Comparison Tests ✅ **UNBLOCKED**
*Test the TensorRT comparison features from Epic 22.*

- [ ] **Task 42.4.1**: Test `--compare-trt` with ResNet ONNX + compiled engine
- [ ] **Task 42.4.2**: Verify fusion detection accuracy (Conv+BN+ReLU → single kernel)
- [ ] **Task 42.4.3**: Verify precision change detection (FP32 → FP16/INT8)
- [ ] **Task 42.4.4**: Test layer rewrite detection (FlashAttention, GELU)
- [ ] **Task 42.4.5**: Test HTML comparison report generation
- [ ] **Task 42.4.6**: Test quantization bottleneck analysis accuracy

### Story 42.5: Round-Trip and Metadata Validation
*Verify conversions preserve essential information.*

- [x] **Task 42.5.1**: Create test harness for conversion round-trips ✅
- [x] **Task 42.5.2**: Test ONNX → TFLite → ONNX round-trip ✅ (skips on TF/Keras 3.x compat issues)
- [x] **Task 42.5.3**: Test ONNX → CoreML → ONNX round-trip (measures lossy delta) ✅
- [x] **Task 42.5.4**: Validate op_type_counts preserved across conversions ✅
- [x] **Task 42.5.5**: Validate precision_breakdown preserved across conversions ✅
- [ ] **Task 42.5.6**: Test conversion error handling (unsupported ops, invalid models)
- [x] **Task 42.5.7**: Verify param_counts match before/after conversion (within tolerance) ✅

### Story 42.6: Weight Export Tests
*Test weight extraction and SafeTensors export paths.*

- [ ] **Task 42.6.1**: Test PyTorch state_dict → SafeTensors export
- [ ] **Task 42.6.2**: Test ONNX initializers → SafeTensors export
- [ ] **Task 42.6.3**: Verify tensor name preservation
- [ ] **Task 42.6.4**: Verify dtype preservation (fp16, bf16, int8)

**Task Summary:** 38 total tasks (all unblocked! 8 require TRT GPU)

---

## Epic 43: Performance & Scalability (P3)

*Handle very large models (70B+ params, 20k+ ops) efficiently. Identified gap from Deep Research analysis.*

**Context:** Python analysis works well for typical models but may struggle with extremely large LLMs. Optimize critical paths and consider async/streaming analysis.

### Story 43.1: Large Model Analysis Optimization
*Speed up analysis for models with millions of parameters and thousands of ops.*

- [ ] **Task 43.1.1**: Profile analysis pipeline with 70B parameter model (identify bottlenecks)
- [ ] **Task 43.1.2**: Implement lazy tensor loading (don't load all weights into memory)
- [ ] **Task 43.1.3**: Add streaming progress updates for long analyses
- [ ] **Task 43.1.4**: Implement chunked parameter counting (process in batches)
- [ ] **Task 43.1.5**: Add `--quick-scan` mode (metadata + op counts only, skip deep analysis)

### Story 43.2: Memory Efficiency
*Reduce memory footprint for analyzing large models.*

- [ ] **Task 43.2.1**: Use memory-mapped file access for weight tensors
- [ ] **Task 43.2.2**: Implement tensor shape extraction without loading weights
- [ ] **Task 43.2.3**: Add memory budget option (`--max-memory 4GB`)
- [ ] **Task 43.2.4**: Profile peak memory usage during analysis

### Story 43.3: Activation Memory Liveness Analysis
*More precise peak activation memory estimation by modeling tensor lifetimes.*

- [ ] **Task 43.3.1**: Build tensor lifetime graph (when each tensor is created/consumed)
- [ ] **Task 43.3.2**: Implement topological execution order estimation
- [ ] **Task 43.3.3**: Calculate memory "watermark" at each execution point
- [ ] **Task 43.3.4**: Report true peak activation memory (not worst-case sum)
- [ ] **Task 43.3.5**: Visualize memory timeline in HTML report *(groundwork in Epic 22 TRT: per-layer workspace tracking)*

---

## Epic 44: Expanded Op Type Support (P3)

*Add FLOP/memory formulas for exotic and domain-specific ops. Identified gap from Deep Research analysis.*

**Context:** Current FLOP estimation covers common CNN/Transformer ops well, but falls back to generic estimates for LSTM/RNN, 3D conv, and domain-specific ops.

### Story 44.1: Recurrent Network Ops
*Support LSTM, GRU, and RNN analysis.*

- [ ] **Task 44.1.1**: Add FLOP formula for LSTM op (4 gates × hidden × seq_len)
- [ ] **Task 44.1.2**: Add FLOP formula for GRU op (3 gates × hidden × seq_len)
- [ ] **Task 44.1.3**: Add FLOP formula for RNN op (basic recurrent)
- [ ] **Task 44.1.4**: Add memory estimation for hidden states
- [ ] **Task 44.1.5**: Detect bidirectional variants (2x FLOPs)

### Story 44.2: 3D and Specialized Convolutions
*Support volumetric and specialized conv ops.*

- [ ] **Task 44.2.1**: Add FLOP formula for Conv3D (video, medical imaging)
- [ ] **Task 44.2.2**: Add FLOP formula for ConvTranspose (decoders, upsampling)
- [ ] **Task 44.2.3**: Add FLOP formula for DepthwiseConv (MobileNet-style)
- [ ] **Task 44.2.4**: Add FLOP formula for GroupConv (ResNeXt-style)
- [ ] **Task 44.2.5**: Add FLOP formula for DeformableConv (object detection)

### Story 44.3: Pooling and Reduction Refinements
*More accurate estimates for pooling and reduce ops.*

- [ ] **Task 44.3.1**: Refine pooling FLOP estimates (currently treated as elementwise)
- [ ] **Task 44.3.2**: Add memory estimates for global pooling (reduces tensor size)
- [ ] **Task 44.3.3**: Add FLOP formula for ReduceL1, ReduceL2 (norm operations)
- [ ] **Task 44.3.4**: Document which ops have accurate vs estimated FLOPs

---

## ~~Epic 45: UI Demo Polish~~ → MERGED INTO EPIC 11

*Sample model preloading merged into Epic 11 (Streamlit Web UI). Visual risk indicators and comparison polish already covered by Epic 41.*

**New Story 11.4 (in Epic 11):** Sample Model Preloading ✅ **COMPLETE**
- [x] Bundle 3 demo models (MNIST, SqueezeNet, EfficientNet-Lite4)
- [x] Add "Try a demo model" buttons in Streamlit
- [x] Download + analyze demo models on demand

---

## ~~Epic 46: Enhanced Model Diff~~ → MERGED INTO EPIC 18

*Layer-level diff and visual diff merged into Epic 18 (Universal IR). Structural comparison tools (`diff()`, `is_structurally_equal()`) already exist in `universal_ir.py`.*

**New Story 18.7 (in Epic 18):** Enhanced Diff Visualization (PARTIAL)
- [x] Basic `diff()` exists (op counts, param diffs, missing nodes)
- [x] `is_structurally_equal()` implemented
- [ ] Layer-level alignment by name (not yet)
- [ ] added/removed/modified layer categorization (partial)
- [ ] Visualize diff in graph view (green=added, red=removed, yellow=modified)
- [ ] Generate "what changed" summary paragraph

**New Story 18.8 (in Epic 18):** Universal IR Documentation & Honest Assessment
*Document what the Universal IR can and cannot do for transparency.*

- [ ] **Task 18.8.1**: Add Universal IR capabilities section to Architecture.md
- [ ] **Task 18.8.2**: Document fidelity rating per format (High/Medium/Low with explanation)
- [ ] **Task 18.8.3**: Document limitations: no semantic equivalence, static-only, no fusion awareness
- [ ] **Task 18.8.4**: Add "Format Fidelity" table to README.md (ONNX=High, SafeTensors=Low, etc.)
- [ ] **Task 18.8.5**: Add limitations disclaimer to Streamlit UI ("Static analysis only")
- [ ] **Task 18.8.6**: Document what each format CAN provide (graph vs params vs metadata only)

---

## Epic 47: Model Card & Standards Compliance (P4)

*Align output with industry standards. Report navigation already covered by Epic 41.*

### Story 47.1: Model Card Toolkit Compliance
*Align output with Google's Model Card Toolkit schema for interoperability.*

- [ ] **Task 47.1.1**: Research Model Card Toolkit JSON schema
- [ ] **Task 47.1.2**: Map HaoLine fields to Model Card fields
- [ ] **Task 47.1.3**: Add `--model-card-format` CLI flag for MCT-compliant output
- [ ] **Task 47.1.4**: Include MCT required fields (intended use, limitations, ethical considerations)
- [ ] **Task 47.1.5**: Generate Model Card Toolkit-compatible JSON export

### Story 47.2: Industry Standard Metrics
*Add metrics used by model benchmarking leaderboards.*

- [ ] **Task 47.2.1**: Add Params/FLOP efficiency ratio
- [ ] **Task 47.2.2**: Add memory efficiency score (params per MB)
- [ ] **Task 47.2.3**: Add compute density metric (FLOPs per parameter)
- [ ] **Task 47.2.4**: Compare against reference architectures (vs ResNet50, vs BERT-base)
- [ ] **Task 47.2.5**: Add efficiency percentile ranking

---

## ~~Epic 48: Quantization Efficiency~~ → MERGED INTO EPIC 33

*Quantization efficiency scoring merged into Epic 33 (QAT Linters). Readiness scoring and recommendations already exist. Efficiency metrics extend that work.*

**New Story 33.6 (in Epic 33):** Quantization Efficiency Scoring
*Note: We have quantization DETECTION + LINTING done (is_quantized, quantized_ops, lint warnings). This story is about EFFICIENCY metrics.*
- [ ] Calculate compression ratio (original size / quantized size)
- [ ] Estimate theoretical speedup from precision reduction
- [ ] Add quantization-specific "efficiency score" (0-100)
- [ ] Detect over-quantization (INT4 on accuracy-sensitive layers)
- [ ] Generate quantization efficiency report

---

## Dec 2025 Deep Research Analysis Integration

*Tasks from deep research codebase analysis (Dec 11, 2025) were feathered into existing epics:*

### Output Parity & CLI Optimization
- **Story 41.7** (Epic 41): Output Parity Gap Closure - 6 tasks
- **Story 50.2** (Epic 50): Dependency Prompting - 3 new tasks (50.2.9-50.2.11)

### Universal IR Documentation
- **Story 18.8** (Epic 18): Universal IR Honest Assessment - 6 tasks
  - Document capabilities vs limitations
  - Format fidelity ratings (High/Medium/Low)
  - Static analysis disclaimer

### Format-Aware UX
- **Story 49.2** (Epic 49): Format-Aware UI/CLI - 3 new tasks (49.2.7-49.2.9)
  - "Feature unavailable" messaging
  - Help tooltips explaining limitations
  - Format capabilities report section

### What Was Already Complete (Research Missed)
- HuggingFace Spaces deployment (Epic 11) - LIVE
- Streamlit UI with core features (Epic 11) - COMPLETE
- Hardware profiling (Epic 6) - COMPLETE
- LLM summarization (Epic 7) - COMPLETE
- Compare mode (Epic 6) - COMPLETE
- 9 format readers (Epics 19-24) - COMPLETE or in progress

### Remaining CPU Limitations (Cannot Address)
- TensorRT engine analysis requires GPU
- Runtime benchmarking requires GPU
- These are documented in format capabilities matrix

---

### Deep Research Analysis #2: Data vs Output Parity Audit (Dec 11, 2025)

*Second deep research pass identified specific UI gaps where extracted data isn't surfaced:*

**Parity Gaps Identified:**

| Gap | Current State | Fix Location |
|-----|---------------|--------------|
| Per-layer breakdowns | Only CSV download, no in-app table | Story 11.5 |
| Architecture classification | Stored but not shown | Task 11.5.4 |
| Format-specific metadata | Extracted but not rendered | Tasks 20.2.7, 24.2.3 |
| Quantization lint/advice | CLI only, no UI | Story 11.6 |
| GGUF/LLM charts | Not implemented | Story 24.2 |
| Operational profiling | No frontend viz | Epic 51.4 (GPU required) |
| Format uploader gaps | TFLite/CoreML/OpenVINO/GGUF not in picker | Tasks 20.2.5, 21.2.5, 23.2.5, 24.2.1 |

**New Stories Added:**
- **Story 11.5**: Layer Summary In-App Table (6 tasks)
- **Story 11.6**: Quantization Lint/Advice Display (6 tasks)
- **Story 24.2**: GGUF UI Features (+2 tasks)
- **Story 20.2, 21.2, 23.2**: Format uploader integration (flagged as priority)

**Executive Assessment (from research):**
> "With ~75% of roadmap completed, HaoLine is essentially ready for primetime. The core analysis engine and UI are fully developed, and user-facing parity is nearly achieved."