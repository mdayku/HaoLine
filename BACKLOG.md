# HaoLine (皓线) - Project Backlog

*Universal model analysis and inspection platform. See what's really inside your models.*

**Related Documents:**
- [PRD.md](PRD.md) - Product requirements and specifications
- [BRAINLIFT.md](BRAINLIFT.md) - Daily learning logs
- [Architecture.md](Architecture.md) - System design details

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
| Epic 11: Streamlit Web UI | In Progress | 3 | 16/17 | P0 |
| Epic 12: Eval Import & Comparison | In Progress | 7 | 13/36 | P1 |
| Epic 13-17: MLOps Platform | Future | 5 | 0/? | P5 |
| Epic 18: Universal IR | Not Started | 3 | 0/12 | P1 |
| Epic 19: SafeTensors | In Progress | 2 | 4/10 | P2 |
| Epic 20: CoreML | In Progress | 2 | 5/12 | P2 |
| Epic 21: TFLite | In Progress | 2 | 5/12 | P2 |
| Epic 22: TensorRT Engine Introspection | Not Started | 6 | 0/34 | P3 |
| Epic 23: OpenVINO | In Progress | 2 | 4/10 | P3 |
| Epic 24: GGUF | In Progress | 2 | 4/8 | P3 |
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
| Epic 33: QAT Linters | Not Started | 4 | 0/22 | **P1** |
| Epic 34: Activation Visualization | Not Started | 5 | 0/25 | P2/P3 |
| Epic 39: Pydantic Schema Migration | Not Started | 3 | 0/15 | P1 |
| Epic 35: TRT-Aware Graph UX | Not Started | 3 | 0/16 | P3 |
| Epic 36: Layer Visualization | Not Started | 5 | 0/25 | **P2** |
| Epic 37: Hardware Recommender | Not Started | 2 | 0/10 | P3 |
| Epic 38: Docker Distribution | Not Started | 1 | 0/5 | P3 |

---

## Epic 1: Environment Setup (COMPLETE - 11/11)

- [x] Fork and build ONNX Runtime
- [x] Build Python wheel (`onnxruntime_gpu-1.24.0`)
- [x] Codebase familiarization
- [x] Project scaffolding

*Note: Task "Add to ORT build system" removed - this is our IP, not donating to Microsoft.*

---

## Epic 2: Core Analysis Engine (COMPLETE - 17/17)

- [x] ONNX Graph Loader
- [x] Parameter Counting (with shared weights, quantized params)
- [x] FLOP Estimation (Conv, MatMul, Attention)
- [x] Memory Estimation (activations, KV cache)

---

## Epic 3: Pattern Analysis (COMPLETE - 9/9)

- [x] Block Detection (Conv-BN-ReLU, Residual, Transformer)
- [x] Risk Heuristics (deep networks, dynamic shapes, oversized layers)

---

## Epic 4: CLI and Output (COMPLETE - 18/18)

- [x] CLI Implementation (argparse, progress, error handling)
- [x] JSON Output (schema validation)
- [x] Markdown Output (model cards)
- [x] HTML Report (full parity)

---

## Epic 4B: PyTorch Integration (COMPLETE - 14/14)

- [x] PyTorch to ONNX Conversion
- [x] Dataset/Class Metadata Extraction (Ultralytics, output shape inference)

---

## Epic 4C: TensorFlow and Keras Conversion (COMPLETE - 15/15)

*Big user base, enterprise adoption.*

### Story 4C.1: TensorFlow to ONNX Conversion - **COMPLETE**
- [x] **Task 4C.1.1**: Add `--from-tensorflow` CLI flag with SavedModel path argument
- [x] **Task 4C.1.2**: Implement TensorFlow SavedModel loading
- [x] **Task 4C.1.3**: Integrate tf2onnx conversion with sensible defaults
- [x] **Task 4C.1.4**: Support frozen graph (.pb) files (--from-frozen-graph, --tf-inputs, --tf-outputs)
- [x] **Task 4C.1.5**: Handle conversion errors gracefully (missing tf, export failures)
- [x] **Task 4C.1.6**: Add tests for TensorFlow conversion flow (12 tests)

### Story 4C.2: Keras to ONNX Conversion - **COMPLETE**
- [x] **Task 4C.2.1**: Add `--from-keras` CLI flag with .h5/.keras path argument
- [x] **Task 4C.2.2**: Implement Keras model loading (Sequential, Functional, Subclassed)
- [x] **Task 4C.2.3**: Convert via tf2onnx CLI for robustness
- [x] **Task 4C.2.4**: Support both .h5 and .keras formats
- [x] **Task 4C.2.5**: Add tests for Keras conversion flow

### Story 4C.3: JAX/Flax to ONNX Conversion - **COMPLETE**
- [x] **Task 4C.3.1**: Add `--from-jax` CLI flag
- [x] **Task 4C.3.2**: Implement JAX -> TF SavedModel -> ONNX pipeline via jax2tf
- [x] **Task 4C.3.3**: Support .msgpack, .pkl, .npy params formats
- [x] **Task 4C.3.4**: Support Flax modules via --jax-apply-fn module:function pattern

---

## Epic 5: Visualization Module (COMPLETE - 52/52 tasks)

*Expanded for LLM-scale models (70B+ params, 80+ layers, 20k+ ops)*

### Story 5.1: Chart Infrastructure - **COMPLETE**
- [x] **Task 5.1.1**: Set up matplotlib with Agg backend
- [x] **Task 5.1.2**: Create consistent chart styling/theme (ChartTheme dataclass, dark theme)
- [x] **Task 5.1.3**: Implement asset directory management
- [x] **Task 5.1.4**: Add graceful fallback when matplotlib unavailable

### Story 5.2: Individual Charts - **COMPLETE**
- [x] **Task 5.2.1**: Implement operator type histogram
- [x] **Task 5.2.2**: Implement layer depth profile (cumulative params/FLOPs)
- [x] **Task 5.2.3**: Implement parameter distribution chart (pie chart)
- [x] **Task 5.2.4**: Implement FLOPs distribution chart
- [x] **Task 5.2.5**: Implement complexity summary dashboard (3-panel)

### Story 5.3: Report Integration - **COMPLETE**
- [x] **Task 5.3.1**: Embed charts in Markdown output
- [x] **Task 5.3.2**: Add chart captions and descriptions
- [x] **Task 5.3.3**: Support HTML output with embedded images (base64, single shareable file)
- [x] **Task 5.3.4**: Support PDF output (Playwright-based, --out-pdf flag)

### Story 5.4: LLM-Scale Pattern Detection - **COMPLETE**
*Must handle 70B+ param models with 80+ transformer layers*
- [x] **Task 5.4.1**: Detect attention patterns (Q/K/V projections, Softmax, Output proj)
- [x] **Task 5.4.2**: Detect MLP/FFN patterns (up-proj, activation, down-proj, SwiGLU)
- [x] **Task 5.4.3**: Detect embedding patterns (token embed, position embed, RoPE/sinusoidal)
- [x] **Task 5.4.4**: Detect normalization placement (pre-norm vs post-norm)
- [x] **Task 5.4.5**: Detect repetition - "N identical blocks" → collapse with ×N count
- [x] **Task 5.4.6**: Add `AttentionHead`, `MLPBlock`, `PositionEncoding`, `MoERouter` types
- [x] **Task 5.4.7**: Handle MoE (Mixture of Experts) routing patterns (TopK detection)
- [x] **Task 5.4.8**: Tests with BERT, GPT-2, LLaMA (test_llm_patterns.py with mini models + model zoo tests)

### Story 5.5: Op Type Icon System and Visual Vocabulary - **COMPLETE**
*180+ ONNX ops → ~20 visual categories*
- [x] **Task 5.5.1**: Define icon/shape for each op category (23 categories)
- [x] **Task 5.5.2**: Map all 180 ONNX ops to visual categories (165 mapped)
- [x] **Task 5.5.3**: Define size scaling function (FLOPs → node size, log scale)
- [x] **Task 5.5.4**: Define color mapping (compute intensity, precision, memory)
- [x] **Task 5.5.5**: Create SVG icon set for embedding in HTML
- [x] **Task 5.5.6**: Add legend/key to visualization output

*Op Category Table:*
| Category | Ops | Shape |
|----------|-----|-------|
| Conv | Conv, ConvTranspose | ◼️ Square |
| Linear | MatMul, Gemm | ◆ Diamond |
| Attention | (pattern) | ◉ Target |
| Norm | BatchNorm, LayerNorm, etc. | ▬ Bar |
| Activation | Relu, GELU, Softmax, etc. | ⚡ Bolt |
| Pool | MaxPool, AvgPool, Global* | ▼ Down |
| Residual | Add (skip connection) | ⊕ Plus |
| Reshape | Reshape, Transpose, etc. | 🔄 Transform |
| Slice | Gather, Slice, Split | ✂️ Cut |
| Concat | Concat | ⊕ Join |
| Reduce | ReduceSum, ReduceMean | Σ Sigma |
| Elementwise | Add, Mul, Div, etc. | ± Math |
| Embed | Gather on weights | 📖 Lookup |
| KV Cache | Concat on seq dim | 💾 Cache |

### Story 5.6: Edge-Centric Visualization - **COMPLETE**
*Edges show tensor flow - THE key insight for bottleneck detection*
- [x] **Task 5.6.1**: Calculate tensor size at every edge (shape × dtype bytes)
- [x] **Task 5.6.2**: Map edge thickness to tensor size (log scale for LLMs)
- [x] **Task 5.6.3**: Color edges by precision (fp32=blue, fp16=green, int8=yellow, bf16=purple)
- [x] **Task 5.6.4**: Highlight memory bottleneck edges (red for top 20%)
- [x] **Task 5.6.5**: Show tensor shape on hover: "[batch, seq, hidden]"
- [x] **Task 5.6.6**: Detect and highlight skip connections (dashed lines)
- [x] **Task 5.6.7**: Calculate peak memory point in graph (memory profile)
- [x] **Task 5.6.8**: For attention: detect O(seq²) edges (is_attention_qk flag)

### Story 5.7: Interactive Hierarchical Graph Visualization - **COMPLETE**
*Depends on: 5.4 (patterns), 5.5 (icons), 5.6 (edges) - ALL DONE*
- [x] **Task 5.7.1**: Build hierarchical graph data structure (Model → Layers → Blocks → Ops)
- [x] **Task 5.7.2**: Implement D3.js or Cytoscape.js renderer (D3.js in HTML export)
- [x] **Task 5.7.3**: Default view: collapsed (Input → [Block×N] → Output)
- [x] **Task 5.7.4**: Click-to-expand: show internal ops of any block
- [x] **Task 5.7.5**: Pan/zoom for large graphs (d3-zoom)
- [x] **Task 5.7.6**: Search by op type, layer name, or tensor name (search input + highlightAndFocusNode)
- [x] **Task 5.7.7**: Export as standalone HTML (self-contained, shareable)
- [x] **Task 5.7.8**: Integrate with existing HTML report (--include-graph flag embeds via iframe)
- [x] **Task 5.7.9**: Performance: handle 20k+ nodes via virtualization/culling (performanceMode, node limit)

### Story 5.8: Per-Layer Summary Table - **COMPLETE**
- [x] **Task 5.8.1**: Create per-layer summary table (LayerSummary, LayerSummaryBuilder in layer_summary.py)
- [x] **Task 5.8.2**: Add sortable/filterable table to HTML report (generate_html_table with JS)
- [x] **Task 5.8.3**: Click row to highlight in graph visualization (layer-selected custom event)
- [x] **Task 5.8.4**: Export table as CSV (--layer-csv flag, LayerSummary.to_csv())

---

## Epic 6: Hardware Profiles and Compare Mode (P3) - **COMPLETE**

### Story 6.1: Hardware Profile System - **COMPLETE**
- [x] **Task 6.1.1**: Define hardware profile dataclass (HardwareProfile)
- [x] **Task 6.1.2**: Create comprehensive profile library (30+ profiles)
- [x] **Task 6.1.3**: Implement profile loading and auto-detection via nvidia-smi
- [x] **Task 6.1.4**: Add CLI flags (--hardware, --list-hardware, --precision, --batch-size)

### Story 6.2: Hardware Estimates - **COMPLETE**
- [x] **Task 6.2.1**: Implement VRAM requirement estimation
- [x] **Task 6.2.2**: Implement theoretical latency bounds
- [x] **Task 6.2.3**: Estimate compute utilization (roofline-based)
- [x] **Task 6.2.4**: Identify bottleneck (compute vs memory vs vram)
- [x] **Task 6.2.5**: Add GPU Saturation metric

### Story 6.3: Compare Mode CLI - **COMPLETE**
- [x] **Task 6.3.1**: Implement multi-model argument parsing
- [x] **Task 6.3.2**: Load and validate eval metrics JSONs
- [x] **Task 6.3.3**: Verify architecture compatibility
- [x] **Task 6.3.4**: Compute deltas vs baseline

### Story 6.4: Quantization Impact Report (TRT EngineXplorer-inspired) - **COMPLETE**
- [x] **Task 6.4.1**: Generate comparison JSON schema
- [x] **Task 6.4.2**: Create comparison Markdown table
- [x] **Task 6.4.3**: Add trade-off analysis section
- [x] **Task 6.4.4**: Add layer-wise precision breakdown visualization
- [x] **Task 6.4.5**: Show accuracy vs speedup tradeoff chart
- [x] **Task 6.4.6**: Display memory savings per layer analysis
- [x] **Task 6.4.7**: Add engine summary panel
- [x] **Task 6.4.8**: Show quantization calibration recommendations

### Story 6.10: Multi-Model Comparison Report - **COMPLETE**
*Compare 2+ models side-by-side - different architectures, sizes, or precisions*
*Note: Many tasks completed in Story 6.3/6.4 via model_inspect_compare CLI*
- [x] **Task 6.10.1**: Design comparison report layout (side-by-side vs tabular) - *Done in 6.4*
- [x] **Task 6.10.2**: Implement normalized metrics (FLOPs/param, memory/param, etc.)
- [x] **Task 6.10.3**: Add radar chart comparing key metrics across models
- [x] **Task 6.10.4**: Add bar charts for params, FLOPs, memory comparison - *Done in 6.4*
- [x] **Task 6.10.5**: Generate executive summary ("Model A is 2x faster, 30% smaller") - *Done in 6.4*
- [x] **Task 6.10.6**: Handle different architectures gracefully (apples-to-oranges warning) - *Done in 6.3*
- [x] **Task 6.10.7**: Add `--compare` CLI flag for N models - *Done in 6.3 (model_inspect_compare)*
- [x] **Task 6.10.8**: Generate comparison HTML report - *Done in 6.4*
- [x] **Task 6.10.9**: Generate comparison PDF report - `--out-pdf` flag

### Story 6.11: MOVED to Epic 12 (Story 12.6)
*Model Leaderboard requires inference metrics - moved to Inference Platform epic*

### Story 6.5: Expanded VRAM Variants - **COMPLETE**
- [x] **Task 6.5.1-6.5.8**: All GPU variants added (40+ profiles)

### Story 6.6: Multi-GPU / Cluster Support - **COMPLETE**
- [x] **Task 6.6.1-6.6.6**: Multi-GPU support complete

### Story 6.7: Cloud Instance Profiles - **COMPLETE**
- [x] **Task 6.7.1-6.7.4**: AWS/Azure/GCP profiles complete

### Story 6.8: Resolution and Batch Size Impact Analysis - **COMPLETE**
- [x] **Task 6.8.1**: Add `--input-resolution` CLI flag
- [x] **Task 6.8.2**: Implement resolution scaling impact estimator
- [x] **Task 6.8.3**: Add batch size sweep analysis
- [x] **Task 6.8.4**: Generate resolution/batch vs latency/memory charts
- [x] **Task 6.8.5**: Add resolution recommendations for target hardware

### Story 6.9: Hardware Requirements Recommendations (Steam-style) - **COMPLETE**
- [x] **Task 6.9.1**: Define deployment target categories
- [x] **Task 6.9.2**: Calculate minimum HW spec
- [x] **Task 6.9.3**: Calculate recommended HW spec
- [x] **Task 6.9.4**: Add `--deployment-target` CLI flag
- [x] **Task 6.9.5**: Generate "System Requirements" section
- [x] **Task 6.9.6**: Factor in latency/throughput requirements

---

## Epic 7: LLM Integration (P3 - 5/9 done)

### Story 7.1: LLM Summarizer - **COMPLETE**
- [x] **Task 7.1.1**: Implement API client abstraction
- [x] **Task 7.1.2**: Create prompt templates
- [x] **Task 7.1.3**: Generate short summary
- [x] **Task 7.1.4**: Generate detailed summary
- [x] **Task 7.1.5**: Handle API failures gracefully

### Story 7.2: Secure Config File - **CANCELLED**
*Decision: Environment variables are the standard approach for API keys.*
*CLI now auto-loads .env files and provides clear prompts when keys are missing.*
- [x] ~~**Task 7.2.1**: Read API key from config file~~ → Using env vars + .env auto-load
- [x] ~~**Task 7.2.2**: Add `--api-key` CLI flag~~ → Not needed, env vars work
- [x] ~~**Task 7.2.3**: Priority order~~ → Standard: env var (from shell or .env)
- [x] ~~**Task 7.2.4**: Document config file location~~ → Documented in README

---

## Epic 8: Testing, CI/CD, and Quality (P3) - **COMPLETE**

### Story 8.1: Unit Tests - **COMPLETE**
- [x] **Task 8.1.1-8.1.4**: All unit tests complete

### Story 8.2: Integration Tests - **COMPLETE**
- [x] **Task 8.2.1**: Test CLI end-to-end with ResNet (PDF, graph, LLM summary)
- [x] **Task 8.2.2**: Test CLI end-to-end with BERT (PDF, graph, LLM summary)
- [x] **Task 8.2.3**: Test compare mode with quantized variants (real benchmarks)
- [x] **Task 8.2.4**: Test visualization generation (17 tests)

### Story 8.3: Documentation - **COMPLETE**
- [x] **Task 8.3.1**: Write tool README
- [x] **Task 8.3.2**: Add inline code documentation (844 docstrings across 35 files)
- [x] **Task 8.3.3**: Create example scripts (`autodoc/examples/`)
- [x] **Task 8.3.4**: Document JSON schemas (`docs/marcu/JSON_SCHEMAS.md`)

### Story 8.4: CI/CD Pipeline - **COMPLETE**
- [x] **Task 8.4.1-8.4.6**: GitHub Actions workflow complete

---

## Epic 9: Runtime Profiling and Actual Measurements (P2) - **COMPLETE**

*Replace theoretical estimates with actual ONNX Runtime measurements.*

### Story 9.1: Batch Size Benchmarking - **COMPLETE**
- [x] **Task 9.1.1**: Implement `run_batch_sweep_benchmark()` with ONNX Runtime
- [x] **Task 9.1.2**: Measure actual latency (p50) per batch size
- [x] **Task 9.1.3**: Calculate real throughput from measured latency
- [x] **Task 9.1.4**: Make benchmarking the default (`--no-benchmark` for theoretical)

### Story 9.2: GPU Memory Profiling - **COMPLETE**
- [x] **Task 9.2.1**: Integrate `pynvml` for GPU memory measurement
- [x] **Task 9.2.2**: Track VRAM usage during inference
- [x] **Task 9.2.3**: Measure peak GPU memory per batch size
- [x] **Task 9.2.4**: Add GPU utilization tracking

### Story 9.3: Per-Layer Profiling - **COMPLETE**
- [x] **Task 9.3.1**: Enable ONNX Runtime profiling (`enable_profiling=True`)
- [x] **Task 9.3.2**: Parse profiling JSON output
- [x] **Task 9.3.3**: Identify slowest layers/operators
- [x] **Task 9.3.4**: Generate per-layer timing breakdown chart
- [x] **Task 9.3.5**: Highlight bottleneck layers in graph visualization (via layer_timing_chart)

### Story 9.4: Bottleneck Detection - **COMPLETE**
- [x] **Task 9.4.1**: Compare compute time vs memory transfer time
- [x] **Task 9.4.2**: Classify as compute-bound or memory-bound
- [x] **Task 9.4.3**: Provide optimization recommendations based on bottleneck
- [x] **Task 9.4.4**: Show theoretical vs actual performance gap

### Story 9.5: Resolution Benchmarking - **COMPLETE**
- [x] **Task 9.5.1**: Benchmark actual inference at different resolutions
- [x] **Task 9.5.2**: Measure real throughput scaling with resolution
- [x] **Task 9.5.3**: Find optimal resolution for target latency

### Story 9.6: Multi-Input Model Profiling - **COMPLETE**
*Support profiling for BERT, LLMs, and other multi-input models.*
- [x] **Task 9.6.1**: Detect all model inputs and their shapes/dtypes
- [x] **Task 9.6.2**: Generate appropriate dummy inputs based on dtype (int64 for tokens, float32 for vision)
- [x] **Task 9.6.3**: Support common input patterns (text: input_ids/attention_mask, multimodal: image+text)
- [x] **Task 9.6.4**: Auto-detect sequence length from model or use sensible defaults (128, 512)
- [x] **Task 9.6.5**: Handle dynamic axes gracefully (batch, seq_len)

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

---

## Epic 10B: Standalone Package and Distribution (P0)

*Extract from ORT fork, ship as independent product. This is our IP.*

**New Repository:** [github.com/mdayku/HaoLine](https://github.com/mdayku/HaoLine)

### Story 10B.0: Greenfield Extraction - **COMPLETE**
- [x] **Task 10B.0.1**: Create new GitHub repo (standalone, not ORT fork)
- [x] **Task 10B.0.2**: Copy autodoc modules from `tools/python/util/autodoc/`
- [x] **Task 10B.0.3**: Copy `model_inspect.py` and `model_inspect_compare.py` as CLI entrypoints
- [x] **Task 10B.0.4**: Update all imports to standalone package structure (`from haoline import ...`)
- [x] **Task 10B.0.5**: Remove `ModelProtoWithShapeInfo` import (confirm fallback works)
- [x] **Task 10B.0.6**: Copy test fixtures (sample ONNX models for unit tests)
- [x] **Task 10B.0.7**: Verify all tests pass in new repo (229 passed)
- [x] **Task 10B.0.8**: Update README for standalone usage

### Story 10B.1: Python Wheel Packaging - **COMPLETE**
- [x] **Task 10B.1.1**: Create pyproject.toml with proper metadata (name, version, author, license)
- [x] **Task 10B.1.2**: Configure build system (hatchling recommended)
- [x] **Task 10B.1.3**: Define core dependencies: `onnx>=1.14`, `numpy>=1.20`
- [x] **Task 10B.1.4**: Define optional dependencies:
  - `[runtime]`: onnxruntime
  - `[viz]`: matplotlib
  - `[llm]`: anthropic
  - `[pdf]`: playwright
  - `[gpu]`: nvidia-ml-py
  - `[full]`: all of the above
- [x] **Task 10B.1.5**: Add CLI entrypoints:
  - `haoline` → `haoline.cli:run_inspect`
- [x] **Task 10B.1.6**: Test wheel installation in clean virtual environment
- [x] **Task 10B.1.7**: Publish to TestPyPI first, verify install works
- [x] **Task 10B.1.8**: Publish to PyPI (v0.2.2 live at pypi.org/project/haoline/)

### Story 10B.2: CI/CD Pipeline (New Repo) - **COMPLETE**
- [x] **Task 10B.2.1**: Create GitHub Actions workflow for testing
- [x] **Task 10B.2.2**: Add Black + Ruff linting checks
- [x] **Task 10B.2.3**: Add mypy type checking
- [x] **Task 10B.2.4**: Add pytest with coverage
- [x] **Task 10B.2.5**: Auto-publish to PyPI on release tag
- [x] **Task 10B.2.6**: Add badge shields (build status, coverage, PyPI version) - already in README

### Story 10B.3: Pre-built Docker Image - **MOVED to Epic 38**
*Deferred - no user demand yet. See Epic 38 for Docker distribution.*

### Story 10B.4: Documentation and Branding - **COMPLETE**
- [x] **Task 10B.4.1**: Create standalone README.md with installation instructions
- [x] **Task 10B.4.2**: Add quickstart examples (analyze, compare, profile)
- [x] **Task 10B.4.3**: Document all CLI flags and options (see README CLI section)
- [x] **Task 10B.4.4**: Add architecture overview for contributors (Architecture.md)
- [x] **Task 10B.4.5**: ~~Choose final product name~~ **DONE: HaoLine (皓线)**

---

## Epic 11: Streamlit Web UI (P0)

*Demo-able web interface for non-CLI users.*

### Story 11.1: Basic Streamlit App - **COMPLETE**
- [x] **Task 11.1.1**: Create `streamlit_app.py` with file upload widget
- [x] **Task 11.1.2**: Wire upload to analysis engine
- [x] **Task 11.1.3**: Display HTML report in Streamlit iframe/component (+ interactive graph)
- [x] **Task 11.1.4**: Add hardware profile dropdown selector
- [x] **Task 11.1.5**: Add download buttons (JSON, Markdown, HTML, PDF)

### Story 11.2: Enhanced UI Features - **COMPLETE**
- [x] **Task 11.2.1**: Modern dark theme with emerald accents, high-contrast UI
- [x] **Task 11.2.2**: Hardware dropdown with search and categorization (50+ GPUs)
- [x] **Task 11.2.3**: Add LLM summary toggle with API key input (secure use-and-forget)
- [x] **Task 11.2.4**: Full interactive D3.js graph embedded (matches standalone HTML)
- [x] **Task 11.2.5**: FLOPs-based node sizing (log scale, expensive ops visually larger)
- [x] **Task 11.2.6**: Collapsible sidebar in graph for maximum viewing area
- [x] **Task 11.2.7**: PDF export functionality
- [x] **Task 11.2.8**: Add model comparison tab (upload 2 models, side-by-side metrics)
- [x] **Task 11.2.9**: Add session history (stores last 10 analyses, quick re-select)
- ~~**Task 11.2.10**: Responsive layout for mobile~~ **CANCELLED** (not targeting mobile users)

### Story 11.3: Deployment
- [ ] **Task 11.3.1**: Deploy to Hugging Face Spaces (free, GPU available)
  - [ ] **11.3.1a**: Create HF Space at huggingface.co/new-space (SDK: Streamlit)
  - [ ] **11.3.1b**: Get HF token from huggingface.co/settings/tokens (write access)
  - [ ] **11.3.1c**: Add `HF_TOKEN` secret in GitHub repo settings
  - [ ] **11.3.1d**: Update `HF_SPACE_NAME` in `.github/workflows/deploy-hf-spaces.yml`
  - [ ] **11.3.1e**: Trigger workflow and verify deployment works
- [x] **Task 11.3.2**: ~~Add Streamlit Cloud deployment option~~ **CANCELLED** - HuggingFace Spaces is sufficient
- [x] **Task 11.3.3**: Create deployment documentation - see [DEPLOYMENT.md](DEPLOYMENT.md)
- [x] **Task 11.3.4**: Set up CI/CD for auto-deploy on push - `.github/workflows/deploy-hf-spaces.yml`

---

## Epic 12: Eval Import & Comparison (P1)

*Import eval metrics from existing tools (Ultralytics, HF evaluate, etc.) — don't reinvent the wheel.*
*Combine with architecture analysis for unified accuracy × speed × cost comparison.*

**Philosophy:** Users already have eval pipelines. We import their results, not run inference ourselves.

```
User's Eval Tool → JSON/CSV → HaoLine Import → Unified Report
(Ultralytics, HF evaluate, lm-eval, timm, custom)
```

### Story 12.1: Base Eval Schema - **COMPLETE**
*Task-agnostic fields all eval results share*
- [x] **Task 12.1.1**: Define `EvalResult` base schema (model_id, task_type, timestamp, metrics dict)
- [x] **Task 12.1.2**: Define `EvalMetric` schema (name, value, unit, higher_is_better)
- [x] **Task 12.1.3**: Create `eval_schema.json` for validation (embedded in schemas.py)
- [x] **Task 12.1.4**: Add `haoline import-eval` CLI command skeleton - `src/haoline/eval/cli.py`

### Story 12.2: Task-Specific Schemas - **COMPLETE**
*Standard metrics for common ML tasks*
- [x] **Task 12.2.1**: Detection schema (mAP@50, mAP@50:95, P/R/F1 per class) - `DetectionEvalResult`
- [x] **Task 12.2.2**: Classification schema (top-1, top-5 accuracy, per-class accuracy) - `ClassificationEvalResult`
- [x] **Task 12.2.3**: NLP schema (accuracy, F1, exact_match, BLEU) - `NLPEvalResult`
- [x] **Task 12.2.4**: LLM schema (perplexity, mmlu, hellaswag, truthfulqa) - `LLMEvalResult`
- [x] **Task 12.2.5**: Segmentation schema (mIoU, dice, per-class IoU) - `SegmentationEvalResult`
- [x] **Task 12.2.6**: Generic schema (user-defined metrics) - `GenericEvalResult` in schemas.py

### Story 12.3: Import Adapters
*Parse output from popular eval tools*
- [x] **Task 12.3.1**: Ultralytics adapter (parse YOLO val results) - `adapters.py`
- [ ] **Task 12.3.2**: HuggingFace evaluate adapter (parse evaluate output)
- [ ] **Task 12.3.3**: lm-eval-harness adapter (parse LLM benchmark results)
- [ ] **Task 12.3.4**: timm adapter (parse timm benchmark output)
- [x] **Task 12.3.5**: Generic CSV/JSON adapter (user maps columns to schema) - `adapters.py`
- [ ] **Task 12.3.6**: Auto-detect adapter from file format/contents

### Story 12.4: Merge Eval + Architecture
*Combine imported eval metrics with HaoLine's architecture analysis*
- [ ] **Task 12.4.1**: Link eval results to model files (by path or hash)
- [x] **Task 12.4.2**: Create `CombinedReport` dataclass (architecture + eval) - `schemas.py`
- [ ] **Task 12.4.3**: Handle multiple eval runs per model (aggregate or select)
- [ ] **Task 12.4.4**: Validate eval task matches model type (warn if mismatch)

### Story 12.5: Unified Comparison Report
*Architecture + Eval + Hardware in one view*
- [ ] **Task 12.5.1**: Multi-model comparison table (accuracy, speed, size, cost)
- [ ] **Task 12.5.2**: Accuracy vs Speed scatter plot (Pareto frontier)
- [ ] **Task 12.5.3**: Per-class metric comparison (radar chart or grouped bars)
- [ ] **Task 12.5.4**: Add eval metrics to existing HTML/PDF reports
- [ ] **Task 12.5.5**: Export comparison as CSV/JSON

### Story 12.6: Deployment Cost Calculator
*Answer: "What does it cost to run this model at X fps?"*
- [ ] **Task 12.6.1**: Define deployment scenario inputs (fps, hours/day, cloud vs edge)
- [ ] **Task 12.6.2**: Calculate required hardware tier for latency SLA
- [ ] **Task 12.6.3**: Estimate $/day and $/month for deployment
- [ ] **Task 12.6.4**: Compare cost across precision variants (fp32 vs fp16 vs int8)
- [ ] **Task 12.6.5**: Generate "Deployment Recommendation" section in report
- [x] **Task 12.6.6**: Add `--deployment-fps` and `--deployment-hours` CLI flags

### Story 12.7: YOLO Quantization Demo (Reference Implementation)
*End-to-end demo: fp32 → fp16 → int8, eval all, compare, deploy*
- [x] **Task 12.7.1**: Document YOLO quantization workflow - `docs/YOLO_QUANTIZATION_WORKFLOW.md`
- [ ] **Task 12.7.2**: Run batchtestv1.py on all 3 variants
- [ ] **Task 12.7.3**: Import results with Ultralytics adapter
- [ ] **Task 12.7.4**: Generate comparison report with deployment costs
- [ ] **Task 12.7.5**: Create demo script/notebook for showcase

---

## Epics 13-17: MLOps Platform Vision (P5)

*High-level placeholder. Do not implement until there's demand.*

- **Epic 13**: Cloud Provider Integration (AWS/GCP/Azure)
- **Epic 14**: GPU Orchestration (on-demand spin-up/down)
- **Epic 15**: Training Estimation (dataset → training time)
- **Epic 16**: Model Inventory Management (versions, lineage)
- **Epic 17**: Billing and Usage (metered tracking)

---

## Epic 18: Universal Internal Representation (P1)

*Foundation for format-agnostic analysis. Must be built before other format adapters.*

### Story 18.1: Universal Graph IR
- [ ] **Task 18.1.1**: Design `UniversalGraph` dataclass (nodes, edges, weights, metadata)
- [ ] **Task 18.1.2**: Design `UniversalNode` with op-type abstraction (not tied to ONNX ops)
- [ ] **Task 18.1.3**: Design `UniversalTensor` for weights/activations
- [ ] **Task 18.1.4**: Add source_format tracking and round-trip metadata
- [ ] **Task 18.1.5**: Document IR design decisions in Architecture.md

### Story 18.2: Format Adapter Interface
- [ ] **Task 18.2.1**: Define `FormatAdapter` protocol (can_read, read, can_write, write)
- [ ] **Task 18.2.2**: Implement adapter registry and auto-detection by file extension
- [ ] **Task 18.2.3**: Refactor ONNX loader to use FormatAdapter interface
- [ ] **Task 18.2.4**: Refactor PyTorch loader to use FormatAdapter interface

### Story 18.3: Conversion Matrix
- [ ] **Task 18.3.1**: Define conversion capability enum (FULL, LOSSY, PARTIAL, NONE)
- [ ] **Task 18.3.2**: Implement conversion matrix lookup
- [ ] **Task 18.3.3**: Add CLI flag `--convert-to <format>` for format conversion

---

## Epic 19: SafeTensors Format (P2)

*HuggingFace ecosystem, widely used for LLM weights. Easy win.*

### Story 19.1: SafeTensors Reader - **COMPLETE**
- [x] **Task 19.1.1**: Add safetensors dependency (optional) - in `[formats]` extra
- [x] **Task 19.1.2**: Implement SafeTensorsReader.read() - load tensor dict
- [x] **Task 19.1.3**: Extract metadata (tensor names, shapes, dtypes)
- [x] **Task 19.1.4**: Integrate with analysis pipeline (param counts, memory)
- [ ] **Task 19.1.5**: Test with real SafeTensors model (e.g., HuggingFace LLM weights)
- [ ] **Task 19.1.6**: Write unit tests for SafeTensorsReader

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
- [ ] **Task 20.1.6**: Test with real CoreML model (.mlmodel or .mlpackage)
- [ ] **Task 20.1.7**: Write unit tests for CoreMLReader

### Story 20.2: CoreML Writer
- [ ] **Task 20.2.1**: Implement CoreMLAdapter.write() via coremltools conversion
- [ ] **Task 20.2.2**: Support ONNX → CoreML conversion path
- [ ] **Task 20.2.3**: Support PyTorch → CoreML conversion path
- [ ] **Task 20.2.4**: Add iOS/macOS deployment target options
- [ ] **Task 20.2.5**: Add `--convert-to coreml` CLI flag

---

## Epic 21: TFLite Format (P2)

*TensorFlow Lite for mobile and edge deployment.*

### Story 21.1: TFLite Reader - **COMPLETE**
- [x] **Task 21.1.1**: Add tflite-runtime dependency (optional) - in `[formats]` extra
- [x] **Task 21.1.2**: Implement TFLiteReader.read() - load .tflite
- [x] **Task 21.1.3**: Parse FlatBuffer schema for ops and tensors
- [x] **Task 21.1.4**: Map TFLite ops to op_type_counts
- [x] **Task 21.1.5**: Extract quantization info (int8, float16)
- [ ] **Task 21.1.6**: Test with real TFLite model (e.g., TF Model Garden mobile model)
- [ ] **Task 21.1.7**: Write unit tests for TFLiteReader

### Story 21.2: TFLite Writer
- [ ] **Task 21.2.1**: Implement TFLiteAdapter.write() via tf.lite.TFLiteConverter
- [ ] **Task 21.2.2**: Support ONNX → TFLite conversion (via TF intermediary)
- [ ] **Task 21.2.3**: Add quantization options (dynamic, full int8)
- [ ] **Task 21.2.4**: Add representative dataset hook for calibration
- [ ] **Task 21.2.5**: Add `--convert-to tflite` CLI flag

---

## Epic 22: TensorRT Engine Introspection (P3)

*Deep analysis of NVIDIA TensorRT compiled engines. Inspired by [TRT Engine Explorer](https://github.com/NVIDIA/TensorRT/tree/main/tools/experimental/trt-engine-explorer).*

### Story 22.1: Engine File Loader
*Load .engine/.plan TRT blobs using TensorRT runtime APIs.*
- [ ] **Task 22.1.1**: Add tensorrt dependency (optional, requires NVIDIA GPU)
- [ ] **Task 22.1.2**: Implement `TRTEngineLoader.load()` to deserialize engine files
- [ ] **Task 22.1.3**: Extract engine metadata (TRT version, build flags, calibration info)
- [ ] **Task 22.1.4**: Handle engine compatibility checks (GPU arch, TRT version)
- [ ] **Task 22.1.5**: Support both `.engine` and `.plan` file formats

### Story 22.2: Fused Graph Reconstruction
*Parse the optimized TRT graph and reconstruct the execution plan.*
- [ ] **Task 22.2.1**: Extract layer list from engine (names, types, shapes)
- [ ] **Task 22.2.2**: Identify fused operations (Conv+BN+ReLU → single kernel)
- [ ] **Task 22.2.3**: Detect removed/optimized-away layers
- [ ] **Task 22.2.4**: Extract kernel substitutions (cuDNN vs custom kernels)
- [ ] **Task 22.2.5**: Parse timing cache if present
- [ ] **Task 22.2.6**: Identify precision per layer (FP32/FP16/INT8/TF32)

### Story 22.3: ONNX ↔ TRT Diff View
*Visual comparison between source ONNX and compiled TRT engine.*
- [ ] **Task 22.3.1**: Map TRT layers back to original ONNX nodes
- [ ] **Task 22.3.2**: Highlight fused operations (N ONNX ops → 1 TRT layer)
- [ ] **Task 22.3.3**: Show precision auto-selection decisions
- [ ] **Task 22.3.4**: Visualize layer rewrites (e.g., attention → Flash Attention)
- [ ] **Task 22.3.5**: Display shape changes (dynamic → static binding)
- [ ] **Task 22.3.6**: Generate side-by-side graph comparison HTML

### Story 22.4: TRT Performance Metadata Panel
*Extract and display engine profiling information.*
- [ ] **Task 22.4.1**: Extract per-layer latency (if profiling was enabled)
- [ ] **Task 22.4.2**: Show workspace size allocation per layer
- [ ] **Task 22.4.3**: Display kernel/tactic selection choices
- [ ] **Task 22.4.4**: Identify memory-bound vs compute-bound layers
- [ ] **Task 22.4.5**: Show layer timing breakdown chart
- [ ] **Task 22.4.6**: Extract device memory footprint

### Story 22.5: TRT Engine Summary Block
*Comprehensive summary matching PRD format.*
- [ ] **Task 22.5.1**: Generate engine overview (layers, params, memory)
- [ ] **Task 22.5.2**: Show optimization summary (fusions applied, precision mix)
- [ ] **Task 22.5.3**: Display hardware binding info (GPU arch, compute capability)
- [ ] **Task 22.5.4**: List builder configuration used (max batch, workspace, etc.)

### Story 22.6: ONNX vs TRT Comparison Mode
*Side-by-side analysis showing what changed and performance impact.*
- [ ] **Task 22.6.1**: Load both ONNX source and TRT engine
- [ ] **Task 22.6.2**: Compute layer count delta (before/after fusion)
- [ ] **Task 22.6.3**: Show speedup contributions per optimization
- [ ] **Task 22.6.4**: Display precision changes with accuracy impact notes
- [ ] **Task 22.6.5**: Generate comparison report (JSON/MD/HTML)
- [ ] **Task 22.6.6**: Visualize memory reduction from optimizations

---

## Epic 23: OpenVINO Format (P3)

*Intel's inference toolkit for CPU/GPU/VPU deployment.*

### Story 23.1: OpenVINO Reader - **COMPLETE**
- [x] **Task 23.1.1**: Add openvino dependency (optional) - in `[openvino]` extra
- [x] **Task 23.1.2**: Implement OpenVINOReader.read() - load .xml/.bin
- [x] **Task 23.1.3**: Map OpenVINO ops to layer_type_counts
- [x] **Task 23.1.4**: Extract precision breakdown
- [ ] **Task 23.1.5**: Test with real OpenVINO model (.xml + .bin)
- [ ] **Task 23.1.6**: Write unit tests for OpenVINOReader

### Story 23.2: OpenVINO Writer
- [ ] **Task 23.2.1**: Implement OpenVINOAdapter.write()
- [ ] **Task 23.2.2**: Support ONNX → OpenVINO conversion
- [ ] **Task 23.2.3**: Add precision options
- [ ] **Task 23.2.4**: Add `--convert-to openvino` CLI flag

---

## Epic 24: GGUF Format (P3 - Read-Only)

*llama.cpp format for running LLMs locally.*

### Story 24.1: GGUF Reader - **COMPLETE**
- [x] **Task 24.1.1**: Implement GGUF header parser (pure Python, no deps)
- [x] **Task 24.1.2**: Extract model metadata (arch, context_length, etc.)
- [x] **Task 24.1.3**: Extract quantization type per tensor
- [x] **Task 24.1.4**: Estimate memory footprint (VRAM estimation)
- [ ] **Task 24.1.5**: Test with real GGUF model (e.g., llama.cpp quantized model)
- [ ] **Task 24.1.6**: Write unit tests for GGUFReader

### Story 24.2: GGUF Analysis Features
- [ ] **Task 24.2.1**: Show quantization breakdown (data available, needs UI)
- [ ] **Task 24.2.2**: Estimate VRAM for different context lengths (method exists)

---

## Epic 25: Privacy and Trust Architecture (P1)

*Enterprise customers need proof we won't steal their model IP.*

### Story 25.1: Local-First Architecture
- [x] **Task 25.1.1**: Document "model never leaves your machine" guarantee - see [PRIVACY.md](PRIVACY.md)
- [x] **Task 25.1.2**: Audit code for any network calls or telemetry - only LLM (optional) and test downloads
- [x] **Task 25.1.3**: Add `--offline` CLI flag (skips LLM features)
- [x] **Task 25.1.4**: Create architecture diagram showing data flow - ASCII diagram in PRIVACY.md

### Story 25.2: Output Controls
- [x] **Task 25.2.1**: Add `--redact-names` flag (anonymize layer/tensor names) - `src/haoline/privacy.py`
- [x] **Task 25.2.2**: Add `--summary-only` flag (stats only, no graph structure) - `src/haoline/privacy.py`
- [x] **Task 25.2.3**: Document what information each output format reveals - table in PRIVACY.md

### Story 25.3: Enterprise Trust Documentation
- [x] **Task 25.3.1**: Write Privacy Policy / Data Handling document - expanded in [PRIVACY.md](PRIVACY.md)
- [x] **Task 25.3.2**: Document open-source audit path ("read our code") - in PRIVACY.md

*Future: Confidential Computing (TEE) for cloud analysis - see Epic 10 SaaS.*

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

*Provide "preflight checks" and guidance to help users build QAT/quant-friendly models. Static analysis only - no runtime required.*

**Value Proposition:**
- Tell users if their model is "good quantization material" BEFORE they invest time in QAT
- Catch anti-patterns and mistakes that cause accuracy drops
- Actionable recommendations, not just warnings

### Story 33.1: Quantization-Unfriendly Op Detection
*Detect ops that are commonly problematic in INT8.*
- [ ] **Task 33.1.1**: Build list of quantization-unfriendly ops (custom ops, certain activations)
- [ ] **Task 33.1.2**: Detect dynamic shapes in problematic positions
- [ ] **Task 33.1.3**: Flag ops with no ONNX quantization support
- [ ] **Task 33.1.4**: Identify ops that typically cause large accuracy drops (e.g., LayerNorm, Softmax)
- [ ] **Task 33.1.5**: Generate severity-ranked warning list

### Story 33.2: QAT Graph Validation
*Validate QAT-annotated graphs for correctness.*
- [ ] **Task 33.2.1**: Detect missing fake-quantization nodes
- [ ] **Task 33.2.2**: Check for inconsistent fake-quant placement across branches
- [ ] **Task 33.2.3**: Validate per-tensor vs per-channel quantization consistency
- [ ] **Task 33.2.4**: Flag suspiciously wide activation ranges (suggests calibration issue)
- [ ] **Task 33.2.5**: Detect inconsistent scales/zero points across residual connections

### Story 33.3: Quantization Readiness Score
*Emit an overall "QAT readiness score" with breakdown.*
- [ ] **Task 33.3.1**: Define scoring rubric (op support, graph structure, precision mix)
- [ ] **Task 33.3.2**: Calculate per-layer quantization risk scores
- [ ] **Task 33.3.3**: Aggregate into overall readiness score (0-100)
- [ ] **Task 33.3.4**: Generate "problem layers" list with reasons
- [ ] **Task 33.3.5**: Add `--lint-quantization` CLI flag

### Story 33.4: Actionable Recommendations
*Don't just warn - tell users what to do.*
- [ ] **Task 33.4.1**: Recommend keeping sensitive layers at FP16 (classifier, final conv)
- [ ] **Task 33.4.2**: Suggest fake-quant insertion points for QAT
- [ ] **Task 33.4.3**: Recommend op substitutions (e.g., LayerNorm → RMSNorm for INT8)
- [ ] **Task 33.4.4**: Suggest per-channel vs per-tensor for specific layers
- [ ] **Task 33.4.5**: Generate "QAT Readiness Report" (Markdown/HTML)
- [ ] **Task 33.4.6**: Integrate recommendations into compare mode (FP32 vs INT8)

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

---

## Epic 35: TensorRT-Aware Graph Visualization (P3)

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

### Story 37.2: Streamlit Hardware Selector
- [ ] **Task 37.2.1**: Add hardware profile dropdown in UI
- [ ] **Task 37.2.2**: Add "Recommend for me" button with requirements form
- [ ] **Task 37.2.3**: Display recommendation cards with key metrics
- [ ] **Task 37.2.4**: Allow side-by-side hardware comparison
- [ ] **Task 37.2.5**: Show "Your Hardware" vs "Recommended" comparison

---

## Epic 36: Layer-Level Visualization Expansion (P2)

*Leverage per-layer metrics (layers.csv) for rich analysis visualizations in PDF/HTML reports.*

### Story 36.1: Layer Waterfall Chart
*Show per-layer latency as stacked horizontal bars (flame graph style).*
- [ ] **Task 36.1.1**: Create horizontal waterfall chart from layer timing data
- [ ] **Task 36.1.2**: Color-code by op type (Conv=blue, MatMul=green, etc.)
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
- [ ] **Task 36.3.3**: Compute/Memory ratio heatmap by layer
- [ ] **Task 36.3.4**: Per-layer roofline positioning
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
- [ ] **Task 38.1.1**: Create Dockerfile with all dependencies
- [ ] **Task 38.1.2**: Optimize image size (multi-stage build)
- [ ] **Task 38.1.3**: Add GPU support variant (CUDA base image)
- [ ] **Task 38.1.4**: Publish to Docker Hub / GitHub Container Registry
- [ ] **Task 38.1.5**: Create docker-compose.yml for easy local setup

---

## Epic 39: Pydantic Schema Migration (P1)

*Replace manual JSON Schema with Pydantic models for better validation, type safety, and maintainability.*

**Why:** Current `schema.py` has 450 lines of manually-maintained JSON Schema that duplicates the dataclass definitions in `report.py`. Pydantic provides auto-generated schemas, runtime validation, and better IDE support.

### Story 39.1: Core Model Migration
- [ ] **Task 39.1.1**: Add `pydantic>=2.0` to core dependencies
- [ ] **Task 39.1.2**: Convert `ParamCounts` dataclass to Pydantic model
- [ ] **Task 39.1.3**: Convert `FLOPCounts` dataclass to Pydantic model
- [ ] **Task 39.1.4**: Convert `MemoryEstimates` dataclass to Pydantic model
- [ ] **Task 39.1.5**: Convert `GraphSummary` dataclass to Pydantic model
- [ ] **Task 39.1.6**: Convert `HardwareEstimates` dataclass to Pydantic model
- [ ] **Task 39.1.7**: Convert `InspectionReport` dataclass to Pydantic model
- [ ] **Task 39.1.8**: Update `to_dict()` / `to_json()` to use Pydantic methods

### Story 39.2: Schema Cleanup
- [ ] **Task 39.2.1**: Replace manual `INSPECTION_REPORT_SCHEMA` with `InspectionReport.model_json_schema()`
- [ ] **Task 39.2.2**: Remove redundant validation functions (Pydantic handles this)
- [ ] **Task 39.2.3**: Export schema for external consumers (e.g., `haoline --schema`)
- [ ] **Task 39.2.4**: Update tests to use Pydantic validation

### Story 39.3: Eval Schema Migration
- [ ] **Task 39.3.1**: Convert `EvalMetric` to Pydantic model
- [ ] **Task 39.3.2**: Convert `EvalResult` and task-specific variants to Pydantic
- [ ] **Task 39.3.3**: Convert `CombinedReport` to Pydantic model
- [ ] **Task 39.3.4**: Update adapters to use Pydantic serialization
