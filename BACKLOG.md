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
| Epic 11: Streamlit Web UI | **COMPLETE** | 3 | 17/17 | Done |
| Epic 12: Eval Import & Comparison | **COMPLETE** | 7 | 30/30 | Done |
| Epic 13-17: MLOps Platform | Future | 5 | 0/? | P5 |
| Epic 18: Universal IR | **COMPLETE** | 6 | 25/25 | Done |
| Epic 19: SafeTensors | In Progress | 2 | 4/10 | P2 |
| Epic 20: CoreML | In Progress | 2 | 5/12 | P2 |
| Epic 21: TFLite | In Progress | 2 | 5/12 | P2 |
| Epic 22: TensorRT Engine Introspection | Not Started | 6 | 0/34 | **P2** |
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
| Epic 33: QAT Linters | **COMPLETE** | 5 | 41/41 | **P1** |
| Epic 34: Activation Visualization | Not Started | 5 | 0/25 | P2/P3 |
| Epic 35: TRT-Aware Graph UX | Not Started | 3 | 0/16 | **P2** |
| Epic 36: Layer Visualization | Not Started | 5 | 0/25 | **P2** |
| Epic 37: Hardware Recommender | Not Started | 2 | 0/10 | P3 |
| Epic 38: Docker Distribution | Not Started | 1 | 0/5 | P3 |
| Epic 39: Pydantic Schema Migration | **COMPLETE** | 3 | 12/12 | Done |
| Epic 40: Full Pydantic Dataclass Migration | **In Progress** | 6 | 22/29 | P2 |
| Epic 41: Standardized Reporting | **COMPLETE** | 5 | 44/44 | Done |
| Epic 42: Format Conversion Testing | Blocked | 4 | 0/24 | P1 (after 19-24) |
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
- **Epic 11: Streamlit Web UI** (17/17) - Web interface, HuggingFace Spaces deployment
- **Epic 12: Eval Import** (30/30) - Eval schemas, adapters, cost calculator, YOLO demo
- **Epic 18: Universal IR** (25/25) - Universal graph, format adapters, conversion matrix
- **Epic 25: Privacy/Trust** (9/9) - Local-first, output controls, enterprise documentation
- **Epic 33: QAT Linters** (41/41) - Quantization readiness, QAT validation, recommendations
- **Epic 39: Pydantic Migration** (12/12) - Schema validation, Pydantic models
- **Epic 41: Standardized Reporting** (44/44) - CLI-Streamlit parity, enhanced LLM prompts

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

## Epic 22: TensorRT Engine Introspection (P2)

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

## Epic 40: Full Pydantic Dataclass Migration (P2)

*Complete migration from Python dataclasses to Pydantic BaseModel across the entire codebase.*

**Why:** Epic 39 established Pydantic for validation, but the actual data structures remain as dataclasses. Full migration enables:
- Automatic JSON serialization via `model_dump()` / `model_dump_json()`
- Built-in validation on construction
- Better IDE autocomplete and type inference
- Field validators and computed properties
- Immutability options

**Scope:** 12+ dataclasses across 5 modules:
- `report.py`: `ModelMetadata`, `GraphSummary`, `DatasetInfo`, `InspectionReport`
- `analyzer.py`: `ParamCounts`, `FlopCounts`, `MemoryEstimates`
- `hardware.py`: `HardwareProfile`, `HardwareEstimates`
- `patterns.py`: `Block` and related types
- `risks.py`: `RiskSignal`
- `operational_profiling.py`: `BatchSizeSweep`, `ResolutionSweep`, `SystemRequirements`

### Story 40.1: Core Report Models - **COMPLETE**
- [x] **Task 40.1.1**: Convert `ModelMetadata` to Pydantic `BaseModel`
- [x] **Task 40.1.2**: Convert `GraphSummary` to Pydantic `BaseModel`
- [x] **Task 40.1.3**: Convert `DatasetInfo` to Pydantic `BaseModel`
- [x] **Task 40.1.4**: Convert `InspectionReport` to Pydantic `BaseModel`
- [x] **Task 40.1.5**: Replace `to_dict()` with Pydantic `model_dump()`
- [x] **Task 40.1.6**: Replace `to_json()` with Pydantic `model_dump_json()`

### Story 40.2: Analyzer Models - **COMPLETE**
- [x] **Task 40.2.1**: Convert `ParamCounts` to Pydantic `BaseModel`
- [x] **Task 40.2.2**: Convert `FlopCounts` to Pydantic `BaseModel`
- [x] **Task 40.2.3**: Convert `MemoryEstimates` to Pydantic `BaseModel`
- [x] **Task 40.2.4**: Update `MetricsEngine` to return Pydantic models
- [x] **Task 40.2.5**: Handle multiple eval runs per model (aggregate or select) — *from Epic 12.4.3*
- [x] **Task 40.2.6**: Validate eval task matches model type (warn if mismatch) — *from Epic 12.4.4*

### Story 40.3: Hardware and Risk Models - **COMPLETE**
- [x] **Task 40.3.1**: Convert `HardwareProfile` to Pydantic `BaseModel`
- [x] **Task 40.3.2**: Convert `HardwareEstimates` to Pydantic `BaseModel`
- [x] **Task 40.3.3**: Convert `RiskSignal` to Pydantic `BaseModel`
- [x] **Task 40.3.4**: Convert `Block` and pattern types to Pydantic

### Story 40.4: Schema Consolidation - **COMPLETE**
- [x] **Task 40.4.1**: Consolidate schema validation to use report.py models
- [x] **Task 40.4.2**: Update all imports across codebase
- [x] **Task 40.4.3**: Update CLI to work with Pydantic models
- [x] **Task 40.4.4**: Update Streamlit app to work with Pydantic models
- [x] **Task 40.4.5**: Delete `pydantic_models.py` (superseded by report.py)
- [x] **Task 40.4.6**: Update all unit tests for Pydantic models

### Story 40.5: Format Readers Migration
*Convert format reader dataclasses to Pydantic.*

- [ ] **Task 40.5.1**: Convert `formats/safetensors.py` classes to Pydantic
- [ ] **Task 40.5.2**: Convert `formats/coreml.py` classes to Pydantic
- [ ] **Task 40.5.3**: Convert `formats/tflite.py` classes to Pydantic
- [ ] **Task 40.5.4**: Convert `formats/openvino.py` classes to Pydantic
- [ ] **Task 40.5.5**: Convert `formats/gguf.py` classes to Pydantic
- [ ] **Task 40.5.6**: Update format reader tests
- [ ] **Task 40.5.7**: Performance testing (Pydantic overhead vs dataclass)

---

---

## Epic 42: Format Conversion Testing (P1)

*Comprehensive test suite for all format conversions in the conversion matrix (Epics 19-24).*

**Goal:** Ensure every `to` and `from` conversion path works correctly, preserves metadata, and handles edge cases gracefully.

### Story 42.1: ONNX Hub Conversions
*Test ONNX as the interchange format (most common path).*

- [ ] **Task 42.1.1**: Test ONNX → TensorRT conversion (requires TRT runtime)
- [ ] **Task 42.1.2**: Test ONNX → TFLite conversion (via tf2onnx reverse)
- [ ] **Task 42.1.3**: Test ONNX → CoreML conversion (via coremltools)
- [ ] **Task 42.1.4**: Test ONNX → OpenVINO conversion (via openvino)
- [ ] **Task 42.1.5**: Test TFLite → ONNX conversion (tflite2onnx)
- [ ] **Task 42.1.6**: Test CoreML → ONNX conversion (lossy path)

### Story 42.2: Framework-to-ONNX Conversions
*Test native framework exports to ONNX.*

- [ ] **Task 42.2.1**: Test PyTorch → ONNX with simple CNN model
- [ ] **Task 42.2.2**: Test PyTorch → ONNX with Ultralytics YOLO model
- [ ] **Task 42.2.3**: Test PyTorch → ONNX with transformer model
- [ ] **Task 42.2.4**: Test TensorFlow SavedModel → ONNX conversion
- [ ] **Task 42.2.5**: Test Keras .h5 → ONNX conversion
- [ ] **Task 42.2.6**: Test TensorFlow frozen graph → ONNX conversion

### Story 42.3: Framework Multi-Hop Conversions
*Test conversions that go through ONNX as intermediary.*

- [ ] **Task 42.3.1**: Test PyTorch → TFLite (via ONNX)
- [ ] **Task 42.3.2**: Test PyTorch → CoreML (via coremltools direct)
- [ ] **Task 42.3.3**: Test PyTorch → OpenVINO (via ONNX)
- [ ] **Task 42.3.4**: Test TensorFlow → TFLite (direct, should be FULL)
- [ ] **Task 42.3.5**: Test TensorFlow → CoreML (via coremltools)
- [ ] **Task 42.3.6**: Test TensorFlow → OpenVINO (via ONNX)

### Story 42.4: Round-Trip and Metadata Validation
*Verify conversions preserve essential information.*

- [ ] **Task 42.4.1**: Create test harness for conversion round-trips
- [ ] **Task 42.4.2**: Test ONNX → TFLite → ONNX round-trip (check param counts)
- [ ] **Task 42.4.3**: Test ONNX → CoreML → ONNX round-trip (check lossy delta)
- [ ] **Task 42.4.4**: Validate op_type_counts preserved across conversions
- [ ] **Task 42.4.5**: Validate precision_breakdown preserved across conversions
- [ ] **Task 42.4.6**: Test conversion error handling (unsupported ops, invalid models)

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
- [ ] **Task 43.3.5**: Visualize memory timeline in HTML report

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

**New Story 11.4 (in Epic 11):** Sample Model Preloading
- [ ] Bundle 2-3 small sample models (ResNet18, TinyBERT, YOLO-nano)
- [ ] Add "Try a sample model" dropdown in Streamlit
- [ ] Cache sample model analysis results for instant display

---

## ~~Epic 46: Enhanced Model Diff~~ → MERGED INTO EPIC 18

*Layer-level diff and visual diff merged into Epic 18 (Universal IR). Structural comparison tools (`diff()`, `is_structurally_equal()`) already exist in `universal_ir.py`.*

**New Story 18.7 (in Epic 18):** Enhanced Diff Visualization
- [ ] Extend `diff()` to include layer-level alignment by name
- [ ] Add added/removed/modified layer categorization
- [ ] Visualize diff in graph view (green=added, red=removed, yellow=modified)
- [ ] Generate "what changed" summary paragraph

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
- [ ] Calculate compression ratio (original size / quantized size)
- [ ] Estimate theoretical speedup from precision reduction
- [ ] Add "efficiency score" (0-100) based on size + speed gains
- [ ] Detect over-quantization (INT4 on accuracy-sensitive layers)
- [ ] Generate quantization efficiency report