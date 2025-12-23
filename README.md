# HaoLine (皓线)

**The Model Decision Layer — Prevent teams from shipping the wrong model.**

[![PyPI version](https://badge.fury.io/py/haoline.svg)](https://badge.fury.io/py/haoline)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-509%20passed-brightgreen.svg)]()

HaoLine is a universal model inspector that makes neural network internals **legible** across formats — so you can make informed decisions about what to deploy.

### The Problem

ML teams inherit models they didn't train. They ship quantized variants without understanding the tradeoffs. They deploy to hardware that doesn't match their model's needs. The result: **silent regressions, cost overruns, and production failures.**

### The Solution

HaoLine provides a **single source of truth** for model decisions:

| What You're Doing | What HaoLine Prevents |
|-------------------|----------------------|
| Comparing model variants | Deploying a regression |
| Quantizing to INT8 | Shipping broken quantization |
| Choosing deployment hardware | Paying for the wrong GPU tier |
| Reviewing inherited models | Making decisions based on guesswork |

Works with **ONNX, PyTorch, TensorFlow, TensorRT, CoreML, TFLite, OpenVINO, GGUF, and SafeTensors**.

---

## Why Universal IR?

HaoLine is powered by a **Universal Internal Representation (IR)** — a format-agnostic graph representation that enables true cross-format comparison.

```
PyTorch  ─┐
ONNX     ─┼──▶  Universal IR  ──▶  Compare  ──▶  Decide
TensorRT ─┤          │
CoreML   ─┘          ▼
              Single source of truth
```

**Why this matters:**
- **Apples-to-apples comparison**: Compare a PyTorch model to its TensorRT-compiled version and see exactly what changed
- **Format-agnostic analysis**: Same metrics, same visualization, regardless of source format
- **Structural diff**: See which ops were fused, which precision changed, which layers were rewritten

Without a universal IR, format-specific graphs can obscure what's actually happening inside your model.

---

## Quick Start

**Try it now:** [huggingface.co/spaces/mdayku/haoline](https://huggingface.co/spaces/mdayku/haoline) — no installation required!

```bash
# Install
pip install haoline

# Analyze a model
python -m haoline model.onnx --out-html report.html --hardware auto

# Compare two variants
python -m haoline compare --models v1.onnx v2.onnx --out-html comparison.html
```

---

## Complete Beginner Guide

**Don't have a model yet?** No problem. Follow these steps to analyze your first model in under 5 minutes.

### Step 1: Install HaoLine

```bash
pip install haoline[llm]
```

This installs HaoLine with chart generation and AI summary support.

**Verify installation:**
```bash
python -m haoline --help
```

> **Troubleshooting:** If `haoline` command is not found after install, use `python -m haoline` instead. This happens when pip installs to a directory not on your PATH (common on Windows and user-level installs). See [Troubleshooting Installation](#troubleshooting-installation) below.

### Step 2: Get a Model to Analyze

**Option A: Download a pre-trained model from Hugging Face**

```bash
# Install huggingface_hub if you don't have it
pip install huggingface_hub

# Download a small image classification model (MobileNet, ~14MB)
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('onnx/models', 'validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx', local_dir='.')"
```

**Option B: Use your own model**

If you have a `.onnx`, `.pt`, `.pth`, or TensorFlow SavedModel, you can analyze it directly.

**Option C: Convert a PyTorch model**

```bash
# HaoLine can convert PyTorch models on the fly
haoline --from-pytorch your_model.pt --input-shape 1,3,224,224 --out-html report.html
```

### Step 3: Set Up AI Summaries (Optional but Recommended)

To get AI-generated executive summaries, set your OpenAI API key:

```bash
# Linux/macOS
export OPENAI_API_KEY="sk-..."

# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."

# Or create a .env file in your working directory
echo "OPENAI_API_KEY=sk-..." > .env
```

Get your API key at: https://platform.openai.com/api-keys

### Step 4: Generate Your Full Report

```bash
haoline mobilenetv2-7.onnx \
  --out-html report.html \
  --include-graph \
  --llm-summary \
  --hardware auto
```

This generates `report.html` containing:
- Model architecture overview
- Parameter counts and FLOPs analysis  
- Memory requirements
- Interactive neural network graph (zoomable, searchable)
- AI-generated executive summary
- Hardware performance estimates for your GPU

**Open `report.html` in your browser to explore your model!**

---

## Web Interface

**Try it now:** [huggingface.co/spaces/mdayku/haoline](https://huggingface.co/spaces/mdayku/haoline) — no installation required!

Or run locally with a single command:

```bash
pip install haoline[web]
haoline-web
```

This opens an interactive dashboard at `http://localhost:8501` with:

- Drag-and-drop model upload (ONNX, PyTorch, TFLite, CoreML, OpenVINO, TensorRT, GGUF, SafeTensors)
- Hardware selection with 50+ GPU profiles (searchable)
- **NEW:** Batch size and GPU count controls
- **NEW:** System Requirements (Steam-style min/rec/optimal)
- **NEW:** Deployment Cost Calculator (monthly cloud cost estimates)
- **NEW:** Cloud instance selector (T4, A10G, A100, H100, Jetson)
- Full interactive D3.js neural network graph
- Model comparison mode (side-by-side analysis)
- **NEW:** Per-layer timing breakdown (when benchmarked)
- **NEW:** Memory usage overview chart
- **NEW:** Run Benchmark button (actual ONNX Runtime measurements)
- **NEW:** Privacy controls (redact layer names, summary-only mode)
- **NEW:** Quantization Analysis (readiness score, QAT linting, recommendations)
- **NEW:** Layer Details tab (search/filter, CSV/JSON download)
- **NEW:** Quantization tab (readiness score, warnings, recommendations, layer sensitivity)
- AI-powered summaries (bring your own API key)
- Export to PDF, HTML, JSON, Markdown, **Universal IR**, **DOT graph**
- **NEW:** "Export as CLI" command generator (copy-paste equivalent `haoline` command)

> **Want to deploy your own?** See [DEPLOYMENT.md](DEPLOYMENT.md) for HuggingFace Spaces, Docker, and self-hosted options.

---

## Installation Options

| Command | What You Get |
|---------|--------------|
| `pip install haoline` | Core analysis (ONNX, GGUF) + charts |
| `pip install haoline[llm]` | + AI-powered summaries |
| `pip install haoline[full]` | **Everything** - all formats, LLM, PDF, GPU, web UI |

### Format-Specific Extras

Install only what you need:

| Extra | Command | Adds Support For |
|-------|---------|------------------|
| `pytorch` | `pip install haoline[pytorch]` | `.pt`, `.pth` model conversion |
| `tensorflow` | `pip install haoline[tensorflow]` | SavedModel, `.h5`, `.keras` conversion |
| `ultralytics` | `pip install haoline[ultralytics]` | YOLO models (v5, v8, v11) |
| `jax` | `pip install haoline[jax]` | JAX/Flax model conversion |
| `safetensors` | `pip install haoline[safetensors]` | `.safetensors` (HuggingFace weights) |
| `tflite` | `pip install haoline[tflite]` | `.tflite` + ONNX↔TFLite conversion |
| `coreml` | `pip install haoline[coreml]` | `.mlmodel`, `.mlpackage` (Apple) |
| `openvino` | `pip install haoline[openvino]` | `.xml`/`.bin` (Intel) |
| `tensorrt` | `pip install haoline[tensorrt]` | `.engine`, `.plan` (NVIDIA GPU required) |
| `gguf` | *included by default* | `.gguf` (llama.cpp) - pure Python |

### Other Extras

| Extra | Command | What It Adds |
|-------|---------|--------------|
| `llm` | `pip install haoline[llm]` | OpenAI, Anthropic, Google AI summaries |
| `web` | `pip install haoline[web]` | Streamlit web interface |
| `pdf` | `pip install haoline[pdf]` | PDF report generation |
| `gpu` | `pip install haoline[gpu]` | NVIDIA GPU metrics via pynvml |
| `runtime` | `pip install haoline[runtime]` | ONNX Runtime for benchmarking |

---

## Troubleshooting Installation

### "haoline: command not found"

This happens when pip installs scripts to a directory not on your PATH (common on Windows and user-level installs).

**Solution 1: Use module invocation (recommended)**
```bash
# Works on all platforms, no PATH changes needed
python -m haoline model.onnx --out-html report.html

# For the web interface
python -c "from haoline.web import main; main()"

# For model comparison
python -c "from haoline.compare import main; main()"
```

**Solution 2: Add pip scripts to PATH**

*Windows (PowerShell):*
```powershell
# Find where pip installed the scripts
python -c "import site; print(site.USER_SITE.replace('site-packages', 'Scripts'))"

# Add that path to your PATH environment variable
# Example: C:\Users\YourName\AppData\Roaming\Python\Python311\Scripts
```

*Linux/macOS:*
```bash
# Add to your ~/.bashrc or ~/.zshrc
export PATH="$HOME/.local/bin:$PATH"
```

### Verify Installation

```bash
# Check if haoline is installed and working
python -m haoline --help

# Check which extras are installed
python -c "import haoline; print(haoline.__version__)"
```

---

## Common Commands

```bash
# Basic analysis (prints to console)
haoline model.onnx

# Generate HTML report with charts
haoline model.onnx --out-html report.html --with-plots

# Full analysis with interactive graph and AI summary
haoline model.onnx --out-html report.html --include-graph --llm-summary

# Specify hardware for performance estimates
haoline model.onnx --hardware rtx4090 --out-html report.html

# Auto-detect your GPU
haoline model.onnx --hardware auto --out-html report.html

# List all available hardware profiles
haoline --list-hardware

# Convert and analyze a PyTorch model
haoline --from-pytorch model.pt --input-shape 1,3,224,224 --out-html report.html

# Convert and analyze a TensorFlow SavedModel
haoline --from-tensorflow ./saved_model_dir --out-html report.html

# Generate JSON for programmatic use
haoline model.onnx --out-json report.json
```

---

## Compare Model Variants

Compare different quantizations or architectures side-by-side:

```bash
haoline-compare \
  --models resnet_fp32.onnx resnet_fp16.onnx resnet_int8.onnx \
  --eval-metrics eval_fp32.json eval_fp16.json eval_int8.json \
  --baseline-precision fp32 \
  --out-html comparison.html \
  --with-charts
```

Or use the web UI's comparison mode for an interactive experience.

---

## CLI Reference

> **Tip:** The web interface includes an "Export as CLI" feature in the Export tab. After analyzing a model, you can copy the equivalent `haoline` command to replicate the analysis locally with full features.

### Output Options

| Flag | Description |
|------|-------------|
| `--out-json PATH` | Write JSON report |
| `--out-md PATH` | Write Markdown model card |
| `--out-html PATH` | Write HTML report (single shareable file) |
| `--out-pdf PATH` | Write PDF report (requires playwright) |
| `--html-graph PATH` | Write standalone interactive graph HTML |
| `--layer-csv PATH` | Write per-layer metrics CSV |

### Report Options

| Flag | Description |
|------|-------------|
| `--include-graph` | Embed interactive D3.js graph in HTML report |
| `--include-layer-table` | Include sortable per-layer table in HTML |
| `--with-plots` | Generate matplotlib visualization charts |
| `--assets-dir PATH` | Directory for chart PNG files |

### Hardware Options

| Flag | Description |
|------|-------------|
| `--hardware PROFILE` | GPU profile (`auto`, `rtx4090`, `a100`, `h100`, etc.) |
| `--list-hardware` | Show all 50+ available GPU profiles |
| `--precision {fp32,fp16,bf16,int8}` | Precision for estimates |
| `--batch-size N` | Batch size for estimates |
| `--gpu-count N` | Multi-GPU scaling (2, 4, 8) |
| `--cloud INSTANCE` | Cloud instance (e.g., `aws-p4d-24xlarge`) |
| `--list-cloud` | Show available cloud instances |
| `--system-requirements` | Generate Steam-style min/recommended specs |
| `--sweep-batch-sizes` | Find optimal batch size |
| `--sweep-resolutions` | Analyze resolution scaling |

### LLM Options

| Flag | Description |
|------|-------------|
| `--llm-summary` | Generate AI-powered executive summary |
| `--llm-model MODEL` | Model to use (default: `gpt-4o-mini`) |

### Conversion Options

| Flag | Description |
|------|-------------|
| `--from-pytorch PATH` | Convert PyTorch model to ONNX |
| `--from-tensorflow PATH` | Convert TensorFlow SavedModel |
| `--from-keras PATH` | Convert Keras .h5/.keras model |
| `--from-jax PATH` | Convert JAX/Flax model |
| `--input-shape SHAPE` | Input shape for conversion (e.g., `1,3,224,224`) |
| `--keep-onnx PATH` | Save converted ONNX to path |

### Privacy Options

| Flag | Description |
|------|-------------|
| `--redact-names` | Anonymize layer names for IP protection |
| `--summary-only` | Show only aggregate statistics |
| `--offline` | Disable all network requests |

### Quantization Analysis Options

| Flag | Description |
|------|-------------|
| `--lint-quantization` | Run quantization readiness analysis |
| `--quant-report PATH` | Write quantization report (Markdown) |
| `--quant-report-html PATH` | Write quantization report (HTML) |
| `--quant-llm-advice` | Get LLM-powered quantization recommendations |

### TensorRT Options

| Flag | Description |
|------|-------------|
| `--compare-trt ENGINE` | Compare ONNX model with its compiled TensorRT engine |
| `--quant-bottlenecks` | Show detailed quantization bottleneck analysis |

**TensorRT Engine Analysis:** Analyze compiled `.engine` or `.plan` files directly:

```bash
# Analyze TensorRT engine
python -m haoline model.engine --out-json report.json

# Compare ONNX source with TRT engine (shows fusions, precision changes)
python -m haoline model.onnx --compare-trt model.engine --out-html comparison.html
```

**Guaranteed Features:**
- Layer enumeration with names and types
- Precision breakdown (INT8/FP16/FP32 distribution)
- Fusion detection (Conv+BN+ReLU, LayerNorm, FlashAttention, etc.)
- Workspace and device memory allocation tracking
- Interactive side-by-side ONNX vs TRT comparison HTML

**Best-Effort Features** (may vary by engine):
- Layer rewrite visualization (attention optimizations, GELU, etc.)
- Per-layer timing breakdown (requires profiling data)
- Quantization bottleneck zone identification

**Known Limitations:**
- **Dynamic shapes**: HaoLine can detect when an engine was built with dynamic shapes, but cannot reconstruct the full optimization profile ranges
- **Plugin layers**: Custom TensorRT plugins may show as opaque nodes without internal details
- **Precision inference**: When explicit precision flags weren't set at build time, precision is inferred from layer names (heuristic)
- **No ONNX→TRT conversion**: HaoLine reads existing engines but doesn't build them—use NVIDIA's `trtexec` to compile

**Troubleshooting:**

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: tensorrt` | Install with `pip install haoline[tensorrt]` (NVIDIA GPU required) |
| `Engine deserialization failed` | Engine was built for different GPU/TensorRT version—rebuild with `trtexec` |
| `No layers found in engine` | Engine may be corrupt or built with incompatible TensorRT version |
| `Cannot read .trt file` | Rename to `.engine` or use `--format tensorrt` flag |

### Universal IR Export

| Flag | Description |
|------|-------------|
| `--export-ir PATH` | Export format-agnostic graph as JSON |
| `--export-graph PATH` | Export graph as DOT or PNG (Graphviz) |
| `--list-conversions` | Show all supported format conversions |

### Other Options

| Flag | Description |
|------|-------------|
| `--quiet` | Suppress console output |
| `--progress` | Show progress for large models |
| `--log-level {debug,info,warning,error}` | Logging verbosity |

---

## Python API

```python
from haoline import ModelInspector

inspector = ModelInspector()
report = inspector.inspect("model.onnx")

# Access metrics
print(f"Parameters: {report.param_counts.total:,}")
print(f"FLOPs: {report.flop_counts.total:,}")
print(f"Peak Memory: {report.memory_estimates.peak_activation_bytes / 1e9:.2f} GB")

# Export reports
report.to_json("report.json")
report.to_markdown("model_card.md")
report.to_html("report.html")
```

---

## CI/CD Integration

HaoLine can act as a **gatekeeper** in your ML pipelines, failing builds when model quality regresses.

### Threshold-Based Failure

Use `--fail-on` flags to set thresholds that cause non-zero exit codes:

```bash
python -m haoline compare \
  --models baseline.onnx candidate.onnx \
  --eval-metrics baseline.json candidate.json \
  --fail-on latency_increase=10% \
  --fail-on memory_increase=20% \
  --fail-on new_risk_signals
# Exit code 1 if any threshold violated
```

| Threshold | Example | Description |
|-----------|---------|-------------|
| `latency_increase` | `latency_increase=10%` | Fail if estimated latency increases >10% |
| `memory_increase` | `memory_increase=20%` | Fail if memory usage increases >20% |
| `param_increase` | `param_increase=5%` | Fail if parameter count increases >5% |
| `new_risk_signals` | `new_risk_signals` | Fail if new high-severity risks appear |

### Decision Reports (Audit Trail)

Generate a decision report for compliance and governance:

```bash
python -m haoline compare \
  --models baseline.onnx candidate.onnx \
  --eval-metrics baseline.json candidate.json \
  --decision-report decision.json  # or decision.md for Markdown
```

The decision report captures:
- **Models compared**: paths, MD5 hashes, file sizes, timestamps
- **Constraints applied**: all `--fail-on` thresholds
- **Results**: pass/fail status for each constraint
- **Decision**: APPROVED or REJECTED
- **Recommendations**: from quantization advisor and hardware estimator
- **Metadata**: timestamp, HaoLine version

### GitHub Actions

Copy the example workflow to your repository:

```bash
cp .github/examples/model-check.yml .github/workflows/
```

Or create `.github/workflows/model-check.yml`:

```yaml
name: Model Check
on:
  pull_request:
    paths: ['models/**', '*.onnx']

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install haoline
      - run: |
          python -m haoline compare \
            --models models/baseline.onnx models/candidate.onnx \
            --eval-metrics baseline.json candidate.json \
            --fail-on latency_increase=10% \
            --fail-on memory_increase=20% \
            --out-md comparison.md
      - uses: actions/github-script@v7
        if: always()
        with:
          script: |
            const fs = require('fs');
            const body = fs.readFileSync('comparison.md', 'utf8');
            github.rest.issues.createComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
              body: '## Model Check\n\n' + body
            });
```

See [.github/examples/model-check.yml](.github/examples/model-check.yml) for a full-featured template with artifact uploads and baseline detection.

---

## Features

| Feature | Description |
|---------|-------------|
| **Parameter Counts** | Per-node, per-block, and total parameter analysis |
| **FLOP Estimates** | Identify compute hotspots in your model |
| **Memory Analysis** | Peak activation memory and VRAM requirements |
| **Risk Signals** | Detect problematic architecture patterns |
| **Hardware Estimates** | GPU utilization predictions for 30+ NVIDIA profiles |
| **Runtime Profiling** | Actual inference benchmarks with ONNX Runtime |
| **Visualizations** | Operator histograms, parameter/FLOPs distribution charts |
| **Interactive Graph** | Zoomable D3.js neural network visualization |
| **AI Summaries** | GPT-powered executive summaries of your architecture |
| **Multiple Formats** | Export to HTML, Markdown, PDF, JSON, or CSV |
| **Universal IR** | Format-agnostic intermediate representation for cross-format analysis |
| **Quantization Analysis** | QAT readiness scoring, problem layer detection, deployment recommendations; Streamlit Quant tab with readiness score, warnings, recommendations, layer sensitivity |
| **Layer Details** | In-app per-layer table (search/filter, CSV/JSON download) |

---

## Quantization Analysis

HaoLine includes comprehensive quantization readiness analysis to help you prepare models for INT8/INT4 deployment:

```bash
# Run quantization analysis
haoline model.onnx --lint-quantization --quant-report quant_analysis.md

# Get LLM-powered recommendations
haoline model.onnx --lint-quantization --quant-llm-advice
```

**Features:**
- **Readiness Score (0-100)**: Letter grade (A-F) indicating how well the model will quantize
- **Problem Layer Detection**: Identifies ops that typically cause accuracy loss when quantized
- **QAT Validation**: Checks fake-quantization node placement in QAT-trained models
- **Deployment Recommendations**: Target-specific guidance (TensorRT, ONNX Runtime, TFLite)
- **LLM-Powered Advice**: Context-aware quantization strategy from AI

---

## Universal IR (Internal Representation)

HaoLine uses a Universal IR to represent models in a format-agnostic way, enabling:

- **Cross-format comparison**: Compare PyTorch vs ONNX vs TensorFlow architectures
- **Structural analysis**: Check if two models are architecturally identical
- **Graph visualization**: Export to Graphviz DOT or PNG

```bash
# Export model as Universal IR (JSON)
haoline model.onnx --export-ir model_ir.json

# Export graph visualization
haoline model.onnx --export-graph graph.dot
haoline model.onnx --export-graph graph.png --graph-max-nodes 200

# List available format conversions
haoline --list-conversions
```

The Universal IR includes:
- **UniversalGraph**: Container for nodes, tensors, and metadata
- **UniversalNode**: Format-agnostic operation representation
- **UniversalTensor**: Weight, input, output, and activation metadata

---

## Supported Model Formats

| Format | Support | Notes |
|--------|---------|-------|
| ONNX (.onnx) | ✅ Full | Native support |
| PyTorch (.pt, .pth) | ✅ Full | Auto-converts to ONNX |
| TensorFlow SavedModel | ✅ Full | Requires tf2onnx |
| Keras (.h5, .keras) | ✅ Full | Requires tf2onnx |
| GGUF (.gguf) | ✅ Read | llama.cpp LLMs (`pip install haoline`) |
| SafeTensors (.safetensors) | ⚠️ Weights Only | HuggingFace weights (`pip install haoline[safetensors]`) |
| TFLite (.tflite) | ✅ Full | Mobile/edge, ONNX↔TFLite conversion (`pip install haoline[tflite]`) |
| CoreML (.mlmodel, .mlpackage) | ✅ Read | Apple devices (`pip install haoline[coreml]`) |
| OpenVINO (.xml) | ✅ Read | Intel inference (`pip install haoline[openvino]`) |
| TensorRT (.engine, .plan) | ✅ Read | NVIDIA optimized engines (`pip install haoline[tensorrt]`) |

### Format Capabilities Matrix

Not all formats support all features. Here's what you get with each:

| Feature | ONNX | PyTorch | TFLite | CoreML | OpenVINO | TensorRT | GGUF | SafeTensors |
|---------|------|---------|--------|--------|----------|----------|------|-------------|
| **Parameter Count** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| **Memory Estimate** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **FLOPs Estimate** | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Interactive Graph** | ✅ | ✅ | 🔜 | 🔜 | 🔜 | ❌ | ❌ | ❌ |
| **Layer-by-Layer Table** | ✅ | ✅ | 🔜 | 🔜 | 🔜 | ✅ | ❌ | ❌ |
| **Op Type Breakdown** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **Quantization Analysis** | ✅ | ✅ | ✅ | ❓ | ✅ | ✅ | ✅ | ❌ |
| **Runtime Benchmarking** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **ONNX Comparison** | N/A | N/A | 🔜 | 🔜 | 🔜 | ✅ | ❌ | ❌ |

**Legend:** ✅ = Supported | 🔜 = Planned | ❌ = Not available | ❓ = Partial

**Why the differences?**

- **ONNX/PyTorch**: Full graph structure with UniversalGraph adapters → all features work
- **TensorRT**: Optimized fused graph with layer info, precision breakdown, and ONNX comparison (requires NVIDIA GPU)
- **TFLite/CoreML/OpenVINO**: Graph structure available; convert to ONNX externally for full FLOPs analysis
- **GGUF**: LLM architecture metadata (layers, heads, quantization) but no computational graph - weights only
- **SafeTensors**: Weights only - tensor shapes and dtypes, no graph structure

### Format Fidelity & Universal IR

| Format | Fidelity | Notes |
| --- | --- | --- |
| ONNX | High | Full graph + params + FLOPs + interactive map |
| PyTorch | Medium | Convert to ONNX for full UI; CLI can export ONNX |
| TFLite | Medium (CLI) | Graph/params via CLI; convert to ONNX for UI |
| CoreML | Medium (CLI) | Graph/params via CLI; convert to ONNX for UI |
| OpenVINO | Medium (CLI) | Graph/params via CLI; convert to ONNX for UI |
| TensorRT | Metadata | Engine metadata only; graph not available |
| GGUF | Metadata | LLM arch/quant metadata; no graph |
| SafeTensors | Weights | Weights only; no graph |

Streamlit renders graph-based views only when the format includes a graph; otherwise, convert to ONNX for full visualization and Universal IR features.

### Auto-conversion to ONNX (app + CLI)

| Source format | Auto-convert in Streamlit | CLI flag |
| --- | --- | --- |
| PyTorch (.pt/.pth) | ✅ (requires input shape prompt) | `--from-pytorch` |
| TFLite (.tflite) | ✅ (uses `tflite2onnx` if installed) | `--from-tflite` |
| CoreML (.mlmodel/.mlpackage) | ✅ (uses `coremltools` if installed) | `--from-coreml` |
| TensorFlow/Keras/JAX | CLI-only | `--from-tensorflow`, `--from-keras`, `--from-jax` |
| OpenVINO (.xml/.bin) | Not auto-converted; analyzed directly | n/a |
| GGUF / SafeTensors | No (metadata/weights only) | n/a |

If conversion dependencies are missing, the app falls back to native readers with limited features; provide input shapes for PyTorch or use the CLI for full control.

**Want Full Analysis for TFLite/CoreML/OpenVINO?**

These formats have graph structure but limited FLOPs/memory analysis. Convert to ONNX externally for complete metrics:

```bash
# Example: TFLite to ONNX (requires tflite2onnx)
pip install tflite2onnx
tflite2onnx model.tflite model.onnx
python -m haoline model.onnx --out-html report.html
```

**HuggingFace Models:** For models stored as SafeTensors, you need the original ONNX export or PyTorch weights. SafeTensors contains only weights, not computational graph.

---

## LLM Providers

HaoLine supports multiple AI providers for generating summaries:

| Provider | Environment Variable | Get API Key |
|----------|---------------------|-------------|
| OpenAI | `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com/api-keys) |
| Anthropic | `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com/) |
| Google Gemini | `GOOGLE_API_KEY` | [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| xAI Grok | `XAI_API_KEY` | [console.x.ai](https://console.x.ai/) |

---

## Where to Find Models

| Source | URL | Notes |
|--------|-----|-------|
| Hugging Face ONNX | [huggingface.co/onnx](https://huggingface.co/onnx) | Pre-converted ONNX models |
| ONNX Model Zoo | [github.com/onnx/models](https://github.com/onnx/models) | Official ONNX examples |
| Hugging Face Hub | [huggingface.co/models](https://huggingface.co/models) | PyTorch/TF models (convert with HaoLine) |
| TorchVision | `torchvision.models` | Classic vision models |
| Timm | [github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) | State-of-the-art vision models |

---

## Security Notice

⚠️ **Loading untrusted models is inherently risky.**

Like PyTorch's `torch.load()`, HaoLine uses `pickle` when loading certain model formats. These can execute arbitrary code if the model file is malicious.

**Best practices:**
- Only analyze models from trusted sources
- Run in a sandboxed environment (Docker, VM) when analyzing unknown models
- Review model provenance before loading

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Etymology

**HaoLine** (皓线) combines:
- 皓 (hào) = "bright, luminous" in Chinese
- Line = the paths through your neural network

*"Illuminating the architecture of your models."*
