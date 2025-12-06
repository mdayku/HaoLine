# HaoLine (皓线)

**Universal Model Inspector — See what's really inside your neural networks.**

[![PyPI version](https://badge.fury.io/py/haoline.svg)](https://badge.fury.io/py/haoline)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-317%20passed-brightgreen.svg)]()

HaoLine analyzes neural network architectures and generates comprehensive reports with metrics, visualizations, and AI-powered summaries. Works with ONNX, PyTorch, and TensorFlow models.

---

## Complete Beginner Guide

**Don't have a model yet?** No problem. Follow these steps to analyze your first model in under 5 minutes.

### Step 1: Install HaoLine

```bash
pip install haoline[llm]
```

This installs HaoLine with chart generation and AI summary support.

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

- Drag-and-drop model upload (ONNX, PyTorch)
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
- AI-powered summaries (bring your own API key)
- Export to PDF, HTML, JSON, Markdown, **Universal IR**, **DOT graph**

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
| `tflite` | `pip install haoline[tflite]` | `.tflite` (Linux/Mac only) |
| `coreml` | `pip install haoline[coreml]` | `.mlmodel`, `.mlpackage` (Apple) |
| `openvino` | `pip install haoline[openvino]` | `.xml`/`.bin` (Intel) |
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
| **Quantization Analysis** | QAT readiness scoring, problem layer detection, deployment recommendations |

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
| TFLite (.tflite) | ✅ Read | Mobile/edge, Linux/Mac (`pip install haoline[tflite]`) |
| CoreML (.mlmodel, .mlpackage) | ✅ Read | Apple devices (`pip install haoline[coreml]`) |
| OpenVINO (.xml) | ✅ Read | Intel inference (`pip install haoline[openvino]`) |
| TensorRT Engine | 🔜 Coming | NVIDIA optimized engines |

### Format Capabilities Matrix

Not all formats support all features. Here's what you get with each:

| Feature | ONNX | PyTorch | TFLite | CoreML | OpenVINO | GGUF | SafeTensors |
|---------|------|---------|--------|--------|----------|------|-------------|
| **Parameter Count** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Memory Estimate** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **FLOPs Estimate** | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Interactive Graph** | ✅ | ✅ | 🔜 | 🔜 | 🔜 | ❌ | ❌ |
| **Layer-by-Layer Table** | ✅ | ✅ | 🔜 | 🔜 | 🔜 | ❌ | ❌ |
| **Op Type Breakdown** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **Quantization Analysis** | ✅ | ✅ | ✅ | ❓ | ✅ | ✅ | ❌ |
| **Runtime Benchmarking** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

**Legend:** ✅ = Supported | 🔜 = Planned | ❌ = Not available | ❓ = Partial

**Why the differences?**

- **ONNX/PyTorch**: Full graph structure with UniversalGraph adapters → all features work
- **TFLite/CoreML/OpenVINO**: Readers exist with op counts, but UniversalGraph adapters pending for interactive visualization
- **GGUF**: LLM architecture metadata (layers, heads, context length, quantization) but no computational graph - weights only
- **SafeTensors**: Weights only - tensor shapes and dtypes, no graph structure

**Tip:** For full analysis of HuggingFace models stored as SafeTensors, load the complete model:
```bash
# Coming soon: --from-huggingface flag
haoline --from-huggingface meta-llama/Llama-2-7b --out-html report.html
```

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
