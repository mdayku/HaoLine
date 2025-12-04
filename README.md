# HaoLine (皓线)

**Universal Model Inspector — See what's really inside your neural networks.**

[![PyPI version](https://badge.fury.io/py/haoline.svg)](https://badge.fury.io/py/haoline)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

Launch the HaoLine web UI with a single command:

```bash
pip install haoline[web]
haoline-web
```

This opens an interactive dashboard at `http://localhost:8501` with:
- Drag-and-drop model upload (ONNX, PyTorch)
- Hardware selection with 50+ GPU profiles (searchable)
- Full interactive D3.js neural network graph
- Model comparison mode (side-by-side analysis)
- AI-powered summaries (bring your own API key)
- Export to PDF, HTML, JSON, Markdown

---

## Installation Options

| Command | What You Get |
|---------|--------------|
| `pip install haoline` | Core analysis + charts |
| `pip install haoline[llm]` | + AI-powered summaries |
| `pip install haoline[full]` | + PDF export, GPU metrics, runtime profiling |

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

---

## Supported Model Formats

| Format | Support | Notes |
|--------|---------|-------|
| ONNX (.onnx) | ✅ Full | Native support |
| PyTorch (.pt, .pth) | ✅ Full | Auto-converts to ONNX |
| TensorFlow SavedModel | ✅ Full | Requires tf2onnx |
| Keras (.h5, .keras) | ✅ Full | Requires tf2onnx |
| GGUF (.gguf) | ✅ Read | llama.cpp LLMs (`pip install haoline`) |
| SafeTensors (.safetensors) | ✅ Read | HuggingFace weights (`pip install haoline[formats]`) |
| TFLite (.tflite) | ✅ Read | Mobile/edge (`pip install haoline[formats]`) |
| CoreML (.mlmodel, .mlpackage) | ✅ Read | Apple devices (`pip install haoline[coreml]`) |
| OpenVINO (.xml) | ✅ Read | Intel inference (`pip install haoline[openvino]`) |
| TensorRT Engine | 🔜 Coming | NVIDIA optimized engines |

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
