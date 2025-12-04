# HaoLine (好线)

**Universal Model Inspector — See what's really inside your models.**

[![PyPI version](https://badge.fury.io/py/haoline.svg)](https://badge.fury.io/py/haoline)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A graph-level analysis tool that inspects neural network architectures (ONNX, PyTorch, TensorFlow, TensorRT), computes static complexity metrics, surfaces structural risk signals, and auto-generates human-readable reports.

---

## Installation

```bash
pip install haoline

# With visualization support
pip install haoline[viz]

# With LLM summaries (requires Anthropic API key)
pip install haoline[llm]

# With all features
pip install haoline[full]
```

---

## Quick Start

### Analyze a Model

```bash
# Basic analysis
haoline analyze model.onnx

# With hardware profile
haoline analyze model.onnx --hardware rtx4090

# Generate HTML report with charts
haoline analyze model.onnx --out-html report.html --with-plots

# Full analysis with AI summary
haoline analyze model.onnx --hardware auto --with-plots --llm-summary --out-html full_report.html
```

### Compare Model Variants

```bash
# Compare FP32, FP16, INT8 quantizations
haoline compare \
  --models resnet_fp32.onnx resnet_fp16.onnx resnet_int8.onnx \
  --baseline-precision fp32 \
  --out-html comparison.html --with-charts
```

---

## Features

- **Parameter counts** - Per node, per block, and globally
- **FLOP estimates** - Identify compute hotspots
- **Memory analysis** - Peak activation memory and VRAM requirements
- **Risk signals** - Detect problematic architecture patterns
- **Hardware estimates** - GPU utilization predictions for 30+ NVIDIA profiles
- **Runtime profiling** - Actual inference benchmarks with ONNX Runtime
- **Visualizations** - Operator histograms, parameter/FLOPs distribution charts
- **LLM Summaries** - AI-generated executive summaries
- **Interactive graphs** - D3.js neural network visualizations
- **Shareable Reports** - HTML, Markdown, PDF, and JSON output formats

---

## Supported Formats

| Format | Read | Write |
|--------|------|-------|
| ONNX | ✅ | - |
| PyTorch (.pt, .pth) | ✅ | - |
| TensorFlow SavedModel | ✅ | - |
| TensorRT Engine | 🔜 | - |
| SafeTensors | 🔜 | - |

---

## Python API

```python
from haoline import ModelInspector

inspector = ModelInspector()
report = inspector.inspect("model.onnx")

# Get metrics
print(f"Parameters: {report.params.total:,}")
print(f"FLOPs: {report.flops.total:,}")
print(f"Peak Memory: {report.memory.peak_activation_bytes / 1e9:.2f} GB")

# Export
report.to_json("report.json")
report.to_markdown("model_card.md")
report.to_html("report.html")
```

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Etymology

**HaoLine** (好线) combines:
- 好 (hǎo) = "good" in Chinese
- A personal touch from the author

*"The good path through your neural network."*

