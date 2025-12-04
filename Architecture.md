# HaoLine (皓线) - System Architecture

## Overview

This document describes the system architecture for HaoLine, a universal model architecture inspection tool supporting ONNX, PyTorch, TensorFlow, and TensorRT.

---

## Table of Contents

1. [System Context](#1-system-context)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Component Details](#3-component-details)
4. [Data Flow](#4-data-flow)
5. [File Structure](#5-file-structure)
6. [Integration Points](#6-integration-points)
7. [Deployment Architecture](#7-deployment-architecture)
8. [Design Decisions](#8-design-decisions)

---

## 1. System Context

### 1.1 Context Diagram

```
+------------------+     +------------------+     +------------------+
|                  |     |                  |     |                  |
|  ML Engineers    |     |  MLOps/Platform  |     |  Leadership      |
|                  |     |                  |     |                  |
+--------+---------+     +--------+---------+     +--------+---------+
         |                        |                        |
         |   Inspect models       |   Registry integration |   Reports
         v                        v                        v
+------------------------------------------------------------------------+
|                                                                        |
|                          HaoLine (皓线)                                 |
|                                                                        |
|   CLI Interface  -->  Analysis Engine  -->  Report Generators          |
|                                                                        |
+------------------------------------------------------------------------+
         |                        |                        |
         v                        v                        v
+------------------+     +------------------+     +------------------+
|                  |     |                  |     |                  |
|  ONNX Models     |     |  Hardware        |     |  External Eval   |
|  (.onnx files)   |     |  Profiles        |     |  Pipelines       |
|                  |     |                  |     |                  |
+------------------+     +------------------+     +------------------+
```

### 1.2 External Dependencies

| Dependency | Purpose | Required |
|------------|---------|----------|
| `onnx` library | Model loading and parsing | Yes |
| `numpy` | Numerical computations | Yes |
| `protobuf` | ONNX serialization | Yes |
| `matplotlib` | Visualization generation | No (optional) |
| `openai` / `anthropic` | LLM summarization | No (optional) |
| `jinja2` | HTML templating | No (optional) |

---

## 2. High-Level Architecture

### 2.1 Layered Architecture

```
+------------------------------------------------------------------+
|                        Presentation Layer                         |
|                                                                   |
|   +------------+   +------------+   +------------+                |
|   |   CLI      |   | JSON API   |   | Python API |                |
|   +------------+   +------------+   +------------+                |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                        Application Layer                          |
|                                                                   |
|   +------------------+   +------------------+                     |
|   | Report Generator |   | Compare Engine   |                     |
|   +------------------+   +------------------+                     |
|                                                                   |
|   +------------------+   +------------------+                     |
|   | Visualization    |   | LLM Summarizer   |                     |
|   +------------------+   +------------------+                     |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                         Domain Layer                              |
|                                                                   |
|   +------------------+   +------------------+                     |
|   | Model Inspector  |   | Hardware Profile |                     |
|   +------------------+   +------------------+                     |
|                                                                   |
|   +------------------+   +------------------+                     |
|   | Metrics Engine   |   | Risk Analyzer    |                     |
|   +------------------+   +------------------+                     |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                      Infrastructure Layer                         |
|                                                                   |
|   +------------------+   +------------------+                     |
|   | ONNX Graph       |   | File System      |                     |
|   | Loader           |   | I/O              |                     |
|   +------------------+   +------------------+                     |
+------------------------------------------------------------------+
```

### 2.2 Component Overview

| Layer | Components | Responsibility |
|-------|------------|----------------|
| **Presentation** | CLI, JSON API, Python API | User interaction, argument parsing |
| **Application** | Report Generator, Compare Engine, Visualization, LLM Summarizer | Orchestration, output formatting |
| **Domain** | Model Inspector, Metrics Engine, Risk Analyzer, Hardware Profile | Core business logic |
| **Infrastructure** | ONNX Graph Loader, File System I/O | External resource access |

---

## 3. Component Details

### 3.1 ONNX Graph Loader

**Purpose**: Load and parse ONNX models into an internal representation.

```python
class ONNXGraphLoader:
    """Load ONNX models and extract graph structure."""

    def load(self, path: str) -> ModelProto:
        """Load ONNX model from file."""
        pass

    def extract_graph(self, model: ModelProto) -> GraphInfo:
        """Extract graph nodes, edges, and metadata."""
        pass

    def infer_shapes(self, model: ModelProto) -> Dict[str, TensorShape]:
        """Run shape inference to get tensor dimensions."""
        pass
```

**Key Classes:**

```
+------------------+     +------------------+     +------------------+
|   ModelProto     | --> |   GraphInfo      | --> |   NodeInfo       |
+------------------+     +------------------+     +------------------+
| path             |     | name             |     | name             |
| opset_version    |     | nodes            |     | op_type          |
| producer         |     | inputs           |     | inputs           |
| ir_version       |     | outputs          |     | outputs          |
+------------------+     | initializers     |     | attributes       |
                         +------------------+     +------------------+
```

### 3.2 Metrics Engine

**Purpose**: Compute structural complexity metrics.

```python
class MetricsEngine:
    """Compute model complexity metrics."""

    def count_parameters(self, graph: GraphInfo) -> ParamCounts:
        """Count parameters per node, block, and globally."""
        pass

    def estimate_flops(self, graph: GraphInfo) -> FlopCounts:
        """Estimate FLOPs for each operation."""
        pass

    def estimate_memory(self, graph: GraphInfo) -> MemoryEstimates:
        """Estimate activation memory and peak usage."""
        pass

    def extract_attention_metrics(self, graph: GraphInfo) -> AttentionMetrics:
        """Extract transformer-specific metrics."""
        pass
```

**FLOP Calculation Matrix:**

| Op Type | FLOP Formula |
|---------|--------------|
| Conv2D | `2 * K_h * K_w * C_in * C_out * H_out * W_out` |
| MatMul | `2 * M * N * K` |
| Gemm | `2 * M * N * K + M * N` (with bias) |
| Add/Mul | `N` (element count) |
| Softmax | `5 * N` (approximation) |

### 3.3 Pattern Analyzer

**Purpose**: Detect common architectural patterns and group nodes into blocks.

```python
class PatternAnalyzer:
    """Detect architectural patterns in the graph."""

    def detect_conv_bn_relu(self, graph: GraphInfo) -> List[Block]:
        """Find Conv-BatchNorm-ReLU sequences."""
        pass

    def detect_residual_blocks(self, graph: GraphInfo) -> List[Block]:
        """Find skip connection patterns."""
        pass

    def detect_transformer_blocks(self, graph: GraphInfo) -> List[Block]:
        """Find attention + MLP patterns."""
        pass

    def group_into_blocks(self, graph: GraphInfo) -> List[Block]:
        """Aggregate all pattern detections."""
        pass
```

**Pattern Detection State Machine:**

```
                    Conv
          +----------+----------+
          |                     |
          v                     v
    +----------+          +----------+
    |   BN     |          | No Match |
    +----------+          +----------+
          |
          v
    +----------+
    |  ReLU    | --> Block(Conv-BN-ReLU)
    +----------+
```

### 3.4 Risk Analyzer

**Purpose**: Apply heuristics to detect potentially problematic patterns.

```python
class RiskAnalyzer:
    """Detect architectural risk signals."""

    def check_deep_without_skips(self, graph: GraphInfo) -> Optional[RiskSignal]:
        """Flag deep networks without skip connections."""
        pass

    def check_oversized_dense(self, graph: GraphInfo) -> Optional[RiskSignal]:
        """Flag excessively large fully-connected layers."""
        pass

    def check_dynamic_shapes(self, graph: GraphInfo) -> Optional[RiskSignal]:
        """Flag problematic dynamic dimensions."""
        pass

    def analyze(self, graph: GraphInfo) -> List[RiskSignal]:
        """Run all heuristics and return signals."""
        pass
```

**Risk Signal Schema:**

```python
@dataclass
class RiskSignal:
    id: str              # e.g., "no_skip_connections"
    severity: str        # "info" | "warning" | "high"
    description: str     # Human-readable explanation
    nodes: List[str]     # Affected nodes
    recommendation: str  # Suggested action
```

### 3.5 Model Inspector (Orchestrator)

**Purpose**: Coordinate all analysis components and produce the final report.

```python
class ModelInspector:
    """Main orchestrator for model analysis."""

    def __init__(
        self,
        loader: ONNXGraphLoader,
        metrics: MetricsEngine,
        patterns: PatternAnalyzer,
        risks: RiskAnalyzer,
    ):
        pass

    def inspect(self, model_path: str) -> InspectionReport:
        """Run full analysis pipeline."""
        pass

    def to_json(self) -> Dict[str, Any]:
        """Serialize report to JSON."""
        pass

    def to_markdown(self) -> str:
        """Generate Markdown model card."""
        pass
```

### 3.6 Visualization Module

**Purpose**: Generate matplotlib charts for reports.

```python
class VisualizationEngine:
    """Generate charts for model analysis."""

    def __init__(self, style: str = "default"):
        pass

    def operator_histogram(
        self,
        op_counts: Dict[str, int],
        output_path: Path
    ) -> str:
        """Generate operator type distribution chart."""
        pass

    def layer_depth_profile(
        self,
        layers: List[LayerInfo],
        output_path: Path
    ) -> str:
        """Generate cumulative compute distribution."""
        pass

    def parameter_distribution(
        self,
        param_counts: Dict[str, int],
        output_path: Path
    ) -> str:
        """Generate parameter distribution by layer."""
        pass

    def generate_all(
        self,
        report: InspectionReport,
        assets_dir: Path
    ) -> Dict[str, str]:
        """Generate all charts and return paths."""
        pass
```

### 3.7 Hardware Profile System

**Purpose**: Estimate hardware requirements and utilization.

```python
class HardwareProfile:
    """Hardware specification for estimates."""

    name: str
    vendor: str
    type: str  # "gpu" | "cpu" | "npu"
    vram_bytes: int
    peak_fp16_flops: int
    peak_fp32_flops: int
    memory_bandwidth_bytes_per_s: int

class HardwareEstimator:
    """Estimate hardware requirements."""

    def estimate_vram(
        self,
        report: InspectionReport,
        profile: HardwareProfile,
        batch_size: int,
        precision: str
    ) -> int:
        """Estimate VRAM requirement in bytes."""
        pass

    def estimate_latency(
        self,
        report: InspectionReport,
        profile: HardwareProfile,
        batch_size: int
    ) -> float:
        """Estimate theoretical latency in ms."""
        pass

    def identify_bottleneck(
        self,
        report: InspectionReport,
        profile: HardwareProfile
    ) -> str:
        """Identify whether compute or memory limited."""
        pass
```

### 3.8 Compare Engine

**Purpose**: Compare multiple model variants.

```python
class CompareEngine:
    """Compare multiple model variants."""

    def load_variants(
        self,
        model_paths: List[str],
        eval_metrics_paths: List[str]
    ) -> List[VariantInfo]:
        """Load models and their eval metrics."""
        pass

    def verify_compatibility(
        self,
        variants: List[VariantInfo]
    ) -> bool:
        """Check if models are comparable."""
        pass

    def compute_deltas(
        self,
        variants: List[VariantInfo],
        baseline_precision: str
    ) -> CompareReport:
        """Compute differences vs baseline."""
        pass
```

### 3.9 Operational Profiler

**Purpose**: Analyze scaling characteristics (batch size, resolution).

```python
class OperationalProfiler:
    """Analyzes model operational characteristics."""

    def run_batch_sweep(
        self,
        model_params: int,
        model_flops: int,
        hardware: HardwareProfile,
        batch_sizes: List[int] = None
    ) -> BatchSizeSweep:
        """Analyze performance scaling across batch sizes."""
        pass

    def run_resolution_sweep(
        self,
        base_flops: int,
        base_resolution: Tuple[int, int],
        model_params: int,
        hardware: HardwareProfile,
        resolutions: List[Tuple[int, int]] = None
    ) -> ResolutionSweep:
        """
        Analyze performance scaling across resolutions.

        Key constraints:
        1. Only sweep UP TO training resolution (not above)
        2. Match aspect ratio of training data
        3. Round to nearest 32 for GPU efficiency
        """
        pass

    def recommend_resolution(
        self,
        base_flops: int,
        base_resolution: Tuple[int, int],
        hardware: HardwareProfile,
        target_fps: float = 30.0
    ) -> Dict[str, Any]:
        """Recommend optimal resolution for target FPS."""
        pass
```

### 3.10 Compare Visualizations

**Purpose**: Generate charts for multi-model comparison reports.

```python
# compare_visualizations.py

def compute_tradeoff_points(compare_json: Dict) -> List[TradeoffPoint]:
    """Compute speedup/accuracy tradeoff for each variant."""

def generate_tradeoff_chart(points: List[TradeoffPoint]) -> bytes:
    """Generate accuracy vs speedup scatter chart."""

def generate_memory_savings_chart(compare_json: Dict) -> bytes:
    """Generate size/memory reduction bar chart."""

def generate_compare_html(compare_json: Dict) -> str:
    """Generate HTML report with engine summary panel."""

def analyze_tradeoffs(compare_json: Dict) -> Dict[str, Any]:
    """Identify best variants and generate recommendations."""

def generate_calibration_recommendations(compare_json: Dict) -> List[CalibrationRecommendation]:
    """Generate INT8/INT4 calibration guidance."""
```

---

## 4. Data Flow

### 4.1 Single Model Inspection

```
                                    +------------------+
                                    |  Command Line    |
                                    |  Arguments       |
                                    +--------+---------+
                                             |
                                             v
+------------------+              +------------------+
|  ONNX Model      | ----------> |  ONNX Graph      |
|  (.onnx file)    |             |  Loader          |
+------------------+              +--------+---------+
                                           |
                                           v
                                  +------------------+
                                  |  GraphInfo       |
                                  |  (parsed graph)  |
                                  +--------+---------+
                                           |
              +----------------------------+----------------------------+
              |                            |                            |
              v                            v                            v
    +------------------+         +------------------+         +------------------+
    |  Metrics         |         |  Pattern         |         |  Risk            |
    |  Engine          |         |  Analyzer        |         |  Analyzer        |
    +--------+---------+         +--------+---------+         +--------+---------+
             |                            |                            |
             v                            v                            v
    +------------------+         +------------------+         +------------------+
    |  ParamCounts     |         |  Blocks          |         |  RiskSignals     |
    |  FlopCounts      |         |  Patterns        |         |                  |
    |  MemoryEstimates |         |                  |         |                  |
    +------------------+         +------------------+         +------------------+
              \                           |                           /
               \                          |                          /
                \                         v                         /
                 +-----------> +------------------+ <--------------+
                               |  InspectionReport|
                               +--------+---------+
                                        |
              +-------------------------+-------------------------+
              |                         |                         |
              v                         v                         v
    +------------------+      +------------------+      +------------------+
    |  JSON            |      |  Markdown        |      |  Visualization   |
    |  Serializer      |      |  Renderer        |      |  Engine          |
    +--------+---------+      +--------+---------+      +--------+---------+
             |                         |                         |
             v                         v                         v
    +------------------+      +------------------+      +------------------+
    |  report.json     |      |  model_card.md   |      |  assets/*.png    |
    +------------------+      +------------------+      +------------------+
```

### 4.2 Compare Mode Flow

```
+------------------+     +------------------+     +------------------+
|  Model FP32      |     |  Model FP16      |     |  Model INT8      |
+--------+---------+     +--------+---------+     +--------+---------+
         |                        |                        |
         v                        v                        v
+------------------+     +------------------+     +------------------+
|  Inspector       |     |  Inspector       |     |  Inspector       |
+--------+---------+     +--------+---------+     +--------+---------+
         |                        |                        |
         v                        v                        v
+------------------+     +------------------+     +------------------+
|  Report FP32     |     |  Report FP16     |     |  Report INT8     |
+--------+---------+     +--------+---------+     +--------+---------+
         \                        |                       /
          \                       |                      /
           +----------------------+---------------------+
                                  |
                                  v
                         +------------------+
                         |  Compare Engine  |
                         +--------+---------+
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
           +------------------+        +------------------+
           |  Eval Metrics    |        |  Hardware        |
           |  (external JSON) |        |  Profile         |
           +--------+---------+        +--------+---------+
                    \                          /
                     \                        /
                      +----------+-----------+
                                 |
                                 v
                        +------------------+
                        |  CompareReport   |
                        +--------+---------+
                                 |
                    +------------+------------+
                    |                         |
                    v                         v
           +------------------+      +------------------+
           |  quant_impact    |      |  quant_impact    |
           |  .json           |      |  .md             |
           +------------------+      +------------------+
```

---

## 5. File Structure

### 5.1 Standalone Package (HaoLine)

**Repository:** [github.com/mdayku/HaoLine](https://github.com/mdayku/HaoLine)

```
HaoLine/
|
+-- pyproject.toml           # Package metadata, dependencies, CLI entrypoints
+-- README.md                # Installation and usage guide
+-- LICENSE                  # MIT License
|
+-- src/haoline/             # Main package
|   +-- __init__.py          # Public API exports, version
|   +-- cli.py               # Main CLI entrypoint (haoline command)
|   +-- compare.py           # Compare mode CLI
|   +-- analyzer.py          # ONNX graph analysis, FLOPs, params, memory
|   +-- patterns.py          # Architecture pattern detection
|   +-- risks.py             # Risk signal detection
|   +-- report.py            # InspectionReport dataclass, ModelInspector
|   +-- hardware.py          # GPU profiles, estimator, multi-GPU
|   +-- operational_profiling.py  # Runtime benchmarking, profiling
|   +-- visualizations.py    # Matplotlib chart generation
|   +-- compare_visualizations.py # Compare mode charts, HTML
|   +-- html_export.py       # Interactive D3.js graph visualization
|   +-- pdf_generator.py     # Playwright PDF generation
|   +-- llm_summarizer.py    # AI-powered summaries (Anthropic)
|   +-- schema.py            # JSON schema validation
|   +-- layer_summary.py     # Per-layer metrics
|   +-- hierarchical_graph.py # Collapsible graph structure
|   +-- edge_analysis.py     # Tensor flow analysis
|   |
|   +-- formats/             # Multi-format readers (Epics 19-24)
|   |   +-- __init__.py      # detect_format(), reader exports
|   |   +-- gguf.py          # GGUF reader (llama.cpp, pure Python)
|   |   +-- safetensors.py   # SafeTensors reader (HuggingFace)
|   |   +-- tflite.py        # TFLite reader (mobile/edge)
|   |   +-- coreml.py        # CoreML reader (Apple)
|   |   +-- openvino.py      # OpenVINO reader (Intel)
|   |
|   +-- tests/               # 229 unit tests
|   |   +-- conftest.py
|   |   +-- test_*.py
|   |
|   +-- examples/            # Usage examples
|       +-- basic_inspection.py
|       +-- compare_models.py
|       +-- hardware_estimation.py
|
+-- .github/workflows/ci.yml # GitHub Actions CI/CD
```

### 5.2 Legacy Location (ORT Fork)

The original development occurred in the ORT fork at `tools/python/util/autodoc/`.
This has been **extracted** to the standalone HaoLine package above.

```
onnxruntime/tools/python/util/
|
+-- autodoc/                 # Original development location (legacy)
+-- model_inspect.py         # Original CLI (legacy)
+-- model_inspect_compare.py # Original compare CLI (legacy)

docs/marcu/
|
+-- README.md             # Project overview
+-- PRD.md                # Product requirements
+-- Architecture.md       # This document
+-- BACKLOG.md            # Epic/Story tracking
```

### 5.2 Module Dependencies

```
model_inspect.py (CLI)
    |
    +-- analysis.py
    |       |
    |       +-- onnx (external)
    |       +-- numpy (external)
    |
    +-- visualizations.py
    |       |
    |       +-- matplotlib (external, optional)
    |
    +-- render_markdown.py
    |
    +-- render_html.py (optional)
            |
            +-- jinja2 (external, optional)
```

---

## 6. Integration Points

### 6.1 ONNX Runtime Integration

The tool integrates with existing ONNX Runtime tooling:

| Existing Tool | Integration Point |
|--------------|-------------------|
| `check_onnx_model_mobile_usability` | Similar CLI pattern, can share utilities |
| `onnxruntime_perf_test` | Autodoc can consume perf test output |
| ONNX Runtime Python bindings | Leverage existing model loading code |

### 6.2 External Pipeline Integration

```
+------------------+     +------------------+     +------------------+
|  YOLO Batch      |     |  ResNet Eval     |     |  BERT Eval       |
|  Eval Pipeline   |     |  Pipeline        |     |  Pipeline        |
+--------+---------+     +--------+---------+     +--------+---------+
         |                        |                        |
         v                        v                        v
+------------------------------------------------------------------------+
|                    Generic Eval/Perf JSON Schema                       |
|                                                                        |
|  { "model_id": "...", "precision": "...", "eval": {...}, "perf": {...}}|
+------------------------------------------------------------------------+
                                  |
                                  v
                         +------------------+
                         |  ONNX Autodoc    |
                         |  Compare Mode    |
                         +------------------+
```

### 6.3 CI/CD Integration

```yaml
# Example GitHub Actions workflow
name: Model Analysis

on:
  push:
    paths:
      - 'models/*.onnx'

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Install dependencies
        run: pip install onnxruntime onnx matplotlib

      - name: Analyze models
        run: |
          for model in models/*.onnx; do
            python -m onnxruntime.tools.model_inspect "$model" \
              --out-json "reports/$(basename $model .onnx).json" \
              --out-md "reports/$(basename $model .onnx).md"
          done

      - name: Upload reports
        uses: actions/upload-artifact@v2
        with:
          name: model-reports
          path: reports/
```

### 6.4 ONNX Graph Representation Classes

Understanding the ONNX graph data structures is essential for extending the autodoc tool.

#### 6.4.1 ONNX Python API (used by Autodoc)

| Class | Purpose | Key Attributes |
|-------|---------|----------------|
| `onnx.ModelProto` | Top-level model container | `graph`, `ir_version`, `producer_name`, `opset_import` |
| `onnx.GraphProto` | Computational graph | `node`, `input`, `output`, `initializer`, `value_info` |
| `onnx.NodeProto` | Single operation | `op_type`, `name`, `input`, `output`, `attribute`, `domain` |
| `onnx.TensorProto` | Weight/initializer tensor | `dims`, `data_type`, `raw_data`, `name` |
| `onnx.ValueInfoProto` | Input/output tensor info | `name`, `type` (includes shape via `tensor_type.shape`) |
| `onnx.AttributeProto` | Node attribute | `name`, `type`, `i`, `f`, `s`, `ints`, `floats`, etc. |

#### 6.4.2 How Autodoc Uses These Classes

```python
# Loading and traversing the graph
model = onnx.load("model.onnx")
graph = model.graph

# Iterate over nodes
for node in graph.node:
    print(f"{node.name}: {node.op_type}")
    print(f"  Inputs: {list(node.input)}")
    print(f"  Outputs: {list(node.output)}")

# Access initializers (weights)
for init in graph.initializer:
    tensor = onnx.numpy_helper.to_array(init)
    print(f"{init.name}: shape={tensor.shape}, dtype={tensor.dtype}")

# Get input/output shapes
for vi in graph.input:
    shape = [d.dim_value or d.dim_param for d in vi.type.tensor_type.shape.dim]
    print(f"{vi.name}: {shape}")
```

#### 6.4.3 ONNX Runtime C++ Classes (for reference)

| C++ Class | Python Equivalent | Location |
|-----------|-------------------|----------|
| `onnxruntime::Graph` | `onnx.GraphProto` | `onnxruntime/core/graph/graph.h` |
| `onnxruntime::Node` | `onnx.NodeProto` | `onnxruntime/core/graph/graph.h` |
| `onnxruntime::NodeArg` | Input/output tensor | `onnxruntime/core/graph/graph.h` |
| `ONNX_NAMESPACE::TensorProto` | `onnx.TensorProto` | Uses ONNX proto directly |

The C++ API provides additional methods for graph traversal and mutation that aren't available in pure ONNX Python API:
- `Graph::Nodes()` - iterator over all nodes
- `Node::InputDefs()` / `OutputDefs()` - typed tensor access
- `Graph::GetProducerNode()` / `GetConsumerNodes()` - dependency tracking

### 6.5 Extension Points and Patterns

#### 6.5.1 Adding New Operator Analysis

To add FLOP estimation for a new operator in `analyzer.py`:

```python
# In MetricsEngine._estimate_flops_for_node()
def _estimate_flops_for_node(self, node: NodeInfo, graph_info: GraphInfo) -> int:
    if node.op_type == "MyNewOp":
        return self._estimate_mynewop_flops(node, graph_info)
    # ... existing handlers

def _estimate_mynewop_flops(self, node: NodeInfo, graph_info: GraphInfo) -> int:
    # Extract shapes from graph_info.value_shapes
    # Calculate FLOPs based on operator semantics
    return flops
```

#### 6.5.2 Adding New Pattern Detection

To detect a new architectural pattern in `patterns.py`:

```python
# In PatternAnalyzer.group_into_blocks()
def group_into_blocks(self, graph_info: GraphInfo) -> list[Block]:
    blocks = []
    blocks.extend(self.detect_conv_bn_relu(graph_info))
    blocks.extend(self.detect_my_new_pattern(graph_info))  # Add here
    # ...

def detect_my_new_pattern(self, graph_info: GraphInfo) -> list[Block]:
    blocks: list[Block] = []
    for node in graph_info.nodes:
        if self._matches_my_pattern(node, graph_info):
            blocks.append(Block(
                block_type="MyPattern",
                name=f"mypattern_{len(blocks)}",
                nodes=[node.name],
                # ...
            ))
    return blocks
```

#### 6.5.3 Adding New Hardware Profiles

To add a new GPU profile in `hardware.py`:

```python
HARDWARE_PROFILES["my-new-gpu"] = HardwareProfile(
    name="My New GPU",
    vram_bytes=16 * 1024**3,           # 16 GB
    peak_fp32_tflops=20.0,              # 20 TFLOPS FP32
    peak_fp16_tflops=40.0,              # 40 TFLOPS FP16
    memory_bandwidth_gbps=500,          # 500 GB/s
    tdp_watts=200,                      # 200W TDP
)
```

#### 6.5.4 Adding New Risk Signals

To add a new risk heuristic in `risks.py`:

```python
# In RiskAnalyzer.analyze()
def analyze(self, graph_info: GraphInfo, blocks: list[Block]) -> list[RiskSignal]:
    signals = []
    # ... existing checks
    signal = self.check_my_new_risk(graph_info, blocks)
    if signal:
        signals.append(signal)
    return signals

def check_my_new_risk(self, graph_info: GraphInfo, blocks: list[Block]) -> RiskSignal | None:
    # Implement detection logic
    if detected:
        return RiskSignal(
            id="my_new_risk",
            severity="warning",  # info | warning | high
            description="Description of what was detected",
            nodes=affected_node_names,
            recommendation="What the user should do",
        )
    return None
```

#### 6.5.5 Adding New Format Readers

To add a new model format in `formats/`:

```python
# In formats/myformat.py
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

@dataclass
class MyFormatInfo:
    """Metadata extracted from MyFormat files."""
    total_params: int
    total_size_bytes: int
    tensors: List[MyFormatTensorInfo]
    # Format-specific fields...

class MyFormatReader:
    """Reader for MyFormat model files."""

    def read(self, path: Path) -> MyFormatInfo:
        """Read and parse the model file."""
        pass

    def read_header_only(self, path: Path) -> MyFormatInfo:
        """Read metadata without loading weights (for large files)."""
        pass

def is_myformat_file(path: Path) -> bool:
    """Check if file is a MyFormat file by extension/magic bytes."""
    return path.suffix.lower() == ".myformat"

def is_available() -> bool:
    """Check if required dependencies are installed."""
    try:
        import myformat_lib
        return True
    except ImportError:
        return False
```

Then register in `formats/__init__.py`:

```python
from .myformat import MyFormatReader, MyFormatInfo, is_myformat_file, is_available as myformat_available
```

#### 6.5.6 Key Integration Files

| File | Extension Point |
|------|-----------------|
| `analyzer.py` | New operators, metrics, memory estimation |
| `patterns.py` | New architectural patterns, block detection |
| `risks.py` | New risk heuristics, severity thresholds |
| `hardware.py` | New GPU profiles, estimation formulas |
| `report.py` | New output sections, report formats |
| `visualizations.py` | New chart types, themes |
| `llm_summarizer.py` | New LLM providers, prompt templates |
| `formats/*.py` | New model format readers |

---

## 7. Deployment Architecture

### 7.1 Standalone CLI

```
+------------------+
|  User Machine    |
|                  |
|  +------------+  |
|  | Python     |  |
|  | 3.10+      |  |
|  +------------+  |
|       |          |
|       v          |
|  +------------+  |
|  | model_     |  |     +------------------+
|  | inspect    |--+---> |  .json / .md     |
|  +------------+  |     |  reports         |
|                  |     +------------------+
+------------------+
```

### 7.2 Integration with Model Registry

```
+------------------+     +------------------+     +------------------+
|  Model Registry  | --> |  ONNX Autodoc    | --> |  Registry        |
|  (upload event)  |     |  (webhook/job)   |     |  Metadata Store  |
+------------------+     +------------------+     +------------------+
                                                          |
                                                          v
                                                 +------------------+
                                                 |  Search/Filter   |
                                                 |  by Metrics      |
                                                 +------------------+
```

### 7.3 Batch Processing Mode

```
+------------------+
|  Model Directory |
|                  |
|  +-- model1.onnx |
|  +-- model2.onnx |
|  +-- model3.onnx |
+--------+---------+
         |
         v
+------------------+     +------------------+
|  Batch Script    | --> |  reports/        |
|                  |     |  +-- model1.json |
|  for model in    |     |  +-- model1.md   |
|    models/*.onnx |     |  +-- model2.json |
|  do inspect      |     |  +-- model2.md   |
|  done            |     |  +-- ...         |
+------------------+     +------------------+
```

---

## 8. Design Decisions

### 8.1 Decision Log

| Decision | Rationale | Alternatives Considered |
|----------|-----------|------------------------|
| Python-first implementation | Faster development, easier ONNX integration | C++-first with Python bindings |
| Matplotlib for visualization | Widely available, simple API, no JS dependencies | Plotly, D3.js, Vega-Lite |
| JSON as primary output format | Machine-readable, widely supported | Protobuf, MessagePack |
| Optional LLM integration | Not everyone has API access; core functionality works without | Required LLM dependency |
| Hardware profiles as JSON files | Easy to extend, no code changes for new hardware | Hardcoded profiles |
| Graceful degradation | Tool should always produce some output | Fail fast on any error |

### 8.2 Trade-offs

| Trade-off | Choice | Consequence |
|-----------|--------|-------------|
| Accuracy vs Speed | Approximate FLOPs | May not match exact profiler results |
| Simplicity vs Completeness | Focus on common ops | Exotic ops get generic estimates |
| Bundled vs External | Integrated into ORT | Requires ORT build; could be standalone |

### 8.3 Future Considerations

- **C++ Core**: For performance-critical deployments, implement core analysis in C++
- **Interactive Mode**: Web-based UI for exploring model architecture
- **Model Zoo Integration**: Pre-computed reports for ONNX Model Zoo
- **Diff Mode**: Visual diff between model versions
- **Custom Risk Rules**: User-defined heuristics via YAML config

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | Dec 2024 | Marcus | Initial architecture document |
