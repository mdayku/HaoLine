# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
DEPRECATED: Legacy argparse-based CLI.

This module is kept for backwards compatibility and helper functions.
The main CLI has been migrated to Typer in `cli_typer.py`.

DO NOT add new features to this file. Use cli_typer.py instead.

Helper functions still used by cli_typer.py:
- convert_pytorch_to_onnx()
- generate_markdown()
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .quantization_advisor import QuantizationAdvice
    from .quantization_linter import QuantizationLintResult

from . import ModelInspector
from .edge_analysis import EdgeAnalyzer
from .hardware import (
    CLOUD_INSTANCES,
    HardwareEstimator,
    create_multi_gpu_profile,
    detect_local_hardware,
    get_cloud_instance,
    get_profile,
)
from .hierarchical_graph import HierarchicalGraphBuilder
from .html_export import HTMLExporter
from .html_export import generate_html as generate_graph_html
from .layer_summary import LayerSummaryBuilder, generate_html_table
from .llm_summarizer import (
    LLMSummarizer,
)
from .llm_summarizer import (
    has_api_key as has_llm_api_key,
)
from .llm_summarizer import (
    is_available as is_llm_available,
)
from .operational_profiling import BatchSizeSweep, OperationalProfiler
from .patterns import PatternAnalyzer
from .pdf_generator import (
    PDFGenerator,
)
from .pdf_generator import (
    is_available as is_pdf_available,
)
from .visualizations import (
    VisualizationGenerator,
)
from .visualizations import (
    is_available as is_viz_available,
)


class ProgressIndicator:
    """Simple progress indicator for CLI operations."""

    def __init__(self, enabled: bool = True, quiet: bool = False):
        self.enabled = enabled and not quiet
        self._current_step = 0
        self._total_steps = 0

    def start(self, total_steps: int, description: str = "Processing"):
        """Start progress tracking."""
        self._total_steps = total_steps
        self._current_step = 0
        if self.enabled:
            print(f"\n{description}...")

    def step(self, message: str):
        """Mark completion of a step."""
        self._current_step += 1
        if self.enabled:
            pct = (self._current_step / self._total_steps * 100) if self._total_steps else 0
            print(f"  [{self._current_step}/{self._total_steps}] {message} ({pct:.0f}%)")

    def finish(self, message: str = "Done"):
        """Mark completion of all steps."""
        if self.enabled:
            print(f"  {message}\n")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description="Analyze an ONNX model and generate architecture documentation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inspection with console output (auto-detects local hardware)
  python -m haoline model.onnx

  # Use specific NVIDIA GPU profile for estimates
  python -m haoline model.onnx --hardware a100

  # List available hardware profiles
  python -m haoline --list-hardware

  # Generate JSON report with hardware estimates
  python -m haoline model.onnx --hardware rtx4090 --out-json report.json

  # Specify precision and batch size for hardware estimates
  python -m haoline model.onnx --hardware t4 --precision fp16 --batch-size 8

  # Convert PyTorch model to ONNX and analyze
  python -m haoline --from-pytorch model.pt --input-shape 1,3,224,224

  # Convert TensorFlow SavedModel to ONNX and analyze
  python -m haoline --from-tensorflow ./saved_model_dir --out-html report.html

  # Convert Keras .h5 model to ONNX and analyze
  python -m haoline --from-keras model.h5 --keep-onnx converted.onnx

  # Convert TFLite model to ONNX and analyze
  python -m haoline --from-tflite model.tflite --out-html report.html

  # Convert TensorFlow frozen graph to ONNX (requires input/output names)
  python -m haoline --from-frozen-graph model.pb --tf-inputs input:0 --tf-outputs output:0

  # Convert JAX model to ONNX (requires apply function and input shape)
  python -m haoline --from-jax params.pkl --jax-apply-fn my_model:apply --input-shape 1,3,224,224

  # Generate Steam-style system requirements
  python -m haoline model.onnx --system-requirements

  # Run batch size sweep
  python -m haoline model.onnx --hardware a100 --sweep-batch-sizes

  # Run resolution sweep for vision models
  python -m haoline model.onnx --hardware rtx4090 --sweep-resolutions auto

  # Custom resolutions for object detection
  python -m haoline yolo.onnx --hardware rtx4090 --sweep-resolutions "320x320,640x640,1280x1280"

  # List supported input formats
  python -m haoline --list-formats

  # List available format conversions
  python -m haoline --list-conversions

  # Convert PyTorch to ONNX and save
  python -m haoline --from-pytorch model.pt --input-shape 1,3,224,224 --convert-to onnx --convert-output model.onnx

  # Note: ONNX to TFLite conversion is currently unavailable due to
  # upstream compatibility issues with TensorFlow 2.16+ / Keras 3.x.
  # Use TFLite -> ONNX direction instead:

  # Export model as Universal IR (JSON)
  python -m haoline model.onnx --export-ir model_ir.json

  # Export graph visualization (DOT or PNG)
  python -m haoline model.onnx --export-graph graph.dot
  python -m haoline model.onnx --export-graph graph.png --graph-max-nodes 200
""",
    )

    parser.add_argument(
        "model_path",
        type=pathlib.Path,
        nargs="?",  # Optional now since --list-hardware doesn't need it
        help="Path to model file. Supports ONNX (.onnx), TensorRT engines (.engine, .plan).",
    )

    parser.add_argument(
        "--schema",
        action="store_true",
        help="Output the JSON Schema for InspectionReport and exit. "
        "Useful for integrating HaoLine output with other tools.",
    )

    parser.add_argument(
        "--out-json",
        type=pathlib.Path,
        default=None,
        help="Output path for JSON report. If not specified, no JSON is written.",
    )

    parser.add_argument(
        "--out-md",
        type=pathlib.Path,
        default=None,
        help="Output path for Markdown model card. If not specified, no Markdown is written.",
    )

    parser.add_argument(
        "--out-html",
        type=pathlib.Path,
        default=None,
        help="Output path for HTML report with embedded images. Single shareable file.",
    )

    parser.add_argument(
        "--out-pdf",
        type=pathlib.Path,
        default=None,
        help="Output path for PDF report. Requires playwright: pip install playwright && playwright install chromium",
    )

    parser.add_argument(
        "--html-graph",
        type=pathlib.Path,
        default=None,
        help="Output path for interactive graph visualization (standalone HTML with D3.js).",
    )

    parser.add_argument(
        "--layer-csv",
        type=pathlib.Path,
        default=None,
        help="Output path for per-layer metrics CSV (params, FLOPs, memory per layer).",
    )

    parser.add_argument(
        "--include-graph",
        action="store_true",
        help="Include interactive graph in --out-html report (makes HTML larger but more informative).",
    )

    parser.add_argument(
        "--include-layer-table",
        action="store_true",
        help="Include per-layer summary table in --out-html report.",
    )

    # PyTorch conversion options
    pytorch_group = parser.add_argument_group("PyTorch Conversion Options")
    pytorch_group.add_argument(
        "--from-pytorch",
        type=pathlib.Path,
        default=None,
        metavar="MODEL_PATH",
        help="Convert a PyTorch model (.pth, .pt) to ONNX before analysis. Requires torch.",
    )
    pytorch_group.add_argument(
        "--input-shape",
        type=str,
        default=None,
        metavar="SHAPE",
        help="Input shape for PyTorch conversion, e.g., '1,3,224,224'. Required with --from-pytorch.",
    )
    pytorch_group.add_argument(
        "--keep-onnx",
        type=pathlib.Path,
        default=None,
        metavar="PATH",
        help="Save the converted ONNX model to this path (otherwise uses temp file).",
    )
    pytorch_group.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNX opset version for PyTorch export (default: 17).",
    )
    pytorch_group.add_argument(
        "--pytorch-weights",
        type=pathlib.Path,
        default=None,
        metavar="PATH",
        help="Path to original PyTorch weights (.pt) to extract class names/metadata. "
        "Useful when analyzing a pre-converted ONNX file.",
    )

    # TensorFlow/Keras conversion options
    tf_group = parser.add_argument_group("TensorFlow/Keras Conversion Options")
    tf_group.add_argument(
        "--from-tensorflow",
        type=pathlib.Path,
        default=None,
        metavar="MODEL_PATH",
        help="Convert a TensorFlow SavedModel directory to ONNX before analysis. Requires tf2onnx.",
    )
    tf_group.add_argument(
        "--from-keras",
        type=pathlib.Path,
        default=None,
        metavar="MODEL_PATH",
        help="Convert a Keras model (.h5, .keras) to ONNX before analysis. Requires tf2onnx.",
    )
    tf_group.add_argument(
        "--from-tflite",
        type=pathlib.Path,
        default=None,
        metavar="MODEL_PATH",
        help="Convert a TFLite model (.tflite) to ONNX before analysis. Requires tflite2onnx.",
    )
    tf_group.add_argument(
        "--from-frozen-graph",
        type=pathlib.Path,
        default=None,
        metavar="MODEL_PATH",
        help="Convert a TensorFlow frozen graph (.pb) to ONNX. Requires --tf-inputs and --tf-outputs.",
    )
    tf_group.add_argument(
        "--tf-inputs",
        type=str,
        default=None,
        metavar="NAMES",
        help="Comma-separated input tensor names for frozen graph conversion, e.g., 'input:0'.",
    )
    tf_group.add_argument(
        "--tf-outputs",
        type=str,
        default=None,
        metavar="NAMES",
        help="Comma-separated output tensor names for frozen graph conversion, e.g., 'output:0'.",
    )

    # JAX conversion options
    jax_group = parser.add_argument_group("JAX/Flax Conversion Options")
    jax_group.add_argument(
        "--from-jax",
        type=pathlib.Path,
        default=None,
        metavar="MODEL_PATH",
        help="Convert a JAX/Flax model to ONNX before analysis. Requires jax and tf2onnx. "
        "Expects a saved params file (.msgpack, .pkl) with --jax-apply-fn.",
    )
    jax_group.add_argument(
        "--jax-apply-fn",
        type=str,
        default=None,
        metavar="MODULE:FUNCTION",
        help="JAX apply function path, e.g., 'my_model:apply'. Required with --from-jax.",
    )

    # Hardware options
    hardware_group = parser.add_argument_group("Hardware Options")
    hardware_group.add_argument(
        "--hardware",
        type=str,
        default=None,
        metavar="PROFILE",
        help="Hardware profile for performance estimates. Use 'auto' to detect local hardware, "
        "or specify a profile name (e.g., 'a100', 'rtx4090', 't4'). Use --list-hardware to see all options.",
    )

    hardware_group.add_argument(
        "--list-hardware",
        action="store_true",
        help="List all available hardware profiles and exit.",
    )

    hardware_group.add_argument(
        "--precision",
        choices=["fp32", "fp16", "bf16", "int8"],
        default="fp32",
        help="Precision for hardware estimates (default: fp32).",
    )

    hardware_group.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for hardware estimates (default: 1).",
    )

    hardware_group.add_argument(
        "--gpu-count",
        type=int,
        default=1,
        metavar="N",
        help="Number of GPUs for multi-GPU estimates (default: 1). "
        "Scales compute and memory with efficiency factors for tensor parallelism.",
    )

    hardware_group.add_argument(
        "--cloud",
        type=str,
        default=None,
        metavar="INSTANCE",
        help="Cloud instance type for cost/performance estimates. "
        "E.g., 'aws-p4d-24xlarge', 'azure-nd-h100-v5'. Use --list-cloud to see options.",
    )

    hardware_group.add_argument(
        "--list-cloud",
        action="store_true",
        help="List all available cloud instance profiles and exit.",
    )

    hardware_group.add_argument(
        "--deployment-target",
        type=str,
        choices=["edge", "local", "cloud"],
        default=None,
        help="High-level deployment target to guide system requirement recommendations "
        "(edge device, local server, or cloud server).",
    )

    hardware_group.add_argument(
        "--target-latency-ms",
        type=float,
        default=None,
        help="Optional latency target (ms) for system requirements. "
        "If set, this is converted into a throughput target for recommendations.",
    )

    hardware_group.add_argument(
        "--target-throughput-fps",
        type=float,
        default=None,
        help="Optional throughput target (frames/requests per second) for system requirements.",
    )

    hardware_group.add_argument(
        "--deployment-fps",
        type=float,
        default=None,
        metavar="FPS",
        help="Target inference rate for deployment cost calculation (e.g., 3 for 3 fps continuous). "
        "Combined with --deployment-hours to estimate $/day and $/month.",
    )

    hardware_group.add_argument(
        "--deployment-hours",
        type=float,
        default=24.0,
        metavar="HOURS",
        help="Hours per day the model runs for deployment cost calculation (default: 24). "
        "E.g., 8 for business hours only.",
    )

    # Epic 6C features
    hardware_group.add_argument(
        "--system-requirements",
        action="store_true",
        help="Generate Steam-style system requirements (Minimum, Recommended, Optimal).",
    )

    hardware_group.add_argument(
        "--sweep-batch-sizes",
        action="store_true",
        help="Run analysis across multiple batch sizes (1, 2, 4, ..., 128) to find optimal throughput.",
    )

    hardware_group.add_argument(
        "--no-benchmark",
        action="store_true",
        help="Use theoretical estimates instead of actual inference for batch sweeps (faster but less accurate).",
    )

    hardware_group.add_argument(
        "--input-resolution",
        type=str,
        default=None,
        help=(
            "Override input resolution for analysis. Format: HxW (e.g., 640x640). "
            "For vision models, affects FLOPs and memory estimates."
        ),
    )

    hardware_group.add_argument(
        "--sweep-resolutions",
        type=str,
        default=None,
        help=(
            "Run resolution sweep analysis. Provide comma-separated resolutions "
            "(e.g., '224x224,384x384,512x512,640x640') or 'auto' for common resolutions."
        ),
    )

    # Epic 9: Runtime Profiling Options (defaults to ON for real measurements)
    profiling_group = parser.add_argument_group("Runtime Profiling Options")
    profiling_group.add_argument(
        "--no-profile",
        action="store_true",
        help="Disable per-layer ONNX Runtime profiling (faster but less detailed).",
    )

    profiling_group.add_argument(
        "--profile-runs",
        type=int,
        default=10,
        metavar="N",
        help="Number of inference runs for profiling (default: 10).",
    )

    profiling_group.add_argument(
        "--no-gpu-metrics",
        action="store_true",
        help="Disable GPU metrics capture (VRAM, utilization, temperature).",
    )

    profiling_group.add_argument(
        "--no-bottleneck-analysis",
        action="store_true",
        help="Disable compute vs memory bottleneck analysis.",
    )

    profiling_group.add_argument(
        "--no-benchmark-resolutions",
        action="store_true",
        help="Disable resolution benchmarking (use theoretical estimates instead).",
    )

    # Visualization options
    viz_group = parser.add_argument_group("Visualization Options")
    viz_group.add_argument(
        "--with-plots",
        action="store_true",
        help="Generate visualization assets (requires matplotlib).",
    )

    viz_group.add_argument(
        "--assets-dir",
        type=pathlib.Path,
        default=None,
        metavar="PATH",
        help="Directory for plot PNG files (default: same directory as output files, or 'assets/').",
    )

    # Quantization Linting options
    quant_group = parser.add_argument_group("Quantization Linting Options")
    quant_group.add_argument(
        "--lint-quantization",
        action="store_true",
        help="Analyze model for INT8 quantization readiness. "
        "Detects problematic ops, dynamic shapes, and provides recommendations.",
    )
    quant_group.add_argument(
        "--lint-quant",
        action="store_true",
        help="Alias for --lint-quantization.",
    )
    quant_group.add_argument(
        "--quant-report",
        type=pathlib.Path,
        default=None,
        metavar="PATH",
        help="Output path for quantization lint report (Markdown). Default: prints to console.",
    )
    quant_group.add_argument(
        "--quant-report-html",
        type=pathlib.Path,
        default=None,
        metavar="PATH",
        help="Output path for quantization lint report (HTML format).",
    )
    quant_group.add_argument(
        "--quant-llm-advice",
        action="store_true",
        help="Use LLM to generate intelligent quantization recommendations. "
        "Requires OPENAI_API_KEY env var or --llm-model.",
    )
    quant_group.add_argument(
        "--quant-advice-report",
        type=pathlib.Path,
        default=None,
        metavar="PATH",
        help="Output path for QAT readiness report with LLM advice (Markdown).",
    )

    # LLM options
    llm_group = parser.add_argument_group("LLM Summarization Options")
    llm_group.add_argument(
        "--llm-summary",
        action="store_true",
        help="Generate LLM-powered summaries (requires openai package and OPENAI_API_KEY env var).",
    )

    llm_group.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4o-mini",
        metavar="MODEL",
        help="OpenAI model to use for summaries (default: gpt-4o-mini).",
    )

    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Logging verbosity level (default: info).",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output. Only write to files if --out-json or --out-md specified.",
    )

    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress indicators during analysis (useful for large models).",
    )

    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run in offline mode. Fails if any network access is attempted. "
        "Disables LLM summaries and other features requiring internet.",
    )

    # Privacy controls
    privacy_group = parser.add_argument_group("Privacy Options")
    privacy_group.add_argument(
        "--redact-names",
        action="store_true",
        help="Anonymize layer and tensor names in output (e.g., layer_001, tensor_042). "
        "Useful for sharing reports without revealing proprietary architecture details.",
    )
    privacy_group.add_argument(
        "--summary-only",
        action="store_true",
        help="Output only aggregate statistics (params, FLOPs, memory). "
        "Omit per-layer details and graph structure for maximum privacy.",
    )

    # Format conversion
    convert_group = parser.add_argument_group("Format Conversion Options")
    convert_group.add_argument(
        "--convert-to",
        type=str,
        choices=["onnx"],
        default=None,
        metavar="FORMAT",
        help="Convert the model to ONNX format. "
        "Use with --from-pytorch, --from-tensorflow, or other input formats. "
        "Note: ONNX->TFLite is unavailable due to upstream TF/Keras 3.x issues.",
    )
    convert_group.add_argument(
        "--convert-output",
        type=pathlib.Path,
        default=None,
        metavar="PATH",
        help="Output path for format conversion. Required when using --convert-to.",
    )
    convert_group.add_argument(
        "--list-conversions",
        action="store_true",
        help="List available format conversion paths and exit.",
    )
    convert_group.add_argument(
        "--list-formats",
        action="store_true",
        help="List all supported input formats and exit.",
    )

    # Universal IR export
    ir_group = parser.add_argument_group("Universal IR Export Options")
    ir_group.add_argument(
        "--export-ir",
        type=pathlib.Path,
        default=None,
        metavar="PATH",
        help="Export model as Universal IR to JSON file. "
        "Provides format-agnostic representation of the model graph.",
    )
    ir_group.add_argument(
        "--export-graph",
        type=pathlib.Path,
        default=None,
        metavar="PATH",
        help="Export model graph visualization. Supports .dot (Graphviz) and .png formats. "
        "Requires graphviz package for PNG: pip install graphviz",
    )
    ir_group.add_argument(
        "--graph-max-nodes",
        type=int,
        default=500,
        metavar="N",
        help="Maximum nodes to include in graph visualization (default: 500). "
        "Prevents huge graphs from crashing.",
    )

    # TensorRT comparison
    trt_group = parser.add_argument_group("TensorRT Comparison Options")
    trt_group.add_argument(
        "--compare-trt",
        type=pathlib.Path,
        default=None,
        metavar="ENGINE_PATH",
        help="Compare ONNX model with its compiled TensorRT engine. "
        "Shows layer mappings, fusions, precision changes. "
        "Usage: haoline model.onnx --compare-trt model.engine",
    )
    trt_group.add_argument(
        "--quant-bottlenecks",
        action="store_true",
        default=False,
        help="Show quantization bottleneck analysis for TRT engines. "
        "Identifies failed fusions, FP32 fallback zones, and speed impact. "
        "Usage: haoline model.engine --quant-bottlenecks",
    )

    return parser.parse_args()


def setup_logging(log_level: str) -> logging.Logger:
    """Configure logging for the CLI."""
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }

    logging.basicConfig(
        level=level_map.get(log_level, logging.INFO),
        format="%(levelname)s - %(message)s",
    )

    return logging.getLogger("haoline")


def _generate_markdown_with_extras(
    report, viz_paths: dict, report_dir: pathlib.Path, llm_summary=None
) -> str:
    """Generate markdown with embedded visualizations and LLM summaries."""
    lines = []
    base_md = report.to_markdown()

    # If we have an LLM summary, insert it after the header
    if llm_summary and llm_summary.success:
        # Insert executive summary after the metadata section
        header_end = base_md.find("## Graph Summary")
        if header_end != -1:
            lines.append(base_md[:header_end])
            lines.append("## Executive Summary\n")
            if llm_summary.short_summary:
                lines.append(f"{llm_summary.short_summary}\n")
            if llm_summary.detailed_summary:
                lines.append(f"\n{llm_summary.detailed_summary}\n")
            lines.append(f"\n*Generated by {llm_summary.model_used}*\n\n")
            base_md = base_md[header_end:]

    # Split the markdown at the Complexity Metrics section to insert plots
    sections = base_md.split("## Complexity Metrics")

    if len(sections) < 2:
        # No complexity section found, just append plots at end
        lines.append(base_md)
    else:
        lines.append(sections[0])

        # Insert visualizations section before Complexity Metrics
        if viz_paths:
            lines.append("## Visualizations\n")

            if "complexity_summary" in viz_paths:
                rel_path = (
                    viz_paths["complexity_summary"].relative_to(report_dir)
                    if viz_paths["complexity_summary"].is_relative_to(report_dir)
                    else viz_paths["complexity_summary"]
                )
                lines.append("### Complexity Overview\n")
                lines.append(f"![Complexity Summary]({rel_path})\n")

            if "op_histogram" in viz_paths:
                rel_path = (
                    viz_paths["op_histogram"].relative_to(report_dir)
                    if viz_paths["op_histogram"].is_relative_to(report_dir)
                    else viz_paths["op_histogram"]
                )
                lines.append("### Operator Distribution\n")
                lines.append(f"![Operator Histogram]({rel_path})\n")

            if "param_distribution" in viz_paths:
                rel_path = (
                    viz_paths["param_distribution"].relative_to(report_dir)
                    if viz_paths["param_distribution"].is_relative_to(report_dir)
                    else viz_paths["param_distribution"]
                )
                lines.append("### Parameter Distribution\n")
                lines.append(f"![Parameter Distribution]({rel_path})\n")

            if "flops_distribution" in viz_paths:
                rel_path = (
                    viz_paths["flops_distribution"].relative_to(report_dir)
                    if viz_paths["flops_distribution"].is_relative_to(report_dir)
                    else viz_paths["flops_distribution"]
                )
                lines.append("### FLOPs Distribution\n")
                lines.append(f"![FLOPs Distribution]({rel_path})\n")

            lines.append("")

        lines.append("## Complexity Metrics" + sections[1])

    return "\n".join(lines)


def _extract_ultralytics_metadata(
    weights_path: pathlib.Path,
    logger: logging.Logger,
) -> dict[str, Any] | None:
    """
    Extract metadata from an Ultralytics model (.pt file).

    Returns dict with task, num_classes, class_names or None if not Ultralytics.
    """
    try:
        from ultralytics import YOLO

        model = YOLO(str(weights_path))

        return {
            "task": model.task,
            "num_classes": len(model.names),
            "class_names": list(model.names.values()),
            "source": "ultralytics",
        }
    except ImportError:
        logger.debug("ultralytics not installed, skipping metadata extraction")
        return None
    except Exception as e:
        logger.debug(f"Could not extract Ultralytics metadata: {e}")
        return None


def _convert_pytorch_to_onnx(
    pytorch_path: pathlib.Path,
    input_shape_str: str | None,
    output_path: pathlib.Path | None,
    opset_version: int,
    logger: logging.Logger,
) -> tuple[pathlib.Path | None, Any]:
    """
    Convert a PyTorch model to ONNX format.

    Args:
        pytorch_path: Path to PyTorch model (.pth, .pt)
        input_shape_str: Input shape as comma-separated string, e.g., "1,3,224,224"
        output_path: Where to save ONNX file (None = temp file)
        opset_version: ONNX opset version
        logger: Logger instance

    Returns:
        Tuple of (onnx_path, temp_file_handle_or_None)
    """
    # Check if torch is available
    try:
        import torch
    except ImportError:
        logger.error("PyTorch not installed. Install with: pip install torch")
        return None, None

    pytorch_path = pytorch_path.resolve()
    if not pytorch_path.exists():
        logger.error(f"PyTorch model not found: {pytorch_path}")
        return None, None

    # Parse input shape
    if not input_shape_str:
        logger.error(
            "--input-shape is required for PyTorch conversion. Example: --input-shape 1,3,224,224"
        )
        return None, None

    try:
        input_shape = tuple(int(x.strip()) for x in input_shape_str.split(","))
        logger.info(f"Input shape: {input_shape}")
    except ValueError:
        logger.error(
            f"Invalid --input-shape format: '{input_shape_str}'. "
            "Use comma-separated integers, e.g., '1,3,224,224'"
        )
        return None, None

    # Load PyTorch model
    logger.info(f"Loading PyTorch model from: {pytorch_path}")
    model = None

    # Try 1: TorchScript model (.pt files from torch.jit.save)
    try:
        model = torch.jit.load(str(pytorch_path), map_location="cpu")
        logger.info(f"Loaded TorchScript model: {type(model).__name__}")
    except Exception:
        pass

    # Try 2: Check for Ultralytics YOLO format first
    if model is None:
        try:
            loaded = torch.load(pytorch_path, map_location="cpu", weights_only=False)

            if isinstance(loaded, dict):
                # Check if it's an Ultralytics model (has 'model' key with the actual model)
                if "model" in loaded and hasattr(loaded.get("model"), "forward"):
                    logger.info("Detected Ultralytics YOLO format, using native export...")
                    try:
                        from ultralytics import YOLO

                        yolo_model = YOLO(str(pytorch_path))

                        # Determine output path for Ultralytics export
                        if output_path:
                            onnx_out = output_path.resolve()
                        else:
                            import tempfile as tf

                            temp = tf.NamedTemporaryFile(suffix=".onnx", delete=False)
                            onnx_out = pathlib.Path(temp.name)
                            temp.close()

                        # Export using Ultralytics (handles all the complexity)
                        yolo_model.export(
                            format="onnx",
                            imgsz=input_shape[2] if len(input_shape) >= 3 else 640,
                            simplify=True,
                            opset=opset_version,
                        )

                        # Ultralytics saves next to the .pt file, move if needed
                        default_onnx = pytorch_path.with_suffix(".onnx")
                        if default_onnx.exists() and default_onnx != onnx_out:
                            import shutil

                            shutil.move(str(default_onnx), str(onnx_out))

                        logger.info(f"ONNX model saved to: {onnx_out}")
                        return onnx_out, None if output_path else onnx_out

                    except ImportError:
                        logger.error(
                            "Ultralytics YOLO model detected but 'ultralytics' package not installed.\n"
                            "Install with: pip install ultralytics\n"
                            "Then re-run this command."
                        )
                        return None, None

                # It's a regular state_dict - we can't use it directly
                logger.error(
                    "Model file appears to be a state_dict (weights only). "
                    "To convert, you need either:\n"
                    "  1. A TorchScript model: torch.jit.save(torch.jit.script(model), 'model.pt')\n"
                    "  2. A full model: torch.save(model, 'model.pth')  # run from same codebase\n"
                    "  3. Export to ONNX directly in your training code using torch.onnx.export()"
                )
                return None, None

            model = loaded
            logger.info(f"Loaded PyTorch model: {type(model).__name__}")

        except Exception as e:
            error_msg = str(e)
            if "Can't get attribute" in error_msg:
                logger.error(
                    "Failed to load model - class definition not found.\n"
                    "The model was saved with torch.save(model, ...) which requires "
                    "the original class to be importable.\n\n"
                    "Solutions:\n"
                    "  1. Save as TorchScript: torch.jit.save(torch.jit.script(model), 'model.pt')\n"
                    "  2. Export to ONNX in your code: torch.onnx.export(model, dummy_input, 'model.onnx')\n"
                    "  3. Run this tool from the directory containing your model definition"
                )
            else:
                logger.error(f"Failed to load PyTorch model: {e}")
            return None, None

    if model is None:
        logger.error("Could not load the PyTorch model.")
        return None, None

    model.eval()

    # Create dummy input
    try:
        dummy_input = torch.randn(*input_shape)
        logger.info(f"Created dummy input with shape: {dummy_input.shape}")
    except Exception as e:
        logger.error(f"Failed to create input tensor: {e}")
        return None, None

    # Determine output path
    temp_file = None
    if output_path:
        onnx_path = output_path.resolve()
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        import tempfile

        temp_file = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        onnx_path = pathlib.Path(temp_file.name)
        temp_file.close()

    # Export to ONNX
    logger.info(f"Exporting to ONNX (opset {opset_version})...")
    try:
        torch.onnx.export(
            model,
            (dummy_input,),  # Wrap in tuple as required by torch.onnx.export
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
        )
        logger.info(f"ONNX model saved to: {onnx_path}")

        # Verify the export
        import onnx

        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model validated successfully")

    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        if temp_file:
            try:
                onnx_path.unlink()
            except Exception:
                pass
        return None, None

    return onnx_path, temp_file


def _convert_tensorflow_to_onnx(
    tf_path: pathlib.Path,
    output_path: pathlib.Path | None,
    opset_version: int,
    logger: logging.Logger,
) -> tuple[pathlib.Path | None, Any]:
    """
    Convert a TensorFlow SavedModel to ONNX format.

    Args:
        tf_path: Path to TensorFlow SavedModel directory
        output_path: Where to save ONNX file (None = temp file)
        opset_version: ONNX opset version
        logger: Logger instance

    Returns:
        Tuple of (onnx_path, temp_file_handle_or_None)
    """
    # Check if tf2onnx is available
    import importlib.util

    if importlib.util.find_spec("tf2onnx") is None:
        logger.error("tf2onnx not installed. Install with: pip install tf2onnx tensorflow")
        return None, None

    tf_path = tf_path.resolve()
    if not tf_path.exists():
        logger.error(f"TensorFlow model not found: {tf_path}")
        return None, None

    # Determine output path
    temp_file = None
    if output_path:
        onnx_path = output_path.resolve()
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        import tempfile

        temp_file = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        onnx_path = pathlib.Path(temp_file.name)
        temp_file.close()

    logger.info("Converting TensorFlow SavedModel to ONNX...")
    logger.info(f"  Source: {tf_path}")
    logger.info(f"  Target: {onnx_path}")

    try:
        import subprocess
        import sys

        # Use tf2onnx CLI for robustness (handles TF version quirks)
        cmd = [
            sys.executable,
            "-m",
            "tf2onnx.convert",
            "--saved-model",
            str(tf_path),
            "--output",
            str(onnx_path),
            "--opset",
            str(opset_version),
        ]

        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout for large models
        )

        if result.returncode != 0:
            logger.error(f"tf2onnx conversion failed:\n{result.stderr}")
            if temp_file:
                try:
                    onnx_path.unlink()
                except Exception:
                    pass
            return None, None

        logger.info("TensorFlow model converted successfully")

        # Verify the export
        import onnx

        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model validated successfully")

    except subprocess.TimeoutExpired:
        logger.error("tf2onnx conversion timed out after 10 minutes")
        if temp_file:
            try:
                onnx_path.unlink()
            except Exception:
                pass
        return None, None
    except Exception as e:
        logger.error(f"TensorFlow conversion failed: {e}")
        if temp_file:
            try:
                onnx_path.unlink()
            except Exception:
                pass
        return None, None

    return onnx_path, temp_file


def _convert_keras_to_onnx(
    keras_path: pathlib.Path,
    output_path: pathlib.Path | None,
    opset_version: int,
    logger: logging.Logger,
) -> tuple[pathlib.Path | None, Any]:
    """
    Convert a Keras model (.h5, .keras) to ONNX format.

    Args:
        keras_path: Path to Keras model file (.h5 or .keras)
        output_path: Where to save ONNX file (None = temp file)
        opset_version: ONNX opset version
        logger: Logger instance

    Returns:
        Tuple of (onnx_path, temp_file_handle_or_None)
    """
    # Check if tf2onnx is available
    import importlib.util

    if importlib.util.find_spec("tf2onnx") is None:
        logger.error("tf2onnx not installed. Install with: pip install tf2onnx tensorflow")
        return None, None

    keras_path = keras_path.resolve()
    if not keras_path.exists():
        logger.error(f"Keras model not found: {keras_path}")
        return None, None

    suffix = keras_path.suffix.lower()
    if suffix not in (".h5", ".keras", ".hdf5"):
        logger.warning(f"Unexpected Keras file extension: {suffix}. Proceeding anyway.")

    # Determine output path
    temp_file = None
    if output_path:
        onnx_path = output_path.resolve()
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        import tempfile

        temp_file = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        onnx_path = pathlib.Path(temp_file.name)
        temp_file.close()

    logger.info("Converting Keras model to ONNX...")
    logger.info(f"  Source: {keras_path}")
    logger.info(f"  Target: {onnx_path}")

    try:
        import subprocess
        import sys

        # Use tf2onnx CLI with --keras flag
        cmd = [
            sys.executable,
            "-m",
            "tf2onnx.convert",
            "--keras",
            str(keras_path),
            "--output",
            str(onnx_path),
            "--opset",
            str(opset_version),
        ]

        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        if result.returncode != 0:
            logger.error(f"tf2onnx conversion failed:\n{result.stderr}")
            if temp_file:
                try:
                    onnx_path.unlink()
                except Exception:
                    pass
            return None, None

        logger.info("Keras model converted successfully")

        # Verify the export
        import onnx

        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model validated successfully")

    except subprocess.TimeoutExpired:
        logger.error("tf2onnx conversion timed out after 10 minutes")
        if temp_file:
            try:
                onnx_path.unlink()
            except Exception:
                pass
        return None, None
    except Exception as e:
        logger.error(f"Keras conversion failed: {e}")
        if temp_file:
            try:
                onnx_path.unlink()
            except Exception:
                pass
        return None, None

    return onnx_path, temp_file


def _convert_onnx_to_tflite(
    onnx_path: pathlib.Path,
    output_path: pathlib.Path | None,
    logger: logging.Logger,
) -> tuple[pathlib.Path | None, Any]:
    """
    Convert an ONNX model to TFLite format.

    NOTE: This feature is currently UNAVAILABLE due to upstream compatibility
    issues. Both onnx2tf and onnx-tf are broken with TensorFlow 2.16+ / Keras 3.x.

    Args:
        onnx_path: Path to ONNX model file
        output_path: Where to save TFLite file (None = temp file)
        logger: Logger instance

    Returns:
        Tuple of (None, None) - conversion is not available
    """
    logger.error(
        "ONNX to TFLite conversion is currently unavailable.\n"
        "\n"
        "Both onnx2tf and onnx-tf have compatibility issues with TensorFlow 2.16+ / Keras 3.x.\n"
        "This is a known upstream issue affecting all ONNX-to-TFLite converters.\n"
        "\n"
        "Workarounds:\n"
        "  1. Use TFLite -> ONNX direction instead (--from-tflite works)\n"
        "  2. Convert via PyTorch: PyTorch -> ONNX -> analyze with HaoLine\n"
        "  3. Use TensorFlow directly: TF SavedModel -> TFLite (tf.lite.TFLiteConverter)\n"
        "\n"
        "Track issue: https://github.com/onnx/onnx-tensorflow/issues/1124"
    )
    return None, None


def _convert_tflite_to_onnx(
    tflite_path: pathlib.Path,
    output_path: pathlib.Path | None,
    opset_version: int,
    logger: logging.Logger,
) -> tuple[pathlib.Path | None, Any]:
    """
    Convert a TFLite model (.tflite) to ONNX format.

    Args:
        tflite_path: Path to TFLite model file
        output_path: Where to save ONNX file (None = temp file)
        opset_version: ONNX opset version (currently ignored, tflite2onnx uses own defaults)
        logger: Logger instance

    Returns:
        Tuple of (onnx_path, temp_file_handle_or_None)
    """
    # Suppress opset_version unused warning - kept for API consistency
    _ = opset_version

    # Check if tflite2onnx is available
    try:
        import tflite2onnx
    except ImportError:
        logger.error("tflite2onnx not installed. Install with: pip install haoline[tflite]")
        return None, None

    tflite_path = tflite_path.resolve()
    if not tflite_path.exists():
        logger.error(f"TFLite model not found: {tflite_path}")
        return None, None

    suffix = tflite_path.suffix.lower()
    if suffix != ".tflite":
        logger.warning(f"Unexpected TFLite file extension: {suffix}. Proceeding anyway.")

    # Determine output path
    temp_file = None
    if output_path:
        onnx_path = output_path.resolve()
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        import tempfile

        temp_file = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        onnx_path = pathlib.Path(temp_file.name)
        temp_file.close()

    logger.info("Converting TFLite model to ONNX...")
    logger.info(f"  Source: {tflite_path}")
    logger.info(f"  Target: {onnx_path}")

    try:
        # tflite2onnx.convert(tflite_path, onnx_path)
        tflite2onnx.convert(str(tflite_path), str(onnx_path))
        logger.info("TFLite model converted successfully")

        # Verify the export
        import onnx

        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model validated successfully")

    except Exception as e:
        logger.error(f"TFLite conversion failed: {e}")
        if temp_file:
            try:
                onnx_path.unlink()
            except Exception:
                pass
        return None, None

    return onnx_path, temp_file


def _convert_frozen_graph_to_onnx(
    pb_path: pathlib.Path,
    inputs: str,
    outputs: str,
    output_path: pathlib.Path | None,
    opset_version: int,
    logger: logging.Logger,
) -> tuple[pathlib.Path | None, Any]:
    """
    Convert a TensorFlow frozen graph (.pb) to ONNX format.

    Args:
        pb_path: Path to frozen graph .pb file
        inputs: Comma-separated input tensor names (e.g., "input:0")
        outputs: Comma-separated output tensor names (e.g., "output:0")
        output_path: Where to save ONNX file (None = temp file)
        opset_version: ONNX opset version
        logger: Logger instance

    Returns:
        Tuple of (onnx_path, temp_file_handle_or_None)
    """
    # Check if tf2onnx is available
    import importlib.util

    if importlib.util.find_spec("tf2onnx") is None:
        logger.error("tf2onnx not installed. Install with: pip install tf2onnx tensorflow")
        return None, None

    pb_path = pb_path.resolve()
    if not pb_path.exists():
        logger.error(f"Frozen graph not found: {pb_path}")
        return None, None

    if not inputs or not outputs:
        logger.error(
            "--tf-inputs and --tf-outputs are required for frozen graph conversion.\n"
            "Example: --from-frozen-graph model.pb --tf-inputs input:0 --tf-outputs output:0"
        )
        return None, None

    # Determine output path
    temp_file = None
    if output_path:
        onnx_path = output_path.resolve()
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        import tempfile

        temp_file = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        onnx_path = pathlib.Path(temp_file.name)
        temp_file.close()

    logger.info("Converting TensorFlow frozen graph to ONNX...")
    logger.info(f"  Source: {pb_path}")
    logger.info(f"  Inputs: {inputs}")
    logger.info(f"  Outputs: {outputs}")
    logger.info(f"  Target: {onnx_path}")

    try:
        import subprocess
        import sys

        # Use tf2onnx CLI with --graphdef flag
        cmd = [
            sys.executable,
            "-m",
            "tf2onnx.convert",
            "--graphdef",
            str(pb_path),
            "--inputs",
            inputs,
            "--outputs",
            outputs,
            "--output",
            str(onnx_path),
            "--opset",
            str(opset_version),
        ]

        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        if result.returncode != 0:
            logger.error(f"tf2onnx conversion failed:\n{result.stderr}")
            if temp_file:
                try:
                    onnx_path.unlink()
                except Exception:
                    pass
            return None, None

        logger.info("Frozen graph converted successfully")

        # Verify the export
        import onnx

        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model validated successfully")

    except subprocess.TimeoutExpired:
        logger.error("tf2onnx conversion timed out after 10 minutes")
        if temp_file:
            try:
                onnx_path.unlink()
            except Exception:
                pass
        return None, None
    except Exception as e:
        logger.error(f"Frozen graph conversion failed: {e}")
        if temp_file:
            try:
                onnx_path.unlink()
            except Exception:
                pass
        return None, None

    return onnx_path, temp_file


def _convert_jax_to_onnx(
    jax_path: pathlib.Path,
    apply_fn_path: str | None,
    input_shape_str: str | None,
    output_path: pathlib.Path | None,
    opset_version: int,
    logger: logging.Logger,
) -> tuple[pathlib.Path | None, Any]:
    """
    Convert a JAX/Flax model to ONNX format.

    This is more complex than other conversions because JAX doesn't have a standard
    serialization format. The typical flow is:
    1. Load model params from file (.msgpack, .pkl, etc.)
    2. Import the apply function from user's code
    3. Convert JAX -> TF SavedModel -> ONNX

    Args:
        jax_path: Path to JAX params file (.msgpack, .pkl, .npy)
        apply_fn_path: Module:function path to the apply function
        input_shape_str: Input shape for tracing
        output_path: Where to save ONNX file (None = temp file)
        opset_version: ONNX opset version
        logger: Logger instance

    Returns:
        Tuple of (onnx_path, temp_file_handle_or_None)
    """
    # Check dependencies
    import importlib.util

    if importlib.util.find_spec("jax") is None:
        logger.error("JAX not installed. Install with: pip install jax jaxlib")
        return None, None

    if importlib.util.find_spec("tf2onnx") is None:
        logger.error("tf2onnx not installed. Install with: pip install tf2onnx tensorflow")
        return None, None

    jax_path = jax_path.resolve()
    if not jax_path.exists():
        logger.error(f"JAX params file not found: {jax_path}")
        return None, None

    if not apply_fn_path:
        logger.error(
            "--jax-apply-fn is required for JAX conversion.\n"
            "Example: --from-jax params.pkl --jax-apply-fn my_model:apply --input-shape 1,3,224,224"
        )
        return None, None

    if not input_shape_str:
        logger.error(
            "--input-shape is required for JAX conversion.\n"
            "Example: --from-jax params.pkl --jax-apply-fn my_model:apply --input-shape 1,3,224,224"
        )
        return None, None

    # Parse input shape
    try:
        input_shape = tuple(int(x.strip()) for x in input_shape_str.split(","))
    except ValueError:
        logger.error(
            f"Invalid --input-shape format: '{input_shape_str}'. "
            "Use comma-separated integers, e.g., '1,3,224,224'"
        )
        return None, None

    # Parse apply function path (module:function)
    if ":" not in apply_fn_path:
        logger.error(
            f"Invalid --jax-apply-fn format: '{apply_fn_path}'. "
            "Use module:function format, e.g., 'my_model:apply'"
        )
        return None, None

    module_path, fn_name = apply_fn_path.rsplit(":", 1)

    # Determine output path
    temp_file = None
    if output_path:
        onnx_path = output_path.resolve()
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        import tempfile

        temp_file = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        onnx_path = pathlib.Path(temp_file.name)
        temp_file.close()

    logger.info("Converting JAX model to ONNX...")
    logger.info(f"  Params: {jax_path}")
    logger.info(f"  Apply fn: {apply_fn_path}")
    logger.info(f"  Input shape: {input_shape}")
    logger.info(f"  Target: {onnx_path}")

    try:
        import importlib.util
        import sys

        # Load params
        suffix = jax_path.suffix.lower()
        if suffix == ".msgpack":
            try:
                from flax.serialization import msgpack_restore

                with open(jax_path, "rb") as f:
                    params = msgpack_restore(f.read())
                logger.info("Loaded Flax msgpack params")
            except ImportError:
                logger.error("Flax not installed. Install with: pip install flax")
                return None, None
        elif suffix in (".pkl", ".pickle"):
            import pickle

            with open(jax_path, "rb") as f:
                params = pickle.load(f)
            logger.info("Loaded pickle params")
        elif suffix == ".npy":
            import numpy as np

            params = np.load(jax_path, allow_pickle=True).item()
            logger.info("Loaded numpy params")
        else:
            logger.error(f"Unsupported params format: {suffix}. Use .msgpack, .pkl, or .npy")
            return None, None

        # Import apply function
        # Add current directory to path for local imports
        sys.path.insert(0, str(pathlib.Path.cwd()))

        try:
            module = importlib.import_module(module_path)
            apply_fn = getattr(module, fn_name)
            logger.info(f"Loaded apply function: {apply_fn_path}")
        except (ImportError, AttributeError) as e:
            logger.error(f"Could not import {apply_fn_path}: {e}")
            return None, None

        # Convert via jax2tf (JAX -> TF -> ONNX)
        try:
            import jax.numpy as jnp
            import tensorflow as tf
            from jax.experimental import jax2tf
        except ImportError:
            logger.error("jax2tf or TensorFlow not available. Install with: pip install tensorflow")
            return None, None

        # Create a concrete function
        jnp.zeros(input_shape, dtype=jnp.float32)

        # Convert JAX function to TF
        tf_fn = jax2tf.convert(
            lambda x: apply_fn(params, x),
            enable_xla=False,
        )

        # Create TF SavedModel
        import tempfile as tf_tempfile

        with tf_tempfile.TemporaryDirectory() as tf_model_dir:
            # Wrap in tf.Module for SavedModel export
            class TFModule(tf.Module):
                @tf.function(input_signature=[tf.TensorSpec(input_shape, tf.float32)])
                def __call__(self, x):
                    return tf_fn(x)

            tf_module = TFModule()
            tf.saved_model.save(tf_module, tf_model_dir)
            logger.info("Created temporary TF SavedModel")

            # Convert TF SavedModel to ONNX
            import subprocess

            cmd = [
                sys.executable,
                "-m",
                "tf2onnx.convert",
                "--saved-model",
                tf_model_dir,
                "--output",
                str(onnx_path),
                "--opset",
                str(opset_version),
            ]

            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                logger.error(f"tf2onnx conversion failed:\n{result.stderr}")
                if temp_file:
                    try:
                        onnx_path.unlink()
                    except Exception:
                        pass
                return None, None

        logger.info("JAX model converted successfully")

        # Verify the export
        import onnx

        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model validated successfully")

    except Exception as e:
        logger.error(f"JAX conversion failed: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        if temp_file:
            try:
                onnx_path.unlink()
            except Exception:
                pass
        return None, None

    return onnx_path, temp_file


def _generate_quant_report_markdown(result: QuantizationLintResult, model_name: str) -> str:
    """Generate a Markdown report for quantization linting results."""
    from .quantization_linter import Severity

    lines = [
        f"# Quantization Readiness Report: {model_name}",
        "",
        "## Summary",
        "",
        f"**Readiness Score:** {result.readiness_score}/100",
        f"**Quant-Friendly Ops:** {result.quant_friendly_pct:.1f}%",
        f"**Critical Issues:** {result.critical_count}",
        f"**High Severity Issues:** {result.high_count}",
        "",
    ]

    if result.is_already_quantized:
        lines.extend(
            [
                "> **Note:** This model already contains quantization ops.",
                "",
            ]
        )

    # Warnings by severity
    if result.warnings:
        lines.extend(["## Issues", ""])
        for severity in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]:
            severity_warnings = [w for w in result.warnings if w.severity == severity]
            if severity_warnings:
                lines.append(f"### {severity.value.title()} ({len(severity_warnings)})")
                lines.append("")
                for w in severity_warnings:
                    lines.append(f"- **{w.message}**")
                    if w.recommendation:
                        lines.append(f"  - *Recommendation:* {w.recommendation}")
                lines.append("")

    # Op breakdown
    lines.extend(["## Op Analysis", ""])

    if result.unsupported_ops:
        lines.append("### Ops Without INT8 Kernel")
        lines.append("")
        lines.append("| Op Type | Count |")
        lines.append("|---------|-------|")
        for op, count in sorted(result.unsupported_ops.items(), key=lambda x: -x[1]):
            lines.append(f"| {op} | {count} |")
        lines.append("")

    if result.accuracy_sensitive_ops:
        lines.append("### Accuracy-Sensitive Ops")
        lines.append("")
        lines.append("| Op Type | Count |")
        lines.append("|---------|-------|")
        for op, count in sorted(result.accuracy_sensitive_ops.items(), key=lambda x: -x[1]):
            lines.append(f"| {op} | {count} |")
        lines.append("")

    if result.quant_friendly_ops:
        lines.append("### Quant-Friendly Ops")
        lines.append("")
        lines.append("| Op Type | Count |")
        lines.append("|---------|-------|")
        for op, count in sorted(result.quant_friendly_ops.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"| {op} | {count} |")
        if len(result.quant_friendly_ops) > 10:
            lines.append(f"| ... | ({len(result.quant_friendly_ops) - 10} more) |")
        lines.append("")

    # Problem layers
    if result.problem_layers:
        lines.extend(["## Problem Layers", ""])
        lines.append("| Layer Name | Op Type | Reason | Recommendation |")
        lines.append("|------------|---------|--------|----------------|")
        for layer in result.problem_layers[:20]:
            lines.append(
                f"| {layer['name'][:30]} | {layer['op_type']} | "
                f"{layer['reason']} | {layer['recommendation']} |"
            )
        if len(result.problem_layers) > 20:
            lines.append(f"| ... | | ({len(result.problem_layers) - 20} more) | |")
        lines.append("")

    return "\n".join(lines)


def _handle_tensorrt_analysis(
    args: argparse.Namespace, model_path: pathlib.Path, logger: logging.Logger
) -> None:
    """Handle TensorRT engine file analysis.

    TensorRT engines require special handling since they're compiled,
    optimized models without the same structure as ONNX.
    """
    try:
        from .formats.tensorrt import TRTEngineReader, format_bytes, is_available
    except ImportError:
        logger.error(
            "TensorRT support not installed.\n"
            "Install with: pip install haoline[tensorrt]\n"
            "Note: Requires NVIDIA GPU and CUDA 12.x"
        )
        return

    if not is_available():
        logger.error(
            "TensorRT not available. Install with: pip install tensorrt\n"
            "Note: Requires NVIDIA GPU and CUDA 12.x"
        )
        return

    print(f"\n{'=' * 60}")
    print("TensorRT Engine Analysis")
    print(f"{'=' * 60}\n")

    try:
        reader = TRTEngineReader(model_path)
        info = reader.read()
    except RuntimeError as e:
        logger.error(f"Failed to load TensorRT engine: {e}")
        return

    # Engine overview
    print(f"File: {model_path.name}")
    print(f"TensorRT Version: {info.trt_version}")
    print(f"Device: {info.device_name}")
    print(f"Compute Capability: SM {info.compute_capability[0]}.{info.compute_capability[1]}")
    print(f"Device Memory: {format_bytes(info.device_memory_bytes)}")
    print()

    # Builder configuration
    cfg = info.builder_config
    print("Builder Configuration:")
    print(f"  Max Batch Size: {cfg.max_batch_size}")
    print(f"  Workspace Size: {format_bytes(cfg.device_memory_size)}")
    if cfg.num_optimization_profiles > 0:
        print(f"  Optimization Profiles: {cfg.num_optimization_profiles}")
    if cfg.dla_core >= 0:
        print(f"  DLA Core: {cfg.dla_core}")
    else:
        print("  DLA: Not used (GPU only)")
    if cfg.engine_capability != "Standard":
        print(f"  Engine Capability: {cfg.engine_capability}")
    if cfg.has_implicit_batch:
        print("  Mode: Implicit Batch (legacy)")
    print()

    # Bindings
    print("Input/Output Bindings:")
    for b in info.input_bindings:
        print(f"  [Input]  {b.name}: {b.shape} ({b.dtype})")
    for b in info.output_bindings:
        print(f"  [Output] {b.name}: {b.shape} ({b.dtype})")
    print()

    # Layer summary
    print(f"Layer Count: {info.layer_count}")
    if info.fused_layer_count > 0:
        fused_pct = int(info.fusion_ratio * 100)
        print(f"Fused Layers: {info.fused_layer_count}/{info.layer_count} ({fused_pct}%)")
        print(f"Original Ops Fused: ~{info.original_ops_fused} → {info.fused_layer_count} kernels")
    print()

    # Layer type distribution
    print("Layer Type Distribution:")
    for ltype, count in sorted(info.layer_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {ltype}: {count}")
    print()

    # Precision breakdown
    if info.precision_breakdown:
        print("Precision Breakdown:")
        for prec, count in info.precision_breakdown.items():
            print(f"  {prec}: {count}")
        print()

    # Performance Metadata (Task 22.4 - TRT Performance Metadata Panel)
    perf = info.performance
    print("Performance Analysis:")

    # Task 22.4.2: Workspace allocation
    if perf.total_workspace_bytes > 0:
        print(f"  Total Workspace: {format_bytes(perf.total_workspace_bytes)}")

    # Task 22.4.4: Compute vs memory bound distribution
    total_classified = perf.compute_bound_layers + perf.memory_bound_layers + perf.balanced_layers
    if total_classified > 0:
        print("  Layer Classification:")
        if perf.compute_bound_layers > 0:
            print(f"    Compute-bound: {perf.compute_bound_layers}")
        if perf.memory_bound_layers > 0:
            print(f"    Memory-bound: {perf.memory_bound_layers}")
        if perf.balanced_layers > 0:
            print(f"    Balanced: {perf.balanced_layers}")
        if perf.unknown_bound_layers > 0:
            print(f"    Unknown: {perf.unknown_bound_layers}")

    # Task 22.4.1: Per-layer timing (if available from profiling)
    if perf.has_layer_timing and perf.total_time_ms is not None:
        print(f"\n  Total Inference Time: {perf.total_time_ms:.2f} ms")
        if perf.slowest_layers:
            print("  Slowest Layers:")
            for name, time_ms in perf.slowest_layers[:5]:
                pct = (time_ms / perf.total_time_ms) * 100 if perf.total_time_ms > 0 else 0
                print(f"    {name[:40]}: {time_ms:.3f} ms ({pct:.1f}%)")
    else:
        print("  Note: Run with profiling enabled for per-layer timing")

    # Task 22.4.3: Tactic/kernel selection (show sample)
    layers_with_tactics = [lyr for lyr in info.layers if lyr.tactic_name]
    if layers_with_tactics:
        print(f"\n  Kernel/Tactic Selection: ({len(layers_with_tactics)} layers have tactics)")
        for lyr in layers_with_tactics[:3]:
            tactic_short = (lyr.tactic_name or "")[:50]
            print(f"    {lyr.name[:30]}: {tactic_short}")
        if len(layers_with_tactics) > 3:
            print(f"    ... and {len(layers_with_tactics) - 3} more")
    print()

    # Quantization bottleneck analysis (--quant-bottlenecks or always show summary)
    show_bottlenecks = getattr(args, "quant_bottlenecks", False)
    from .formats.tensorrt import analyze_quant_bottlenecks

    bottleneck_analysis = analyze_quant_bottlenecks(info)

    # Always show summary, detailed analysis with --quant-bottlenecks
    print("=" * 40)
    print("Quantization Fusion Summary")
    print("=" * 40)
    print(
        f"  INT8 layers: {bottleneck_analysis.int8_layer_count} ({int(bottleneck_analysis.quantization_ratio * 100)}%)"
    )
    print(f"  FP16 layers: {bottleneck_analysis.fp16_layer_count}")
    print(
        f"  FP32 fallback: {bottleneck_analysis.fp32_layer_count} ({int(bottleneck_analysis.fp32_fallback_ratio * 100)}%)"
    )

    if bottleneck_analysis.largest_bottleneck:
        lb = bottleneck_analysis.largest_bottleneck
        print(f"  Largest bottleneck: {lb.layer_count} consecutive FP32 layers ({lb.severity})")

    if bottleneck_analysis.estimated_speedup_potential > 1.05:
        print(
            f"  Estimated speedup potential: {bottleneck_analysis.estimated_speedup_potential:.1f}x"
        )

    print()

    # Detailed analysis with --quant-bottlenecks
    if show_bottlenecks:
        if bottleneck_analysis.failed_fusions:
            print("Failed Fusions (should have fused but didn't):")
            for ff in bottleneck_analysis.failed_fusions[:5]:  # Show top 5
                print(f"  [{ff.speed_impact}] {ff.pattern_type}: {', '.join(ff.layer_names[:3])}")
            if len(bottleneck_analysis.failed_fusions) > 5:
                print(f"  ... and {len(bottleneck_analysis.failed_fusions) - 5} more")
            print()

        if bottleneck_analysis.bottleneck_zones:
            print("FP32 Bottleneck Zones:")
            for zone in sorted(bottleneck_analysis.bottleneck_zones, key=lambda z: -z.layer_count)[
                :5
            ]:
                types_str = ", ".join(set(zone.layer_types[:3]))
                print(f"  [{zone.severity}] {zone.layer_count} layers: {types_str}")
            if len(bottleneck_analysis.bottleneck_zones) > 5:
                print(f"  ... and {len(bottleneck_analysis.bottleneck_zones) - 5} more zones")
            print()

        if bottleneck_analysis.recommendations:
            print("Recommendations:")
            for rec in bottleneck_analysis.recommendations:
                print(f"  - {rec}")
            print()

    # JSON output if requested
    if args.out_json:
        import json

        cfg = info.builder_config
        output_data = {
            "format": "tensorrt",
            "path": str(model_path),
            "trt_version": info.trt_version,
            "device": {
                "name": info.device_name,
                "compute_capability": list(info.compute_capability),
            },
            "device_memory_bytes": info.device_memory_bytes,
            "builder_config": {
                "max_batch_size": cfg.max_batch_size,
                "workspace_bytes": cfg.device_memory_size,
                "num_optimization_profiles": cfg.num_optimization_profiles,
                "dla_core": cfg.dla_core,
                "engine_capability": cfg.engine_capability,
                "has_implicit_batch": cfg.has_implicit_batch,
            },
            "layer_count": info.layer_count,
            "fused_layer_count": info.fused_layer_count,
            "fusion_ratio": info.fusion_ratio,
            "original_ops_fused": info.original_ops_fused,
            "layer_type_counts": info.layer_type_counts,
            "precision_breakdown": info.precision_breakdown,
            "bindings": [
                {
                    "name": b.name,
                    "shape": list(b.shape),
                    "dtype": b.dtype,
                    "is_input": b.is_input,
                }
                for b in info.bindings
            ],
            "layers": [
                {
                    "name": layer.name,
                    "type": layer.type,
                    "precision": layer.precision,
                    "is_fused": layer.is_fused,
                    "fused_ops": layer.fused_ops,
                    "tactic": layer.tactic,
                    "tactic_name": layer.tactic_name,
                    "workspace_size_bytes": layer.workspace_size_bytes,
                    "output_memory_bytes": layer.output_memory_bytes,
                    "bound_type": layer.bound_type,
                }
                for layer in info.layers
            ],
            # Task 22.4: Performance metadata
            "performance": {
                "total_time_ms": perf.total_time_ms,
                "has_layer_timing": perf.has_layer_timing,
                "slowest_layers": [
                    {"name": name, "time_ms": time} for name, time in perf.slowest_layers
                ],
                "total_workspace_bytes": perf.total_workspace_bytes,
                "compute_bound_layers": perf.compute_bound_layers,
                "memory_bound_layers": perf.memory_bound_layers,
                "balanced_layers": perf.balanced_layers,
                "unknown_bound_layers": perf.unknown_bound_layers,
            },
            "quant_bottleneck_analysis": {
                "int8_layer_count": bottleneck_analysis.int8_layer_count,
                "fp16_layer_count": bottleneck_analysis.fp16_layer_count,
                "fp32_layer_count": bottleneck_analysis.fp32_layer_count,
                "quantization_ratio": bottleneck_analysis.quantization_ratio,
                "fp32_fallback_ratio": bottleneck_analysis.fp32_fallback_ratio,
                "estimated_speedup_potential": bottleneck_analysis.estimated_speedup_potential,
                "failed_fusion_count": len(bottleneck_analysis.failed_fusions),
                "bottleneck_zone_count": len(bottleneck_analysis.bottleneck_zones),
                "largest_bottleneck_size": bottleneck_analysis.largest_bottleneck.layer_count
                if bottleneck_analysis.largest_bottleneck
                else 0,
                "recommendations": bottleneck_analysis.recommendations,
            },
        }
        with open(args.out_json, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"JSON report written to: {args.out_json}")

    # Markdown output if requested
    if args.out_md:
        cfg = info.builder_config
        dla_str = f"Core {cfg.dla_core}" if cfg.dla_core >= 0 else "Not used (GPU)"
        md_lines = [
            f"# TensorRT Engine Analysis: {model_path.name}",
            "",
            "## Engine Overview",
            "",
            "| Property | Value |",
            "|----------|-------|",
            f"| TensorRT Version | {info.trt_version} |",
            f"| Device | {info.device_name} |",
            f"| Compute Capability | SM {info.compute_capability[0]}.{info.compute_capability[1]} |",
            f"| Device Memory | {format_bytes(info.device_memory_bytes)} |",
            f"| Layer Count | {info.layer_count} |",
            f"| Fused Layers | {info.fused_layer_count} ({int(info.fusion_ratio * 100)}%) |",
            f"| Original Ops Fused | ~{info.original_ops_fused} ops → {info.fused_layer_count} kernels |",
            "",
            "## Builder Configuration",
            "",
            "| Setting | Value |",
            "|---------|-------|",
            f"| Max Batch Size | {cfg.max_batch_size} |",
            f"| Workspace Size | {format_bytes(cfg.device_memory_size)} |",
            f"| Optimization Profiles | {cfg.num_optimization_profiles} |",
            f"| DLA | {dla_str} |",
            f"| Engine Capability | {cfg.engine_capability} |",
            "",
            "## Bindings",
            "",
            "| Name | Type | Shape | Dtype |",
            "|------|------|-------|-------|",
        ]
        for b in info.bindings:
            io_type = "Input" if b.is_input else "Output"
            md_lines.append(f"| {b.name} | {io_type} | {b.shape} | {b.dtype} |")

        md_lines.extend(
            [
                "",
                "## Layer Type Distribution",
                "",
                "| Type | Count |",
                "|------|-------|",
            ]
        )
        for ltype, count in sorted(info.layer_type_counts.items(), key=lambda x: -x[1]):
            md_lines.append(f"| {ltype} | {count} |")

        with open(args.out_md, "w") as f:
            f.write("\n".join(md_lines))
        print(f"Markdown report written to: {args.out_md}")

    print("\n[TensorRT analysis complete]")


def _handle_trt_comparison(
    args: argparse.Namespace,
    onnx_path: pathlib.Path,
    trt_path: pathlib.Path,
    logger: logging.Logger,
) -> None:
    """Handle ONNX vs TensorRT engine comparison."""
    try:
        from .formats.trt_comparison import compare_onnx_trt
    except ImportError as e:
        logger.error(f"Failed to import TRT comparison module: {e}")
        return

    print(f"\n{'=' * 60}")
    print("ONNX ↔ TensorRT Comparison")
    print(f"{'=' * 60}\n")
    print(f"ONNX:  {onnx_path.name}")
    print(f"TRT:   {trt_path.name}")
    print()

    try:
        report = compare_onnx_trt(onnx_path, trt_path)
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        return

    # Summary
    compression = report.onnx_node_count / max(report.trt_layer_count, 1)
    print(f"ONNX Nodes:      {report.onnx_node_count}")
    print(f"TRT Layers:      {report.trt_layer_count}")
    print(f"Compression:     {compression:.1f}x")
    print()

    print(f"Fusions:         {report.fusion_count}")
    print(f"Removed Nodes:   {report.removed_node_count}")
    print(f"Precision Changes: {len(report.precision_changes)}")
    print()

    # Memory metrics
    mm = report.memory_metrics
    if mm.onnx_file_size_bytes > 0:
        print("Memory Comparison:")
        print(f"  ONNX file:      {mm.onnx_file_size_bytes / 1024 / 1024:.1f} MB")
        print(f"  TRT engine:     {mm.trt_engine_size_bytes / 1024 / 1024:.1f} MB")
        print(f"  File ratio:     {mm.file_size_ratio:.2f}x")
        if mm.trt_device_memory_bytes > 0:
            print(f"  Device memory:  {mm.trt_device_memory_bytes / 1024 / 1024:.1f} MB")
        if mm.estimated_precision_savings_ratio > 0:
            print(f"  Precision savings: ~{mm.estimated_precision_savings_ratio * 100:.0f}%")
        print()

    # Top fusions
    if report.layer_mappings:
        fusions = [m for m in report.layer_mappings if m.is_fusion]
        if fusions:
            print("Top Fusions (N ONNX ops → 1 TRT kernel):")
            for fusion in fusions[:10]:
                ops = fusion.fusion_description or f"{len(fusion.onnx_nodes)} ops"
                print(f"  {fusion.trt_layer_name[:50]}")
                print(f"    → {ops}")
            if len(fusions) > 10:
                print(f"  ... and {len(fusions) - 10} more fusions")
            print()

    # Precision changes
    if report.precision_changes:
        print("Precision Changes:")
        prec_summary: dict[str, int] = {}
        for change in report.precision_changes:
            prec_summary[change.trt_precision] = prec_summary.get(change.trt_precision, 0) + 1
        for prec, count in sorted(prec_summary.items(), key=lambda x: -x[1]):
            print(f"  {prec}: {count} layers")
        print()

    # Layer rewrites (attention → Flash Attention, etc.)
    if hasattr(report, "layer_rewrites") and report.layer_rewrites:
        print("Layer Rewrites (optimized kernel substitutions):")
        for rewrite in report.layer_rewrites[:8]:
            print(f"  {rewrite.rewrite_type}: {rewrite.description}")
            if rewrite.original_ops:
                ops_str = ", ".join(rewrite.original_ops[:3])
                if len(rewrite.original_ops) > 3:
                    ops_str += f" (+{len(rewrite.original_ops) - 3} more)"
                print(f"    ONNX ops: {ops_str}")
            if rewrite.speedup_estimate:
                print(f"    Speedup: {rewrite.speedup_estimate}")
        if len(report.layer_rewrites) > 8:
            print(f"  ... and {len(report.layer_rewrites) - 8} more rewrites")
        print()

    # Removed nodes (optimized away)
    if report.removed_nodes:
        print(f"Removed/Optimized Nodes ({len(report.removed_nodes)}):")
        for node in report.removed_nodes[:10]:
            print(f"  • {node}")
        if len(report.removed_nodes) > 10:
            print(f"  ... and {len(report.removed_nodes) - 10} more")
        print()

    # Unmapped warnings
    if report.unmapped_onnx_nodes:
        print(f"⚠ Unmapped ONNX Nodes: {len(report.unmapped_onnx_nodes)}")
        for node in report.unmapped_onnx_nodes[:5]:
            print(f"    {node}")
        if len(report.unmapped_onnx_nodes) > 5:
            print(f"    ... and {len(report.unmapped_onnx_nodes) - 5} more")
        print()

    # JSON output
    if args.out_json:
        import json

        mm = report.memory_metrics
        output_data = {
            "comparison": "onnx_vs_trt",
            "onnx_path": str(onnx_path),
            "trt_path": str(trt_path),
            "onnx_node_count": report.onnx_node_count,
            "trt_layer_count": report.trt_layer_count,
            "compression_ratio": compression,
            "fusion_count": report.fusion_count,
            "removed_node_count": report.removed_node_count,
            "memory": {
                "onnx_file_bytes": mm.onnx_file_size_bytes,
                "trt_engine_bytes": mm.trt_engine_size_bytes,
                "trt_device_memory_bytes": mm.trt_device_memory_bytes,
                "file_size_ratio": mm.file_size_ratio,
                "precision_savings_ratio": mm.estimated_precision_savings_ratio,
            },
            "precision_changes": [
                {
                    "layer": c.layer_name,
                    "from": c.original_precision,
                    "to": c.trt_precision,
                }
                for c in report.precision_changes
            ],
            "fusions": [
                {
                    "trt_layer": m.trt_layer_name,
                    "onnx_nodes": m.onnx_nodes,
                    "description": m.fusion_description,
                }
                for m in report.layer_mappings
                if m.is_fusion
            ],
            "layer_rewrites": [
                {
                    "type": r.rewrite_type,
                    "original_ops": r.original_ops,
                    "original_op_types": r.original_op_types,
                    "trt_layer": r.trt_layer_name,
                    "kernel": r.trt_kernel,
                    "description": r.description,
                    "speedup_estimate": r.speedup_estimate,
                }
                for r in (report.layer_rewrites if hasattr(report, "layer_rewrites") else [])
            ],
            "removed_nodes": report.removed_nodes,
        }
        with open(args.out_json, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"JSON report written to: {args.out_json}")

    # Markdown output
    if args.out_md:
        mm = report.memory_metrics
        md_lines = [
            "# ONNX ↔ TensorRT Comparison",
            "",
            f"**ONNX:** `{onnx_path.name}`",
            f"**TRT:** `{trt_path.name}`",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| ONNX Nodes | {report.onnx_node_count} |",
            f"| TRT Layers | {report.trt_layer_count} |",
            f"| Compression | {compression:.1f}x |",
            f"| Fusions | {report.fusion_count} |",
            f"| Removed Nodes | {report.removed_node_count} |",
            f"| Precision Changes | {len(report.precision_changes)} |",
            "",
            "## Memory Comparison",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| ONNX File | {mm.onnx_file_size_bytes / 1024 / 1024:.1f} MB |",
            f"| TRT Engine | {mm.trt_engine_size_bytes / 1024 / 1024:.1f} MB |",
            f"| File Ratio | {mm.file_size_ratio:.2f}x |",
            f"| Device Memory | {mm.trt_device_memory_bytes / 1024 / 1024:.1f} MB |",
            "",
        ]

        if report.fusion_count > 0:
            md_lines.extend(
                [
                    "## Top Fusions",
                    "",
                    "| TRT Layer | Fused ONNX Ops |",
                    "|-----------|----------------|",
                ]
            )
            fusions = [m for m in report.layer_mappings if m.is_fusion][:15]
            for fusion in fusions:
                desc = fusion.fusion_description or f"{len(fusion.onnx_nodes)} ops"
                md_lines.append(f"| {fusion.trt_layer_name[:40]} | {desc} |")
            md_lines.append("")

        # Layer rewrites section
        if hasattr(report, "layer_rewrites") and report.layer_rewrites:
            md_lines.extend(
                [
                    "## Layer Rewrites",
                    "",
                    "TensorRT replaced these ONNX patterns with optimized kernels:",
                    "",
                    "| Rewrite Type | Description | Speedup |",
                    "|--------------|-------------|---------|",
                ]
            )
            for rewrite in report.layer_rewrites[:10]:
                md_lines.append(
                    f"| {rewrite.rewrite_type} | {rewrite.description} | {rewrite.speedup_estimate} |"
                )
            if len(report.layer_rewrites) > 10:
                md_lines.append(f"| ... | {len(report.layer_rewrites) - 10} more | |")
            md_lines.append("")

        if report.removed_nodes:
            md_lines.extend(
                [
                    "## Removed/Optimized Nodes",
                    "",
                    "These ONNX nodes were optimized away by TensorRT:",
                    "",
                ]
            )
            for node in report.removed_nodes[:20]:
                md_lines.append(f"- `{node}`")
            if len(report.removed_nodes) > 20:
                md_lines.append(f"- ... and {len(report.removed_nodes) - 20} more")
            md_lines.append("")

        with open(args.out_md, "w") as f:
            f.write("\n".join(md_lines))
        print(f"Markdown report written to: {args.out_md}")

    # HTML output with side-by-side comparison (Task 22.3.6)
    if args.out_html:
        try:
            from .formats.trt_comparison import generate_comparison_html

            html_content = generate_comparison_html(report)
            with open(args.out_html, "w", encoding="utf-8") as f:
                f.write(html_content)
            print(f"HTML comparison report written to: {args.out_html}")
        except Exception as e:
            logger.warning(f"Could not generate HTML comparison: {e}")

    print("\n[ONNX ↔ TRT comparison complete]")


def run_inspect() -> None:
    """Main entry point for the model_inspect CLI."""
    # Load environment variables from .env file if present
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass  # python-dotenv not installed, use environment variables directly

    args = parse_args()
    logger = setup_logging(args.log_level)

    # Handle --schema
    if args.schema:
        import json

        from .schema import get_schema

        schema = get_schema()
        print(json.dumps(schema, indent=2))
        return

    # Handle --list-hardware
    if args.list_hardware:
        print("\n" + "=" * 70)
        print("Available Hardware Profiles")
        print("=" * 70)

        print("\nData Center GPUs - H100 Series:")
        for name in ["h100-sxm", "h100-pcie", "h100-nvl"]:
            profile = get_profile(name)
            if profile:
                print(
                    f"  {name:20} {profile.name:30} {profile.vram_bytes // (1024**3):3} GB  {profile.peak_fp16_tflops:6.1f} TF16"
                )

        print("\nData Center GPUs - A100 Series:")
        for name in [
            "a100-80gb-sxm",
            "a100-80gb-pcie",
            "a100-40gb-sxm",
            "a100-40gb-pcie",
        ]:
            profile = get_profile(name)
            if profile:
                print(
                    f"  {name:20} {profile.name:30} {profile.vram_bytes // (1024**3):3} GB  {profile.peak_fp16_tflops:6.1f} TF16"
                )

        print("\nData Center GPUs - Other:")
        for name in ["a10", "l4", "l40", "l40s", "t4"]:
            profile = get_profile(name)
            if profile:
                print(
                    f"  {name:20} {profile.name:30} {profile.vram_bytes // (1024**3):3} GB  {profile.peak_fp16_tflops:6.1f} TF16"
                )

        print("\nData Center GPUs - V100 Series:")
        for name in [
            "v100-32gb-sxm",
            "v100-32gb-pcie",
            "v100-16gb-sxm",
            "v100-16gb-pcie",
        ]:
            profile = get_profile(name)
            if profile:
                print(
                    f"  {name:20} {profile.name:30} {profile.vram_bytes // (1024**3):3} GB  {profile.peak_fp16_tflops:6.1f} TF16"
                )

        print("\nDGX Systems (Multi-GPU):")
        for name in ["dgx-h100", "dgx-a100-640gb", "dgx-a100-320gb"]:
            profile = get_profile(name)
            if profile:
                print(
                    f"  {name:20} {profile.name:30} {profile.vram_bytes // (1024**3):3} GB  {profile.peak_fp16_tflops:6.1f} TF16"
                )

        print("\nJetson Edge/Embedded (Orin Series - Recommended for new projects):")
        for name in [
            "jetson-agx-orin-64gb",
            "jetson-agx-orin-32gb",
            "jetson-orin-nx-16gb",
            "jetson-orin-nx-8gb",
            "jetson-orin-nano-8gb",
            "jetson-orin-nano-4gb",
        ]:
            profile = get_profile(name)
            if profile:
                print(
                    f"  {name:20} {profile.name:30} {profile.vram_bytes // (1024**3):3} GB  {profile.peak_fp16_tflops:6.1f} TF16"
                )

        print("\nJetson Edge/Embedded (Xavier Series):")
        for name in ["jetson-agx-xavier", "jetson-xavier-nx-8gb"]:
            profile = get_profile(name)
            if profile:
                print(
                    f"  {name:20} {profile.name:30} {profile.vram_bytes // (1024**3):3} GB  {profile.peak_fp16_tflops:6.1f} TF16"
                )

        print("\nJetson Edge/Embedded (Legacy - Very Constrained!):")
        for name in ["jetson-tx2", "jetson-nano", "jetson-nano-2gb"]:
            profile = get_profile(name)
            if profile:
                vram_gb = profile.vram_bytes / (1024**3)
                print(
                    f"  {name:20} {profile.name:30} {vram_gb:3.0f} GB  {profile.peak_fp16_tflops:6.3f} TF16"
                )

        print("\nConsumer GPUs - RTX 40 Series:")
        for name in [
            "rtx4090",
            "4080-super",
            "rtx4080",
            "4070-ti-super",
            "4070-ti",
            "4070-super",
            "rtx4070",
            "4060-ti-16gb",
            "rtx4060",
        ]:
            profile = get_profile(name)
            if profile:
                print(
                    f"  {name:20} {profile.name:30} {profile.vram_bytes // (1024**3):3} GB  {profile.peak_fp16_tflops:6.1f} TF16"
                )

        print("\nConsumer GPUs - RTX 30 Series:")
        for name in [
            "3090-ti",
            "rtx3090",
            "3080-ti",
            "3080-12gb",
            "rtx3080",
            "3070-ti",
            "rtx3070",
            "3060-ti",
            "rtx3060",
            "rtx3050",
        ]:
            profile = get_profile(name)
            if profile:
                print(
                    f"  {name:20} {profile.name:30} {profile.vram_bytes // (1024**3):3} GB  {profile.peak_fp16_tflops:6.1f} TF16"
                )

        print("\nLaptop/Mobile GPUs:")
        for name in [
            "4090-mobile",
            "4080-mobile",
            "4070-mobile",
            "3080-mobile",
            "3070-mobile",
        ]:
            profile = get_profile(name)
            if profile:
                print(
                    f"  {name:20} {profile.name:30} {profile.vram_bytes // (1024**3):3} GB  {profile.peak_fp16_tflops:6.1f} TF16"
                )

        print("\nOther:")
        print("  auto                 Auto-detect local GPU/CPU")
        print("  cpu                  Generic CPU profile")

        print("\n" + "-" * 70)
        print("TF16 = Peak FP16 TFLOPS (higher = faster)")
        print("Use --gpu-count N for multi-GPU estimates")
        print("Use --list-cloud for cloud instance options")
        print("=" * 70 + "\n")
        sys.exit(0)

    # Handle --list-cloud
    if args.list_cloud:
        print("\n" + "=" * 70)
        print("Available Cloud Instance Profiles")
        print("=" * 70)

        print("\nAWS GPU Instances:")
        for name, instance in CLOUD_INSTANCES.items():
            if instance.provider == "aws":
                vram_gb = instance.hardware.vram_bytes * instance.gpu_count // (1024**3)
                print(
                    f"  {name:25} {instance.gpu_count}x GPU  {vram_gb:4} GB  ${instance.hourly_cost_usd:6.2f}/hr"
                )

        print("\nAzure GPU Instances:")
        for name, instance in CLOUD_INSTANCES.items():
            if instance.provider == "azure":
                vram_gb = instance.hardware.vram_bytes * instance.gpu_count // (1024**3)
                print(
                    f"  {name:25} {instance.gpu_count}x GPU  {vram_gb:4} GB  ${instance.hourly_cost_usd:6.2f}/hr"
                )

        print("\nGCP GPU Instances:")
        for name, instance in CLOUD_INSTANCES.items():
            if instance.provider == "gcp":
                vram_gb = instance.hardware.vram_bytes * instance.gpu_count // (1024**3)
                print(
                    f"  {name:25} {instance.gpu_count}x GPU  {vram_gb:4} GB  ${instance.hourly_cost_usd:6.2f}/hr"
                )

        print("\n" + "-" * 70)
        print("Prices are approximate on-demand rates (us-east-1 or equivalent)")
        print("Use --cloud <instance> to get cost estimates for your model")
        print("=" * 70 + "\n")
        sys.exit(0)

    # Handle --list-formats
    if args.list_formats:
        print("\n" + "=" * 70)
        print("Supported Input Formats")
        print("=" * 70)

        print("\n[FULL ANALYSIS] - Graph structure + parameters")
        print("-" * 50)
        print("  ONNX          .onnx              Native support")
        print("  TensorRT      .engine, .plan     Requires NVIDIA GPU")

        print("\n[AUTO-CONVERT TO ONNX] - Converted then analyzed")
        print("-" * 50)
        print("  PyTorch       .pt, .pth          --from-pytorch + --input-shape")
        print("  TensorFlow    SavedModel/        --from-tensorflow")
        print("  Keras         .h5, .keras        --from-keras")
        print("  TFLite        .tflite            --from-tflite")
        print("  Frozen Graph  .pb                --from-frozen-graph + --tf-inputs/outputs")
        print("  JAX/Flax      .pkl, .msgpack     --from-jax + --jax-apply-fn + --input-shape")

        print("\n[WEIGHTS-ONLY] - Limited analysis (no graph)")
        print("-" * 50)
        print("  SafeTensors   .safetensors       Tensor shapes + dtypes only")
        print("  GGUF          .gguf              LLM quantization info")

        print("\n[CLI-ONLY READERS] - Requires optional dependencies")
        print("-" * 50)
        print("  TFLite        .tflite            pip install haoline  (header parsing)")
        print("  CoreML        .mlpackage         pip install haoline[coreml]")
        print("  OpenVINO      .xml + .bin        pip install haoline[openvino]")

        print("\n" + "-" * 70)
        print("Use --list-conversions to see format conversion paths")
        print("=" * 70 + "\n")
        sys.exit(0)

    # Handle --list-conversions
    if args.list_conversions:
        from .format_adapters import list_conversion_paths

        print("\n" + "=" * 70)
        print("Available Format Conversion Paths")
        print("=" * 70)

        paths = list_conversion_paths()
        current_source = None

        for path in paths:
            if path["source"] != current_source:
                current_source = path["source"]
                print(f"\nFrom {current_source.upper()}:")

            level_icons = {
                "full": "[FULL]    ",
                "partial": "[PARTIAL] ",
                "lossy": "[LOSSY]   ",
                "none": "[NONE]    ",
            }
            icon = level_icons.get(path["level"], "          ")
            print(f"  {icon} -> {path['target']}")

        print("\n" + "-" * 70)
        print("Conversion Levels:")
        print("  FULL    = Lossless, complete conversion")
        print("  PARTIAL = Some limitations or multi-step required")
        print("  LOSSY   = Information may be lost")
        print("  NONE    = No conversion path available")
        print("\nUse --convert-to FORMAT to convert a model")
        print("=" * 70 + "\n")
        sys.exit(0)

    # Handle model conversion if requested
    temp_onnx_file = None
    conversion_sources = [
        ("--from-pytorch", args.from_pytorch),
        ("--from-tensorflow", args.from_tensorflow),
        ("--from-keras", args.from_keras),
        ("--from-tflite", args.from_tflite),
        ("--from-frozen-graph", args.from_frozen_graph),
        ("--from-jax", args.from_jax),
    ]
    active_conversions = [(name, path) for name, path in conversion_sources if path]

    if len(active_conversions) > 1:
        names = [name for name, _ in active_conversions]
        logger.error(f"Cannot use multiple conversion flags together: {', '.join(names)}")
        sys.exit(1)

    if args.from_pytorch:
        model_path, temp_onnx_file = _convert_pytorch_to_onnx(
            args.from_pytorch,
            args.input_shape,
            args.keep_onnx,
            args.opset_version,
            logger,
        )
        if model_path is None:
            sys.exit(1)
    elif args.from_tensorflow:
        model_path, temp_onnx_file = _convert_tensorflow_to_onnx(
            args.from_tensorflow,
            args.keep_onnx,
            args.opset_version,
            logger,
        )
        if model_path is None:
            sys.exit(1)
    elif args.from_keras:
        model_path, temp_onnx_file = _convert_keras_to_onnx(
            args.from_keras,
            args.keep_onnx,
            args.opset_version,
            logger,
        )
        if model_path is None:
            sys.exit(1)
    elif args.from_tflite:
        model_path, temp_onnx_file = _convert_tflite_to_onnx(
            args.from_tflite,
            args.keep_onnx,
            args.opset_version,
            logger,
        )
        if model_path is None:
            sys.exit(1)
    elif args.from_frozen_graph:
        model_path, temp_onnx_file = _convert_frozen_graph_to_onnx(
            args.from_frozen_graph,
            args.tf_inputs,
            args.tf_outputs,
            args.keep_onnx,
            args.opset_version,
            logger,
        )
        if model_path is None:
            sys.exit(1)
    elif args.from_jax:
        model_path, temp_onnx_file = _convert_jax_to_onnx(
            args.from_jax,
            args.jax_apply_fn,
            args.input_shape,
            args.keep_onnx,
            args.opset_version,
            logger,
        )
        if model_path is None:
            sys.exit(1)
    else:
        # Validate model path (no conversion requested)
        if args.model_path is None:
            logger.error(
                "Model path is required. Use --list-hardware to see available profiles, "
                "or use a conversion flag (--from-pytorch, --from-tensorflow, --from-keras, "
                "--from-tflite, --from-frozen-graph, --from-jax)."
            )
            sys.exit(1)

        model_path = args.model_path.resolve()
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            sys.exit(1)

        # Handle --compare-trt: ONNX vs TensorRT comparison
        if args.compare_trt:
            if model_path.suffix.lower() != ".onnx":
                logger.error("--compare-trt requires an ONNX model as the primary input")
                sys.exit(1)
            trt_path = args.compare_trt.resolve()
            if not trt_path.exists():
                logger.error(f"TensorRT engine not found: {trt_path}")
                sys.exit(1)
            _handle_trt_comparison(args, model_path, trt_path, logger)
            sys.exit(0)

        # Check for TensorRT engine files - handle separately
        if model_path.suffix.lower() in (".engine", ".plan"):
            _handle_tensorrt_analysis(args, model_path, logger)
            sys.exit(0)

        if model_path.suffix.lower() not in (".onnx", ".pb", ".ort"):
            logger.warning(f"Unexpected file extension: {model_path.suffix}. Proceeding anyway.")

    # Handle --convert-to for format conversion
    if args.convert_to:
        from .format_adapters import (
            ConversionLevel,
            OnnxAdapter,
            get_conversion_level,
            load_model,
        )
        from .universal_ir import SourceFormat

        if not args.convert_output:
            logger.error("--convert-output is required when using --convert-to")
            sys.exit(1)

        target_format = SourceFormat(args.convert_to.lower())
        output_path = args.convert_output.resolve()

        # Determine source format from file extension
        source_ext = model_path.suffix.lower()
        source_format_map = {
            ".onnx": SourceFormat.ONNX,
            ".pt": SourceFormat.PYTORCH,
            ".pth": SourceFormat.PYTORCH,
        }
        source_format = source_format_map.get(source_ext, SourceFormat.UNKNOWN)

        # Check conversion level
        level = get_conversion_level(source_format, target_format)
        if level == ConversionLevel.NONE:
            logger.error(
                f"Cannot convert from {source_format.value} to {target_format.value}. "
                f"Use --list-conversions to see available paths."
            )
            sys.exit(1)
        elif level == ConversionLevel.LOSSY:
            logger.warning(
                f"Converting {source_format.value} to {target_format.value} may lose information"
            )

        # Load to Universal IR and convert
        try:
            logger.info(f"Loading {model_path} as Universal IR...")
            graph = load_model(model_path)

            if target_format == SourceFormat.ONNX:
                logger.info(f"Exporting to ONNX: {output_path}")
                OnnxAdapter().write(graph, output_path)
                logger.info(f"Successfully converted to {output_path}")
                print(f"\nConverted: {model_path} -> {output_path}")
                print(f"  Nodes: {graph.num_nodes}")
                print(f"  Parameters: {graph.total_parameters:,}")
                print(f"  Source: {graph.metadata.source_format.value}")
            elif target_format == SourceFormat.TFLITE:
                # ONNX -> TFLite conversion
                # First ensure we have an ONNX file to convert from
                if source_format != SourceFormat.ONNX:
                    # Need to convert to ONNX first
                    logger.info("Converting to ONNX intermediate format first...")
                    import tempfile

                    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
                        intermediate_onnx = pathlib.Path(tmp.name)
                    OnnxAdapter().write(graph, intermediate_onnx)
                    onnx_source = intermediate_onnx
                else:
                    onnx_source = model_path
                    intermediate_onnx = None

                # Convert ONNX to TFLite
                tflite_path, _ = _convert_onnx_to_tflite(
                    onnx_source,
                    output_path,
                    logger,
                )

                # Cleanup intermediate if created
                if intermediate_onnx:
                    intermediate_onnx.unlink(missing_ok=True)

                if tflite_path is None:
                    sys.exit(1)

                logger.info(f"Successfully converted to {output_path}")
                print(f"\nConverted: {model_path} -> {output_path}")
                print(f"  Original format: {source_format.value}")
                print("  Target format: tflite")
            else:
                logger.error(f"Conversion to {target_format.value} not yet implemented")
                sys.exit(1)

            # If no analysis requested, exit after conversion
            if not (args.out_json or args.out_md or args.out_html or args.out_pdf):
                sys.exit(0)

            # Use the converted file for analysis
            model_path = output_path

        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            sys.exit(1)

    # Handle --export-ir and --export-graph (Universal IR exports)
    if args.export_ir or args.export_graph:
        from .format_adapters import load_model

        try:
            logger.info(f"Loading {model_path} as Universal IR...")
            ir_graph = load_model(model_path)

            if args.export_ir:
                logger.info(f"Exporting Universal IR to {args.export_ir}")
                ir_graph.to_json(args.export_ir)
                print(f"Exported IR: {args.export_ir}")
                print(f"  Nodes: {ir_graph.num_nodes}")
                print(f"  Parameters: {ir_graph.total_parameters:,}")

            if args.export_graph:
                export_path = args.export_graph
                suffix = export_path.suffix.lower()

                if suffix == ".dot":
                    logger.info(f"Exporting graph to DOT: {export_path}")
                    ir_graph.save_dot(export_path)
                    print(f"Exported graph: {export_path}")
                elif suffix == ".png":
                    logger.info(f"Rendering graph to PNG: {export_path}")
                    ir_graph.save_png(export_path, max_nodes=args.graph_max_nodes)
                    print(f"Rendered graph: {export_path}")
                else:
                    logger.error(f"Unsupported graph format: {suffix}. Use .dot or .png")
                    sys.exit(1)

            # If no other outputs requested, exit after IR export
            if not (args.out_json or args.out_md or args.out_html or args.out_pdf):
                sys.exit(0)

        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"IR export failed: {e}")
            sys.exit(1)

    # Determine hardware profile
    hardware_profile = None
    cloud_instance = None

    if args.cloud:
        # Cloud instance takes precedence
        cloud_instance = get_cloud_instance(args.cloud)
        if cloud_instance is None:
            logger.error(f"Unknown cloud instance: {args.cloud}")
            logger.error("Use --list-cloud to see available instances.")
            sys.exit(1)
        hardware_profile = cloud_instance.hardware
        # Override gpu_count from cloud instance
        args.gpu_count = cloud_instance.gpu_count
        logger.info(
            f"Using cloud instance: {cloud_instance.name} "
            f"({cloud_instance.gpu_count}x GPU, ${cloud_instance.hourly_cost_usd:.2f}/hr)"
        )
    elif args.hardware:
        if args.hardware.lower() == "auto":
            logger.info("Auto-detecting local hardware...")
            hardware_profile = detect_local_hardware()
            logger.info(f"Detected: {hardware_profile.name}")
        else:
            hardware_profile = get_profile(args.hardware)
            if hardware_profile is None:
                logger.error(f"Unknown hardware profile: {args.hardware}")
                logger.error("Use --list-hardware to see available profiles.")
                sys.exit(1)
            logger.info(f"Using hardware profile: {hardware_profile.name}")

    # Apply multi-GPU scaling if requested
    if hardware_profile and args.gpu_count > 1 and not args.cloud:
        multi_gpu = create_multi_gpu_profile(args.hardware or "auto", args.gpu_count)
        if multi_gpu:
            hardware_profile = multi_gpu.get_effective_profile()
            logger.info(
                f"Multi-GPU: {args.gpu_count}x scaling with "
                f"{multi_gpu.compute_efficiency:.0%} efficiency"
            )

    # Setup progress indicator
    progress = ProgressIndicator(enabled=args.progress, quiet=args.quiet)

    # Calculate total steps based on what will be done
    total_steps = 2  # Load + Analyze always
    if hardware_profile:
        total_steps += 1
    if args.system_requirements:
        total_steps += 1
    if args.sweep_batch_sizes and hardware_profile:
        total_steps += 1
    if args.sweep_resolutions and hardware_profile:
        total_steps += 1
    if not args.no_gpu_metrics:
        total_steps += 1
    if not args.no_profile:
        total_steps += 1
    if not args.no_profile and not args.no_bottleneck_analysis and hardware_profile:
        total_steps += 1
    if not args.no_benchmark_resolutions:
        total_steps += 1
    if args.with_plots and is_viz_available():
        total_steps += 1
    if args.llm_summary and is_llm_available() and has_llm_api_key():
        total_steps += 1
    if args.out_json or args.out_md or args.out_html:
        total_steps += 1

    progress.start(total_steps, f"Analyzing {model_path.name}")

    # Check format capabilities and warn about limitations

    from .format_adapters import get_format_capabilities
    from .universal_ir import SourceFormat

    # Detect format from file extension
    file_ext = model_path.suffix.lower()
    detected_format: SourceFormat | None = None
    if file_ext == ".onnx":
        detected_format = SourceFormat.ONNX
    elif file_ext in (".pt", ".pth"):
        detected_format = SourceFormat.PYTORCH
    elif file_ext == ".safetensors":
        detected_format = SourceFormat.SAFETENSORS
    elif file_ext == ".gguf":
        detected_format = SourceFormat.GGUF
    elif file_ext == ".tflite":
        detected_format = SourceFormat.TFLITE
    elif file_ext in (".mlmodel", ".mlpackage"):
        detected_format = SourceFormat.COREML
    elif file_ext == ".xml":
        detected_format = SourceFormat.OPENVINO
    elif file_ext in (".engine", ".plan"):
        detected_format = SourceFormat.TENSORRT

    capabilities = get_format_capabilities(detected_format) if detected_format else None

    # Warn about format limitations
    if capabilities and detected_format and not capabilities.has_graph:
        logger.warning(f"Format {detected_format.value}: {capabilities.description}")
        logger.warning("Skipping graph visualization and FLOP estimation for weight-only formats")

    if capabilities and detected_format and not capabilities.has_flops:
        logger.warning(f"Format {detected_format.value}: FLOP estimation not available")
        logger.warning("Consider converting to ONNX for complete analysis")

    # Run inspection
    try:
        progress.step("Loading model and extracting graph structure")
        inspector = ModelInspector(logger=logger)
        report = inspector.inspect(model_path)
        progress.step("Computing metrics (params, FLOPs, memory)")

        # Add hardware estimates if profile specified
        if (
            hardware_profile
            and report.param_counts
            and report.flop_counts
            and report.memory_estimates
        ):
            progress.step(f"Estimating performance on {hardware_profile.name}")
            estimator = HardwareEstimator(logger=logger)
            hw_estimates = estimator.estimate(
                model_params=report.param_counts.total,
                model_flops=report.flop_counts.total,
                peak_activation_bytes=report.memory_estimates.peak_activation_bytes,
                hardware=hardware_profile,
                batch_size=args.batch_size,
                precision=args.precision,
            )
            report.hardware_estimates = hw_estimates
            report.hardware_profile = hardware_profile

            # Batch Size Sweep (Story 6C.1)
            if args.sweep_batch_sizes:
                profiler = OperationalProfiler(logger=logger)

                sweep_result: BatchSizeSweep | None = None

                if args.no_benchmark:
                    # Use theoretical estimates (faster but less accurate)
                    progress.step("Running batch size sweep (theoretical)")
                    sweep_result = profiler.run_batch_sweep(
                        model_params=report.param_counts.total,
                        model_flops=report.flop_counts.total,
                        peak_activation_bytes=report.memory_estimates.peak_activation_bytes,
                        hardware=hardware_profile,
                        precision=args.precision,
                    )
                else:
                    # Default: use actual inference benchmarking
                    progress.step("Benchmarking batch sizes (actual inference)")
                    sweep_result = profiler.run_batch_sweep_benchmark(
                        model_path=str(model_path),
                        batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128],
                    )
                    if sweep_result is None:
                        # Fall back to theoretical if benchmark fails
                        logger.warning("Benchmark failed, using theoretical estimates")
                        sweep_result = profiler.run_batch_sweep(
                            model_params=report.param_counts.total,
                            model_flops=report.flop_counts.total,
                            peak_activation_bytes=report.memory_estimates.peak_activation_bytes,
                            hardware=hardware_profile,
                            precision=args.precision,
                        )

                report.batch_size_sweep = sweep_result
                logger.info(
                    f"Batch sweep complete. Optimal batch size: {sweep_result.optimal_batch_size}"
                )

            # Resolution Sweep (Story 6.8)
            if args.sweep_resolutions:
                progress.step("Running resolution sweep")
                profiler = OperationalProfiler(logger=logger)

                # Determine base/training resolution from model input shape
                base_resolution = (224, 224)  # Default
                if report.graph_summary and report.graph_summary.input_shapes:
                    for shape in report.graph_summary.input_shapes.values():
                        if len(shape) >= 3:
                            # Assume NCHW or NHWC format
                            h, w = shape[-2], shape[-1]
                            if isinstance(h, int) and isinstance(w, int) and h > 1 and w > 1:
                                base_resolution = (h, w)
                                break

                # Parse resolutions from CLI argument
                # Note: Only resolutions UP TO training resolution are reliable
                resolutions: list[tuple[int, int]] | None = None
                if args.sweep_resolutions != "auto":
                    resolutions = []
                    for res_part in args.sweep_resolutions.split(","):
                        res_str = res_part.strip()
                        if "x" in res_str:
                            h, w = res_str.split("x")
                            res_h, res_w = int(h), int(w)
                            # Warn if resolution exceeds training resolution
                            if res_h > base_resolution[0] or res_w > base_resolution[1]:
                                logger.warning(
                                    f"Resolution {res_str} exceeds training resolution "
                                    f"{base_resolution[0]}x{base_resolution[1]}. "
                                    "Results may be unreliable."
                                )
                            resolutions.append((res_h, res_w))

                res_sweep_result = profiler.run_resolution_sweep(
                    base_flops=report.flop_counts.total,
                    base_activation_bytes=report.memory_estimates.peak_activation_bytes,
                    base_resolution=base_resolution,
                    model_params=report.param_counts.total,
                    hardware=hardware_profile,
                    resolutions=resolutions,
                    batch_size=args.batch_size,
                    precision=args.precision,
                )
                report.resolution_sweep = res_sweep_result
                logger.info(
                    f"Resolution sweep complete. Max resolution: {res_sweep_result.max_resolution}, "
                    f"Optimal: {res_sweep_result.optimal_resolution}"
                )

        # System Requirements (Story 6C.2)
        if (
            args.system_requirements
            and report.param_counts
            and report.flop_counts
            and report.memory_estimates
        ):
            progress.step("Generating system requirements")
            profiler = OperationalProfiler(logger=logger)

            # Derive a target FPS based on deployment target / explicit knobs
            target_fps: float | None = None
            if getattr(args, "target_throughput_fps", None):
                target_fps = float(args.target_throughput_fps)
            elif getattr(args, "target_latency_ms", None):
                # latency (ms) -> fps
                if args.target_latency_ms > 0:
                    target_fps = 1000.0 / float(args.target_latency_ms)

            # Fallback targets based on deployment target category
            if target_fps is None:
                if args.deployment_target == "edge":
                    target_fps = 30.0
                elif args.deployment_target == "local":
                    target_fps = 60.0
                elif args.deployment_target == "cloud":
                    target_fps = 120.0
                else:
                    target_fps = 30.0

            sys_reqs = profiler.determine_system_requirements(
                model_params=report.param_counts.total,
                model_flops=report.flop_counts.total,
                peak_activation_bytes=report.memory_estimates.peak_activation_bytes,
                precision=args.precision,
                target_fps=target_fps,
            )
            report.system_requirements = sys_reqs
            if sys_reqs.recommended_gpu:
                logger.info(f"Recommended device: {sys_reqs.recommended_gpu.device}")
            elif sys_reqs.minimum_gpu:
                logger.info(f"Minimum device: {sys_reqs.minimum_gpu.device}")

        # Epic 9: Runtime Profiling (defaults ON for real measurements)
        profiler = OperationalProfiler(logger=logger)
        profiling_result = None  # Store for use in NN graph visualization

        # GPU Metrics (Story 9.2) - default ON
        if not args.no_gpu_metrics:
            progress.step("Capturing GPU metrics")
            gpu_metrics = profiler.get_gpu_metrics()
            if gpu_metrics:
                logger.info(
                    f"GPU: {gpu_metrics.vram_used_bytes / (1024**3):.2f} GB VRAM used, "
                    f"{gpu_metrics.gpu_utilization_percent:.0f}% utilization, "
                    f"{gpu_metrics.temperature_c}C"
                )
                # Store in report (add to JSON output)
                report.extra_data = report.extra_data or {}
                report.extra_data["gpu_metrics"] = gpu_metrics.to_dict()
            else:
                logger.debug("GPU metrics unavailable (pynvml not installed)")

        # Per-Layer Profiling (Story 9.3) - default ON
        if not args.no_profile:
            progress.step("Running ONNX Runtime profiler")
            profiling_result = profiler.profile_model(
                model_path=str(model_path),
                batch_size=args.batch_size,
                num_runs=args.profile_runs,
            )
            if profiling_result:
                logger.info(
                    f"Profiling complete: {profiling_result.total_time_ms:.2f}ms "
                    f"({len(profiling_result.layer_profiles)} layers)"
                )
                # Show slowest layers
                slowest = profiling_result.get_slowest_layers(5)
                if slowest:
                    logger.info("Top 5 slowest layers:")
                    for lp in slowest:
                        logger.info(f"  {lp.name}: {lp.duration_ms:.3f}ms ({lp.op_type})")

                # Store in report
                report.extra_data = report.extra_data or {}
                report.extra_data["profiling"] = profiling_result.to_dict()

                # Bottleneck Analysis (Story 9.4) - default ON
                if not args.no_bottleneck_analysis and hardware_profile and report.flop_counts:
                    progress.step("Analyzing bottlenecks")
                    bottleneck = profiler.analyze_bottleneck(
                        model_flops=report.flop_counts.total,
                        profiling_result=profiling_result,
                        hardware=hardware_profile,
                        precision=args.precision,
                    )
                    logger.info(
                        f"Bottleneck: {bottleneck.bottleneck_type} "
                        f"(compute: {bottleneck.compute_ratio:.0%}, "
                        f"memory: {bottleneck.memory_ratio:.0%})"
                    )
                    logger.info(f"Efficiency: {bottleneck.efficiency_percent:.1f}%")
                    for rec in bottleneck.recommendations[:3]:
                        logger.info(f"  - {rec}")
                    report.extra_data["bottleneck_analysis"] = bottleneck.to_dict()
            else:
                logger.debug("Profiling unavailable (onnxruntime not installed)")

        # Resolution Benchmarking (Story 9.5) - default ON
        if not args.no_benchmark_resolutions:
            progress.step("Benchmarking resolutions (actual inference)")
            res_benchmark = profiler.benchmark_resolutions(
                model_path=str(model_path),
                batch_size=args.batch_size,
            )
            if res_benchmark:
                logger.info(
                    f"Resolution benchmark complete. Optimal: {res_benchmark.optimal_resolution}"
                )
                report.extra_data = report.extra_data or {}
                report.extra_data["resolution_benchmark"] = res_benchmark.to_dict()
            else:
                logger.debug("Resolution benchmark unavailable")

    except Exception as e:
        logger.error(f"Failed to inspect model: {e}")
        if args.log_level == "debug":
            import traceback

            traceback.print_exc()
        sys.exit(1)

    # Extract dataset metadata if PyTorch weights provided
    if args.pytorch_weights or args.from_pytorch:
        weights_path = args.pytorch_weights or args.from_pytorch
        if weights_path.exists():
            logger.info(f"Extracting metadata from: {weights_path}")
            metadata = _extract_ultralytics_metadata(weights_path, logger)
            if metadata:
                from .report import DatasetInfo

                report.dataset_info = DatasetInfo(
                    task=metadata.get("task"),
                    num_classes=metadata.get("num_classes"),
                    class_names=metadata.get("class_names", []),
                    source=metadata.get("source"),
                )
                logger.info(
                    f"Extracted {report.dataset_info.num_classes} class(es): "
                    f"{', '.join(report.dataset_info.class_names[:5])}"
                    f"{'...' if len(report.dataset_info.class_names) > 5 else ''}"
                )

    # Quantization linting if requested
    quant_lint_result = None
    if args.lint_quantization or args.lint_quant:
        progress.step("Analyzing quantization readiness")
        from .quantization_linter import QuantizationLinter, QuantWarning

        linter = QuantizationLinter(logger=logger)
        # Need graph_info for linting
        from .analyzer import ONNXGraphLoader

        graph_loader = ONNXGraphLoader(logger=logger)
        _, graph_info = graph_loader.load(model_path)
        quant_lint_result = linter.lint(graph_info)

        # Print summary to console
        print("\n" + "=" * 60)
        print("QUANTIZATION READINESS ANALYSIS")
        print("=" * 60)
        print(quant_lint_result.get_summary())
        print()

        if quant_lint_result.warnings:
            print(f"Issues Found ({len(quant_lint_result.warnings)}):")
            print("-" * 40)
            warning: QuantWarning
            for warning in sorted(quant_lint_result.warnings, key=lambda x: x.severity.value):
                severity_icon = {
                    "critical": "[!!]",
                    "high": "[!] ",
                    "medium": "[~] ",
                    "low": "[.] ",
                    "info": "[i] ",
                }.get(warning.severity.value, "    ")
                print(f"  {severity_icon} {warning.message}")
                if warning.recommendation:
                    print(f"       -> {warning.recommendation}")
            print()

        # Get recommendations
        recommendations = linter.get_recommendations(quant_lint_result)
        if recommendations:
            print("Recommendations:")
            print("-" * 40)
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
            print()
        print("=" * 60 + "\n")

        # Write report to file if requested
        if args.quant_report:
            report_md = _generate_quant_report_markdown(quant_lint_result, model_path.name)
            args.quant_report.write_text(report_md, encoding="utf-8")
            logger.info(f"Quantization report written to {args.quant_report}")

        # Generate HTML report if requested
        if args.quant_report_html:
            from .quantization_advisor import (
                advise_quantization,
                generate_qat_readiness_report,
            )

            advice = advise_quantization(quant_lint_result, graph_info, use_llm=False)
            html_report = generate_qat_readiness_report(
                quant_lint_result, advice, model_path.name, format="html"
            )
            args.quant_report_html.write_text(html_report, encoding="utf-8")
            logger.info(f"HTML quantization report written to {args.quant_report_html}")

        # Generate LLM-powered advice if requested
        quant_advice: QuantizationAdvice | None = None
        if args.quant_llm_advice:
            from .quantization_advisor import QuantizationAdvisor, generate_qat_readiness_report

            if not has_llm_api_key():
                print("\n[WARNING] --quant-llm-advice requires OPENAI_API_KEY env var")
                print("Using heuristic-based advice instead.\n")
                advisor = QuantizationAdvisor(use_llm=False)
            else:
                progress.step("Generating LLM quantization advice")
                advisor = QuantizationAdvisor(model=args.llm_model, use_llm=True)

            quant_advice = advisor.advise(quant_lint_result, graph_info)

            # Print advice to console
            print("\n" + "=" * 60)
            print("QUANTIZATION RECOMMENDATIONS")
            print("=" * 60)
            print(f"\nArchitecture: {quant_advice.architecture_type.value.upper()}")
            print(f"Strategy: {quant_advice.strategy}")
            print(f"\nExpected Accuracy Impact: {quant_advice.expected_accuracy_impact}")
            print()

            if quant_advice.sensitive_layers:
                print("Sensitive Layers (keep at FP16):")
                for layer in quant_advice.sensitive_layers[:5]:
                    print(f"  - {layer}")
            print()

            if quant_advice.op_substitutions:
                print("Recommended Op Substitutions:")
                for sub in quant_advice.op_substitutions:
                    print(f"  - {sub.original_op} -> {sub.replacement_op}: {sub.reason}")
            print()

            print("QAT Workflow:")
            for i, step in enumerate(quant_advice.qat_workflow[:4], 1):
                print(f"  {i}. {step}")
            if len(quant_advice.qat_workflow) > 4:
                print(f"  ... ({len(quant_advice.qat_workflow) - 4} more steps)")
            print("=" * 60 + "\n")

            # Write QAT readiness report if requested
            if args.quant_advice_report:
                advice_report = generate_qat_readiness_report(
                    quant_lint_result, quant_advice, model_path.name, format="markdown"
                )
                args.quant_advice_report.write_text(advice_report, encoding="utf-8")
                logger.info(f"QAT readiness report written to {args.quant_advice_report}")

        # Store in report for JSON output
        report.quantization_lint = quant_lint_result

    # Generate LLM summaries if requested
    llm_summary = None
    if args.llm_summary:
        if args.offline:
            print("\n[OFFLINE MODE] Skipping LLM summary (requires network access)\n")
        elif not is_llm_available():
            print("\n" + "=" * 60)
            print("LLM PACKAGE NOT INSTALLED")
            print("=" * 60)
            print("To enable AI-powered summaries, install the LLM extras:\n")
            print("  pip install haoline[llm]")
            print("\nThen set your API key and try again.")
            print("=" * 60 + "\n")
        elif not has_llm_api_key():
            print("\n" + "=" * 60)
            print("API KEY REQUIRED FOR LLM SUMMARIES")
            print("=" * 60)
            print("Set one of the following environment variables:\n")
            print("  PowerShell:  $env:OPENAI_API_KEY = 'sk-...'")
            print("  Bash/Zsh:    export OPENAI_API_KEY='sk-...'")
            print("\nGet your API key at: https://platform.openai.com/api-keys")
            print("=" * 60 + "\n")
        else:
            try:
                progress.step(f"Generating LLM summary with {args.llm_model}")
                logger.info(f"Generating LLM summaries with {args.llm_model}...")
                summarizer = LLMSummarizer(model=args.llm_model, logger=logger)
                llm_summary = summarizer.summarize(report)
                if llm_summary.success:
                    logger.info(f"LLM summaries generated ({llm_summary.tokens_used} tokens used)")
                else:
                    logger.warning(f"LLM summarization failed: {llm_summary.error_message}")
            except Exception as e:
                logger.warning(f"Failed to generate LLM summaries: {e}")

    # Store LLM summary in report for output
    if llm_summary and llm_summary.success:
        # Add to report dict for JSON output
        report._llm_summary = llm_summary  # type: ignore

    # Apply privacy transformations if requested
    report_dict = report.to_dict()
    if args.summary_only:
        from .privacy import create_summary_only_dict

        logger.info("Applying summary-only mode (omitting per-layer details)")
        report_dict = create_summary_only_dict(report_dict)
    elif args.redact_names:
        from .privacy import collect_names_from_dict, create_name_mapping, redact_dict

        logger.info("Applying name redaction")
        names_to_redact = collect_names_from_dict(report_dict)
        mapping = create_name_mapping(names_to_redact)
        report_dict = redact_dict(report_dict, mapping)
        logger.debug(f"Redacted {len(mapping)} names")

    # Output results
    has_output = (
        args.out_json
        or args.out_md
        or args.out_html
        or args.out_pdf
        or args.html_graph
        or args.layer_csv
    )
    if has_output:
        progress.step("Writing output files")

    if args.out_json:
        try:
            import json

            args.out_json.parent.mkdir(parents=True, exist_ok=True)
            args.out_json.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
            logger.info(f"JSON report written to: {args.out_json}")
        except Exception as e:
            logger.error(f"Failed to write JSON report: {e}")
            sys.exit(1)

    # Generate visualizations if requested
    viz_paths = {}
    if args.with_plots:
        if not is_viz_available():
            logger.warning(
                "matplotlib not installed. Skipping visualizations. Install with: pip install matplotlib"
            )
        else:
            progress.step("Generating visualizations")
            # Determine assets directory
            if args.assets_dir:
                assets_dir = args.assets_dir
            elif args.out_html:
                # HTML embeds images, but we still generate them for the file
                assets_dir = args.out_html.parent / "assets"
            elif args.out_md:
                assets_dir = args.out_md.parent / "assets"
            elif args.out_json:
                assets_dir = args.out_json.parent / "assets"
            else:
                assets_dir = pathlib.Path("assets")

            try:
                viz_gen = VisualizationGenerator(logger=logger)
                viz_paths = viz_gen.generate_all(report, assets_dir)
                logger.info(f"Generated {len(viz_paths)} visualization assets in {assets_dir}")
            except Exception as e:
                logger.warning(f"Failed to generate some visualizations: {e}")
                if args.log_level == "debug":
                    import traceback

                    traceback.print_exc()

    if args.out_md:
        try:
            args.out_md.parent.mkdir(parents=True, exist_ok=True)
            # Generate markdown with visualizations and/or LLM summaries
            if viz_paths or llm_summary:
                md_content = _generate_markdown_with_extras(
                    report, viz_paths, args.out_md.parent, llm_summary
                )
            else:
                md_content = report.to_markdown()
            args.out_md.write_text(md_content, encoding="utf-8")
            logger.info(f"Markdown model card written to: {args.out_md}")
        except Exception as e:
            logger.error(f"Failed to write Markdown report: {e}")
            sys.exit(1)

    if args.out_html or args.out_pdf:
        # Add LLM summary to report if available
        if llm_summary and llm_summary.success:
            report.llm_summary = {
                "success": True,
                "short_summary": llm_summary.short_summary,
                "detailed_summary": llm_summary.detailed_summary,
                "model": args.llm_model,
            }
        # Generate layer table HTML if requested
        layer_table_html = None
        if args.include_layer_table or args.layer_csv:
            try:
                # Re-load graph info if needed
                if hasattr(inspector, "_graph_info") and inspector._graph_info:
                    graph_info = inspector._graph_info
                else:
                    from .analyzer import ONNXGraphLoader

                    loader = ONNXGraphLoader(logger=logger)
                    _, graph_info = loader.load(model_path)

                layer_builder = LayerSummaryBuilder(logger=logger)
                layer_summary = layer_builder.build(
                    graph_info,
                    report.param_counts,
                    report.flop_counts,
                    report.memory_estimates,
                )

                if args.include_layer_table:
                    layer_table_html = generate_html_table(layer_summary)
                    logger.debug("Generated layer summary table for HTML report")

                if args.layer_csv:
                    args.layer_csv.parent.mkdir(parents=True, exist_ok=True)
                    layer_summary.save_csv(args.layer_csv)
                    logger.info(f"Layer summary CSV written to: {args.layer_csv}")

            except Exception as e:
                logger.warning(f"Could not generate layer summary: {e}")

        # Generate embedded graph HTML if requested
        graph_html = None
        if args.include_graph:
            try:
                # Re-load graph info if needed
                if hasattr(inspector, "_graph_info") and inspector._graph_info:
                    graph_info = inspector._graph_info
                else:
                    from .analyzer import ONNXGraphLoader

                    loader = ONNXGraphLoader(logger=logger)
                    _, graph_info = loader.load(model_path)

                pattern_analyzer = PatternAnalyzer(logger=logger)
                blocks = pattern_analyzer.group_into_blocks(graph_info)

                edge_analyzer = EdgeAnalyzer(logger=logger)
                edge_result = edge_analyzer.analyze(graph_info)

                builder = HierarchicalGraphBuilder(logger=logger)
                hier_graph = builder.build(graph_info, blocks, model_path.stem)

                # Generate graph HTML (just the interactive part, not full document)
                # For embedding, we'll use an iframe approach

                # Extract layer timing from profiling results if available
                layer_timing = None
                if (
                    report.extra_data
                    and "profiling" in report.extra_data
                    and "slowest_layers" in report.extra_data["profiling"]
                ):
                    layer_timing = {}
                    for layer in report.extra_data["profiling"]["slowest_layers"]:
                        layer_timing[layer["name"]] = layer["duration_ms"]

                full_graph_html = generate_graph_html(
                    hier_graph, edge_result, model_path.stem, layer_timing=layer_timing
                )
                # Wrap in iframe data URI for embedding
                import base64

                graph_data = base64.b64encode(full_graph_html.encode()).decode()
                graph_html = f'<iframe src="data:text/html;base64,{graph_data}" style="width:100%;height:100%;border:none;"></iframe>'
                logger.debug("Generated interactive graph for HTML report")

            except Exception as e:
                logger.warning(f"Could not generate embedded graph: {e}")

        # Generate HTML with embedded images, graph, and layer table
        html_content = report.to_html(
            image_paths=viz_paths,
            graph_html=graph_html,
            layer_table_html=layer_table_html,
        )

        if args.out_html:
            try:
                args.out_html.parent.mkdir(parents=True, exist_ok=True)
                args.out_html.write_text(html_content, encoding="utf-8")
                logger.info(f"HTML report written to: {args.out_html}")
            except Exception as e:
                logger.error(f"Failed to write HTML report: {e}")
                sys.exit(1)

        if args.out_pdf:
            if not is_pdf_available():
                logger.error(
                    "Playwright not installed. Install with: pip install playwright && playwright install chromium"
                )
                sys.exit(1)
            try:
                args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
                pdf_gen = PDFGenerator(logger=logger)
                success = pdf_gen.generate_from_html(html_content, args.out_pdf)
                if success:
                    logger.info(f"PDF report written to: {args.out_pdf}")
                else:
                    logger.error("PDF generation failed")
                    sys.exit(1)
            except Exception as e:
                logger.error(f"Failed to write PDF report: {e}")
                sys.exit(1)

    # Interactive graph visualization
    if args.html_graph:
        try:
            args.html_graph.parent.mkdir(parents=True, exist_ok=True)

            # Use graph_info from the inspector if available
            if hasattr(inspector, "_graph_info") and inspector._graph_info:
                graph_info = inspector._graph_info
            else:
                # Re-load the model to get graph_info
                from .analyzer import ONNXGraphLoader

                loader = ONNXGraphLoader(logger=logger)
                _, graph_info = loader.load(model_path)

            # Detect patterns
            pattern_analyzer = PatternAnalyzer(logger=logger)
            blocks = pattern_analyzer.group_into_blocks(graph_info)

            # Analyze edges
            edge_analyzer = EdgeAnalyzer(logger=logger)
            edge_result = edge_analyzer.analyze(graph_info)

            # Build hierarchy
            builder = HierarchicalGraphBuilder(logger=logger)
            hier_graph = builder.build(graph_info, blocks, model_path.stem)

            # Export HTML with model size and layer timing
            model_size = model_path.stat().st_size if model_path.exists() else None

            # Extract layer timing from profiling results if available
            layer_timing_data: dict[str, float] | None = None
            if (
                report.extra_data
                and "profiling" in report.extra_data
                and "slowest_layers" in report.extra_data["profiling"]
            ):
                # Build timing dict from profiling results
                layer_timing_data = {}
                for layer in report.extra_data["profiling"]["slowest_layers"]:
                    layer_timing_data[layer["name"]] = layer["duration_ms"]

            exporter = HTMLExporter(logger=logger)
            exporter.export(
                hier_graph,
                edge_result,
                args.html_graph,
                model_path.stem,
                model_size_bytes=model_size,
                layer_timing=layer_timing_data,
            )

            logger.info(f"Interactive graph visualization written to: {args.html_graph}")
        except Exception as e:
            logger.error(f"Failed to generate graph visualization: {e}")
            if not args.quiet:
                import traceback

                traceback.print_exc()
            sys.exit(1)

    # Console output
    if (
        not args.quiet
        and not args.out_json
        and not args.out_md
        and not args.out_html
        and not args.out_pdf
        and not args.html_graph
    ):
        # No output files specified - print summary to console
        print("\n" + "=" * 60)
        print(f"Model: {model_path.name}")
        print("=" * 60)

        if report.graph_summary:
            print(f"\nNodes: {report.graph_summary.num_nodes}")
            print(f"Inputs: {report.graph_summary.num_inputs}")
            print(f"Outputs: {report.graph_summary.num_outputs}")
            print(f"Initializers: {report.graph_summary.num_initializers}")

        if report.param_counts:
            print(f"\nParameters: {report._format_number(report.param_counts.total)}")

        if report.flop_counts:
            print(f"FLOPs: {report._format_number(report.flop_counts.total)}")

        if report.memory_estimates:
            print(f"Model Size: {report._format_bytes(report.memory_estimates.model_size_bytes)}")

        print(f"\nArchitecture: {report.architecture_type}")
        print(f"Detected Blocks: {len(report.detected_blocks)}")

        # Hardware estimates
        if hasattr(report, "hardware_estimates") and report.hardware_estimates:
            hw = report.hardware_estimates
            print(f"\n--- Hardware Estimates ({hw.device}) ---")
            print(f"Precision: {hw.precision}, Batch Size: {hw.batch_size}")
            print(f"VRAM Required: {report._format_bytes(hw.vram_required_bytes)}")
            print(f"Fits in VRAM: {'Yes' if hw.fits_in_vram else 'NO'}")
            if hw.fits_in_vram:
                print(f"Theoretical Latency: {hw.theoretical_latency_ms:.2f} ms")
                print(f"Bottleneck: {hw.bottleneck}")

        # System Requirements (Console)
        if hasattr(report, "system_requirements") and report.system_requirements:
            reqs = report.system_requirements
            print("\n--- System Requirements ---")
            print(f"Minimum:     {reqs.minimum_gpu.name} ({reqs.minimum_vram_gb} GB VRAM)")
            print(f"Recommended: {reqs.recommended_gpu.name} ({reqs.recommended_vram_gb} GB VRAM)")
            print(f"Optimal:     {reqs.optimal_gpu.name}")

        # Batch Scaling (Console)
        if hasattr(report, "batch_size_sweep") and report.batch_size_sweep:
            sweep = report.batch_size_sweep
            print("\n--- Batch Size Scaling ---")
            print(f"Optimal Batch Size: {sweep.optimal_batch_size}")
            print(f"Max Throughput: {max(sweep.throughputs):.1f} inf/s")

        if report.risk_signals:
            print(f"\nRisk Signals: {len(report.risk_signals)}")
            risk_icons = {
                "info": "[INFO]",
                "warning": "[WARN]",
                "high": "[HIGH]",
            }
            for risk in report.risk_signals:
                print(f"  {risk_icons.get(risk.severity, '')} {risk.id}")

        # LLM Summary
        if llm_summary and llm_summary.success:
            print(f"\n--- LLM Summary ({llm_summary.model_used}) ---")
            if llm_summary.short_summary:
                print(f"{llm_summary.short_summary}")

        print("\n" + "=" * 60)
        print("Use --out-json or --out-md for detailed reports.")
        if not args.hardware:
            print("Use --hardware auto or --hardware <profile> for hardware estimates.")
        print("=" * 60 + "\n")

    elif not args.quiet:
        # Files written - just confirm
        print(f"\nInspection complete for: {model_path.name}")
        if args.out_json:
            print(f"  JSON report: {args.out_json}")
        if args.out_md:
            print(f"  Markdown card: {args.out_md}")
        if args.out_html:
            print(f"  HTML report: {args.out_html}")
        if args.out_pdf:
            print(f"  PDF report: {args.out_pdf}")
        if args.html_graph:
            print(f"  Graph visualization: {args.html_graph}")
        if args.layer_csv:
            print(f"  Layer CSV: {args.layer_csv}")

    # Finish progress indicator
    progress.finish("Analysis complete!")

    # Cleanup temp ONNX file if we created one
    if temp_onnx_file is not None:
        try:
            pathlib.Path(temp_onnx_file.name).unlink()
            logger.debug(f"Cleaned up temp ONNX file: {temp_onnx_file.name}")
        except Exception:
            pass


if __name__ == "__main__":
    run_inspect()
