# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
HaoLine CLI - Universal Model Inspector.

Entry point for the `haoline` command-line tool.
"""
from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys
from typing import Any

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
from .operational_profiling import OperationalProfiler
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


def parse_args():
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
""",
    )

    parser.add_argument(
        "model_path",
        type=pathlib.Path,
        nargs="?",  # Optional now since --list-hardware doesn't need it
        help="Path to the ONNX model file to analyze.",
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
            "--input-shape is required for PyTorch conversion. "
            "Example: --input-shape 1,3,224,224"
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

    # Try 2: Regular torch.load (full model or state_dict)
    if model is None:
        try:
            loaded = torch.load(pytorch_path, map_location="cpu", weights_only=False)

            if isinstance(loaded, dict):
                # It's a state_dict - we can't use it directly
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
            dummy_input,
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
    try:
        import tf2onnx
        from tf2onnx import tf_loader
    except ImportError:
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
    try:
        import tf2onnx
    except ImportError:
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
    try:
        import tf2onnx
    except ImportError:
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
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        logger.error("JAX not installed. Install with: pip install jax jaxlib")
        return None, None

    try:
        import tf2onnx
    except ImportError:
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
            import tensorflow as tf
            from jax.experimental import jax2tf
        except ImportError:
            logger.error(
                "jax2tf or TensorFlow not available. Install with: " "pip install tensorflow"
            )
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


def run_inspect():
    """Main entry point for the model_inspect CLI."""
    # Load environment variables from .env file if present
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass  # python-dotenv not installed, use environment variables directly

    args = parse_args()
    logger = setup_logging(args.log_level)

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

    # Handle model conversion if requested
    temp_onnx_file = None
    conversion_sources = [
        ("--from-pytorch", args.from_pytorch),
        ("--from-tensorflow", args.from_tensorflow),
        ("--from-keras", args.from_keras),
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
                "--from-frozen-graph, --from-jax)."
            )
            sys.exit(1)

        model_path = args.model_path.resolve()
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            sys.exit(1)

        if model_path.suffix.lower() not in (".onnx", ".pb", ".ort"):
            logger.warning(f"Unexpected file extension: {model_path.suffix}. Proceeding anyway.")

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
                if not args.no_bottleneck_analysis and hardware_profile:
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
                    f"Resolution benchmark complete. "
                    f"Optimal: {res_benchmark.optimal_resolution}"
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
        names = collect_names_from_dict(report_dict)
        mapping = create_name_mapping(names)
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
            layer_timing = None
            if (
                report.extra_data
                and "profiling" in report.extra_data
                and "slowest_layers" in report.extra_data["profiling"]
            ):
                # Build timing dict from profiling results
                layer_timing = {}
                for layer in report.extra_data["profiling"]["slowest_layers"]:
                    layer_timing[layer["name"]] = layer["duration_ms"]

            exporter = HTMLExporter(logger=logger)
            exporter.export(
                hier_graph,
                edge_result,
                args.html_graph,
                model_path.stem,
                model_size_bytes=model_size,
                layer_timing=layer_timing,
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
            for risk in report.risk_signals:
                severity_icon = {
                    "info": "[INFO]",
                    "warning": "[WARN]",
                    "high": "[HIGH]",
                }
                print(f"  {severity_icon.get(risk.severity, '')} {risk.id}")

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
