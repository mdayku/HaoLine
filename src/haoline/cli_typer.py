#!/usr/bin/env python
# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
HaoLine CLI - Universal Model Inspector (Typer version).

Modern CLI built with Typer for better UX, shell completion, and rich output.
"""

from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Rich console for output
console = Console()
err_console = Console(stderr=True)


def _version_callback(value: bool) -> None:
    """Print version and exit if --version is passed."""
    if value:
        from haoline import __version__

        console.print(f"[bold]HaoLine[/bold] version [cyan]{__version__}[/cyan]")
        raise typer.Exit()


# Initialize Typer app with rich markup
app = typer.Typer(
    name="haoline",
    help="HaoLine - Universal Model Inspector. See what's really inside your models.",
    add_completion=True,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


def _list_hardware_callback(value: bool) -> None:
    """Callback for --list-hardware flag (backwards compatibility)."""
    if value:
        from haoline.hardware import HARDWARE_PROFILES

        table = Table(
            title="Available Hardware Profiles", show_header=True, header_style="bold cyan"
        )
        table.add_column("Key", style="dim")
        table.add_column("Name")
        table.add_column("Type")
        table.add_column("VRAM", justify="right")

        for key, profile in sorted(HARDWARE_PROFILES.items()):
            vram_gb = profile.vram_bytes / (1024**3)
            table.add_row(key, profile.name, profile.device_type, f"{vram_gb:.0f} GB")

        console.print(table)
        raise typer.Exit()


def _list_formats_callback(value: bool) -> None:
    """Callback for --list-formats flag (backwards compatibility)."""
    if value:
        from haoline.format_adapters import (
            FORMAT_CAPABILITIES,
            SourceFormat,
            list_adapters,
        )

        table = Table(title="Supported Formats", show_header=True, header_style="bold cyan")
        table.add_column("Format", style="dim")
        table.add_column("Extensions")
        table.add_column("Graph")
        table.add_column("FLOPs")

        for adapter_info in list_adapters():
            name = str(adapter_info["name"])
            ext_list = adapter_info["extensions"]
            extensions = ", ".join(ext_list) if isinstance(ext_list, list) else str(ext_list)
            source_fmt_str = adapter_info["source_format"]
            # Convert string to SourceFormat enum
            try:
                source_fmt = SourceFormat(source_fmt_str)
                caps = FORMAT_CAPABILITIES.get(source_fmt)
                if caps:
                    graph = "[green]Yes[/green]" if caps.has_graph else "[dim]No[/dim]"
                    flops = "[green]Yes[/green]" if caps.has_flops else "[dim]No[/dim]"
                else:
                    graph = "[dim]?[/dim]"
                    flops = "[dim]?[/dim]"
            except (ValueError, KeyError):
                graph = "[dim]?[/dim]"
                flops = "[dim]?[/dim]"
            table.add_row(name.upper(), extensions, graph, flops)

        console.print(table)
        raise typer.Exit()


@app.callback()
def callback(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-V",
            help="Show version and exit",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = None,
    list_hardware: Annotated[
        bool,
        typer.Option(
            "--list-hardware",
            help="List available hardware profiles and exit",
            callback=_list_hardware_callback,
            is_eager=True,
        ),
    ] = False,
    list_formats: Annotated[
        bool,
        typer.Option(
            "--list-formats",
            help="List supported model formats and exit",
            callback=_list_formats_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """HaoLine - Universal Model Inspector."""
    pass


# Enums for choices
class Precision(str, Enum):
    fp32 = "fp32"
    fp16 = "fp16"
    bf16 = "bf16"
    int8 = "int8"


class LogLevel(str, Enum):
    debug = "debug"
    info = "info"
    warning = "warning"
    error = "error"


class DeploymentTarget(str, Enum):
    edge = "edge"
    local = "local"
    cloud = "cloud"


# =============================================================================
# Helper functions
# =============================================================================


def check_dependency(module: str, extra: str, feature: str) -> bool:
    """Check if a dependency is available, show install hint if not."""
    try:
        __import__(module)
        return True
    except ImportError:
        err_console.print(
            f"[yellow]Warning:[/yellow] {feature} requires [cyan]{module}[/cyan]\n"
            f"  Install with: [bold]pip install haoline[{extra}][/bold]"
        )
        return False


def format_size(bytes_val: int | float) -> str:
    """Format bytes as human-readable size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(bytes_val) < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} PB"


def format_number(n: int | float) -> str:
    """Format large numbers with K/M/B suffixes."""
    if n >= 1e12:
        return f"{n / 1e12:.2f}T"
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return str(int(n))


# =============================================================================
# Main inspect command
# =============================================================================


@app.command()
def inspect(
    model_path: Annotated[
        Path | None,
        typer.Argument(
            help="Path to model file (ONNX, TensorRT, PyTorch, etc.)",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    # Output options
    out_json: Annotated[
        Path | None,
        typer.Option("--out-json", "-j", help="Output path for JSON report"),
    ] = None,
    out_md: Annotated[
        Path | None,
        typer.Option("--out-md", "-m", help="Output path for Markdown model card"),
    ] = None,
    out_html: Annotated[
        Path | None,
        typer.Option("--out-html", help="Output path for HTML report"),
    ] = None,
    out_pdf: Annotated[
        Path | None,
        typer.Option("--out-pdf", help="Output path for PDF report (requires playwright)"),
    ] = None,
    include_graph: Annotated[
        bool,
        typer.Option("--include-graph", help="Include interactive D3.js graph in HTML"),
    ] = False,
    # Hardware options
    hardware: Annotated[
        str | None,
        typer.Option("--hardware", "-H", help="Hardware profile (auto, rtx4090, a100, etc.)"),
    ] = None,
    precision: Annotated[
        Precision,
        typer.Option("--precision", "-p", help="Precision for estimates"),
    ] = Precision.fp32,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", "-b", help="Batch size for estimates"),
    ] = 1,
    gpu_count: Annotated[
        int,
        typer.Option("--gpu-count", help="Number of GPUs for multi-GPU estimates"),
    ] = 1,
    # Conversion options
    from_pytorch: Annotated[
        Path | None,
        typer.Option("--from-pytorch", help="Convert PyTorch model to ONNX first"),
    ] = None,
    input_shape: Annotated[
        str | None,
        typer.Option("--input-shape", help="Input shape for conversion (e.g., 1,3,224,224)"),
    ] = None,
    # LLM options
    llm_summary: Annotated[
        bool,
        typer.Option("--llm-summary", help="Generate AI-powered summary (requires API key)"),
    ] = False,
    llm_model: Annotated[
        str,
        typer.Option("--llm-model", help="LLM model for summaries"),
    ] = "gpt-4o-mini",
    # Quantization options
    lint_quant: Annotated[
        bool,
        typer.Option("--lint-quant/--no-lint-quant", help="Analyze quantization readiness"),
    ] = False,
    # Visualization options
    with_plots: Annotated[
        bool,
        typer.Option("--with-plots", help="Generate visualization charts"),
    ] = False,
    # General options
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Suppress console output"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed output"),
    ] = False,
) -> None:
    """
    Analyze a neural network model and generate comprehensive reports.

    [bold]Examples:[/bold]

        haoline model.onnx
        haoline model.onnx --out-html report.html --include-graph
        haoline model.onnx --hardware rtx4090 --out-json report.json
        haoline --from-pytorch model.pt --input-shape 1,3,224,224
    """
    # Handle no model path
    if model_path is None and from_pytorch is None:
        console.print("[red]Error:[/red] No model path provided")
        console.print("Run [bold]haoline --help[/bold] for usage")
        raise typer.Exit(1)

    # Wrap everything in error handler
    try:
        _run_inspect(
            model_path=model_path,
            from_pytorch=from_pytorch,
            input_shape=input_shape,
            out_json=out_json,
            out_md=out_md,
            out_html=out_html,
            out_pdf=out_pdf,
            include_graph=include_graph,
            hardware=hardware,
            precision=precision,
            batch_size=batch_size,
            gpu_count=gpu_count,
            llm_summary=llm_summary,
            llm_model=llm_model,
            lint_quant=lint_quant,
            with_plots=with_plots,
            quiet=quiet,
            verbose=verbose,
        )
    except Exception as e:
        if verbose:
            # Show full traceback
            console.print_exception(show_locals=True)
        else:
            # User-friendly error with suggestions
            error_type = type(e).__name__
            err_console.print(f"[red]Error:[/red] {error_type}: {e}")

            # Suggest fixes for common errors
            suggestion = _get_error_suggestion(e, model_path, from_pytorch)
            if suggestion:
                err_console.print(f"\n[yellow]Suggestion:[/yellow] {suggestion}")

            err_console.print("\n[dim]Run with --verbose for full traceback[/dim]")
        raise typer.Exit(1) from None


def _get_error_suggestion(
    error: Exception,
    model_path: Path | None,
    from_pytorch: Path | None,
) -> str | None:
    """Return a helpful suggestion for common errors."""
    error_msg = str(error).lower()
    error_type = type(error).__name__

    # File not found
    if error_type == "FileNotFoundError" or "no such file" in error_msg:
        return "Check that the model file exists and the path is correct."

    # ONNX format errors
    if "onnx" in error_msg and ("invalid" in error_msg or "corrupt" in error_msg):
        return "The ONNX file may be corrupted. Try re-exporting from your framework."

    # Missing dependency
    if error_type == "ModuleNotFoundError":
        module = error_msg.replace("no module named ", "").strip("'\"")
        return f"Missing dependency. Try: pip install {module}"

    # PyTorch conversion without input shape
    if from_pytorch and "shape" in error_msg:
        return "Ensure --input-shape matches your model's expected input (e.g., 1,3,224,224)."

    # Memory errors
    if "memory" in error_msg or error_type == "MemoryError":
        return "Model too large for available memory. Try closing other applications."

    # Permission errors
    if error_type == "PermissionError":
        return "Check file permissions and ensure you have read access."

    # TensorRT errors
    if model_path and str(model_path).endswith(".engine"):
        if "tensorrt" in error_msg:
            return "TensorRT engine may be incompatible. Engines are GPU-specific."

    return None


def _run_inspect(
    *,
    model_path: Path | None,
    from_pytorch: Path | None,
    input_shape: str | None,
    out_json: Path | None,
    out_md: Path | None,
    out_html: Path | None,
    out_pdf: Path | None,
    include_graph: bool,
    hardware: str | None,
    precision: Precision,
    batch_size: int,
    gpu_count: int,
    llm_summary: bool,
    llm_model: str,
    lint_quant: bool,
    with_plots: bool,
    quiet: bool,
    verbose: bool,
) -> None:
    """Internal implementation of inspect command."""
    # Import the analysis engine
    from haoline import ModelInspector
    from haoline.hardware import (
        HardwareEstimator,
        HardwareProfile,
        detect_local_hardware,
        get_profile,
    )

    # Determine model to analyze
    if from_pytorch:
        if not check_dependency("torch", "pytorch", "PyTorch conversion"):
            raise typer.Exit(1)
        if not input_shape:
            err_console.print("[red]Error:[/red] --input-shape required with --from-pytorch")
            raise typer.Exit(1)

        # Convert PyTorch to ONNX
        with console.status("[bold blue]Converting PyTorch model to ONNX...[/bold blue]"):
            import logging

            from haoline._cli_legacy import _convert_pytorch_to_onnx

            logger = logging.getLogger("haoline.cli")
            result_path, _ = _convert_pytorch_to_onnx(
                pytorch_path=from_pytorch,
                input_shape_str=input_shape,
                output_path=None,  # Use temp file
                opset_version=17,
                logger=logger,
            )
            if not result_path:
                err_console.print("[red]Error:[/red] PyTorch conversion failed")
                raise typer.Exit(1)
            analysis_path = str(result_path)
    else:
        analysis_path = str(model_path)

    # Run analysis
    if not quiet:
        console.print(f"\n[bold]Analyzing:[/bold] {analysis_path}")

    with (
        console.status("[bold blue]Running analysis...[/bold blue]") if not quiet else nullcontext()
    ):
        inspector = ModelInspector()
        report = inspector.inspect(analysis_path)

    # Apply hardware estimates
    hw_profile: HardwareProfile | None = None
    if hardware:
        if hardware == "auto":
            hw_profile = detect_local_hardware()
        else:
            hw_profile = get_profile(hardware)

        if hw_profile and report.param_counts and report.flop_counts and report.memory_estimates:
            estimator = HardwareEstimator()
            report.hardware_profile = hw_profile
            report.hardware_estimates = estimator.estimate(
                model_params=report.param_counts.total,
                model_flops=report.flop_counts.total,
                peak_activation_bytes=report.memory_estimates.peak_activation_bytes,
                hardware=hw_profile,
            )

    # Quantization linting
    if lint_quant:
        from haoline.quantization_linter import QuantizationLinter

        linter = QuantizationLinter()
        # Load graph for linting
        from haoline.analyzer import ONNXGraphLoader

        loader = ONNXGraphLoader()
        _, graph_info = loader.load(analysis_path)
        report.quantization_lint = linter.lint(graph_info)

    # LLM summary
    if llm_summary:
        if not check_dependency("openai", "llm", "LLM summaries"):
            err_console.print("[yellow]Skipping LLM summary[/yellow]")
        else:
            from haoline.llm_summarizer import LLMSummarizer, has_api_key

            if has_api_key():
                with console.status("[bold blue]Generating AI summary...[/bold blue]"):
                    summarizer = LLMSummarizer(model=llm_model)
                    summary_result = summarizer.summarize(report)
                    # Convert Pydantic model to dict for report storage
                    report.llm_summary = summary_result.model_dump()
            else:
                err_console.print(
                    "[yellow]Warning:[/yellow] No API key found. "
                    "Set OPENAI_API_KEY environment variable."
                )

    # Output results
    if not quiet:
        display_report_summary(report)

    # Write outputs
    if out_json:
        out_json.write_text(report.to_json())
        console.print(f"[green]Wrote:[/green] {out_json}")

    if out_md:
        md_content = report.to_markdown()
        out_md.write_text(md_content)
        console.print(f"[green]Wrote:[/green] {out_md}")

    if out_html:
        # Generate HTML from report (include_graph would require HierarchicalGraph)
        html_content = report.to_html()
        out_html.write_text(html_content)
        console.print(f"[green]Wrote:[/green] {out_html}")

    if out_pdf:
        if not check_dependency("playwright", "pdf", "PDF export"):
            err_console.print("[yellow]Skipping PDF export[/yellow]")
        else:
            import pathlib

            from haoline.pdf_generator import PDFGenerator

            gen = PDFGenerator()
            success = gen.generate_from_report(report, pathlib.Path(out_pdf))
            if success:
                console.print(f"[green]Wrote:[/green] {out_pdf}")
            else:
                err_console.print("[yellow]Warning:[/yellow] PDF generation failed")


def display_report_summary(report) -> None:
    """Display a rich summary of the analysis."""

    # Create summary table
    table = Table(title="Model Analysis Summary", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")

    if report.param_counts:
        table.add_row("Parameters", format_number(report.param_counts.total))
    if report.flop_counts:
        table.add_row("FLOPs", format_number(report.flop_counts.total))
    if report.memory_estimates:
        table.add_row("Peak Memory", format_size(report.memory_estimates.peak_activation_bytes))
        table.add_row("Model Size", format_size(report.memory_estimates.model_size_bytes))
    if report.graph_summary:
        table.add_row("Operators", str(report.graph_summary.num_nodes))
        table.add_row("Inputs", str(report.graph_summary.num_inputs))
        table.add_row("Outputs", str(report.graph_summary.num_outputs))

    console.print()
    console.print(table)

    # Show hardware estimates if available
    if report.hardware_estimates:
        hw_table = Table(title="Hardware Estimates", show_header=True, header_style="bold green")
        hw_table.add_column("Metric", style="dim")
        hw_table.add_column("Value", justify="right")

        hw = report.hardware_estimates
        hw_table.add_row("VRAM Required", format_size(hw.vram_required_bytes))
        hw_table.add_row("Est. Latency", f"{hw.theoretical_latency_ms:.2f} ms")
        # Calculate throughput from latency (inferences per second)
        if hw.theoretical_latency_ms > 0:
            throughput = (hw.batch_size * 1000.0) / hw.theoretical_latency_ms
            hw_table.add_row("Est. Throughput", f"{throughput:.1f} inf/s")
        hw_table.add_row("GPU Utilization", f"{hw.compute_utilization_estimate * 100:.0f}%")
        hw_table.add_row("Bottleneck", hw.bottleneck)

        console.print()
        console.print(hw_table)

    # Show detected patterns
    if report.detected_blocks:
        console.print(f"\n[bold]Detected Patterns:[/bold] {len(report.detected_blocks)}")
        for block in report.detected_blocks[:5]:
            console.print(f"  - {block.name} ({block.block_type})")
        if len(report.detected_blocks) > 5:
            console.print(f"  ... and {len(report.detected_blocks) - 5} more")

    # Show risk signals
    if report.risk_signals:
        console.print(f"\n[bold yellow]Risk Signals:[/bold yellow] {len(report.risk_signals)}")
        for risk in report.risk_signals[:3]:
            console.print(f"  [yellow]![/yellow] {risk.message}")
        if len(report.risk_signals) > 3:
            console.print(f"  ... and {len(report.risk_signals) - 3} more")

    console.print()


# Context manager for optional status
class nullcontext:
    """Null context manager for Python < 3.10 compatibility."""

    def __enter__(self):
        return None

    def __exit__(self, *args):
        pass


# =============================================================================
# List commands
# =============================================================================


@app.command("list-hardware")
def list_hardware() -> None:
    """List all available hardware profiles."""
    from haoline.hardware import HARDWARE_PROFILES

    table = Table(title="Available Hardware Profiles", show_header=True, header_style="bold cyan")
    table.add_column("Key", style="dim")
    table.add_column("Name")
    table.add_column("VRAM", justify="right")
    table.add_column("FP16 TFLOPS", justify="right")

    # Group by category
    categories = {
        "H100": ["h100-sxm", "h100-pcie", "h100-nvl"],
        "A100": ["a100-80gb-sxm", "a100-80gb-pcie", "a100-40gb-sxm", "a100-40gb-pcie"],
        "RTX 40": ["rtx4090", "rtx4080", "rtx4070", "rtx4060"],
        "RTX 30": ["rtx3090", "rtx3080", "rtx3070", "rtx3060"],
        "Cloud": ["t4", "a10", "l4", "l40s"],
    }

    for category, keys in categories.items():
        table.add_row(f"[bold]{category}[/bold]", "", "", "", style="bold")
        for key in keys:
            if key in HARDWARE_PROFILES:
                p = HARDWARE_PROFILES[key]
                table.add_row(
                    f"  {key}",
                    p.name,
                    f"{p.vram_bytes // (1024**3)} GB",
                    f"{p.peak_fp16_tflops:.1f}",
                )

    console.print(table)
    console.print("\n[dim]Use --hardware <key> to select a profile[/dim]")


@app.command("list-formats")
def list_formats() -> None:
    """List all supported model formats."""
    table = Table(title="Supported Model Formats", show_header=True, header_style="bold cyan")
    table.add_column("Format", style="bold")
    table.add_column("Extensions")
    table.add_column("Status")
    table.add_column("Install With")

    formats = [
        ("ONNX", ".onnx", "[green]Built-in[/green]", "-"),
        ("PyTorch", ".pt, .pth", check_format("torch"), r"pip install haoline\[pytorch]"),
        (
            "TensorFlow",
            "SavedModel, .h5",
            check_format("tensorflow"),
            r"pip install haoline\[tensorflow]",
        ),
        ("TensorRT", ".engine, .plan", check_format("tensorrt"), r"pip install haoline\[tensorrt]"),
        ("TFLite", ".tflite", check_format("tflite_runtime"), r"pip install haoline\[tflite]"),
        (
            "CoreML",
            ".mlmodel, .mlpackage",
            check_format("coremltools"),
            r"pip install haoline\[coreml]",
        ),
        ("OpenVINO", ".xml + .bin", check_format("openvino"), r"pip install haoline\[openvino]"),
        ("GGUF", ".gguf", "[green]Built-in[/green]", "-"),
        (
            "SafeTensors",
            ".safetensors",
            check_format("safetensors"),
            r"pip install haoline\[safetensors]",
        ),
    ]

    for name, ext, status, install in formats:
        table.add_row(name, ext, status, install)

    console.print(table)


def check_format(module: str) -> str:
    """Check if format module is available."""
    try:
        __import__(module)
        return "[green]Available[/green]"
    except ImportError:
        return "[yellow]Not installed[/yellow]"


# =============================================================================
# Subcommands
# =============================================================================


@app.command()
def web(
    port: Annotated[int, typer.Option("--port", "-p", help="Port to run on")] = 8501,
    host: Annotated[str, typer.Option("--host", help="Host to bind to")] = "localhost",
) -> None:
    """Launch the HaoLine web interface (Streamlit)."""
    if not check_dependency("streamlit", "web", "Web interface"):
        raise typer.Exit(1)

    from haoline.web import main as web_main

    sys.argv = ["haoline-web", "--port", str(port), "--host", host]
    web_main()


def _parse_threshold(spec: str) -> tuple[str, float | None, bool]:
    """Parse a threshold specification like 'latency_increase=10%' or 'new_risk_signals'.

    Returns:
        (metric_name, threshold_value, is_percentage)
        For boolean thresholds like 'new_risk_signals', threshold_value is None.
    """
    if "=" not in spec:
        # Boolean threshold (e.g., 'new_risk_signals')
        return (spec.strip(), None, False)

    key, value = spec.split("=", 1)
    key = key.strip()
    value = value.strip()

    is_percentage = value.endswith("%")
    if is_percentage:
        value = value[:-1]

    try:
        threshold = float(value)
    except ValueError:
        raise typer.BadParameter(f"Invalid threshold value: {value}") from None

    return (key, threshold, is_percentage)


def _check_thresholds(
    compare_json: dict,
    fail_on_specs: list[str],
) -> list[tuple[str, str, bool]]:
    """Check thresholds against comparison results.

    Returns:
        List of (metric, message, passed) tuples.
    """
    results: list[tuple[str, str, bool]] = []

    # Get variants and find non-baseline ones (those with deltas)
    variants = compare_json.get("variants", [])

    for spec in fail_on_specs:
        metric, threshold, is_pct = _parse_threshold(spec)

        # Check each non-baseline variant
        for variant in variants:
            deltas = variant.get("deltas_vs_baseline")
            if deltas is None:
                continue  # This is the baseline

            precision = variant.get("precision", "unknown")

            if metric == "latency_increase":
                base_lat = _get_baseline_latency(compare_json)
                delta_lat = deltas.get("latency_ms", 0)
                if base_lat and base_lat > 0:
                    pct_change = (delta_lat / base_lat) * 100
                    if threshold is not None and pct_change > threshold:
                        results.append(
                            (
                                metric,
                                f"{precision}: latency increased {pct_change:.1f}% (threshold: {threshold}%)",
                                False,
                            )
                        )
                    else:
                        results.append(
                            (
                                metric,
                                f"{precision}: latency change {pct_change:.1f}% (within {threshold}%)",
                                True,
                            )
                        )

            elif metric == "memory_increase":
                base_mem = _get_baseline_memory(compare_json)
                delta_mem = deltas.get("peak_activation_bytes", 0) or deltas.get("memory_bytes", 0)
                if base_mem and base_mem > 0:
                    pct_change = (delta_mem / base_mem) * 100
                    if threshold is not None and pct_change > threshold:
                        results.append(
                            (
                                metric,
                                f"{precision}: memory increased {pct_change:.1f}% (threshold: {threshold}%)",
                                False,
                            )
                        )
                    else:
                        results.append(
                            (
                                metric,
                                f"{precision}: memory change {pct_change:.1f}% (within {threshold}%)",
                                True,
                            )
                        )

            elif metric == "param_increase":
                base_params = _get_baseline_params(compare_json)
                delta_params = deltas.get("total_params", 0)
                if base_params and base_params > 0:
                    pct_change = (delta_params / base_params) * 100
                    if threshold is not None and pct_change > threshold:
                        results.append(
                            (
                                metric,
                                f"{precision}: params increased {pct_change:.1f}% (threshold: {threshold}%)",
                                False,
                            )
                        )
                    else:
                        results.append(
                            (
                                metric,
                                f"{precision}: params change {pct_change:.1f}% (within {threshold}%)",
                                True,
                            )
                        )

            elif metric == "new_risk_signals":
                # Check if variant has new high-severity risk signals
                new_risks = variant.get("new_risk_signals", [])
                high_severity = [r for r in new_risks if r.get("severity") == "high"]
                if high_severity:
                    results.append(
                        (
                            metric,
                            f"{precision}: {len(high_severity)} new high-severity risk signals",
                            False,
                        )
                    )
                else:
                    results.append((metric, f"{precision}: no new high-severity risks", True))

    return results


def _get_baseline_latency(compare_json: dict) -> float | None:
    """Get baseline latency from comparison JSON."""
    for v in compare_json.get("variants", []):
        if v.get("deltas_vs_baseline") is None:  # This is baseline
            hw = v.get("hardware_estimates", {})
            if hw:
                lat = hw.get("theoretical_latency_ms") or hw.get("estimated_latency_ms")
                return float(lat) if lat is not None else None
    return None


def _get_baseline_memory(compare_json: dict) -> int | None:
    """Get baseline memory from comparison JSON."""
    for v in compare_json.get("variants", []):
        if v.get("deltas_vs_baseline") is None:  # This is baseline
            mem = v.get("memory_bytes")
            return int(mem) if mem is not None else None
    return None


def _get_baseline_params(compare_json: dict) -> int | None:
    """Get baseline params from comparison JSON."""
    for v in compare_json.get("variants", []):
        if v.get("deltas_vs_baseline") is None:  # This is baseline
            params = v.get("total_params")
            return int(params) if params is not None else None
    return None


def _build_decision_report(
    compare_json: dict,
    fail_on_specs: list[str] | None,
    threshold_results: list[tuple[str, str, bool]],
    models: list[Path],
) -> dict:
    """Build a decision report for audit/compliance purposes.

    The decision report captures:
    - What models were compared
    - What constraints were applied
    - What the results were
    - What the overall decision was
    """
    import hashlib
    from datetime import datetime, timezone

    from haoline import __version__

    # Compute file hashes
    models_info = []
    for model_path in models:
        model_info: dict = {
            "path": str(model_path),
            "exists": model_path.exists(),
        }
        if model_path.exists():
            model_info["size_bytes"] = model_path.stat().st_size
            model_info["modified"] = datetime.fromtimestamp(
                model_path.stat().st_mtime, tz=timezone.utc
            ).isoformat()
            # Compute MD5 hash (fast enough for audit purposes)
            with open(model_path, "rb") as f:
                model_info["hash_md5"] = hashlib.md5(f.read()).hexdigest()
        models_info.append(model_info)

    # Build constraints section from threshold results
    constraints: dict = {}
    for metric, message, passed in threshold_results:
        if metric not in constraints:
            constraints[metric] = {
                "threshold": None,
                "results": [],
            }
        # Parse threshold from fail_on_specs
        if fail_on_specs:
            for spec in fail_on_specs:
                parsed_metric, threshold, is_pct = _parse_threshold(spec)
                if parsed_metric == metric:
                    if threshold is not None:
                        constraints[metric]["threshold"] = (
                            f"{threshold}%" if is_pct else str(threshold)
                        )
                    break
        constraints[metric]["results"].append(
            {
                "message": message,
                "passed": passed,
            }
        )

    # Determine overall decision
    all_passed = all(passed for _, _, passed in threshold_results) if threshold_results else True
    decision = "APPROVED" if all_passed else "REJECTED"

    # Extract recommendations if available
    recommendations: list[str] = []
    for variant in compare_json.get("variants", []):
        quant_advice = variant.get("quantization_advice")
        if quant_advice:
            recs = quant_advice.get("recommendations", [])
            recommendations.extend(recs[:2])  # Take top 2 from each variant

    # Build the report
    report = {
        "decision_report": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "haoline_version": __version__,
            "models_compared": models_info,
            "baseline": compare_json.get("baseline_precision", "unknown"),
            "constraints": constraints,
            "decision": decision,
            "architecture_compatible": compare_json.get("architecture_compatible", True),
            "compatibility_warnings": compare_json.get("compatibility_warnings", []),
            "recommendations": recommendations[:5],  # Limit to 5
        }
    }

    return report


def _decision_report_to_markdown(report: dict) -> str:
    """Convert decision report to Markdown format."""
    dr = report.get("decision_report", {})

    lines = [
        "# Model Decision Report",
        "",
        f"**Generated:** {dr.get('timestamp', 'unknown')}",
        f"**HaoLine Version:** {dr.get('haoline_version', 'unknown')}",
        "",
        "## Models Compared",
        "",
    ]

    for model in dr.get("models_compared", []):
        lines.append(f"- **{model.get('path', 'unknown')}**")
        if model.get("hash_md5"):
            lines.append(f"  - Hash (MD5): `{model['hash_md5'][:12]}...`")
        if model.get("size_bytes"):
            size_mb = model["size_bytes"] / (1024 * 1024)
            lines.append(f"  - Size: {size_mb:.2f} MB")

    lines.extend(
        [
            "",
            f"**Baseline:** {dr.get('baseline', 'unknown')}",
            "",
            "## Constraints",
            "",
        ]
    )

    constraints = dr.get("constraints", {})
    if constraints:
        for metric, data in constraints.items():
            threshold = data.get("threshold", "N/A")
            lines.append(f"### {metric} (threshold: {threshold})")
            for result in data.get("results", []):
                status = "PASS" if result.get("passed") else "**FAIL**"
                lines.append(f"- {status}: {result.get('message', '')}")
            lines.append("")
    else:
        lines.append("*No constraints specified*")
        lines.append("")

    decision = dr.get("decision", "UNKNOWN")
    decision_emoji = "APPROVED" if decision == "APPROVED" else "REJECTED"
    lines.extend(
        [
            "## Decision",
            "",
            f"**{decision_emoji}**",
            "",
        ]
    )

    if not dr.get("architecture_compatible", True):
        lines.append("**Warning:** Models have architecture differences")
        for warning in dr.get("compatibility_warnings", []):
            lines.append(f"- {warning}")
        lines.append("")

    recommendations = dr.get("recommendations", [])
    if recommendations:
        lines.extend(
            [
                "## Recommendations",
                "",
            ]
        )
        for rec in recommendations:
            lines.append(f"- {rec}")
        lines.append("")

    lines.extend(
        [
            "---",
            "*Generated by HaoLine*",
        ]
    )

    return "\n".join(lines)


@app.command()
def compare(
    models: Annotated[
        list[Path],
        typer.Option("--models", "-m", help="Model files to compare"),
    ],
    eval_metrics: Annotated[
        list[Path],
        typer.Option("--eval-metrics", "-e", help="Eval metrics JSON files"),
    ],
    out_json: Annotated[
        Path | None,
        typer.Option("--out-json", help="Output comparison JSON"),
    ] = None,
    out_md: Annotated[
        Path | None,
        typer.Option("--out-md", help="Output comparison Markdown"),
    ] = None,
    out_html: Annotated[
        Path | None,
        typer.Option("--out-html", help="Output comparison HTML"),
    ] = None,
    fail_on: Annotated[
        list[str] | None,
        typer.Option(
            "--fail-on",
            help="Threshold to fail on (e.g., latency_increase=10%, memory_increase=20%, new_risk_signals). Can be used multiple times.",
        ),
    ] = None,
    decision_report: Annotated[
        Path | None,
        typer.Option(
            "--decision-report",
            help="Output path for decision report (JSON audit trail). Use .json or .md extension.",
        ),
    ] = None,
) -> None:
    """Compare multiple model variants (quantization, architecture).

    Use --fail-on for CI/CD pipelines to exit with code 1 if thresholds are exceeded.
    Use --decision-report for compliance/audit trails.

    Examples:
        python -m haoline compare --models base.onnx candidate.onnx \\
            --eval-metrics base_eval.json candidate_eval.json \\
            --fail-on latency_increase=10% \\
            --fail-on memory_increase=20% \\
            --decision-report decision.json
    """
    import json
    import tempfile

    from haoline.compare import main as compare_main

    # Determine if we need to capture JSON (for --fail-on or --decision-report)
    need_json = fail_on or decision_report

    if need_json:
        # Create temp file for JSON output
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        # Build args for legacy compare CLI
        args = ["--models"] + [str(m) for m in models]
        args += ["--eval-metrics"] + [str(e) for e in eval_metrics]
        args += ["--out-json", str(tmp_path)]  # Always capture JSON
        if out_md:
            args += ["--out-md", str(out_md)]
        if out_html:
            args += ["--out-html", str(out_html)]

        sys.argv = ["haoline-compare"] + args
        exit_code = compare_main()

        if exit_code != 0:
            raise typer.Exit(exit_code)

        # Read the comparison JSON
        try:
            compare_json = json.loads(tmp_path.read_text())
        finally:
            tmp_path.unlink(missing_ok=True)

        # Copy to user-specified path if requested
        if out_json:
            out_json.write_text(json.dumps(compare_json, indent=2))
            console.print(f"[green]Wrote:[/green] {out_json}")

        # Check thresholds if specified
        threshold_results: list[tuple[str, str, bool]] = []
        any_failed = False

        if fail_on:
            threshold_results = _check_thresholds(compare_json, fail_on)

            # Display results
            console.print("\n[bold]Threshold Checks:[/bold]")
            for _metric, message, passed in threshold_results:
                if passed:
                    console.print(f"  [green]PASS[/green] {message}")
                else:
                    console.print(f"  [red]FAIL[/red] {message}")
                    any_failed = True

        # Generate decision report if requested
        if decision_report:
            dr = _build_decision_report(
                compare_json,
                fail_on,
                threshold_results,
                models,
            )

            # Determine output format based on extension
            if decision_report.suffix.lower() == ".md":
                md_content = _decision_report_to_markdown(dr)
                decision_report.write_text(md_content)
            else:
                decision_report.write_text(json.dumps(dr, indent=2))

            console.print(f"[green]Wrote decision report:[/green] {decision_report}")

        # Exit with appropriate code
        if any_failed:
            err_console.print("\n[red bold]Threshold violation(s) detected. Failing.[/red bold]")
            raise typer.Exit(1)
        elif fail_on:
            console.print("\n[green bold]All thresholds passed.[/green bold]")

    else:
        # No --fail-on or --decision-report, just run the comparison normally
        args = ["--models"] + [str(m) for m in models]
        args += ["--eval-metrics"] + [str(e) for e in eval_metrics]
        if out_json:
            args += ["--out-json", str(out_json)]
        if out_md:
            args += ["--out-md", str(out_md)]
        if out_html:
            args += ["--out-html", str(out_html)]

        sys.argv = ["haoline-compare"] + args
        compare_main()


@app.command("check-install")
def check_install_cmd() -> None:
    """Check installation status and report issues."""
    import shutil

    console.print(Panel("[bold]HaoLine Installation Check[/bold]", style="cyan"))

    # Version
    from haoline import __version__

    console.print(f"\n[bold]Version:[/bold] {__version__}")

    # CLI commands
    console.print("\n[bold]CLI Commands:[/bold]")
    cli_commands = {
        "haoline": "python -m haoline",
        "haoline-compare": "python -m haoline compare",
        "haoline-web": "python -m haoline web",
    }

    for cmd, alt in cli_commands.items():
        path = shutil.which(cmd)
        if path:
            console.print(f"  [green]{cmd}[/green]: {path}")
        else:
            console.print(f"  [yellow]{cmd}[/yellow]: NOT ON PATH (use: {alt})")

    # Quick dependency summary
    console.print("\n[bold]Optional Features:[/bold]")
    console.print("  Run [cyan]python -m haoline check-deps[/cyan] for detailed dependency info")

    console.print(f"\n[bold]Python:[/bold] {sys.version.split()[0]}")
    console.print(f"[bold]Executable:[/bold] {sys.executable}")


# Dependency categories for check-deps
DEPENDENCY_CATEGORIES: dict[str, dict[str, tuple[str, str, str]]] = {
    "Format Converters": {
        "torch": ("pytorch", "PyTorch → ONNX conversion", "--from-pytorch"),
        "tensorflow": ("tensorflow", "TensorFlow → ONNX conversion", "--from-tensorflow"),
        "tf2onnx": ("tensorflow", "TF/Keras to ONNX", "--from-keras"),
        "jax": ("jax", "JAX → ONNX conversion", "--from-jax"),
    },
    "Format Readers": {
        "tensorrt": ("tensorrt", "TensorRT .engine analysis", "model.engine"),
        "safetensors": ("safetensors", "SafeTensors .safetensors", "model.safetensors"),
        "coremltools": ("coreml", "CoreML .mlmodel/.mlpackage", "model.mlmodel"),
        "openvino": ("openvino", "OpenVINO .xml/.bin", "model.xml"),
        "tflite_runtime": ("tflite", "TFLite .tflite", "model.tflite"),
    },
    "Features": {
        "streamlit": ("web", "Web UI (Streamlit)", "python -m haoline web"),
        "openai": ("llm", "AI summaries (OpenAI)", "--llm-summary"),
        "anthropic": ("llm", "AI summaries (Claude)", "--llm-provider anthropic"),
        "playwright": ("pdf", "PDF export", "--out-pdf report.pdf"),
        "onnxruntime": ("runtime", "Actual benchmarking", "--sweep-batch-sizes"),
    },
    "GPU & Optimization": {
        "onnxruntime-gpu": ("gpu", "GPU acceleration", "CUDA provider"),
        "pynvml": ("gpu", "GPU memory monitoring", "VRAM tracking"),
    },
}


def _check_module(module: str) -> bool:
    """Check if a module is importable."""
    import importlib.util

    # Handle special case for onnxruntime-gpu (same module name)
    if module == "onnxruntime-gpu":
        try:
            import onnxruntime

            providers = onnxruntime.get_available_providers()
            return "CUDAExecutionProvider" in providers
        except Exception:
            return False
    return importlib.util.find_spec(module.replace("-", "_")) is not None


@app.command("check-deps")
def check_deps_cmd(
    install: Annotated[
        bool,
        typer.Option("--install", "-i", help="Offer to install missing dependencies"),
    ] = False,
) -> None:
    """Check optional dependencies and show what features are available.

    Groups dependencies by feature category and shows install commands
    for missing ones.

    [bold]Examples:[/bold]

        python -m haoline check-deps
        python -m haoline check-deps --install
    """
    from haoline import __version__

    console.print(
        Panel(
            f"[bold]HaoLine Dependency Check[/bold]\nVersion {__version__}",
            style="cyan",
        )
    )

    installed_count = 0
    missing_count = 0
    missing_by_extra: dict[str, list[str]] = {}

    for category, deps in DEPENDENCY_CATEGORIES.items():
        console.print(f"\n[bold]{category}[/bold]")

        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
        table.add_column("Module", style="dim")
        table.add_column("Status")
        table.add_column("Feature")
        table.add_column("Usage Example", style="dim")

        for module, (extra, feature, usage) in deps.items():
            available = _check_module(module)
            if available:
                status = "[green]✓ Installed[/green]"
                installed_count += 1
            else:
                status = "[yellow]✗ Missing[/yellow]"
                missing_count += 1
                if extra not in missing_by_extra:
                    missing_by_extra[extra] = []
                missing_by_extra[extra].append(feature)

            table.add_row(module, status, feature, usage)

        console.print(table)

    # Summary
    console.print("\n" + "─" * 60)
    console.print(f"\n[bold]Summary:[/bold] {installed_count} installed, {missing_count} missing")

    if missing_by_extra:
        console.print("\n[bold]Install missing features:[/bold]")
        for extra, features in sorted(missing_by_extra.items()):
            console.print(f"  [cyan]pip install haoline[{extra}][/cyan]")
            for f in features:
                console.print(f"    → {f}")

        # Full install hint
        console.print("\n  [dim]Or install everything:[/dim]")
        console.print("  [cyan]pip install haoline[full][/cyan]")

        # Offer to install if --install flag used
        if install:
            console.print()
            if typer.confirm("Would you like to install all missing dependencies?"):
                extras = list(missing_by_extra.keys())
                extras_str = ",".join(extras)
                cmd = f"pip install haoline[{extras_str}]"
                console.print(f"\n[bold]Running:[/bold] {cmd}")
                import subprocess

                result = subprocess.run(cmd, shell=True)
                if result.returncode == 0:
                    console.print("\n[green]Installation complete![/green]")
                else:
                    err_console.print("\n[red]Installation failed[/red]")
                    raise typer.Exit(1)
    else:
        console.print("\n[green]All optional dependencies are installed![/green]")


# =============================================================================
# Entry point
# =============================================================================


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
