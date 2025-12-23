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
        table.add_row("Inputs", str(len(report.graph_summary.input_names)))
        table.add_row("Outputs", str(len(report.graph_summary.output_names)))

    console.print()
    console.print(table)

    # Show hardware estimates if available
    if report.hardware_estimates:
        hw_table = Table(title="Hardware Estimates", show_header=True, header_style="bold green")
        hw_table.add_column("Metric", style="dim")
        hw_table.add_column("Value", justify="right")

        hw_table.add_row(
            "VRAM Required", format_size(report.hardware_estimates.vram_required_bytes)
        )
        hw_table.add_row("Est. Latency", f"{report.hardware_estimates.estimated_latency_ms:.1f} ms")
        hw_table.add_row("Est. Throughput", f"{report.hardware_estimates.throughput_fps:.1f} FPS")
        hw_table.add_row("Bottleneck", report.hardware_estimates.bottleneck)

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
) -> None:
    """Compare multiple model variants (quantization, architecture)."""
    from haoline.compare import main as compare_main

    # Build args for legacy compare CLI
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
