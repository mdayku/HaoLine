#!/usr/bin/env python
# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
HaoLine Compare CLI - Multi-model comparison.

This CLI takes multiple model variants plus corresponding eval/perf
metrics JSON files and produces a comparison JSON + Markdown
summary focused on quantization impact / multi-variant trade-offs.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from . import ModelInspector
from .compare_visualizations import (
    build_enhanced_markdown,
    compute_tradeoff_points,
    generate_compare_html,
    generate_compare_pdf,
    generate_memory_savings_chart,
    generate_tradeoff_chart,
)
from .compare_visualizations import (
    is_available as viz_available,
)

LOGGER = logging.getLogger("haoline.compare")


class VariantInputs(BaseModel):
    """Inputs required to build a single variant entry in the compare report."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_path: Path
    eval_metrics_path: Path
    precision: str | None = None


class VariantReport(BaseModel):
    """Bundle of inspection + eval metrics for a variant."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_path: Path
    precision: str
    quantization_scheme: str
    size_bytes: int
    report: Any  # InspectionReport
    metrics: dict[str, Any]


class ArchCompatResult(BaseModel):
    """Result of architecture compatibility check between model variants."""

    model_config = ConfigDict(frozen=True)

    is_compatible: bool
    warnings: list[str]
    details: dict[str, Any]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="model_inspect_compare",
        description=(
            "Compare multiple model variants (e.g., fp32/fp16/int8) and "
            "summarize quantization / architecture trade-offs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Basic quantization impact comparison
  python -m onnxruntime.tools.model_inspect_compare \\
    --models resnet_fp32.onnx resnet_fp16.onnx resnet_int8.onnx \\
    --eval-metrics eval_fp32.json eval_fp16.json eval_int8.json \\
    --baseline-precision fp32 \\
    --out-json quant_impact.json \\
    --out-md quant_impact.md
""",
    )

    parser.add_argument(
        "--models",
        type=Path,
        nargs="+",
        required=True,
        help="List of ONNX model paths to compare (e.g., fp32/fp16/int8 variants).",
    )
    parser.add_argument(
        "--eval-metrics",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "List of eval/perf metrics JSON files, one per model, "
            "produced by a batch-eval or perf script."
        ),
    )
    parser.add_argument(
        "--precisions",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional list of precision labels for each model "
            "(e.g., fp32 fp16 int8). If omitted, precisions are inferred "
            "from filenames where possible."
        ),
    )
    parser.add_argument(
        "--baseline-precision",
        type=str,
        default=None,
        help=(
            "Precision label to use as baseline for delta computation "
            "(e.g., fp32). If omitted, the first model is the baseline."
        ),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Output path for comparison JSON report.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Output path for human-readable Markdown summary.",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging noise; only errors are printed.",
    )
    parser.add_argument(
        "--with-charts",
        action="store_true",
        help="Generate accuracy vs speedup and memory savings charts (requires matplotlib).",
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=None,
        help="Directory for chart assets. Defaults to same directory as --out-md.",
    )
    parser.add_argument(
        "--out-html",
        type=Path,
        default=None,
        help="Output path for HTML comparison report with engine summary panel.",
    )
    parser.add_argument(
        "--out-pdf",
        type=Path,
        default=None,
        help="Output path for PDF comparison report (requires Playwright).",
    )

    args = parser.parse_args(argv)

    if len(args.models) != len(args.eval_metrics):
        parser.error(
            f"--models ({len(args.models)}) and --eval-metrics "
            f"({len(args.eval_metrics)}) must have the same length."
        )

    if args.precisions is not None and len(args.precisions) != len(args.models):
        parser.error(
            f"--precisions length ({len(args.precisions)}) must match "
            f"--models length ({len(args.models)})."
        )

    return args


def _infer_precision_from_name(path: Path) -> str | None:
    """Heuristic to infer precision from filename."""
    name = path.stem.lower()
    if "int8" in name or "qdq" in name or "int_8" in name:
        return "int8"
    if "int4" in name or "int_4" in name:
        return "int4"
    if "fp16" in name or "half" in name:
        return "fp16"
    if "bf16" in name:
        return "bf16"
    if "fp32" in name or "float32" in name:
        return "fp32"
    return None


def _load_eval_metrics(path: Path) -> dict[str, Any]:
    """
    Load eval/perf metrics JSON.

    We accept two common layouts:
      1) Root-level metrics dict:
         {"f1_macro": 0.93, "latency_ms_p50": 14.5, ...}
      2) Wrapped metrics:
         {"metrics": {...}}
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "metrics" in data and isinstance(data["metrics"], dict):
        return data["metrics"]

    if isinstance(data, dict):
        return data

    raise ValueError(f"Unsupported metrics JSON structure in {path}")


def _build_variant_inputs(args: argparse.Namespace) -> list[VariantInputs]:
    variants: list[VariantInputs] = []
    for idx, (model_path, metrics_path) in enumerate(
        zip(args.models, args.eval_metrics, strict=True)
    ):
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Eval metrics JSON not found: {metrics_path}")

        precision: str | None = None
        if args.precisions is not None:
            precision = args.precisions[idx]
        else:
            precision = _infer_precision_from_name(model_path)

        variants.append(
            VariantInputs(
                model_path=model_path,
                eval_metrics_path=metrics_path,
                precision=precision,
            )
        )
    return variants


def _run_inspection(model_path: Path, logger: logging.Logger) -> Any:
    """
    Run ModelInspector for a single model and return an InspectionReport.

    We deliberately reuse the same inspector class used by model_inspect.py.
    """
    inspector = ModelInspector(logger=logger)
    report = inspector.inspect(str(model_path))
    return report


def _determine_baseline_index(
    variants: Sequence[VariantReport], baseline_precision: str | None
) -> int:
    if baseline_precision is None:
        return 0

    for idx, v in enumerate(variants):
        if v.precision.lower() == baseline_precision.lower():
            return idx

    # Fallback to first if requested precision not found
    LOGGER.warning(
        "Requested baseline precision '%s' not found among variants; "
        "using first variant as baseline.",
        baseline_precision,
    )
    return 0


def _check_architecture_compatibility(
    variants: Sequence[VariantReport],
) -> ArchCompatResult:
    """
    Check if all model variants share the same architecture.

    Task 6.3.3: Verify architecture compatibility
    - Same architecture_type (cnn, transformer, mlp, etc.)
    - Compatible input/output shapes
    - Similar detected block patterns
    """
    warnings: list[str] = []
    details: dict[str, Any] = {}

    if len(variants) < 2:
        return ArchCompatResult(is_compatible=True, warnings=[], details={})

    baseline = variants[0]
    baseline_report = baseline.report

    # Extract baseline characteristics
    baseline_arch_type = getattr(baseline_report, "architecture_type", "unknown")
    baseline_graph = getattr(baseline_report, "graph_summary", None)
    baseline_blocks = getattr(baseline_report, "detected_blocks", [])

    # Get baseline I/O shapes
    baseline_inputs: dict[str, Any] = {}
    baseline_outputs: dict[str, Any] = {}
    if baseline_graph:
        baseline_inputs = getattr(baseline_graph, "input_shapes", {}) or {}
        baseline_outputs = getattr(baseline_graph, "output_shapes", {}) or {}

    details["baseline_architecture"] = baseline_arch_type
    details["baseline_inputs"] = baseline_inputs
    details["baseline_outputs"] = baseline_outputs

    # Count block types in baseline
    baseline_block_counts: dict[str, int] = {}
    for block in baseline_blocks:
        block_type = getattr(block, "block_type", "unknown")
        baseline_block_counts[block_type] = baseline_block_counts.get(block_type, 0) + 1

    details["baseline_block_counts"] = baseline_block_counts

    # Check each variant against baseline
    is_compatible = True
    for idx, variant in enumerate(variants[1:], start=1):
        v_report = variant.report
        v_arch_type = getattr(v_report, "architecture_type", "unknown")
        v_graph = getattr(v_report, "graph_summary", None)
        v_blocks = getattr(v_report, "detected_blocks", [])

        # Check architecture type
        if v_arch_type != baseline_arch_type:
            warnings.append(
                f"Variant {idx} ({variant.precision}): architecture_type mismatch "
                f"({v_arch_type} vs baseline {baseline_arch_type})"
            )
            is_compatible = False

        # Check input shapes (flexible: same keys, same ranks)
        if v_graph:
            v_inputs = getattr(v_graph, "input_shapes", {}) or {}
            v_outputs = getattr(v_graph, "output_shapes", {}) or {}

            # Compare input count and ranks
            if len(v_inputs) != len(baseline_inputs):
                warnings.append(
                    f"Variant {idx} ({variant.precision}): different number of inputs "
                    f"({len(v_inputs)} vs baseline {len(baseline_inputs)})"
                )

            # Compare output count
            if len(v_outputs) != len(baseline_outputs):
                warnings.append(
                    f"Variant {idx} ({variant.precision}): different number of outputs "
                    f"({len(v_outputs)} vs baseline {len(baseline_outputs)})"
                )

        # Check block pattern counts (some variation is OK for quantization)
        v_block_counts: dict[str, int] = {}
        for block in v_blocks:
            block_type = getattr(block, "block_type", "unknown")
            v_block_counts[block_type] = v_block_counts.get(block_type, 0) + 1

        # Major block type differences are a warning
        for block_type, count in baseline_block_counts.items():
            v_count = v_block_counts.get(block_type, 0)
            # Allow some variation but flag major differences
            if abs(v_count - count) > max(count * 0.2, 2):
                warnings.append(
                    f"Variant {idx} ({variant.precision}): block count mismatch for "
                    f"{block_type} ({v_count} vs baseline {count})"
                )

    return ArchCompatResult(
        is_compatible=is_compatible,
        warnings=warnings,
        details=details,
    )


def _get_numeric_metric(report: Any, *path: str) -> float | None:
    """Safely extract a numeric metric from a nested report structure."""
    obj = report
    for key in path:
        if obj is None:
            return None
        if hasattr(obj, key):
            obj = getattr(obj, key)
        elif isinstance(obj, dict) and key in obj:
            obj = obj[key]
        else:
            return None
    if isinstance(obj, (int, float)):
        return float(obj)
    return None


def _compute_deltas(baseline: VariantReport, other: VariantReport) -> dict[str, Any]:
    """
    Compute deltas between baseline and another variant.

    Task 6.3.4: Compute comprehensive deltas including:
    - File size
    - Parameter counts
    - FLOPs
    - Memory estimates
    - Hardware estimates (if available)
    - Eval/perf metrics
    """
    deltas: dict[str, Any] = {}

    # Model file size
    deltas["size_bytes"] = other.size_bytes - baseline.size_bytes

    # Parameter counts
    base_params = _get_numeric_metric(baseline.report, "param_counts", "total")
    other_params = _get_numeric_metric(other.report, "param_counts", "total")
    if base_params is not None and other_params is not None:
        deltas["total_params"] = int(other_params - base_params)

    # FLOPs
    base_flops = _get_numeric_metric(baseline.report, "flop_counts", "total")
    other_flops = _get_numeric_metric(other.report, "flop_counts", "total")
    if base_flops is not None and other_flops is not None:
        deltas["total_flops"] = int(other_flops - base_flops)

    # Memory estimates (model size)
    base_mem = _get_numeric_metric(baseline.report, "memory_estimates", "model_size_bytes")
    other_mem = _get_numeric_metric(other.report, "memory_estimates", "model_size_bytes")
    if base_mem is not None and other_mem is not None:
        deltas["memory_bytes"] = int(other_mem - base_mem)

    # Peak activation memory
    base_peak = _get_numeric_metric(baseline.report, "memory_estimates", "peak_activation_bytes")
    other_peak = _get_numeric_metric(other.report, "memory_estimates", "peak_activation_bytes")
    if base_peak is not None and other_peak is not None:
        deltas["peak_activation_bytes"] = int(other_peak - base_peak)

    # Hardware estimates (if available)
    base_hw = getattr(baseline.report, "hardware_estimates", None)
    other_hw = getattr(other.report, "hardware_estimates", None)
    if base_hw is not None and other_hw is not None:
        # Latency
        base_lat = _get_numeric_metric(base_hw, "estimated_latency_ms")
        other_lat = _get_numeric_metric(other_hw, "estimated_latency_ms")
        if base_lat is not None and other_lat is not None:
            deltas["latency_ms"] = other_lat - base_lat

        # VRAM
        base_vram = _get_numeric_metric(base_hw, "vram_required_bytes")
        other_vram = _get_numeric_metric(other_hw, "vram_required_bytes")
        if base_vram is not None and other_vram is not None:
            deltas["vram_required_bytes"] = int(other_vram - base_vram)

        # Compute utilization
        base_util = _get_numeric_metric(base_hw, "compute_utilization")
        other_util = _get_numeric_metric(other_hw, "compute_utilization")
        if base_util is not None and other_util is not None:
            deltas["compute_utilization"] = other_util - base_util

    # Metric-wise deltas from eval/perf JSON (only for overlapping numeric fields)
    for key, base_val in baseline.metrics.items():
        other_val = other.metrics.get(key)
        if isinstance(base_val, (int, float)) and isinstance(other_val, (int, float)):
            # Don't overwrite structural deltas we already computed
            if key not in deltas:
                deltas[key] = other_val - base_val

    return deltas


def _build_compare_json(
    variants: Sequence[VariantReport],
    baseline_index: int,
    arch_compat: ArchCompatResult,
) -> dict[str, Any]:
    baseline = variants[baseline_index]

    # Derive a simple model_family_id from baseline metadata
    report = baseline.report
    metadata = getattr(report, "metadata", None)
    if metadata and getattr(metadata, "name", None):
        model_family_id = metadata.name
    else:
        model_family_id = baseline.model_path.stem

    out: dict[str, Any] = {
        "model_family_id": model_family_id,
        "baseline_precision": baseline.precision,
        "architecture_compatible": arch_compat.is_compatible,
        "compatibility_warnings": arch_compat.warnings,
        "variants": [],
    }

    for idx, v in enumerate(variants):
        deltas_vs_baseline: dict[str, Any] | None
        if idx == baseline_index:
            deltas_vs_baseline = None
        else:
            deltas_vs_baseline = _compute_deltas(baseline, v)

        hw_estimates = getattr(v.report, "hardware_estimates", None)
        hw_profile = getattr(v.report, "hardware_profile", None)

        # Extract key metrics from inspection report for the variant summary
        param_counts = getattr(v.report, "param_counts", None)
        flop_counts = getattr(v.report, "flop_counts", None)
        memory_estimates = getattr(v.report, "memory_estimates", None)

        out_variant: dict[str, Any] = {
            "precision": v.precision,
            "quantization_scheme": v.quantization_scheme,
            "model_path": str(v.model_path),
            "size_bytes": int(v.size_bytes),
            # Structural metrics from inspection
            "total_params": (param_counts.total if param_counts is not None else None),
            "total_flops": (flop_counts.total if flop_counts is not None else None),
            "memory_bytes": (
                memory_estimates.model_size_bytes if memory_estimates is not None else None
            ),
            # Eval/perf metrics from JSON
            "metrics": v.metrics,
            "hardware_estimates": (
                hw_estimates.to_dict()
                if hw_estimates is not None and hasattr(hw_estimates, "to_dict")
                else None
            ),
            "hardware_profile": (
                hw_profile.to_dict()
                if hw_profile is not None and hasattr(hw_profile, "to_dict")
                else None
            ),
            "deltas_vs_baseline": deltas_vs_baseline,
        }
        out["variants"].append(out_variant)

    return out


def _format_number(n: float | None, suffix: str = "") -> str:
    """Format a number with K/M/G suffixes for readability."""
    if n is None:
        return "-"
    if abs(n) >= 1e9:
        return f"{n / 1e9:.2f}G{suffix}"
    if abs(n) >= 1e6:
        return f"{n / 1e6:.2f}M{suffix}"
    if abs(n) >= 1e3:
        return f"{n / 1e3:.2f}K{suffix}"
    if isinstance(n, float) and not n.is_integer():
        return f"{n:.2f}{suffix}"
    return f"{int(n)}{suffix}"


def _format_delta(val: float | None, suffix: str = "") -> str:
    """Format a delta with +/- prefix and K/M/G suffixes."""
    if val is None:
        return "-"
    sign = "+" if val >= 0 else ""
    if abs(val) >= 1e9:
        return f"{sign}{val / 1e9:.2f}G{suffix}"
    if abs(val) >= 1e6:
        return f"{sign}{val / 1e6:.2f}M{suffix}"
    if abs(val) >= 1e3:
        return f"{sign}{val / 1e3:.2f}K{suffix}"
    if isinstance(val, float) and not val.is_integer():
        return f"{sign}{val:.2f}{suffix}"
    return f"{sign}{int(val)}{suffix}"


def _build_markdown_summary(
    compare_json: dict[str, Any],
) -> str:
    """Generate a Markdown summary for compare mode with rich metrics."""
    lines: list[str] = []

    model_family_id = compare_json.get("model_family_id", "unknown_model")
    baseline_precision = compare_json.get("baseline_precision", "unknown")
    arch_compatible = compare_json.get("architecture_compatible", True)
    warnings = compare_json.get("compatibility_warnings", [])

    lines.append(f"# Quantization Impact: {model_family_id}")
    lines.append("")
    lines.append(
        f"Baseline precision: **{baseline_precision}**  (deltas are relative to this variant)."
    )
    lines.append("")

    # Architecture compatibility notice
    if not arch_compatible:
        lines.append("## Compatibility Warnings")
        lines.append("")
        lines.append(
            "> **Warning**: Model variants may not be directly comparable due to "
            "architecture differences."
        )
        lines.append("")
        for warn in warnings:
            lines.append(f"- {warn}")
        lines.append("")

    # Summary table with comprehensive metrics
    lines.append("## Variant Comparison")
    lines.append("")
    lines.append("| Precision | Size | Params | FLOPs | Δ Size | Δ Params | Δ FLOPs |")
    lines.append("|-----------|------|--------|-------|--------|----------|---------|")

    for v in compare_json.get("variants", []):
        precision = v.get("precision", "unknown")
        size_bytes = v.get("size_bytes", 0)
        total_params = v.get("total_params")
        total_flops = v.get("total_flops")
        deltas = v.get("deltas_vs_baseline")

        # Format absolute values
        size_str = _format_number(size_bytes, "B")
        params_str = _format_number(total_params)
        flops_str = _format_number(total_flops)

        # Format deltas
        if deltas is None:
            delta_size = "-"
            delta_params = "-"
            delta_flops = "-"
        else:
            delta_size = _format_delta(deltas.get("size_bytes"), "B")
            delta_params = _format_delta(deltas.get("total_params"))
            delta_flops = _format_delta(deltas.get("total_flops"))

        lines.append(
            f"| {precision} | {size_str} | {params_str} | {flops_str} | "
            f"{delta_size} | {delta_params} | {delta_flops} |"
        )

    lines.append("")

    # Performance metrics table (if available in eval metrics)
    has_perf_metrics = False
    for v in compare_json.get("variants", []):
        metrics = v.get("metrics", {})
        if any(k in metrics for k in ["latency_ms_p50", "throughput_qps", "f1_macro", "accuracy"]):
            has_perf_metrics = True
            break

    if has_perf_metrics:
        lines.append("## Performance Metrics")
        lines.append("")
        lines.append(
            "| Precision | Latency (ms) | Throughput | Accuracy | Δ Latency | Δ Accuracy |"
        )
        lines.append(
            "|-----------|--------------|------------|----------|-----------|------------|"
        )

        for v in compare_json.get("variants", []):
            precision = v.get("precision", "unknown")
            metrics = v.get("metrics", {})
            deltas = v.get("deltas_vs_baseline")

            latency = metrics.get("latency_ms_p50") or metrics.get("latency_ms")
            throughput = metrics.get("throughput_qps") or metrics.get("throughput")
            accuracy = metrics.get("f1_macro") or metrics.get("accuracy")

            lat_str = f"{latency:.2f}" if latency is not None else "-"
            tput_str = _format_number(throughput, " qps") if throughput is not None else "-"
            acc_str = f"{accuracy:.4f}" if accuracy is not None else "-"

            if deltas is None:
                delta_lat = "-"
                delta_acc = "-"
            else:
                d_lat = deltas.get("latency_ms_p50") or deltas.get("latency_ms")
                d_acc = deltas.get("f1_macro") or deltas.get("accuracy")
                delta_lat = f"{d_lat:+.2f}ms" if d_lat is not None else "-"
                delta_acc = f"{d_acc:+.4f}" if d_acc is not None else "-"

            lines.append(
                f"| {precision} | {lat_str} | {tput_str} | {acc_str} | {delta_lat} | {delta_acc} |"
            )

        lines.append("")

    lines.append("> Full details including hardware estimates are in the JSON report.")
    lines.append("")

    return "\n".join(lines)


def _build_variants(
    variant_inputs: Sequence[VariantInputs],
    logger: logging.Logger,
) -> list[VariantReport]:
    variants: list[VariantReport] = []

    for v in variant_inputs:
        metrics = _load_eval_metrics(v.eval_metrics_path)
        report = _run_inspection(v.model_path, logger=logger)
        size_bytes = v.model_path.stat().st_size

        precision = v.precision or "unknown"
        quant_scheme = precision if precision != "fp32" else "none"

        variants.append(
            VariantReport(
                model_path=v.model_path,
                precision=precision,
                quantization_scheme=quant_scheme,
                size_bytes=size_bytes,
                report=report,
                metrics=metrics,
            )
        )

    return variants


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    logger = LOGGER

    try:
        variant_inputs = _build_variant_inputs(args)
        variants = _build_variants(variant_inputs, logger=logger)

        baseline_index = _determine_baseline_index(variants, args.baseline_precision)

        # Task 6.3.3: Check architecture compatibility
        arch_compat = _check_architecture_compatibility(variants)
        if not arch_compat.is_compatible:
            logger.warning(
                "Model variants have architecture differences; comparison may not be meaningful."
            )
            for warn in arch_compat.warnings:
                logger.warning("  - %s", warn)

        compare_json = _build_compare_json(variants, baseline_index, arch_compat)

        # Write JSON if requested
        if args.out_json:
            args.out_json.parent.mkdir(parents=True, exist_ok=True)
            with args.out_json.open("w", encoding="utf-8") as f:
                json.dump(compare_json, f, indent=2)
            logger.info("Comparison JSON written to %s", args.out_json)

        # Write Markdown if requested
        if args.out_md:
            assets_dir = args.assets_dir or args.out_md.parent
            if args.with_charts:
                md = build_enhanced_markdown(
                    compare_json,
                    include_charts=True,
                    assets_dir=assets_dir,
                )
            else:
                md = _build_markdown_summary(compare_json)
            args.out_md.parent.mkdir(parents=True, exist_ok=True)
            args.out_md.write_text(md, encoding="utf-8")
            logger.info("Comparison Markdown written to %s", args.out_md)

            # Generate standalone charts if requested
            if args.with_charts and viz_available():
                points = compute_tradeoff_points(compare_json)
                if points:
                    chart_path = assets_dir / "tradeoff_chart.png"
                    generate_tradeoff_chart(points, chart_path)
                    logger.info("Tradeoff chart written to %s", chart_path)

                    mem_path = assets_dir / "memory_savings.png"
                    generate_memory_savings_chart(compare_json, mem_path)
                    logger.info("Memory savings chart written to %s", mem_path)

        # Write HTML if requested
        if args.out_html:
            generate_compare_html(
                compare_json,
                output_path=args.out_html,
                include_charts=True,
            )
            logger.info("Comparison HTML written to %s", args.out_html)

        # Write PDF if requested (Task 6.10.9)
        if args.out_pdf:
            pdf_path = generate_compare_pdf(
                compare_json,
                output_path=args.out_pdf,
                include_charts=True,
            )
            if pdf_path:
                logger.info("Comparison PDF written to %s", pdf_path)
            else:
                logger.warning("PDF generation failed (Playwright may not be installed)")

        if not args.out_json and not args.out_md and not args.out_html and not args.out_pdf:
            # Default to printing JSON to stdout if no outputs specified
            print(json.dumps(compare_json, indent=2))

        return 0

    except Exception as exc:  # pragma: no cover - top-level safety
        logger.error("Compare-mode failed: %s", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
