"""Extracted tab rendering functions for Streamlit app.

These functions are separated to:
1. Enable unit testing of data preparation logic
2. Provide a single source of truth (no duplicate tab code)
3. Make the main streamlit_app.py cleaner
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from haoline.format_adapters import FormatCapabilities
    from haoline.report import InspectionReport


# =============================================================================
# Format Capability Helpers (Tasks 49.2.4-7)
# =============================================================================


def get_capabilities_from_extension(file_ext: str) -> FormatCapabilities:
    """Get format capabilities from file extension.

    Args:
        file_ext: File extension including dot (e.g., ".onnx", ".gguf")

    Returns:
        FormatCapabilities for the format, or default capabilities if unknown.
    """
    from haoline.format_adapters import (
        FORMAT_CAPABILITIES,
        FormatCapabilities,
        SourceFormat,
    )

    ext_to_format = {
        ".onnx": SourceFormat.ONNX,
        ".pt": SourceFormat.PYTORCH,
        ".pth": SourceFormat.PYTORCH,
        ".pb": SourceFormat.TENSORFLOW,
        ".tflite": SourceFormat.TFLITE,
        ".mlmodel": SourceFormat.COREML,
        ".mlpackage": SourceFormat.COREML,
        ".engine": SourceFormat.TENSORRT,
        ".plan": SourceFormat.TENSORRT,
        ".gguf": SourceFormat.GGUF,
        ".safetensors": SourceFormat.SAFETENSORS,
        ".xml": SourceFormat.OPENVINO,
    }

    source_format = ext_to_format.get(file_ext.lower())
    if source_format:
        return FORMAT_CAPABILITIES.get(source_format, FormatCapabilities())
    return FormatCapabilities()


def render_format_tier_badge(capabilities: FormatCapabilities) -> str:
    """Render a format tier badge as HTML.

    Args:
        capabilities: FormatCapabilities for the current format.

    Returns:
        HTML string for the badge.
    """
    tier = capabilities.tier
    colors = {
        "Full": ("#10b981", "#065f46"),  # Green
        "Graph": ("#3b82f6", "#1e40af"),  # Blue
        "Metadata": ("#f59e0b", "#92400e"),  # Amber
        "Weights": ("#6b7280", "#374151"),  # Gray
    }
    bg, border = colors.get(tier, ("#6b7280", "#374151"))

    return f"""
    <span style="
        background: {bg}20;
        color: {bg};
        border: 1px solid {border};
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    ">{tier}</span>
    """


def render_feature_unavailable(
    feature_name: str,
    reason: str,
    upgrade_path: str | None = None,
) -> None:
    """Render a "Feature unavailable" message with optional upgrade path.

    Args:
        feature_name: Name of the unavailable feature.
        reason: Why the feature is unavailable.
        upgrade_path: Optional suggestion to enable the feature.
    """
    import streamlit as st

    st.info(f"**{feature_name}** is not available for this format.")
    st.caption(reason)
    if upgrade_path:
        st.markdown(f"💡 **Tip:** {upgrade_path}")


def format_number(n: float) -> str:
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


def format_bytes(b: float) -> str:
    """Format bytes with KB/MB/GB suffixes."""
    if b >= 1e12:
        return f"{b / 1e12:.2f} TB"
    if b >= 1e9:
        return f"{b / 1e9:.2f} GB"
    if b >= 1e6:
        return f"{b / 1e6:.2f} MB"
    if b >= 1e3:
        return f"{b / 1e3:.2f} KB"
    return f"{int(b)} B"


def generate_cli_command(
    model_name: str,
    hardware: str | None = None,
    batch_size: int | None = None,
    include_graph: bool = True,
    output_format: str = "html",
) -> str:
    """Generate equivalent CLI command for current analysis settings.

    Args:
        model_name: Name of the model file.
        hardware: Hardware profile name (e.g., 'rtx4090', 'auto').
        batch_size: Batch size for hardware estimates.
        include_graph: Whether to include interactive graph.
        output_format: Output format ('html', 'json', 'md').

    Returns:
        CLI command string that would replicate the web analysis.
    """
    # Use python -m haoline to avoid PATH issues on Windows/user installs
    parts = ["python -m haoline", model_name]

    # Hardware profile
    if hardware and hardware != "auto":
        parts.append(f"--hardware {hardware}")
    elif hardware == "auto":
        parts.append("--hardware auto")

    # Batch size (only if non-default)
    if batch_size and batch_size != 1:
        parts.append(f"--batch-size {batch_size}")

    # Output format
    output_name = model_name.replace(".onnx", "").replace(".pt", "").replace(".pth", "")
    if output_format == "html":
        parts.append(f"--out-html {output_name}_report.html")
        if include_graph:
            parts.append("--include-graph")
    elif output_format == "json":
        parts.append(f"--out-json {output_name}_report.json")
    elif output_format == "md":
        parts.append(f"--out-md {output_name}_report.md")

    return " \\\n  ".join(parts)


# =============================================================================
# Data preparation functions (testable, no Streamlit dependency)
# =============================================================================


def prepare_model_info_table(
    model_name: str,
    report: InspectionReport,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Prepare model info tables for Overview tab.

    Returns:
        Tuple of (properties_table, metrics_table) as list of dicts.
    """
    properties = [
        {"Property": "Model", "Value": f"`{model_name}`"},
        {"Property": "IR Version", "Value": str(report.metadata.ir_version)},
        {"Property": "Producer", "Value": report.metadata.producer_name or "Unknown"},
        {
            "Property": "Opset",
            "Value": str(
                list(report.metadata.opsets.values())[0] if report.metadata.opsets else "Unknown"
            ),
        },
    ]

    params_total = report.param_counts.total if report.param_counts else 0
    flops_total = report.flop_counts.total if report.flop_counts else 0
    peak_mem = report.memory_estimates.peak_activation_bytes if report.memory_estimates else 0
    model_size = report.memory_estimates.model_size_bytes if report.memory_estimates else 0

    metrics = [
        {"Metric": "Total Parameters", "Value": f"{params_total:,}"},
        {"Metric": "Total FLOPs", "Value": f"{flops_total:,}"},
        {"Metric": "Peak Memory", "Value": format_bytes(peak_mem)},
        {"Metric": "Model Size", "Value": format_bytes(model_size)},
    ]

    return properties, metrics


def prepare_op_distribution(
    report: InspectionReport,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """Prepare operator distribution data for charts.

    Returns:
        List of dicts with Operator and Count keys.
    """
    if not report.graph_summary or not report.graph_summary.op_type_counts:
        return []

    op_counts = report.graph_summary.op_type_counts
    sorted_ops = sorted(op_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [{"Operator": op, "Count": count} for op, count in sorted_ops]


def prepare_layer_table(
    layer_summary: Any,
    redact_names: bool = False,
    max_layers: int = 100,
) -> list[dict[str, Any]]:
    """Prepare layer breakdown table data.

    Args:
        layer_summary: LayerSummary object with layers list.
        redact_names: If True, replace layer names with generic "layer_N".
        max_layers: Maximum number of layers to include.

    Returns:
        List of dicts for DataFrame construction.
    """
    if not layer_summary or not layer_summary.layers:
        return []

    data = []
    for idx, layer in enumerate(layer_summary.layers[:max_layers]):
        if redact_names:
            display_name = f"layer_{idx}"
        else:
            display_name = layer.name[:30] + ("..." if len(layer.name) > 30 else "")

        data.append(
            {
                "Name": display_name,
                "Op Type": layer.op_type,
                "Parameters": format_number(layer.params) if layer.params > 0 else "-",
                "FLOPs": format_number(layer.flops) if layer.flops > 0 else "-",
                "Memory": format_bytes(layer.memory_bytes) if layer.memory_bytes > 0 else "-",
                "% Params": f"{layer.pct_params:.1f}%" if layer.pct_params > 0 else "-",
                "% FLOPs": f"{layer.pct_flops:.1f}%" if layer.pct_flops > 0 else "-",
                "Output Shape": ", ".join(layer.output_shapes) if layer.output_shapes else "-",
            }
        )

    return data


def prepare_quantization_data(
    lint_result: Any,
    advice: Any | None = None,
) -> dict[str, Any]:
    """Prepare quantization analysis data.

    Returns:
        Dict with score, warnings, unsupported_ops, sensitive_ops,
        recommendations, and layer_risks.
    """
    data: dict[str, Any] = {
        "score": lint_result.readiness_score,
        "warnings": list(lint_result.warnings) if lint_result.warnings else [],
        "unsupported_ops": sorted(lint_result.unsupported_ops)
        if lint_result.unsupported_ops
        else [],
        "sensitive_ops": sorted(lint_result.accuracy_sensitive_ops)
        if lint_result.accuracy_sensitive_ops
        else [],
        "recommendations": [],
        "layer_risks": [],
    }

    # Build recommendations from QuantizationAdvice fields (no 'recommendations' attr)
    if advice:
        recs = []
        # Strategy is the main recommendation
        if hasattr(advice, "strategy") and advice.strategy:
            recs.append(advice.strategy)
        # Add first few QAT workflow steps
        if hasattr(advice, "qat_workflow") and advice.qat_workflow:
            recs.extend(advice.qat_workflow[:2])
        data["recommendations"] = recs

    if lint_result.layer_risk_scores:
        data["layer_risks"] = [
            {
                "Layer": risk.layer_name,
                "Op": risk.op_type,
                "Risk": risk.risk_score,
                "Reason": risk.reason,
            }
            for risk in lint_result.layer_risk_scores
        ]

    return data


def prepare_details_data(report: InspectionReport) -> dict[str, Any]:
    """Prepare data for the Details tab.

    Returns:
        Dict with blocks, op_types, and risk_signals.
    """
    data: dict[str, Any] = {
        "blocks": [],
        "op_types": [],
        "risk_signals": [],
    }

    # Detected blocks
    if report.detected_blocks:
        for block in report.detected_blocks[:15]:
            block_data = {
                "name": block.name,
                "type": block.block_type,
                "node_count": len(block.nodes),
                "params": getattr(block, "params", None),
            }
            data["blocks"].append(block_data)

    # Op type breakdown
    if report.graph_summary and report.graph_summary.op_type_counts:
        op_counts = report.graph_summary.op_type_counts
        sorted_ops = sorted(op_counts.items(), key=lambda x: x[1], reverse=True)
        data["op_types"] = [{"Operator": op, "Count": count} for op, count in sorted_ops[:20]]

    # Risk signals
    if report.risk_signals:
        for risk in report.risk_signals:
            data["risk_signals"].append(
                {
                    "id": risk.id,
                    "severity": risk.severity,
                    "description": risk.description,
                }
            )

    return data


# =============================================================================
# Streamlit rendering functions (use prepared data)
# =============================================================================


def render_overview_tab(
    report: InspectionReport,
    model_name: str,
    capabilities: FormatCapabilities | None = None,
    model_path: str | Path | None = None,
) -> None:
    """Render the Overview tab content.

    Args:
        report: The inspection report.
        model_name: Display name for the model.
        capabilities: Format capabilities (optional, will be inferred from path).
        model_path: Path to the model file (used to infer capabilities).
    """
    import pandas as pd
    import streamlit as st

    # Infer capabilities if not provided
    if capabilities is None and model_path:
        capabilities = get_capabilities_from_extension(Path(model_path).suffix)

    # Task 49.2.6: Show format tier badge
    header_html = "### Model Information"
    if capabilities:
        badge = render_format_tier_badge(capabilities)
        header_html = f'<div style="display: flex; align-items: center; gap: 12px;"><h3 style="margin: 0;">Model Information</h3>{badge}</div>'
        st.markdown(header_html, unsafe_allow_html=True)
    else:
        st.markdown(header_html)

    properties, metrics = prepare_model_info_table(model_name, report)

    info_col1, info_col2 = st.columns(2)

    with info_col1:
        props_md = "| Property | Value |\n|----------|-------|\n"
        for p in properties:
            props_md += f"| **{p['Property']}** | {p['Value']} |\n"
        st.markdown(props_md)

    with info_col2:
        metrics_md = "| Metric | Value |\n|--------|-------|\n"
        for m in metrics:
            metrics_md += f"| **{m['Metric']}** | {m['Value']} |\n"
        st.markdown(metrics_md)

    # Op distribution chart
    op_data = prepare_op_distribution(report)
    if op_data:
        st.markdown("### Operator Distribution")
        df = pd.DataFrame(op_data)
        st.bar_chart(df.set_index("Operator"))


def render_graph_tab(
    report: InspectionReport,
    model_path: str | Path | None,
    model_name: str,
    file_size: int,
    graph_info: Any = None,
    capabilities: FormatCapabilities | None = None,
) -> None:
    """Render the Interactive Graph tab content.

    Args:
        report: The inspection report.
        model_path: Path to the model file.
        model_name: Display name for the model.
        file_size: Size of the model file in bytes.
        graph_info: Pre-loaded graph info (optional).
        capabilities: Format capabilities (optional, will be inferred from path).
    """
    import streamlit as st
    import streamlit.components.v1 as components

    # Infer capabilities if not provided
    if capabilities is None and model_path:
        capabilities = get_capabilities_from_extension(Path(model_path).suffix)

    # Task 49.2.4: Check if graph is available for this format
    if capabilities and not capabilities.has_graph:
        render_feature_unavailable(
            "Interactive Graph",
            capabilities.description or "This format does not include a computational graph.",
            "Convert to ONNX for full analysis with interactive graph visualization."
            if capabilities.can_convert_to_onnx
            else None,
        )
        return

    if not model_path or not Path(model_path).exists():
        st.info("Interactive graph not available - model file not cached.")
        _render_blocks_fallback(report)
        return

    st.markdown(
        """
        <p style="font-size: 0.85rem; color: #737373; margin-bottom: 1rem;">
        Scroll to zoom | Drag to pan | Click nodes to expand/collapse
        </p>
        """,
        unsafe_allow_html=True,
    )

    try:
        from haoline.analyzer import ONNXGraphLoader
        from haoline.edge_analysis import EdgeAnalyzer
        from haoline.hierarchical_graph import HierarchicalGraphBuilder
        from haoline.html_export import generate_html as generate_graph_html
        from haoline.patterns import PatternAnalyzer

        if graph_info is None:
            if Path(model_path).suffix.lower() != ".onnx":
                # Task 49.2.5: Show conversion prompt for non-ONNX formats
                render_feature_unavailable(
                    "Interactive Graph",
                    "Interactive graph visualization requires ONNX format.",
                    "Convert to ONNX for full analysis with graph visualization.",
                )
                return
            graph_loader = ONNXGraphLoader()
            _, graph_info = graph_loader.load(str(model_path))

        # Analyze patterns
        pattern_analyzer = PatternAnalyzer()
        blocks = pattern_analyzer.group_into_blocks(graph_info)

        # Edge analysis
        edge_analyzer = EdgeAnalyzer()
        edge_result = edge_analyzer.analyze(graph_info)

        # Build hierarchical graph
        builder = HierarchicalGraphBuilder()
        hier_graph = builder.build(
            graph_info,
            blocks,
            model_name.replace(".onnx", ""),
        )

        # Generate the full D3.js HTML
        graph_html = generate_graph_html(
            hier_graph,
            edge_result,
            title=model_name,
            model_size_bytes=file_size,
        )

        # Embed with generous height
        components.html(graph_html, height=800, scrolling=False)

    except Exception as e:
        st.warning(f"Could not generate interactive graph: {e}")
        _render_blocks_fallback(report)


def _render_blocks_fallback(report: InspectionReport) -> None:
    """Render block list as fallback when graph unavailable."""
    import streamlit as st

    if report.detected_blocks:
        st.markdown("#### Detected Architecture Blocks")
        for i, block in enumerate(report.detected_blocks[:15]):
            with st.expander(f"{block.block_type}: {block.name}", expanded=(i < 3)):
                st.write(f"**Type:** {block.block_type}")
                st.write(f"**Nodes:** {len(block.nodes)}")


def render_details_tab(report: InspectionReport) -> None:
    """Render the Details tab content."""
    import streamlit as st

    data = prepare_details_data(report)

    # Architecture blocks
    st.markdown("### Architecture Blocks")
    if data["blocks"]:
        for i, block in enumerate(data["blocks"]):
            with st.expander(f"{block['type']}: {block['name']}", expanded=(i < 3)):
                st.write(f"**Type:** {block['type']}")
                st.write(f"**Nodes:** {block['node_count']}")
                if block["params"]:
                    st.write(f"**Parameters:** {format_number(block['params'])}")
    else:
        st.info("No architecture blocks detected.")

    # Op type breakdown
    if data["op_types"]:
        st.markdown("### Operator Types")
        st.dataframe(data["op_types"], width="stretch")

    # Risk signals
    st.markdown("### Risk Signals")
    if data["risk_signals"]:
        for risk in data["risk_signals"]:
            severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(risk["severity"], "⚪")
            st.markdown(f"{severity_icon} **{risk['id']}** ({risk['severity']})")
            st.caption(risk["description"])
    else:
        st.success("No risk signals detected!")


def render_layer_details_tab(
    report: InspectionReport,
    graph_info: Any,
    model_name: str,
    redact_names: bool = False,
    summary_only: bool = False,
) -> None:
    """Render the Layer Details tab content."""
    import pandas as pd
    import streamlit as st

    st.markdown("### Layer Details")

    if summary_only:
        st.info("**Summary Only mode enabled** - Per-layer details hidden for IP protection.")
        _render_aggregate_stats(report)
        return

    if graph_info is None:
        st.info(
            "Layer table is available for ONNX models. Upload an ONNX model to view per-layer metrics."
        )
        return

    try:
        from haoline.layer_summary import LayerSummaryBuilder

        builder = LayerSummaryBuilder()
        layer_summary = builder.build(
            graph_info,
            param_counts=report.param_counts,
            flop_counts=report.flop_counts,
            memory_estimates=report.memory_estimates,
        )

        layer_data = prepare_layer_table(layer_summary, redact_names=redact_names)

        if layer_data:
            df = pd.DataFrame(layer_data)

            # Simple search/filter
            filter_text = st.text_input("Filter by name or op type", "", key="layer_filter")
            if filter_text:
                mask = df["Name"].str.contains(filter_text, case=False, na=False) | df[
                    "Op Type"
                ].str.contains(filter_text, case=False, na=False)
                df_filtered = df[mask]
            else:
                df_filtered = df

            st.dataframe(df_filtered, width="stretch", hide_index=True, height=400)

            # Download buttons
            csv_data = layer_summary.to_csv()
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    "Download Layer CSV",
                    data=csv_data,
                    file_name=f"{model_name.replace('.onnx', '')}_layers.csv",
                    mime="text/csv",
                )
            with col_dl2:
                json_bytes = df.to_json(orient="records", indent=2).encode("utf-8")
                st.download_button(
                    "Download Layer JSON",
                    data=json_bytes,
                    file_name=f"{model_name.replace('.onnx', '')}_layers.json",
                    mime="application/json",
                )
        else:
            st.info("No layer details available.")

    except Exception as e:
        st.warning(f"Could not generate layer table: {e}")


def _render_aggregate_stats(report: InspectionReport) -> None:
    """Render aggregate statistics when summary_only is enabled."""
    import streamlit as st

    st.markdown("### Aggregate Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Total Parameters",
            format_number(report.param_counts.total) if report.param_counts else "N/A",
        )
    with col2:
        st.metric(
            "Total FLOPs",
            format_number(report.flop_counts.total) if report.flop_counts else "N/A",
        )
    with col3:
        st.metric(
            "Total Layers",
            str(report.graph_summary.num_nodes) if report.graph_summary else "N/A",
        )


def render_quantization_tab(
    report: InspectionReport,
    graph_info: Any,
    capabilities: FormatCapabilities | None = None,
    model_path: str | Path | None = None,
) -> None:
    """Render the Quantization tab content.

    Args:
        report: The inspection report.
        graph_info: Pre-loaded graph info (optional).
        capabilities: Format capabilities (optional, will be inferred from path).
        model_path: Path to the model file (used to infer capabilities).
    """
    import pandas as pd
    import streamlit as st

    st.markdown("### Quantization Readiness")

    # Infer capabilities if not provided
    if capabilities is None and model_path:
        capabilities = get_capabilities_from_extension(Path(model_path).suffix)

    # Task 49.2.7: Show unavailable message for formats without quantization lint
    if capabilities and not capabilities.has_quantization_lint:
        render_feature_unavailable(
            "Quantization Analysis",
            capabilities.description or "Quantization linting requires ONNX graph structure.",
            "Convert to ONNX for quantization readiness analysis."
            if capabilities.can_convert_to_onnx
            else None,
        )
        return

    if graph_info is None:
        st.info("Quantization lint is available for ONNX models. Upload an ONNX model to view.")
        return

    try:
        from haoline.quantization_advisor import advise_quantization
        from haoline.quantization_linter import QuantizationLinter

        linter = QuantizationLinter()
        lint_result = linter.lint(graph_info)

        # Get advice (heuristic only to avoid API key requirement)
        advice = advise_quantization(lint_result, graph_info, api_key=None, use_llm=False)

        quant_data = prepare_quantization_data(lint_result, advice)

        # Score card
        st.metric("Readiness Score", f"{quant_data['score']:.0f}")

        # Warnings
        if quant_data["warnings"]:
            st.markdown("#### Warnings")
            for w in quant_data["warnings"]:
                st.markdown(f"- {w}")

        # Unsupported ops
        if quant_data["unsupported_ops"]:
            st.markdown("#### Unsupported Ops")
            st.write(", ".join(quant_data["unsupported_ops"]))

        # Accuracy-sensitive ops
        if quant_data["sensitive_ops"]:
            st.markdown("#### Accuracy-Sensitive Ops")
            st.write(", ".join(quant_data["sensitive_ops"]))

        # Recommendations
        if quant_data["recommendations"]:
            st.markdown("#### Recommendations")
            for rec in quant_data["recommendations"]:
                st.markdown(f"- {rec}")

        # Per-layer sensitivity
        if quant_data["layer_risks"]:
            st.markdown("#### Layer Sensitivity")
            risk_df = pd.DataFrame(quant_data["layer_risks"])
            st.dataframe(risk_df, width="stretch", hide_index=True)

    except Exception as e:
        st.warning(f"Quantization lint not available: {e}")


def render_export_tab(
    report: InspectionReport,
    model_name: str,
    tmp_path: str | None = None,
    hardware: str | None = None,
    batch_size: int | None = None,
) -> None:
    """Render the Export tab content.

    Args:
        report: The inspection report to export.
        model_name: Name of the model file.
        tmp_path: Path to temp model file (for advanced exports).
        hardware: Hardware profile used for analysis.
        batch_size: Batch size used for hardware estimates.
    """
    import streamlit as st

    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h3 style="color: #f5f5f5; margin-bottom: 0.25rem;">Export Reports</h3>
            <p style="color: #737373; font-size: 0.9rem; margin: 0;">
                Download your analysis in various formats
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Generate export data
    json_data = report.to_json()
    md_data = report.to_markdown()
    html_data = report.to_html()

    # Custom styled export grid CSS
    st.markdown(
        """
        <style>
            .export-card {
                background: #1a1a1a;
                border: 1px solid rgba(255,255,255,0.1);
                border-radius: 12px;
                padding: 1.25rem;
                transition: all 0.2s ease;
                margin-bottom: 0.5rem;
            }
            .export-card:hover {
                border-color: #10b981;
                background: #1f1f1f;
            }
            .export-icon { font-size: 1.5rem; margin-bottom: 0.5rem; }
            .export-title {
                color: #f5f5f5;
                font-weight: 600;
                font-size: 1rem;
                margin-bottom: 0.25rem;
            }
            .export-desc {
                color: #737373;
                font-size: 0.8rem;
                line-height: 1.4;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="export-card">
                <div class="export-icon">📊</div>
                <div class="export-title">HTML Report</div>
                <div class="export-desc">Interactive report with D3.js graph visualization</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            label="Download HTML",
            data=html_data,
            file_name=f"{model_name}_report.html",
            mime="text/html",
            width="stretch",
        )

    with col2:
        st.markdown(
            """
            <div class="export-card">
                <div class="export-icon">📄</div>
                <div class="export-title">JSON Data</div>
                <div class="export-desc">Raw analysis data for programmatic use</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name=f"{model_name}_report.json",
            mime="application/json",
            width="stretch",
        )

    col3, col4 = st.columns(2)

    with col3:
        st.markdown(
            """
            <div class="export-card">
                <div class="export-icon">📝</div>
                <div class="export-title">Markdown</div>
                <div class="export-desc">Text report for docs, READMEs, or wikis</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            label="Download Markdown",
            data=md_data,
            file_name=f"{model_name}_report.md",
            mime="text/markdown",
            width="stretch",
        )

    with col4:
        # PDF export (if available)
        pdf_data = None
        try:
            from haoline.pdf_generator import PDFGenerator
            from haoline.pdf_generator import is_available as pdf_available

            if pdf_available():
                import tempfile as tf_pdf

                pdf_gen = PDFGenerator()
                with tf_pdf.NamedTemporaryFile(suffix=".pdf", delete=False) as pdf_tmp:
                    if pdf_gen.generate_from_html(html_data, Path(pdf_tmp.name)):
                        with open(pdf_tmp.name, "rb") as f:
                            pdf_data = f.read()
        except Exception:
            pass

        st.markdown(
            """
            <div class="export-card">
                <div class="export-icon">📑</div>
                <div class="export-title">PDF Report <span style="background:#dc2626;color:white;font-size:0.65rem;padding:1px 4px;border-radius:3px;">CLI Only</span></div>
                <div class="export-desc">Requires Playwright installation - use CLI locally</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if pdf_data:
            st.download_button(
                label="Download PDF",
                data=pdf_data,
                file_name=f"{model_name}_report.pdf",
                mime="application/pdf",
                width="stretch",
            )
        else:
            st.button("PDF unavailable", disabled=True, width="stretch")

    # Advanced exports
    if tmp_path and Path(tmp_path).exists():
        st.markdown("---")
        st.markdown("### Advanced Exports")

        ir_col1, ir_col2 = st.columns(2)

        with ir_col1:
            st.markdown(
                """
                <div class="export-card">
                    <div class="export-icon">🔄</div>
                    <div class="export-title">Universal IR</div>
                    <div class="export-desc">Format-agnostic graph representation (JSON)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            try:
                from haoline.format_adapters import load_model

                ir_graph = load_model(Path(tmp_path))
                ir_dict = ir_graph.model_dump(exclude={"tensors": {"__all__": {"data"}}})
                ir_json = json.dumps(ir_dict, indent=2, default=str)
                st.download_button(
                    label="Download Universal IR",
                    data=ir_json,
                    file_name=f"{model_name}_universal_ir.json",
                    mime="application/json",
                    width="stretch",
                )
            except Exception as e:
                st.button(
                    f"IR export failed: {str(e)[:30]}",
                    disabled=True,
                    width="stretch",
                )

        with ir_col2:
            st.markdown(
                """
                <div class="export-card">
                    <div class="export-icon">📊</div>
                    <div class="export-title">Graph DOT</div>
                    <div class="export-desc">Graphviz format for visualization tools</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            try:
                from haoline.format_adapters import load_model

                ir_graph = load_model(Path(tmp_path))
                dot_data = ir_graph.to_dot()
                st.download_button(
                    label="Download DOT Graph",
                    data=dot_data,
                    file_name=f"{model_name}_graph.dot",
                    mime="text/plain",
                    width="stretch",
                )
            except Exception as e:
                st.button(
                    f"DOT export failed: {str(e)[:30]}",
                    disabled=True,
                    width="stretch",
                )

    # CLI Command section (Task 41.7.4)
    st.markdown("---")
    st.markdown("### CLI Command")
    st.markdown(
        """
        <p style="color: #737373; font-size: 0.9rem; margin-bottom: 0.75rem;">
            Run this command locally to replicate the analysis with full features
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Generate CLI command for different output formats
    model_filename = f"{model_name}.onnx" if not model_name.endswith(".onnx") else model_name
    cli_html = generate_cli_command(
        model_filename,
        hardware=hardware,
        batch_size=batch_size,
        include_graph=True,
        output_format="html",
    )

    st.code(cli_html, language="bash")

    # Copy hint
    st.markdown(
        """
        <p style="color: #525252; font-size: 0.75rem; margin-top: 0.5rem;">
            Install HaoLine: <code>pip install haoline</code> &nbsp;|&nbsp;
            Full features: <code>pip install haoline[full]</code>
        </p>
        """,
        unsafe_allow_html=True,
    )
