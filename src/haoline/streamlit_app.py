# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
HaoLine Streamlit Web UI.

A web interface for analyzing neural network models without installing anything.
Upload an ONNX model, get instant architecture analysis with interactive visualizations.

Run locally:
    streamlit run streamlit_app.py

Deploy to HuggingFace Spaces or Streamlit Cloud for public access.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from pydantic import BaseModel, ConfigDict, computed_field

# Page config must be first Streamlit command
st.set_page_config(
    page_title="HaoLine - Model Inspector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


class AnalysisResult(BaseModel):
    """Stored analysis result for session history."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    timestamp: datetime
    report: Any  # InspectionReport
    file_size: int
    model_path: str | None = None  # Path to temp file for interactive graph

    @computed_field  # type: ignore[prop-decorator]
    @property
    def summary(self) -> str:
        """Get a brief summary for display."""
        params = self.report.param_counts.total if self.report.param_counts else 0
        flops = self.report.flop_counts.total if self.report.flop_counts else 0
        return f"{format_number(params)} params, {format_number(flops)} FLOPs"


def init_session_state():
    """Initialize session state for history and comparison."""
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []
    if "compare_models" not in st.session_state:
        st.session_state.compare_models = {"model_a": None, "model_b": None}
    if "current_mode" not in st.session_state:
        st.session_state.current_mode = "analyze"  # "analyze" or "compare"
    if "current_result" not in st.session_state:
        st.session_state.current_result = None  # Currently displayed analysis


def add_to_history(
    name: str, report: Any, file_size: int, model_path: str | None = None
) -> AnalysisResult:
    """Add an analysis result to session history."""
    result = AnalysisResult(
        name=name,
        timestamp=datetime.now(),
        report=report,
        file_size=file_size,
        model_path=model_path,
    )
    # Keep max 10 results, newest first
    st.session_state.analysis_history.insert(0, result)
    if len(st.session_state.analysis_history) > 10:
        st.session_state.analysis_history.pop()
    return result


# Import haoline after page config
import streamlit.components.v1 as components

from haoline import ModelInspector, __version__
from haoline.edge_analysis import EdgeAnalyzer
from haoline.hardware import (
    HARDWARE_PROFILES,
    HardwareEstimator,
    detect_local_hardware,
    get_profile,
)
from haoline.hierarchical_graph import HierarchicalGraphBuilder
from haoline.html_export import generate_html as generate_graph_html
from haoline.patterns import PatternAnalyzer
from haoline.streamlit_tabs import (
    render_details_tab,
    render_export_tab,
    render_graph_tab,
    render_layer_details_tab,
    render_overview_tab,
    render_quantization_tab,
)

# Demo models for quick start
DEMO_MODELS = {
    "MNIST": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12.onnx",
        "size": "26 KB",
        "desc": "Handwritten digits",
    },
    "MobileNetV2": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx",
        "size": "14 MB",
        "desc": "Mobile-optimized CNN",
    },
    "EfficientNet": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx",
        "size": "50 MB",
        "desc": "State-of-the-art efficiency",
    },
}

# Custom CSS - Sleek dark theme with mint/emerald accents
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Root variables for consistency */
    :root {
        --bg-primary: #0d0d0d;
        --bg-secondary: #161616;
        --bg-tertiary: #1f1f1f;
        --bg-card: #1a1a1a;
        --accent-primary: #10b981;
        --accent-secondary: #34d399;
        --accent-glow: rgba(16, 185, 129, 0.3);
        --text-primary: #f5f5f5;
        --text-secondary: #a3a3a3;
        --text-muted: #737373;
        --border-subtle: rgba(255, 255, 255, 0.08);
        --border-accent: rgba(16, 185, 129, 0.3);
    }

    /* Global app background */
    .stApp {
        background: var(--bg-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-subtle);
    }

    [data-testid="stSidebar"] > div {
        background: transparent !important;
    }

    /* Header styling - high specificity */
    .main-header, h1.main-header, .stApp h1.main-header {
        font-family: 'Inter', sans-serif !important;
        background: linear-gradient(135deg, #10b981 0%, #34d399 50%, #6ee7b7 100%) !important;
        -webkit-background-clip: text !important;
        background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        color: transparent !important;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0;
        letter-spacing: -0.03em;
    }

    .sub-header {
        text-align: center;
        color: var(--text-secondary);
        font-size: 1.1rem;
        font-weight: 400;
        margin-top: 0.5rem;
        margin-bottom: 2.5rem;
        letter-spacing: 0.02em;
    }

    /* Metric styling - high specificity */
    [data-testid="stMetricValue"],
    .stApp [data-testid="stMetricValue"],
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #10b981 !important;
        font-weight: 600 !important;
        font-size: 2rem !important;
    }

    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.75rem !important;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Text colors */
    .stMarkdown, .stText, p, span, label, li {
        color: var(--text-primary) !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }

    /* Sidebar section headers */
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--accent-primary) !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 1.5rem !important;
        margin-bottom: 0.75rem !important;
        font-weight: 600 !important;
    }

    /* Input fields */
    .stTextInput input, .stSelectbox > div > div {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
        transition: all 0.2s ease;
    }

    .stTextInput input:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 2px var(--accent-glow) !important;
    }

    /* Checkboxes */
    .stCheckbox label span {
        color: var(--text-primary) !important;
    }

    [data-testid="stCheckbox"] > label > div:first-child {
        background: var(--bg-tertiary) !important;
        border-color: var(--border-subtle) !important;
    }

    [data-testid="stCheckbox"][aria-checked="true"] > label > div:first-child {
        background: var(--accent-primary) !important;
        border-color: var(--accent-primary) !important;
    }

    /* Tabs - modern pill style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: var(--bg-tertiary);
        padding: 4px;
        border-radius: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 8px !important;
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        padding: 8px 16px !important;
        border: none !important;
    }

    .stTabs [aria-selected="true"] {
        background: var(--accent-primary) !important;
        color: var(--bg-primary) !important;
    }

    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        background: rgba(255, 255, 255, 0.05) !important;
        color: var(--text-primary) !important;
    }

    /* File uploader - clean dark style */
    [data-testid="stFileUploader"] {
        background: transparent !important;
    }

    [data-testid="stFileUploader"] section {
        background: var(--bg-secondary) !important;
        border: 2px dashed var(--border-accent) !important;
        border-radius: 16px !important;
        padding: 2.5rem 2rem !important;
        transition: all 0.3s ease;
    }

    [data-testid="stFileUploader"] section:hover {
        border-color: var(--accent-primary) !important;
        background: rgba(16, 185, 129, 0.05) !important;
    }

    [data-testid="stFileUploader"] section div,
    [data-testid="stFileUploader"] section span {
        color: var(--text-secondary) !important;
    }

    [data-testid="stFileUploader"] button {
        background: var(--accent-primary) !important;
        color: var(--bg-primary) !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease;
    }

    [data-testid="stFileUploader"] button:hover {
        background: var(--accent-secondary) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px var(--accent-glow);
    }

    /* Alerts - amber for warnings, mint for info */
    .stAlert {
        border-radius: 12px !important;
        border: none !important;
    }

    [data-testid="stNotificationContentWarning"] {
        background: rgba(251, 191, 36, 0.1) !important;
        border-left: 4px solid #fbbf24 !important;
    }

    [data-testid="stNotificationContentWarning"] p {
        color: #fcd34d !important;
    }

    [data-testid="stNotificationContentInfo"] {
        background: rgba(16, 185, 129, 0.1) !important;
        border-left: 4px solid var(--accent-primary) !important;
    }

    [data-testid="stNotificationContentInfo"] p {
        color: var(--accent-secondary) !important;
    }

    [data-testid="stNotificationContentError"] {
        background: rgba(239, 68, 68, 0.1) !important;
        border-left: 4px solid #ef4444 !important;
    }

    [data-testid="stNotificationContentError"] p {
        color: #fca5a5 !important;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background: var(--bg-tertiary) !important;
        border-radius: 8px !important;
        border: 1px solid var(--border-subtle) !important;
    }

    .streamlit-expanderHeader:hover {
        border-color: var(--accent-primary) !important;
    }

    /* Caption/muted text */
    .stCaption, small {
        color: var(--text-muted) !important;
    }

    /* Download buttons */
    .stDownloadButton button {
        background: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease;
    }

    .stDownloadButton button:hover {
        background: var(--accent-primary) !important;
        color: var(--bg-primary) !important;
        border-color: var(--accent-primary) !important;
    }

    /* Dividers */
    hr {
        border-color: var(--border-subtle) !important;
    }

    /* Code blocks */
    code {
        background: var(--bg-tertiary) !important;
        color: var(--accent-secondary) !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
    }

    /* Links */
    a {
        color: var(--accent-primary) !important;
    }

    a:hover {
        color: var(--accent-secondary) !important;
    }

    /* Uploaded file chip */
    [data-testid="stFileUploaderFile"] {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px !important;
    }

    [data-testid="stFileUploaderFile"] button {
        background: transparent !important;
        color: var(--text-secondary) !important;
    }

    [data-testid="stFileUploaderFile"] button:hover {
        color: #ef4444 !important;
        background: rgba(239, 68, 68, 0.1) !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: var(--accent-primary) !important;
    }

    /* Privacy notice */
    .privacy-notice {
        background: rgba(16, 185, 129, 0.08);
        border-left: 3px solid var(--accent-primary);
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
        color: var(--text-secondary);
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--bg-tertiary);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-muted);
    }
</style>
""",
    unsafe_allow_html=True,
)


# Helper functions (defined early for use in dataclasses)
def format_number(n: float) -> str:
    """Format large numbers with K/M/B suffixes."""
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.2f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.2f}K"
    else:
        return f"{n:.0f}"


def format_bytes(b: float) -> str:
    """Format bytes with KB/MB/GB suffixes."""
    if b >= 1e9:
        return f"{b / 1e9:.2f} GB"
    elif b >= 1e6:
        return f"{b / 1e6:.2f} MB"
    elif b >= 1e3:
        return f"{b / 1e3:.2f} KB"
    else:
        return f"{b:.0f} B"


def render_comparison_view(model_a: AnalysisResult, model_b: AnalysisResult):
    """Render side-by-side model comparison."""
    st.markdown("## Model Comparison")

    # Header with model names
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
        <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                    padding: 1rem 1.5rem; border-radius: 12px; text-align: center;">
            <div style="font-size: 0.75rem; color: rgba(255,255,255,0.7); text-transform: uppercase; letter-spacing: 0.1em;">
                Model A
            </div>
            <div style="font-size: 1.25rem; font-weight: 600; color: white; margin-top: 0.25rem;">
                {model_a.name}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div style="background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
                    padding: 1rem 1.5rem; border-radius: 12px; text-align: center;">
            <div style="font-size: 0.75rem; color: rgba(255,255,255,0.7); text-transform: uppercase; letter-spacing: 0.1em;">
                Model B
            </div>
            <div style="font-size: 1.25rem; font-weight: 600; color: white; margin-top: 0.25rem;">
                {model_b.name}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Metrics comparison
    st.markdown("### Key Metrics")

    # Get metrics
    params_a = model_a.report.param_counts.total if model_a.report.param_counts else 0
    params_b = model_b.report.param_counts.total if model_b.report.param_counts else 0
    flops_a = model_a.report.flop_counts.total if model_a.report.flop_counts else 0
    flops_b = model_b.report.flop_counts.total if model_b.report.flop_counts else 0
    mem_a = (
        model_a.report.memory_estimates.peak_activation_bytes
        if model_a.report.memory_estimates
        else 0
    )
    mem_b = (
        model_b.report.memory_estimates.peak_activation_bytes
        if model_b.report.memory_estimates
        else 0
    )
    ops_a = model_a.report.graph_summary.num_nodes
    ops_b = model_b.report.graph_summary.num_nodes

    # Calculate deltas
    def delta_str(a, b, is_bytes=False):
        if a == 0 and b == 0:
            return ""
        diff = b - a
        pct = (diff / a * 100) if a != 0 else 0
        sign = "+" if diff > 0 else ""
        if is_bytes:
            return f"{sign}{format_bytes(abs(diff))} ({sign}{pct:.1f}%)"
        return f"{sign}{format_number(abs(diff))} ({sign}{pct:.1f}%)"

    # Comparison table
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**Parameters**")
        st.markdown(f"🟢 A: **{format_number(params_a)}**")
        st.markdown(f"🟣 B: **{format_number(params_b)}**")
        if params_a != params_b:
            diff_pct = ((params_b - params_a) / params_a * 100) if params_a else 0
            color = "#ef4444" if diff_pct > 0 else "#10b981"
            st.markdown(
                f"<span style='color: {color};'>Δ {diff_pct:+.1f}%</span>", unsafe_allow_html=True
            )

    with col2:
        st.markdown("**FLOPs**")
        st.markdown(f"🟢 A: **{format_number(flops_a)}**")
        st.markdown(f"🟣 B: **{format_number(flops_b)}**")
        if flops_a != flops_b:
            diff_pct = ((flops_b - flops_a) / flops_a * 100) if flops_a else 0
            color = "#ef4444" if diff_pct > 0 else "#10b981"
            st.markdown(
                f"<span style='color: {color};'>Δ {diff_pct:+.1f}%</span>", unsafe_allow_html=True
            )

    with col3:
        st.markdown("**Peak Memory**")
        st.markdown(f"🟢 A: **{format_bytes(mem_a)}**")
        st.markdown(f"🟣 B: **{format_bytes(mem_b)}**")
        if mem_a != mem_b:
            diff_pct = ((mem_b - mem_a) / mem_a * 100) if mem_a else 0
            color = "#ef4444" if diff_pct > 0 else "#10b981"
            st.markdown(
                f"<span style='color: {color};'>Δ {diff_pct:+.1f}%</span>", unsafe_allow_html=True
            )

    with col4:
        st.markdown("**Operators**")
        st.markdown(f"🟢 A: **{ops_a}**")
        st.markdown(f"🟣 B: **{ops_b}**")
        if ops_a != ops_b:
            diff_pct = ((ops_b - ops_a) / ops_a * 100) if ops_a else 0
            color = "#ef4444" if diff_pct > 0 else "#10b981"
            st.markdown(
                f"<span style='color: {color};'>Δ {diff_pct:+.1f}%</span>", unsafe_allow_html=True
            )

    st.markdown("---")

    # Operator distribution comparison
    st.markdown("### Operator Distribution Comparison")

    # Merge operator counts
    ops_a_dict = model_a.report.graph_summary.op_type_counts or {}
    ops_b_dict = model_b.report.graph_summary.op_type_counts or {}
    all_ops = set(ops_a_dict.keys()) | set(ops_b_dict.keys())

    comparison_data = []
    for op in sorted(all_ops):
        count_a = ops_a_dict.get(op, 0)
        count_b = ops_b_dict.get(op, 0)
        comparison_data.append(
            {
                "Operator": op,
                f"Model A ({model_a.name})": count_a,
                f"Model B ({model_b.name})": count_b,
                "Difference": count_b - count_a,
            }
        )

    df = pd.DataFrame(comparison_data)

    # Bar chart
    chart_df = df.set_index("Operator")[[f"Model A ({model_a.name})", f"Model B ({model_b.name})"]]
    st.bar_chart(chart_df)

    # Table
    with st.expander("View detailed comparison table"):
        st.dataframe(df, width="stretch")

    # Summary
    st.markdown("### Summary")

    # Auto-generate comparison summary
    summary_points = []

    if params_b < params_a:
        reduction = (1 - params_b / params_a) * 100 if params_a else 0
        summary_points.append(f"Model B has **{reduction:.1f}% fewer parameters** than Model A")
    elif params_b > params_a:
        increase = (params_b / params_a - 1) * 100 if params_a else 0
        summary_points.append(f"Model B has **{increase:.1f}% more parameters** than Model A")

    if flops_b < flops_a:
        reduction = (1 - flops_b / flops_a) * 100 if flops_a else 0
        summary_points.append(
            f"Model B requires **{reduction:.1f}% fewer FLOPs** (faster inference)"
        )
    elif flops_b > flops_a:
        increase = (flops_b / flops_a - 1) * 100 if flops_a else 0
        summary_points.append(f"Model B requires **{increase:.1f}% more FLOPs** (slower inference)")

    if mem_b < mem_a:
        reduction = (1 - mem_b / mem_a) * 100 if mem_a else 0
        summary_points.append(f"Model B uses **{reduction:.1f}% less memory**")
    elif mem_b > mem_a:
        increase = (mem_b / mem_a - 1) * 100 if mem_a else 0
        summary_points.append(f"Model B uses **{increase:.1f}% more memory**")

    if summary_points:
        for point in summary_points:
            st.markdown(f"- {point}")
    else:
        st.info("Models have similar characteristics.")

    # Quantization Recommendations (Task 33.4.11)
    # Detect if comparing FP32 vs INT8/quantized models
    is_a_quantized = model_a.report.param_counts and model_a.report.param_counts.is_quantized
    is_b_quantized = model_b.report.param_counts and model_b.report.param_counts.is_quantized

    # Show quantization analysis if one is FP32 and one is INT8
    if is_a_quantized != is_b_quantized:
        st.markdown("---")
        st.markdown("### Quantization Analysis")

        if is_b_quantized and not is_a_quantized:
            fp32_model, int8_model = model_a, model_b
            fp32_label, int8_label = "A (FP32)", "B (INT8)"
        else:
            fp32_model, int8_model = model_b, model_a
            fp32_label, int8_label = "B (FP32)", "A (INT8)"

        st.info(f"Comparing {fp32_label} against {int8_label} quantized version")

        # Calculate savings

        mem_fp32 = (
            fp32_model.report.memory_estimates.model_size_bytes
            if fp32_model.report.memory_estimates
            else 0
        )
        mem_int8 = (
            int8_model.report.memory_estimates.model_size_bytes
            if int8_model.report.memory_estimates
            else 0
        )

        # Size reduction metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            if mem_fp32 > 0:
                size_reduction = (1 - mem_int8 / mem_fp32) * 100
                st.metric(
                    "Size Reduction",
                    f"{size_reduction:.1f}%",
                    delta=f"-{format_bytes(mem_fp32 - mem_int8)}",
                )
        with col2:
            st.metric(
                "FP32 Size",
                format_bytes(mem_fp32),
            )
        with col3:
            st.metric(
                "INT8 Size",
                format_bytes(mem_int8),
            )

        # Run quantization analysis on FP32 model to show what to watch
        try:
            from haoline.analyzer import ONNXGraphLoader
            from haoline.quantization_advisor import advise_quantization
            from haoline.quantization_linter import QuantizationLinter

            # Check if we can load the FP32 model for analysis
            if hasattr(fp32_model, "path") and fp32_model.path:
                with st.expander("Quantization Recommendations", expanded=True):
                    try:
                        graph_loader = ONNXGraphLoader()
                        _, graph_info = graph_loader.load(fp32_model.path)
                        linter = QuantizationLinter()
                        lint_result = linter.lint(graph_info)

                        # Get advice
                        advice = advise_quantization(lint_result, graph_info, use_llm=False)

                        # Display
                        st.markdown(
                            f"**FP32 Model Readiness Score:** {lint_result.readiness_score}/100"
                        )
                        st.markdown(f"> {advice.strategy}")

                        if advice.op_substitutions:
                            st.markdown("**Recommended optimizations before quantization:**")
                            for sub in advice.op_substitutions:
                                st.markdown(
                                    f"- Replace `{sub.original_op}` with `{sub.replacement_op}`"
                                )

                        if lint_result.warnings:
                            st.warning(
                                f"FP32 model has {len(lint_result.warnings)} quantization-related warnings. "
                                "Review these if accuracy degradation is observed."
                            )

                    except Exception as e:
                        st.caption(f"Could not analyze FP32 model: {e}")
        except ImportError:
            pass  # Quantization advisor not available

    # Also show if both models might benefit from quantization (neither is quantized)
    elif not is_a_quantized and not is_b_quantized:
        with st.expander("Quantization Potential"):
            st.markdown("Neither model is quantized. Consider INT8 quantization for:")
            st.markdown("- **2-4x smaller model size**")
            st.markdown("- **2-4x faster inference** (with INT8-optimized hardware)")
            st.markdown("- **Lower memory bandwidth** requirements")
            st.markdown("")
            st.markdown(
                "Use the **Analyze** mode with **Quantization Analysis** enabled to assess readiness."
            )


def render_compare_mode():
    """Render the model comparison interface."""
    model_a = st.session_state.compare_models.get("model_a")
    model_b = st.session_state.compare_models.get("model_b")

    # Show comparison if both models are selected
    if model_a and model_b:
        # Clear selection buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Clear Comparison", type="secondary", width="stretch"):
                st.session_state.compare_models = {"model_a": None, "model_b": None}
                st.rerun()

        render_comparison_view(model_a, model_b)
        return

    # Model selection interface
    st.markdown("## Compare Two Models")
    st.markdown("Select models from your session history, or upload new models to compare.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.05) 100%);
                    border: 2px dashed rgba(16, 185, 129, 0.3); border-radius: 16px; padding: 2rem; text-align: center;">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🟢</div>
            <div style="font-size: 1rem; font-weight: 600; color: #10b981;">Model A</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        if model_a:
            st.success(f"Selected: **{model_a.name}**")
            st.caption(model_a.summary)
            if st.button("Clear Model A"):
                st.session_state.compare_models["model_a"] = None
                st.rerun()
        else:
            # Upload option
            file_a = st.file_uploader(
                "Upload Model A",
                type=["onnx"],
                key="compare_file_a",
                help="Upload an ONNX model",
            )
            if file_a:
                with st.spinner("Analyzing Model A..."):
                    result = analyze_model_file(file_a)
                    if result:
                        st.session_state.compare_models["model_a"] = result
                        st.rerun()

            # Or select from history
            if st.session_state.analysis_history:
                st.markdown("**Or select from history:**")
                for i, result in enumerate(st.session_state.analysis_history[:3]):
                    if st.button(f"{result.name}", key=f"select_a_{i}"):
                        st.session_state.compare_models["model_a"] = result
                        st.rerun()

    with col2:
        st.markdown(
            """
        <div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(79, 70, 229, 0.05) 100%);
                    border: 2px dashed rgba(99, 102, 241, 0.3); border-radius: 16px; padding: 2rem; text-align: center;">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🟣</div>
            <div style="font-size: 1rem; font-weight: 600; color: #6366f1;">Model B</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        if model_b:
            st.success(f"Selected: **{model_b.name}**")
            st.caption(model_b.summary)
            if st.button("Clear Model B"):
                st.session_state.compare_models["model_b"] = None
                st.rerun()
        else:
            # Upload option
            file_b = st.file_uploader(
                "Upload Model B",
                type=["onnx"],
                key="compare_file_b",
                help="Upload an ONNX model",
            )
            if file_b:
                with st.spinner("Analyzing Model B..."):
                    result = analyze_model_file(file_b)
                    if result:
                        st.session_state.compare_models["model_b"] = result
                        st.rerun()

            # Or select from history
            if st.session_state.analysis_history:
                st.markdown("**Or select from history:**")
                for i, result in enumerate(st.session_state.analysis_history[:3]):
                    if st.button(f"{result.name}", key=f"select_b_{i}"):
                        st.session_state.compare_models["model_b"] = result
                        st.rerun()

    # Tips
    if not st.session_state.analysis_history:
        st.info(
            "💡 **Tip:** First analyze some models in **Analyze** mode. They'll appear in your session history for easy comparison."
        )


def _check_feature_availability() -> dict[str, dict[str, Any]]:
    """Check which features are available in the current environment."""
    features: dict[str, dict[str, Any]] = {}

    # Core ONNX analysis (always available)
    features["ONNX Analysis"] = {"available": True}
    features["Interactive Graph"] = {"available": True}
    features["Quantization Linting"] = {"available": True}

    # TensorRT (requires GPU + tensorrt package)
    try:
        from haoline.formats.tensorrt import is_available

        trt_available = is_available()
    except ImportError:
        trt_available = False
    features["TensorRT Analysis"] = {
        "available": trt_available,
        "requires_gpu": True,
    }

    # Runtime Profiling (requires GPU + onnxruntime-gpu)
    try:
        import onnxruntime

        has_gpu_ep = "CUDAExecutionProvider" in onnxruntime.get_available_providers()
    except ImportError:
        has_gpu_ep = False
    features["GPU Inference"] = {
        "available": has_gpu_ep,
        "requires_gpu": True,
    }

    # LLM Summary (requires openai)
    try:
        import openai  # noqa: F401

        llm_available = True
    except ImportError:
        llm_available = False
    features["AI Summary"] = {
        "available": llm_available,
        "requires_dep": "openai",
    }

    # PDF Export (requires playwright)
    try:
        import playwright  # noqa: F401

        pdf_available = True
    except ImportError:
        pdf_available = False
    features["PDF Export"] = {
        "available": pdf_available,
        "requires_dep": "playwright",
    }

    # PyTorch conversion (requires torch)
    try:
        import torch  # noqa: F401

        torch_available = True
    except ImportError:
        torch_available = False
    features["PyTorch Conversion"] = {
        "available": torch_available,
        "requires_dep": "torch",
    }

    return features


def _handle_tensorrt_streamlit(uploaded_file) -> None:
    """Handle TensorRT engine file analysis in Streamlit."""
    import tempfile

    # Check TensorRT availability
    try:
        from haoline.formats.tensorrt import (
            TRTEngineReader,
            analyze_quant_bottlenecks,
            format_bytes,
            is_available,
        )
    except ImportError:
        st.error("""
        **TensorRT support not installed.**

        Install with: `pip install haoline[tensorrt]`

        Note: Requires NVIDIA GPU and CUDA 12.x
        """)
        return

    if not is_available():
        st.warning("""
        **TensorRT requires NVIDIA GPU** — This feature is not available on HuggingFace Spaces free tier.

        **Options:**
        1. Use the CLI locally with a GPU:
           ```bash
           pip install haoline[tensorrt]
           haoline model.engine --quant-bottlenecks
           ```

        2. Use our AWS-hosted version (coming soon) for full GPU features.
        """)
        return

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".engine", delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        with st.spinner("Analyzing TensorRT engine..."):
            reader = TRTEngineReader(tmp_path)
            info = reader.read()
            bottleneck_analysis = analyze_quant_bottlenecks(info)

        # Display results
        st.markdown("## TensorRT Engine Analysis")

        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Layers", info.layer_count)
        with col2:
            fused_pct = int(info.fusion_ratio * 100)
            st.metric("Fused", f"{info.fused_layer_count} ({fused_pct}%)")
        with col3:
            st.metric("Device Memory", format_bytes(info.device_memory_bytes))
        with col4:
            quant_pct = int(bottleneck_analysis.quantization_ratio * 100)
            st.metric("INT8 Layers", f"{bottleneck_analysis.int8_layer_count} ({quant_pct}%)")

        # Engine info
        st.markdown("### Engine Overview")
        st.markdown(f"""
        | Property | Value |
        |----------|-------|
        | **TensorRT Version** | {info.trt_version} |
        | **Device** | {info.device_name} |
        | **Compute Capability** | SM {info.compute_capability[0]}.{info.compute_capability[1]} |
        | **Max Batch Size** | {info.builder_config.max_batch_size} |
        """)

        # Quantization Bottleneck Analysis (Task 22.8.7)
        st.markdown("### Quantization Bottleneck Analysis")

        # Precision breakdown with color-coded bar
        total = bottleneck_analysis.total_layer_count
        if total > 0:
            int8_pct = bottleneck_analysis.int8_layer_count / total
            fp16_pct = bottleneck_analysis.fp16_layer_count / total
            fp32_pct = bottleneck_analysis.fp32_layer_count / total

            # Visual bar
            st.markdown(
                f"""
                <div style="display: flex; height: 30px; border-radius: 5px; overflow: hidden; margin-bottom: 1rem;">
                    <div style="width: {int8_pct * 100}%; background: #10b981;" title="INT8: {bottleneck_analysis.int8_layer_count}"></div>
                    <div style="width: {fp16_pct * 100}%; background: #f59e0b;" title="FP16: {bottleneck_analysis.fp16_layer_count}"></div>
                    <div style="width: {fp32_pct * 100}%; background: #ef4444;" title="FP32: {bottleneck_analysis.fp32_layer_count}"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #a3a3a3;">
                    <span style="color: #10b981;">■ INT8: {bottleneck_analysis.int8_layer_count}</span>
                    <span style="color: #f59e0b;">■ FP16: {bottleneck_analysis.fp16_layer_count}</span>
                    <span style="color: #ef4444;">■ FP32: {bottleneck_analysis.fp32_layer_count}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Speedup potential
        if bottleneck_analysis.estimated_speedup_potential > 1.05:
            st.warning(
                f"**Estimated speedup potential:** {bottleneck_analysis.estimated_speedup_potential:.1f}x "
                f"— {int(bottleneck_analysis.fp32_fallback_ratio * 100)}% of layers are FP32 fallback"
            )

        # Bottleneck zones heatmap
        if bottleneck_analysis.bottleneck_zones:
            st.markdown("#### FP32 Bottleneck Zones")

            for zone in sorted(bottleneck_analysis.bottleneck_zones, key=lambda z: -z.layer_count)[
                :5
            ]:
                severity_color = {
                    "Critical": "#ef4444",
                    "High": "#f97316",
                    "Medium": "#f59e0b",
                    "Low": "#84cc16",
                }.get(zone.severity, "#a3a3a3")

                types_str = ", ".join(set(zone.layer_types[:3]))
                st.markdown(
                    f"""
                    <div style="background: #1a1a1a; border-left: 4px solid {severity_color}; padding: 0.5rem 1rem; margin-bottom: 0.5rem; border-radius: 0 4px 4px 0;">
                        <strong style="color: {severity_color};">[{zone.severity}]</strong>
                        <span style="color: #f5f5f5;">{zone.layer_count} consecutive FP32 layers</span>
                        <br><span style="color: #737373; font-size: 0.85rem;">Types: {types_str}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if len(bottleneck_analysis.bottleneck_zones) > 5:
                st.caption(f"... and {len(bottleneck_analysis.bottleneck_zones) - 5} more zones")

        # Failed fusions
        if bottleneck_analysis.failed_fusions:
            with st.expander(
                f"Failed Fusions ({len(bottleneck_analysis.failed_fusions)})", expanded=False
            ):
                for ff in bottleneck_analysis.failed_fusions[:10]:
                    impact_color = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#84cc16"}.get(
                        ff.speed_impact, "#a3a3a3"
                    )
                    st.markdown(
                        f"<span style='color: {impact_color};'>■</span> **{ff.pattern_type}**: {', '.join(ff.layer_names[:3])}",
                        unsafe_allow_html=True,
                    )

        # Recommendations
        if bottleneck_analysis.recommendations:
            st.markdown("#### Recommendations")
            for rec in bottleneck_analysis.recommendations:
                st.markdown(f"- {rec}")

        # Layer type distribution
        with st.expander("Layer Type Distribution", expanded=False):
            import pandas as pd

            layer_types = info.layer_type_counts
            df = pd.DataFrame(
                sorted(layer_types.items(), key=lambda x: -x[1]),
                columns=["Type", "Count"],
            )
            st.bar_chart(df.set_index("Type"))

    except Exception as e:
        st.error(f"Failed to analyze TensorRT engine: {e}")


def analyze_model_file(uploaded_file) -> AnalysisResult | None:
    """Analyze an uploaded model file and return the result."""
    from haoline import ModelInspector

    file_ext = Path(uploaded_file.name).suffix.lower()

    if file_ext not in [".onnx"]:
        st.error("Only ONNX files are supported in compare mode. Convert your model first.")
        return None

    try:
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        inspector = ModelInspector()
        report = inspector.inspect(tmp_path)

        # Clean up
        Path(tmp_path).unlink(missing_ok=True)

        # Add to history and return
        result = add_to_history(uploaded_file.name, report, len(uploaded_file.getvalue()))
        return result

    except Exception as e:
        st.error(f"Error analyzing model: {e}")
        return None


def get_hardware_options() -> dict[str, dict]:
    """Get hardware profile options organized by category."""
    categories = {
        "🔧 Auto": {"auto": {"name": "Auto-detect local GPU", "vram": 0, "tflops": 0}},
        "🏢 Data Center - H100": {},
        "🏢 Data Center - A100": {},
        "🏢 Data Center - Other": {},
        "🎮 Consumer - RTX 40 Series": {},
        "🎮 Consumer - RTX 30 Series": {},
        "💼 Workstation": {},
        "🤖 Edge / Jetson": {},
        "☁️ Cloud Instances": {},
    }

    for name, profile in HARDWARE_PROFILES.items():
        if profile.device_type != "gpu":
            continue

        vram_gb = profile.vram_bytes // (1024**3)
        tflops = profile.peak_fp16_tflops or profile.peak_fp32_tflops

        entry = {
            "name": profile.name,
            "vram": vram_gb,
            "tflops": tflops,
        }

        # Categorize
        name_lower = name.lower()
        if "h100" in name_lower:
            categories["🏢 Data Center - H100"][name] = entry
        elif "a100" in name_lower:
            categories["🏢 Data Center - A100"][name] = entry
        elif any(x in name_lower for x in ["a10", "l4", "t4", "v100", "a40", "a30"]):
            categories["🏢 Data Center - Other"][name] = entry
        elif (
            "rtx40" in name_lower
            or "4090" in name_lower
            or "4080" in name_lower
            or "4070" in name_lower
            or "4060" in name_lower
        ):
            categories["🎮 Consumer - RTX 40 Series"][name] = entry
        elif (
            "rtx30" in name_lower
            or "3090" in name_lower
            or "3080" in name_lower
            or "3070" in name_lower
            or "3060" in name_lower
        ):
            categories["🎮 Consumer - RTX 30 Series"][name] = entry
        elif any(x in name_lower for x in ["rtxa", "a6000", "a5000", "a4000"]):
            categories["💼 Workstation"][name] = entry
        elif (
            "jetson" in name_lower
            or "orin" in name_lower
            or "xavier" in name_lower
            or "nano" in name_lower
        ):
            categories["🤖 Edge / Jetson"][name] = entry
        elif any(x in name_lower for x in ["aws", "azure", "gcp"]):
            categories["☁️ Cloud Instances"][name] = entry
        else:
            categories["🏢 Data Center - Other"][name] = entry

    # Remove empty categories
    return {k: v for k, v in categories.items() if v}


def main():
    # Initialize session state
    init_session_state()

    # Header
    st.markdown('<h1 class="main-header">HaoLine 皓线</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Universal Model Inspector — See what\'s really inside your neural networks</p>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        # Mode selector
        st.markdown("### Mode")
        mode = st.radio(
            "Select mode",
            options=["Analyze", "Compare"],
            index=0 if st.session_state.current_mode == "analyze" else 1,
            horizontal=True,
            label_visibility="collapsed",
        )
        st.session_state.current_mode = mode.lower()

        st.markdown("---")

        # Session history
        if st.session_state.analysis_history:
            st.markdown("### Recent Analyses")
            for i, result in enumerate(st.session_state.analysis_history[:5]):
                time_str = result.timestamp.strftime("%H:%M")
                col1, col2 = st.columns([3, 1])
                with col1:
                    # Make the whole item clickable in analyze mode
                    if st.session_state.current_mode == "analyze":
                        if st.button(
                            f"{result.name[:18]}{'...' if len(result.name) > 18 else ''}",
                            key=f"hist_view_{i}",
                            help=f"View {result.name}",
                            width="stretch",
                        ):
                            st.session_state.current_result = result
                            st.rerun()
                        st.caption(f"{result.summary} · {time_str}")
                    else:
                        st.markdown(
                            f"""
                        <div style="font-size: 0.85rem; color: #f5f5f5; margin-bottom: 0.1rem;">
                            {result.name[:20]}{"..." if len(result.name) > 20 else ""}
                        </div>
                        <div style="font-size: 0.7rem; color: #737373;">
                            {result.summary} · {time_str}
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                with col2:
                    if st.session_state.current_mode == "compare":
                        if st.button("A", key=f"hist_a_{i}", help="Set as Model A"):
                            st.session_state.compare_models["model_a"] = result
                            st.rerun()
                        if st.button("B", key=f"hist_b_{i}", help="Set as Model B"):
                            st.session_state.compare_models["model_b"] = result
                            st.rerun()

            if st.button("Clear History", type="secondary"):
                st.session_state.analysis_history = []
                st.rerun()

            st.markdown("---")

        st.markdown("### Settings")

        # Hardware selection with categorized picker
        st.markdown("#### Target Hardware")
        hardware_categories = get_hardware_options()

        # Search filter
        search_query = st.text_input(
            "Search GPUs",
            placeholder="e.g., RTX 4090, A100, H100...",
            help="Filter hardware by name",
        )

        # Build flat list with category info for filtering
        all_hardware = []
        for category, profiles in hardware_categories.items():
            for hw_key, hw_info in profiles.items():
                display_name = hw_info["name"]
                if hw_info["vram"] > 0:
                    display_name += f" ({hw_info['vram']}GB"
                    if hw_info["tflops"]:
                        display_name += f", {hw_info['tflops']:.0f} TFLOPS"
                    display_name += ")"
                all_hardware.append(
                    {
                        "key": hw_key,
                        "display": display_name,
                        "category": category,
                        "vram": hw_info["vram"],
                        "tflops": hw_info["tflops"],
                    }
                )

        # Filter by search
        if search_query:
            filtered_hardware = [
                h
                for h in all_hardware
                if search_query.lower() in h["display"].lower()
                or search_query.lower() in h["key"].lower()
            ]
        else:
            filtered_hardware = all_hardware

        # Category filter
        available_categories = sorted({h["category"] for h in filtered_hardware})
        if len(available_categories) > 1:
            selected_category = st.selectbox(
                "Category",
                options=["All Categories"] + available_categories,
                index=0,
            )
            if selected_category != "All Categories":
                filtered_hardware = [
                    h for h in filtered_hardware if h["category"] == selected_category
                ]

        # Final hardware dropdown
        if filtered_hardware:
            hw_options = {h["key"]: h["display"] for h in filtered_hardware}
            default_key = "auto" if "auto" in hw_options else list(hw_options.keys())[0]
            selected_hardware = st.selectbox(
                "Select GPU",
                options=list(hw_options.keys()),
                format_func=lambda x: hw_options[x],
                index=list(hw_options.keys()).index(default_key)
                if default_key in hw_options
                else 0,
            )
        else:
            st.warning("No GPUs match your search. Try a different query.")
            selected_hardware = "auto"

        # Batch size and GPU count (Story 41.4.1, 41.4.5)
        hw_detail_col1, hw_detail_col2 = st.columns(2)
        with hw_detail_col1:
            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=128,
                value=1,
                help="Inference batch size for hardware estimates",
            )
        with hw_detail_col2:
            st.number_input(
                "GPU Count",
                min_value=1,
                max_value=8,
                value=1,
                help="Number of GPUs for multi-GPU estimates",
            )

        # Deployment Target (Story 41.4.6)
        st.selectbox(
            "Deployment Target",
            options=["cloud", "local", "edge"],
            index=0,
            help="Target environment affects cost and latency estimates",
            format_func=lambda x: {
                "cloud": "☁️ Cloud",
                "local": "🖥️ Local/On-Prem",
                "edge": "📱 Edge Device",
            }[x],
        )

        # Show selected hardware specs
        if selected_hardware != "auto":
            try:
                profile = HARDWARE_PROFILES.get(selected_hardware)
                if profile:
                    st.markdown(
                        f"""
                    <div style="background: #1f1f1f;
                                border: 1px solid rgba(16, 185, 129, 0.2);
                                padding: 0.75rem 1rem; border-radius: 10px; margin-top: 0.5rem;">
                        <div style="font-size: 0.85rem; color: #10b981; font-weight: 600;">
                            {profile.name}
                        </div>
                        <div style="font-size: 0.75rem; color: #737373; margin-top: 0.25rem; font-family: 'SF Mono', monospace;">
                            {profile.vram_bytes // (1024**3)} GB VRAM · {profile.peak_fp16_tflops or "—"} TF
                        </div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
            except Exception:
                pass

        # Analysis options
        st.markdown("### Analysis Options")
        include_graph = st.checkbox(
            "Interactive Graph", value=True, help="Include zoomable D3.js network visualization"
        )
        st.checkbox("Charts", value=True, help="Include matplotlib visualizations")
        include_quant_analysis = st.checkbox(
            "Quantization Analysis",
            value=False,
            help="Analyze model readiness for INT8 quantization",
        )

        # Privacy Controls (Story 41.3.12, 41.4.8)
        st.markdown("### Privacy Controls")
        redact_names = st.checkbox(
            "Redact Layer Names",
            value=False,
            help="Anonymize layer names (layer_0, layer_1, ...) for IP protection",
        )
        summary_only = st.checkbox(
            "Summary Only",
            value=False,
            help="Show only aggregate statistics, hide per-layer details",
        )

        # Feature Availability Matrix (Task 50.2.7)
        with st.expander("Feature Availability", expanded=False):
            # Check what's available in this environment
            features = _check_feature_availability()

            st.markdown(
                """
                <style>
                    .feat-avail { font-size: 0.85rem; }
                    .feat-yes { color: #10b981; }
                    .feat-no { color: #ef4444; }
                    .feat-badge {
                        font-size: 0.7rem;
                        padding: 2px 6px;
                        border-radius: 4px;
                        margin-left: 4px;
                    }
                    .badge-gpu { background: #7c3aed; color: white; }
                    .badge-dep { background: #f59e0b; color: black; }
                </style>
                """,
                unsafe_allow_html=True,
            )

            for feat_name, feat_info in features.items():
                status = "✓" if feat_info["available"] else "✗"
                status_class = "feat-yes" if feat_info["available"] else "feat-no"
                badge = ""
                if feat_info.get("requires_gpu"):
                    badge = '<span class="feat-badge badge-gpu">GPU</span>'
                elif feat_info.get("requires_dep"):
                    badge = f'<span class="feat-badge badge-dep">{feat_info["requires_dep"]}</span>'

                st.markdown(
                    f'<div class="feat-avail"><span class="{status_class}">{status}</span> {feat_name}{badge}</div>',
                    unsafe_allow_html=True,
                )

            if not features.get("TensorRT Analysis", {}).get("available"):
                st.caption("GPU features require `pip install haoline[tensorrt]` and NVIDIA GPU")

        # LLM Summary
        st.markdown("### AI Summary")
        import os

        env_api_key = os.environ.get("OPENAI_API_KEY", "")
        enable_llm = st.checkbox(
            "Generate AI Summary",
            value=st.session_state.get("enable_llm", False),
            help="Requires OpenAI API key",
            key="enable_llm_checkbox",
        )
        st.session_state["enable_llm"] = enable_llm

        if enable_llm:
            if env_api_key:
                st.success("Using API key from environment")
                st.session_state["openai_api_key_value"] = env_api_key
            else:
                api_key = st.text_input(
                    "OpenAI API Key",
                    type="password",
                    help="Used once per analysis, never stored",
                    key="openai_api_key_input",
                )
                st.session_state["openai_api_key_value"] = api_key
            st.caption("For maximum security, run `haoline` locally instead.")

        # Privacy notice
        st.markdown("---")
        st.markdown(
            '<div class="privacy-notice">'
            "<strong>Privacy:</strong> Models and API keys are processed in memory only. "
            "Nothing is stored. For sensitive work, self-host with <code>pip install haoline[web]</code> "
            "and run <code>streamlit run streamlit_app.py</code> locally."
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(f"---\n*HaoLine v{__version__}*")

    # Main content - different views based on mode
    if st.session_state.current_mode == "compare":
        render_compare_mode()
        return

    # Analyze mode
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # File upload - support multiple formats
        uploaded_file = st.file_uploader(
            "Upload your model",
            type=[
                "onnx",  # ONNX (full support)
                "pt",
                "pth",  # PyTorch
                "safetensors",  # HuggingFace weights
                "engine",
                "plan",  # TensorRT
                "tflite",  # TensorFlow Lite
                "mlmodel",
                "mlpackage",  # CoreML (macOS)
                "xml",  # OpenVINO IR
                "gguf",  # GGUF (LLM weights)
            ],
            help="Limit 500MB per file. PyTorch models (.pt/.pth) require local PyTorch installation for conversion.",
        )

        # Use demo model if one was downloaded (same code path as uploaded files)
        if uploaded_file is None and "demo_uploaded_file" in st.session_state:
            uploaded_file = st.session_state.pop("demo_uploaded_file")

        if uploaded_file is None:
            # Show format capability matrix
            st.markdown(
                """
            <div style="text-align: center; padding: 0.5rem 2rem; margin-top: -0.5rem;">
                <p style="font-size: 0.8rem; color: #737373; margin-bottom: 1rem;">
                    Need a model? Browse the
                    <a href="https://huggingface.co/models?library=onnx" target="_blank" style="color: #10b981; text-decoration: none;">HuggingFace ONNX Hub</a>
                </p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Capability matrix as an expander
            with st.expander("Format Capabilities", expanded=False):
                st.markdown(
                    """
                <style>
                    .cap-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; table-layout: fixed; }
                    .cap-table th { text-align: left; padding: 0.4rem; border-bottom: 1px solid #333; color: #10b981; }
                    .cap-table td { padding: 0.3rem 0.4rem; border-bottom: 1px solid #222; white-space: nowrap; }
                    .cap-table td:last-child { white-space: normal; font-size: 0.75rem; }
                    .cap-table tr:hover { background: rgba(16, 185, 129, 0.05); }
                    .cap-yes { color: #10b981; }
                    .cap-no { color: #666; }
                    .cap-cli { color: #60a5fa; }
                    .cap-warn { color: #f59e0b; }
                </style>
                <p style="font-size: 0.8rem; color: #888; margin-bottom: 0.5rem;"><strong>Tier 1 - Full Analysis</strong></p>
                <table class="cap-table">
                    <tr>
                        <th>Format</th>
                        <th>Graph</th>
                        <th>Params</th>
                        <th>FLOPs</th>
                        <th>Map</th>
                        <th>Notes</th>
                    </tr>
                    <tr>
                        <td><strong>ONNX</strong></td>
                        <td class="cap-yes">Yes</td>
                        <td class="cap-yes">Yes</td>
                        <td class="cap-yes">Yes</td>
                        <td class="cap-yes">Yes</td>
                        <td class="cap-yes">Full support</td>
                    </tr>
                    <tr>
                        <td><strong>PyTorch</strong></td>
                        <td class="cap-cli">CLI</td>
                        <td class="cap-cli">CLI</td>
                        <td class="cap-cli">CLI</td>
                        <td class="cap-cli">CLI</td>
                        <td class="cap-warn">CLI + PyTorch</td>
                    </tr>
                </table>
                <p style="font-size: 0.8rem; color: #888; margin: 0.75rem 0 0.5rem 0;"><strong>Tier 2 - Graph Analysis</strong></p>
                <table class="cap-table">
                    <tr>
                        <td><strong>TFLite</strong></td>
                        <td class="cap-cli">CLI</td>
                        <td class="cap-cli">CLI</td>
                        <td class="cap-no">No</td>
                        <td class="cap-cli">CLI</td>
                        <td class="cap-warn">CLI only</td>
                    </tr>
                    <tr>
                        <td><strong>CoreML</strong></td>
                        <td class="cap-cli">CLI</td>
                        <td class="cap-cli">CLI</td>
                        <td class="cap-no">No</td>
                        <td class="cap-cli">CLI</td>
                        <td class="cap-warn">macOS CLI</td>
                    </tr>
                    <tr>
                        <td><strong>OpenVINO</strong></td>
                        <td class="cap-cli">CLI</td>
                        <td class="cap-cli">CLI</td>
                        <td class="cap-no">No</td>
                        <td class="cap-cli">CLI</td>
                        <td class="cap-warn">Needs .bin file</td>
                    </tr>
                    <tr>
                        <td><strong>TensorRT</strong> <span style="background:#7c3aed;color:white;font-size:0.65rem;padding:1px 4px;border-radius:3px;">GPU</span></td>
                        <td class="cap-warn">GPU</td>
                        <td class="cap-no">N/A</td>
                        <td class="cap-no">N/A</td>
                        <td class="cap-no">No</td>
                        <td class="cap-warn">Requires NVIDIA GPU</td>
                    </tr>
                </table>
                <p style="font-size: 0.8rem; color: #888; margin: 0.75rem 0 0.5rem 0;"><strong>Tier 3/4 - Metadata Only</strong></p>
                <table class="cap-table">
                    <tr>
                        <td><strong>GGUF</strong></td>
                        <td class="cap-no">No</td>
                        <td class="cap-yes">Yes</td>
                        <td class="cap-no">No</td>
                        <td class="cap-no">No</td>
                        <td class="cap-warn">LLM metadata + quant info</td>
                    </tr>
                    <tr>
                        <td><strong>SafeTensors</strong></td>
                        <td class="cap-no">No</td>
                        <td class="cap-yes">Yes</td>
                        <td class="cap-no">No</td>
                        <td class="cap-no">No</td>
                        <td class="cap-warn">Weights only</td>
                    </tr>
                </table>
                <p style="font-size: 0.7rem; color: #bbb; margin-top: 0.75rem;">
                    <strong>Legend:</strong> Yes = In-app (Streamlit) | CLI = Use <code>pip install haoline</code> locally | GPU = Requires NVIDIA GPU.<br>
                    For full analysis, convert to ONNX format.
                </p>
                """,
                    unsafe_allow_html=True,
                )

            # Demo models - quick start buttons
            st.markdown("#### Try a Demo Model")
            demo_col1, demo_col2, demo_col3 = st.columns(3)

            with demo_col1:
                if st.button("MNIST (26 KB)", width="stretch", help="Tiny model - instant"):
                    st.session_state["demo_model"] = "MNIST"
            with demo_col2:
                if st.button("MobileNetV2 (14 MB)", width="stretch", help="Medium model"):
                    st.session_state["demo_model"] = "MobileNetV2"
            with demo_col3:
                if st.button("EfficientNet (50 MB)", width="stretch", help="Larger model"):
                    st.session_state["demo_model"] = "EfficientNet"

    # Handle demo model download - convert to fake "uploaded file" and rerun
    # This ensures demo models use the SAME code path as uploaded files (no duplication)
    if "demo_model" in st.session_state and st.session_state["demo_model"]:
        demo_name = st.session_state["demo_model"]
        demo_info = DEMO_MODELS[demo_name]
        st.session_state["demo_model"] = None  # Clear to prevent re-download

        with st.spinner(f"Downloading {demo_name} ({demo_info['size']})..."):
            import urllib.request

            try:
                # Download into memory
                with urllib.request.urlopen(demo_info["url"]) as response:
                    demo_bytes = response.read()

                # Create a simple object that mimics UploadedFile
                class DemoUploadedFile:
                    def __init__(self, name: str, data: bytes):
                        self._name = name
                        self._data = data

                    @property
                    def name(self) -> str:
                        return self._name

                    def getvalue(self) -> bytes:
                        return self._data

                # Store as "uploaded file" so normal path handles it
                st.session_state["demo_uploaded_file"] = DemoUploadedFile(
                    f"{demo_name}.onnx", demo_bytes
                )
                st.rerun()

            except Exception as e:
                st.error(f"Failed to download {demo_name}: {e}")

    # Display current result from history/demo
    if st.session_state.current_result is not None and uploaded_file is None:
        result = st.session_state.current_result
        report = result.report
        model_name = result.name

        # Load graph_info for layer/quantization views (ONNX only)
        graph_info = None
        if result.model_path and Path(result.model_path).exists():
            if Path(result.model_path).suffix.lower() == ".onnx":
                try:
                    from haoline.analyzer import ONNXGraphLoader

                    graph_loader = ONNXGraphLoader()
                    _, graph_info = graph_loader.load(result.model_path)
                except Exception:
                    pass  # graph_info remains None

        # Clear button
        if st.button("← Back to upload", type="secondary"):
            st.session_state.current_result = None
            st.rerun()

        st.markdown("---")
        st.markdown(f"## Analysis Results: {model_name}")

        # Metrics cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            params = report.param_counts.total if report.param_counts else 0
            st.metric("Parameters", format_number(params))

        with col2:
            flops = report.flop_counts.total if report.flop_counts else 0
            st.metric("FLOPs", format_number(flops))

        with col3:
            memory = report.memory_estimates.peak_activation_bytes if report.memory_estimates else 0
            st.metric("Memory", format_bytes(memory))

        with col4:
            st.metric("Operators", str(report.graph_summary.num_nodes))

        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            ["Overview", "Interactive Graph", "Details", "Layer Details", "Quantization", "Export"]
        )

        # Use shared tab render functions (single source of truth)
        with tab1:
            render_overview_tab(report, model_name)

        with tab2:
            render_graph_tab(
                report,
                result.model_path,
                model_name,
                result.file_size,
                graph_info,
            )

        with tab3:
            render_details_tab(report)

        with tab4:
            render_layer_details_tab(report, graph_info, model_name)

        with tab5:
            render_quantization_tab(report, graph_info)

        with tab6:
            render_export_tab(
                report,
                model_name.replace(".onnx", ""),
                result.model_path,
                hardware=selected_hardware,
                batch_size=batch_size,
            )

    # Analysis (file upload)
    elif uploaded_file is not None:
        file_ext = Path(uploaded_file.name).suffix.lower()
        tmp_path = None

        # Check if format needs conversion
        if file_ext in [".pt", ".pth"]:
            # Check if PyTorch is available
            try:
                import torch

                pytorch_available = True
            except ImportError:
                pytorch_available = False

            if pytorch_available:
                st.info(
                    "**PyTorch model detected** — We'll try to convert it to ONNX for analysis."
                )

                # Input shape is required for conversion
                input_shape_str = st.text_input(
                    "Input Shape (required)",
                    placeholder="1,3,224,224",
                    help="Batch, Channels, Height, Width for image models. E.g., 1,3,224,224",
                )

                if not input_shape_str:
                    st.warning("⚠️ Please enter the input shape to convert and analyze this model.")
                    st.caption(
                        "**Common shapes:** `1,3,224,224` (ResNet), `1,3,384,384` (ViT-Large), `1,768` (BERT tokens)"
                    )
                    st.stop()

                # Try conversion
                try:
                    input_shape = tuple(int(x.strip()) for x in input_shape_str.split(","))
                except ValueError:
                    st.error(
                        f"Invalid input shape: `{input_shape_str}`. Use comma-separated integers like `1,3,224,224`"
                    )
                    st.stop()

                # Save uploaded file
                with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as pt_tmp:
                    pt_tmp.write(uploaded_file.getvalue())
                    pt_path = pt_tmp.name

                # Attempt conversion
                with st.spinner("Converting PyTorch → ONNX..."):
                    try:
                        # Try TorchScript first
                        try:
                            model = torch.jit.load(pt_path, map_location="cpu")
                        except Exception:
                            loaded = torch.load(pt_path, map_location="cpu", weights_only=False)
                            if isinstance(loaded, dict):
                                st.error("""
                                **State dict detected** — This file contains only weights, not the model architecture.

                                To analyze, you need the full model. Export to ONNX from your training code:
                                ```python
                                torch.onnx.export(model, dummy_input, "model.onnx")
                                ```
                                """)
                                st.stop()
                            model = loaded

                        model.eval()
                        dummy_input = torch.randn(*input_shape)

                        # Convert to ONNX
                        onnx_tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
                        torch.onnx.export(
                            model,
                            dummy_input,
                            onnx_tmp.name,
                            opset_version=17,
                            input_names=["input"],
                            output_names=["output"],
                            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                        )
                        tmp_path = onnx_tmp.name
                        st.success("✅ Conversion successful!")

                    except Exception as e:
                        st.error(f"""
                        **Conversion failed:** {str(e)[:200]}

                        Try exporting to ONNX directly from your training code, or use the CLI:
                        ```bash
                        haoline --from-pytorch model.pt --input-shape {input_shape_str} --html
                        ```
                        """)
                        st.stop()
            else:
                st.warning(f"""
                **PyTorch model detected**, but PyTorch is not installed in this environment.

                **Options:**
                1. Use the CLI locally (supports conversion):
                   ```bash
                   pip install haoline torch
                   haoline --from-pytorch {uploaded_file.name} --input-shape 1,3,224,224 --html
                   ```

                2. Convert to ONNX first in your code:
                   ```python
                   torch.onnx.export(model, dummy_input, "model.onnx")
                   ```
                """)
                st.stop()

        elif file_ext == ".safetensors":
            st.warning("""
            **SafeTensors format detected** — This format contains only weights, not architecture.

            To analyze, export to ONNX from your training code. If using HuggingFace:
            ```python
            from optimum.exporters.onnx import main_export
            main_export("model-name", output="model.onnx")
            ```
            """)
            st.stop()

        elif file_ext in [".engine", ".plan"]:
            # TensorRT engine analysis
            _handle_tensorrt_streamlit(uploaded_file)
            st.stop()

        elif file_ext in [".tflite"]:
            # TFLite analysis
            st.info("**TFLite model detected** - Analyzing with TFLite reader...")
            with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

        elif file_ext in [".mlmodel", ".mlpackage"]:
            # CoreML analysis
            st.info("**CoreML model detected** - Analyzing with CoreML reader...")
            st.warning("Note: Full CoreML analysis requires macOS with coremltools installed.")
            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

        elif file_ext in [".xml"]:
            # OpenVINO IR analysis
            st.info("**OpenVINO IR detected** - Analyzing with OpenVINO reader...")
            st.warning(
                "Note: Full analysis requires the .bin file in the same directory. Upload may be partial."
            )
            with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

        elif file_ext in [".gguf"]:
            # GGUF (LLM) analysis
            st.info("**GGUF model detected** - Analyzing LLM architecture metadata...")
            with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

        # Save ONNX to temp file (if not already set by conversion)
        if tmp_path is None:
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

        try:
            with st.spinner("Analyzing model architecture..."):
                # Run analysis
                inspector = ModelInspector()
                report = inspector.inspect(tmp_path)

                # Apply hardware estimates
                if selected_hardware == "auto":
                    profile = detect_local_hardware()
                else:
                    profile = get_profile(selected_hardware)

                if (
                    profile
                    and report.param_counts
                    and report.flop_counts
                    and report.memory_estimates
                ):
                    estimator = HardwareEstimator()
                    report.hardware_profile = profile
                    report.hardware_estimates = estimator.estimate(
                        model_params=report.param_counts.total,
                        model_flops=report.flop_counts.total,
                        peak_activation_bytes=report.memory_estimates.peak_activation_bytes,
                        hardware=profile,
                        batch_size=batch_size,  # From sidebar (41.4.1)
                    )

                # Save to session history (keep temp path for graph/layer views)
                add_to_history(
                    uploaded_file.name,
                    report,
                    len(uploaded_file.getvalue()),
                    model_path=tmp_path,
                )

                # Display results
                st.markdown("---")
                st.markdown("## Analysis Results")

                # Metrics cards
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    params = report.param_counts.total if report.param_counts else 0
                    st.metric("Parameters", format_number(params))

                with col2:
                    flops = report.flop_counts.total if report.flop_counts else 0
                    st.metric("FLOPs", format_number(flops))

                with col3:
                    memory = (
                        report.memory_estimates.peak_activation_bytes
                        if report.memory_estimates
                        else 0
                    )
                    st.metric("Memory", format_bytes(memory))

                with col4:
                    st.metric("Operators", str(report.graph_summary.num_nodes))

                # Prepare graph_info (ONNX only) for layer/quant views
                graph_info = None
                if tmp_path and Path(tmp_path).exists():
                    if Path(tmp_path).suffix.lower() == ".onnx":
                        try:
                            from haoline.analyzer import ONNXGraphLoader

                            graph_loader = ONNXGraphLoader()
                            _, graph_info = graph_loader.load(tmp_path)
                        except Exception as e:
                            st.warning(f"Could not load graph for detailed views: {e}")

                # Tabs for different views
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
                    [
                        "Overview",
                        "Interactive Graph",
                        "Details",
                        "Layer Details",
                        "Quantization",
                        "Export",
                    ]
                )

                with tab1:
                    st.markdown("### Model Information")

                    info_col1, info_col2 = st.columns(2)

                    with info_col1:
                        st.markdown(f"""
                        | Property | Value |
                        |----------|-------|
                        | **Model** | `{uploaded_file.name}` |
                        | **IR Version** | {report.metadata.ir_version} |
                        | **Producer** | {report.metadata.producer_name or "Unknown"} |
                        | **Opset** | {list(report.metadata.opsets.values())[0] if report.metadata.opsets else "Unknown"} |
                        """)

                    with info_col2:
                        params_total = report.param_counts.total if report.param_counts else 0
                        flops_total = report.flop_counts.total if report.flop_counts else 0
                        peak_mem = (
                            report.memory_estimates.peak_activation_bytes
                            if report.memory_estimates
                            else 0
                        )
                        model_size = (
                            report.memory_estimates.model_size_bytes
                            if report.memory_estimates
                            else 0
                        )

                        st.markdown(f"""
                        | Metric | Value |
                        |--------|-------|
                        | **Total Parameters** | {params_total:,} |
                        | **Total FLOPs** | {flops_total:,} |
                        | **Peak Memory** | {format_bytes(peak_mem)} |
                        | **Model Size** | {format_bytes(model_size)} |
                        """)

                    # Operator distribution
                    if report.graph_summary.op_type_counts:
                        st.markdown("### Operator Distribution")

                        import pandas as pd

                        op_data = pd.DataFrame(
                            [
                                {"Operator": op, "Count": count}
                                for op, count in sorted(
                                    report.graph_summary.op_type_counts.items(),
                                    key=lambda x: x[1],
                                    reverse=True,
                                )
                            ]
                        )
                        st.bar_chart(op_data.set_index("Operator"))

                    # Parameter distribution by op type (Story 41.2.5)
                    if report.param_counts and report.param_counts.by_op_type:
                        st.markdown("### Parameter Distribution by Op Type")

                        import pandas as pd

                        param_by_op = report.param_counts.by_op_type
                        total_params = sum(param_by_op.values())
                        param_data = pd.DataFrame(
                            [
                                {
                                    "Op Type": op,
                                    "Parameters": count,
                                    "Percentage": 100.0 * count / total_params
                                    if total_params > 0
                                    else 0,
                                }
                                for op, count in sorted(
                                    param_by_op.items(), key=lambda x: x[1], reverse=True
                                )[:10]
                            ]
                        )
                        st.bar_chart(param_data.set_index("Op Type")["Parameters"])

                    # FLOPs distribution by op type (Story 41.2.5)
                    if report.flop_counts and report.flop_counts.by_op_type:
                        st.markdown("### FLOPs Distribution by Op Type")

                        import pandas as pd

                        flops_by_op = report.flop_counts.by_op_type
                        total_flops = sum(flops_by_op.values())
                        flops_data = pd.DataFrame(
                            [
                                {
                                    "Op Type": op,
                                    "FLOPs": count,
                                    "Percentage": 100.0 * count / total_flops
                                    if total_flops > 0
                                    else 0,
                                }
                                for op, count in sorted(
                                    flops_by_op.items(), key=lambda x: x[1], reverse=True
                                )[:10]
                            ]
                        )
                        st.bar_chart(flops_data.set_index("Op Type")["FLOPs"])

                    # Hardware estimates
                    if report.hardware_estimates:
                        st.markdown("### Hardware Estimates")
                        hw = report.hardware_estimates

                        hw_col1, hw_col2, hw_col3 = st.columns(3)

                        with hw_col1:
                            st.metric("VRAM Required", format_bytes(hw.vram_required_bytes))

                        with hw_col2:
                            fits = "Yes" if hw.fits_in_vram else "No"
                            st.metric("Fits in VRAM", fits)

                        with hw_col3:
                            st.metric("Theoretical Latency", f"{hw.theoretical_latency_ms:.2f} ms")

                        # Bottleneck analysis
                        if hasattr(hw, "bottleneck") and hw.bottleneck:
                            st.markdown("#### Performance Bottleneck")
                            bottleneck_colors = {
                                "compute": "#10b981",  # Green - compute bound
                                "memory_bandwidth": "#f59e0b",  # Amber - memory bound
                                "vram": "#ef4444",  # Red - VRAM limited
                            }
                            color = bottleneck_colors.get(hw.bottleneck, "#6b7280")
                            st.markdown(
                                f'<span style="color: {color}; font-weight: bold; font-size: 1.1rem;">'
                                f"{hw.bottleneck.replace('_', ' ').title()}</span>",
                                unsafe_allow_html=True,
                            )
                            if hw.bottleneck == "compute":
                                st.caption(
                                    "Model is compute-bound. Consider quantization (FP16/INT8) for speedup."
                                )
                            elif hw.bottleneck == "memory_bandwidth":
                                st.caption(
                                    "Model is memory-bound. Consider reducing activations or batch size."
                                )
                            elif hw.bottleneck == "vram":
                                st.caption(
                                    "Model exceeds VRAM. Consider a larger GPU or model compression."
                                )

                    # KV Cache section (for transformers)
                    if (
                        report.memory_estimates
                        and hasattr(report.memory_estimates, "kv_cache_bytes_per_token")
                        and report.memory_estimates.kv_cache_bytes_per_token > 0
                    ):
                        st.markdown("### KV Cache (Transformer Inference)")
                        kv = report.memory_estimates
                        config = getattr(kv, "kv_cache_config", {}) or {}

                        kv_col1, kv_col2 = st.columns(2)
                        with kv_col1:
                            st.metric("Per Token", format_bytes(kv.kv_cache_bytes_per_token))
                            if config.get("num_layers"):
                                st.caption(f"Layers: {config['num_layers']}")
                        with kv_col2:
                            if kv.kv_cache_bytes_full_context > 0:
                                seq_len = config.get("seq_len", "?")
                                st.metric(
                                    f"Full Context (seq={seq_len})",
                                    format_bytes(kv.kv_cache_bytes_full_context),
                                )
                                if config.get("hidden_dim"):
                                    st.caption(f"Hidden dim: {config['hidden_dim']}")

                    # Precision breakdown
                    if (
                        report.param_counts
                        and hasattr(report.param_counts, "precision_breakdown")
                        and report.param_counts.precision_breakdown
                    ):
                        st.markdown("### Precision Breakdown")
                        precision_data = report.param_counts.precision_breakdown
                        total = sum(precision_data.values())

                        import pandas as pd

                        prec_df = pd.DataFrame(
                            [
                                {
                                    "Data Type": dtype,
                                    "Parameters": count,
                                    "Percentage": f"{100.0 * count / total:.1f}%"
                                    if total > 0
                                    else "0%",
                                }
                                for dtype, count in sorted(
                                    precision_data.items(), key=lambda x: -x[1]
                                )
                            ]
                        )
                        st.dataframe(prec_df, width="stretch", hide_index=True)

                        # Quantization indicator
                        if report.param_counts.is_quantized:
                            st.success("Model is quantized")
                            if report.param_counts.quantized_ops:
                                st.caption(
                                    f"Quantized ops: {', '.join(report.param_counts.quantized_ops[:5])}"
                                )

                    # Quantization Readiness Analysis (Epic 33)
                    if include_quant_analysis:
                        with st.expander("INT8 Quantization Readiness", expanded=True):
                            try:
                                from haoline.analyzer import ONNXGraphLoader
                                from haoline.quantization_linter import (
                                    QuantizationLinter,
                                    Severity,
                                )

                                # Get model name for reports
                                model_name = uploaded_file.name.replace(".onnx", "")

                                # Load graph and run linting
                                graph_loader = ONNXGraphLoader()
                                _, graph_info = graph_loader.load(tmp_path)
                                linter = QuantizationLinter()
                                quant_result = linter.lint(graph_info)

                                # Readiness Score with letter grade
                                score = quant_result.readiness_score
                                if score >= 90:
                                    grade, color = "A", "#22c55e"
                                elif score >= 75:
                                    grade, color = "B", "#84cc16"
                                elif score >= 60:
                                    grade, color = "C", "#eab308"
                                elif score >= 40:
                                    grade, color = "D", "#f97316"
                                else:
                                    grade, color = "F", "#ef4444"

                                # Score display
                                score_col1, score_col2, score_col3 = st.columns([1, 2, 1])
                                with score_col2:
                                    st.markdown(
                                        f"""
                                        <div style="text-align: center; padding: 1rem;
                                             background: linear-gradient(135deg, {color}22, {color}11);
                                             border: 2px solid {color}; border-radius: 12px;">
                                            <div style="font-size: 3rem; font-weight: bold; color: {color};">
                                                {grade}
                                            </div>
                                            <div style="font-size: 1.5rem; color: #e5e5e5;">
                                                {score}/100
                                            </div>
                                            <div style="font-size: 0.9rem; color: #a3a3a3;">
                                                Quantization Readiness
                                            </div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )

                                st.markdown("")

                                # Op breakdown
                                op_col1, op_col2 = st.columns(2)
                                with op_col1:
                                    st.metric(
                                        "Quant-Friendly Ops",
                                        f"{quant_result.quant_friendly_pct:.1f}%",
                                        help="Percentage of ops that work well with INT8",
                                    )
                                with op_col2:
                                    st.metric(
                                        "Issues Found",
                                        len(quant_result.warnings),
                                        delta=f"{quant_result.critical_count} critical"
                                        if quant_result.critical_count > 0
                                        else None,
                                        delta_color="inverse",
                                    )

                                # Warnings by severity
                                if quant_result.warnings:
                                    st.markdown("#### Issues")
                                    severity_icons = {
                                        Severity.CRITICAL: "!!",
                                        Severity.HIGH: "!",
                                        Severity.MEDIUM: "~",
                                        Severity.LOW: ".",
                                        Severity.INFO: "i",
                                    }
                                    severity_colors = {
                                        Severity.CRITICAL: "#ef4444",
                                        Severity.HIGH: "#f97316",
                                        Severity.MEDIUM: "#eab308",
                                        Severity.LOW: "#84cc16",
                                        Severity.INFO: "#3b82f6",
                                    }

                                    for w in sorted(
                                        quant_result.warnings,
                                        key=lambda x: list(Severity).index(x.severity),
                                    ):
                                        icon = severity_icons.get(w.severity, "?")
                                        color = severity_colors.get(w.severity, "#737373")
                                        st.markdown(
                                            f"""
                                            <div style="padding: 0.5rem; margin: 0.25rem 0;
                                                 border-left: 3px solid {color}; background: {color}11;">
                                                <span style="color: {color}; font-weight: bold;">[{icon}]</span>
                                                <span style="color: #e5e5e5;">{w.message}</span>
                                                {f'<br><span style="color: #a3a3a3; font-size: 0.85rem; margin-left: 2rem;">→ {w.recommendation}</span>' if w.recommendation else ""}
                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )

                                # Problem layers table
                                if quant_result.problem_layers:
                                    st.markdown("#### Problem Layers")
                                    prob_df = pd.DataFrame(quant_result.problem_layers[:10])
                                    st.dataframe(prob_df, width="stretch", hide_index=True)

                                # QAT-specific results
                                if quant_result.is_qat_model:
                                    st.markdown("#### QAT Model Validation")
                                    qat_col1, qat_col2 = st.columns(2)
                                    with qat_col1:
                                        st.metric(
                                            "Missing Fake-Quant",
                                            len(quant_result.missing_fake_quant_nodes),
                                        )
                                    with qat_col2:
                                        st.metric(
                                            "Scale Mismatches",
                                            len(quant_result.scale_mismatches),
                                        )

                                # Download report button
                                recommendations = linter.get_recommendations(quant_result)
                                report_lines = [
                                    f"# Quantization Readiness Report: {model_name}",
                                    "",
                                    f"**Score:** {score}/100 ({grade})",
                                    f"**Quant-Friendly Ops:** {quant_result.quant_friendly_pct:.1f}%",
                                    "",
                                    "## Recommendations",
                                    "",
                                ]
                                for rec in recommendations:
                                    report_lines.append(f"- {rec}")
                                report_lines.append("")
                                report_lines.append("## Warnings")
                                report_lines.append("")
                                for w in quant_result.warnings:
                                    report_lines.append(f"- **[{w.severity.value}]** {w.message}")
                                    if w.recommendation:
                                        report_lines.append(f"  - {w.recommendation}")

                                st.download_button(
                                    "Download Quant Report",
                                    data="\n".join(report_lines),
                                    file_name=f"{model_name}_quant_report.md",
                                    mime="text/markdown",
                                )

                                # LLM-powered advice (Task 33.4.12)
                                st.markdown("---")
                                st.markdown("#### Quantization Recommendations")

                                # Check for API key
                                use_llm_for_quant = st.session_state.get(
                                    "enable_llm", False
                                ) and st.session_state.get("openai_api_key_value", "")

                                try:
                                    from haoline.quantization_advisor import (
                                        advise_quantization,
                                        generate_qat_readiness_report,
                                    )

                                    advice = advise_quantization(
                                        quant_result,
                                        graph_info,
                                        api_key=st.session_state.get("openai_api_key_value", "")
                                        if use_llm_for_quant
                                        else None,
                                        use_llm=use_llm_for_quant,
                                    )

                                    # Architecture badge
                                    arch_colors = {
                                        "cnn": "#22c55e",
                                        "transformer": "#8b5cf6",
                                        "rnn": "#f97316",
                                        "hybrid": "#06b6d4",
                                        "unknown": "#737373",
                                    }
                                    arch_color = arch_colors.get(
                                        advice.architecture_type.value, "#737373"
                                    )
                                    st.markdown(
                                        f"""
                                        <div style="display: inline-block; padding: 0.25rem 0.75rem;
                                             background: {arch_color}22; border: 1px solid {arch_color};
                                             border-radius: 999px; margin-bottom: 0.5rem;">
                                            <span style="color: {arch_color}; font-weight: 600;">
                                                {advice.architecture_type.value.upper()}
                                            </span>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )

                                    # Strategy
                                    st.markdown(f"> {advice.strategy}")

                                    # Expected accuracy impact
                                    st.info(
                                        f"Expected Accuracy Impact: {advice.expected_accuracy_impact}"
                                    )

                                    # Op Substitutions
                                    if advice.op_substitutions:
                                        st.markdown("**Recommended Op Substitutions:**")
                                        for sub in advice.op_substitutions:
                                            st.markdown(
                                                f"- `{sub.original_op}` → `{sub.replacement_op}` "
                                                f"({sub.accuracy_impact} accuracy impact)"
                                            )

                                    # QAT Workflow (collapsed)
                                    with st.expander("QAT Workflow Steps"):
                                        for i, step in enumerate(advice.qat_workflow, 1):
                                            st.markdown(f"{i}. {step}")
                                        st.markdown("")
                                        st.markdown(
                                            f"**Calibration Tips:** {advice.calibration_tips}"
                                        )

                                    # Runtime recommendations (collapsed)
                                    with st.expander("Runtime-Specific Recommendations"):
                                        for runtime, rec in advice.runtime_recommendations.items():
                                            st.markdown(f"**{runtime.upper()}**")
                                            st.markdown(rec)
                                            st.markdown("")

                                    # Download full QAT report
                                    full_report = generate_qat_readiness_report(
                                        quant_result, advice, model_name, format="markdown"
                                    )
                                    st.download_button(
                                        "Download Full QAT Report",
                                        data=full_report,
                                        file_name=f"{model_name}_qat_readiness.md",
                                        mime="text/markdown",
                                        key="download_qat_report",
                                    )

                                except Exception as e:
                                    st.warning(f"Could not generate recommendations: {e}")

                            except Exception as e:
                                st.warning(f"Could not run quantization analysis: {e}")

                    # Memory breakdown by op type
                    if (
                        report.memory_estimates
                        and hasattr(report.memory_estimates, "breakdown")
                        and report.memory_estimates.breakdown
                    ):
                        bd = report.memory_estimates.breakdown
                        if hasattr(bd, "weights_by_op_type") and bd.weights_by_op_type:
                            st.markdown("### Memory Breakdown by Op Type")

                            import pandas as pd

                            total_w = sum(bd.weights_by_op_type.values())
                            mem_df = pd.DataFrame(
                                [
                                    {
                                        "Op Type": op_type,
                                        "Size": format_bytes(size),
                                        "Percentage": f"{100.0 * size / total_w:.1f}%"
                                        if total_w > 0
                                        else "0%",
                                    }
                                    for op_type, size in sorted(
                                        bd.weights_by_op_type.items(), key=lambda x: -x[1]
                                    )[:8]
                                ]
                            )
                            st.dataframe(mem_df, width="stretch", hide_index=True)

                    # Memory Usage Overview (Story 41.3.2)
                    if report.memory_estimates:
                        st.markdown("### Memory Usage Overview")
                        mem = report.memory_estimates

                        import pandas as pd

                        # Build memory components
                        memory_components = []

                        # Model weights
                        if mem.model_size_bytes > 0:
                            memory_components.append(
                                {
                                    "Component": "Model Weights",
                                    "Size (MB)": mem.model_size_bytes / (1024**2),
                                }
                            )

                        # Peak activations
                        if mem.peak_activation_bytes > 0:
                            memory_components.append(
                                {
                                    "Component": "Peak Activations",
                                    "Size (MB)": mem.peak_activation_bytes / (1024**2),
                                }
                            )

                        # KV Cache (if transformer)
                        if mem.kv_cache_bytes_full_context > 0:
                            memory_components.append(
                                {
                                    "Component": "KV Cache (full)",
                                    "Size (MB)": mem.kv_cache_bytes_full_context / (1024**2),
                                }
                            )

                        if memory_components:
                            mem_overview_df = pd.DataFrame(memory_components)
                            st.bar_chart(
                                mem_overview_df.set_index("Component")["Size (MB)"],
                                horizontal=True,
                            )

                            # Show total
                            total_mb = sum(c["Size (MB)"] for c in memory_components)
                            st.caption(
                                f"**Total Estimated Memory:** {total_mb:.1f} MB ({total_mb / 1024:.2f} GB)"
                            )

                    # System Requirements (Story 41.3.6)
                    if report.param_counts and report.flop_counts and report.memory_estimates:
                        st.markdown("### System Requirements")
                        st.caption("Steam-style hardware tiers for deployment")

                        try:
                            import logging

                            from haoline.operational_profiling import OperationalProfiler

                            req_logger = logging.getLogger("haoline.sysreq")
                            profiler = OperationalProfiler(logger=req_logger)

                            sys_reqs = profiler.determine_system_requirements(
                                model_params=report.param_counts.total,
                                model_flops=report.flop_counts.total,
                                peak_activation_bytes=report.memory_estimates.peak_activation_bytes,
                                precision="fp16",
                                target_fps=30.0,
                            )

                            req_col1, req_col2, req_col3 = st.columns(3)

                            with req_col1:
                                st.markdown("#### 🟢 Minimum")
                                if sys_reqs.minimum:
                                    st.markdown(f"**{sys_reqs.minimum.device}**")
                                    st.caption(f"{sys_reqs.minimum_vram_gb} GB VRAM")
                                else:
                                    st.caption("N/A")

                            with req_col2:
                                st.markdown("#### 🟡 Recommended")
                                if sys_reqs.recommended:
                                    st.markdown(f"**{sys_reqs.recommended.device}**")
                                    st.caption(f"{sys_reqs.recommended_vram_gb} GB VRAM")
                                else:
                                    st.caption("N/A")

                            with req_col3:
                                st.markdown("#### 🔵 Optimal")
                                if sys_reqs.optimal:
                                    st.markdown(f"**{sys_reqs.optimal.device}**")
                                    st.caption(f"{sys_reqs.optimal_vram_gb} GB VRAM")
                                else:
                                    st.caption("N/A")

                        except Exception as e:
                            st.warning(f"Could not generate system requirements: {e}")

                    # Deployment Cost Calculator (Story 41.3.7)
                    if report.flop_counts and report.memory_estimates:
                        with st.expander("💰 Deployment Cost Calculator", expanded=False):
                            st.caption(
                                "Estimate monthly cloud costs for running this model in production"
                            )

                            # Cloud Instance Selector (Story 41.3.11, 41.4.4)
                            from haoline.eval.deployment import HARDWARE_TIERS

                            tier_options = {
                                k: f"{v.name} (${v.cost_per_hour_usd:.2f}/hr, {v.memory_gb}GB)"
                                for k, v in HARDWARE_TIERS.items()
                            }
                            st.selectbox(
                                "Cloud Instance",
                                options=list(tier_options.keys()),
                                format_func=lambda x: tier_options[x],
                                index=1,  # Default to A10G
                                help="Select cloud GPU instance for cost estimation",
                            )

                            cost_col1, cost_col2 = st.columns(2)
                            with cost_col1:
                                target_fps = st.number_input(
                                    "Target Throughput (FPS)",
                                    min_value=1,
                                    max_value=1000,
                                    value=30,
                                    help="How many inferences per second you need",
                                )
                                hours_per_day = st.slider(
                                    "Hours/Day",
                                    min_value=1,
                                    max_value=24,
                                    value=24,
                                    help="How many hours per day the model runs",
                                )

                            with cost_col2:
                                precision_choice = st.selectbox(
                                    "Precision",
                                    options=["fp32", "fp16", "int8"],
                                    index=1,
                                    help="Inference precision affects speed and cost",
                                )
                                replicas = st.number_input(
                                    "Replicas",
                                    min_value=1,
                                    max_value=10,
                                    value=1,
                                    help="Number of model instances for redundancy",
                                )

                            if st.button("Calculate Cost", type="primary"):
                                try:
                                    from haoline.eval.deployment import (
                                        DeploymentScenario,
                                        calculate_deployment_cost,
                                    )

                                    scenario = DeploymentScenario(
                                        target_fps=float(target_fps),
                                        hours_per_day=float(hours_per_day),
                                        precision=precision_choice,
                                        replicas=replicas,
                                    )

                                    cost_estimate = calculate_deployment_cost(
                                        model_flops=report.flop_counts.total,
                                        scenario=scenario,
                                        model_memory_bytes=report.memory_estimates.model_size_bytes,
                                    )

                                    # Display results
                                    st.markdown("---")
                                    result_col1, result_col2, result_col3 = st.columns(3)

                                    with result_col1:
                                        st.metric(
                                            "Monthly Cost",
                                            f"${cost_estimate.cost_per_month_usd:.2f}",
                                        )

                                    with result_col2:
                                        st.metric(
                                            "Instances Needed",
                                            str(cost_estimate.num_instances),
                                        )

                                    with result_col3:
                                        if cost_estimate.hardware_tier:
                                            st.metric(
                                                "Recommended",
                                                cost_estimate.hardware_tier.name,
                                            )

                                    if cost_estimate.warnings:
                                        for warning in cost_estimate.warnings:
                                            st.warning(warning)

                                except Exception as e:
                                    st.error(f"Cost calculation failed: {e}")

                    # Run Benchmark Button (Story 41.4.2)
                    if not (hasattr(report, "batch_size_sweep") and report.batch_size_sweep):
                        with st.expander("🔬 Run Benchmark (Optional)", expanded=False):
                            st.caption(
                                "Run actual inference benchmarks to find optimal batch size. "
                                "This requires ONNX Runtime and may take a minute."
                            )

                            bench_col1, bench_col2 = st.columns(2)
                            with bench_col1:
                                bench_batch_sizes = st.multiselect(
                                    "Batch Sizes to Test",
                                    options=[1, 2, 4, 8, 16, 32, 64],
                                    default=[1, 2, 4, 8],
                                )
                            with bench_col2:
                                num_runs = st.number_input(
                                    "Warmup + Test Runs",
                                    min_value=3,
                                    max_value=20,
                                    value=5,
                                )

                            if st.button("Run Batch Benchmark", type="secondary"):
                                try:
                                    import logging

                                    from haoline.operational_profiling import OperationalProfiler

                                    bench_logger = logging.getLogger("haoline.bench")
                                    profiler = OperationalProfiler(logger=bench_logger)

                                    with st.spinner("Running benchmark..."):
                                        sweep_result = profiler.sweep_batch_sizes(
                                            model_path=str(tmp_path),
                                            batch_sizes=bench_batch_sizes,
                                            num_runs=num_runs,
                                        )

                                    if sweep_result:
                                        report.batch_size_sweep = sweep_result
                                        st.success(
                                            f"Benchmark complete! Optimal batch: {sweep_result.optimal_batch_size}"
                                        )
                                        st.rerun()
                                    else:
                                        st.warning(
                                            "Benchmark failed - ONNX Runtime may not be available"
                                        )

                                except Exception as e:
                                    st.error(f"Benchmark failed: {e}")

                    # Batch Size Sweep Results (Story 41.3.8)
                    if hasattr(report, "batch_size_sweep") and report.batch_size_sweep:
                        st.markdown("### Batch Size Sweep Results")
                        sweep = report.batch_size_sweep

                        import pandas as pd

                        sweep_col1, sweep_col2 = st.columns(2)
                        with sweep_col1:
                            st.metric("Optimal Batch Size", str(sweep.optimal_batch_size))
                        with sweep_col2:
                            st.metric(
                                "Peak Throughput",
                                f"{max(sweep.throughputs):.1f} inf/s",
                            )

                        # Chart
                        sweep_df = pd.DataFrame(
                            {
                                "Batch Size": sweep.batch_sizes,
                                "Throughput (inf/s)": sweep.throughputs,
                                "Latency (ms)": sweep.latencies,
                            }
                        )
                        st.line_chart(sweep_df.set_index("Batch Size")["Throughput (inf/s)"])

                    # Resolution Sweep Results (Story 41.3.9)
                    if hasattr(report, "resolution_sweep") and report.resolution_sweep:
                        st.markdown("### Resolution Sweep Results")
                        res_sweep = report.resolution_sweep

                        import pandas as pd

                        res_col1, res_col2 = st.columns(2)
                        with res_col1:
                            st.metric("Optimal Resolution", res_sweep.optimal_resolution)
                        with res_col2:
                            st.metric("Max Resolution (fits VRAM)", res_sweep.max_resolution)

                        # Chart
                        res_df = pd.DataFrame(
                            {
                                "Resolution": res_sweep.resolutions,
                                "Throughput (inf/s)": res_sweep.throughputs,
                                "VRAM (GB)": res_sweep.vram_usage_gb,
                            }
                        )
                        st.line_chart(res_df.set_index("Resolution")["Throughput (inf/s)"])

                    # Per-Layer Timing (Story 41.3.10)
                    if (
                        hasattr(report, "extra_data")
                        and report.extra_data
                        and "profiling" in report.extra_data
                    ):
                        st.markdown("### Per-Layer Timing")
                        profiling_data = report.extra_data["profiling"]

                        import pandas as pd

                        if "layer_profiles" in profiling_data and profiling_data["layer_profiles"]:
                            layers = profiling_data["layer_profiles"]

                            # Show total time
                            total_ms = profiling_data.get("total_time_ms", 0)
                            st.metric("Total Inference Time", f"{total_ms:.2f} ms")

                            # Show slowest layers
                            st.markdown("#### Slowest Layers")
                            sorted_layers = sorted(layers, key=lambda x: -x.get("duration_ms", 0))[
                                :10
                            ]

                            timing_df = pd.DataFrame(
                                [
                                    {
                                        "Layer": lp.get("name", "?")[:25],
                                        "Op Type": lp.get("op_type", "?"),
                                        "Time (ms)": f"{lp.get('duration_ms', 0):.3f}",
                                        "Provider": lp.get("provider", "?"),
                                    }
                                    for lp in sorted_layers
                                ]
                            )
                            st.dataframe(timing_df, width="stretch", hide_index=True)

                            # Time by op type chart
                            time_by_op: dict[str, float] = {}
                            for lp in layers:
                                op = lp.get("op_type", "Unknown")
                                time_by_op[op] = time_by_op.get(op, 0) + lp.get("duration_ms", 0)

                            if time_by_op:
                                st.markdown("#### Time by Op Type")
                                op_df = pd.DataFrame(
                                    [
                                        {"Op Type": op, "Time (ms)": t}
                                        for op, t in sorted(
                                            time_by_op.items(), key=lambda x: -x[1]
                                        )[:10]
                                    ]
                                )
                                st.bar_chart(op_df.set_index("Op Type")["Time (ms)"])

                    # AI Summary (if enabled and API key provided)
                    llm_enabled = st.session_state.get("enable_llm", False)
                    llm_api_key = st.session_state.get("openai_api_key_value", "")

                    if llm_enabled:
                        st.markdown("### AI Analysis")

                        if not llm_api_key:
                            st.warning(
                                "AI Summary is enabled but no API key is set. "
                                "Enter your OpenAI API key in the sidebar."
                            )
                        elif not llm_api_key.startswith("sk-"):
                            st.error(
                                f"Invalid API key format. Keys should start with 'sk-'. "
                                f"Got: {llm_api_key[:10]}..."
                            )
                        else:
                            with st.spinner("Generating AI summary..."):
                                try:
                                    from haoline.llm_summarizer import LLMSummarizer

                                    summarizer = LLMSummarizer(api_key=llm_api_key)
                                    llm_result = summarizer.summarize(report)

                                    if llm_result and llm_result.success:
                                        # Short summary
                                        if llm_result.short_summary:
                                            st.markdown(
                                                f"""<div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.05) 100%);
                                                border-left: 4px solid #10b981; border-radius: 8px; padding: 1rem; margin: 1rem 0;">
                                                <p style="font-weight: 600; color: #10b981; margin-bottom: 0.5rem;">AI Summary</p>
                                                <p style="color: #e5e5e5; line-height: 1.6;">{llm_result.short_summary}</p>
                                                </div>""",
                                                unsafe_allow_html=True,
                                            )

                                        # Detailed analysis
                                        if llm_result.detailed_summary:
                                            with st.expander("Detailed Analysis", expanded=True):
                                                st.markdown(llm_result.detailed_summary)

                                        # Show model/tokens info
                                        st.caption(
                                            f"Generated by {llm_result.model_used} "
                                            f"({llm_result.tokens_used} tokens)"
                                        )
                                    elif llm_result and llm_result.error_message:
                                        st.error(f"AI summary failed: {llm_result.error_message}")
                                    else:
                                        st.warning("AI summary generation returned empty result.")
                                except ImportError:
                                    st.error(
                                        "LLM summarizer not available. Install with: "
                                        "`pip install haoline[llm]`"
                                    )
                                except Exception as e:
                                    st.error(f"Error generating AI summary: {e}")

                with tab2:
                    if include_graph:
                        st.markdown("### Interactive Architecture Graph")
                        st.caption(
                            "🖱️ Scroll to zoom | Drag to pan | Click nodes to expand/collapse | Use sidebar controls"
                        )

                        try:
                            # Build the full interactive D3.js graph
                            import logging

                            from haoline.analyzer import ONNXGraphLoader

                            graph_logger = logging.getLogger("haoline.graph")

                            # Load graph info
                            loader = ONNXGraphLoader(logger=graph_logger)
                            _, graph_info = loader.load(tmp_path)

                            # Detect patterns/blocks
                            pattern_analyzer = PatternAnalyzer(logger=graph_logger)
                            blocks = pattern_analyzer.group_into_blocks(graph_info)

                            # Analyze edges
                            edge_analyzer = EdgeAnalyzer(logger=graph_logger)
                            edge_result = edge_analyzer.analyze(graph_info)

                            # Build hierarchical graph
                            builder = HierarchicalGraphBuilder(logger=graph_logger)
                            model_name = Path(uploaded_file.name).stem
                            hier_graph = builder.build(graph_info, blocks, model_name)

                            # Generate the full D3.js HTML
                            # The HTML template auto-detects embedded mode (iframe) and:
                            # - Collapses sidebar for more graph space
                            # - Auto-fits the view
                            graph_html = generate_graph_html(
                                hier_graph,
                                edge_result,
                                title=model_name,
                                model_size_bytes=len(uploaded_file.getvalue()),
                            )

                            # Embed with generous height for comfortable viewing
                            components.html(graph_html, height=800, scrolling=False)

                        except Exception as e:
                            st.warning(f"Could not generate interactive graph: {e}")
                            # Fallback to block list
                            if report.detected_blocks:
                                st.markdown("#### Detected Architecture Blocks")
                                for i, block in enumerate(report.detected_blocks[:15]):
                                    with st.expander(
                                        f"{block.block_type}: {block.name}", expanded=(i < 3)
                                    ):
                                        st.write(f"**Type:** {block.block_type}")
                                        st.write(f"**Nodes:** {len(block.nodes)}")
                    else:
                        st.info(
                            "Enable 'Interactive Graph' in the sidebar to see the architecture visualization."
                        )

                with tab3:
                    # Details tab - patterns and risk signals
                    render_details_tab(report)

                with tab4:
                    # Layer Details tab
                    model_name = uploaded_file.name.replace(".onnx", "")
                    render_layer_details_tab(
                        report,
                        graph_info,
                        model_name,
                        redact_names=redact_names,
                        summary_only=summary_only,
                    )

                with tab5:
                    # Quantization tab
                    render_quantization_tab(report, graph_info)

                with tab6:
                    # Export tab
                    model_name = uploaded_file.name.replace(".onnx", "")
                    render_export_tab(
                        report,
                        model_name,
                        tmp_path,
                        hardware=selected_hardware,
                        batch_size=batch_size,
                    )

        except Exception as e:
            st.error(f"Error analyzing model: {e}")
            st.exception(e)

        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
