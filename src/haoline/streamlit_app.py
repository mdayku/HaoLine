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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

# Page config must be first Streamlit command
st.set_page_config(
    page_title="HaoLine - Model Inspector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


@dataclass
class AnalysisResult:
    """Stored analysis result for session history."""

    name: str
    timestamp: datetime
    report: Any  # InspectionReport
    file_size: int

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
    if "demo_model" not in st.session_state:
        st.session_state.demo_model = None  # Tuple of (bytes, name) when demo requested


def add_to_history(name: str, report: Any, file_size: int) -> AnalysisResult:
    """Add an analysis result to session history."""
    result = AnalysisResult(
        name=name,
        timestamp=datetime.now(),
        report=report,
        file_size=file_size,
    )
    # Keep max 10 results, newest first
    st.session_state.analysis_history.insert(0, result)
    if len(st.session_state.analysis_history) > 10:
        st.session_state.analysis_history.pop()
    return result


# Import haoline after page config
import streamlit.components.v1 as components

from haoline import ModelInspector, __version__

# Demo models from ONNX Model Zoo (small, real models)
DEMO_MODELS = {
    "mnist": {
        "name": "MNIST CNN",
        "file": "mnist-12.onnx",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12.onnx",
        "description": "Tiny CNN for handwritten digits (26 KB)",
        "size": "26 KB",
    },
    "squeezenet": {
        "name": "SqueezeNet 1.0",
        "file": "squeezenet1.0-12.onnx",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.0-12.onnx",
        "description": "Compact CNN for ImageNet (5 MB)",
        "size": "5 MB",
    },
    "efficientnet": {
        "name": "EfficientNet-Lite4",
        "file": "efficientnet-lite4-11.onnx",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx",
        "description": "Efficient CNN architecture (49 MB)",
        "size": "49 MB",
    },
}


def download_demo_model(model_key: str) -> tuple[bytes, str]:
    """Download a demo model from ONNX Model Zoo.

    Args:
        model_key: Key from DEMO_MODELS dict

    Returns:
        Tuple of (model_bytes, model_name)
    """
    import urllib.request

    model_info = DEMO_MODELS[model_key]
    url = model_info["url"]
    filename = model_info["file"]

    # Download with timeout
    with urllib.request.urlopen(url, timeout=30) as response:
        model_bytes = response.read()

    return model_bytes, filename


from haoline.analyzer import ONNXGraphLoader
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

    /* Header styling */
    .main-header {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #10b981 0%, #34d399 50%, #6ee7b7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
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

    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: var(--accent-primary) !important;
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

    /* Sidebar - remove top padding */
    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:first-child {
        padding-top: 0 !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem !important;
    }

    /* Sidebar section headers */
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--accent-primary) !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 1rem !important;
        margin-bottom: 0.5rem !important;
        font-weight: 600 !important;
    }

    /* First header in sidebar - no top margin */
    [data-testid="stSidebar"] .stMarkdown:first-of-type h3 {
        margin-top: 0 !important;
    }

    /* Tighten sidebar dividers */
    [data-testid="stSidebar"] hr {
        margin: 0.75rem 0 !important;
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
    """Render CLI-style model comparison report."""
    import pandas as pd

    # Extract all metrics
    params_a = model_a.report.param_counts.total if model_a.report.param_counts else 0
    params_b = model_b.report.param_counts.total if model_b.report.param_counts else 0
    flops_a = model_a.report.flop_counts.total if model_a.report.flop_counts else 0
    flops_b = model_b.report.flop_counts.total if model_b.report.flop_counts else 0
    size_a = (
        model_a.report.memory_estimates.model_size_bytes if model_a.report.memory_estimates else 0
    )
    size_b = (
        model_b.report.memory_estimates.model_size_bytes if model_b.report.memory_estimates else 0
    )
    ops_a = model_a.report.graph_summary.num_nodes
    ops_b = model_b.report.graph_summary.num_nodes

    # Precision detection
    bytes_per_param_a = (size_a / params_a) if params_a > 0 else 0
    bytes_per_param_b = (size_b / params_b) if params_b > 0 else 0

    def get_precision(bpp: float) -> str:
        if bpp < 1.5:
            return "INT8"
        elif bpp < 2.5:
            return "FP16"
        elif bpp < 4.5:
            return "FP32"
        return "FP64"

    precision_a = get_precision(bytes_per_param_a)
    precision_b = get_precision(bytes_per_param_b)

    # Size ratio (B relative to A)
    size_ratio = size_b / size_a if size_a > 0 else 1.0

    # Title
    st.markdown(
        f"""
    <h2 style="margin-bottom: 0.25rem;">Quantization Impact Report</h2>
    <p style="color: #a3a3a3; font-size: 0.9rem;">
        Baseline: <strong>{model_a.name}</strong> ({precision_a})
    </p>
    """,
        unsafe_allow_html=True,
    )

    # Trade-off Analysis box
    st.markdown("### Trade-off Analysis")

    # Determine best characteristics
    smaller = model_b.name if size_b < size_a else model_a.name
    fewer_params = model_b.name if params_b < params_a else model_a.name
    fewer_flops = model_b.name if flops_b < flops_a else model_a.name

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Smallest",
            smaller,
            f"{(1 - min(size_a, size_b) / max(size_a, size_b)) * 100:.1f}% smaller",
        )
    with col2:
        st.metric("Fewest Params", fewer_params)
    with col3:
        st.metric("Fewest FLOPs", fewer_flops)

    # Recommendations
    st.markdown("#### Recommendations")
    recommendations = []

    if precision_a != precision_b:
        if size_ratio < 0.6:
            recommendations.append(
                f"**{model_b.name}** offers **{(1 - size_ratio) * 100:.0f}% smaller** model size "
                f"({precision_a} → {precision_b})"
            )
        if size_ratio > 1.4:
            recommendations.append(
                f"**{model_a.name}** is **{(1 - 1 / size_ratio) * 100:.0f}% smaller** than {model_b.name}"
            )

    if abs(size_ratio - 0.5) < 0.1 and precision_b == "FP16":
        recommendations.append(
            "FP16 achieves expected ~50% size reduction with minimal accuracy impact"
        )
    elif abs(size_ratio - 0.25) < 0.1 and precision_b == "INT8":
        recommendations.append(
            "INT8 achieves expected ~75% size reduction - verify accuracy on your dataset"
        )

    if params_a == params_b and flops_a == flops_b:
        recommendations.append("Same architecture - only precision/quantization differs")

    if not recommendations:
        recommendations.append("Models have similar characteristics")

    for rec in recommendations:
        st.markdown(f"- {rec}")

    st.markdown("---")

    # Variant Comparison Table (CLI-style)
    st.markdown("### Variant Comparison")

    table_data = [
        {
            "Model": model_a.name,
            "Precision": precision_a,
            "Size": format_bytes(size_a),
            "Params": format_number(params_a),
            "FLOPs": format_number(flops_a),
            "Size vs Baseline": "baseline",
            "Ops": ops_a,
        },
        {
            "Model": model_b.name,
            "Precision": precision_b,
            "Size": format_bytes(size_b),
            "Params": format_number(params_b),
            "FLOPs": format_number(flops_b),
            "Size vs Baseline": f"{size_ratio:.2f}x ({(size_ratio - 1) * 100:+.1f}%)",
            "Ops": ops_b,
        },
    ]

    df = pd.DataFrame(table_data)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Model": st.column_config.TextColumn("Model", width="medium"),
            "Precision": st.column_config.TextColumn("Precision", width="small"),
            "Size": st.column_config.TextColumn("Size", width="small"),
            "Params": st.column_config.TextColumn("Params", width="small"),
            "FLOPs": st.column_config.TextColumn("FLOPs", width="small"),
            "Size vs Baseline": st.column_config.TextColumn("Δ Size", width="medium"),
            "Ops": st.column_config.NumberColumn("Ops", width="small"),
        },
    )

    st.markdown("---")

    # Memory Savings visualization
    st.markdown("### Memory Comparison")

    col1, col2 = st.columns(2)

    with col1:
        # Size comparison bar
        size_data = pd.DataFrame(
            {"Model": [model_a.name, model_b.name], "Size (MB)": [size_a / 1e6, size_b / 1e6]}
        )
        st.bar_chart(size_data.set_index("Model"), height=200)
        st.caption("Model Size (weights)")

    with col2:
        # Savings indicator
        if size_a > size_b:
            savings_pct = (1 - size_b / size_a) * 100
            savings_bytes = size_a - size_b
            st.markdown(
                f"""
            <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.1) 100%);
                        border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 12px; padding: 1.5rem; text-align: center;">
                <div style="font-size: 2rem; font-weight: 700; color: #10b981;">{savings_pct:.1f}%</div>
                <div style="color: #a3a3a3; font-size: 0.9rem;">Size Reduction</div>
                <div style="color: #6b7280; font-size: 0.8rem; margin-top: 0.5rem;">
                    Saves {format_bytes(savings_bytes)}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        elif size_b > size_a:
            increase_pct = (size_b / size_a - 1) * 100
            st.markdown(
                f"""
            <div style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.1) 100%);
                        border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 12px; padding: 1.5rem; text-align: center;">
                <div style="font-size: 2rem; font-weight: 700; color: #ef4444;">+{increase_pct:.1f}%</div>
                <div style="color: #a3a3a3; font-size: 0.9rem;">Size Increase</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Models have identical size")

    st.markdown("---")

    # Operator Distribution
    st.markdown("### Operator Distribution")

    ops_a_dict = model_a.report.graph_summary.op_type_counts or {}
    ops_b_dict = model_b.report.graph_summary.op_type_counts or {}
    all_ops = sorted(set(ops_a_dict.keys()) | set(ops_b_dict.keys()))

    if all_ops:
        op_data = []
        for op in all_ops:
            count_a = ops_a_dict.get(op, 0)
            count_b = ops_b_dict.get(op, 0)
            if count_a > 0 or count_b > 0:
                op_data.append({"Operator": op, model_a.name: count_a, model_b.name: count_b})

        op_df = pd.DataFrame(op_data)
        st.bar_chart(op_df.set_index("Operator"), height=300)

        with st.expander("View operator details"):
            # Add difference column
            for row in op_data:
                row["Difference"] = row[model_b.name] - row[model_a.name]
            detail_df = pd.DataFrame(op_data)
            st.dataframe(detail_df, use_container_width=True, hide_index=True)

    # Architecture compatibility check
    if params_a != params_b or flops_a != flops_b:
        st.markdown("---")
        st.warning(
            "**Architecture Difference Detected**: Models have different parameter counts or FLOPs. "
            "This may indicate structural changes beyond precision conversion."
        )

    # Universal IR Structural Comparison (if available)
    ir_a = model_a.report.universal_graph
    ir_b = model_b.report.universal_graph

    if ir_a and ir_b:
        st.markdown("---")
        st.markdown("### Structural Analysis (Universal IR)")

        # Check structural equality
        is_equal = ir_a.is_structurally_equal(ir_b)

        if is_equal:
            st.success(
                "**Architectures are structurally identical** — same ops in same order. "
                "Differences are limited to precision/weights."
            )
        else:
            st.warning(
                "**Structural differences detected** — graphs differ in ops or connectivity."
            )

            # Show detailed diff
            with st.expander("View IR Diff", expanded=True):
                diff_result = ir_a.diff(ir_b)

                diff_cols = st.columns(2)

                with diff_cols[0]:
                    st.markdown("**Summary:**")
                    st.text(f"  Node count: {ir_a.num_nodes} → {ir_b.num_nodes}")
                    st.text(f"  Parameters: {ir_a.total_parameters:,} → {ir_b.total_parameters:,}")

                with diff_cols[1]:
                    st.markdown("**Changes:**")
                    if diff_result.get("node_count_diff", 0) != 0:
                        st.text(f"  Δ Nodes: {diff_result['node_count_diff']:+d}")
                    if diff_result.get("param_count_diff", 0) != 0:
                        st.text(f"  Δ Params: {diff_result['param_count_diff']:+,}")

                # Op type differences
                op_diff = diff_result.get("op_type_diff", {})
                if op_diff:
                    st.markdown("**Op Type Changes:**")
                    added = op_diff.get("added", [])
                    removed = op_diff.get("removed", [])
                    if added:
                        st.text(f"  Added: {', '.join(added)}")
                    if removed:
                        st.text(f"  Removed: {', '.join(removed)}")

    # Footer
    st.markdown("---")
    st.caption("*Generated by HaoLine Compare Mode*")


def render_compare_mode():
    """Render the model comparison interface."""
    model_a = st.session_state.compare_models.get("model_a")
    model_b = st.session_state.compare_models.get("model_b")

    # Show comparison if both models are selected
    if model_a and model_b:
        # Clear selection buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Clear Comparison", type="secondary", use_container_width=True):
                st.session_state.compare_models = {"model_a": None, "model_b": None}
                st.rerun()

        render_comparison_view(model_a, model_b)
        return

    # Model selection interface
    st.markdown("## Compare Two Models")
    st.markdown("Upload two models at once, or select from session history.")

    # Quick dual upload
    with st.expander("📤 Quick Upload (both models at once)", expanded=not (model_a or model_b)):
        dual_files = st.file_uploader(
            "Select two ONNX models",
            type=["onnx"],
            accept_multiple_files=True,
            key="dual_upload",
            help="Select exactly 2 models to compare",
        )
        if dual_files:
            if len(dual_files) == 2:
                with st.spinner("Analyzing both models..."):
                    result_a = analyze_model_file(dual_files[0])
                    result_b = analyze_model_file(dual_files[1])
                    if result_a and result_b:
                        st.session_state.compare_models["model_a"] = result_a
                        st.session_state.compare_models["model_b"] = result_b
                        st.rerun()
            elif len(dual_files) == 1:
                st.warning("Please select 2 models to compare")
            else:
                st.warning(f"Please select exactly 2 models (you selected {len(dual_files)})")

    st.markdown("**Or select individually:**")
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
                index=(
                    list(hw_options.keys()).index(default_key) if default_key in hw_options else 0
                ),
            )
        else:
            st.warning("No GPUs match your search. Try a different query.")
            selected_hardware = "auto"

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

        # LLM Summary
        st.markdown("### AI Summary")
        enable_llm = st.checkbox("Generate AI Summary", value=False, help="Requires OpenAI API key")

        openai_api_key = None
        if enable_llm:
            openai_api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Used once per analysis, never stored",
                key="openai_api_key",
            )
            if not openai_api_key:
                st.warning("Enter your OpenAI API key to generate AI summaries.")
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
            type=["onnx", "pt", "pth", "safetensors"],
            help="ONNX (recommended), PyTorch (.pt/.pth), or SafeTensors",
        )

        if uploaded_file is None:
            st.markdown(
                """
            <div style="text-align: center; padding: 1rem 2rem; margin-top: -0.5rem;">
                <p style="font-size: 0.9rem; margin-bottom: 0.75rem; color: #a3a3a3;">
                    <span style="color: #10b981; font-weight: 600;">ONNX</span> ✓ &nbsp;&nbsp;
                    <span style="color: #a3a3a3;">PyTorch</span> ↻ &nbsp;&nbsp;
                    <span style="color: #a3a3a3;">SafeTensors</span> ↻
                </p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Demo model options
            st.markdown(
                """<div style="text-align: center; margin: 1rem 0 0.5rem 0;">
                    <span style="font-size: 0.9rem; color: #a3a3a3; font-weight: 500;">
                        No model handy? Try a demo:
                    </span>
                </div>""",
                unsafe_allow_html=True,
            )

            # Demo model buttons in a row
            demo_cols = st.columns(len(DEMO_MODELS))
            for i, (key, info) in enumerate(DEMO_MODELS.items()):
                with demo_cols[i]:
                    if st.button(
                        f"{info['name']}\n({info['size']})",
                        key=f"demo_{key}",
                        use_container_width=True,
                        help=info["description"],
                    ):
                        with st.spinner(f"Downloading {info['name']}..."):
                            try:
                                demo_bytes, demo_name = download_demo_model(key)
                                st.session_state.demo_model = (demo_bytes, demo_name)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to download: {e}")

            st.markdown(
                """<div style="text-align: center; margin-top: 1rem;">
                <p style="font-size: 0.8rem; color: #737373;">
                    Or browse the
                    <a href="https://huggingface.co/models?library=onnx" target="_blank"
                       style="color: #10b981; text-decoration: none;">HuggingFace ONNX Hub</a>
                </p>
            </div>""",
                unsafe_allow_html=True,
            )

    # Handle demo model if requested
    demo_model_bytes = None
    demo_model_name = None
    if st.session_state.demo_model is not None:
        demo_model_bytes, demo_model_name = st.session_state.demo_model
        st.session_state.demo_model = None  # Clear after use

    # Analysis - either uploaded file or demo model
    if uploaded_file is not None or demo_model_bytes is not None:
        if demo_model_bytes is not None:
            # Use demo model
            file_ext = ".onnx"
            file_name = demo_model_name
            file_bytes = demo_model_bytes
        else:
            # Use uploaded file
            file_ext = Path(uploaded_file.name).suffix.lower()
            file_name = uploaded_file.name
            file_bytes = uploaded_file.getvalue()

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
                    pt_tmp.write(file_bytes)
                    pt_path = pt_tmp.name

                # Attempt conversion
                with st.spinner("Converting PyTorch → ONNX..."):
                    try:
                        # Try TorchScript first
                        try:
                            model = torch.jit.load(pt_path, map_location="cpu")
                            is_ultralytics = False
                        except Exception:
                            loaded = torch.load(pt_path, map_location="cpu", weights_only=False)
                            is_ultralytics = False

                            if isinstance(loaded, dict):
                                # Check for Ultralytics YOLO format
                                if "model" in loaded and hasattr(loaded.get("model"), "forward"):
                                    is_ultralytics = True
                                else:
                                    st.error(
                                        """
                                    **State dict detected** — This file contains only weights, not the model architecture.

                                    To analyze, you need the full model. Export to ONNX from your training code:
                                    ```python
                                    torch.onnx.export(model, dummy_input, "model.onnx")
                                    ```
                                    """
                                    )
                                    st.stop()

                            if not is_ultralytics:
                                model = loaded

                        # Handle Ultralytics models with their native export
                        if is_ultralytics:
                            try:
                                from ultralytics import YOLO

                                st.info("🔄 Ultralytics YOLO detected — using native export...")
                                yolo_model = YOLO(pt_path)
                                onnx_tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
                                yolo_model.export(
                                    format="onnx",
                                    imgsz=input_shape[2] if len(input_shape) >= 3 else 640,
                                    simplify=True,
                                )
                                # Ultralytics saves next to .pt, move to our temp
                                import shutil

                                default_onnx = Path(pt_path).with_suffix(".onnx")
                                if default_onnx.exists():
                                    shutil.move(str(default_onnx), onnx_tmp.name)
                                tmp_path = onnx_tmp.name
                                st.success("✅ YOLO conversion successful!")
                            except ImportError:
                                st.error(
                                    "**Ultralytics required** — Install with: `pip install ultralytics`"
                                )
                                st.stop()
                        else:
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
                        st.error(
                            f"""
                        **Conversion failed:** {str(e)[:200]}

                        Try exporting to ONNX directly from your training code, or use the CLI:
                        ```bash
                        haoline --from-pytorch model.pt --input-shape {input_shape_str} --html
                        ```
                        """
                        )
                        st.stop()
            else:
                st.warning(
                    f"""
                **PyTorch model detected**, but PyTorch is not installed in this environment.

                **Options:**
                1. Use the CLI locally (supports conversion):
                   ```bash
                   pip install haoline torch
                   haoline --from-pytorch {file_name} --input-shape 1,3,224,224 --html
                   ```

                2. Convert to ONNX first in your code:
                   ```python
                   torch.onnx.export(model, dummy_input, "model.onnx")
                   ```
                """
                )
                st.stop()

        elif file_ext == ".safetensors":
            st.warning(
                """
            **SafeTensors format detected** — This format contains only weights, not architecture.

            To analyze, export to ONNX from your training code. If using HuggingFace:
            ```python
            from optimum.exporters.onnx import main_export
            main_export("model-name", output="model.onnx")
            ```
            """
            )
            st.stop()

        # Save ONNX to temp file (if not already set by conversion)
        if tmp_path is None:
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
                tmp.write(file_bytes)
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
                    )

                # Save to session history
                add_to_history(file_name, report, len(file_bytes))

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

                # Tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs(
                    ["Overview", "Interactive Graph", "Details", "Export"]
                )

                with tab1:
                    st.markdown("### Model Information")

                    info_col1, info_col2 = st.columns(2)

                    with info_col1:
                        st.markdown(
                            f"""
                        | Property | Value |
                        |----------|-------|
                        | **Model** | `{file_name}` |
                        | **IR Version** | {report.metadata.ir_version} |
                        | **Producer** | {report.metadata.producer_name or "Unknown"} |
                        | **Opset** | {list(report.metadata.opsets.values())[0] if report.metadata.opsets else "Unknown"} |
                        """
                        )

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

                        st.markdown(
                            f"""
                        | Metric | Value |
                        |--------|-------|
                        | **Total Parameters** | {params_total:,} |
                        | **Total FLOPs** | {flops_total:,} |
                        | **Peak Memory** | {format_bytes(peak_mem)} |
                        | **Model Size** | {format_bytes(model_size)} |
                        """
                        )

                    # AI Summary (if enabled and API key provided)
                    if enable_llm and openai_api_key:
                        st.markdown("### AI Analysis")
                        with st.spinner("Generating AI summary..."):
                            try:
                                from haoline.llm_summarizer import LLMSummarizer

                                summarizer = LLMSummarizer(api_key=openai_api_key)
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
                                    "LLM module not available. Install with: `pip install haoline[llm]`"
                                )
                            except Exception as e:
                                st.error(f"AI summary generation failed: {e}")

                    # Universal IR Summary (if available)
                    if report.universal_graph:
                        with st.expander("Universal IR View", expanded=False):
                            ir = report.universal_graph
                            st.markdown(
                                f"""
                                <div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(79, 70, 229, 0.05) 100%);
                                border-left: 4px solid #6366f1; border-radius: 8px; padding: 1rem; margin: 0.5rem 0;">
                                <p style="font-weight: 600; color: #6366f1; margin-bottom: 0.5rem;">Format-Agnostic Graph</p>
                                <p style="color: #e5e5e5; line-height: 1.6;">
                                <strong>Source:</strong> {ir.metadata.source_format.value.upper()}<br>
                                <strong>Nodes:</strong> {ir.num_nodes}<br>
                                <strong>Tensors:</strong> {len(ir.tensors)}<br>
                                <strong>Parameters:</strong> {ir.total_parameters:,}<br>
                                <strong>Weight Size:</strong> {ir.total_weight_bytes / (1024 * 1024):.2f} MB
                                </p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            # Op type distribution from IR
                            op_counts = ir.op_type_counts
                            if op_counts:
                                st.markdown("**Operation Types (from IR):**")
                                top_ops = sorted(
                                    op_counts.items(), key=lambda x: x[1], reverse=True
                                )[:10]
                                for op, count in top_ops:
                                    st.text(f"  {op}: {count}")

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

                with tab2:
                    if include_graph:
                        st.markdown("### Interactive Architecture Graph")
                        st.caption(
                            "🖱️ Scroll to zoom | Drag to pan | Click nodes to expand/collapse | Use sidebar controls"
                        )

                        try:
                            # Build the full interactive D3.js graph
                            import logging

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
                            model_name = Path(file_name).stem
                            hier_graph = builder.build(graph_info, blocks, model_name)

                            # Generate the full D3.js HTML
                            # The HTML template auto-detects embedded mode (iframe) and:
                            # - Collapses sidebar for more graph space
                            # - Auto-fits the view
                            graph_html = generate_graph_html(
                                hier_graph,
                                edge_result,
                                title=model_name,
                                model_size_bytes=len(file_bytes),
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
                    st.markdown("### Detected Patterns")

                    if report.detected_blocks:
                        for block in report.detected_blocks[:10]:  # Limit to first 10
                            with st.expander(f"{block.block_type}: {block.name}"):
                                st.write(
                                    f"**Nodes:** {', '.join(block.nodes[:5])}{'...' if len(block.nodes) > 5 else ''}"
                                )
                    else:
                        st.info("No architectural patterns detected.")

                    st.markdown("### Risk Signals")

                    if report.risk_signals:
                        for risk in report.risk_signals:
                            severity_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                                risk.severity, "⚪"
                            )

                            st.markdown(f"{severity_color} **{risk.id}** ({risk.severity})")
                            st.caption(risk.description)
                    else:
                        st.success("No risk signals detected!")

                with tab4:
                    model_name = file_name.replace(".onnx", "")

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

                    # Generate all export data
                    json_data = report.to_json()
                    md_data = report.to_markdown()
                    html_data = report.to_html()

                    # Try to generate PDF
                    pdf_data = None
                    try:
                        from haoline.pdf_generator import (
                            PDFGenerator,
                        )
                        from haoline.pdf_generator import (
                            is_available as pdf_available,
                        )

                        if pdf_available():
                            import tempfile as tf_pdf

                            pdf_gen = PDFGenerator()
                            with tf_pdf.NamedTemporaryFile(suffix=".pdf", delete=False) as pdf_tmp:
                                if pdf_gen.generate_from_html(html_data, pdf_tmp.name):
                                    with open(pdf_tmp.name, "rb") as f:
                                        pdf_data = f.read()
                    except Exception:
                        pass

                    # Custom styled export grid
                    st.markdown(
                        """
                    <style>
                        .export-grid {
                            display: grid;
                            grid-template-columns: repeat(2, 1fr);
                            gap: 1rem;
                            margin-top: 1rem;
                        }
                        .export-card {
                            background: #1a1a1a;
                            border: 1px solid rgba(255,255,255,0.1);
                            border-radius: 12px;
                            padding: 1.25rem;
                            transition: all 0.2s ease;
                        }
                        .export-card:hover {
                            border-color: #10b981;
                            background: #1f1f1f;
                        }
                        .export-icon {
                            font-size: 1.5rem;
                            margin-bottom: 0.5rem;
                        }
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
                            use_container_width=True,
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
                            use_container_width=True,
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
                            use_container_width=True,
                        )

                    with col4:
                        if pdf_data:
                            st.markdown(
                                """
                            <div class="export-card">
                                <div class="export-icon">📑</div>
                                <div class="export-title">PDF Report</div>
                                <div class="export-desc">Print-ready document for sharing</div>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )
                            st.download_button(
                                label="Download PDF",
                                data=pdf_data,
                                file_name=f"{model_name}_report.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                            )
                        else:
                            st.markdown(
                                """
                            <div class="export-card" style="opacity: 0.5;">
                                <div class="export-icon">📑</div>
                                <div class="export-title">PDF Report</div>
                                <div class="export-desc">Requires Playwright · Use CLI for PDF export</div>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )
                            st.button("PDF unavailable", disabled=True, use_container_width=True)

        except Exception as e:
            st.error(f"Error analyzing model: {e}")
            st.exception(e)

        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
