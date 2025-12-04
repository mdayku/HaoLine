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

import io
import tempfile
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Any

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


def add_to_history(name: str, report: Any, file_size: int):
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
from haoline import ModelInspector, __version__
from haoline.hardware import HARDWARE_PROFILES, get_profile, HardwareEstimator, detect_local_hardware
from haoline.analyzer import ONNXGraphLoader
from haoline.patterns import PatternAnalyzer
from haoline.edge_analysis import EdgeAnalyzer
from haoline.hierarchical_graph import HierarchicalGraphBuilder
from haoline.html_export import generate_html as generate_graph_html
import streamlit.components.v1 as components

# Custom CSS - Sleek dark theme with mint/emerald accents
st.markdown("""
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
""", unsafe_allow_html=True)


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
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                    padding: 1rem 1.5rem; border-radius: 12px; text-align: center;">
            <div style="font-size: 0.75rem; color: rgba(255,255,255,0.7); text-transform: uppercase; letter-spacing: 0.1em;">
                Model A
            </div>
            <div style="font-size: 1.25rem; font-weight: 600; color: white; margin-top: 0.25rem;">
                {model_a.name}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%); 
                    padding: 1rem 1.5rem; border-radius: 12px; text-align: center;">
            <div style="font-size: 0.75rem; color: rgba(255,255,255,0.7); text-transform: uppercase; letter-spacing: 0.1em;">
                Model B
            </div>
            <div style="font-size: 1.25rem; font-weight: 600; color: white; margin-top: 0.25rem;">
                {model_b.name}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Metrics comparison
    st.markdown("### Key Metrics")
    
    # Get metrics
    params_a = model_a.report.param_counts.total if model_a.report.param_counts else 0
    params_b = model_b.report.param_counts.total if model_b.report.param_counts else 0
    flops_a = model_a.report.flop_counts.total if model_a.report.flop_counts else 0
    flops_b = model_b.report.flop_counts.total if model_b.report.flop_counts else 0
    mem_a = model_a.report.memory_estimates.peak_activation_bytes if model_a.report.memory_estimates else 0
    mem_b = model_b.report.memory_estimates.peak_activation_bytes if model_b.report.memory_estimates else 0
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
            st.markdown(f"<span style='color: {color};'>Δ {diff_pct:+.1f}%</span>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("**FLOPs**")
        st.markdown(f"🟢 A: **{format_number(flops_a)}**")
        st.markdown(f"🟣 B: **{format_number(flops_b)}**")
        if flops_a != flops_b:
            diff_pct = ((flops_b - flops_a) / flops_a * 100) if flops_a else 0
            color = "#ef4444" if diff_pct > 0 else "#10b981"
            st.markdown(f"<span style='color: {color};'>Δ {diff_pct:+.1f}%</span>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("**Peak Memory**")
        st.markdown(f"🟢 A: **{format_bytes(mem_a)}**")
        st.markdown(f"🟣 B: **{format_bytes(mem_b)}**")
        if mem_a != mem_b:
            diff_pct = ((mem_b - mem_a) / mem_a * 100) if mem_a else 0
            color = "#ef4444" if diff_pct > 0 else "#10b981"
            st.markdown(f"<span style='color: {color};'>Δ {diff_pct:+.1f}%</span>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("**Operators**")
        st.markdown(f"🟢 A: **{ops_a}**")
        st.markdown(f"🟣 B: **{ops_b}**")
        if ops_a != ops_b:
            diff_pct = ((ops_b - ops_a) / ops_a * 100) if ops_a else 0
            color = "#ef4444" if diff_pct > 0 else "#10b981"
            st.markdown(f"<span style='color: {color};'>Δ {diff_pct:+.1f}%</span>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Operator distribution comparison
    st.markdown("### Operator Distribution Comparison")
    
    import pandas as pd
    
    # Merge operator counts
    ops_a_dict = model_a.report.graph_summary.op_type_counts or {}
    ops_b_dict = model_b.report.graph_summary.op_type_counts or {}
    all_ops = set(ops_a_dict.keys()) | set(ops_b_dict.keys())
    
    comparison_data = []
    for op in sorted(all_ops):
        count_a = ops_a_dict.get(op, 0)
        count_b = ops_b_dict.get(op, 0)
        comparison_data.append({
            "Operator": op,
            f"Model A ({model_a.name})": count_a,
            f"Model B ({model_b.name})": count_b,
            "Difference": count_b - count_a,
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Bar chart
    chart_df = df.set_index("Operator")[[f"Model A ({model_a.name})", f"Model B ({model_b.name})"]]
    st.bar_chart(chart_df)
    
    # Table
    with st.expander("View detailed comparison table"):
        st.dataframe(df, use_container_width=True)
    
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
        summary_points.append(f"Model B requires **{reduction:.1f}% fewer FLOPs** (faster inference)")
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
    st.markdown("Select models from your session history, or upload new models to compare.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.05) 100%); 
                    border: 2px dashed rgba(16, 185, 129, 0.3); border-radius: 16px; padding: 2rem; text-align: center;">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🟢</div>
            <div style="font-size: 1rem; font-weight: 600; color: #10b981;">Model A</div>
        </div>
        """, unsafe_allow_html=True)
        
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
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(79, 70, 229, 0.05) 100%); 
                    border: 2px dashed rgba(99, 102, 241, 0.3); border-radius: 16px; padding: 2rem; text-align: center;">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🟣</div>
            <div style="font-size: 1rem; font-weight: 600; color: #6366f1;">Model B</div>
        </div>
        """, unsafe_allow_html=True)
        
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
        st.info("💡 **Tip:** First analyze some models in **Analyze** mode. They'll appear in your session history for easy comparison.")


def analyze_model_file(uploaded_file) -> Optional[AnalysisResult]:
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
        "🔧 Auto": {
            "auto": {"name": "Auto-detect local GPU", "vram": 0, "tflops": 0}
        },
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
        elif "rtx40" in name_lower or "4090" in name_lower or "4080" in name_lower or "4070" in name_lower or "4060" in name_lower:
            categories["🎮 Consumer - RTX 40 Series"][name] = entry
        elif "rtx30" in name_lower or "3090" in name_lower or "3080" in name_lower or "3070" in name_lower or "3060" in name_lower:
            categories["🎮 Consumer - RTX 30 Series"][name] = entry
        elif any(x in name_lower for x in ["rtxa", "a6000", "a5000", "a4000"]):
            categories["💼 Workstation"][name] = entry
        elif "jetson" in name_lower or "orin" in name_lower or "xavier" in name_lower or "nano" in name_lower:
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
    st.markdown('<p class="sub-header">Universal Model Inspector — See what\'s really inside your neural networks</p>', unsafe_allow_html=True)
    
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
                    st.markdown(f"""
                    <div style="font-size: 0.85rem; color: #f5f5f5; margin-bottom: 0.1rem;">
                        {result.name[:20]}{'...' if len(result.name) > 20 else ''}
                    </div>
                    <div style="font-size: 0.7rem; color: #737373;">
                        {result.summary} · {time_str}
                    </div>
                    """, unsafe_allow_html=True)
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
                all_hardware.append({
                    "key": hw_key,
                    "display": display_name,
                    "category": category,
                    "vram": hw_info["vram"],
                    "tflops": hw_info["tflops"],
                })
        
        # Filter by search
        if search_query:
            filtered_hardware = [
                h for h in all_hardware 
                if search_query.lower() in h["display"].lower() or search_query.lower() in h["key"].lower()
            ]
        else:
            filtered_hardware = all_hardware
        
        # Category filter
        available_categories = sorted(set(h["category"] for h in filtered_hardware))
        if len(available_categories) > 1:
            selected_category = st.selectbox(
                "Category",
                options=["All Categories"] + available_categories,
                index=0,
            )
            if selected_category != "All Categories":
                filtered_hardware = [h for h in filtered_hardware if h["category"] == selected_category]
        
        # Final hardware dropdown
        if filtered_hardware:
            hw_options = {h["key"]: h["display"] for h in filtered_hardware}
            default_key = "auto" if "auto" in hw_options else list(hw_options.keys())[0]
            selected_hardware = st.selectbox(
                "Select GPU",
                options=list(hw_options.keys()),
                format_func=lambda x: hw_options[x],
                index=list(hw_options.keys()).index(default_key) if default_key in hw_options else 0,
            )
        else:
            st.warning("No GPUs match your search. Try a different query.")
            selected_hardware = "auto"
        
        # Show selected hardware specs
        if selected_hardware != "auto":
            try:
                profile = HARDWARE_PROFILES.get(selected_hardware)
                if profile:
                    st.markdown(f"""
                    <div style="background: #1f1f1f; 
                                border: 1px solid rgba(16, 185, 129, 0.2);
                                padding: 0.75rem 1rem; border-radius: 10px; margin-top: 0.5rem;">
                        <div style="font-size: 0.85rem; color: #10b981; font-weight: 600;">
                            {profile.name}
                        </div>
                        <div style="font-size: 0.75rem; color: #737373; margin-top: 0.25rem; font-family: 'SF Mono', monospace;">
                            {profile.vram_bytes // (1024**3)} GB VRAM · {profile.peak_fp16_tflops or '—'} TF
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception:
                pass
        
        # Analysis options
        st.markdown("### Analysis Options")
        include_graph = st.checkbox("Interactive Graph", value=True, help="Include zoomable D3.js network visualization")
        include_charts = st.checkbox("Charts", value=True, help="Include matplotlib visualizations")
        
        # LLM Summary
        st.markdown("### AI Summary")
        enable_llm = st.checkbox("Generate AI Summary", value=False, help="Requires OpenAI API key")
        
        api_key = None
        if enable_llm:
            api_key = st.text_input(
                "OpenAI API Key", 
                type="password", 
                help="Used once per analysis, never stored"
            )
            st.caption("For maximum security, run `haoline` locally instead.")
        
        # Privacy notice
        st.markdown("---")
        st.markdown(
            '<div class="privacy-notice">'
            '<strong>Privacy:</strong> Models and API keys are processed in memory only. '
            'Nothing is stored. For sensitive work, self-host with <code>pip install haoline[web]</code> '
            'and run <code>streamlit run streamlit_app.py</code> locally.'
            '</div>',
            unsafe_allow_html=True
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
            st.markdown("""
            <div style="text-align: center; padding: 1rem 2rem; margin-top: -0.5rem;">
                <p style="font-size: 0.9rem; margin-bottom: 0.75rem; color: #a3a3a3;">
                    <span style="color: #10b981; font-weight: 600;">ONNX</span> ✓ &nbsp;&nbsp;
                    <span style="color: #a3a3a3;">PyTorch</span> ↻ &nbsp;&nbsp;
                    <span style="color: #a3a3a3;">SafeTensors</span> ↻
                </p>
                <p style="font-size: 0.8rem; color: #737373;">
                    Need a model? Browse the 
                    <a href="https://huggingface.co/models?library=onnx" target="_blank" style="color: #10b981; text-decoration: none;">HuggingFace ONNX Hub →</a>
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Analysis
    if uploaded_file is not None:
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
                st.info("**PyTorch model detected** — We'll try to convert it to ONNX for analysis.")
                
                # Input shape is required for conversion
                input_shape_str = st.text_input(
                    "Input Shape (required)",
                    placeholder="1,3,224,224",
                    help="Batch, Channels, Height, Width for image models. E.g., 1,3,224,224"
                )
                
                if not input_shape_str:
                    st.warning("⚠️ Please enter the input shape to convert and analyze this model.")
                    st.caption("**Common shapes:** `1,3,224,224` (ResNet), `1,3,384,384` (ViT-Large), `1,768` (BERT tokens)")
                    st.stop()
                
                # Try conversion
                try:
                    input_shape = tuple(int(x.strip()) for x in input_shape_str.split(","))
                except ValueError:
                    st.error(f"Invalid input shape: `{input_shape_str}`. Use comma-separated integers like `1,3,224,224`")
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
                
                if profile and report.param_counts and report.flop_counts and report.memory_estimates:
                    estimator = HardwareEstimator()
                    report.hardware_profile = profile
                    report.hardware_estimates = estimator.estimate(
                        model_params=report.param_counts.total,
                        model_flops=report.flop_counts.total,
                        peak_activation_bytes=report.memory_estimates.peak_activation_bytes,
                        hardware=profile,
                    )
                
                # Save to session history
                add_to_history(uploaded_file.name, report, len(uploaded_file.getvalue()))
                
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
                    memory = report.memory_estimates.peak_activation_bytes if report.memory_estimates else 0
                    st.metric("Memory", format_bytes(memory))
                
                with col4:
                    st.metric("Operators", str(report.graph_summary.num_nodes))
                
                # Tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Interactive Graph", "Details", "Export"])
                
                with tab1:
                    st.markdown("### Model Information")
                    
                    info_col1, info_col2 = st.columns(2)
                    
                    with info_col1:
                        st.markdown(f"""
                        | Property | Value |
                        |----------|-------|
                        | **Model** | `{uploaded_file.name}` |
                        | **IR Version** | {report.metadata.ir_version} |
                        | **Producer** | {report.metadata.producer_name or 'Unknown'} |
                        | **Opset** | {list(report.metadata.opsets.values())[0] if report.metadata.opsets else 'Unknown'} |
                        """)
                    
                    with info_col2:
                        params_total = report.param_counts.total if report.param_counts else 0
                        flops_total = report.flop_counts.total if report.flop_counts else 0
                        peak_mem = report.memory_estimates.peak_activation_bytes if report.memory_estimates else 0
                        model_size = report.memory_estimates.model_size_bytes if report.memory_estimates else 0
                        
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
                        op_data = pd.DataFrame([
                            {"Operator": op, "Count": count}
                            for op, count in sorted(
                                report.graph_summary.op_type_counts.items(),
                                key=lambda x: x[1],
                                reverse=True
                            )
                        ])
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
                        st.caption("🖱️ Scroll to zoom | Drag to pan | Click nodes to expand/collapse | Use sidebar controls")
                        
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
                                    with st.expander(f"{block.block_type}: {block.name}", expanded=(i < 3)):
                                        st.write(f"**Type:** {block.block_type}")
                                        st.write(f"**Nodes:** {len(block.nodes)}")
                    else:
                        st.info("Enable 'Interactive Graph' in the sidebar to see the architecture visualization.")
                
                with tab3:
                    st.markdown("### Detected Patterns")
                    
                    if report.detected_blocks:
                        for block in report.detected_blocks[:10]:  # Limit to first 10
                            with st.expander(f"{block.block_type}: {block.name}"):
                                st.write(f"**Nodes:** {', '.join(block.nodes[:5])}{'...' if len(block.nodes) > 5 else ''}")
                    else:
                        st.info("No architectural patterns detected.")
                    
                    st.markdown("### Risk Signals")
                    
                    if report.risk_signals:
                        for risk in report.risk_signals:
                            severity_color = {
                                "high": "🔴",
                                "medium": "🟡", 
                                "low": "🟢"
                            }.get(risk.severity, "⚪")
                            
                            st.markdown(f"{severity_color} **{risk.id}** ({risk.severity})")
                            st.caption(risk.description)
                    else:
                        st.success("No risk signals detected!")
                
                with tab4:
                    model_name = uploaded_file.name.replace('.onnx', '')
                    
                    st.markdown("""
                    <div style="margin-bottom: 1.5rem;">
                        <h3 style="color: #f5f5f5; margin-bottom: 0.25rem;">Export Reports</h3>
                        <p style="color: #737373; font-size: 0.9rem; margin: 0;">
                            Download your analysis in various formats
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Generate all export data
                    json_data = report.to_json()
                    md_data = report.to_markdown()
                    html_data = report.to_html()
                    
                    # Try to generate PDF
                    pdf_data = None
                    try:
                        from haoline.pdf_generator import PDFGenerator, is_available as pdf_available
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
                    st.markdown("""
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
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("""
                        <div class="export-card">
                            <div class="export-icon">📊</div>
                            <div class="export-title">HTML Report</div>
                            <div class="export-desc">Interactive report with D3.js graph visualization</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.download_button(
                            label="Download HTML",
                            data=html_data,
                            file_name=f"{model_name}_report.html",
                            mime="text/html",
                            use_container_width=True,
                        )
                    
                    with col2:
                        st.markdown("""
                        <div class="export-card">
                            <div class="export-icon">📄</div>
                            <div class="export-title">JSON Data</div>
                            <div class="export-desc">Raw analysis data for programmatic use</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.download_button(
                            label="Download JSON",
                            data=json_data,
                            file_name=f"{model_name}_report.json",
                            mime="application/json",
                            use_container_width=True,
                        )
                    
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        st.markdown("""
                        <div class="export-card">
                            <div class="export-icon">📝</div>
                            <div class="export-title">Markdown</div>
                            <div class="export-desc">Text report for docs, READMEs, or wikis</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.download_button(
                            label="Download Markdown",
                            data=md_data,
                            file_name=f"{model_name}_report.md",
                            mime="text/markdown",
                            use_container_width=True,
                        )
                    
                    with col4:
                        if pdf_data:
                            st.markdown("""
                            <div class="export-card">
                                <div class="export-icon">📑</div>
                                <div class="export-title">PDF Report</div>
                                <div class="export-desc">Print-ready document for sharing</div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.download_button(
                                label="Download PDF",
                                data=pdf_data,
                                file_name=f"{model_name}_report.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                            )
                        else:
                            st.markdown("""
                            <div class="export-card" style="opacity: 0.5;">
                                <div class="export-icon">📑</div>
                                <div class="export-title">PDF Report</div>
                                <div class="export-desc">Requires Playwright · Use CLI for PDF export</div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.button("PDF unavailable", disabled=True, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error analyzing model: {e}")
            st.exception(e)
        
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()

