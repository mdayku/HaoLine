# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
HaoLine (皓线) - Universal Model Inspector.

See what's really inside your models.

This module provides utilities for analyzing neural network models to extract:
- Parameter counts and memory estimates
- FLOP estimates per operation
- Architectural pattern detection (transformers, CNNs, residual blocks)
- Risk signals for deployment
- Hardware performance estimates

Supports: ONNX, PyTorch, TensorFlow, TensorRT

Example usage:
    from haoline import ModelInspector

    inspector = ModelInspector()
    report = inspector.inspect("model.onnx")
    print(report.to_json())
"""

__version__ = "0.9.11"

from .analyzer import MetricsEngine, ONNXGraphLoader
from .compare_visualizations import (
    CalibrationRecommendation,
    LayerPrecisionBreakdown,
    NormalizedMetrics,
    TradeoffPoint,
    analyze_tradeoffs,
    build_enhanced_markdown,
    compute_normalized_metrics,
    compute_tradeoff_points,
    extract_layer_precision_breakdown,
    generate_calibration_recommendations,
    generate_compare_html,
    generate_compare_pdf,
    generate_layer_precision_chart,
    generate_memory_savings_chart,
    generate_radar_chart,
    generate_tradeoff_chart,
)
from .compare_visualizations import is_available as is_compare_viz_available
from .edge_analysis import EdgeAnalysisResult, EdgeAnalyzer
from .hardware import (
    HARDWARE_PROFILES,
    HardwareDetector,
    HardwareEstimates,
    HardwareEstimator,
    HardwareProfile,
    detect_local_hardware,
    get_profile,
    list_available_profiles,
)
from .hierarchical_graph import (
    HierarchicalGraph,
    HierarchicalGraphBuilder,
    HierarchicalNode,
)
from .hierarchical_graph import generate_summary as generate_graph_summary
from .html_export import HTMLExporter
from .html_export import generate_html as generate_graph_html
from .layer_summary import (
    LayerMetrics,
    LayerSummary,
    LayerSummaryBuilder,
    generate_html_table,
    generate_markdown_table,
)
from .llm_summarizer import LLMSummarizer, LLMSummary, summarize_report
from .llm_summarizer import has_api_key as has_llm_api_key
from .llm_summarizer import is_available as is_llm_available
from .operational_profiling import (
    BatchSizeSweep,
    BatchSweepPoint,
    BottleneckAnalysis,
    GPUMetrics,
    LayerProfile,
    OperationalProfiler,
    ProfilingResult,
    ResolutionPoint,
    ResolutionSweep,
    SystemRequirements,
)
from .patterns import PatternAnalyzer
from .pdf_generator import PDFGenerator, generate_pdf
from .pdf_generator import is_available as is_pdf_available
from .quantization_advisor import (
    ArchitectureType,
    DeploymentRuntime,
    FakeQuantInsertionPoint,
    OpSubstitution,
    QuantGranularityRec,
    QuantizationAdvice,
    QuantizationAdvisor,
    advise_quantization,
    generate_qat_readiness_report,
)
from .quantization_linter import (
    LayerRiskScore,
    QuantIssueType,
    QuantizationLinter,
    QuantizationLintResult,
    QuantWarning,
    Severity,
    lint_model,
)
from .report import (
    DatasetInfo,
    InspectionReport,
    ModelInspector,
    infer_num_classes_from_output,
)
from .report_sections import (
    BlockSummaryItem,
    BottleneckSection,
    DetectedBlocksSection,
    ExtractedReportSections,
    HardwareEstimatesSection,
    KVCacheSection,
    MemoryBreakdownRow,
    MemoryBreakdownSection,
    MetricsCard,
    MetricsSummary,
    OperatorDistribution,
    PrecisionBreakdown,
    PrecisionBreakdownRow,
    RiskSignalItem,
    RiskSignalsSection,
    SharedWeightsSection,
    format_bytes,
    format_number,
)
from .risks import RiskAnalyzer, RiskSignal, RiskThresholds
from .schema import (
    INSPECTION_REPORT_SCHEMA,
    ValidationError,
    get_schema,
    validate_report,
    validate_report_strict,
)
from .universal_ir import (
    DataType,
    GraphMetadata,
    SourceFormat,
    TensorOrigin,
    UniversalGraph,
    UniversalNode,
    UniversalTensor,
)
from .visualizations import (
    THEME,
    ChartTheme,
    VisualizationGenerator,
    generate_visualizations,
)
from .visualizations import is_available as is_visualization_available

__all__ = [
    # Universal IR (Epic 18)
    "DataType",
    "GraphMetadata",
    "SourceFormat",
    "TensorOrigin",
    "UniversalGraph",
    "UniversalNode",
    "UniversalTensor",
    # Core
    "HARDWARE_PROFILES",
    "INSPECTION_REPORT_SCHEMA",
    "THEME",
    "BatchSizeSweep",
    "BatchSweepPoint",
    "BottleneckAnalysis",
    "CalibrationRecommendation",
    "ChartTheme",
    "DatasetInfo",
    "EdgeAnalysisResult",
    "EdgeAnalyzer",
    "GPUMetrics",
    "HTMLExporter",
    "HardwareDetector",
    "HardwareEstimates",
    "HardwareEstimator",
    "HardwareProfile",
    "HierarchicalGraph",
    "HierarchicalGraphBuilder",
    "HierarchicalNode",
    "InspectionReport",
    "LLMSummarizer",
    "LLMSummary",
    "LayerMetrics",
    "LayerPrecisionBreakdown",
    "LayerProfile",
    "LayerSummary",
    "LayerSummaryBuilder",
    "MetricsEngine",
    "ModelInspector",
    "NormalizedMetrics",
    "ONNXGraphLoader",
    "OperationalProfiler",
    "PDFGenerator",
    "PatternAnalyzer",
    "ProfilingResult",
    "ResolutionPoint",
    "ResolutionSweep",
    "RiskAnalyzer",
    "RiskSignal",
    "RiskThresholds",
    "QuantizationLinter",
    "QuantizationLintResult",
    "QuantWarning",
    "LayerRiskScore",
    "Severity",
    "QuantIssueType",
    "lint_model",
    "QuantizationAdvisor",
    "QuantizationAdvice",
    "ArchitectureType",
    "DeploymentRuntime",
    "FakeQuantInsertionPoint",
    "OpSubstitution",
    "QuantGranularityRec",
    "advise_quantization",
    "generate_qat_readiness_report",
    "SystemRequirements",
    "TradeoffPoint",
    "ValidationError",
    "VisualizationGenerator",
    "analyze_tradeoffs",
    "build_enhanced_markdown",
    "compute_normalized_metrics",
    "compute_tradeoff_points",
    "detect_local_hardware",
    "extract_layer_precision_breakdown",
    "generate_calibration_recommendations",
    "generate_compare_html",
    "generate_compare_pdf",
    "generate_graph_html",
    "generate_graph_summary",
    "generate_html_table",
    "generate_layer_precision_chart",
    "generate_markdown_table",
    "generate_memory_savings_chart",
    "generate_pdf",
    "generate_radar_chart",
    "generate_tradeoff_chart",
    "generate_visualizations",
    "get_profile",
    "get_schema",
    "has_llm_api_key",
    "infer_num_classes_from_output",
    "is_compare_viz_available",
    "is_llm_available",
    "is_pdf_available",
    "is_visualization_available",
    "list_available_profiles",
    "summarize_report",
    "validate_report",
    "validate_report_strict",
    # Report Sections (Story 41.2)
    "BlockSummaryItem",
    "BottleneckSection",
    "DetectedBlocksSection",
    "ExtractedReportSections",
    "HardwareEstimatesSection",
    "KVCacheSection",
    "MemoryBreakdownRow",
    "MemoryBreakdownSection",
    "MetricsCard",
    "MetricsSummary",
    "OperatorDistribution",
    "PrecisionBreakdown",
    "PrecisionBreakdownRow",
    "RiskSignalItem",
    "RiskSignalsSection",
    "SharedWeightsSection",
    "format_bytes",
    "format_number",
]
