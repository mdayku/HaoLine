# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
Reusable report section generators for HaoLine.

This module provides format-agnostic functions that generate report sections
as structured data, which can then be rendered to HTML, Markdown, or Streamlit.

Story 41.2: Extract report sections into reusable functions for CLI-Streamlit parity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from .operational_profiling import BottleneckAnalysis
    from .report import InspectionReport


def format_number(n: int | float) -> str:
    """Format a number with SI suffixes (K, M, B, T)."""
    if n >= 1e12:
        return f"{n / 1e12:.1f}T"
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(int(n))


def format_bytes(b: int | float) -> str:
    """Format bytes with appropriate units."""
    if b >= 1024**4:
        return f"{b / 1024**4:.2f} TB"
    if b >= 1024**3:
        return f"{b / 1024**3:.2f} GB"
    if b >= 1024**2:
        return f"{b / 1024**2:.2f} MB"
    if b >= 1024:
        return f"{b / 1024:.2f} KB"
    return f"{int(b)} bytes"


class MetricsCard(BaseModel):
    """A single metrics card for display."""

    model_config = ConfigDict(frozen=True)

    label: str
    value: str
    raw_value: int | float | None = None
    color: str | None = None  # Optional accent color


class MetricsSummary(BaseModel):
    """Summary metrics for a model."""

    model_config = ConfigDict(frozen=True)

    cards: list[MetricsCard] = Field(default_factory=list)

    @classmethod
    def from_report(cls, report: InspectionReport) -> MetricsSummary:
        """Build metrics summary from an InspectionReport."""
        cards = []

        # Parameters
        if report.param_counts:
            cards.append(
                MetricsCard(
                    label="Parameters",
                    value=format_number(report.param_counts.total),
                    raw_value=report.param_counts.total,
                )
            )

        # FLOPs
        if report.flop_counts:
            cards.append(
                MetricsCard(
                    label="FLOPs",
                    value=format_number(report.flop_counts.total),
                    raw_value=report.flop_counts.total,
                )
            )

        # Model Size
        if report.memory_estimates:
            cards.append(
                MetricsCard(
                    label="Model Size",
                    value=format_bytes(report.memory_estimates.model_size_bytes),
                    raw_value=report.memory_estimates.model_size_bytes,
                )
            )

        # Peak Memory
        if report.memory_estimates:
            cards.append(
                MetricsCard(
                    label="Peak Memory",
                    value=format_bytes(report.memory_estimates.peak_activation_bytes),
                    raw_value=report.memory_estimates.peak_activation_bytes,
                )
            )

        # Operators
        if report.graph_summary:
            cards.append(
                MetricsCard(
                    label="Operators",
                    value=str(report.graph_summary.num_nodes),
                    raw_value=report.graph_summary.num_nodes,
                )
            )

        # Architecture Type
        if report.architecture_type and report.architecture_type != "unknown":
            cards.append(
                MetricsCard(
                    label="Architecture",
                    value=report.architecture_type.upper(),
                )
            )

        # Quantization indicator
        if report.param_counts and report.param_counts.is_quantized:
            cards.append(
                MetricsCard(
                    label="Quantized",
                    value="Yes",
                    color="#4CAF50",  # Green
                )
            )

        return cls(cards=cards)


class KVCacheSection(BaseModel):
    """KV Cache analysis for transformer models."""

    model_config = ConfigDict(frozen=True)

    bytes_per_token: int
    bytes_full_context: int
    num_layers: int | None = None
    hidden_dim: int | None = None
    seq_len: int | None = None

    @classmethod
    def from_report(cls, report: InspectionReport) -> KVCacheSection | None:
        """Extract KV cache info from report, or None if not a transformer."""
        if not report.memory_estimates:
            return None
        if report.memory_estimates.kv_cache_bytes_per_token <= 0:
            return None

        config = report.memory_estimates.kv_cache_config or {}
        return cls(
            bytes_per_token=report.memory_estimates.kv_cache_bytes_per_token,
            bytes_full_context=report.memory_estimates.kv_cache_bytes_full_context,
            num_layers=config.get("num_layers"),
            hidden_dim=config.get("hidden_dim"),
            seq_len=config.get("seq_len"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "bytes_per_token": self.bytes_per_token,
            "bytes_per_token_formatted": format_bytes(self.bytes_per_token),
            "bytes_full_context": self.bytes_full_context,
            "bytes_full_context_formatted": format_bytes(self.bytes_full_context),
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "seq_len": self.seq_len,
        }


class PrecisionBreakdownRow(BaseModel):
    """A row in the precision breakdown table."""

    model_config = ConfigDict(frozen=True)

    dtype: str
    count: int
    percentage: float


class PrecisionBreakdown(BaseModel):
    """Precision breakdown for model parameters."""

    model_config = ConfigDict(frozen=True)

    rows: list[PrecisionBreakdownRow] = Field(default_factory=list)
    is_quantized: bool = False
    quantized_ops: list[str] = Field(default_factory=list)

    @classmethod
    def from_report(cls, report: InspectionReport) -> PrecisionBreakdown | None:
        """Extract precision breakdown from report."""
        if not report.param_counts:
            return None
        if not report.param_counts.precision_breakdown:
            return None

        total = sum(report.param_counts.precision_breakdown.values())
        if total == 0:
            return None

        rows = []
        for dtype, count in sorted(
            report.param_counts.precision_breakdown.items(),
            key=lambda x: -x[1],
        ):
            rows.append(
                PrecisionBreakdownRow(
                    dtype=dtype,
                    count=count,
                    percentage=100.0 * count / total,
                )
            )

        return cls(
            rows=rows,
            is_quantized=report.param_counts.is_quantized,
            quantized_ops=report.param_counts.quantized_ops or [],
        )


class MemoryBreakdownRow(BaseModel):
    """A row in the memory breakdown table."""

    model_config = ConfigDict(frozen=True)

    component: str
    size_bytes: int
    percentage: float | None = None


class MemoryBreakdownSection(BaseModel):
    """Memory breakdown by op type."""

    model_config = ConfigDict(frozen=True)

    weights_by_op: list[MemoryBreakdownRow] = Field(default_factory=list)
    activations_by_op: list[MemoryBreakdownRow] = Field(default_factory=list)
    total_weights: int = 0
    total_activations: int = 0

    @classmethod
    def from_report(cls, report: InspectionReport) -> MemoryBreakdownSection | None:
        """Extract memory breakdown from report."""
        if not report.memory_estimates:
            return None
        if not report.memory_estimates.breakdown:
            return None

        bd = report.memory_estimates.breakdown
        weights_rows = []
        activations_rows = []

        # Weights by op type
        if bd.weights_by_op_type:
            total_w = sum(bd.weights_by_op_type.values())
            for op_type, size in sorted(bd.weights_by_op_type.items(), key=lambda x: -x[1])[:10]:
                weights_rows.append(
                    MemoryBreakdownRow(
                        component=op_type,
                        size_bytes=size,
                        percentage=100.0 * size / total_w if total_w > 0 else 0,
                    )
                )

        # Activations by op type
        if bd.activations_by_op_type:
            total_a = sum(bd.activations_by_op_type.values())
            for op_type, size in sorted(bd.activations_by_op_type.items(), key=lambda x: -x[1])[
                :10
            ]:
                activations_rows.append(
                    MemoryBreakdownRow(
                        component=op_type,
                        size_bytes=size,
                        percentage=100.0 * size / total_a if total_a > 0 else 0,
                    )
                )

        if not weights_rows and not activations_rows:
            return None

        return cls(
            weights_by_op=weights_rows,
            activations_by_op=activations_rows,
            total_weights=sum(bd.weights_by_op_type.values()) if bd.weights_by_op_type else 0,
            total_activations=sum(bd.activations_by_op_type.values())
            if bd.activations_by_op_type
            else 0,
        )


class HardwareEstimatesSection(BaseModel):
    """Hardware performance estimates."""

    model_config = ConfigDict(frozen=True)

    device: str
    precision: str
    batch_size: int
    vram_required_bytes: int
    fits_in_vram: bool
    theoretical_latency_ms: float
    compute_utilization: float
    gpu_saturation: float
    bottleneck: str

    @classmethod
    def from_report(cls, report: InspectionReport) -> HardwareEstimatesSection | None:
        """Extract hardware estimates from report."""
        if not report.hardware_estimates:
            return None

        hw = report.hardware_estimates
        return cls(
            device=hw.device,
            precision=hw.precision,
            batch_size=hw.batch_size,
            vram_required_bytes=hw.vram_required_bytes,
            fits_in_vram=hw.fits_in_vram,
            theoretical_latency_ms=hw.theoretical_latency_ms,
            compute_utilization=hw.compute_utilization_estimate,
            gpu_saturation=hw.gpu_saturation,
            bottleneck=hw.bottleneck,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for display."""
        return {
            "device": self.device,
            "precision": self.precision,
            "batch_size": self.batch_size,
            "vram_required": format_bytes(self.vram_required_bytes),
            "vram_required_bytes": self.vram_required_bytes,
            "fits_in_vram": self.fits_in_vram,
            "theoretical_latency_ms": round(self.theoretical_latency_ms, 2),
            "compute_utilization": round(self.compute_utilization * 100, 1),
            "gpu_saturation": round(self.gpu_saturation * 100, 4),
            "bottleneck": self.bottleneck,
        }


class BottleneckSection(BaseModel):
    """Bottleneck analysis results."""

    model_config = ConfigDict(frozen=True)

    classification: str  # "compute_bound", "memory_bound", "balanced"
    compute_time_ms: float
    memory_time_ms: float
    ratio: float
    recommendations: list[str] = Field(default_factory=list)

    @classmethod
    def from_bottleneck_analysis(cls, analysis: BottleneckAnalysis) -> BottleneckSection:
        """Create from BottleneckAnalysis object."""
        return cls(
            classification=analysis.bottleneck_type,
            compute_time_ms=analysis.compute_time_ms,
            memory_time_ms=analysis.memory_time_ms,
            ratio=analysis.compute_ratio,
            recommendations=analysis.recommendations,
        )


class OperatorDistribution(BaseModel):
    """Operator type distribution."""

    model_config = ConfigDict(frozen=True)

    op_counts: dict[str, int]
    total_ops: int

    @classmethod
    def from_report(cls, report: InspectionReport) -> OperatorDistribution | None:
        """Extract operator distribution from report."""
        if not report.graph_summary:
            return None
        if not report.graph_summary.op_type_counts:
            return None

        return cls(
            op_counts=report.graph_summary.op_type_counts,
            total_ops=report.graph_summary.num_nodes,
        )

    def top_n(self, n: int = 10) -> list[tuple[str, int, float]]:
        """Get top N operators with counts and percentages."""
        sorted_ops = sorted(self.op_counts.items(), key=lambda x: -x[1])
        return [
            (op, count, 100.0 * count / self.total_ops if self.total_ops > 0 else 0)
            for op, count in sorted_ops[:n]
        ]


class RiskSignalItem(BaseModel):
    """A single risk signal for display."""

    model_config = ConfigDict(frozen=True)

    id: str
    severity: str  # "high", "medium", "low"
    description: str
    nodes: list[str] = Field(default_factory=list)


class RiskSignalsSection(BaseModel):
    """Risk signals summary."""

    model_config = ConfigDict(frozen=True)

    signals: list[RiskSignalItem] = Field(default_factory=list)
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0

    @classmethod
    def from_report(cls, report: InspectionReport) -> RiskSignalsSection:
        """Extract risk signals from report."""
        signals = []
        high = medium = low = 0

        for risk in report.risk_signals or []:
            signals.append(
                RiskSignalItem(
                    id=risk.id,
                    severity=risk.severity,
                    description=risk.description,
                    nodes=risk.nodes or [],
                )
            )
            if risk.severity == "high":
                high += 1
            elif risk.severity == "medium":
                medium += 1
            else:
                low += 1

        return cls(
            signals=signals,
            high_count=high,
            medium_count=medium,
            low_count=low,
        )


class BlockSummaryItem(BaseModel):
    """A single detected block for display."""

    model_config = ConfigDict(frozen=True)

    block_type: str
    name: str
    node_count: int
    nodes: list[str] = Field(default_factory=list)


class DetectedBlocksSection(BaseModel):
    """Detected architecture blocks summary."""

    model_config = ConfigDict(frozen=True)

    blocks: list[BlockSummaryItem] = Field(default_factory=list)
    block_type_counts: dict[str, int] = Field(default_factory=dict)

    @classmethod
    def from_report(cls, report: InspectionReport) -> DetectedBlocksSection:
        """Extract detected blocks from report."""
        blocks = []
        type_counts: dict[str, int] = {}

        for block in report.detected_blocks or []:
            blocks.append(
                BlockSummaryItem(
                    block_type=block.block_type,
                    name=block.name,
                    node_count=len(block.nodes) if block.nodes else 0,
                    nodes=block.nodes or [],
                )
            )
            type_counts[block.block_type] = type_counts.get(block.block_type, 0) + 1

        return cls(blocks=blocks, block_type_counts=type_counts)


class SharedWeightsSection(BaseModel):
    """Shared weights information."""

    model_config = ConfigDict(frozen=True)

    num_shared: int
    shared_weights: dict[str, list[str]]  # weight_name -> list of nodes using it

    @classmethod
    def from_report(cls, report: InspectionReport) -> SharedWeightsSection | None:
        """Extract shared weights info from report."""
        if not report.param_counts:
            return None
        if report.param_counts.num_shared_weights <= 0:
            return None

        return cls(
            num_shared=report.param_counts.num_shared_weights,
            shared_weights=report.param_counts.shared_weights or {},
        )


# =============================================================================
# Full Report Extraction
# =============================================================================


class ExtractedReportSections(BaseModel):
    """All extracted sections from an InspectionReport."""

    model_config = ConfigDict(frozen=True)

    metrics: MetricsSummary
    kv_cache: KVCacheSection | None
    precision: PrecisionBreakdown | None
    memory_breakdown: MemoryBreakdownSection | None
    hardware: HardwareEstimatesSection | None
    operators: OperatorDistribution | None
    risks: RiskSignalsSection
    blocks: DetectedBlocksSection
    shared_weights: SharedWeightsSection | None

    @classmethod
    def from_report(cls, report: InspectionReport) -> ExtractedReportSections:
        """Extract all sections from an InspectionReport."""
        return cls(
            metrics=MetricsSummary.from_report(report),
            kv_cache=KVCacheSection.from_report(report),
            precision=PrecisionBreakdown.from_report(report),
            memory_breakdown=MemoryBreakdownSection.from_report(report),
            hardware=HardwareEstimatesSection.from_report(report),
            operators=OperatorDistribution.from_report(report),
            risks=RiskSignalsSection.from_report(report),
            blocks=DetectedBlocksSection.from_report(report),
            shared_weights=SharedWeightsSection.from_report(report),
        )
