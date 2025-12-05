"""
Multi-Model Comparison Module for HaoLine.

Generate comparison tables and reports across multiple models with:
- Architecture metrics (params, FLOPs, memory)
- Evaluation metrics (accuracy, mAP, etc.)
- Hardware estimates (latency, throughput)
- Deployment costs

Supports export to:
- Console tables (rich formatting)
- CSV/JSON for further analysis
- HTML for reports
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any

from .deployment import (
    DeploymentCostEstimate,
    DeploymentScenario,
    calculate_deployment_cost,
)
from .schemas import CombinedReport


@dataclass
class ModelComparisonRow:
    """
    A single row in the comparison table.

    Represents one model with all its metrics for comparison.
    """

    model_id: str
    model_path: str = ""

    # Architecture metrics
    params_total: int = 0
    flops_total: int = 0
    model_size_mb: float = 0.0

    # Primary accuracy metric (task-dependent)
    primary_metric_name: str = ""
    primary_metric_value: float = 0.0

    # Speed metrics
    latency_ms: float = 0.0
    throughput_fps: float = 0.0
    hardware_tier: str = ""

    # Cost metrics
    cost_per_month_usd: float = 0.0
    cost_per_1k_inferences_usd: float = 0.0

    # Additional metrics (for detailed comparison)
    extra_metrics: dict[str, float] = field(default_factory=dict)

    # Source data references
    combined_report: CombinedReport | None = None
    cost_estimate: DeploymentCostEstimate | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_id": self.model_id,
            "model_path": self.model_path,
            "params_total": self.params_total,
            "flops_total": self.flops_total,
            "model_size_mb": self.model_size_mb,
            "primary_metric_name": self.primary_metric_name,
            "primary_metric_value": self.primary_metric_value,
            "latency_ms": self.latency_ms,
            "throughput_fps": self.throughput_fps,
            "hardware_tier": self.hardware_tier,
            "cost_per_month_usd": self.cost_per_month_usd,
            "cost_per_1k_inferences_usd": self.cost_per_1k_inferences_usd,
            "extra_metrics": self.extra_metrics,
        }

    @classmethod
    def from_combined_report(
        cls,
        report: CombinedReport,
        cost_estimate: DeploymentCostEstimate | None = None,
    ) -> ModelComparisonRow:
        """
        Create a comparison row from a CombinedReport.

        Args:
            report: CombinedReport with architecture and eval data.
            cost_estimate: Optional pre-computed cost estimate.

        Returns:
            ModelComparisonRow with extracted metrics.
        """
        arch = report.architecture
        row = cls(
            model_id=report.model_id,
            model_path=report.model_path,
            params_total=arch.get("params_total", 0),
            flops_total=arch.get("flops_total", 0),
            model_size_mb=arch.get("model_size_bytes", 0) / (1024 * 1024),
            primary_metric_name=report.primary_accuracy_metric,
            primary_metric_value=report.primary_accuracy_value,
            latency_ms=report.latency_ms,
            throughput_fps=report.throughput_fps,
            hardware_tier=report.hardware_profile,
            combined_report=report,
        )

        # Add cost if available
        if cost_estimate:
            row.cost_per_month_usd = cost_estimate.cost_per_month_usd
            row.cost_per_1k_inferences_usd = cost_estimate.cost_per_1k_inferences_usd
            row.cost_estimate = cost_estimate
            if not row.hardware_tier:
                row.hardware_tier = cost_estimate.hardware_tier.name

        # Extract extra metrics from eval results
        for eval_result in report.eval_results:
            for metric in eval_result.metrics:
                if metric.name != row.primary_metric_name:
                    row.extra_metrics[metric.name] = metric.value

        return row


@dataclass
class ModelComparisonTable:
    """
    Multi-model comparison table.

    Holds comparison data for multiple models and provides
    various output formats.
    """

    rows: list[ModelComparisonRow] = field(default_factory=list)
    scenario: DeploymentScenario | None = None

    # Table metadata
    title: str = "Model Comparison"
    description: str = ""

    def add_model(
        self,
        report: CombinedReport,
        scenario: DeploymentScenario | None = None,
    ) -> ModelComparisonRow:
        """
        Add a model to the comparison table.

        Args:
            report: CombinedReport with model data.
            scenario: Optional deployment scenario for cost calculation.

        Returns:
            The created ModelComparisonRow.
        """
        # Calculate cost if scenario provided
        cost_estimate = None
        if scenario and report.architecture.get("flops_total"):
            cost_estimate = calculate_deployment_cost(
                report.architecture["flops_total"],
                scenario,
                report.architecture.get("model_size_bytes", 0),
            )

        row = ModelComparisonRow.from_combined_report(report, cost_estimate)
        self.rows.append(row)
        return row

    def sort_by(
        self,
        key: str,
        reverse: bool = False,
    ) -> None:
        """
        Sort the table by a given metric.

        Args:
            key: Attribute name to sort by (e.g., 'primary_metric_value', 'cost_per_month_usd').
            reverse: If True, sort in descending order.
        """
        self.rows.sort(key=lambda r: getattr(r, key, 0), reverse=reverse)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "title": self.title,
            "description": self.description,
            "scenario": self.scenario.to_dict() if self.scenario else None,
            "rows": [row.to_dict() for row in self.rows],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_csv(self) -> str:
        """
        Export comparison table to CSV format.

        Returns:
            CSV string with all comparison metrics.
        """
        if not self.rows:
            return ""

        output = StringIO()
        fieldnames = [
            "model_id",
            "params_total",
            "flops_total",
            "model_size_mb",
            "primary_metric_name",
            "primary_metric_value",
            "latency_ms",
            "throughput_fps",
            "hardware_tier",
            "cost_per_month_usd",
            "cost_per_1k_inferences_usd",
        ]

        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in self.rows:
            writer.writerow(row.to_dict())

        return output.getvalue()

    def to_markdown(self) -> str:
        """
        Generate a markdown table for display.

        Returns:
            Markdown-formatted comparison table.
        """
        if not self.rows:
            return "*No models to compare*"

        lines = []

        # Title
        if self.title:
            lines.append(f"## {self.title}")
            lines.append("")

        if self.description:
            lines.append(self.description)
            lines.append("")

        # Table header
        lines.append(
            "| Model | Params | FLOPs | Size | Accuracy | Latency | Throughput | $/Month |"
        )
        lines.append(
            "|-------|--------|-------|------|----------|---------|------------|---------|"
        )

        # Table rows
        for row in self.rows:
            params_str = _format_number(row.params_total)
            flops_str = _format_number(row.flops_total)
            size_str = f"{row.model_size_mb:.1f} MB"
            acc_str = f"{row.primary_metric_value:.1f}%" if row.primary_metric_value else "N/A"
            lat_str = f"{row.latency_ms:.1f} ms" if row.latency_ms else "N/A"
            thr_str = f"{row.throughput_fps:.1f} fps" if row.throughput_fps else "N/A"
            cost_str = f"${row.cost_per_month_usd:.0f}" if row.cost_per_month_usd else "N/A"

            lines.append(
                f"| {row.model_id} | {params_str} | {flops_str} | {size_str} | "
                f"{acc_str} | {lat_str} | {thr_str} | {cost_str} |"
            )

        return "\n".join(lines)

    def to_console(self) -> str:
        """
        Generate a console-friendly table with rich formatting.

        Returns:
            Formatted table string for terminal output.
        """
        if not self.rows:
            return "No models to compare"

        # Calculate column widths
        headers = ["Model", "Params", "FLOPs", "Size", "Accuracy", "Latency", "$/Month"]
        rows_data = []

        for row in self.rows:
            rows_data.append(
                [
                    row.model_id[:20],  # Truncate long names
                    _format_number(row.params_total),
                    _format_number(row.flops_total),
                    f"{row.model_size_mb:.1f}MB",
                    f"{row.primary_metric_value:.1f}%" if row.primary_metric_value else "N/A",
                    f"{row.latency_ms:.1f}ms" if row.latency_ms else "N/A",
                    f"${row.cost_per_month_usd:.0f}" if row.cost_per_month_usd else "N/A",
                ]
            )

        # Calculate column widths
        widths = [len(h) for h in headers]
        for rd in rows_data:
            for i, cell in enumerate(rd):
                widths[i] = max(widths[i], len(cell))

        # Build table
        def row_str(cells: list[str]) -> str:
            return " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

        lines = [
            f"\n{self.title}",
            "=" * (sum(widths) + len(widths) * 3),
            row_str(headers),
            "-" * (sum(widths) + len(widths) * 3),
        ]
        for rd in rows_data:
            lines.append(row_str(rd))
        lines.append("=" * (sum(widths) + len(widths) * 3))

        return "\n".join(lines)

    def save_csv(self, path: str | Path) -> None:
        """Save comparison to CSV file."""
        Path(path).write_text(self.to_csv(), encoding="utf-8")

    def save_json(self, path: str | Path) -> None:
        """Save comparison to JSON file."""
        Path(path).write_text(self.to_json(), encoding="utf-8")


def _format_number(n: int | float) -> str:
    """Format large numbers with K/M/B suffixes."""
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.1f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.1f}K"
    else:
        return str(int(n))


def compare_models(
    reports: list[CombinedReport],
    scenario: DeploymentScenario | None = None,
    sort_by: str = "primary_metric_value",
    sort_descending: bool = True,
    title: str = "Model Comparison",
) -> ModelComparisonTable:
    """
    Compare multiple models and generate a comparison table.

    This is the main entry point for model comparison.

    Args:
        reports: List of CombinedReport objects to compare.
        scenario: Optional deployment scenario for cost calculation.
        sort_by: Metric to sort by (default: primary accuracy).
        sort_descending: Sort order (default: highest first).
        title: Table title.

    Returns:
        ModelComparisonTable with all models.

    Example:
        >>> reports = [report1, report2, report3]
        >>> scenario = DeploymentScenario.realtime_video(fps=30)
        >>> table = compare_models(reports, scenario)
        >>> print(table.to_console())
    """
    table = ModelComparisonTable(
        title=title,
        scenario=scenario,
    )

    for report in reports:
        table.add_model(report, scenario)

    if sort_by:
        table.sort_by(sort_by, reverse=sort_descending)

    return table


def compare_models_from_paths(
    model_paths: list[str | Path],
    eval_paths: list[str | Path] | None = None,
    scenario: DeploymentScenario | None = None,
) -> ModelComparisonTable:
    """
    Compare models from file paths.

    Runs haoline analysis on each model and optionally imports eval results.

    Args:
        model_paths: List of paths to model files.
        eval_paths: Optional list of eval result files (matched by index).
        scenario: Deployment scenario for cost calculation.

    Returns:
        ModelComparisonTable with comparison data.
    """
    from .schemas import create_combined_report

    reports = []

    for i, model_path in enumerate(model_paths):
        # Import eval if available
        eval_results = None
        if eval_paths and i < len(eval_paths):
            from .adapters import detect_and_parse

            eval_result = detect_and_parse(Path(eval_paths[i]))
            if eval_result:
                eval_results = [eval_result]

        # Create combined report
        report = create_combined_report(
            str(model_path),
            eval_results=eval_results,
            run_inspection=True,
        )
        reports.append(report)

    return compare_models(reports, scenario)
