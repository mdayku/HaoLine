# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
Per-Layer Summary Table for HaoLine.

Story 5.8: Creates sortable, filterable tables showing per-layer metrics
(params, FLOPs, latency estimate, memory).
"""

from __future__ import annotations

import csv
import io
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from .analyzer import FlopCounts, GraphInfo, MemoryEstimates, ParamCounts
    from .report import InspectionReport


class LayerMetrics(BaseModel):
    """Metrics for a single layer/node."""

    model_config = ConfigDict(frozen=True)

    name: str
    op_type: str
    input_shapes: list[str] = Field(default_factory=list)
    output_shapes: list[str] = Field(default_factory=list)
    params: int = 0
    flops: int = 0
    memory_bytes: int = 0
    latency_ms: float = 0.0  # Estimated
    pct_params: float = 0.0
    pct_flops: float = 0.0
    depth: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export."""
        return {
            "name": self.name,
            "op_type": self.op_type,
            "input_shapes": self.input_shapes,
            "output_shapes": self.output_shapes,
            "params": self.params,
            "flops": self.flops,
            "memory_bytes": self.memory_bytes,
            "latency_ms": self.latency_ms,
            "pct_params": self.pct_params,
            "pct_flops": self.pct_flops,
            "depth": self.depth,
        }


class LayerSummary(BaseModel):
    """Complete layer summary for a model."""

    model_config = ConfigDict(frozen=True)

    layers: list[LayerMetrics] = Field(default_factory=list)
    total_params: int = 0
    total_flops: int = 0
    total_memory: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "layers": [layer.to_dict() for layer in self.layers],
            "total_params": self.total_params,
            "total_flops": self.total_flops,
            "total_memory": self.total_memory,
        }

    def to_json(self, indent: int = 2) -> str:
        """Export to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_csv(self) -> str:
        """
        Export to CSV format.

        Task 5.8.4: Export table as CSV.
        """
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(
            [
                "Layer Name",
                "Op Type",
                "Input Shape",
                "Output Shape",
                "Parameters",
                "FLOPs",
                "Memory (bytes)",
                "Latency (ms)",
                "% Params",
                "% FLOPs",
            ]
        )

        # Data rows
        for layer in self.layers:
            writer.writerow(
                [
                    layer.name,
                    layer.op_type,
                    "; ".join(layer.input_shapes),
                    "; ".join(layer.output_shapes),
                    layer.params,
                    layer.flops,
                    layer.memory_bytes,
                    f"{layer.latency_ms:.4f}",
                    f"{layer.pct_params:.2f}",
                    f"{layer.pct_flops:.2f}",
                ]
            )

        return output.getvalue()

    def save_csv(self, path: Path | str) -> None:
        """Save to CSV file."""
        path = Path(path)
        path.write_text(self.to_csv(), encoding="utf-8")

    def filter_by_op_type(self, op_types: list[str]) -> LayerSummary:
        """Return a new summary filtered by op type."""
        filtered = [layer for layer in self.layers if layer.op_type in op_types]
        return LayerSummary(
            layers=filtered,
            total_params=self.total_params,
            total_flops=self.total_flops,
            total_memory=self.total_memory,
        )

    def filter_by_threshold(
        self,
        min_params: int = 0,
        min_flops: int = 0,
        min_pct_params: float = 0.0,
        min_pct_flops: float = 0.0,
    ) -> LayerSummary:
        """Return a new summary filtered by thresholds."""
        filtered = [
            layer
            for layer in self.layers
            if layer.params >= min_params
            and layer.flops >= min_flops
            and layer.pct_params >= min_pct_params
            and layer.pct_flops >= min_pct_flops
        ]
        return LayerSummary(
            layers=filtered,
            total_params=self.total_params,
            total_flops=self.total_flops,
            total_memory=self.total_memory,
        )

    def sort_by(self, key: str, descending: bool = True) -> LayerSummary:
        """
        Return a new summary sorted by the specified key.

        Args:
            key: One of 'name', 'op_type', 'params', 'flops', 'memory_bytes',
                 'latency_ms', 'pct_params', 'pct_flops', 'depth'
            descending: Sort in descending order if True
        """
        valid_keys = {
            "name",
            "op_type",
            "params",
            "flops",
            "memory_bytes",
            "latency_ms",
            "pct_params",
            "pct_flops",
            "depth",
            "input_shapes",
            "output_shapes",
        }
        key_normalized = key.replace("-", "_")
        if key_normalized not in valid_keys:
            raise ValueError(f"Invalid sort key: {key}")

        sorted_layers = sorted(
            self.layers,
            key=lambda layer: getattr(layer, key_normalized),
            reverse=descending,
        )
        return LayerSummary(
            layers=sorted_layers,
            total_params=self.total_params,
            total_flops=self.total_flops,
            total_memory=self.total_memory,
        )

    def top_n(self, n: int, key: str = "flops") -> LayerSummary:
        """Get top N layers by the specified metric."""
        sorted_summary = self.sort_by(key, descending=True)
        return LayerSummary(
            layers=sorted_summary.layers[:n],
            total_params=self.total_params,
            total_flops=self.total_flops,
            total_memory=self.total_memory,
        )


class LayerSummaryBuilder:
    """
    Build per-layer summary from ONNX graph analysis.

    Task 5.8.1: Create per-layer summary table.
    """

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger("haoline.layer_summary")

    def build(
        self,
        graph_info: GraphInfo,
        param_counts: ParamCounts | None = None,
        flop_counts: FlopCounts | None = None,
        memory_estimates: MemoryEstimates | None = None,
    ) -> LayerSummary:
        """
        Build layer summary from graph analysis.

        Args:
            graph_info: Parsed ONNX graph.
            param_counts: Parameter counts from analysis.
            flop_counts: FLOPs from analysis.
            memory_estimates: Memory estimates from analysis.

        Returns:
            LayerSummary with per-layer metrics.
        """
        layers = []

        # Calculate totals for percentages
        total_params = param_counts.total if param_counts else 0
        total_flops = flop_counts.total if flop_counts else 0
        total_memory = memory_estimates.model_size_bytes if memory_estimates else 0

        # Build tensor shape map for input/output shapes
        tensor_shapes: dict[str, str] = {}
        for name, shape in graph_info.input_shapes.items():
            tensor_shapes[name] = str(shape)
        for name, shape in graph_info.output_shapes.items():
            tensor_shapes[name] = str(shape)

        # Process each node
        for idx, node in enumerate(graph_info.nodes):
            # Get metrics for this node
            node_params = 0
            node_flops = 0

            if param_counts and param_counts.by_node:
                node_params = int(param_counts.by_node.get(node.name, 0))

            if flop_counts and flop_counts.by_node:
                node_flops = flop_counts.by_node.get(node.name, 0)

            # Get shapes
            input_shapes = []
            output_shapes = []

            for inp in node.inputs:
                if inp in tensor_shapes:
                    input_shapes.append(tensor_shapes[inp])
                elif inp in graph_info.initializers:
                    init = graph_info.initializers[inp]
                    if hasattr(init, "dims"):
                        input_shapes.append(str(list(init.dims)))

            for out in node.outputs:
                if out in tensor_shapes:
                    output_shapes.append(tensor_shapes[out])

            # Calculate percentages
            pct_params = (node_params / total_params * 100) if total_params > 0 else 0.0
            pct_flops = (node_flops / total_flops * 100) if total_flops > 0 else 0.0

            # Estimate memory for this layer
            node_memory = 0
            if memory_estimates and memory_estimates.breakdown:
                # Use weight memory by op type as approximation
                op_weights = memory_estimates.breakdown.weights_by_op_type
                if op_weights and node.op_type in op_weights:
                    # Distribute among nodes of same type
                    nodes_of_type = sum(1 for n in graph_info.nodes if n.op_type == node.op_type)
                    if nodes_of_type > 0:
                        node_memory = op_weights[node.op_type] // nodes_of_type

            # Estimate latency (rough: assume linear with FLOPs)
            # This is a placeholder - real latency requires profiling
            latency_ms = node_flops / 1e9 * 0.01 if node_flops > 0 else 0.0

            layer = LayerMetrics(
                name=node.name,
                op_type=node.op_type,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                params=node_params,
                flops=node_flops,
                memory_bytes=node_memory,
                latency_ms=latency_ms,
                pct_params=pct_params,
                pct_flops=pct_flops,
                depth=idx,
            )
            layers.append(layer)

        return LayerSummary(
            layers=layers,
            total_params=total_params,
            total_flops=total_flops,
            total_memory=total_memory,
        )

    def build_from_report(self, report: InspectionReport) -> LayerSummary:
        """Build layer summary from an inspection report."""
        # We need the graph_info which isn't stored in the report
        # This method requires re-loading the model
        raise NotImplementedError(
            "build_from_report requires model path. Use build() with graph_info instead."
        )


def generate_html_table(summary: LayerSummary, include_js: bool = True) -> str:
    """
    Generate HTML table with sortable columns.

    Task 5.8.2: Add sortable/filterable table to HTML report.

    Args:
        summary: Layer summary to render.
        include_js: Include JavaScript for sorting/filtering.

    Returns:
        HTML string for the table.
    """

    def format_number(n: int | float) -> str:
        if isinstance(n, float):
            if n >= 1e9:
                return f"{n / 1e9:.2f}B"
            if n >= 1e6:
                return f"{n / 1e6:.2f}M"
            if n >= 1e3:
                return f"{n / 1e3:.2f}K"
            return f"{n:.2f}"
        else:
            if n >= 1e9:
                return f"{n / 1e9:.2f}B"
            if n >= 1e6:
                return f"{n / 1e6:.2f}M"
            if n >= 1e3:
                return f"{n / 1e3:.2f}K"
            return str(n)

    def format_bytes(b: int) -> str:
        if b >= 1e9:
            return f"{b / 1e9:.2f} GB"
        if b >= 1e6:
            return f"{b / 1e6:.2f} MB"
        if b >= 1e3:
            return f"{b / 1e3:.2f} KB"
        return f"{b} B"

    html_parts = []

    # CSS styles
    html_parts.append(
        """
    <style>
        .layer-table-container {
            margin: 20px 0;
            overflow-x: auto;
        }

        .layer-controls {
            display: flex;
            gap: 16px;
            margin-bottom: 12px;
            flex-wrap: wrap;
            align-items: center;
        }

        .layer-search {
            padding: 8px 12px;
            border: 1px solid var(--border, #30363d);
            border-radius: 6px;
            background: var(--bg-card, #21262d);
            color: var(--text-primary, #e6edf3);
            font-size: 0.875rem;
            min-width: 200px;
        }

        .layer-filter {
            padding: 8px 12px;
            border: 1px solid var(--border, #30363d);
            border-radius: 6px;
            background: var(--bg-card, #21262d);
            color: var(--text-primary, #e6edf3);
            font-size: 0.875rem;
        }

        .layer-export-btn {
            padding: 8px 16px;
            border: 1px solid var(--accent-cyan, #00d4ff);
            border-radius: 6px;
            background: transparent;
            color: var(--accent-cyan, #00d4ff);
            cursor: pointer;
            font-size: 0.875rem;
            transition: all 0.2s;
        }

        .layer-export-btn:hover {
            background: var(--accent-cyan, #00d4ff);
            color: white;
        }

        .layer-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.8125rem;
        }

        .layer-table th {
            background: var(--bg-secondary, #161b22);
            color: var(--accent-cyan, #00d4ff);
            padding: 10px 12px;
            text-align: left;
            cursor: pointer;
            user-select: none;
            white-space: nowrap;
            border-bottom: 2px solid var(--border, #30363d);
            position: sticky;
            top: 0;
        }

        .layer-table th:hover {
            background: var(--bg-card, #21262d);
        }

        .layer-table th .sort-icon {
            margin-left: 4px;
            opacity: 0.5;
        }

        .layer-table th.sorted .sort-icon {
            opacity: 1;
        }

        .layer-table td {
            padding: 8px 12px;
            border-bottom: 1px solid var(--border, #30363d);
            vertical-align: top;
        }

        .layer-table tr:hover {
            background: var(--bg-secondary, #161b22);
        }

        .layer-table tr.highlight {
            background: rgba(0, 212, 255, 0.1);
            outline: 1px solid var(--accent-cyan, #00d4ff);
        }

        .layer-table .op-type {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 500;
        }

        .layer-table .op-type.conv { background: rgba(74, 144, 217, 0.2); color: #4A90D9; }
        .layer-table .op-type.linear { background: rgba(191, 90, 242, 0.2); color: #BF5AF2; }
        .layer-table .op-type.norm { background: rgba(100, 210, 255, 0.2); color: #64D2FF; }
        .layer-table .op-type.activation { background: rgba(255, 214, 10, 0.2); color: #FFD60A; }
        .layer-table .op-type.pool { background: rgba(48, 209, 88, 0.2); color: #30D158; }
        .layer-table .op-type.reshape { background: rgba(94, 92, 230, 0.2); color: #5E5CE6; }
        .layer-table .op-type.elementwise { background: rgba(255, 100, 130, 0.2); color: #FF6482; }
        .layer-table .op-type.default { background: rgba(99, 99, 102, 0.2); color: #636366; }

        .layer-table .shape {
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 0.7rem;
            color: var(--text-secondary, #8b949e);
        }

        .layer-table .pct-bar {
            display: inline-block;
            height: 4px;
            background: var(--accent-cyan, #00d4ff);
            border-radius: 2px;
            margin-right: 6px;
            vertical-align: middle;
        }

        .layer-count {
            font-size: 0.75rem;
            color: var(--text-secondary, #8b949e);
            margin-top: 8px;
        }
    </style>
    """
    )

    # Controls
    html_parts.append(
        """
    <div class="layer-table-container">
        <div class="layer-controls">
            <input type="text" class="layer-search" id="layerSearch"
                   placeholder="Search layers..." oninput="filterLayers()">
            <select class="layer-filter" id="opFilter" onchange="filterLayers()">
                <option value="">All Op Types</option>
    """
    )

    # Get unique op types
    op_types = sorted({layer.op_type for layer in summary.layers})
    for op_type in op_types:
        html_parts.append(f'            <option value="{op_type}">{op_type}</option>\n')

    html_parts.append(
        """
            </select>
            <button class="layer-export-btn" onclick="exportLayersCSV()">Export CSV</button>
        </div>
    """
    )

    # Table
    html_parts.append(
        """
        <table class="layer-table" id="layerTable">
            <thead>
                <tr>
                    <th onclick="sortTable(0)" data-col="name">Layer Name <span class="sort-icon">^v</span></th>
                    <th onclick="sortTable(1)" data-col="op_type">Op Type <span class="sort-icon">^v</span></th>
                    <th onclick="sortTable(2)" data-col="input_shape">Input Shape</th>
                    <th onclick="sortTable(3)" data-col="output_shape">Output Shape</th>
                    <th onclick="sortTable(4)" data-col="params">Parameters <span class="sort-icon">^v</span></th>
                    <th onclick="sortTable(5)" data-col="flops">FLOPs <span class="sort-icon">^v</span></th>
                    <th onclick="sortTable(6)" data-col="memory">Memory <span class="sort-icon">^v</span></th>
                    <th onclick="sortTable(7)" data-col="pct_flops">% Compute <span class="sort-icon">^v</span></th>
                </tr>
            </thead>
            <tbody>
    """
    )

    # Get op type category for styling
    def get_op_category(op_type: str) -> str:
        op = op_type.lower()
        if "conv" in op:
            return "conv"
        if "matmul" in op or "gemm" in op:
            return "linear"
        if "norm" in op:
            return "norm"
        if any(x in op for x in ["relu", "gelu", "softmax", "sigmoid", "silu", "tanh"]):
            return "activation"
        if "pool" in op:
            return "pool"
        if any(x in op for x in ["reshape", "transpose", "flatten", "squeeze"]):
            return "reshape"
        if any(x in op for x in ["add", "mul", "sub", "div", "concat"]):
            return "elementwise"
        return "default"

    # Data rows
    for layer in summary.layers:
        op_cat = get_op_category(layer.op_type)
        input_str = "; ".join(layer.input_shapes) if layer.input_shapes else "-"
        output_str = "; ".join(layer.output_shapes) if layer.output_shapes else "-"

        # Calculate bar width for percentage
        bar_width = min(100, layer.pct_flops * 3)  # Scale for visibility

        html_parts.append(
            f"""
                <tr data-name="{layer.name}" data-op="{layer.op_type}"
                    data-params="{layer.params}" data-flops="{layer.flops}">
                    <td><code>{layer.name}</code></td>
                    <td><span class="op-type {op_cat}">{layer.op_type}</span></td>
                    <td class="shape">{input_str}</td>
                    <td class="shape">{output_str}</td>
                    <td>{format_number(layer.params)}</td>
                    <td>{format_number(layer.flops)}</td>
                    <td>{format_bytes(layer.memory_bytes)}</td>
                    <td>
                        <span class="pct-bar" style="width: {bar_width}px"></span>
                        {layer.pct_flops:.1f}%
                    </td>
                </tr>
        """
        )

    html_parts.append(
        """
            </tbody>
        </table>
        <div class="layer-count" id="layerCount">
    """
    )
    html_parts.append(
        f'        Showing <span id="visibleCount">{len(summary.layers)}</span> of {len(summary.layers)} layers'
    )
    html_parts.append(
        """
        </div>
    </div>
    """
    )

    # JavaScript for sorting and filtering
    if include_js:
        # Build CSV data for export
        csv_data = summary.to_csv().replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")

        html_parts.append(
            f"""
    <script>
        const layerCSVData = `{csv_data}`;

        let sortColumn = -1;
        let sortAsc = true;

        function sortTable(col) {{
            const table = document.getElementById('layerTable');
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const headers = table.querySelectorAll('th');

            // Toggle sort direction if same column
            if (sortColumn === col) {{
                sortAsc = !sortAsc;
            }} else {{
                sortAsc = false;  // Default descending for numeric columns
                sortColumn = col;
            }}

            // Update header styling
            headers.forEach((h, i) => {{
                h.classList.remove('sorted');
                if (i === col) h.classList.add('sorted');
            }});

            // Sort rows
            rows.sort((a, b) => {{
                let aVal = a.cells[col].textContent.trim();
                let bVal = b.cells[col].textContent.trim();

                // Try numeric comparison for columns 4-7
                if (col >= 4) {{
                    aVal = parseFloat(aVal.replace(/[^0-9.-]/g, '')) || 0;
                    bVal = parseFloat(bVal.replace(/[^0-9.-]/g, '')) || 0;
                    return sortAsc ? aVal - bVal : bVal - aVal;
                }}

                // String comparison
                return sortAsc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
            }});

            // Re-add rows
            rows.forEach(row => tbody.appendChild(row));
        }}

        function filterLayers() {{
            const searchTerm = document.getElementById('layerSearch').value.toLowerCase();
            const opFilter = document.getElementById('opFilter').value;
            const table = document.getElementById('layerTable');
            const rows = table.querySelectorAll('tbody tr');

            let visibleCount = 0;
            rows.forEach(row => {{
                const name = row.dataset.name.toLowerCase();
                const op = row.dataset.op;

                const matchesSearch = !searchTerm || name.includes(searchTerm);
                const matchesOp = !opFilter || op === opFilter;

                if (matchesSearch && matchesOp) {{
                    row.style.display = '';
                    visibleCount++;
                }} else {{
                    row.style.display = 'none';
                }}
            }});

            document.getElementById('visibleCount').textContent = visibleCount;
        }}

        function exportLayersCSV() {{
            const blob = new Blob([layerCSVData], {{ type: 'text/csv;charset=utf-8;' }});
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = 'layer_summary.csv';
            link.click();
        }}

        // Highlight row when clicked (for graph integration)
        document.querySelectorAll('#layerTable tbody tr').forEach(row => {{
            row.addEventListener('click', () => {{
                document.querySelectorAll('#layerTable tbody tr').forEach(r => r.classList.remove('highlight'));
                row.classList.add('highlight');

                // Dispatch event for graph integration (Task 5.8.3)
                const event = new CustomEvent('layer-selected', {{
                    detail: {{ name: row.dataset.name, op: row.dataset.op }}
                }});
                document.dispatchEvent(event);
            }});
        }});
    </script>
        """
        )

    return "".join(html_parts)


def generate_markdown_table(summary: LayerSummary, max_rows: int = 50) -> str:
    """Generate Markdown table for layer summary."""

    def format_number(n: int | float) -> str:
        if isinstance(n, float):
            if n >= 1e9:
                return f"{n / 1e9:.1f}B"
            if n >= 1e6:
                return f"{n / 1e6:.1f}M"
            if n >= 1e3:
                return f"{n / 1e3:.1f}K"
            return f"{n:.2f}"
        else:
            if n >= 1e9:
                return f"{n / 1e9:.1f}B"
            if n >= 1e6:
                return f"{n / 1e6:.1f}M"
            if n >= 1e3:
                return f"{n / 1e3:.1f}K"
            return str(n)

    lines = []
    lines.append("| Layer | Op Type | Params | FLOPs | % Compute |")
    lines.append("|-------|---------|--------|-------|-----------|")

    # Sort by FLOPs descending to show most important first
    sorted_layers = sorted(summary.layers, key=lambda layer: layer.flops, reverse=True)

    for layer in sorted_layers[:max_rows]:
        lines.append(
            f"| `{layer.name}` | {layer.op_type} | "
            f"{format_number(layer.params)} | {format_number(layer.flops)} | "
            f"{layer.pct_flops:.1f}% |"
        )

    if len(summary.layers) > max_rows:
        lines.append(f"| ... | ({len(summary.layers) - max_rows} more layers) | ... | ... | ... |")

    return "\n".join(lines)
