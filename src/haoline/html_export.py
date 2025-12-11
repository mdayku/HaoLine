# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
Interactive HTML Export for graph visualization.

Task 5.8: Creates standalone HTML files with embedded visualization
that can be opened in any browser without a server.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .edge_analysis import EdgeAnalysisResult
    from .hierarchical_graph import HierarchicalGraph


# HTML template with embedded D3.js visualization - Jony Ive Edition
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Neural Architecture</title>
    <link href="https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600&display=swap" rel="stylesheet">
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        :root {{
            --bg-deep: #000000;
            --bg-primary: #0a0a0a;
            --bg-elevated: #1a1a1a;
            --bg-glass: rgba(255, 255, 255, 0.03);
            --text-primary: rgba(255, 255, 255, 0.92);
            --text-secondary: rgba(255, 255, 255, 0.55);
            --text-tertiary: rgba(255, 255, 255, 0.35);
            --accent: #0A84FF;
            --accent-glow: rgba(10, 132, 255, 0.3);
            --border: rgba(255, 255, 255, 0.08);
            --success: #30D158;
            --warning: #FFD60A;
            --error: #FF453A;
            --purple: #BF5AF2;
            --orange: #FF9F0A;
            --teal: #64D2FF;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', sans-serif;
            background: var(--bg-deep);
            color: var(--text-primary);
            overflow: hidden;
            -webkit-font-smoothing: antialiased;
        }}

        .container {{
            display: flex;
            height: 100vh;
        }}

        .sidebar {{
            width: 280px;
            background: var(--bg-primary);
            border-right: 1px solid var(--border);
            padding: 32px 24px;
            overflow-y: auto;
            backdrop-filter: blur(20px);
            transition: all 0.3s cubic-bezier(0.25, 0.1, 0.25, 1);
            position: relative;
        }}

        .sidebar.collapsed {{
            width: 0;
            padding: 0;
            overflow: hidden;
            border-right: none;
        }}

        .sidebar-toggle {{
            position: fixed;
            top: 12px;
            left: 248px;
            width: 28px;
            height: 28px;
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text-secondary);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            z-index: 1000;
            transition: all 0.3s cubic-bezier(0.25, 0.1, 0.25, 1);
        }}

        .sidebar-toggle:hover {{
            background: var(--accent);
            color: white;
            border-color: var(--accent);
        }}

        .sidebar.collapsed + .main .sidebar-toggle,
        .sidebar.collapsed ~ .sidebar-toggle {{
            left: 8px;
        }}

        .sidebar h1 {{
            font-size: 1.25rem;
            font-weight: 600;
            letter-spacing: -0.02em;
            margin-bottom: 32px;
            background: linear-gradient(135deg, #fff 0%, rgba(255,255,255,0.7) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .sidebar h2 {{
            font-size: 0.6875rem;
            font-weight: 500;
            color: var(--text-tertiary);
            margin: 24px 0 12px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}

        .stats {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }}

        .stat-card {{
            background: var(--bg-glass);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            transition: all 0.3s cubic-bezier(0.25, 0.1, 0.25, 1);
        }}

        .stat-card:hover {{
            background: rgba(255, 255, 255, 0.06);
            transform: translateY(-1px);
        }}

        .stat-value {{
            font-size: 1.5rem;
            font-weight: 600;
            letter-spacing: -0.03em;
            background: linear-gradient(135deg, var(--accent) 0%, var(--teal) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .stat-label {{
            font-size: 0.6875rem;
            color: var(--text-tertiary);
            margin-top: 4px;
        }}

        .controls {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }}

        .btn {{
            padding: 8px 14px;
            background: var(--bg-glass);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 0.75rem;
            font-weight: 500;
            transition: all 0.2s cubic-bezier(0.25, 0.1, 0.25, 1);
        }}

        .btn:hover {{
            background: var(--accent);
            border-color: var(--accent);
            color: white;
            box-shadow: 0 4px 16px var(--accent-glow);
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            padding: 6px 8px;
            margin: 2px 0;
            font-size: 0.75rem;
            color: var(--text-secondary);
            cursor: pointer;
            border-radius: 6px;
            transition: all 0.15s ease;
        }}

        .legend-item:hover {{
            background: rgba(255,255,255,0.05);
        }}

        .legend-item.active {{
            background: rgba(10, 132, 255, 0.2);
            color: var(--text-primary);
        }}

        .legend-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 12px;
            flex-shrink: 0;
        }}

        .legend-symbol {{
            font-size: 1rem;
            width: 20px;
            text-align: center;
            margin-right: 10px;
        }}

        .main {{
            flex: 1;
            position: relative;
            background: radial-gradient(ellipse at center, #0a0a0a 0%, #000 100%);
        }}

        /* Subtle grid pattern */
        .main::before {{
            content: '';
            position: absolute;
            inset: 0;
            background-image:
                linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px);
            background-size: 40px 40px;
            pointer-events: none;
        }}

        svg {{
            width: 100%;
            height: 100%;
        }}

        .node {{
            cursor: pointer;
        }}

        .node:hover .node-circle {{
            filter: brightness(1.2) drop-shadow(0 0 12px currentColor);
        }}

        .node-circle {{
            stroke: rgba(255,255,255,0.1);
            stroke-width: 1;
            filter: drop-shadow(0 2px 8px rgba(0,0,0,0.4));
        }}

        .node-label {{
            font-size: 9px;
            font-weight: 500;
            fill: white;
            text-anchor: middle;
            dominant-baseline: middle;
            pointer-events: none;
            text-shadow: 0 1px 2px rgba(0,0,0,0.8);
            opacity: 0;
            transition: opacity 0.2s ease;
        }}

        /* Show labels on hover */
        .node:hover .node-label {{
            opacity: 1;
        }}

        /* Always show labels for large/important nodes */
        .node.show-label .node-label {{
            opacity: 1;
        }}

        .node-sublabel {{
            font-size: 7px;
            fill: rgba(255,255,255,0.5);
            text-anchor: middle;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s ease;
        }}

        .node:hover .node-sublabel {{
            opacity: 1;
        }}

        .edge {{
            fill: none;
            stroke-linecap: round;
            opacity: 0.6;
        }}

        .edge:hover {{
            opacity: 1;
            stroke-width: 3;
        }}

        .tooltip {{
            position: fixed;
            background: rgba(20, 20, 20, 0.95);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            font-size: 0.8125rem;
            pointer-events: none;
            opacity: 0;
            transform: translateY(4px);
            transition: all 0.2s cubic-bezier(0.25, 0.1, 0.25, 1);
            max-width: 280px;
            z-index: 1000;
            box-shadow: 0 8px 32px rgba(0,0,0,0.6);
        }}

        .tooltip.visible {{
            opacity: 1;
            transform: translateY(0);
        }}

        .tooltip-title {{
            font-weight: 600;
            font-size: 0.9375rem;
            margin-bottom: 6px;
            color: var(--accent);
        }}

        .tooltip-desc {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            line-height: 1.4;
            margin-bottom: 12px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--border);
        }}

        .tooltip-row {{
            display: flex;
            justify-content: space-between;
            padding: 3px 0;
            font-size: 0.75rem;
        }}

        .tooltip-label {{
            color: var(--text-tertiary);
        }}

        .tooltip-value {{
            color: var(--text-primary);
            font-weight: 500;
            font-family: 'SF Mono', 'Menlo', monospace;
        }}

        .block-indicator {{
            position: absolute;
            top: -6px;
            right: -6px;
            width: 14px;
            height: 14px;
            background: var(--accent);
            border-radius: 50%;
            font-size: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
            box-shadow: 0 2px 8px var(--accent-glow);
        }}

        /* Zoom controls */
        .zoom-controls {{
            position: absolute;
            bottom: 24px;
            right: 24px;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}

        .zoom-btn {{
            width: 36px;
            height: 36px;
            background: rgba(20, 20, 20, 0.9);
            backdrop-filter: blur(10px);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 1.25rem;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
        }}

        .zoom-btn:hover {{
            background: var(--bg-elevated);
            color: var(--text-primary);
        }}

        /* Search functionality - Task 5.7.6 */
        .search-input {{
            width: 100%;
            padding: 10px 14px;
            background: var(--bg-glass);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text-primary);
            font-size: 0.875rem;
            margin-bottom: 8px;
            transition: all 0.2s;
        }}

        .search-input:focus {{
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }}

        .search-input::placeholder {{
            color: var(--text-tertiary);
        }}

        .search-results {{
            max-height: 200px;
            overflow-y: auto;
            margin-bottom: 16px;
        }}

        .search-result {{
            padding: 8px 12px;
            font-size: 0.75rem;
            cursor: pointer;
            border-radius: 6px;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: background 0.15s;
        }}

        .search-result:hover {{
            background: rgba(255,255,255,0.05);
        }}

        .search-result .result-name {{
            flex: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}

        .search-result .result-type {{
            font-size: 0.65rem;
            color: var(--text-tertiary);
            padding: 2px 6px;
            background: var(--bg-glass);
            border-radius: 4px;
        }}

        .search-highlight {{
            filter: drop-shadow(0 0 20px var(--accent)) brightness(1.5);
            z-index: 100;
        }}

        /* Performance mode indicator */
        .perf-mode {{
            font-size: 0.65rem;
            color: var(--warning);
            padding: 4px 8px;
            background: rgba(255, 214, 10, 0.1);
            border-radius: 4px;
            margin-bottom: 12px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <button class="sidebar-toggle" id="sidebarToggle" onclick="toggleSidebar()" title="Toggle sidebar (more graph space)">◀</button>
        <aside class="sidebar" id="graphSidebar">
            <h1>{title}</h1>

            <h2>Overview</h2>
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-value" id="node-count">0</div>
                    <div class="stat-label">Nodes</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="edge-count">0</div>
                    <div class="stat-label">Edges</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="peak-memory">0</div>
                    <div class="stat-label">Peak Activation</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="model-size">-</div>
                    <div class="stat-label">Model Size</div>
                </div>
            </div>

            <h2>Search</h2>
            <input type="text" id="nodeSearch" class="search-input"
                   placeholder="Search nodes..." oninput="searchNodes(this.value)">
            <div id="searchResults" class="search-results"></div>

            <h2>Navigation</h2>
            <div class="controls">
                <button class="btn" onclick="expandAll()">Expand</button>
                <button class="btn" onclick="collapseAll()">Collapse</button>
                <button class="btn" onclick="fitToScreen()">Fit</button>
                <button class="btn" onclick="resetZoom()">Reset</button>
            </div>

            <h2>Visualization</h2>
            <div class="controls">
                <button class="btn" id="labels-btn" onclick="toggleAllLabels()">Show All Labels</button>
                <button class="btn" id="heatmap-btn" onclick="toggleHeatMap()" style="margin-top:6px;">FLOPs Heat Map</button>
            </div>

            <h2>On Load</h2>
            <div class="controls" style="display:flex;flex-direction:column;gap:6px;">
                <label style="display:flex;align-items:center;gap:8px;cursor:pointer;font-size:0.85rem;">
                    <input type="checkbox" id="autoFitCheck" checked onchange="savePrefs()">
                    Auto-fit to screen
                </label>
                <label style="display:flex;align-items:center;gap:8px;cursor:pointer;font-size:0.85rem;">
                    <input type="checkbox" id="autoExpandCheck" onchange="savePrefs()">
                    Expand all nodes
                </label>
            </div>

            <h2>Op Types <span style="font-size:0.6rem;color:var(--text-tertiary)">(click to filter)</span></h2>
            <div class="legend" id="op-legend">
                <div class="legend-item" data-category="conv" onclick="filterByCategory('conv')">
                    <div class="legend-symbol" style="color: #4A90D9;">▣</div>
                    <span>Convolution</span>
                </div>
                <div class="legend-item" data-category="linear" onclick="filterByCategory('linear')">
                    <div class="legend-symbol" style="color: #BF5AF2;">◆</div>
                    <span>Linear/MatMul</span>
                </div>
                <div class="legend-item" data-category="attention" onclick="filterByCategory('attention')">
                    <div class="legend-symbol" style="color: #FF9F0A;">◎</div>
                    <span>Attention</span>
                </div>
                <div class="legend-item" data-category="norm" onclick="filterByCategory('norm')">
                    <div class="legend-symbol" style="color: #64D2FF;">≡</div>
                    <span>Normalization</span>
                </div>
                <div class="legend-item" data-category="activation" onclick="filterByCategory('activation')">
                    <div class="legend-symbol" style="color: #FFD60A;">⚡</div>
                    <span>Activation</span>
                </div>
                <div class="legend-item" data-category="pool" onclick="filterByCategory('pool')">
                    <div class="legend-symbol" style="color: #30D158;">▼</div>
                    <span>Pooling</span>
                </div>
                <div class="legend-item" data-category="reshape" onclick="filterByCategory('reshape')">
                    <div class="legend-symbol" style="color: #5E5CE6;">⤨</div>
                    <span>Reshape</span>
                </div>
                <div class="legend-item" data-category="elementwise" onclick="filterByCategory('elementwise')">
                    <div class="legend-symbol" style="color: #FF6482;">+</div>
                    <span>Math ops</span>
                </div>
                <div class="legend-item" data-category="default" onclick="filterByCategory('default')">
                    <div class="legend-symbol" style="color: #636366;">●</div>
                    <span>Other</span>
                </div>
                <div class="legend-item" data-category="all" onclick="filterByCategory('all')" style="margin-top:8px;border-top:1px solid var(--border);padding-top:8px;">
                    <span style="color:var(--accent)">↺ Show all</span>
                </div>
            </div>

            <h2>Heat Map Scale</h2>
            <div class="legend" id="heatmap-legend" style="display: none;">
                <div class="legend-item">
                    <div class="legend-dot" style="background: #0A84FF; box-shadow: 0 0 8px #0A84FF;"></div>
                    <span>Low FLOPs</span>
                </div>
                <div class="legend-item">
                    <div class="legend-dot" style="background: #64D2FF; box-shadow: 0 0 8px #64D2FF;"></div>
                    <span>Light</span>
                </div>
                <div class="legend-item">
                    <div class="legend-dot" style="background: #30D158; box-shadow: 0 0 8px #30D158;"></div>
                    <span>Medium</span>
                </div>
                <div class="legend-item">
                    <div class="legend-dot" style="background: #FFD60A; box-shadow: 0 0 8px #FFD60A;"></div>
                    <span>Heavy</span>
                </div>
                <div class="legend-item">
                    <div class="legend-dot" style="background: #FF9F0A; box-shadow: 0 0 8px #FF9F0A;"></div>
                    <span>Very heavy</span>
                </div>
                <div class="legend-item">
                    <div class="legend-dot" style="background: #FF453A; box-shadow: 0 0 8px #FF453A;"></div>
                    <span>Compute hotspot</span>
                </div>
            </div>
            <div class="legend" id="optype-legend-note">
                <span style="font-size: 0.7rem; color: var(--text-tertiary);">Toggle heat maps to visualize compute or timing</span>
            </div>
        </aside>

        <main class="main">
            <svg id="graph"></svg>
            <div class="tooltip" id="tooltip"></div>
            <div class="zoom-controls">
                <button class="zoom-btn" onclick="zoomIn()">+</button>
                <button class="zoom-btn" onclick="zoomOut()">-</button>
            </div>
        </main>
    </div>

    <script>
        // Embedded graph data
        const graphData = {graph_json};
        const edgeData = {edge_json};

        // Get label for a node
        function getNodeLabel(node) {{
            if (node.node_type === 'op' && node.op_type) {{
                return node.op_type;
            }}
            if (node.display_name) {{
                return node.display_name;
            }}
            return node.name;
        }}

        // Apple-inspired color palette with symbols
        const opStyles = {{
            conv:        {{ color: '#4A90D9', symbol: '▣', name: 'Convolution' }},
            linear:      {{ color: '#BF5AF2', symbol: '◆', name: 'Linear' }},
            attention:   {{ color: '#FF9F0A', symbol: '◎', name: 'Attention' }},
            norm:        {{ color: '#64D2FF', symbol: '≡', name: 'Normalize' }},
            activation:  {{ color: '#FFD60A', symbol: '⚡', name: 'Activation' }},
            pool:        {{ color: '#30D158', symbol: '▼', name: 'Pooling' }},
            embed:       {{ color: '#AF52DE', symbol: '⊞', name: 'Embed' }},
            reshape:     {{ color: '#5E5CE6', symbol: '⤨', name: 'Reshape' }},
            elementwise: {{ color: '#FF6482', symbol: '+', name: 'Math' }},
            reduce:      {{ color: '#FF453A', symbol: 'Σ', name: 'Reduce' }},
            default:     {{ color: '#636366', symbol: '●', name: 'Other' }}
        }};

        // For backwards compat
        const colors = Object.fromEntries(
            Object.entries(opStyles).map(([k, v]) => [k, v.color])
        );

        // Get op category key
        function getOpCategory(node) {{
            if (node.node_type === 'block') {{
                const blockType = (node.attributes?.block_type || '').toLowerCase();
                if (blockType.includes('attention')) return 'attention';
                if (blockType.includes('mlp') || blockType.includes('ffn')) return 'linear';
                if (blockType.includes('conv')) return 'conv';
                if (blockType.includes('norm')) return 'norm';
                if (blockType.includes('embed')) return 'embed';
                return 'default';
            }}

            const op = (node.op_type || '').toLowerCase();
            if (op.includes('conv')) return 'conv';
            if (op.includes('matmul') || op.includes('gemm')) return 'linear';
            if (op.includes('norm') || op.includes('layer')) return 'norm';
            if (op.includes('relu') || op.includes('gelu') || op.includes('softmax') || op.includes('sigmoid') || op.includes('silu') || op.includes('tanh')) return 'activation';
            if (op.includes('pool')) return 'pool';
            if (op.includes('reshape') || op.includes('transpose') || op.includes('flatten') || op.includes('squeeze') || op.includes('unsqueeze')) return 'reshape';
            if (op.includes('add') || op.includes('mul') || op.includes('sub') || op.includes('div') || op.includes('concat') || op.includes('split')) return 'elementwise';
            if (op.includes('reduce')) return 'reduce';
            if (op.includes('gather') || op.includes('embed')) return 'embed';
            return 'default';
        }}

        // Get color for op type
        function getNodeColor(node) {{
            return opStyles[getOpCategory(node)].color;
        }}

        // Get symbol for op type
        function getNodeSymbol(node) {{
            return opStyles[getOpCategory(node)].symbol;
        }}

        // Get node size based on type and compute
        // Track max FLOPs for scaling (computed once during render)
        let globalMaxFlops = 1;

        function getNodeSize(node) {{
            // Scale by FLOPs - expensive ops are visually bigger
            const flops = node.total_flops || 0;

            if (flops > 0 && globalMaxFlops > 1) {{
                // Log scale: 12px min, 45px max based on FLOPs
                const logFlops = Math.log10(flops + 1);
                const logMax = Math.log10(globalMaxFlops + 1);
                const ratio = logFlops / logMax;
                return 12 + ratio * 33;
            }}

            // Fallback for nodes without FLOPs data - use hierarchy
            const base = node.node_type === 'model' ? 40 :
                         node.node_type === 'layer' ? 30 :
                         node.node_type === 'block' ? 25 : 15;
            return base;
        }}

        // Heat map mode toggle
        let heatMapMode = false;

        // Get heat map color based on compute intensity
        function getHeatColor(flops, maxFlops) {{
            if (!flops || flops === 0) return null;
            const intensity = Math.log10(flops + 1) / Math.log10(maxFlops + 1);
            // Blue -> Cyan -> Green -> Yellow -> Orange -> Red
            if (intensity < 0.2) return '#0A84FF';
            if (intensity < 0.4) return '#64D2FF';
            if (intensity < 0.6) return '#30D158';
            if (intensity < 0.8) return '#FFD60A';
            if (intensity < 0.9) return '#FF9F0A';
            return '#FF453A';
        }}

        // Initialize visualization
        const svg = d3.select('#graph');
        const container = svg.append('g');

        // Zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on('zoom', (event) => {{
                container.attr('transform', event.transform);
            }});

        svg.call(zoom);

        // Tooltip
        const tooltip = d3.select('#tooltip');

        // Op type explanations for demystifying the model
        const opDescriptions = {{
            'Conv': 'Convolution: Slides filters across input to detect features like edges, textures, shapes',
            'MatMul': 'Matrix Multiply: Core linear transformation - learns weighted combinations of inputs',
            'Gemm': 'General Matrix Multiply: Linear layer with weights and bias',
            'Relu': 'ReLU Activation: Keeps positive values, zeros negatives - adds non-linearity',
            'Gelu': 'GELU Activation: Smooth activation used in transformers - better gradients',
            'Softmax': 'Softmax: Converts scores to probabilities (sums to 1) - used in attention',
            'Sigmoid': 'Sigmoid: Squashes values to 0-1 range - used for gates/probabilities',
            'LayerNormalization': 'Layer Norm: Normalizes activations for stable training',
            'BatchNormalization': 'Batch Norm: Normalizes across batch - speeds up training',
            'Add': 'Addition: Element-wise sum - often a residual/skip connection',
            'Mul': 'Multiply: Element-wise product - used in attention and gating',
            'Div': 'Division: Often scaling (e.g., by sqrt(d) in attention)',
            'Transpose': 'Transpose: Rearranges tensor dimensions',
            'Reshape': 'Reshape: Changes tensor shape without changing data',
            'Gather': 'Gather: Lookup operation - retrieves embeddings by index',
            'Concat': 'Concatenate: Joins tensors along an axis',
            'Split': 'Split: Divides tensor into parts',
            'MaxPool': 'Max Pooling: Downsamples by taking maximum in each window',
            'GlobalAveragePool': 'Global Avg Pool: Reduces spatial dims to single values',
            'Flatten': 'Flatten: Collapses dimensions into 1D for dense layers',
            'Dropout': 'Dropout: Randomly zeros values during training for regularization',
            'Attention': 'Self-Attention: Computes relationships between all positions',
            'AttentionHead': 'Attention Head: Q/K/V projections + scaled dot-product attention',
            'MLPBlock': 'MLP/FFN: Feed-forward network - expands then contracts dimensions',
        }};

        function getOpDescription(node) {{
            const opType = node.op_type || node.attributes?.block_type || '';
            return opDescriptions[opType] || `${{opType}}: Neural network operation`;
        }}

        function showTooltip(event, node) {{
            const opType = node.op_type || node.node_type;
            const description = getOpDescription(node);

            // Build detailed info
            let details = '';

            // Input/output info
            if (node.inputs && node.inputs.length > 0) {{
                details += `<div class="tooltip-row"><span class="tooltip-label">Inputs:</span><span class="tooltip-value">${{node.inputs.length}} tensor(s)</span></div>`;
            }}
            if (node.outputs && node.outputs.length > 0) {{
                details += `<div class="tooltip-row"><span class="tooltip-label">Outputs:</span><span class="tooltip-value">${{node.outputs.length}} tensor(s)</span></div>`;
            }}

            // Child count for blocks
            if (node.children && node.children.length > 0) {{
                details += `<div class="tooltip-row"><span class="tooltip-label">Contains:</span><span class="tooltip-value">${{node.node_count || node.children.length}} ops</span></div>`;
            }}

            // Compute info
            if (node.total_flops > 0) {{
                const intensity = node.total_flops > 1e9 ? 'high' : node.total_flops > 1e6 ? 'medium' : 'low';
                details += `<div class="tooltip-row"><span class="tooltip-label">Compute:</span><span class="tooltip-value">${{formatNumber(node.total_flops)}} FLOPs (${{intensity}})</span></div>`;
            }}

            // Memory info
            if (node.total_memory_bytes > 0) {{
                details += `<div class="tooltip-row"><span class="tooltip-label">Memory:</span><span class="tooltip-value">${{formatBytes(node.total_memory_bytes)}}</span></div>`;
            }}

            const html = `
                <div class="tooltip-title">${{opType}}</div>
                <div class="tooltip-desc">${{description}}</div>
                ${{details}}
            `;

            tooltip.html(html);

            // Smart positioning using fixed coordinates (relative to viewport)
            const tooltipNode = tooltip.node();
            const tooltipWidth = tooltipNode.offsetWidth || 280;
            const tooltipHeight = tooltipNode.offsetHeight || 150;
            const viewportWidth = window.innerWidth;
            const viewportHeight = window.innerHeight;

            // Use clientX/clientY for fixed positioning (viewport coords)
            let left = event.clientX + 15;
            let top = event.clientY + 15;

            // Check right edge - flip to left side if needed
            if (left + tooltipWidth > viewportWidth - 20) {{
                left = event.clientX - tooltipWidth - 15;
            }}

            // Check bottom edge - flip to top if needed
            if (top + tooltipHeight > viewportHeight - 20) {{
                top = event.clientY - tooltipHeight - 15;
            }}

            // Ensure not off left or top edge
            left = Math.max(10, left);
            top = Math.max(10, top);

            tooltip
                .style('left', left + 'px')
                .style('top', top + 'px')
                .classed('visible', true);
        }}

        function hideTooltip() {{
            tooltip.classed('visible', false);
        }}

        // Improved grid layout with depth calculation attempt
        // Falls back to clean grid if depth calc fails
        function layoutNodes(nodes) {{
            // Check if sidebar is collapsed for width calculation
            const sidebar = document.getElementById('graphSidebar');
            const sidebarWidth = sidebar && sidebar.classList.contains('collapsed') ? 0 : 280;
            const width = window.innerWidth - sidebarWidth - 40;
            const height = window.innerHeight;
            const padding = 40;

            // Try to calculate depths based on inputs/outputs
            const outputToNode = {{}};
            const depths = {{}};

            nodes.forEach(node => {{
                if (node.outputs) {{
                    node.outputs.forEach(out => {{ outputToNode[out] = node; }});
                }}
            }});

            // Simple depth: count parent chain length
            function getDepth(node, visited) {{
                if (!node) return 0;
                if (depths[node.id] !== undefined) return depths[node.id];
                if (visited[node.id]) return 0;
                visited[node.id] = true;

                let maxParent = -1;
                if (node.inputs && node.inputs.length > 0) {{
                    for (const inp of node.inputs) {{
                        const parent = outputToNode[inp];
                        if (parent) {{
                            maxParent = Math.max(maxParent, getDepth(parent, visited));
                        }}
                    }}
                }}
                depths[node.id] = maxParent + 1;
                return depths[node.id];
            }}

            // Calculate all depths
            nodes.forEach(n => getDepth(n, {{}}));

            // Group by depth
            const byDepth = {{}};
            let maxDepth = 0;
            nodes.forEach(node => {{
                const d = depths[node.id] || 0;
                maxDepth = Math.max(maxDepth, d);
                if (!byDepth[d]) byDepth[d] = [];
                byDepth[d].push(node);
            }});

            // If we got meaningful depths, use horizontal flow layout
            if (maxDepth > 0) {{
                // Prefer horizontal spread - use full width
                const colWidth = (width - padding * 2) / (maxDepth + 1);
                const availableHeight = height - padding * 2;

                for (let d = 0; d <= maxDepth; d++) {{
                    const nodesAtDepth = byDepth[d] || [];
                    const x = padding + d * colWidth + colWidth / 2;

                    // Distribute vertically within available height
                    const rowH = availableHeight / Math.max(nodesAtDepth.length, 1);

                    nodesAtDepth.forEach((node, i) => {{
                        node.x = x;
                        node.y = padding + i * rowH + rowH / 2;
                        node.r = getNodeSize(node);
                    }});
                }}
            }} else {{
                // Fallback: horizontal-biased grid (more columns than rows)
                const cols = Math.ceil(Math.sqrt(nodes.length * 2.5));
                const rows = Math.ceil(nodes.length / cols);
                const cellW = (width - padding * 2) / cols;
                const cellH = (height - padding * 2) / rows;

                nodes.forEach((node, i) => {{
                    node.x = padding + (i % cols) * cellW + cellW / 2;
                    node.y = padding + Math.floor(i / cols) * cellH + cellH / 2;
                    node.r = getNodeSize(node);
                }});
            }}

            return nodes;
        }}

        // Render graph with performance optimizations (Task 5.7.9)
        function render() {{
            container.selectAll('*').remove();

            // Flatten visible nodes
            let visibleNodes = [];
            function collectVisible(node) {{
                visibleNodes.push(node);
                if (!node.is_collapsed && node.children) {{
                    node.children.forEach(collectVisible);
                }}
            }}

            if (graphData.root) {{
                collectVisible(graphData.root);
            }}

            // Performance: limit visible nodes in performance mode
            if (performanceMode && visibleNodes.length > 1000) {{
                console.log(`Limiting display to 1000 nodes (had ${{visibleNodes.length}})`);
                // Prioritize by FLOPs/importance
                visibleNodes = visibleNodes
                    .sort((a, b) => (b.total_flops || 0) - (a.total_flops || 0))
                    .slice(0, 1000);
            }}

            // Compute max FLOPs for size scaling (expensive ops = bigger nodes)
            globalMaxFlops = Math.max(1, ...visibleNodes.map(n => n.total_flops || 0));

            // Layout
            layoutNodes(visibleNodes);

            // Create node lookup for edges
            const nodeById = {{}};
            visibleNodes.forEach(n => {{
                nodeById[n.id] = n;
                nodeById[n.name] = n;
            }});

            // Draw edges first (so they're behind nodes)
            const edges = [];
            visibleNodes.forEach(node => {{
                if (node.outputs) {{
                    node.outputs.forEach(output => {{
                        // Find nodes that consume this output
                        visibleNodes.forEach(target => {{
                            if (target.inputs && target.inputs.includes(output)) {{
                                edges.push({{
                                    source: node,
                                    target: target,
                                    output: output
                                }});
                            }}
                        }});
                    }});
                }}
            }});

            // Setup defs for markers and filters
            const defs = container.append('defs');

            // Arrow marker for edges
            defs.append('marker')
                .attr('id', 'arrowhead')
                .attr('viewBox', '0 -5 10 10')
                .attr('refX', 8)
                .attr('refY', 0)
                .attr('markerWidth', 5)
                .attr('markerHeight', 5)
                .attr('orient', 'auto')
                .append('path')
                .attr('d', 'M0,-3L8,0L0,3')
                .attr('fill', 'rgba(255,255,255,0.5)');

            // Glow filter for nodes
            const filter = defs.append('filter')
                .attr('id', 'glow');
            filter.append('feGaussianBlur')
                .attr('stdDeviation', '2')
                .attr('result', 'coloredBlur');

            // Draw edges with arrows
            container.selectAll('.edge')
                .data(edges)
                .enter()
                .append('path')
                .attr('class', 'edge')
                .attr('d', d => {{
                    const sr = d.source.r || 20;
                    const tr = d.target.r || 20;
                    const sx = d.source.x + sr;
                    const sy = d.source.y;
                    const tx = d.target.x - tr;
                    const ty = d.target.y;

                    // Bezier curve
                    const midX = (sx + tx) / 2;
                    return `M${{sx}},${{sy}} C${{midX}},${{sy}} ${{midX}},${{ty}} ${{tx}},${{ty}}`;
                }})
                .attr('stroke', 'rgba(255,255,255,0.3)')
                .attr('stroke-width', 1.5)
                .attr('marker-end', 'url(#arrowhead)');

            // Draw nodes as circles
            const nodeGroups = container.selectAll('.node')
                .data(visibleNodes)
                .enter()
                .append('g')
                .attr('class', d => {{
                    // Show labels for important nodes: model, layers, blocks
                    const isImportant = d.node_type === 'model' ||
                                       d.node_type === 'layer' ||
                                       d.node_type === 'block' ||
                                       d.r > 35;
                    return 'node' + (isImportant ? ' show-label' : '');
                }})
                .attr('transform', d => `translate(${{d.x}}, ${{d.y}})`)
                .on('mouseover', showTooltip)
                .on('mouseout', hideTooltip)
                .on('click', (event, d) => {{
                    if (d.children && d.children.length > 0) {{
                        d.is_collapsed = !d.is_collapsed;
                        render();
                    }}
                }});

            // Calculate max FLOPs for heat map
            const maxFlops = getMaxFlops(visibleNodes);

            // Circle nodes - use heat map or category colors
            nodeGroups.append('circle')
                .attr('class', 'node-circle')
                .attr('r', d => d.r)
                .attr('fill', d => {{
                    if (heatMapMode && d.total_flops > 0) {{
                        return getHeatColor(d.total_flops, maxFlops);
                    }}
                    return getNodeColor(d);
                }})
                .attr('opacity', d => d.node_type === 'model' ? 0.9 : 0.85)
                .style('filter', 'url(#glow)');

            // Inner highlight
            nodeGroups.append('circle')
                .attr('r', d => d.r - 2)
                .attr('fill', 'none')
                .attr('stroke', 'rgba(255,255,255,0.2)')
                .attr('stroke-width', 1);

            // Symbol in center
            nodeGroups.append('text')
                .attr('class', 'node-symbol')
                .attr('y', d => d.r > 28 ? -4 : 0)
                .attr('font-size', d => d.r > 28 ? '14px' : '12px')
                .attr('fill', 'white')
                .attr('text-anchor', 'middle')
                .attr('dominant-baseline', 'middle')
                .text(d => getNodeSymbol(d));

            // Labels below symbol for larger nodes
            nodeGroups.filter(d => d.r > 28)
                .append('text')
                .attr('class', 'node-label')
                .attr('y', 10)
                .text(d => truncate(getNodeLabel(d), 8));

            // Just label for small nodes
            nodeGroups.filter(d => d.r <= 28)
                .append('text')
                .attr('class', 'node-label')
                .attr('y', d => d.r + 14)
                .text(d => truncate(getNodeLabel(d), 6));

            // Expand indicator for blocks
            nodeGroups.filter(d => d.children && d.children.length > 0)
                .append('circle')
                .attr('cx', d => d.r * 0.7)
                .attr('cy', d => -d.r * 0.7)
                .attr('r', 8)
                .attr('fill', '#0A84FF')
                .attr('stroke', '#000')
                .attr('stroke-width', 1);

            nodeGroups.filter(d => d.children && d.children.length > 0)
                .append('text')
                .attr('x', d => d.r * 0.7)
                .attr('y', d => -d.r * 0.7 + 1)
                .attr('text-anchor', 'middle')
                .attr('dominant-baseline', 'middle')
                .attr('fill', 'white')
                .attr('font-size', '10px')
                .attr('font-weight', '600')
                .text(d => d.is_collapsed ? '+' : '-');

            // Update stats
            document.getElementById('node-count').textContent = visibleNodes.length;
            document.getElementById('edge-count').textContent = edges.length || edgeData?.num_edges || 0;
            document.getElementById('peak-memory').textContent = formatBytes(edgeData?.peak_activation_bytes || 0);
            document.getElementById('model-size').textContent = formatBytes(graphData.model_size_bytes || 0);
        }}

        // Sidebar toggle for more graph space
        function toggleSidebar() {{
            const sidebar = document.getElementById('graphSidebar');
            const toggleBtn = document.getElementById('sidebarToggle');
            const isCollapsed = sidebar.classList.toggle('collapsed');
            toggleBtn.textContent = isCollapsed ? '▶' : '◀';
            toggleBtn.style.left = isCollapsed ? '8px' : '248px';
            // Re-fit graph after sidebar toggle
            setTimeout(fitToScreen, 350);
        }}

        function zoomIn() {{
            svg.transition().duration(300).call(zoom.scaleBy, 1.3);
        }}

        function zoomOut() {{
            svg.transition().duration(300).call(zoom.scaleBy, 0.7);
        }}

        let activeFilter = 'all';
        let showAllLabels = false;

        function toggleAllLabels() {{
            showAllLabels = !showAllLabels;
            const btn = document.getElementById('labels-btn');
            btn.style.background = showAllLabels ? 'var(--accent)' : '';
            btn.style.color = showAllLabels ? 'white' : '';
            btn.textContent = showAllLabels ? 'Hide Labels' : 'Show All Labels';

            // Toggle show-label class on all nodes
            container.selectAll('.node').classed('show-label', showAllLabels);
        }}

        function toggleHeatMap() {{
            heatMapMode = !heatMapMode;

            const btn = document.getElementById('heatmap-btn');
            btn.style.background = heatMapMode ? 'var(--accent)' : '';
            btn.style.color = heatMapMode ? 'white' : '';

            // Toggle legend visibility
            document.getElementById('heatmap-legend').style.display = heatMapMode ? 'block' : 'none';
            document.getElementById('optype-legend-note').style.display = heatMapMode ? 'none' : 'block';

            render();
        }}

        function filterByCategory(category) {{
            activeFilter = category;

            // Update legend active state
            document.querySelectorAll('#op-legend .legend-item').forEach(item => {{
                item.classList.remove('active');
                if (item.dataset.category === category) {{
                    item.classList.add('active');
                }}
            }});

            // Update node visibility/opacity
            container.selectAll('.node').each(function(d) {{
                const nodeCategory = getOpCategory(d);
                const visible = category === 'all' || nodeCategory === category;
                d3.select(this)
                    .transition()
                    .duration(200)
                    .style('opacity', visible ? 1 : 0.15);
            }});
        }}

        // Calculate max FLOPs for heat map scaling
        function getMaxFlops(nodes) {{
            let max = 0;
            nodes.forEach(n => {{
                if (n.total_flops > max) max = n.total_flops;
            }});
            return max || 1;
        }}

        // Helper functions
        function truncate(str, len) {{
            return str.length > len ? str.slice(0, len) + '...' : str;
        }}

        function formatNumber(n) {{
            if (n >= 1e12) return (n / 1e12).toFixed(1) + 'T';
            if (n >= 1e9) return (n / 1e9).toFixed(1) + 'G';
            if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
            if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
            return n.toString();
        }}

        function formatBytes(bytes) {{
            if (bytes >= 1e9) return (bytes / 1e9).toFixed(2) + ' GB';
            if (bytes >= 1e6) return (bytes / 1e6).toFixed(1) + ' MB';
            if (bytes >= 1e3) return (bytes / 1e3).toFixed(1) + ' KB';
            return bytes + ' B';
        }}

        // Control functions
        function expandAll() {{
            function expand(node) {{
                node.is_collapsed = false;
                if (node.children) node.children.forEach(expand);
            }}
            if (graphData.root) expand(graphData.root);
            render();
        }}

        function collapseAll() {{
            function collapse(node) {{
                if (node.children && node.children.length > 0) {{
                    node.is_collapsed = true;
                }}
                if (node.children) node.children.forEach(collapse);
            }}
            if (graphData.root) collapse(graphData.root);
            render();
        }}

        function resetZoom() {{
            svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity);
        }}

        function fitToScreen() {{
            const bounds = container.node().getBBox();
            const parent = svg.node().parentElement;
            const fullWidth = parent.clientWidth;
            const fullHeight = parent.clientHeight;
            const width = bounds.width;
            const height = bounds.height;
            const midX = bounds.x + width / 2;
            const midY = bounds.y + height / 2;

            const scale = 0.9 / Math.max(width / fullWidth, height / fullHeight);
            const translate = [fullWidth / 2 - scale * midX, fullHeight / 2 - scale * midY];

            svg.transition().duration(500).call(
                zoom.transform,
                d3.zoomIdentity.translate(translate[0], translate[1]).scale(scale)
            );
        }}

        // Task 5.7.6: Search functionality
        let allFlatNodes = [];

        function buildSearchIndex() {{
            allFlatNodes = [];
            function collect(node) {{
                allFlatNodes.push(node);
                if (node.children) node.children.forEach(collect);
            }}
            if (graphData.root) collect(graphData.root);
        }}

        function searchNodes(query) {{
            const resultsDiv = document.getElementById('searchResults');
            resultsDiv.innerHTML = '';

            if (!query || query.length < 2) {{
                // Clear highlights
                container.selectAll('.node').classed('search-highlight', false);
                return;
            }}

            query = query.toLowerCase();
            const matches = allFlatNodes.filter(node => {{
                const name = (node.name || '').toLowerCase();
                const opType = (node.op_type || '').toLowerCase();
                const blockType = (node.attributes?.block_type || '').toLowerCase();
                return name.includes(query) || opType.includes(query) || blockType.includes(query);
            }}).slice(0, 20);  // Limit to 20 results

            if (matches.length === 0) {{
                resultsDiv.innerHTML = '<div style="font-size: 0.7rem; color: var(--text-tertiary); padding: 8px;">No matches found</div>';
                return;
            }}

            matches.forEach(node => {{
                const div = document.createElement('div');
                div.className = 'search-result';
                div.innerHTML = `
                    <span class="result-name">${{node.name}}</span>
                    <span class="result-type">${{node.op_type || node.node_type}}</span>
                `;
                div.onclick = () => highlightAndFocusNode(node);
                resultsDiv.appendChild(div);
            }});
        }}

        function highlightAndFocusNode(targetNode) {{
            // Expand path to node if collapsed
            function expandToNode(node, target) {{
                if (node.id === target.id) return true;
                if (node.children) {{
                    for (const child of node.children) {{
                        if (expandToNode(child, target)) {{
                            node.is_collapsed = false;
                            return true;
                        }}
                    }}
                }}
                return false;
            }}

            if (graphData.root) expandToNode(graphData.root, targetNode);
            render();

            // Highlight the node
            setTimeout(() => {{
                container.selectAll('.node').classed('search-highlight', false);
                container.selectAll('.node')
                    .filter(d => d.id === targetNode.id)
                    .classed('search-highlight', true);

                // Pan to node
                const nodeElem = container.selectAll('.node').filter(d => d.id === targetNode.id);
                if (!nodeElem.empty()) {{
                    const d = nodeElem.datum();
                    const width = window.innerWidth - 320;
                    const height = window.innerHeight;
                    const scale = 1.5;
                    const x = width / 2 - d.x * scale;
                    const y = height / 2 - d.y * scale;

                    svg.transition().duration(500).call(
                        zoom.transform,
                        d3.zoomIdentity.translate(x, y).scale(scale)
                    );
                }}
            }}, 100);
        }}

        // Task 5.7.9: Performance mode for large graphs
        const LARGE_GRAPH_THRESHOLD = 500;
        const VERY_LARGE_THRESHOLD = 5000;
        let performanceMode = false;
        let cullingEnabled = false;

        function checkPerformanceMode() {{
            const totalNodes = graphData.total_nodes || 0;

            if (totalNodes > VERY_LARGE_THRESHOLD) {{
                performanceMode = true;
                cullingEnabled = true;
                console.log(`Performance mode: ON (${{totalNodes}} nodes). Culling enabled.`);

                // Add indicator
                const sidebar = document.querySelector('.sidebar');
                const perfDiv = document.createElement('div');
                perfDiv.className = 'perf-mode';
                perfDiv.innerHTML = `Large model (${{formatNumber(totalNodes)}} nodes) - simplified view`;
                sidebar.insertBefore(perfDiv, sidebar.firstChild.nextSibling);

                // Auto-collapse to depth 1
                if (graphData.root) {{
                    function collapseDeep(node, depth) {{
                        if (depth > 1 && node.children && node.children.length > 0) {{
                            node.is_collapsed = true;
                        }}
                        if (node.children) node.children.forEach(c => collapseDeep(c, depth + 1));
                    }}
                    collapseDeep(graphData.root, 0);
                }}
            }} else if (totalNodes > LARGE_GRAPH_THRESHOLD) {{
                performanceMode = true;
                console.log(`Performance mode: ON (${{totalNodes}} nodes). Simplified rendering.`);
            }}
        }}

        // Viewport culling for very large graphs
        function getViewportBounds() {{
            const transform = d3.zoomTransform(svg.node());
            const width = window.innerWidth - 320;
            const height = window.innerHeight;

            return {{
                x: -transform.x / transform.k - 100,
                y: -transform.y / transform.k - 100,
                width: width / transform.k + 200,
                height: height / transform.k + 200
            }};
        }}

        function isInViewport(node, bounds) {{
            if (!cullingEnabled) return true;
            if (!node.x || !node.y) return true;
            const r = node.r || 30;
            return node.x + r >= bounds.x &&
                   node.x - r <= bounds.x + bounds.width &&
                   node.y + r >= bounds.y &&
                   node.y - r <= bounds.y + bounds.height;
        }}

        // Initialize
        buildSearchIndex();
        checkPerformanceMode();

        // Check if embedded in iframe (Streamlit) - auto-collapse sidebar for more space
        const isEmbedded = window.self !== window.top;
        if (isEmbedded) {{
            // Start with sidebar collapsed in embedded mode
            setTimeout(() => {{
                const sidebar = document.getElementById('graphSidebar');
                const toggleBtn = document.getElementById('sidebarToggle');
                if (sidebar && toggleBtn) {{
                    sidebar.classList.add('collapsed');
                    toggleBtn.textContent = '▶';
                    toggleBtn.style.left = '8px';
                }}
            }}, 50);
        }}

        // Preferences persistence
        function loadPrefs() {{
            try {{
                const prefs = JSON.parse(localStorage.getItem('haoline_graph_prefs') || '{{}}');
                document.getElementById('autoFitCheck').checked = prefs.autoFit !== false; // default true
                document.getElementById('autoExpandCheck').checked = prefs.autoExpand === true; // default false
                return prefs;
            }} catch (e) {{
                return {{ autoFit: true, autoExpand: false }};
            }}
        }}

        function savePrefs() {{
            const prefs = {{
                autoFit: document.getElementById('autoFitCheck').checked,
                autoExpand: document.getElementById('autoExpandCheck').checked
            }};
            localStorage.setItem('haoline_graph_prefs', JSON.stringify(prefs));
        }}

        // Initial render with preferences
        const prefs = loadPrefs();
        render();

        // Apply preferences after render
        setTimeout(() => {{
            if (prefs.autoExpand) {{
                expandAll();
            }}
            if (prefs.autoFit) {{
                setTimeout(fitToScreen, 100);
            }}
        }}, isEmbedded ? 300 : 150);
    </script>
</body>
</html>
"""


def generate_html(
    graph: HierarchicalGraph,
    edge_result: EdgeAnalysisResult | None = None,
    title: str = "Model Architecture",
    output_path: Path | str | None = None,
    model_size_bytes: int | None = None,
    layer_timing: dict[str, float] | None = None,
) -> str:
    """
    Generate interactive HTML visualization.

    Task 5.8.1-5.8.5: Create standalone HTML with embedded visualization.

    Args:
        graph: HierarchicalGraph to visualize.
        edge_result: Optional edge analysis results.
        title: Page title.
        output_path: Optional path to save HTML file.
        model_size_bytes: Optional model file size in bytes.
        layer_timing: Optional dict mapping layer name to execution time (ms).
            Used to color-code nodes by execution time (hotspots).

    Returns:
        HTML content as string.
    """
    # Convert graph to JSON, adding model_size_bytes and layer_timing
    graph_dict = graph.to_dict()
    if model_size_bytes is not None:
        graph_dict["model_size_bytes"] = model_size_bytes
    if layer_timing is not None:
        graph_dict["layer_timing"] = layer_timing
    graph_json = json.dumps(graph_dict)

    # Convert edge analysis to JSON
    if edge_result:
        edge_json = json.dumps(edge_result.to_dict())
    else:
        edge_json = "{}"

    # Generate HTML
    html = HTML_TEMPLATE.format(
        title=title,
        graph_json=graph_json,
        edge_json=edge_json,
    )

    # Save to file if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.write_text(html, encoding="utf-8")

    return html


class HTMLExporter:
    """
    Export ONNX models to interactive HTML visualization.

    Combines HierarchicalGraph and EdgeAnalysis into a single
    standalone HTML file.
    """

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger("haoline.html")

    def export(
        self,
        graph: HierarchicalGraph,
        edge_result: EdgeAnalysisResult | None = None,
        output_path: Path | str = "model_graph.html",
        title: str | None = None,
        model_size_bytes: int | None = None,
        layer_timing: dict[str, float] | None = None,
    ) -> Path:
        """
        Export graph to HTML file.

        Args:
            graph: HierarchicalGraph to export.
            edge_result: Optional EdgeAnalysisResult for edge data.
            output_path: Output file path.
            title: Optional page title.
            model_size_bytes: Optional model file size in bytes.
            layer_timing: Optional dict mapping layer name to execution time (ms).

        Returns:
            Path to generated HTML file.
        """
        output_path = Path(output_path)

        if title is None:
            title = graph.root.name if graph.root else "Model"

        generate_html(
            graph=graph,
            edge_result=edge_result,
            title=title,
            output_path=output_path,
            model_size_bytes=model_size_bytes,
            layer_timing=layer_timing,
        )

        self.logger.info(f"HTML visualization exported to {output_path}")
        return output_path
