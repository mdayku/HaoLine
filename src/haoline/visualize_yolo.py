#!/usr/bin/env python3
"""Visualize a YOLO model."""

import sys
from pathlib import Path

sys.path.insert(0, str(__file__).replace("\\", "/").rsplit("/", 4)[0])

from util.haoline.analyzer import ONNXGraphLoader
from util.haoline.edge_analysis import EdgeAnalyzer
from util.haoline.hierarchical_graph import HierarchicalGraphBuilder
from util.haoline.html_export import HTMLExporter
from util.haoline.patterns import PatternAnalyzer

model_path = Path(
    r"C:\Users\marcu\Roomer\room_detection_training\local_training_output\yolo-v8l-200epoch\weights\best.onnx"
)

print(f"Loading {model_path.name}...")
loader = ONNXGraphLoader()
_, graph_info = loader.load(model_path)

print("Detecting patterns...")
pattern_analyzer = PatternAnalyzer()
blocks = pattern_analyzer.group_into_blocks(graph_info)

print("Analyzing edges...")
edge_analyzer = EdgeAnalyzer()
edge_result = edge_analyzer.analyze(graph_info)

print("Building hierarchy...")
builder = HierarchicalGraphBuilder()
hier_graph = builder.build(graph_info, blocks, model_path.stem)

print("Exporting HTML...")
exporter = HTMLExporter()
output_path = exporter.export(
    hier_graph,
    edge_result,
    output_path=model_path.with_suffix(".html"),
    title=model_path.stem,
)

print(f"Done! Open {output_path}")
