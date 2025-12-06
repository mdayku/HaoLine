# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
Hierarchical Graph View for visualization.

Task 5.7: Creates collapsible, multi-level graph representations
for LLM-scale models where flat visualization would be unreadable.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from .analyzer import GraphInfo
    from .patterns import Block


class HierarchicalNode(BaseModel):
    """
    A node in the hierarchical graph that can contain children.

    Task 5.7.1: Create HierarchicalNode with children.
    """

    model_config = ConfigDict(frozen=False)  # Allow mutation for collapse/expand

    id: str
    name: str
    node_type: str  # "op", "block", "layer", "model"
    op_type: str | None = None  # ONNX op type for leaf nodes

    # Hierarchy
    children: list[HierarchicalNode] = Field(default_factory=list)
    parent_id: str | None = None
    depth: int = 0

    # State
    is_collapsed: bool = True  # Task 5.7.3
    is_repeated: bool = False
    repeat_count: int = 1  # Task 5.7.5: xN notation

    # Aggregated stats (Task 5.7.4)
    total_flops: int = 0
    total_params: int = 0
    total_memory_bytes: int = 0
    node_count: int = 1

    # Inputs/outputs for edges
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)

    # Metadata
    attributes: dict = Field(default_factory=dict)

    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)."""
        return len(self.children) == 0

    def get_display_name(self) -> str:
        """Get display name with repeat notation."""
        if self.repeat_count > 1:
            return f"{self.name} x{self.repeat_count}"
        return self.name

    def collapse(self) -> None:
        """Collapse this node."""
        self.is_collapsed = True

    def expand(self) -> None:
        """Expand this node."""
        self.is_collapsed = False

    def toggle(self) -> None:
        """Toggle collapsed state."""
        self.is_collapsed = not self.is_collapsed

    def aggregate_stats(self) -> None:
        """
        Aggregate statistics from children.

        Task 5.7.4: Aggregate stats for collapsed blocks.
        """
        if self.is_leaf():
            return

        self.total_flops = 0
        self.total_params = 0
        self.total_memory_bytes = 0
        self.node_count = 0

        for child in self.children:
            child.aggregate_stats()
            self.total_flops += child.total_flops
            self.total_params += child.total_params
            self.total_memory_bytes += child.total_memory_bytes
            self.node_count += child.node_count

        # Apply repeat multiplier
        if self.repeat_count > 1:
            self.total_flops *= self.repeat_count
            self.total_memory_bytes *= self.repeat_count
            self.node_count *= self.repeat_count
            # Params are shared, don't multiply

    def to_dict(self, include_children: bool = True) -> dict[str, Any]:
        """
        Convert to dictionary for JSON export.

        Task 5.7.6: JSON export for visualization.
        """
        result: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "display_name": self.get_display_name(),
            "node_type": self.node_type,
            "op_type": self.op_type,
            "depth": self.depth,
            "is_collapsed": self.is_collapsed,
            "is_repeated": self.is_repeated,
            "repeat_count": self.repeat_count,
            "total_flops": self.total_flops,
            "total_params": self.total_params,
            "total_memory_bytes": self.total_memory_bytes,
            "node_count": self.node_count,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "attributes": self.attributes,
        }

        if include_children and self.children:
            result["children"] = [c.to_dict(include_children) for c in self.children]

        return result


class HierarchicalGraph(BaseModel):
    """Complete hierarchical graph representation."""

    model_config = ConfigDict(frozen=False)  # Allow mutation for collapse/expand

    root: HierarchicalNode
    nodes_by_id: dict[str, HierarchicalNode] = Field(default_factory=dict)
    total_nodes: int = 0
    depth: int = 0

    def get_node(self, node_id: str) -> HierarchicalNode | None:
        """Get node by ID."""
        return self.nodes_by_id.get(node_id)

    def collapse_all(self) -> None:
        """Collapse all non-leaf nodes."""
        for node in self.nodes_by_id.values():
            if not node.is_leaf():
                node.collapse()

    def expand_all(self) -> None:
        """Expand all nodes."""
        for node in self.nodes_by_id.values():
            node.expand()

    def expand_to_depth(self, max_depth: int) -> None:
        """Expand nodes up to a certain depth."""
        for node in self.nodes_by_id.values():
            if node.depth <= max_depth:
                node.expand()
            else:
                node.collapse()

    def get_visible_nodes(self) -> list[HierarchicalNode]:
        """Get nodes visible given current collapse state."""
        visible: list[HierarchicalNode] = []
        self._collect_visible(self.root, visible)
        return visible

    def _collect_visible(self, node: HierarchicalNode, visible: list[HierarchicalNode]) -> None:
        """Recursively collect visible nodes."""
        visible.append(node)
        if not node.is_collapsed:
            for child in node.children:
                self._collect_visible(child, visible)

    def to_dict(self) -> dict:
        """Export to dictionary."""
        return {
            "root": self.root.to_dict(),
            "total_nodes": self.total_nodes,
            "depth": self.depth,
        }

    def to_json(self, indent: int | None = 2) -> str:
        """Export to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class HierarchicalGraphBuilder:
    """
    Build hierarchical graph from flat ONNX graph and detected blocks.

    Task 5.7.2: Convert Blocks to HierarchicalNodes.
    """

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger("haoline.hierarchy")

    def build(
        self,
        graph_info: GraphInfo,
        blocks: list[Block],
        model_name: str = "Model",
    ) -> HierarchicalGraph:
        """
        Build hierarchical graph from ONNX graph and detected blocks.

        Args:
            graph_info: Parsed ONNX graph.
            blocks: Detected architectural blocks.
            model_name: Name for the root node.

        Returns:
            HierarchicalGraph with multi-level structure.
        """
        nodes_by_id: dict[str, HierarchicalNode] = {}

        # Create root node
        root = HierarchicalNode(
            id="root",
            name=model_name,
            node_type="model",
            depth=0,
            is_collapsed=False,
        )
        nodes_by_id["root"] = root

        # Map op nodes to block memberships
        node_to_block: dict[str, str] = {}
        for block in blocks:
            for node_name in block.nodes:
                if node_name not in node_to_block:
                    node_to_block[node_name] = block.name

        # Create block nodes
        block_nodes: dict[str, HierarchicalNode] = {}
        for block in blocks:
            block_node = HierarchicalNode(
                id=f"block_{block.name}",
                name=block.name,
                node_type="block",
                depth=1,
                parent_id="root",
                attributes=block.attributes.copy() if block.attributes else {},
            )
            block_node.attributes["block_type"] = block.block_type
            block_nodes[block.name] = block_node
            nodes_by_id[block_node.id] = block_node

        # Process repeated blocks (Task 5.7.5)
        repeated_blocks = [b for b in blocks if b.block_type == "RepeatedBlock"]
        for rb in repeated_blocks:
            rb.attributes.get("repeated_type", "")
            repeat_count = rb.attributes.get("num_repetitions", 1)
            block_names = rb.attributes.get("block_names", [])

            # Mark the first block as repeated, hide others
            if block_names and repeat_count > 1:
                first_block_name = block_names[0]
                if first_block_name in block_nodes:
                    block_nodes[first_block_name].is_repeated = True
                    block_nodes[first_block_name].repeat_count = repeat_count

                # Remove subsequent blocks from display (they're represented by xN)
                for name in block_names[1:]:
                    if name in block_nodes:
                        # Mark as hidden by adding to a "collapsed repeated" group
                        block_nodes[name].attributes["hidden_by_repeat"] = True

        # Create op nodes and assign to blocks or root
        for node in graph_info.nodes:
            op_node = HierarchicalNode(
                id=f"op_{node.name}",
                name=node.name,
                node_type="op",
                op_type=node.op_type,
                inputs=list(node.inputs),
                outputs=list(node.outputs),
            )

            # Get stats from graph_info if available
            if hasattr(graph_info, "node_flops"):
                op_node.total_flops = graph_info.node_flops.get(node.name, 0)

            if node.name in node_to_block:
                # Node belongs to a block
                block_name = node_to_block[node.name]
                if block_name in block_nodes:
                    parent = block_nodes[block_name]
                    op_node.parent_id = parent.id
                    op_node.depth = parent.depth + 1
                    parent.children.append(op_node)
            else:
                # Standalone node at root level
                op_node.parent_id = "root"
                op_node.depth = 1
                root.children.append(op_node)

            nodes_by_id[op_node.id] = op_node

        # Add non-hidden blocks to root
        for block_node in block_nodes.values():
            if not block_node.attributes.get("hidden_by_repeat", False):
                root.children.append(block_node)

        # Aggregate stats up the tree
        root.aggregate_stats()

        # Calculate depth
        max_depth = max(n.depth for n in nodes_by_id.values())

        return HierarchicalGraph(
            root=root,
            nodes_by_id=nodes_by_id,
            total_nodes=len(nodes_by_id),
            depth=max_depth,
        )

    def build_layer_hierarchy(
        self,
        graph_info: GraphInfo,
        blocks: list[Block],
        model_name: str = "Model",
    ) -> HierarchicalGraph:
        """
        Build a 3-level hierarchy: Model -> Layers -> Blocks -> Ops.

        For LLMs, groups attention+MLP into "TransformerLayer" containers.
        """
        basic_graph = self.build(graph_info, blocks, model_name)

        # Group consecutive attention + MLP blocks into layers
        layers: list[HierarchicalNode] = []
        current_layer_blocks: list[HierarchicalNode] = []
        layer_idx = 0

        for child in basic_graph.root.children:
            if child.node_type == "block":
                block_type = child.attributes.get("block_type", "")

                # Check if this starts a new layer
                if block_type in ("AttentionHead", "Attention") and current_layer_blocks:
                    # Previous blocks become a layer
                    if len(current_layer_blocks) > 1:
                        layer = self._create_layer_node(current_layer_blocks, layer_idx)
                        layers.append(layer)
                        layer_idx += 1
                    else:
                        layers.extend(current_layer_blocks)
                    current_layer_blocks = []

                current_layer_blocks.append(child)
            else:
                # Non-block node, flush any pending layer
                if current_layer_blocks:
                    if len(current_layer_blocks) > 1:
                        layer = self._create_layer_node(current_layer_blocks, layer_idx)
                        layers.append(layer)
                        layer_idx += 1
                    else:
                        layers.extend(current_layer_blocks)
                    current_layer_blocks = []
                layers.append(child)

        # Handle remaining blocks
        if current_layer_blocks:
            if len(current_layer_blocks) > 1:
                layer = self._create_layer_node(current_layer_blocks, layer_idx)
                layers.append(layer)
            else:
                layers.extend(current_layer_blocks)

        # Update root children
        basic_graph.root.children = layers

        # Update nodes_by_id
        for layer in layers:
            if layer.node_type == "layer":
                basic_graph.nodes_by_id[layer.id] = layer

        # Recalculate depths
        self._recalculate_depths(basic_graph.root, 0)
        basic_graph.depth = max(n.depth for n in basic_graph.nodes_by_id.values())

        # Re-aggregate stats
        basic_graph.root.aggregate_stats()

        return basic_graph

    def _create_layer_node(
        self, blocks: list[HierarchicalNode], layer_idx: int
    ) -> HierarchicalNode:
        """Create a layer node containing multiple blocks."""
        layer = HierarchicalNode(
            id=f"layer_{layer_idx}",
            name=f"Layer {layer_idx}",
            node_type="layer",
            depth=1,
            parent_id="root",
            children=blocks,
        )

        # Update children's parent
        for block in blocks:
            block.parent_id = layer.id
            block.depth = 2
            # Update children of blocks
            for child in block.children:
                child.depth = 3

        return layer

    def _recalculate_depths(self, node: HierarchicalNode, depth: int) -> None:
        """Recursively recalculate depths."""
        node.depth = depth
        for child in node.children:
            self._recalculate_depths(child, depth + 1)


def generate_summary(graph: HierarchicalGraph) -> str:
    """
    Generate multi-level text summary.

    Task 5.7.5: Generate multi-level summary (xN notation).
    """
    lines = []
    lines.append(f"# {graph.root.name}")
    lines.append(f"Total Nodes: {graph.total_nodes}")
    lines.append(f"Max Depth: {graph.depth}")
    lines.append("")

    def format_node(node: HierarchicalNode, indent: int = 0) -> None:
        prefix = "  " * indent
        display = node.get_display_name()

        if node.node_type == "op":
            lines.append(f"{prefix}- {display} ({node.op_type})")
        elif node.node_type == "block":
            block_type = node.attributes.get("block_type", "Block")
            lines.append(f"{prefix}[{block_type}] {display}")
        elif node.node_type == "layer":
            lines.append(f"{prefix}== {display} ==")
        else:
            lines.append(f"{prefix}{display}")

        if not node.is_collapsed:
            for child in node.children:
                format_node(child, indent + 1)

    for child in graph.root.children:
        format_node(child, 0)

    return "\n".join(lines)
