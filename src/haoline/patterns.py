# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
Pattern detection for HaoLine.

Detects common architectural patterns in ONNX graphs:
- Conv-BatchNorm-ReLU blocks
- Residual/skip connections
- Transformer blocks (attention + MLP)
- Embedding layers
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from .analyzer import GraphInfo, NodeInfo


class Block(BaseModel):
    """
    A detected architectural block (group of related nodes).

    Blocks represent higher-level patterns like "ResidualBlock" or
    "TransformerLayer" that consist of multiple ONNX nodes.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    block_type: str  # e.g., "ConvBNRelu", "ResidualBlock", "TransformerBlock"
    name: str
    nodes: list[str]  # Node names in this block
    start_node: str
    end_node: str
    attributes: dict[str, Any] = Field(default_factory=dict)  # Block-specific metadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_type": self.block_type,
            "name": self.name,
            "nodes": self.nodes,
            "start_node": self.start_node,
            "end_node": self.end_node,
            "attributes": self.attributes,
        }


class PatternAnalyzer:
    """
    Detect architectural patterns in ONNX graphs.

    Identifies common patterns like Conv-BN-ReLU sequences, residual
    blocks, and transformer attention blocks.
    """

    # Operators that commonly appear together
    CONV_ACTIVATIONS: ClassVar[set[str]] = {
        "Relu",
        "LeakyRelu",
        "Sigmoid",
        "Tanh",
        "Clip",
        "HardSwish",
        "Silu",
    }
    NORM_OPS: ClassVar[set[str]] = {
        "BatchNormalization",
        "InstanceNormalization",
        "LayerNormalization",
        "GroupNormalization",
    }
    ATTENTION_OPS: ClassVar[set[str]] = {"MatMul", "Softmax", "Transpose"}
    EMBEDDING_OPS: ClassVar[set[str]] = {"Gather", "Embedding"}

    # LLM-specific activation functions
    LLM_ACTIVATIONS: ClassVar[set[str]] = {
        "Gelu",
        "FastGelu",
        "QuickGelu",  # GPT-style
        "Silu",
        "Swish",  # LLaMA/Mistral style
        "Relu",
        "LeakyRelu",  # Classic
        "NewGelu",  # Some implementations
    }

    # MoE-related ops
    MOE_OPS: ClassVar[set[str]] = {"TopK", "Scatter", "ScatterND", "GatherND"}

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger("haoline.patterns")

    def group_into_blocks(self, graph_info: GraphInfo) -> list[Block]:
        """
        Detect all architectural blocks in the graph.

        Args:
            graph_info: Parsed graph information.

        Returns:
            List of detected Block instances.
        """
        blocks: list[Block] = []

        # Detect various patterns - ordered from specific to general
        blocks.extend(self.detect_conv_bn_relu(graph_info))
        blocks.extend(self.detect_residual_blocks(graph_info))
        blocks.extend(self.detect_nonstandard_residual_blocks(graph_info))

        # LLM-specific patterns (Task 5.4.1-5.4.7)
        blocks.extend(self.detect_attention_heads(graph_info))
        blocks.extend(self.detect_mlp_blocks(graph_info))
        blocks.extend(self.detect_embedding_layers(graph_info))
        blocks.extend(self.detect_position_encoding(graph_info))
        blocks.extend(self.detect_moe_routing(graph_info))

        # High-level transformer detection (uses sub-blocks)
        blocks.extend(self.detect_transformer_blocks(graph_info))

        # Detect repeated blocks
        repeated = self.detect_repeated_blocks(graph_info, blocks)
        if repeated:
            blocks.extend(repeated)

        self.logger.debug(f"Detected {len(blocks)} blocks")
        return blocks

    def detect_conv_bn_relu(self, graph_info: GraphInfo) -> list[Block]:
        """
        Find Conv-BatchNorm-ReLU sequences.

        Matches patterns like:
        - Conv -> BatchNorm -> ReLU
        - Conv -> ReLU
        - Conv -> BatchNorm
        """
        blocks: list[Block] = []
        visited: set[str] = set()

        for node in graph_info.nodes:
            if node.name in visited:
                continue

            if node.op_type == "Conv":
                block_nodes = [node.name]
                current_output = node.outputs[0] if node.outputs else None
                block_type_parts = ["Conv"]

                # Look for BatchNorm
                if current_output:
                    next_node = self._find_consumer(current_output, graph_info)
                    if next_node and next_node.op_type in self.NORM_OPS:
                        block_nodes.append(next_node.name)
                        block_type_parts.append("BN")
                        current_output = next_node.outputs[0] if next_node.outputs else None

                        # Look for activation after BN
                        if current_output:
                            act_node = self._find_consumer(current_output, graph_info)
                            if act_node and act_node.op_type in self.CONV_ACTIVATIONS:
                                block_nodes.append(act_node.name)
                                block_type_parts.append(act_node.op_type)
                    elif next_node and next_node.op_type in self.CONV_ACTIVATIONS:
                        # Conv -> ReLU without BN
                        block_nodes.append(next_node.name)
                        block_type_parts.append(next_node.op_type)

                if len(block_nodes) > 1:
                    visited.update(block_nodes)
                    block = Block(
                        block_type="".join(block_type_parts),
                        name=f"conv_block_{len(blocks)}",
                        nodes=block_nodes,
                        start_node=block_nodes[0],
                        end_node=block_nodes[-1],
                    )
                    blocks.append(block)

        return blocks

    def detect_residual_blocks(self, graph_info: GraphInfo) -> list[Block]:
        """
        Find residual/skip connection patterns.

        Looks for Add nodes where one input comes from earlier in the graph
        (skip connection).
        """
        blocks: list[Block] = []

        for node in graph_info.nodes:
            if node.op_type == "Add" and len(node.inputs) >= 2:
                # Check if this could be a residual connection
                # by looking for inputs that come from different depths
                input_nodes = []
                for inp in node.inputs:
                    if inp in graph_info.node_by_output:
                        input_nodes.append(graph_info.node_by_output[inp])

                if len(input_nodes) >= 2:
                    # Heuristic: if one path is longer (more hops), it's likely the residual path
                    # For now, just detect the pattern exists
                    blocks.append(
                        Block(
                            block_type="ResidualAdd",
                            name=f"residual_{len(blocks)}",
                            nodes=[node.name],
                            start_node=node.name,
                            end_node=node.name,
                            attributes={"inputs": node.inputs, "variant": "standard"},
                        )
                    )

        return blocks

    def detect_nonstandard_residual_blocks(self, graph_info: GraphInfo) -> list[Block]:
        """
        Find non-standard residual/skip connection patterns.

        Detects alternative skip connection implementations:
        - Concat-based skip connections (DenseNet-style)
        - Mul-based gating mechanisms (Highway networks, attention gates)
        - Sub-based connections (rare but possible)

        These may indicate custom architectures that need special handling.
        """
        blocks: list[Block] = []

        # Concat-based skip connections (DenseNet-style)
        for node in graph_info.nodes:
            if node.op_type == "Concat" and len(node.inputs) >= 2:
                # Check if inputs come from different depths (skip connection indicator)
                input_depths = self._estimate_input_depths(node.inputs, graph_info)
                if input_depths and max(input_depths) - min(input_depths) >= 2:
                    blocks.append(
                        Block(
                            block_type="ResidualConcat",
                            name=f"dense_skip_{len(blocks)}",
                            nodes=[node.name],
                            start_node=node.name,
                            end_node=node.name,
                            attributes={
                                "inputs": node.inputs,
                                "variant": "concat",
                                "depth_diff": max(input_depths) - min(input_depths),
                            },
                        )
                    )

        # Mul-based gating (Highway networks, attention gates)
        for node in graph_info.nodes:
            if node.op_type == "Mul" and len(node.inputs) >= 2:
                # Look for Sigmoid before Mul (gating pattern)
                has_sigmoid_input = False
                for inp in node.inputs:
                    if inp in graph_info.node_by_output:
                        prev_node = graph_info.node_by_output[inp]
                        if prev_node.op_type == "Sigmoid":
                            has_sigmoid_input = True
                            break

                if has_sigmoid_input:
                    blocks.append(
                        Block(
                            block_type="ResidualGate",
                            name=f"gate_{len(blocks)}",
                            nodes=[node.name],
                            start_node=node.name,
                            end_node=node.name,
                            attributes={"inputs": node.inputs, "variant": "gated"},
                        )
                    )

        # Sub-based connections (rare, but could be learned residual)
        for node in graph_info.nodes:
            if node.op_type == "Sub" and len(node.inputs) >= 2:
                input_nodes = []
                for inp in node.inputs:
                    if inp in graph_info.node_by_output:
                        input_nodes.append(graph_info.node_by_output[inp])

                if len(input_nodes) >= 2:
                    blocks.append(
                        Block(
                            block_type="ResidualSub",
                            name=f"sub_residual_{len(blocks)}",
                            nodes=[node.name],
                            start_node=node.name,
                            end_node=node.name,
                            attributes={"inputs": node.inputs, "variant": "subtract"},
                        )
                    )

        return blocks

    def _estimate_input_depths(
        self, inputs: list[str], graph_info: GraphInfo, max_depth: int = 20
    ) -> list[int]:
        """
        Estimate the graph depth of each input tensor.

        Returns a list of estimated depths (hops from graph inputs).
        Used to detect skip connections where inputs come from very different depths.
        """
        depths = []
        for inp in inputs:
            depth = self._trace_depth(inp, graph_info, 0, max_depth)
            depths.append(depth)
        return depths

    def _trace_depth(
        self,
        tensor_name: str,
        graph_info: GraphInfo,
        current_depth: int,
        max_depth: int,
    ) -> int:
        """Recursively trace back to find the depth of a tensor."""
        if current_depth >= max_depth:
            return current_depth

        # If it's a graph input or initializer, depth is 0
        if tensor_name in graph_info.input_shapes:
            return 0
        if tensor_name in graph_info.initializers:
            return 0

        # Find the node that produces this tensor
        if tensor_name in graph_info.node_by_output:
            producer = graph_info.node_by_output[tensor_name]
            if producer.inputs:
                # Trace back through the first input
                return 1 + self._trace_depth(
                    producer.inputs[0], graph_info, current_depth + 1, max_depth
                )

        return current_depth

    def detect_transformer_blocks(self, graph_info: GraphInfo) -> list[Block]:
        """
        Find transformer attention patterns.

        Looks for the characteristic Softmax in attention computation
        and MatMul patterns for Q, K, V projections.
        """
        blocks: list[Block] = []
        softmax_nodes = [n for n in graph_info.nodes if n.op_type == "Softmax"]

        for softmax in softmax_nodes:
            # Look for attention pattern: MatMul -> Softmax -> MatMul
            before_nodes: list[str] = []
            after_nodes: list[str] = []

            # Find MatMul before softmax
            if softmax.inputs:
                inp = softmax.inputs[0]
                if inp in graph_info.node_by_output:
                    prev = graph_info.node_by_output[inp]
                    if prev.op_type in (
                        "MatMul",
                        "Gemm",
                        "Div",
                        "Mul",
                    ):  # Div for scaling
                        before_nodes.append(prev.name)

            # Find MatMul after softmax
            if softmax.outputs:
                consumer = self._find_consumer(softmax.outputs[0], graph_info)
                if consumer and consumer.op_type in ("MatMul", "Gemm"):
                    after_nodes.append(consumer.name)

            if before_nodes and after_nodes:
                all_nodes = [*before_nodes, softmax.name, *after_nodes]
                blocks.append(
                    Block(
                        block_type="Attention",
                        name=f"attention_{len(blocks)}",
                        nodes=all_nodes,
                        start_node=before_nodes[0],
                        end_node=after_nodes[-1],
                    )
                )

        # Also look for LayerNorm which often brackets transformer blocks
        layernorm_count = sum(1 for n in graph_info.nodes if n.op_type == "LayerNormalization")
        if layernorm_count >= 2 and blocks:
            # Likely a transformer architecture
            self.logger.debug(
                f"Found {len(blocks)} attention blocks with {layernorm_count} LayerNorms"
            )

        return blocks

    def detect_embedding_layers(self, graph_info: GraphInfo) -> list[Block]:
        """
        Find embedding lookup patterns.

        Looks for Gather operations on large weight tensors.
        """
        blocks: list[Block] = []

        for node in graph_info.nodes:
            if node.op_type == "Gather":
                # Check if first input is a large initializer (embedding table)
                if node.inputs and node.inputs[0] in graph_info.initializers:
                    embed_table = graph_info.initializers[node.inputs[0]]
                    if len(embed_table.shape) == 2:
                        vocab_size, embed_dim = embed_table.shape
                        # Token embedding typically has large vocab (>1000)
                        embed_type = "token" if vocab_size > 1000 else "position"
                        blocks.append(
                            Block(
                                block_type="Embedding",
                                name=f"embedding_{len(blocks)}",
                                nodes=[node.name],
                                start_node=node.name,
                                end_node=node.name,
                                attributes={
                                    "vocab_size": int(vocab_size),
                                    "embed_dim": int(embed_dim),
                                    "embed_type": embed_type,
                                },
                            )
                        )

        return blocks

    def detect_attention_heads(self, graph_info: GraphInfo) -> list[Block]:
        """
        Detect attention head patterns with Q/K/V projections.

        Task 5.4.1: Enhanced attention detection for LLMs.

        Patterns detected:
        - Standard MHA: Q, K, V linear -> attention -> output linear
        - MQA: Single K, V shared across Q heads
        - GQA: Grouped K, V (fewer KV heads than Q heads)
        """
        blocks: list[Block] = []
        visited_softmax: set[str] = set()

        # Find all Softmax nodes (attention core)
        softmax_nodes = [n for n in graph_info.nodes if n.op_type == "Softmax"]

        for softmax in softmax_nodes:
            if softmax.name in visited_softmax:
                continue

            attention_info = self._trace_attention_pattern(softmax, graph_info)
            if attention_info:
                visited_softmax.add(softmax.name)

                # Determine attention type
                num_q_heads = attention_info.get("num_q_heads", 0)
                num_kv_heads = attention_info.get("num_kv_heads", 0)

                if num_kv_heads == 1:
                    attention_type = "MQA"  # Multi-Query Attention
                elif num_kv_heads > 0 and num_kv_heads < num_q_heads:
                    attention_type = "GQA"  # Grouped-Query Attention
                else:
                    attention_type = "MHA"  # Standard Multi-Head Attention

                blocks.append(
                    Block(
                        block_type="AttentionHead",
                        name=f"attention_head_{len(blocks)}",
                        nodes=attention_info.get("nodes", [softmax.name]),
                        start_node=attention_info.get("q_proj", softmax.name),
                        end_node=attention_info.get("o_proj", softmax.name),
                        attributes={
                            "attention_type": attention_type,
                            "num_q_heads": num_q_heads,
                            "num_kv_heads": num_kv_heads,
                            "has_scaling": attention_info.get("has_scaling", False),
                            "has_mask": attention_info.get("has_mask", False),
                            "q_proj": attention_info.get("q_proj"),
                            "k_proj": attention_info.get("k_proj"),
                            "v_proj": attention_info.get("v_proj"),
                            "o_proj": attention_info.get("o_proj"),
                        },
                    )
                )

        return blocks

    def _trace_attention_pattern(
        self, softmax: NodeInfo, graph_info: GraphInfo
    ) -> dict[str, Any] | None:
        """Trace back from Softmax to find Q/K/V projections."""
        result: dict[str, Any] = {
            "nodes": [softmax.name],
            "has_scaling": False,
            "has_mask": False,
            "num_q_heads": 0,
            "num_kv_heads": 0,
        }

        # Trace backward from softmax
        # Pattern: (Q @ K^T) / sqrt(d_k) -> Softmax -> @ V

        if not softmax.inputs:
            return None

        # Look for scaling (Div or Mul) and mask (Add) before softmax
        current = softmax.inputs[0]
        if current in graph_info.node_by_output:
            prev = graph_info.node_by_output[current]

            # Check for mask addition
            if prev.op_type == "Add":
                result["has_mask"] = True
                result["nodes"].append(prev.name)
                if prev.inputs:
                    current = prev.inputs[0]
                    if current in graph_info.node_by_output:
                        prev = graph_info.node_by_output[current]

            # Check for scaling
            if prev.op_type in ("Div", "Mul"):
                result["has_scaling"] = True
                result["nodes"].append(prev.name)
                if prev.inputs:
                    current = prev.inputs[0]
                    if current in graph_info.node_by_output:
                        prev = graph_info.node_by_output[current]

            # Look for Q @ K^T (MatMul)
            if prev.op_type == "MatMul":
                result["nodes"].append(prev.name)

                # Trace Q and K projections
                if len(prev.inputs) >= 2:
                    q_input, k_input = prev.inputs[0], prev.inputs[1]

                    # Find Q projection
                    q_proj = self._find_linear_proj(q_input, graph_info)
                    if q_proj:
                        result["q_proj"] = q_proj["name"]
                        result["nodes"].append(q_proj["name"])
                        result["num_q_heads"] = q_proj.get("num_heads", 0)

                    # Find K projection (may go through Transpose)
                    k_proj = self._find_linear_proj(k_input, graph_info, through_transpose=True)
                    if k_proj:
                        result["k_proj"] = k_proj["name"]
                        result["nodes"].append(k_proj["name"])
                        result["num_kv_heads"] = k_proj.get("num_heads", 0)

        # Trace forward from softmax to find V @ attention and output projection
        if softmax.outputs:
            consumer = self._find_consumer(softmax.outputs[0], graph_info)
            if consumer and consumer.op_type == "MatMul":
                result["nodes"].append(consumer.name)

                # Find V projection
                if len(consumer.inputs) >= 2:
                    v_input = consumer.inputs[1]  # Second input is V
                    v_proj = self._find_linear_proj(v_input, graph_info)
                    if v_proj:
                        result["v_proj"] = v_proj["name"]
                        result["nodes"].append(v_proj["name"])

                # Find output projection
                if consumer.outputs:
                    o_consumer = self._find_consumer(consumer.outputs[0], graph_info)
                    if o_consumer and o_consumer.op_type in ("MatMul", "Gemm"):
                        result["o_proj"] = o_consumer.name
                        result["nodes"].append(o_consumer.name)

        # Only return if we found at least the core attention pattern
        if len(result["nodes"]) >= 3:  # softmax + at least 2 matmuls
            return result
        return None

    def _find_linear_proj(
        self, tensor_name: str, graph_info: GraphInfo, through_transpose: bool = False
    ) -> dict | None:
        """Find a linear projection (MatMul/Gemm) producing this tensor."""
        current = tensor_name

        # Optionally look through Transpose (for K^T in attention)
        if through_transpose and current in graph_info.node_by_output:
            node = graph_info.node_by_output[current]
            if node.op_type == "Transpose":
                if node.inputs:
                    current = node.inputs[0]

        # Also look through Reshape (for multi-head splitting)
        if current in graph_info.node_by_output:
            node = graph_info.node_by_output[current]
            if node.op_type == "Reshape":
                if node.inputs:
                    current = node.inputs[0]

        # Find the MatMul/Gemm
        if current in graph_info.node_by_output:
            node = graph_info.node_by_output[current]
            if node.op_type in ("MatMul", "Gemm"):
                # Try to infer number of heads from weight shape
                num_heads = 0
                if len(node.inputs) >= 2:
                    weight_name = node.inputs[1]
                    if weight_name in graph_info.initializers:
                        weight = graph_info.initializers[weight_name]
                        if len(weight.shape) == 2:
                            # Typical shape: [hidden, num_heads * head_dim]
                            out_features = weight.shape[1]
                            # Common head_dim values: 64, 128
                            for head_dim in [64, 128, 96, 80]:
                                if out_features % head_dim == 0:
                                    num_heads = out_features // head_dim
                                    break

                return {"name": node.name, "num_heads": num_heads}

        return None

    def detect_mlp_blocks(self, graph_info: GraphInfo) -> list[Block]:
        """
        Detect MLP/FFN blocks in transformers.

        Task 5.4.2: Detect MLP/FFN patterns.

        Patterns detected:
        - Standard FFN: Linear -> Activation -> Linear
        - SwiGLU/GeGLU: Linear -> (Gate * Activation(Linear)) -> Linear
        - Gated MLP: Uses element-wise gating
        """
        blocks: list[Block] = []
        visited: set[str] = set()

        # Look for activation functions that typically appear in FFN
        for node in graph_info.nodes:
            if node.name in visited:
                continue

            if node.op_type in self.LLM_ACTIVATIONS or node.op_type == "Gelu":
                mlp_info = self._trace_mlp_pattern(node, graph_info)
                if mlp_info:
                    visited.update(mlp_info["nodes"])

                    blocks.append(
                        Block(
                            block_type="MLPBlock",
                            name=f"mlp_{len(blocks)}",
                            nodes=mlp_info["nodes"],
                            start_node=mlp_info["up_proj"],
                            end_node=mlp_info["down_proj"],
                            attributes={
                                "mlp_type": mlp_info["mlp_type"],
                                "hidden_dim": mlp_info.get("hidden_dim", 0),
                                "intermediate_dim": mlp_info.get("intermediate_dim", 0),
                                "activation": node.op_type,
                                "is_gated": mlp_info.get("is_gated", False),
                            },
                        )
                    )

        return blocks

    def _trace_mlp_pattern(
        self, activation: NodeInfo, graph_info: GraphInfo
    ) -> dict[str, Any] | None:
        """Trace MLP pattern from activation function."""
        result: dict[str, Any] = {
            "nodes": [activation.name],
            "mlp_type": "standard",
            "is_gated": False,
        }

        # Trace backward to find up-projection (first linear)
        if activation.inputs:
            inp = activation.inputs[0]
            if inp in graph_info.node_by_output:
                prev = graph_info.node_by_output[inp]
                if prev.op_type in ("MatMul", "Gemm"):
                    result["up_proj"] = prev.name
                    result["nodes"].append(prev.name)

                    # Get dimensions from weight
                    if len(prev.inputs) >= 2:
                        weight_name = prev.inputs[1]
                        if weight_name in graph_info.initializers:
                            weight = graph_info.initializers[weight_name]
                            if len(weight.shape) == 2:
                                result["hidden_dim"] = int(weight.shape[0])
                                result["intermediate_dim"] = int(weight.shape[1])

        # Trace forward to find down-projection
        # Handle gated patterns (SwiGLU): activation output may go through Mul
        if activation.outputs:
            consumer = self._find_consumer(activation.outputs[0], graph_info)
            if consumer:
                if consumer.op_type == "Mul":
                    # Gated pattern (SwiGLU/GeGLU)
                    result["is_gated"] = True
                    result["mlp_type"] = "gated"
                    result["nodes"].append(consumer.name)

                    if consumer.outputs:
                        down_proj = self._find_consumer(consumer.outputs[0], graph_info)
                        if down_proj and down_proj.op_type in ("MatMul", "Gemm"):
                            result["down_proj"] = down_proj.name
                            result["nodes"].append(down_proj.name)
                            return result

                elif consumer.op_type in ("MatMul", "Gemm"):
                    # Standard FFN
                    result["down_proj"] = consumer.name
                    result["nodes"].append(consumer.name)
                    return result

        # Only return if we found both projections
        if "up_proj" in result and "down_proj" in result:
            return result
        return None

    def detect_position_encoding(self, graph_info: GraphInfo) -> list[Block]:
        """
        Detect position encoding patterns.

        Task 5.4.3: Detect embedding patterns (RoPE, ALiBi, learned, sinusoidal).

        Patterns:
        - RoPE: Complex rotations using Sin/Cos on positions
        - ALiBi: Learned linear biases added to attention
        - Learned: Position embedding table (Gather)
        - Sinusoidal: Sin/Cos computations on positions
        """
        blocks: list[Block] = []

        # Check for RoPE pattern (Sin/Cos operations followed by Mul)
        sin_nodes = [n for n in graph_info.nodes if n.op_type == "Sin"]
        cos_nodes = [n for n in graph_info.nodes if n.op_type == "Cos"]

        if sin_nodes and cos_nodes:
            # Likely RoPE or sinusoidal encoding
            # RoPE typically has paired Sin/Cos that multiply with Q and K
            rope_nodes = []
            for sin in sin_nodes:
                rope_nodes.append(sin.name)
            for cos in cos_nodes:
                rope_nodes.append(cos.name)

            # Check if these feed into Mul operations (rotation pattern)
            has_rotation = False
            for node in graph_info.nodes:
                if node.op_type == "Mul":
                    for inp in node.inputs:
                        if inp in graph_info.node_by_output:
                            prev = graph_info.node_by_output[inp]
                            if prev.op_type in ("Sin", "Cos"):
                                has_rotation = True
                                break

            if has_rotation:
                blocks.append(
                    Block(
                        block_type="PositionEncoding",
                        name="rope_encoding",
                        nodes=rope_nodes,
                        start_node=rope_nodes[0] if rope_nodes else "",
                        end_node=rope_nodes[-1] if rope_nodes else "",
                        attributes={
                            "encoding_type": "RoPE",
                            "num_sin": len(sin_nodes),
                            "num_cos": len(cos_nodes),
                        },
                    )
                )
            else:
                blocks.append(
                    Block(
                        block_type="PositionEncoding",
                        name="sinusoidal_encoding",
                        nodes=rope_nodes,
                        start_node=rope_nodes[0] if rope_nodes else "",
                        end_node=rope_nodes[-1] if rope_nodes else "",
                        attributes={
                            "encoding_type": "sinusoidal",
                        },
                    )
                )

        # Check for learned position embeddings (small Gather, separate from token)
        for node in graph_info.nodes:
            if node.op_type == "Gather":
                if node.inputs and node.inputs[0] in graph_info.initializers:
                    table = graph_info.initializers[node.inputs[0]]
                    if len(table.shape) == 2:
                        size, dim = table.shape
                        # Position embeddings are typically smaller than vocab
                        # (e.g., 512, 1024, 2048, 4096 max positions)
                        if 128 <= size <= 8192 and size not in [
                            30000,
                            32000,
                            50257,
                            50304,
                            65536,
                            128000,
                            151936,
                        ]:
                            # Likely position embedding, not token embedding
                            blocks.append(
                                Block(
                                    block_type="PositionEncoding",
                                    name=f"learned_position_{len(blocks)}",
                                    nodes=[node.name],
                                    start_node=node.name,
                                    end_node=node.name,
                                    attributes={
                                        "encoding_type": "learned",
                                        "max_positions": int(size),
                                        "embed_dim": int(dim),
                                    },
                                )
                            )

        return blocks

    def detect_moe_routing(self, graph_info: GraphInfo) -> list[Block]:
        """
        Detect Mixture of Experts (MoE) routing patterns.

        Task 5.4.7: Handle MoE routing patterns.

        MoE pattern:
        - Router: Linear -> Softmax/TopK -> expert selection
        - Experts: Multiple parallel FFN blocks
        - Combine: Scatter/Gather to route tokens to experts
        """
        blocks: list[Block] = []

        # Look for TopK operations (expert selection)
        topk_nodes = [n for n in graph_info.nodes if n.op_type == "TopK"]

        for topk in topk_nodes:
            # Check if this looks like MoE routing
            # Pattern: Linear -> TopK (for top-k expert selection)
            router_proj = None
            if topk.inputs:
                inp = topk.inputs[0]
                # May go through Softmax first
                if inp in graph_info.node_by_output:
                    prev = graph_info.node_by_output[inp]
                    if prev.op_type == "Softmax":
                        if prev.inputs:
                            inp = prev.inputs[0]
                    if inp in graph_info.node_by_output:
                        router_node = graph_info.node_by_output[inp]
                        if router_node.op_type in ("MatMul", "Gemm"):
                            router_proj = router_node.name

            if router_proj:
                # Try to infer number of experts from router output shape
                num_experts = 0
                k_value = 0

                # Check TopK k attribute
                for attr in (
                    getattr(topk, "attributes", {}).items() if hasattr(topk, "attributes") else []
                ):
                    if attr[0] == "k":
                        k_value = attr[1]

                blocks.append(
                    Block(
                        block_type="MoERouter",
                        name=f"moe_router_{len(blocks)}",
                        nodes=[router_proj, topk.name],
                        start_node=router_proj,
                        end_node=topk.name,
                        attributes={
                            "num_experts": num_experts,
                            "top_k": k_value,
                            "router_type": "top_k",
                        },
                    )
                )

        return blocks

    def detect_repeated_blocks(
        self, graph_info: GraphInfo, existing_blocks: list[Block]
    ) -> list[Block]:
        """
        Detect repeated identical blocks (e.g., N transformer layers).

        Task 5.4.5: Detect repetition - N identical blocks -> collapse with xN.
        """
        blocks: list[Block] = []

        # Group existing blocks by type
        blocks_by_type: dict[str, list[Block]] = {}
        for block in existing_blocks:
            if block.block_type not in blocks_by_type:
                blocks_by_type[block.block_type] = []
            blocks_by_type[block.block_type].append(block)

        # Check for repeated patterns
        for block_type, type_blocks in blocks_by_type.items():
            if len(type_blocks) >= 4:  # At least 4 repetitions to be significant
                # Check if blocks have similar structure
                # (simplified: just count them for now)
                blocks.append(
                    Block(
                        block_type="RepeatedBlock",
                        name=f"repeated_{block_type}",
                        nodes=[],  # Meta-block, doesn't own nodes directly
                        start_node=type_blocks[0].start_node,
                        end_node=type_blocks[-1].end_node,
                        attributes={
                            "repeated_type": block_type,
                            "num_repetitions": len(type_blocks),
                            "block_names": [b.name for b in type_blocks],
                        },
                    )
                )

        return blocks

    def detect_normalization_pattern(self, graph_info: GraphInfo) -> dict:
        """
        Detect normalization placement pattern (pre-norm vs post-norm).

        Task 5.4.4: Detect normalization placement.

        Pre-norm (modern, e.g., LLaMA, GPT-3):
            LayerNorm -> Attention -> Residual Add
            LayerNorm -> FFN -> Residual Add

        Post-norm (original transformer, e.g., BERT):
            Attention -> Residual Add -> LayerNorm
            FFN -> Residual Add -> LayerNorm
        """
        result = {
            "pattern": "unknown",
            "num_layernorms": 0,
            "has_rmsnorm": False,
        }

        # Count normalization ops
        ln_count = graph_info.op_type_counts.get("LayerNormalization", 0)
        rms_count = sum(
            1 for n in graph_info.nodes if "rms" in n.name.lower() or "rmsnorm" in n.name.lower()
        )
        result["num_layernorms"] = ln_count
        result["has_rmsnorm"] = rms_count > 0 or any(
            n.op_type == "SimplifiedLayerNormalization" for n in graph_info.nodes
        )

        if ln_count == 0 and not result["has_rmsnorm"]:
            result["pattern"] = "none"
            return result

        # Analyze pattern by checking what comes after Add (residual)
        add_nodes = [n for n in graph_info.nodes if n.op_type == "Add"]

        ln_after_add = 0
        ln_before_matmul = 0

        for add in add_nodes:
            if add.outputs:
                consumer = self._find_consumer(add.outputs[0], graph_info)
                if consumer and consumer.op_type in (
                    "LayerNormalization",
                    "SimplifiedLayerNormalization",
                ):
                    ln_after_add += 1

        # Check if LayerNorm feeds into MatMul (pre-norm pattern)
        for node in graph_info.nodes:
            if node.op_type in ("LayerNormalization", "SimplifiedLayerNormalization"):
                if node.outputs:
                    consumer = self._find_consumer(node.outputs[0], graph_info)
                    if consumer and consumer.op_type in ("MatMul", "Gemm"):
                        ln_before_matmul += 1

        # Classify
        if ln_after_add > ln_before_matmul:
            result["pattern"] = "post_norm"
        elif ln_before_matmul > ln_after_add:
            result["pattern"] = "pre_norm"
        elif ln_count > 0 or result["has_rmsnorm"]:
            result["pattern"] = "mixed"

        return result

    def classify_architecture(self, graph_info: GraphInfo, blocks: list[Block]) -> str:
        """
        Classify the overall architecture type.

        Args:
            graph_info: Parsed graph information.
            blocks: Detected blocks from group_into_blocks().

        Returns:
            Architecture type: "transformer", "cnn", "mlp", "hybrid", "unknown"
        """
        op_counts = graph_info.op_type_counts
        block_types = [b.block_type for b in blocks]

        # Count key indicators
        has_attention = any("Attention" in bt for bt in block_types)
        has_mlp_block = any("MLPBlock" in bt for bt in block_types)
        has_layernorm = op_counts.get("LayerNormalization", 0) > 0
        has_embedding = any("Embedding" in bt for bt in block_types)
        has_moe = any("MoE" in bt for bt in block_types)
        has_rope = any(
            b.block_type == "PositionEncoding" and b.attributes.get("encoding_type") == "RoPE"
            for b in blocks
        )

        # Include quantized variants (ConvInteger, MatMulInteger) for INT8 models
        conv_count = op_counts.get("Conv", 0) + op_counts.get("ConvInteger", 0)
        matmul_count = (
            op_counts.get("MatMul", 0)
            + op_counts.get("Gemm", 0)
            + op_counts.get("MatMulInteger", 0)
        )
        softmax_count = op_counts.get("Softmax", 0)

        # Classification heuristics - more specific
        if has_moe:
            return "moe_transformer"
        elif has_attention or has_mlp_block or (has_layernorm and softmax_count >= 2):
            if has_rope:
                return "decoder_transformer"  # LLaMA-style
            elif has_embedding:
                return "transformer"
            else:
                return "transformer"
        elif conv_count > matmul_count and conv_count >= 5:
            return "cnn"
        elif conv_count > 0 and (has_attention or has_layernorm):
            return "hybrid"
        elif matmul_count > 0:
            return "mlp"
        else:
            return "unknown"

    def get_architecture_summary(self, graph_info: GraphInfo, blocks: list[Block]) -> dict:
        """
        Get a detailed architecture summary for LLMs.

        Returns comprehensive architecture info for reports.
        """
        arch_type = self.classify_architecture(graph_info, blocks)
        norm_pattern = self.detect_normalization_pattern(graph_info)

        # Count block types
        block_counts: dict[str, int] = {}
        for block in blocks:
            bt = block.block_type
            block_counts[bt] = block_counts.get(bt, 0) + 1

        # Get attention info
        attention_blocks = [b for b in blocks if "Attention" in b.block_type]
        attention_type = "unknown"
        num_heads = 0
        num_kv_heads = 0

        if attention_blocks:
            # Use first attention block's info
            first_attn = attention_blocks[0]
            attention_type = first_attn.attributes.get("attention_type", "unknown")
            num_heads = first_attn.attributes.get("num_q_heads", 0)
            num_kv_heads = first_attn.attributes.get("num_kv_heads", 0)

        # Get MLP info
        mlp_blocks = [b for b in blocks if b.block_type == "MLPBlock"]
        mlp_type = "unknown"
        if mlp_blocks:
            mlp_type = mlp_blocks[0].attributes.get("mlp_type", "unknown")

        # Get position encoding
        pos_blocks = [b for b in blocks if b.block_type == "PositionEncoding"]
        pos_encoding = "none"
        if pos_blocks:
            pos_encoding = pos_blocks[0].attributes.get("encoding_type", "unknown")

        # Get repetition info
        repeated_blocks = [b for b in blocks if b.block_type == "RepeatedBlock"]
        num_layers = 0
        for rb in repeated_blocks:
            if rb.attributes.get("repeated_type") in ("AttentionHead", "Attention"):
                num_layers = rb.attributes.get("num_repetitions", 0)
                break

        return {
            "architecture_type": arch_type,
            "normalization": norm_pattern,
            "block_counts": block_counts,
            "attention": {
                "type": attention_type,
                "num_q_heads": num_heads,
                "num_kv_heads": num_kv_heads,
            },
            "mlp": {
                "type": mlp_type,
            },
            "position_encoding": pos_encoding,
            "num_layers": num_layers,
            "total_blocks": len(blocks),
        }

    def _find_consumer(self, output_name: str, graph_info: GraphInfo) -> NodeInfo | None:
        """Find the first node that consumes a given output."""
        for node in graph_info.nodes:
            if output_name in node.inputs:
                return node
        return None
