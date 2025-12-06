# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
Risk analysis for HaoLine.

Applies heuristics to detect potentially problematic patterns:
- Deep networks without skip connections
- Oversized dense layers
- Dynamic shapes that may cause issues
- Missing normalization
- Unusual activation patterns
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from .analyzer import GraphInfo
    from .patterns import Block


class RiskSignal(BaseModel):
    """
    A detected risk or concern about the model architecture.

    Risk signals are informational - they highlight patterns that
    may cause issues but don't necessarily indicate problems.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str  # e.g., "no_skip_connections", "oversized_dense"
    severity: str  # "info" | "warning" | "high"
    description: str
    nodes: list[str] = Field(default_factory=list)
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "severity": self.severity,
            "description": self.description,
            "nodes": self.nodes,
            "recommendation": self.recommendation,
        }


class RiskThresholds(BaseModel):
    """
    Configurable thresholds for risk detection.

    Allows tuning sensitivity based on model type and use case.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Minimum thresholds - don't bother analyzing tiny models
    min_params_for_analysis: int = 100_000  # 100K params minimum
    min_flops_for_bottleneck: int = 1_000_000_000  # 1B FLOPs before flagging bottlenecks
    min_nodes_for_depth_check: int = 20  # At least 20 nodes before checking depth

    # Thresholds for risk detection
    deep_network_threshold: int = 50  # nodes before considering "deep"
    oversized_dense_threshold: int = 100_000_000  # 100M params in single layer
    large_embedding_threshold: int = 500_000_000  # 500M params for embedding
    high_flop_ratio_threshold: float = 0.5  # Single op using >50% of FLOPs

    # Minimum trainable layers before flagging missing normalization/activations
    min_trainable_for_norm_check: int = 10
    min_trainable_for_activation_check: int = 5


class RiskAnalyzer:
    """
    Detect architectural risk signals in ONNX graphs.

    Applies various heuristics to identify patterns that may
    cause training, inference, or deployment issues.

    Note: Risk signals are only generated for models above minimum
    complexity thresholds to avoid flagging trivial test models.

    Thresholds can be configured via the `thresholds` parameter.
    """

    # Default thresholds (class-level for backward compatibility)
    MIN_PARAMS_FOR_ANALYSIS = 100_000
    MIN_FLOPS_FOR_BOTTLENECK = 1_000_000_000
    MIN_NODES_FOR_DEPTH_CHECK = 20
    DEEP_NETWORK_THRESHOLD = 50
    OVERSIZED_DENSE_THRESHOLD = 100_000_000
    LARGE_EMBEDDING_THRESHOLD = 500_000_000
    HIGH_FLOP_RATIO_THRESHOLD = 0.5

    def __init__(
        self,
        logger: logging.Logger | None = None,
        thresholds: RiskThresholds | None = None,
    ):
        self.logger = logger or logging.getLogger("haoline.risks")
        self.thresholds = thresholds or RiskThresholds()

        # Also update class-level constants for backward compatibility
        if thresholds:
            self.MIN_PARAMS_FOR_ANALYSIS = thresholds.min_params_for_analysis
            self.MIN_FLOPS_FOR_BOTTLENECK = thresholds.min_flops_for_bottleneck
            self.MIN_NODES_FOR_DEPTH_CHECK = thresholds.min_nodes_for_depth_check
            self.DEEP_NETWORK_THRESHOLD = thresholds.deep_network_threshold
            self.OVERSIZED_DENSE_THRESHOLD = thresholds.oversized_dense_threshold
            self.LARGE_EMBEDDING_THRESHOLD = thresholds.large_embedding_threshold
            self.HIGH_FLOP_RATIO_THRESHOLD = thresholds.high_flop_ratio_threshold

    def analyze(self, graph_info: GraphInfo, blocks: list[Block]) -> list[RiskSignal]:
        """
        Run all risk heuristics and return detected signals.

        Args:
            graph_info: Parsed graph information.
            blocks: Detected architectural blocks.

        Returns:
            List of RiskSignal instances.
        """
        signals = []

        # Run all checks
        signal = self.check_deep_without_skips(graph_info, blocks)
        if signal:
            signals.append(signal)

        signal = self.check_oversized_dense(graph_info)
        if signal:
            signals.append(signal)

        signal = self.check_dynamic_shapes(graph_info)
        if signal:
            signals.append(signal)

        signal = self.check_missing_normalization(graph_info, blocks)
        if signal:
            signals.append(signal)

        signal = self.check_compute_bottleneck(graph_info)
        if signal:
            signals.append(signal)

        signal = self.check_large_embedding(graph_info, blocks)
        if signal:
            signals.append(signal)

        signal = self.check_unusual_activations(graph_info)
        if signal:
            signals.append(signal)

        signal = self.check_nonstandard_residuals(graph_info, blocks)
        if signal:
            signals.append(signal)

        self.logger.debug(f"Detected {len(signals)} risk signals")
        return signals

    def check_deep_without_skips(
        self, graph_info: GraphInfo, blocks: list[Block]
    ) -> RiskSignal | None:
        """
        Flag deep networks that lack skip connections.

        Deep networks without residual connections may suffer from
        vanishing gradients during training.
        """
        # Skip very small models - they don't need skip connections
        if graph_info.num_nodes < self.MIN_NODES_FOR_DEPTH_CHECK:
            return None

        if graph_info.num_nodes < self.DEEP_NETWORK_THRESHOLD:
            return None

        # Count residual blocks
        residual_count = sum(1 for b in blocks if "Residual" in b.block_type)

        if residual_count == 0:
            return RiskSignal(
                id="no_skip_connections",
                severity="warning",
                description=(
                    f"Model has {graph_info.num_nodes} nodes but no detected skip connections. "
                    "Deep networks without residual connections may have training difficulties."
                ),
                nodes=[],
                recommendation=(
                    "Consider adding skip/residual connections if this model will be trained. "
                    "If this is a pre-trained inference model, this may not be a concern."
                ),
            )

        return None

    def check_oversized_dense(self, graph_info: GraphInfo) -> RiskSignal | None:
        """
        Flag excessively large fully-connected layers.

        Very large MatMul/Gemm operations can dominate compute and memory.
        """
        large_ops = []

        for node in graph_info.nodes:
            if node.op_type in ("MatMul", "Gemm"):
                # Check weight size
                for inp in node.inputs:
                    if inp in graph_info.initializers:
                        weight = graph_info.initializers[inp]
                        param_count = int(weight.size) if hasattr(weight, "size") else 0
                        if param_count > self.OVERSIZED_DENSE_THRESHOLD:
                            large_ops.append((node.name, param_count))
                        break

        if large_ops:
            total_large = sum(p for _, p in large_ops)
            return RiskSignal(
                id="oversized_dense",
                severity="info",
                description=(
                    f"Found {len(large_ops)} dense layer(s) with >100M parameters "
                    f"(total: {total_large:,} params). These may dominate compute and memory."
                ),
                nodes=[name for name, _ in large_ops],
                recommendation=(
                    "Consider whether these large layers are necessary. "
                    "Techniques like low-rank factorization or pruning may help reduce size."
                ),
            )

        return None

    def check_dynamic_shapes(self, graph_info: GraphInfo) -> RiskSignal | None:
        """
        Flag inputs with dynamic shapes.

        Dynamic shapes can cause issues with some inference backends
        and prevent certain optimizations.
        """
        dynamic_inputs = []

        for name, shape in graph_info.input_shapes.items():
            has_dynamic = any(not isinstance(d, int) for d in shape)
            if has_dynamic:
                dynamic_inputs.append(name)

        if dynamic_inputs:
            return RiskSignal(
                id="dynamic_input_shapes",
                severity="info",
                description=(
                    f"Model has {len(dynamic_inputs)} input(s) with dynamic/symbolic dimensions: "
                    f"{', '.join(dynamic_inputs)}. "
                    "This is normal for variable-length sequences but may affect optimization."
                ),
                nodes=[],
                recommendation=(
                    "For best performance with hardware accelerators, consider providing "
                    "fixed shapes or using onnxruntime.tools.make_dynamic_shape_fixed."
                ),
            )

        return None

    def check_missing_normalization(
        self, graph_info: GraphInfo, blocks: list[Block]
    ) -> RiskSignal | None:
        """
        Flag deep networks without normalization layers.

        Networks without BatchNorm/LayerNorm may have training instabilities.
        """
        # Skip small models
        if graph_info.num_nodes < self.MIN_NODES_FOR_DEPTH_CHECK:
            return None

        norm_ops = {
            "BatchNormalization",
            "LayerNormalization",
            "InstanceNormalization",
            "GroupNormalization",
        }
        has_norm = any(op in graph_info.op_type_counts for op in norm_ops)

        # Count trainable layers (Conv, MatMul, Gemm)
        trainable_count = (
            graph_info.op_type_counts.get("Conv", 0)
            + graph_info.op_type_counts.get("MatMul", 0)
            + graph_info.op_type_counts.get("Gemm", 0)
        )

        # Need at least N trainable layers to care about normalization
        min_trainable = self.thresholds.min_trainable_for_norm_check
        if not has_norm and trainable_count >= min_trainable:
            return RiskSignal(
                id="missing_normalization",
                severity="info",
                description=(
                    f"Model has {trainable_count} trainable layers but no normalization layers detected. "
                    "This may affect training stability."
                ),
                nodes=[],
                recommendation=(
                    "If this model will be fine-tuned, consider adding normalization layers. "
                    "For inference-only, this is typically not a concern."
                ),
            )

        return None

    def check_compute_bottleneck(self, graph_info: GraphInfo) -> RiskSignal | None:
        """
        Flag single operations that dominate compute.

        If one layer uses >50% of FLOPs, it's a potential bottleneck.
        Only flags models with significant compute (>1B FLOPs) to avoid
        noise on trivial models.
        """
        # Need to compute per-node FLOPs
        total_flops = sum(node.flops for node in graph_info.nodes)

        # Skip tiny models - no point optimizing a model with < 1B FLOPs
        if total_flops < self.MIN_FLOPS_FOR_BOTTLENECK:
            return None

        bottlenecks = []
        for node in graph_info.nodes:
            if node.flops > 0:
                ratio = node.flops / total_flops
                if ratio > self.HIGH_FLOP_RATIO_THRESHOLD:
                    bottlenecks.append((node.name, node.op_type, ratio))

        if bottlenecks:
            desc_parts = [f"{name} ({op}: {ratio:.1%})" for name, op, ratio in bottlenecks]
            total_gflops = total_flops / 1e9
            return RiskSignal(
                id="compute_bottleneck",
                severity="info",
                description=(
                    f"The following operations dominate compute ({total_gflops:.1f} GFLOPs total): "
                    f"{', '.join(desc_parts)}. Optimizing these would have the greatest impact."
                ),
                nodes=[name for name, _, _ in bottlenecks],
                recommendation="Focus optimization efforts (quantization, pruning) on these layers.",
            )

        return None

    def check_large_embedding(
        self, graph_info: GraphInfo, blocks: list[Block]
    ) -> RiskSignal | None:
        """
        Flag very large embedding tables.

        Large vocabulary embeddings can dominate model size.
        """
        embedding_blocks = [b for b in blocks if b.block_type == "Embedding"]

        large_embeddings = []
        for block in embedding_blocks:
            vocab_size = block.attributes.get("vocab_size", 0)
            embed_dim = block.attributes.get("embed_dim", 0)
            param_count = vocab_size * embed_dim

            if param_count > self.LARGE_EMBEDDING_THRESHOLD:
                large_embeddings.append((block.name, vocab_size, embed_dim, param_count))

        if large_embeddings:
            details = [
                f"{name}: vocab={v}, dim={d}, params={p:,}" for name, v, d, p in large_embeddings
            ]
            return RiskSignal(
                id="large_embedding",
                severity="info",
                description=(
                    f"Found {len(large_embeddings)} large embedding table(s): {'; '.join(details)}. "
                    "These dominate model size."
                ),
                nodes=[name for name, _, _, _ in large_embeddings],
                recommendation=(
                    "Consider vocabulary pruning, dimensionality reduction, or "
                    "hash embeddings to reduce size."
                ),
            )

        return None

    def check_unusual_activations(self, graph_info: GraphInfo) -> RiskSignal | None:
        """
        Flag unusual activation function patterns.

        Some activation combinations may indicate issues.
        Only checks models with sufficient complexity.
        """
        # Skip small models
        if graph_info.num_nodes < self.MIN_NODES_FOR_DEPTH_CHECK:
            return None

        # Check for deprecated or unusual activations
        unusual_ops = {"Elu", "Selu", "ThresholdedRelu", "Softsign", "Softplus"}
        found_unusual = []

        for op in unusual_ops:
            if op in graph_info.op_type_counts:
                found_unusual.append(f"{op} (x{graph_info.op_type_counts[op]})")

        # Check for missing activations in deep networks
        standard_activations = {
            "Relu",
            "LeakyRelu",
            "Gelu",
            "Silu",
            "Sigmoid",
            "Tanh",
            "Softmax",
        }
        has_standard = any(op in graph_info.op_type_counts for op in standard_activations)

        trainable_count = (
            graph_info.op_type_counts.get("Conv", 0)
            + graph_info.op_type_counts.get("MatMul", 0)
            + graph_info.op_type_counts.get("Gemm", 0)
        )

        # Need at least N trainable layers to care about missing activations
        min_trainable = self.thresholds.min_trainable_for_activation_check
        if not has_standard and trainable_count >= min_trainable:
            return RiskSignal(
                id="no_activations",
                severity="warning",
                description=(
                    f"Model has {trainable_count} linear layers but no standard activation functions. "
                    "This makes the model effectively linear, limiting expressiveness."
                ),
                nodes=[],
                recommendation="Add activation functions between linear layers.",
            )

        if found_unusual:
            return RiskSignal(
                id="unusual_activations",
                severity="info",
                description=(
                    f"Model uses less common activation functions: {', '.join(found_unusual)}. "
                    "These may have limited hardware acceleration support."
                ),
                nodes=[],
                recommendation=(
                    "Consider using more common activations (ReLU, GELU, SiLU) for better "
                    "hardware support, unless these specific activations are required."
                ),
            )

        return None

    def check_nonstandard_residuals(
        self, graph_info: GraphInfo, blocks: list[Block]
    ) -> RiskSignal | None:
        """
        Flag non-standard residual/skip connection patterns.

        Non-standard patterns include:
        - Concat-based skip connections (DenseNet-style)
        - Gated skip connections (Highway networks)
        - Subtraction-based residuals

        These may require special handling for optimization or deployment.
        """
        # Identify non-standard residual blocks
        nonstandard_types = {
            "ResidualConcat": "concat-based (DenseNet-style)",
            "ResidualGate": "gated (Highway/attention gate)",
            "ResidualSub": "subtraction-based",
        }

        found_nonstandard: dict[str, list[str]] = {}
        for block in blocks:
            if block.block_type in nonstandard_types:
                variant = nonstandard_types[block.block_type]
                if variant not in found_nonstandard:
                    found_nonstandard[variant] = []
                found_nonstandard[variant].append(block.name)

        if not found_nonstandard:
            return None

        # Build description
        details = []
        all_nodes = []
        for variant, block_names in found_nonstandard.items():
            details.append(f"{len(block_names)} {variant}")
            all_nodes.extend(block_names)

        total_count = sum(len(names) for names in found_nonstandard.values())

        # Check if model also has standard residuals
        standard_count = sum(1 for b in blocks if b.block_type == "ResidualAdd")
        mixed_msg = ""
        if standard_count > 0:
            mixed_msg = f" Model also has {standard_count} standard Add-based residuals."

        return RiskSignal(
            id="nonstandard_residuals",
            severity="info",
            description=(
                f"Model uses {total_count} non-standard skip connection(s): "
                f"{', '.join(details)}.{mixed_msg} "
                "These patterns may indicate custom architectures requiring special attention."
            ),
            nodes=all_nodes,
            recommendation=(
                "Non-standard skip connections are valid but may need special handling: "
                "Concat-based patterns increase tensor sizes through the network. "
                "Gated patterns add compute overhead but enable selective information flow. "
                "Ensure your deployment target and optimization tools support these patterns."
            ),
        )
