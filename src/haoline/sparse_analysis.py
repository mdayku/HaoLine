# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
Sparse and Efficient Architecture Analysis (Epic 29).

This module analyzes sparse models and efficient architecture patterns:

1. **Mixture of Experts (MoE) Analysis** (Story 29.1):
   - Detect MoE routing patterns (top-k gating)
   - Count total/active experts
   - Calculate effective vs total parameters
   - Analyze expert utilization

2. **Speculative Decoding Detection** (Story 29.2):
   - Detect draft + verify model patterns
   - Identify draft model architecture
   - Calculate speedup potential

3. **Weight Sparsity Analysis** (Story 29.3):
   - Detect structured sparsity (N:M patterns)
   - Detect unstructured sparsity (pruned weights)
   - Calculate actual vs theoretical FLOPs

4. **Efficient Architecture Patterns** (Story 29.4):
   - Depthwise separable convolutions
   - Inverted residual blocks (MobileNet)
   - Squeeze-and-excitation patterns
   - Neural architecture search patterns

Usage:
    from haoline.sparse_analysis import SparseAnalyzer

    analyzer = SparseAnalyzer()
    result = analyzer.analyze(graph_info, blocks)
    print(result.moe_info)  # MoE details if detected
    print(result.sparsity_info)  # Weight sparsity analysis
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from haoline.analyzer import GraphInfo
    from haoline.patterns import Block


# =============================================================================
# Enums and Constants
# =============================================================================


class MoERoutingType(Enum):
    """MoE routing strategy types."""

    TOP_K = "top_k"  # Standard top-k gating (e.g., top-2)
    SWITCH = "switch"  # Switch Transformer (top-1)
    SOFT_MOE = "soft_moe"  # Soft MoE (all experts weighted)
    HASH = "hash"  # Hash-based routing
    UNKNOWN = "unknown"


class SparsityType(Enum):
    """Weight sparsity types."""

    DENSE = "dense"  # No sparsity
    UNSTRUCTURED = "unstructured"  # Random pruned weights
    STRUCTURED_NM = "structured_nm"  # N:M sparsity (e.g., 2:4)
    BLOCK = "block"  # Block-wise sparsity
    CHANNEL = "channel"  # Channel pruning


class EfficientPattern(Enum):
    """Efficient architecture patterns."""

    DEPTHWISE_SEPARABLE = "depthwise_separable"
    INVERTED_RESIDUAL = "inverted_residual"
    SQUEEZE_EXCITATION = "squeeze_excitation"
    NAS_BLOCK = "nas_block"
    GHOST_MODULE = "ghost_module"
    LINEAR_ATTENTION = "linear_attention"


# Hardware requirements for sparse acceleration
SPARSE_HARDWARE_REQUIREMENTS: dict[str, dict[str, Any]] = {
    "structured_nm": {
        "nvidia_ampere": True,  # A100, RTX 30xx support 2:4 sparsity
        "nvidia_hopper": True,  # H100 with enhanced sparse support
        "nvidia_volta": False,
        "intel": False,
        "amd": False,
    },
    "unstructured": {
        "nvidia_ampere": False,  # No native HW acceleration
        "nvidia_hopper": False,
        "intel": True,  # Some sparse ops via MKL
        "amd": False,
    },
}


# =============================================================================
# Data Models
# =============================================================================


class ExpertInfo(BaseModel):
    """Information about a single expert in MoE."""

    model_config = ConfigDict(extra="forbid")

    name: str = ""
    parameter_count: int = 0
    layer_indices: list[int] = Field(default_factory=list)


class MoEInfo(BaseModel):
    """Mixture of Experts analysis result."""

    model_config = ConfigDict(extra="forbid")

    detected: bool = False
    routing_type: str = "unknown"  # MoERoutingType value
    num_experts: int = 0
    active_experts_per_token: int = 0  # top-k value
    num_expert_layers: int = 0  # How many MoE layers in model

    # Parameter analysis
    total_parameters: int = 0
    active_parameters_per_token: int = 0  # Effective params when routing
    parameter_efficiency: float = 0.0  # active/total ratio

    # Memory analysis
    memory_all_experts_gb: float = 0.0
    memory_active_subset_gb: float = 0.0

    # Expert parallelism
    expert_parallelism_detected: bool = False
    recommended_expert_parallel_degree: int = 1

    # Load balancing
    has_load_balance_loss: bool = False
    aux_loss_weight: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "detected": self.detected,
            "routing_type": self.routing_type,
            "num_experts": self.num_experts,
            "active_experts_per_token": self.active_experts_per_token,
            "num_expert_layers": self.num_expert_layers,
            "total_parameters": self.total_parameters,
            "active_parameters_per_token": self.active_parameters_per_token,
            "parameter_efficiency": self.parameter_efficiency,
            "memory_all_experts_gb": self.memory_all_experts_gb,
            "memory_active_subset_gb": self.memory_active_subset_gb,
            "expert_parallelism_detected": self.expert_parallelism_detected,
            "recommended_expert_parallel_degree": self.recommended_expert_parallel_degree,
            "has_load_balance_loss": self.has_load_balance_loss,
            "aux_loss_weight": self.aux_loss_weight,
        }


class SpeculativeDecodingInfo(BaseModel):
    """Speculative decoding detection result."""

    model_config = ConfigDict(extra="forbid")

    detected: bool = False
    has_draft_model: bool = False
    has_verify_model: bool = False

    # Draft model characteristics
    draft_model_layers: int = 0
    draft_model_params: int = 0
    draft_hidden_size: int = 0

    # Speedup potential
    estimated_speedup: float = 1.0  # Expected tokens/second improvement
    recommended_gamma: int = 4  # Number of draft tokens per verify step
    acceptance_rate_needed: float = 0.7  # Min rate for speedup

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "detected": self.detected,
            "has_draft_model": self.has_draft_model,
            "has_verify_model": self.has_verify_model,
            "draft_model_layers": self.draft_model_layers,
            "draft_model_params": self.draft_model_params,
            "draft_hidden_size": self.draft_hidden_size,
            "estimated_speedup": self.estimated_speedup,
            "recommended_gamma": self.recommended_gamma,
            "acceptance_rate_needed": self.acceptance_rate_needed,
        }


class LayerSparsityInfo(BaseModel):
    """Sparsity information for a single layer."""

    model_config = ConfigDict(extra="forbid")

    layer_name: str = ""
    sparsity_type: str = "dense"  # SparsityType value
    sparsity_ratio: float = 0.0  # 0.0 = dense, 0.9 = 90% sparse
    nm_pattern: str | None = None  # e.g., "2:4" for structured
    pruned_elements: int = 0
    total_elements: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "layer_name": self.layer_name,
            "sparsity_type": self.sparsity_type,
            "sparsity_ratio": self.sparsity_ratio,
            "nm_pattern": self.nm_pattern,
            "pruned_elements": self.pruned_elements,
            "total_elements": self.total_elements,
        }


class SparsityInfo(BaseModel):
    """Weight sparsity analysis result."""

    model_config = ConfigDict(extra="forbid")

    detected: bool = False
    primary_sparsity_type: str = "dense"  # SparsityType value
    overall_sparsity_ratio: float = 0.0  # Average across model

    # Per-layer breakdown
    sparse_layer_count: int = 0
    dense_layer_count: int = 0
    layer_sparsity: list[LayerSparsityInfo] = Field(default_factory=list)

    # FLOPs analysis
    theoretical_flops: int = 0  # Without sparsity
    actual_flops: int = 0  # With sparsity applied
    flops_reduction_ratio: float = 0.0  # How much compute is saved

    # Hardware compatibility
    hardware_accelerated: bool = False
    compatible_hardware: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "detected": self.detected,
            "primary_sparsity_type": self.primary_sparsity_type,
            "overall_sparsity_ratio": self.overall_sparsity_ratio,
            "sparse_layer_count": self.sparse_layer_count,
            "dense_layer_count": self.dense_layer_count,
            "layer_sparsity": [ls.to_dict() for ls in self.layer_sparsity],
            "theoretical_flops": self.theoretical_flops,
            "actual_flops": self.actual_flops,
            "flops_reduction_ratio": self.flops_reduction_ratio,
            "hardware_accelerated": self.hardware_accelerated,
            "compatible_hardware": self.compatible_hardware,
        }


class EfficientPatternInfo(BaseModel):
    """Efficient architecture pattern detection result."""

    model_config = ConfigDict(extra="forbid")

    pattern_type: str = ""  # EfficientPattern value
    count: int = 0
    layer_names: list[str] = Field(default_factory=list)

    # Efficiency metrics
    flops_saved_ratio: float = 0.0  # vs baseline implementation
    param_reduction_ratio: float = 0.0  # vs baseline

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pattern_type": self.pattern_type,
            "count": self.count,
            "layer_names": self.layer_names,
            "flops_saved_ratio": self.flops_saved_ratio,
            "param_reduction_ratio": self.param_reduction_ratio,
        }


class EfficientArchInfo(BaseModel):
    """Efficient architecture patterns analysis result."""

    model_config = ConfigDict(extra="forbid")

    architecture_type: str = "standard"  # e.g., "mobilenet", "efficientnet", "nas"
    patterns_detected: list[EfficientPatternInfo] = Field(default_factory=list)

    # Overall efficiency
    total_params: int = 0
    equivalent_baseline_params: int = 0  # If using standard convs
    param_efficiency_ratio: float = 1.0

    total_flops: int = 0
    equivalent_baseline_flops: int = 0
    flops_efficiency_ratio: float = 1.0

    # Comparison to baseline
    baseline_architecture: str | None = None  # e.g., "ResNet" for MobileNet comparison

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "architecture_type": self.architecture_type,
            "patterns_detected": [p.to_dict() for p in self.patterns_detected],
            "total_params": self.total_params,
            "equivalent_baseline_params": self.equivalent_baseline_params,
            "param_efficiency_ratio": self.param_efficiency_ratio,
            "total_flops": self.total_flops,
            "equivalent_baseline_flops": self.equivalent_baseline_flops,
            "flops_efficiency_ratio": self.flops_efficiency_ratio,
            "baseline_architecture": self.baseline_architecture,
        }


class SparseAnalysisResult(BaseModel):
    """Complete sparse and efficient architecture analysis result."""

    model_config = ConfigDict(extra="forbid")

    # MoE Analysis (Story 29.1)
    moe_info: MoEInfo = Field(default_factory=MoEInfo)

    # Speculative Decoding (Story 29.2)
    speculative_decoding_info: SpeculativeDecodingInfo = Field(
        default_factory=SpeculativeDecodingInfo
    )

    # Weight Sparsity (Story 29.3)
    sparsity_info: SparsityInfo = Field(default_factory=SparsityInfo)

    # Efficient Architectures (Story 29.4)
    efficient_arch_info: EfficientArchInfo = Field(default_factory=EfficientArchInfo)

    # Overall summary
    is_sparse_model: bool = False
    is_moe_model: bool = False
    is_efficient_arch: bool = False
    recommendations: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "moe_info": self.moe_info.to_dict(),
            "speculative_decoding_info": self.speculative_decoding_info.to_dict(),
            "sparsity_info": self.sparsity_info.to_dict(),
            "efficient_arch_info": self.efficient_arch_info.to_dict(),
            "is_sparse_model": self.is_sparse_model,
            "is_moe_model": self.is_moe_model,
            "is_efficient_arch": self.is_efficient_arch,
            "recommendations": self.recommendations,
        }

    def get_summary(self) -> str:
        """Generate human-readable summary."""
        lines = ["Sparse & Efficient Architecture Analysis", "=" * 42]

        # MoE summary
        if self.moe_info.detected:
            lines.append("")
            lines.append("Mixture of Experts (MoE) Detected")
            lines.append(f"  Routing: {self.moe_info.routing_type}")
            lines.append(
                f"  Experts: {self.moe_info.num_experts} total, "
                f"{self.moe_info.active_experts_per_token} active/token"
            )
            lines.append(f"  Parameter efficiency: {self.moe_info.parameter_efficiency:.1%}")
            lines.append(
                f"  Memory: {self.moe_info.memory_all_experts_gb:.2f} GB all, "
                f"{self.moe_info.memory_active_subset_gb:.2f} GB active"
            )

        # Sparsity summary
        if self.sparsity_info.detected:
            lines.append("")
            lines.append("Weight Sparsity Detected")
            lines.append(f"  Type: {self.sparsity_info.primary_sparsity_type}")
            lines.append(f"  Overall ratio: {self.sparsity_info.overall_sparsity_ratio:.1%}")
            lines.append(f"  FLOPs reduction: {self.sparsity_info.flops_reduction_ratio:.1%}")
            if self.sparsity_info.hardware_accelerated:
                lines.append(
                    f"  HW accelerated: {', '.join(self.sparsity_info.compatible_hardware)}"
                )

        # Efficient patterns summary
        if self.efficient_arch_info.patterns_detected:
            lines.append("")
            lines.append("Efficient Architecture Patterns")
            for pattern in self.efficient_arch_info.patterns_detected:
                lines.append(f"  {pattern.pattern_type}: {pattern.count} instances")
            lines.append(
                f"  FLOPs efficiency: {self.efficient_arch_info.flops_efficiency_ratio:.2f}x"
            )

        # Recommendations
        if self.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")

        if not (
            self.moe_info.detected
            or self.sparsity_info.detected
            or self.efficient_arch_info.patterns_detected
        ):
            lines.append("")
            lines.append("No sparse/efficient patterns detected (standard dense model)")

        return "\n".join(lines)


# =============================================================================
# Analyzer Class
# =============================================================================


class SparseAnalyzer:
    """
    Analyzer for sparse models and efficient architectures.

    Detects:
    - Mixture of Experts (MoE) patterns
    - Weight sparsity (structured and unstructured)
    - Efficient architecture patterns (MobileNet, EfficientNet, etc.)
    - Speculative decoding configurations
    """

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger("haoline.sparse_analysis")

    def analyze(
        self,
        graph_info: GraphInfo,
        blocks: list[Block] | None = None,
        total_params: int = 0,
        total_flops: int = 0,
    ) -> SparseAnalysisResult:
        """
        Analyze model for sparse and efficient patterns.

        Args:
            graph_info: Graph analysis info from analyzer module.
            blocks: Pattern blocks from PatternAnalyzer.
            total_params: Total parameter count.
            total_flops: Total FLOPs count.

        Returns:
            SparseAnalysisResult with all findings.
        """
        result = SparseAnalysisResult()
        recommendations: list[str] = []

        # Story 29.1: MoE Analysis
        moe_info = self._analyze_moe(graph_info, blocks, total_params)
        result.moe_info = moe_info
        result.is_moe_model = moe_info.detected

        if moe_info.detected:
            if moe_info.num_experts > 8 and moe_info.recommended_expert_parallel_degree > 1:
                recommendations.append(
                    f"Consider expert parallelism with degree "
                    f"{moe_info.recommended_expert_parallel_degree} for {moe_info.num_experts} experts"
                )
            if not moe_info.has_load_balance_loss:
                recommendations.append(
                    "No load balance loss detected - experts may be underutilized"
                )

        # Story 29.2: Speculative Decoding Detection
        spec_info = self._detect_speculative_decoding(graph_info, blocks)
        result.speculative_decoding_info = spec_info

        if spec_info.detected:
            recommendations.append(
                f"Speculative decoding detected: ~{spec_info.estimated_speedup:.1f}x speedup "
                f"with gamma={spec_info.recommended_gamma}"
            )

        # Story 29.3: Sparsity Analysis
        sparsity_info = self._analyze_sparsity(graph_info, total_flops)
        result.sparsity_info = sparsity_info
        result.is_sparse_model = sparsity_info.detected

        if sparsity_info.detected:
            if sparsity_info.primary_sparsity_type == SparsityType.STRUCTURED_NM.value:
                if sparsity_info.hardware_accelerated:
                    recommendations.append(
                        "2:4 structured sparsity detected - use Ampere/Hopper GPU for acceleration"
                    )
            elif sparsity_info.primary_sparsity_type == SparsityType.UNSTRUCTURED.value:
                if sparsity_info.overall_sparsity_ratio > 0.8:
                    recommendations.append(
                        "High unstructured sparsity (>80%) - consider converting to structured for HW acceleration"
                    )

        # Story 29.4: Efficient Architecture Patterns
        efficient_info = self._analyze_efficient_patterns(
            graph_info, blocks, total_params, total_flops
        )
        result.efficient_arch_info = efficient_info
        result.is_efficient_arch = len(efficient_info.patterns_detected) > 0

        if efficient_info.patterns_detected:
            for pattern in efficient_info.patterns_detected:
                if pattern.pattern_type == EfficientPattern.DEPTHWISE_SEPARABLE.value:
                    recommendations.append(
                        f"Depthwise separable convolutions used ({pattern.count}x) - "
                        "optimized for mobile deployment"
                    )

        result.recommendations = recommendations
        return result

    # =========================================================================
    # Story 29.1: MoE Analysis
    # =========================================================================

    def _analyze_moe(
        self,
        graph_info: GraphInfo,
        blocks: list[Block] | None,
        total_params: int,
    ) -> MoEInfo:
        """Analyze Mixture of Experts patterns."""
        info = MoEInfo()

        # Look for MoE blocks from pattern analyzer
        moe_blocks = []
        if blocks:
            moe_blocks = [b for b in blocks if "MoE" in b.block_type]

        # Also detect from graph if not already detected
        if not moe_blocks:
            # Look for TopK operations (expert selection indicator)
            topk_nodes = [n for n in graph_info.nodes if n.op_type == "TopK"]

            # Look for expert-related patterns in tensor names
            expert_pattern_count = 0
            for node in graph_info.nodes:
                if any(pat in node.name.lower() for pat in ["expert", "moe", "router", "gate"]):
                    expert_pattern_count += 1

            if topk_nodes or expert_pattern_count > 5:
                # Likely MoE model
                info.detected = True
                info.routing_type = MoERoutingType.TOP_K.value

                # Try to extract k value from TopK
                for topk in topk_nodes:
                    for attr in topk.attributes if hasattr(topk, "attributes") else []:
                        if attr[0] == "k":
                            info.active_experts_per_token = int(attr[1])
                            break
                    if info.active_experts_per_token:
                        break

                if not info.active_experts_per_token:
                    info.active_experts_per_token = 2  # Common default

        if moe_blocks:
            info.detected = True
            info.num_expert_layers = len(moe_blocks)

            # Extract details from blocks
            for block in moe_blocks:
                attrs = block.attributes if hasattr(block, "attributes") else {}
                if isinstance(attrs, dict):
                    if "num_experts" in attrs and attrs["num_experts"] > 0:
                        info.num_experts = max(info.num_experts, attrs["num_experts"])
                    if "top_k" in attrs and attrs["top_k"] > 0:
                        info.active_experts_per_token = attrs["top_k"]
                    if attrs.get("router_type"):
                        info.routing_type = attrs["router_type"]

        if info.detected:
            # Estimate expert count from tensor patterns if not found
            if info.num_experts == 0:
                info.num_experts = self._estimate_expert_count(graph_info)
                if info.num_experts == 0:
                    info.num_experts = 8  # Common default

            if info.active_experts_per_token == 0:
                info.active_experts_per_token = 2  # Common default

            # Calculate parameter efficiency
            if info.num_experts > 0:
                # Rough estimate: active = total * (active_k / num_experts)
                # But shared components (attention, embed) aren't duplicated
                # Estimate ~60% of params are in expert FFN
                expert_ratio = info.active_experts_per_token / info.num_experts
                info.parameter_efficiency = 0.4 + (0.6 * expert_ratio)

            # Memory estimates (rough)
            if total_params > 0:
                bytes_per_param = 2  # Assume FP16
                info.total_parameters = total_params
                info.memory_all_experts_gb = (total_params * bytes_per_param) / (1024**3)
                info.active_parameters_per_token = int(total_params * info.parameter_efficiency)
                info.memory_active_subset_gb = (
                    info.active_parameters_per_token * bytes_per_param
                ) / (1024**3)

            # Expert parallelism recommendation
            if info.num_experts >= 8:
                info.expert_parallelism_detected = self._detect_expert_parallelism(graph_info)
                # Recommend 1 GPU per N experts, minimum 1
                info.recommended_expert_parallel_degree = max(1, info.num_experts // 4)

            # Check for load balance loss
            info.has_load_balance_loss = self._detect_load_balance_loss(graph_info)

        return info

    def _estimate_expert_count(self, graph_info: GraphInfo) -> int:
        """Estimate number of experts from tensor naming patterns."""
        expert_numbers: set[int] = set()

        for node in graph_info.nodes:
            name_lower = node.name.lower()
            # Look for patterns like "expert_0", "experts.1", "expert[2]"
            import re

            matches = re.findall(r"expert[s_.\[\]]*(\d+)", name_lower)
            for m in matches:
                try:
                    expert_numbers.add(int(m))
                except ValueError:
                    pass

        if expert_numbers:
            return max(expert_numbers) + 1  # 0-indexed

        return 0

    def _detect_expert_parallelism(self, graph_info: GraphInfo) -> bool:
        """Detect if model uses expert parallelism."""
        # Look for AllToAll operations (common in expert parallelism)
        for node in graph_info.nodes:
            if "AllToAll" in node.op_type or "all_to_all" in node.name.lower():
                return True
        return False

    def _detect_load_balance_loss(self, graph_info: GraphInfo) -> bool:
        """Detect if model has load balancing auxiliary loss."""
        for node in graph_info.nodes:
            name_lower = node.name.lower()
            if any(
                pat in name_lower
                for pat in ["balance", "aux_loss", "load_loss", "z_loss", "router_loss"]
            ):
                return True
        return False

    # =========================================================================
    # Story 29.2: Speculative Decoding Detection
    # =========================================================================

    def _detect_speculative_decoding(
        self,
        graph_info: GraphInfo,
        blocks: list[Block] | None,
    ) -> SpeculativeDecodingInfo:
        """Detect speculative decoding (draft + verify) patterns."""
        info = SpeculativeDecodingInfo()

        # Speculative decoding requires two models or a model with explicit draft/verify
        # Look for naming patterns
        has_draft = False
        has_verify = False
        draft_layer_count = 0

        for node in graph_info.nodes:
            name_lower = node.name.lower()
            if any(pat in name_lower for pat in ["draft", "small", "tiny"]):
                has_draft = True
                if "layer" in name_lower or "block" in name_lower:
                    draft_layer_count += 1
            if any(pat in name_lower for pat in ["verify", "main", "large", "target"]):
                has_verify = True

        if has_draft and has_verify:
            info.detected = True
            info.has_draft_model = True
            info.has_verify_model = True
            info.draft_model_layers = max(draft_layer_count, 1)

            # Estimate speedup based on draft size
            # Smaller draft = faster but lower acceptance rate
            # Formula: speedup ~= gamma * acceptance_rate / (1 + gamma * draft_cost)
            # Assume draft is ~10% cost of main model
            info.recommended_gamma = 4
            info.acceptance_rate_needed = 0.7
            info.estimated_speedup = info.recommended_gamma * info.acceptance_rate_needed / 1.4

        return info

    # =========================================================================
    # Story 29.3: Sparsity Analysis
    # =========================================================================

    def _analyze_sparsity(
        self,
        graph_info: GraphInfo,
        total_flops: int,
    ) -> SparsityInfo:
        """Analyze weight sparsity patterns."""
        info = SparsityInfo()
        layer_sparsity: list[LayerSparsityInfo] = []

        # Analyze each node for sparsity indicators
        sparse_count = 0
        dense_count = 0
        total_sparse_ratio = 0.0

        structured_nm_detected = False
        unstructured_detected = False

        for node in graph_info.nodes:
            if node.op_type not in ("Conv", "MatMul", "Gemm", "ConvTranspose"):
                continue

            # Check for sparsity indicators in name
            name_lower = node.name.lower()
            is_sparse = False
            sparsity_type = SparsityType.DENSE.value
            sparsity_ratio = 0.0
            nm_pattern = None

            # Check for structured sparsity patterns
            if any(pat in name_lower for pat in ["sparse", "pruned", "sparsity"]):
                is_sparse = True
                unstructured_detected = True
                sparsity_type = SparsityType.UNSTRUCTURED.value
                sparsity_ratio = 0.5  # Estimate

            # Check for 2:4 structured sparsity (NVIDIA Ampere)
            if "2:4" in name_lower or "sparse_semi_structured" in name_lower:
                is_sparse = True
                structured_nm_detected = True
                sparsity_type = SparsityType.STRUCTURED_NM.value
                sparsity_ratio = 0.5  # 2:4 = 50% sparsity
                nm_pattern = "2:4"

            # Check for mask tensors (indicates pruning)
            has_mask = any("mask" in inp.lower() for inp in node.inputs if isinstance(inp, str))
            if has_mask:
                is_sparse = True
                if not structured_nm_detected:
                    unstructured_detected = True
                    sparsity_type = SparsityType.UNSTRUCTURED.value
                sparsity_ratio = max(sparsity_ratio, 0.5)

            if is_sparse:
                sparse_count += 1
                total_sparse_ratio += sparsity_ratio
                layer_sparsity.append(
                    LayerSparsityInfo(
                        layer_name=node.name,
                        sparsity_type=sparsity_type,
                        sparsity_ratio=sparsity_ratio,
                        nm_pattern=nm_pattern,
                    )
                )
            else:
                dense_count += 1

        # Aggregate results
        if sparse_count > 0:
            info.detected = True
            info.sparse_layer_count = sparse_count
            info.dense_layer_count = dense_count
            info.layer_sparsity = layer_sparsity
            info.overall_sparsity_ratio = total_sparse_ratio / (sparse_count + dense_count)

            # Determine primary type
            if structured_nm_detected:
                info.primary_sparsity_type = SparsityType.STRUCTURED_NM.value
                info.hardware_accelerated = True
                info.compatible_hardware = [
                    "NVIDIA Ampere (A100, RTX 30xx)",
                    "NVIDIA Hopper (H100)",
                ]
            elif unstructured_detected:
                info.primary_sparsity_type = SparsityType.UNSTRUCTURED.value
                info.hardware_accelerated = False
                info.compatible_hardware = []

            # FLOPs reduction
            info.theoretical_flops = total_flops
            info.actual_flops = int(total_flops * (1 - info.overall_sparsity_ratio))
            info.flops_reduction_ratio = info.overall_sparsity_ratio

        return info

    # =========================================================================
    # Story 29.4: Efficient Architecture Patterns
    # =========================================================================

    def _analyze_efficient_patterns(
        self,
        graph_info: GraphInfo,
        blocks: list[Block] | None,
        total_params: int,
        total_flops: int,
    ) -> EfficientArchInfo:
        """Analyze efficient architecture patterns."""
        info = EfficientArchInfo()
        patterns: list[EfficientPatternInfo] = []

        # Detect depthwise separable convolutions
        depthwise_info = self._detect_depthwise_separable(graph_info)
        if depthwise_info.count > 0:
            patterns.append(depthwise_info)

        # Detect inverted residual blocks (MobileNetV2 style)
        inverted_info = self._detect_inverted_residual(graph_info)
        if inverted_info.count > 0:
            patterns.append(inverted_info)

        # Detect squeeze-and-excitation
        se_info = self._detect_squeeze_excitation(graph_info)
        if se_info.count > 0:
            patterns.append(se_info)

        # Detect NAS-style patterns
        nas_info = self._detect_nas_patterns(graph_info)
        if nas_info.count > 0:
            patterns.append(nas_info)

        info.patterns_detected = patterns
        info.total_params = total_params
        info.total_flops = total_flops

        # Classify architecture type
        if depthwise_info.count > 5 and inverted_info.count > 0:
            info.architecture_type = "mobilenet"
            info.baseline_architecture = "ResNet"
            # MobileNet is typically 8-10x more efficient than ResNet
            info.flops_efficiency_ratio = 8.0
            info.param_efficiency_ratio = 7.0
        elif se_info.count > 3:
            info.architecture_type = "efficientnet"
            info.baseline_architecture = "ResNet"
            info.flops_efficiency_ratio = 5.0
            info.param_efficiency_ratio = 6.0
        elif nas_info.count > 0:
            info.architecture_type = "nas"
            info.baseline_architecture = "standard CNN"
            info.flops_efficiency_ratio = 3.0
            info.param_efficiency_ratio = 3.0
        elif depthwise_info.count > 0:
            info.architecture_type = "lightweight"
            info.flops_efficiency_ratio = 2.0
            info.param_efficiency_ratio = 2.0

        return info

    def _detect_depthwise_separable(self, graph_info: GraphInfo) -> EfficientPatternInfo:
        """Detect depthwise separable convolutions."""
        info = EfficientPatternInfo(pattern_type=EfficientPattern.DEPTHWISE_SEPARABLE.value)

        depthwise_layers: list[str] = []

        for node in graph_info.nodes:
            if node.op_type not in ("Conv", "ConvInteger"):
                continue

            name_lower = node.name.lower()

            # Check attributes for group = channels (depthwise indicator)
            is_depthwise = False
            for attr in node.attributes if hasattr(node, "attributes") else []:
                if attr[0] == "group" and int(attr[1]) > 1:
                    is_depthwise = True
                    break

            # Also check naming
            if any(pat in name_lower for pat in ["depthwise", "dw_conv", "depth_conv", "dwconv"]):
                is_depthwise = True

            if is_depthwise:
                depthwise_layers.append(node.name)

        info.count = len(depthwise_layers)
        info.layer_names = depthwise_layers[:10]  # Limit for readability

        # Depthwise separable saves ~k^2 FLOPs vs standard conv
        # Typical kernel 3x3 = 9x reduction, but pointwise adds back
        # Net savings ~8-9x for 3x3 kernels
        if info.count > 0:
            info.flops_saved_ratio = 0.85  # ~85% FLOPs saved vs standard
            info.param_reduction_ratio = 0.80

        return info

    def _detect_inverted_residual(self, graph_info: GraphInfo) -> EfficientPatternInfo:
        """Detect inverted residual blocks (MobileNetV2 style)."""
        info = EfficientPatternInfo(pattern_type=EfficientPattern.INVERTED_RESIDUAL.value)

        inverted_layers: list[str] = []

        # Look for expand -> depthwise -> project pattern
        # Or naming conventions
        for node in graph_info.nodes:
            name_lower = node.name.lower()
            if any(
                pat in name_lower for pat in ["inverted", "mbconv", "expand_conv", "bottleneck"]
            ):
                inverted_layers.append(node.name)

        # Deduplicate by looking at unique block prefixes
        unique_blocks: set[str] = set()
        for name in inverted_layers:
            # Extract block identifier (e.g., "block_3" from "block_3.expand_conv")
            parts = name.split(".")
            if len(parts) > 1:
                unique_blocks.add(parts[0])
            else:
                unique_blocks.add(name)

        info.count = len(unique_blocks) if unique_blocks else len(inverted_layers)
        info.layer_names = list(unique_blocks)[:10]

        if info.count > 0:
            info.flops_saved_ratio = 0.6
            info.param_reduction_ratio = 0.5

        return info

    def _detect_squeeze_excitation(self, graph_info: GraphInfo) -> EfficientPatternInfo:
        """Detect squeeze-and-excitation blocks."""
        info = EfficientPatternInfo(pattern_type=EfficientPattern.SQUEEZE_EXCITATION.value)

        se_layers: list[str] = []

        for node in graph_info.nodes:
            name_lower = node.name.lower()
            if any(
                pat in name_lower
                for pat in ["squeeze", "excite", "se_block", "se_module", "channel_attention"]
            ):
                se_layers.append(node.name)

        # Also look for GlobalAveragePool -> FC -> FC -> Sigmoid pattern
        gap_nodes = [n for n in graph_info.nodes if "GlobalAveragePool" in n.op_type]
        for gap in gap_nodes:
            # Check if followed by small FC layers and sigmoid
            # (simplified detection via naming for now)
            se_layers.append(gap.name)

        info.count = len(se_layers) // 2  # SE has multiple components per block
        info.layer_names = se_layers[:10]

        if info.count > 0:
            info.flops_saved_ratio = 0.02  # SE adds ~2% overhead but improves accuracy
            info.param_reduction_ratio = 0.0  # Adds a few params

        return info

    def _detect_nas_patterns(self, graph_info: GraphInfo) -> EfficientPatternInfo:
        """Detect neural architecture search patterns."""
        info = EfficientPatternInfo(pattern_type=EfficientPattern.NAS_BLOCK.value)

        nas_indicators: list[str] = []

        for node in graph_info.nodes:
            name_lower = node.name.lower()
            if any(
                pat in name_lower
                for pat in [
                    "nas",
                    "searchable",
                    "choice_block",
                    "darts",
                    "proxyless",
                    "fbnet",
                    "mnasnet",
                ]
            ):
                nas_indicators.append(node.name)

        info.count = len(nas_indicators)
        info.layer_names = nas_indicators[:10]

        return info


# =============================================================================
# Convenience Function
# =============================================================================


def analyze_sparse(
    graph_info: GraphInfo,
    blocks: list[Block] | None = None,
    total_params: int = 0,
    total_flops: int = 0,
    logger: logging.Logger | None = None,
) -> SparseAnalysisResult:
    """
    Convenience function to analyze sparse and efficient patterns.

    Args:
        graph_info: Graph analysis info from analyzer module.
        blocks: Pattern blocks from PatternAnalyzer.
        total_params: Total parameter count.
        total_flops: Total FLOPs count.
        logger: Optional logger instance.

    Returns:
        SparseAnalysisResult with all findings.
    """
    analyzer = SparseAnalyzer(logger=logger)
    return analyzer.analyze(graph_info, blocks, total_params, total_flops)
