"""
Deployment Cost Calculator for HaoLine.

Calculate the cost of running ML models in production given:
- Target throughput (FPS or samples/sec)
- Operating hours per day
- Hardware tier and cloud provider pricing

Answers: "What does it cost to run this model at X fps?"
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DeploymentTarget(str, Enum):
    """Deployment target environment."""

    CLOUD_GPU = "cloud_gpu"  # Cloud GPU instances (AWS, GCP, Azure)
    CLOUD_CPU = "cloud_cpu"  # Cloud CPU instances
    EDGE_GPU = "edge_gpu"  # Edge devices with GPU (Jetson, etc.)
    EDGE_CPU = "edge_cpu"  # Edge devices CPU-only
    ON_PREM = "on_prem"  # On-premises servers


class CloudProvider(str, Enum):
    """Cloud provider for cost estimation."""

    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    GENERIC = "generic"  # Use average pricing


class DeploymentScenario(BaseModel):
    """
    Defines a deployment scenario for cost estimation.

    This is the input to the cost calculator - describes what the user
    wants to achieve (throughput, uptime, etc.).
    """

    model_config = ConfigDict(frozen=False)  # Allow mutation for presets

    # Throughput requirements
    target_fps: float = 30.0  # Target frames/samples per second
    batch_size: int = 1  # Inference batch size

    # Operating schedule
    hours_per_day: float = 24.0  # Hours the model runs per day
    days_per_month: int = 30  # Days per month to calculate costs

    # Hardware preferences
    target: DeploymentTarget = DeploymentTarget.CLOUD_GPU
    provider: CloudProvider = CloudProvider.GENERIC
    precision: str = "fp32"  # fp32, fp16, int8

    # Latency constraints
    max_latency_ms: float | None = None  # Maximum acceptable latency (SLA)

    # Redundancy
    replicas: int = 1  # Number of model replicas for availability

    # Optional metadata
    name: str = ""  # Scenario name for reports
    notes: str = ""  # Additional notes

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "target_fps": self.target_fps,
            "batch_size": self.batch_size,
            "hours_per_day": self.hours_per_day,
            "days_per_month": self.days_per_month,
            "target": self.target.value,
            "provider": self.provider.value,
            "precision": self.precision,
            "max_latency_ms": self.max_latency_ms,
            "replicas": self.replicas,
            "name": self.name,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeploymentScenario:
        """Create from dictionary."""
        return cls(
            target_fps=data.get("target_fps", 30.0),
            batch_size=data.get("batch_size", 1),
            hours_per_day=data.get("hours_per_day", 24.0),
            days_per_month=data.get("days_per_month", 30),
            target=DeploymentTarget(data.get("target", "cloud_gpu")),
            provider=CloudProvider(data.get("provider", "generic")),
            precision=data.get("precision", "fp32"),
            max_latency_ms=data.get("max_latency_ms"),
            replicas=data.get("replicas", 1),
            name=data.get("name", ""),
            notes=data.get("notes", ""),
        )

    @classmethod
    def realtime_video(cls, fps: float = 30.0) -> DeploymentScenario:
        """Preset: Real-time video processing (24/7)."""
        return cls(
            target_fps=fps,
            batch_size=1,
            hours_per_day=24.0,
            max_latency_ms=1000.0 / fps,  # Must process faster than frame rate
            name="realtime_video",
        )

    @classmethod
    def batch_processing(
        cls,
        samples_per_hour: int = 10000,
        hours_per_day: float = 8.0,
    ) -> DeploymentScenario:
        """Preset: Batch processing during business hours."""
        fps = samples_per_hour / 3600  # Convert to per-second
        return cls(
            target_fps=fps,
            batch_size=32,
            hours_per_day=hours_per_day,
            max_latency_ms=None,  # Latency not critical
            name="batch_processing",
        )

    @classmethod
    def edge_device(cls, fps: float = 10.0) -> DeploymentScenario:
        """Preset: Edge device deployment."""
        return cls(
            target_fps=fps,
            batch_size=1,
            hours_per_day=24.0,
            target=DeploymentTarget.EDGE_GPU,
            provider=CloudProvider.GENERIC,
            max_latency_ms=100.0,
            name="edge_device",
        )


# =============================================================================
# Hardware Tier Definitions
# =============================================================================


class HardwareTier(BaseModel):
    """
    Defines a hardware tier for deployment.

    Maps to cloud instance types or edge device categories.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    description: str

    # Performance characteristics
    max_tflops_fp32: float  # Peak TFLOPS at FP32
    max_tflops_fp16: float  # Peak TFLOPS at FP16
    max_tflops_int8: float  # Peak TFLOPS at INT8

    # Memory
    memory_gb: float

    # Cost (hourly)
    cost_per_hour_usd: float

    # Provider/target
    provider: CloudProvider = CloudProvider.GENERIC
    target: DeploymentTarget = DeploymentTarget.CLOUD_GPU

    # Instance type (for cloud)
    instance_type: str = ""

    def effective_tflops(self, precision: str = "fp32") -> float:
        """Get effective TFLOPS for a precision level."""
        if precision == "fp16":
            return self.max_tflops_fp16
        elif precision == "int8":
            return self.max_tflops_int8
        else:
            return self.max_tflops_fp32


# Pre-defined hardware tiers (approximate 2024 pricing)
HARDWARE_TIERS: dict[str, HardwareTier] = {
    # Cloud GPU tiers
    "t4": HardwareTier(
        name="T4",
        description="NVIDIA T4 (budget GPU)",
        max_tflops_fp32=8.1,
        max_tflops_fp16=65,
        max_tflops_int8=130,
        memory_gb=16,
        cost_per_hour_usd=0.50,
        instance_type="g4dn.xlarge",
    ),
    "a10g": HardwareTier(
        name="A10G",
        description="NVIDIA A10G (mid-tier GPU)",
        max_tflops_fp32=31.2,
        max_tflops_fp16=125,
        max_tflops_int8=250,
        memory_gb=24,
        cost_per_hour_usd=1.00,
        instance_type="g5.xlarge",
    ),
    "a100_40gb": HardwareTier(
        name="A100-40GB",
        description="NVIDIA A100 40GB (high-end)",
        max_tflops_fp32=19.5,
        max_tflops_fp16=312,
        max_tflops_int8=624,
        memory_gb=40,
        cost_per_hour_usd=3.00,
        instance_type="p4d.24xlarge",
    ),
    "a100_80gb": HardwareTier(
        name="A100-80GB",
        description="NVIDIA A100 80GB (high-memory)",
        max_tflops_fp32=19.5,
        max_tflops_fp16=312,
        max_tflops_int8=624,
        memory_gb=80,
        cost_per_hour_usd=4.00,
        instance_type="p4de.24xlarge",
    ),
    "h100": HardwareTier(
        name="H100",
        description="NVIDIA H100 (latest gen)",
        max_tflops_fp32=67,
        max_tflops_fp16=1979,
        max_tflops_int8=3958,
        memory_gb=80,
        cost_per_hour_usd=8.00,
        instance_type="p5.48xlarge",
    ),
    # Edge devices
    "jetson_nano": HardwareTier(
        name="Jetson Nano",
        description="NVIDIA Jetson Nano (entry edge)",
        max_tflops_fp32=0.472,
        max_tflops_fp16=0.472,
        max_tflops_int8=0.944,
        memory_gb=4,
        cost_per_hour_usd=0.01,  # Amortized device cost
        target=DeploymentTarget.EDGE_GPU,
    ),
    "jetson_orin_nano": HardwareTier(
        name="Jetson Orin Nano",
        description="NVIDIA Jetson Orin Nano (mid edge)",
        max_tflops_fp32=20,
        max_tflops_fp16=40,
        max_tflops_int8=80,
        memory_gb=8,
        cost_per_hour_usd=0.03,
        target=DeploymentTarget.EDGE_GPU,
    ),
    "jetson_orin_nx": HardwareTier(
        name="Jetson Orin NX",
        description="NVIDIA Jetson Orin NX (high edge)",
        max_tflops_fp32=50,
        max_tflops_fp16=100,
        max_tflops_int8=200,
        memory_gb=16,
        cost_per_hour_usd=0.05,
        target=DeploymentTarget.EDGE_GPU,
    ),
    # CPU options
    "cpu_small": HardwareTier(
        name="CPU Small",
        description="4 vCPU cloud instance",
        max_tflops_fp32=0.1,
        max_tflops_fp16=0.1,
        max_tflops_int8=0.2,
        memory_gb=8,
        cost_per_hour_usd=0.10,
        target=DeploymentTarget.CLOUD_CPU,
        instance_type="c5.xlarge",
    ),
    "cpu_large": HardwareTier(
        name="CPU Large",
        description="16 vCPU cloud instance",
        max_tflops_fp32=0.4,
        max_tflops_fp16=0.4,
        max_tflops_int8=0.8,
        memory_gb=32,
        cost_per_hour_usd=0.40,
        target=DeploymentTarget.CLOUD_CPU,
        instance_type="c5.4xlarge",
    ),
}


def get_hardware_tier(name: str) -> HardwareTier | None:
    """Get a hardware tier by name."""
    return HARDWARE_TIERS.get(name.lower())


def list_hardware_tiers(
    target: DeploymentTarget | None = None,
) -> list[HardwareTier]:
    """List available hardware tiers, optionally filtered by target."""
    tiers = list(HARDWARE_TIERS.values())
    if target:
        tiers = [t for t in tiers if t.target == target]
    return sorted(tiers, key=lambda t: t.cost_per_hour_usd)


# =============================================================================
# Cost Estimation Result
# =============================================================================


class DeploymentCostEstimate(BaseModel):
    """
    Result of deployment cost calculation.

    Contains all the computed costs and recommendations.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Input scenario
    scenario: DeploymentScenario

    # Selected hardware
    hardware_tier: HardwareTier
    num_instances: int = 1  # Instances needed to meet throughput

    # Performance estimates
    estimated_latency_ms: float = 0.0
    estimated_throughput_fps: float = 0.0
    meets_latency_sla: bool = True

    # Cost breakdown
    cost_per_hour_usd: float = 0.0
    cost_per_day_usd: float = 0.0
    cost_per_month_usd: float = 0.0

    # Efficiency metrics
    utilization_percent: float = 0.0  # How much of hardware capacity is used
    cost_per_1k_inferences_usd: float = 0.0

    # Warnings/notes
    warnings: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "scenario": self.scenario.to_dict(),
            "hardware_tier": {
                "name": self.hardware_tier.name,
                "description": self.hardware_tier.description,
                "cost_per_hour_usd": self.hardware_tier.cost_per_hour_usd,
            },
            "num_instances": self.num_instances,
            "estimated_latency_ms": self.estimated_latency_ms,
            "estimated_throughput_fps": self.estimated_throughput_fps,
            "meets_latency_sla": self.meets_latency_sla,
            "cost_per_hour_usd": self.cost_per_hour_usd,
            "cost_per_day_usd": self.cost_per_day_usd,
            "cost_per_month_usd": self.cost_per_month_usd,
            "utilization_percent": self.utilization_percent,
            "cost_per_1k_inferences_usd": self.cost_per_1k_inferences_usd,
            "warnings": self.warnings,
        }

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Deployment Cost Estimate: {self.scenario.name or 'Custom Scenario'}",
            f"{'=' * 50}",
            f"Hardware: {self.hardware_tier.name} x {self.num_instances}",
            f"Target:   {self.scenario.target_fps:.1f} fps @ {self.scenario.hours_per_day}h/day",
            "",
            "Performance:",
            f"  Estimated latency:    {self.estimated_latency_ms:.1f} ms",
            f"  Estimated throughput: {self.estimated_throughput_fps:.1f} fps",
            f"  Utilization:          {self.utilization_percent:.0f}%",
        ]

        if self.scenario.max_latency_ms:
            status = "OK" if self.meets_latency_sla else "EXCEEDS SLA"
            lines.append(f"  Latency SLA:          {status}")

        lines.extend(
            [
                "",
                "Costs:",
                f"  Per hour:   ${self.cost_per_hour_usd:.2f}",
                f"  Per day:    ${self.cost_per_day_usd:.2f}",
                f"  Per month:  ${self.cost_per_month_usd:.2f}",
                f"  Per 1K inf: ${self.cost_per_1k_inferences_usd:.4f}",
            ]
        )

        if self.warnings:
            lines.extend(["", "Warnings:"])
            for w in self.warnings:
                lines.append(f"  - {w}")

        return "\n".join(lines)


# =============================================================================
# Cost Calculation Functions (Tasks 12.6.2 and 12.6.3)
# =============================================================================


def estimate_latency_from_flops(
    model_flops: int,
    hardware: HardwareTier,
    precision: str = "fp32",
    utilization_factor: float = 0.3,
) -> float:
    """
    Estimate inference latency from model FLOPs and hardware specs.

    Args:
        model_flops: Model FLOPs per inference.
        hardware: Hardware tier to run on.
        precision: Precision (fp32, fp16, int8).
        utilization_factor: Expected hardware utilization (0.3 = 30% of peak).

    Returns:
        Estimated latency in milliseconds.
    """
    effective_tflops = hardware.effective_tflops(precision) * utilization_factor
    effective_flops_per_sec = effective_tflops * 1e12

    if effective_flops_per_sec == 0:
        return float("inf")

    latency_sec = model_flops / effective_flops_per_sec
    return latency_sec * 1000  # Convert to ms


def select_hardware_tier_for_latency(
    model_flops: int,
    target_latency_ms: float,
    precision: str = "fp32",
    target: DeploymentTarget = DeploymentTarget.CLOUD_GPU,
    utilization_factor: float = 0.3,
) -> HardwareTier | None:
    """
    Select the cheapest hardware tier that meets latency requirements.

    Args:
        model_flops: Model FLOPs per inference.
        target_latency_ms: Maximum acceptable latency.
        precision: Precision (fp32, fp16, int8).
        target: Deployment target (cloud GPU, edge, etc.).
        utilization_factor: Expected hardware utilization.

    Returns:
        Cheapest HardwareTier that meets requirements, or None if none can.
    """
    candidates = list_hardware_tiers(target)

    for tier in candidates:  # Already sorted by cost
        latency = estimate_latency_from_flops(model_flops, tier, precision, utilization_factor)
        if latency <= target_latency_ms:
            return tier

    return None  # No tier can meet the latency requirement


def calculate_deployment_cost(
    model_flops: int,
    scenario: DeploymentScenario,
    model_memory_bytes: int = 0,
    utilization_factor: float = 0.3,
) -> DeploymentCostEstimate:
    """
    Calculate deployment cost for a model given a scenario.

    This is the main cost calculation function that:
    1. Selects appropriate hardware based on latency SLA
    2. Calculates number of instances needed for throughput
    3. Computes hourly, daily, monthly costs

    Args:
        model_flops: Model FLOPs per inference.
        scenario: DeploymentScenario with throughput/latency requirements.
        model_memory_bytes: Model memory footprint (for memory-based selection).
        utilization_factor: Expected hardware utilization (default 30%).

    Returns:
        DeploymentCostEstimate with all computed costs and recommendations.
    """
    warnings: list[str] = []

    # Step 1: Select hardware tier based on latency SLA
    if scenario.max_latency_ms:
        selected_tier = select_hardware_tier_for_latency(
            model_flops,
            scenario.max_latency_ms,
            scenario.precision,
            scenario.target,
            utilization_factor,
        )
        if selected_tier is None:
            # Fall back to most powerful tier
            tiers = list_hardware_tiers(scenario.target)
            selected_tier = tiers[-1] if tiers else list(HARDWARE_TIERS.values())[0]
            warnings.append(
                f"No hardware meets {scenario.max_latency_ms}ms latency SLA. "
                f"Using {selected_tier.name}."
            )
    else:
        # No latency constraint - pick cheapest tier that can run the model
        tiers = list_hardware_tiers(scenario.target)
        selected_tier = tiers[0] if tiers else list(HARDWARE_TIERS.values())[0]

    # Step 2: Calculate estimated latency and throughput
    estimated_latency = estimate_latency_from_flops(
        model_flops, selected_tier, scenario.precision, utilization_factor
    )
    single_instance_fps = 1000.0 / estimated_latency if estimated_latency > 0 else 0

    # Step 3: Calculate instances needed for target throughput
    if single_instance_fps > 0:
        instances_for_throughput = max(
            1,
            int(scenario.target_fps / single_instance_fps + 0.99),  # Round up
        )
    else:
        instances_for_throughput = 1
        warnings.append("Could not estimate throughput. Using 1 instance.")

    # Add replicas for availability
    total_instances = instances_for_throughput * scenario.replicas

    # Step 4: Check memory requirements
    if model_memory_bytes > 0:
        model_gb = model_memory_bytes / (1024**3)
        # Leave ~30% headroom for activations
        required_memory_gb = model_gb * 1.3
        if required_memory_gb > selected_tier.memory_gb:
            warnings.append(
                f"Model requires ~{model_gb:.1f}GB but {selected_tier.name} "
                f"has {selected_tier.memory_gb}GB. Consider larger tier."
            )

    # Step 5: Calculate costs
    cost_per_hour = selected_tier.cost_per_hour_usd * total_instances
    cost_per_day = cost_per_hour * scenario.hours_per_day
    cost_per_month = cost_per_day * scenario.days_per_month

    # Cost per 1000 inferences
    inferences_per_hour = single_instance_fps * 3600 * total_instances
    if inferences_per_hour > 0:
        cost_per_1k = (cost_per_hour / inferences_per_hour) * 1000
    else:
        cost_per_1k = 0

    # Step 6: Calculate utilization
    total_capacity_fps = single_instance_fps * total_instances
    utilization = (scenario.target_fps / total_capacity_fps * 100) if total_capacity_fps > 0 else 0

    # Check latency SLA
    meets_sla = True
    if scenario.max_latency_ms and estimated_latency > scenario.max_latency_ms:
        meets_sla = False
        warnings.append(
            f"Estimated latency ({estimated_latency:.1f}ms) exceeds "
            f"SLA ({scenario.max_latency_ms}ms)"
        )

    return DeploymentCostEstimate(
        scenario=scenario,
        hardware_tier=selected_tier,
        num_instances=total_instances,
        estimated_latency_ms=estimated_latency,
        estimated_throughput_fps=single_instance_fps * total_instances,
        meets_latency_sla=meets_sla,
        cost_per_hour_usd=cost_per_hour,
        cost_per_day_usd=cost_per_day,
        cost_per_month_usd=cost_per_month,
        utilization_percent=utilization,
        cost_per_1k_inferences_usd=cost_per_1k,
        warnings=warnings,
    )


def compare_deployment_costs(
    model_flops: int,
    scenarios: list[DeploymentScenario],
    model_memory_bytes: int = 0,
) -> list[DeploymentCostEstimate]:
    """
    Compare deployment costs across multiple scenarios.

    Useful for comparing different precision levels or deployment targets.

    Args:
        model_flops: Model FLOPs per inference.
        scenarios: List of deployment scenarios to compare.
        model_memory_bytes: Model memory footprint.

    Returns:
        List of DeploymentCostEstimate, one per scenario.
    """
    return [
        calculate_deployment_cost(model_flops, scenario, model_memory_bytes)
        for scenario in scenarios
    ]


def estimate_cost_from_combined_report(
    combined_report: Any,  # CombinedReport
    scenario: DeploymentScenario,
) -> DeploymentCostEstimate:
    """
    Calculate deployment cost from a CombinedReport.

    Extracts FLOPs and memory from the architecture summary and calculates cost.

    Args:
        combined_report: CombinedReport with architecture data.
        scenario: DeploymentScenario defining requirements.

    Returns:
        DeploymentCostEstimate with computed costs.
    """
    arch = combined_report.architecture
    flops = arch.get("flops_total", 0)
    memory = arch.get("model_size_bytes", 0)

    return calculate_deployment_cost(flops, scenario, memory)
