"""
HaoLine Privacy Utilities.

Functions for redacting sensitive information from model analysis reports.
"""

from __future__ import annotations

from typing import Any


def create_name_mapping(names: set[str]) -> dict[str, str]:
    """
    Create a deterministic mapping from original names to anonymized names.

    Args:
        names: Set of original names to anonymize.

    Returns:
        Dictionary mapping original names to anonymized names.
    """
    # Sort for deterministic ordering
    sorted_names = sorted(names)

    mapping: dict[str, str] = {}
    counters: dict[str, int] = {}

    for name in sorted_names:
        # Determine the prefix based on naming patterns
        prefix = _infer_prefix(name)
        count = counters.get(prefix, 0) + 1
        counters[prefix] = count
        mapping[name] = f"{prefix}_{count:04d}"

    return mapping


def _infer_prefix(name: str) -> str:
    """Infer an anonymized prefix based on the original name pattern."""
    name_lower = name.lower()

    # Common ONNX/model patterns
    if any(x in name_lower for x in ["conv", "cnn"]):
        return "conv"
    if any(x in name_lower for x in ["bn", "batchnorm", "batch_norm"]):
        return "bn"
    if any(x in name_lower for x in ["relu", "gelu", "silu", "activation"]):
        return "act"
    if any(x in name_lower for x in ["fc", "linear", "dense", "gemm", "matmul"]):
        return "linear"
    if any(x in name_lower for x in ["attention", "attn", "self_attn"]):
        return "attn"
    if any(x in name_lower for x in ["embed", "embedding"]):
        return "embed"
    if any(x in name_lower for x in ["norm", "layernorm", "layer_norm"]):
        return "norm"
    if any(x in name_lower for x in ["pool", "avgpool", "maxpool"]):
        return "pool"
    if any(x in name_lower for x in ["reshape", "view", "flatten"]):
        return "reshape"
    if any(x in name_lower for x in ["concat", "cat"]):
        return "concat"
    if any(x in name_lower for x in ["add", "sum"]):
        return "add"
    if any(x in name_lower for x in ["mul", "multiply"]):
        return "mul"
    if any(x in name_lower for x in ["split", "chunk"]):
        return "split"
    if any(x in name_lower for x in ["transpose", "permute"]):
        return "transpose"
    if any(x in name_lower for x in ["weight", "bias", "param"]):
        return "param"
    if any(x in name_lower for x in ["input", "inp"]):
        return "input"
    if any(x in name_lower for x in ["output", "out"]):
        return "output"

    # Default
    return "node"


def collect_names_from_dict(data: dict[str, Any]) -> set[str]:
    """
    Recursively collect all string values that look like layer/tensor names.

    Args:
        data: Dictionary to scan (typically from report.to_dict()).

    Returns:
        Set of potential names to anonymize.
    """
    names: set[str] = set()
    _collect_names_recursive(data, names)
    return names


def _collect_names_recursive(obj: Any, names: set[str], key: str = "") -> None:
    """Recursively collect names from nested structures."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            # Keys that typically contain names
            if k in (
                "name",
                "node_name",
                "layer_name",
                "tensor_name",
                "op_name",
                "input_name",
                "output_name",
            ):
                if isinstance(v, str):
                    names.add(v)
            # Keys that map names to values
            elif k in (
                "by_node",
                "by_name",
                "input_shapes",
                "output_shapes",
                "shared_weights",
            ):
                if isinstance(v, dict):
                    names.update(v.keys())
            # Lists like largest_weights, largest_activations
            elif k in ("largest_weights", "largest_activations"):
                if isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict) and "name" in item:
                            names.add(item["name"])
                        elif isinstance(item, (list, tuple)) and len(item) >= 1:
                            if isinstance(item[0], str):
                                names.add(item[0])

            _collect_names_recursive(v, names, k)

    elif isinstance(obj, list):
        for item in obj:
            _collect_names_recursive(item, names, key)


def redact_dict(
    data: dict[str, Any],
    mapping: dict[str, str],
) -> dict[str, Any]:
    """
    Apply name redaction to a dictionary (typically from report.to_dict()).

    Args:
        data: Dictionary to redact.
        mapping: Mapping from original names to anonymized names.

    Returns:
        New dictionary with names replaced.
    """
    result = _redact_recursive(data, mapping)
    # _redact_recursive always returns a dict when given a dict
    assert isinstance(result, dict)
    return result


def _redact_recursive(obj: Any, mapping: dict[str, str]) -> Any:
    """Recursively apply redaction to nested structures."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            # Replace keys if they're in the mapping (for by_node, etc.)
            new_key = mapping.get(k, k) if isinstance(k, str) else k
            result[new_key] = _redact_recursive(v, mapping)
        return result

    elif isinstance(obj, list):
        return [_redact_recursive(item, mapping) for item in obj]

    elif isinstance(obj, str):
        # Replace string values if they match a name
        return mapping.get(obj, obj)

    else:
        return obj


def create_summary_only_dict(data: dict[str, Any]) -> dict[str, Any]:
    """
    Strip a report dictionary to summary-only (no per-layer details).

    Args:
        data: Full report dictionary.

    Returns:
        Stripped dictionary with only aggregate stats.
    """
    # Fields to keep (aggregate only)
    keep_fields = {
        "metadata",
        "generated_at",
        "autodoc_version",
        "architecture_type",
    }

    # Nested fields to summarize
    summary_fields = {
        "graph_summary": ["num_nodes", "num_inputs", "num_outputs", "op_type_counts"],
        "param_counts": ["total", "trainable", "non_trainable", "is_quantized"],
        "flop_counts": ["total"],
        "memory_estimates": ["weights_bytes", "activations_bytes", "total_bytes"],
    }

    result: dict[str, Any] = {}

    # Copy allowed fields
    for field in keep_fields:
        if field in data:
            result[field] = data[field]

    # Extract summary from nested fields
    for field, allowed_keys in summary_fields.items():
        if field in data and data[field]:
            result[field] = {k: data[field][k] for k in allowed_keys if k in data[field]}

    # Add aggregate risk info without details
    if "risk_signals" in data and data["risk_signals"]:
        result["risk_summary"] = {
            "total_risks": len(data["risk_signals"]),
            "high": sum(1 for r in data["risk_signals"] if r.get("severity") == "high"),
            "medium": sum(1 for r in data["risk_signals"] if r.get("severity") == "medium"),
            "low": sum(1 for r in data["risk_signals"] if r.get("severity") == "low"),
        }

    # Add detected block counts without names
    if "detected_blocks" in data and data["detected_blocks"]:
        block_counts: dict[str, int] = {}
        for block in data["detected_blocks"]:
            block_type = block.get("block_type", "unknown")
            block_counts[block_type] = block_counts.get(block_type, 0) + 1
        result["detected_block_counts"] = block_counts

    # Add hardware summary without per-op breakdown
    if "hardware_estimates" in data and data["hardware_estimates"]:
        hw = data["hardware_estimates"]
        result["hardware_estimates"] = {
            k: hw[k]
            for k in [
                "latency_ms",
                "throughput_samples_per_sec",
                "estimated_power_w",
                "bottleneck_summary",
            ]
            if k in hw
        }

    return result
