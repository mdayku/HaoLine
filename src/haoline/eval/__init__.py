"""
HaoLine Eval Import Module

Import evaluation results from external tools and combine with architecture analysis.

Supported adapters:
- Ultralytics YOLO (detection)
- HuggingFace evaluate (classification/NLP)
- lm-eval-harness (LLM benchmarks)
- timm (image classification)
- Generic CSV/JSON
"""

from .adapters import (
    detect_and_parse,
    load_generic_csv,
    load_generic_json,
    load_hf_evaluate,
    load_lm_eval,
    load_timm_benchmark,
    load_ultralytics_json,
    parse_generic_csv,
    parse_generic_json,
    parse_hf_evaluate,
    parse_lm_eval,
    parse_timm_benchmark,
    parse_ultralytics_val,
)
from .schemas import (
    ClassificationEvalResult,
    CombinedReport,
    DetectionEvalResult,
    EvalMetric,
    EvalResult,
    GenericEvalResult,
    LLMEvalResult,
    NLPEvalResult,
    SegmentationEvalResult,
    TaskType,
    compute_model_hash,
    create_combined_report,
    get_combined_report_schema,
    get_eval_schema,
    is_valid_task_type,
    link_eval_to_model,
    validate_eval_result,
)

__all__ = [
    # Schemas
    "EvalMetric",
    "EvalResult",
    "DetectionEvalResult",
    "ClassificationEvalResult",
    "NLPEvalResult",
    "LLMEvalResult",
    "SegmentationEvalResult",
    "GenericEvalResult",
    "CombinedReport",
    "TaskType",
    # Schema utilities
    "get_eval_schema",
    "get_combined_report_schema",
    "validate_eval_result",
    "is_valid_task_type",
    # Linking utilities
    "compute_model_hash",
    "link_eval_to_model",
    "create_combined_report",
    # Adapters - Ultralytics
    "parse_ultralytics_val",
    "load_ultralytics_json",
    # Adapters - HuggingFace evaluate
    "parse_hf_evaluate",
    "load_hf_evaluate",
    # Adapters - lm-eval-harness
    "parse_lm_eval",
    "load_lm_eval",
    # Adapters - timm
    "parse_timm_benchmark",
    "load_timm_benchmark",
    # Adapters - Generic
    "parse_generic_json",
    "load_generic_json",
    "parse_generic_csv",
    "load_generic_csv",
    # Auto-detect
    "detect_and_parse",
]
