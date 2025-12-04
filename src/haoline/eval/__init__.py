"""
HaoLine Eval Import Module

Import evaluation results from external tools (Ultralytics, HF evaluate, etc.)
and combine with architecture analysis.
"""

from .adapters import (
    detect_and_parse,
    load_generic_csv,
    load_generic_json,
    load_ultralytics_json,
    parse_generic_csv,
    parse_generic_json,
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
    get_combined_report_schema,
    get_eval_schema,
    is_valid_task_type,
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
    # Adapters
    "parse_ultralytics_val",
    "load_ultralytics_json",
    "parse_generic_json",
    "load_generic_json",
    "parse_generic_csv",
    "load_generic_csv",
    "detect_and_parse",
]
