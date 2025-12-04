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
    # Adapters
    "parse_ultralytics_val",
    "load_ultralytics_json",
    "parse_generic_json",
    "load_generic_json",
    "parse_generic_csv",
    "load_generic_csv",
    "detect_and_parse",
]
