"""
HaoLine Eval Import Module

Import evaluation results from external tools (Ultralytics, HF evaluate, etc.)
and combine with architecture analysis.
"""

from .schemas import (
    ClassificationEvalResult,
    DetectionEvalResult,
    EvalMetric,
    EvalResult,
    LLMEvalResult,
    NLPEvalResult,
    SegmentationEvalResult,
)

__all__ = [
    "EvalMetric",
    "EvalResult",
    "DetectionEvalResult",
    "ClassificationEvalResult",
    "NLPEvalResult",
    "LLMEvalResult",
    "SegmentationEvalResult",
]
