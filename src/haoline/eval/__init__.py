"""
HaoLine Eval Import Module

Import evaluation results from external tools (Ultralytics, HF evaluate, etc.)
and combine with architecture analysis.
"""

from .schemas import (
    EvalMetric,
    EvalResult,
    DetectionEvalResult,
    ClassificationEvalResult,
    NLPEvalResult,
    LLMEvalResult,
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

