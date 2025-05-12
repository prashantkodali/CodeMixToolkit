"""
Evaluation module for CodeMix Toolkit.

This module provides various evaluation capabilities:
- Zero and few-shot prompting evaluation using LiteLLM
- Fine-tuned model evaluation
- Intrinsic evaluation (perplexity) on code-mix text
- Classification and generation metrics
"""

from .base import BaseEvaluator
from .intrinsic_evaluator import PerplexityEvaluator
from .llm_evaluator import LLMEvaluator
from .metrics import EvaluationMetrics

__version__ = "0.1.0"

__all__ = [
    "BaseEvaluator",
    "LLMEvaluator",
    "PerplexityEvaluator",
    "EvaluationMetrics",
    "EvaluationMetrics",
]
