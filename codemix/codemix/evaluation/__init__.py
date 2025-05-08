"""
Evaluation module for CodeMix Toolkit.

This module provides various evaluation capabilities:
- Zero and few-shot prompting evaluation using LiteLLM
- Fine-tuned model evaluation
- Intrinsic evaluation (perplexity) on code-mix text
- Classification and generation metrics
"""

from .base import BaseEvaluator
from .llm_evaluator import LLMEvaluator
from .model_evaluator import FineTunedModelEvaluator
from .intrinsic_evaluator import PerplexityEvaluator
from .metrics import ClassificationMetrics, GenerationMetrics, compute_metrics

__version__ = "0.1.0"

__all__ = [
    "BaseEvaluator",
    "LLMEvaluator",
    "FineTunedModelEvaluator",
    "PerplexityEvaluator",
    "ClassificationMetrics",
    "GenerationMetrics",
    "compute_metrics",
]
