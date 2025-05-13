"""
Base evaluator class that defines the interface for all evaluators.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from ..data.base import CodeMixDataset


class BaseEvaluator(ABC):
    """Base class for all evaluators in the CodeMix Toolkit."""

    def __init__(self, name: str):
        """Initialize the evaluator.

        Args:
            name: Name of the evaluator
        """
        self.name = name

    @abstractmethod
    def evaluate_dataset(self, dataset: CodeMixDataset, **kwargs) -> Dict[str, Any]:
        """Evaluate the dataset.

        Args:
            dataset: CodeMixDataset to evaluate on
            **kwargs: Additional arguments specific to the evaluator

        Returns:
            Dictionary containing evaluation metrics
        """
        pass

    @abstractmethod
    def evaluate_sample(self, text: str, **kwargs) -> Dict[str, Any]:
        """Evaluate a single text sample.

        Args:
            text: Text to evaluate
            **kwargs: Additional arguments specific to the evaluator

        Returns:
            Dictionary containing evaluation results
        """
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
