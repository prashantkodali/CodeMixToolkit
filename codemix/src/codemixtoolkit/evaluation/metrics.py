"""
Metrics module for evaluation.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sacrebleu.metrics import BLEU, CHRF


class EvaluationMetrics:
    """Class containing various evaluation metrics."""

    @staticmethod
    def classification_metrics(
        predictions: Union[List[int], np.ndarray],
        references: Union[List[int], np.ndarray],
        average: str = "weighted",
    ) -> Dict[str, float]:
        """Compute classification metrics.

        Args:
            predictions: List of predicted labels
            references: List of reference labels
            average: Averaging strategy for F1 score (default: weighted)

        Returns:
            Dictionary containing accuracy and F1 score
        """
        return {
            "accuracy": accuracy_score(references, predictions),
            "f1": f1_score(references, predictions, average=average),
        }

    @staticmethod
    def reference_based_generation_metrics(
        predictions: List[str],
        references: List[Union[str, List[str]]],
        tokenize: Optional[str] = None,
    ) -> Dict[str, float]:
        """Compute reference-based generation metrics.

        Args:
            predictions: List of generated texts
            references: List of reference texts or list of reference text lists
            tokenize: Tokenization method for BLEU (default: None)

        Returns:
            Dictionary containing BLEU and CHRF scores
        """
        # Convert single references to list format
        if references and isinstance(references[0], str):
            references = [[ref] for ref in references]

        # Initialize metrics
        bleu = BLEU()
        chrf = CHRF()

        # Compute BLEU score
        bleu_score = bleu.corpus_score(predictions, references, tokenize=tokenize)

        # Compute CHRF score
        chrf_score = chrf.corpus_score(predictions, references, tokenize=tokenize)

        return {
            "bleu": bleu_score.score,
            "chrf": chrf_score.score,
            "bleu_details": {
                "precisions": bleu_score.precisions,
                "bp": bleu_score.bp,
                "ratio": bleu_score.ratio,
                "sys_len": bleu_score.sys_len,
                "ref_len": bleu_score.ref_len,
            },
            "chrf_details": {
                "char_order": chrf_score.char_order,
                "word_order": chrf_score.word_order,
                "beta": chrf_score.beta,
            },
        }

    @staticmethod
    def compute_metrics(
        predictions: Union[List[str], List[int], np.ndarray],
        references: Union[
            List[str], List[Union[str, List[str]]], List[int], np.ndarray
        ],
        metric_type: str = "classification",
        **kwargs,
    ) -> Dict[str, Any]:
        """Compute metrics based on the type.

        Args:
            predictions: List of predictions
            references: List of references
            metric_type: Type of metrics to compute (classification or generation)
            **kwargs: Additional arguments for specific metrics

        Returns:
            Dictionary containing computed metrics
        """
        if metric_type == "classification":
            return EvaluationMetrics.classification_metrics(
                predictions, references, **kwargs
            )
        elif metric_type == "generation":
            return EvaluationMetrics.reference_based_generation_metrics(
                predictions, references, **kwargs
            )
        else:
            raise ValueError(f"Unsupported metric type: {metric_type}")
