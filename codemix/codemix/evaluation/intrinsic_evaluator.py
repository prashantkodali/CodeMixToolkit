"""
Perplexity evaluator for computing perplexity on code-mix text.
"""

from typing import Any, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

from .base import BaseEvaluator
from ..data.base import CodeMixDataset


class PerplexityEvaluator(BaseEvaluator):
    """Evaluator for computing perplexity on code-mix text."""

    def __init__(
        self,
        name: str,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32,
    ):
        """Initialize the perplexity evaluator.

        Args:
            name: Name of the evaluator
            model_path: Path to the language model
            device: Device to run the model on (default: cuda if available, else cpu)
            batch_size: Batch size for evaluation (default: 32)
        """
        super().__init__(name)
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.model.to(device)
        self.model.eval()

    def _compute_perplexity(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> float:
        """Compute perplexity for a single sequence.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Perplexity score
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
            )
            loss = outputs.loss
            perplexity = torch.exp(loss)

        return perplexity.item()

    def evaluate_sample(self, text: str, **kwargs) -> Dict[str, Any]:
        """Evaluate a single text sample by computing perplexity.

        Args:
            text: Text to evaluate
            **kwargs: Additional arguments for the tokenizer

        Returns:
            Dictionary containing perplexity score and metadata
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, **kwargs
        ).to(self.device)

        perplexity = self._compute_perplexity(
            inputs["input_ids"], inputs["attention_mask"]
        )

        return {
            "input": text,
            "perplexity": perplexity,
        }

    def evaluate_dataset(self, dataset: CodeMixDataset, **kwargs) -> Dict[str, Any]:
        """Evaluate the dataset by computing perplexity for each sample.

        Args:
            dataset: CodeMixDataset to evaluate on
            **kwargs: Additional arguments for the tokenizer

        Returns:
            Dictionary containing perplexity scores and statistics
        """
        results = []
        texts = []

        # Load the dataset if not already loaded
        if dataset.data is None:
            dataset.load()

        # Handle different dataset types
        if isinstance(dataset.data, Dataset):
            # For HuggingFace datasets
            for item in dataset.data:
                text = item.get("text", item.get("input", item.get("sentence", "")))
                if not text:
                    continue
                texts.append(text)
        else:
            # For other dataset types (e.g., pandas DataFrame)
            for item in dataset.data:
                if isinstance(item, dict):
                    text = item.get("text", item.get("input", item.get("sentence", "")))
                elif isinstance(item, str):
                    text = item
                else:
                    continue

                if not text:
                    continue

                texts.append(text)

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]

            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                **kwargs,
            ).to(self.device)

            # Process each sample in the batch individually
            for j, text in enumerate(batch_texts):
                input_ids = inputs["input_ids"][j : j + 1]
                attention_mask = inputs["attention_mask"][j : j + 1]

                perplexity = self._compute_perplexity(input_ids, attention_mask)

                results.append(
                    {
                        "input": text,
                        "perplexity": perplexity,
                    }
                )

        # Compute statistics
        perplexities = [r["perplexity"] for r in results]
        avg_perplexity = sum(perplexities) / len(perplexities)

        return {
            "results": results,
            "model_path": self.model_path,
            "num_samples": len(results),
            "average_perplexity": avg_perplexity,
            "min_perplexity": min(perplexities),
            "max_perplexity": max(perplexities),
        }
