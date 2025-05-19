"""
LLM evaluator for zero and few-shot prompting using LiteLLM.
"""

from typing import Any, Dict, List, Optional, Union

from ..data.base import CodeMixDataset
from ..models.models import LLMPromptModel
from .base import BaseEvaluator
from .metrics import EvaluationMetrics


class Evaluator(BaseEvaluator):
    """Evaluator for zero and few-shot prompting using LiteLLM."""

    def __init__(
        self,
        tasktype: str,
        task: str,
        name: str,
        model: str = "openrouter/meta-llama/llama-3.3-70b-instruct:free",
        temperature: float = 0.1,
        max_tokens: int = 10,
        instruction: Optional[str] = None,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        instruction_label_mapping: Optional[Dict[str, str]] = None,
        dataset_label_mapping: Optional[Dict[str, str]] = None,
        dataset_obj: Optional[CodeMixDataset] = None,
    ):
        """Initialize the LLM evaluator.

        Args:
            tasktype: Type of task (e.g., sequence_classification)
            task: Name of the task
            name: Name of the evaluator
            model: Model to use for evaluation (default: openrouter/meta-llama/llama-3.3-70b-instruct:free)
            temperature: Temperature for generation (default: 0.1)
            max_tokens: Maximum tokens to generate (default: 10)
            instruction: Optional instruction to include in the prompt (default: None)
            few_shot_examples: List of few-shot examples (default: None)
            instruction_label_mapping: Mapping from model output strings to numeric labels
            dataset_label_mapping: Mapping from dataset label strings to numeric labels
            dataset_obj: Optional dataset object for evaluation
        """
        super().__init__(name)
        self.TaskType = tasktype
        self.task = task

        # Initialize the LLM model
        self.model = LLMPromptModel(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            instruction=instruction,
            few_shot_examples=few_shot_examples,
            instruction_label_mapping=instruction_label_mapping,
            dataset_label_mapping=dataset_label_mapping,
        )

    def evaluate_sample(
        self, text: str, label: Union[str, int], **kwargs
    ) -> Dict[str, Any]:
        """Evaluate a single text sample using the LLM.

        Args:
            text: Text to evaluate
            label: Ground truth label
            **kwargs: Additional arguments for the LLM call

        Returns:
            Dictionary containing the evaluation results
        """
        # Get prediction from the model
        result = self.model.predict(text, **kwargs)

        # Convert label if it's a string
        if isinstance(label, str):
            ground_truth = self.model._convert_dataset_label(label)
        else:
            ground_truth = label

        # Print prediction results
        print(f"Response: {result['raw_response']}")
        print(f"Prediction with label mapping: {result['prediction']}")

        if result["prediction"] == ground_truth:
            print(
                f"Correct Prediction! Prediction {result['prediction']} matches ground truth {ground_truth}"
            )
        else:
            print(
                f"Incorrect Prediction! Prediction {result['prediction']} does not match ground truth {ground_truth}"
            )

        return {
            "input": text,
            "prompt": result["prompt"],
            "response": result["prediction"],
            "ground_truth": ground_truth,
            "model": self.model.model,
            "usage": result["usage"],
        }

    def evaluate_dataset(
        self,
        dataset: CodeMixDataset,
        instruction: Optional[str] = None,
        max_eval_samples: Optional[int] = None,
        eval_split: Optional[str] = "test",
        **kwargs,
    ) -> Dict[str, Any]:
        """Evaluate the dataset using the LLM.

        Args:
            dataset: CodeMixDataset to evaluate on
            instruction: Optional instruction to override the default instruction
            max_eval_samples: Maximum number of samples to evaluate
            eval_split: Dataset split to evaluate on (default: test)
            **kwargs: Additional arguments for the LLM calls

        Returns:
            Dictionary containing evaluation results for all samples and metrics
        """
        # Load the dataset if not already loaded
        if dataset.data is None:
            dataset.load()

        # Get predictions from the model
        prediction_results = self.model.predict_dataset(
            dataset=dataset.data,
            max_samples=max_eval_samples,
            eval_split=eval_split,
            **kwargs,
        )

        # Calculate metrics based on task type
        if self.TaskType == "sequence_classification":
            metrics = EvaluationMetrics.classification_metrics(
                prediction_results["predictions"],
                prediction_results["true_labels"],
            )
        elif self.TaskType == "token_classification":
            raise NotImplementedError("Token classification metrics not implemented")
        elif self.TaskType == "translation":
            raise NotImplementedError("Translation metrics not implemented")
        elif self.TaskType == "text_generation":
            raise NotImplementedError("Text generation metrics not implemented")

        return {
            "results": prediction_results["results"],
            "model": self.model.model,
            "instruction": instruction
            if instruction is not None
            else self.model.instruction,
            "num_samples": prediction_results["num_samples"],
            "metrics": metrics,
            "predictions": prediction_results["predictions"],
            "true_labels": prediction_results["true_labels"],
        }
