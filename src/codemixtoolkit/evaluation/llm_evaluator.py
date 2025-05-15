"""
LLM evaluator for zero and few-shot prompting using LiteLLM.
"""

from typing import Any, Dict, List, Optional, Union

from datasets import Dataset, DatasetDict

# import litellm
from litellm import completion
from tqdm import tqdm

from ..config import config
from ..data.base import CodeMixDataset
from .base import BaseEvaluator
from .metrics import EvaluationMetrics


def make_openrouter_api_call(
    messages: List[Dict[str, str]],
    model: str = "openai/gpt-3.5-turbo",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    api_key: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Make an API call to OpenRouter using LiteLLM.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model: Model to use (default: openai/gpt-3.5-turbo)
        temperature: Temperature for generation (default: 0.7)
        max_tokens: Maximum tokens to generate (default: 1000)
        api_key: OpenRouter API key. If not provided, will try to get from OPENROUTER_API_KEY env var
        **kwargs: Additional arguments to pass to the API call

    Returns:
        Dictionary containing the API response

    Raises:
        ValueError: If no API key is provided and OPENROUTER_API_KEY is not set
    """
    # Get API key from argument or environment variable
    # api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    api_key = config.OPENROUTER_API_KEY

    if not api_key:
        raise ValueError(
            "OpenRouter API key must be provided either as an argument or through OPENROUTER_API_KEY environment variable"
        )

    # Set up the API call parameters
    params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "api_key": api_key,
        **kwargs,
    }

    # Make the API call using LiteLLM
    response = completion(**params)

    return {
        "content": response.choices[0].message.content,
        "usage": response.usage,
        "model": model,
        "finish_reason": response.choices[0].finish_reason,
    }


class LLMEvaluator(BaseEvaluator):
    """Evaluator for zero and few-shot prompting using LiteLLM."""

    def __init__(
        self,
        tasktype: str,
        task: str,
        name: str,
        model: str = "openrouter/meta-llama/llama-3.3-70b-instruct:free",  # default model - nocost endpoint
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
            name: Name of the evaluator
            model: Model to use for evaluation (default: gpt-3.5-turbo)
            temperature: Temperature for generation (default: 0.7)
            max_tokens: Maximum tokens to generate (default: 1000)
            instruction: Optional instruction to include in the prompt (default: None)
            few_shot_examples: List of few-shot examples (default: None)
            instruction_label_mapping: Mapping from model output strings to numeric labels
            dataset_label_mapping: Mapping from dataset label strings to numeric labels
        """
        super().__init__(name)
        self.TaskType = tasktype
        self.task = task
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.instruction = instruction
        self.few_shot_examples = few_shot_examples or []
        self.instruction_label_mapping = instruction_label_mapping or {}
        self.dataset_label_mapping = dataset_label_mapping or {}

    def _create_prompt(self, text: str) -> str:
        """Create the prompt for the LLM.

        Args:
            text: Input text

        Returns:
            Formatted prompt string
        """

        # ToDo: create the instruction part of the prompt onlyonce - add only the sample for each sample stage

        prompt = ""

        # Add instruction if provided
        if self.instruction:
            prompt += f"{self.instruction}\n\n"

        # Add few-shot examples if available
        if self.few_shot_examples:
            for example in self.few_shot_examples:
                prompt += f"Input: {example['input']}\n"
                prompt += f"Output: {example['output']}\n\n"

        # Add the current input
        prompt += f"Input: {text}\nOutput:"
        return prompt

    def _prediction_step(self, text: str, **kwargs) -> str:
        """Call the prediction from the LLM.

        Args:
            text: Input text

        """
        prompt = self._create_prompt(text)

        return prompt, completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs,
        )

    def evaluate_sample(
        self, text: str, label: Union[str, int], **kwargs
    ) -> Dict[str, Any]:
        """Evaluate a single text sample using the LLM.

        Args:
            text: Text to evaluate
            instruction: Optional instruction to override the default instruction
            **kwargs: Additional arguments for the LLM call

        Returns:
            Dictionary containing the LLM response and metadata
        """
        # Use provided instruction or fall back to default
        # current_instruction = (
        #     instruction if instruction is not None else self.instruction
        # )

        # # Create prompt with current instruction
        # prompt = ""
        # if current_instruction:
        #     prompt += f"{current_instruction}\n\n"

        # # Add few-shot examples if available
        # if self.few_shot_examples:
        #     for example in self.few_shot_examples:
        #         prompt += f"Input: {example['input']}\n"
        #         prompt += f"Output: {example['output']}\n\n"

        # # Add the current input
        # prompt += f"Input: {text}\nOutput:"

        # prompt = self._create_prompt(text)

        prompt, response = self._prediction_step(text, **kwargs)

        # response = completion(
        #     model=self.model,
        #     messages=[{"role": "user", "content": prompt}],
        #     temperature=self.temperature,
        #     max_tokens=self.max_tokens,
        #     **kwargs,
        # )

        print(f"Response: {response.choices[0].message.content}")

        prediction = self._convert_prediction_to_label(
            response.choices[0].message.content
        )
        print(f"Prediction with label mapping: {prediction}")
        if isinstance(label, str):
            ground_truth = self._convert_dataset_label(label)
        else:
            ground_truth = label

        if prediction == ground_truth:
            print(
                f"Correct Prediction! Prediction {prediction} matches ground truth {ground_truth}"
            )
        else:
            print(
                f"Incorrect Prediction! Prediction {prediction} does not match ground truth {ground_truth}"
            )

        return {
            "input": text,
            "prompt": prompt,
            "response": prediction,
            "ground_truth": ground_truth,
            "model": self.model,
            "usage": response.usage,
        }

    def _convert_prediction_to_label(self, prediction: str) -> int:
        """Convert model's string prediction to numeric label.

        Args:
            prediction: String prediction from the model

        Returns:
            Numeric label corresponding to the prediction
        """
        # Clean the prediction string
        prediction = prediction.strip().lower()

        # If the prediction is in the mapping, return the numeric label
        if prediction in self.instruction_label_mapping:
            return self.instruction_label_mapping[prediction]

        # Try to find the label in the mapping
        found_in = 0
        for label_str, label_num in self.instruction_label_mapping.items():
            if label_str.lower() in prediction:
                found_in += 1
        if found_in == 1:
            return int(label_num)
        elif found_in > 1:
            # raise ValueError(f"Prediction {prediction} found in multiple labels: {self.instruction_label_mapping.keys()}")
            # ToDo: convert printo to logger
            print(
                f"WARNING: Prediction {prediction} found in multiple labels: {self.instruction_label_mapping.keys()}"
            )
            return -100
        elif found_in == 0:
            # raise ValueError(f"Prediction {prediction} not found in any of the labels: {self.instruction_label_mapping.keys()}")
            # ToDo: convert printo to logger
            print(
                f"WARNING: Prediction {prediction} not found in any of the labels: {self.instruction_label_mapping.keys()}"
            )
            return -100
        else:
            # ToDo: convert printo to logger
            print(f"WARNING: Prediction {prediction} is not a valid label")

            return -100

        # If no mapping found, return -1 to indicate error
        return -1

    def _convert_dataset_label(self, label: str) -> int:
        """Convert dataset's string label to numeric label.

        Args:
            label: String label from the dataset

        Returns:
            Numeric label corresponding to the dataset label
        """
        # Clean the label string
        label = str(label).strip().lower()

        # Try to find the label in the mapping
        for label_str, label_num in self.dataset_label_mapping.items():
            if label_str.lower() == label:
                return int(label_num)

        # If no mapping found, return -1 to indicate error
        return -1

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
            **kwargs: Additional arguments for the LLM calls

        Returns:
            Dictionary containing evaluation results for all samples and metrics
        """
        results = []
        predictions = []
        true_labels = []

        # Load the dataset if not already loaded
        if dataset.data is None:
            dataset.load()

        if isinstance(dataset.data, DatasetDict) and eval_split in dataset.data:
            dataset_data = dataset.data[eval_split]
        elif isinstance(dataset.data, Dataset):
            raise ValueError(f"Split {eval_split} not found in dataset")

        if max_eval_samples is not None:
            dataset_data = dataset_data.select(range(max_eval_samples))

        # Handle different dataset types
        if isinstance(dataset_data, Dataset):
            # For HuggingFace datasets
            for item in tqdm(dataset_data, desc="Evaluating dataset"):
                text = item.get("text", item.get("input", item.get("sentence", "")))
                label = item.get("label", item.get("labels", None))

                if not text:
                    continue

                prompt, response = self._prediction_step(
                    text, instruction=instruction, **kwargs
                )
                results.append(response)

                # Convert predictions and labels to numeric values
                pred_label = response.choices[0].message.content
                pred_label = self._convert_prediction_to_label(pred_label)

                if isinstance(label, str):
                    true_label = self._convert_dataset_label(label)
                elif isinstance(label, int):
                    true_label = label
                else:
                    raise ValueError(f"Label type {type(label)} not supported")

                predictions.append(pred_label)
                true_labels.append(true_label)

        else:
            raise ValueError(f"Dataset type {type(dataset.data)} not supported")

        if self.TaskType == "sequence_classification":
            # Calculate metrics
            metrics = EvaluationMetrics.classification_metrics(predictions, true_labels)

            breakpoint()

        elif self.TaskType == "token_classification":
            raise NotImplementedError("Token classification metrics not implemented")

        elif self.TaskType == "translation":
            raise NotImplementedError("Translation metrics not implemented")

        elif self.TaskType == "text_generation":
            raise NotImplementedError("Text generation metrics not implemented")

        return {
            "results": results,
            "model": self.model,
            "instruction": instruction if instruction is not None else self.instruction,
            "num_samples": len(results),
            "metrics": metrics,
            "predictions": predictions,
            "true_labels": true_labels,
        }
