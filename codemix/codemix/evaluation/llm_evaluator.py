"""
LLM evaluator for zero and few-shot prompting using LiteLLM.
"""

from typing import Any, Dict, List, Optional

# import litellm
from litellm import completion
from datasets import Dataset

from .base import BaseEvaluator
from ..data.base import CodeMixDataset


class LLMEvaluator(BaseEvaluator):
    """Evaluator for zero and few-shot prompting using LiteLLM."""

    def __init__(
        self,
        name: str,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        instruction: Optional[str] = None,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
    ):
        """Initialize the LLM evaluator.

        Args:
            name: Name of the evaluator
            model: Model to use for evaluation (default: gpt-3.5-turbo)
            temperature: Temperature for generation (default: 0.7)
            max_tokens: Maximum tokens to generate (default: 1000)
            instruction: Optional instruction to include in the prompt (default: None)
            few_shot_examples: List of few-shot examples (default: None)
        """
        super().__init__(name)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.instruction = instruction
        self.few_shot_examples = few_shot_examples or []

    def _create_prompt(self, text: str) -> str:
        """Create the prompt for the LLM.

        Args:
            text: Input text

        Returns:
            Formatted prompt string
        """
        prompt = ""

        # Add instruction if provided
        if self.instruction:
            prompt += f"Instruction: {self.instruction}\n\n"

        # Add few-shot examples if available
        if self.few_shot_examples:
            for example in self.few_shot_examples:
                prompt += f"Input: {example['input']}\n"
                prompt += f"Output: {example['output']}\n\n"

        # Add the current input
        prompt += f"Input: {text}\nOutput:"
        return prompt

    def evaluate_sample(
        self, text: str, instruction: Optional[str] = None, **kwargs
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
        current_instruction = (
            instruction if instruction is not None else self.instruction
        )

        # Create prompt with current instruction
        prompt = ""
        if current_instruction:
            prompt += f"Instruction: {current_instruction}\n\n"

        # Add few-shot examples if available
        if self.few_shot_examples:
            for example in self.few_shot_examples:
                prompt += f"Input: {example['input']}\n"
                prompt += f"Output: {example['output']}\n\n"

        # Add the current input
        prompt += f"Input: {text}\nOutput:"

        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs,
        )

        return {
            "input": text,
            "instruction": current_instruction,
            "prompt": prompt,
            "response": response.choices[0].message.content,
            "model": self.model,
            "usage": response.usage,
        }

    def evaluate_dataset(
        self, dataset: CodeMixDataset, instruction: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """Evaluate the dataset using the LLM.

        Args:
            dataset: CodeMixDataset to evaluate on
            instruction: Optional instruction to override the default instruction
            **kwargs: Additional arguments for the LLM calls

        Returns:
            Dictionary containing evaluation results for all samples
        """
        results = []

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
                result = self.evaluate_sample(text, instruction=instruction, **kwargs)
                results.append(result)
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

                result = self.evaluate_sample(text, instruction=instruction, **kwargs)
                results.append(result)

        return {
            "results": results,
            "model": self.model,
            "instruction": instruction if instruction is not None else self.instruction,
            "num_samples": len(results),
        }
