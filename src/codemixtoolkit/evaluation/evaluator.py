"""
LLM evaluator for zero and few-shot prompting using LiteLLM.
"""

from typing import Any, Dict, List, Optional, Union

from ..data.base import CodeMixDataset
from ..models.models import BaseModel
from .base import BaseEvaluator
from .metrics import EvaluationMetrics


class Evaluator(BaseEvaluator):
    """Evaluator for zero and few-shot prompting using LiteLLM."""

    def __init__(
        self,
        tasktype: str,
        task: str,
        name: str,
        model: BaseModel,
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
            model: Model instance to use for evaluation (must be a subclass of BaseModel)
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
        self.model = model

    def evaluate_sample(
        self, text: str, label: Union[str, int], **kwargs
    ) -> Dict[str, Any]:
        """Evaluate a single text sample using the model.

        Args:
            text: Text to evaluate
            label: Ground truth label
            **kwargs: Additional arguments for the model call

        Returns:
            Dictionary containing the evaluation results and metrics
        """
        # Get prediction from the model
        result = self.model.predict(text, **kwargs)
        prediction = result["prediction"]

        # Convert label if it's a string
        if isinstance(label, str):
            ground_truth = self.model._convert_dataset_label(label)
        else:
            ground_truth = label

        # Compute metrics based on task type
        metrics = {}
        if self.TaskType == "sequence_classification":
            metrics = EvaluationMetrics.classification_metrics(
                [prediction], [ground_truth]
            )
        elif self.TaskType == "token_classification":
            # For token classification, we expect prediction and ground_truth to be lists
            metrics = EvaluationMetrics.token_classification_metrics(
                [prediction], [ground_truth]
            )
        elif self.TaskType == "translation":
            metrics = EvaluationMetrics.translation_metrics(
                [prediction], [ground_truth]
            )
        elif self.TaskType == "text_generation":
            metrics = EvaluationMetrics.generation_metrics([prediction], [ground_truth])

        return {
            "input": text,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "metrics": metrics,
            "model": getattr(self.model, "model", str(type(self.model).__name__)),
        }

    def evaluate_dataset(
        self,
        dataset: CodeMixDataset,
        max_eval_samples: Optional[int] = None,
        eval_split: Optional[str] = "test",
        **kwargs,
    ) -> Dict[str, Any]:
        """Evaluate the dataset using the model.

        Args:
            dataset: CodeMixDataset to evaluate on
            max_eval_samples: Maximum number of samples to evaluate
            eval_split: Dataset split to evaluate on (default: test)
            **kwargs: Additional arguments for the model calls

        Returns:
            Dictionary containing:
            - samples: List of input samples
            - predictions: List of model predictions
            - labels: List of ground truth labels
            - metrics: Dictionary of evaluation metrics
            - model: Model identifier
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

        predictions = prediction_results["predictions"]
        true_labels = prediction_results["true_labels"]
        samples = prediction_results["samples"]

        # Calculate metrics based on task type
        if self.TaskType == "sequence_classification":
            metrics = EvaluationMetrics.classification_metrics(
                predictions,
                true_labels,
            )
        elif self.TaskType == "token_classification":
            metrics = EvaluationMetrics.token_classification_metrics(
                predictions,
                true_labels,
            )
        elif self.TaskType == "translation":
            metrics = EvaluationMetrics.translation_metrics(
                predictions,
                true_labels,
            )
        elif self.TaskType == "text_generation":
            metrics = EvaluationMetrics.generation_metrics(
                predictions,
                true_labels,
            )

        return {
            "samples": samples,
            "predictions": predictions,
            "labels": true_labels,
            "metrics": metrics,
            "model": getattr(self.model, "model", str(type(self.model).__name__)),
        }


# # Using LLMPromptModel
# llm_model = LLMPromptModel(
#     model="openrouter/meta-llama/llama-3.3-70b-instruct:free",
#     temperature=0.1,
#     max_tokens=10,
#     instruction="Your instruction here"
# )
# evaluator = Evaluator(
#     tasktype="sequence_classification",
#     task="sentiment_analysis",
#     name="llm_evaluator",
#     model=llm_model
# )

# # Or using PoSTagger
# pos_model = PoSTagger(model_path="path/to/model", tokenizer_path="path/to/tokenizer")
# evaluator = Evaluator(
#     tasktype="token_classification",
#     task="pos_tagging",
#     name="pos_evaluator",
#     model=pos_model
# )

# Example usage of Evaluator with different models for sentence classification

# 1. Using LLMPromptModel for sentiment analysis
"""
from codemixtoolkit.models.models import LLMPromptModel
from codemixtoolkit.evaluation.evaluator import Evaluator

# Initialize LLM model for sentiment analysis
llm_model = LLMPromptModel(
    model="openrouter/meta-llama/llama-3.3-70b-instruct:free",
    temperature=0.1,
    max_tokens=10,
    instruction="Classify the sentiment of the following text as positive or negative:",
    instruction_label_mapping={"positive": 1, "negative": 0},
    dataset_label_mapping={"positive": 1, "negative": 0}
)

# Create evaluator
evaluator = Evaluator(
    tasktype="sequence_classification",
    task="sentiment_analysis",
    name="llm_sentiment_evaluator",
    model=llm_model
)

# Evaluate single sample
result = evaluator.evaluate_sample(
    text="I really enjoyed this movie, it was fantastic!",
    label="positive"
)

# Evaluate dataset
dataset_results = evaluator.evaluate_dataset(
    dataset=your_dataset,  # Your CodeMixDataset instance
    max_eval_samples=100
)
"""

# 2. Using PoSTagger for part-of-speech classification
"""
from codemixtoolkit.models.models import PoSTagger
from codemixtoolkit.evaluation.evaluator import Evaluator

# Initialize POS tagging model
pos_model = PoSTagger(
    model_path="path/to/pos/model",
    tokenizer_path="path/to/pos/tokenizer"
)
pos_model.load_model_tokenizer()

# Create evaluator
evaluator = Evaluator(
    tasktype="sequence_classification",
    task="pos_tagging",
    name="pos_evaluator",
    model=pos_model
)

# Evaluate single sample
result = evaluator.evaluate_sample(
    text="The quick brown fox jumps over the lazy dog",
    label=["DET", "ADJ", "ADJ", "NOUN", "VERB", "PREP", "DET", "ADJ", "NOUN"]
)

# Evaluate dataset
dataset_results = evaluator.evaluate_dataset(
    dataset=your_dataset,  # Your CodeMixDataset instance
    max_eval_samples=100
)
"""

# 3. Using NERtagger for named entity classification
"""
from codemixtoolkit.models.models import NERtagger
from codemixtoolkit.evaluation.evaluator import Evaluator

# Initialize NER model
ner_model = NERtagger(
    model_path="path/to/ner/model",
    tokenizer_path="path/to/ner/tokenizer",
    finegrain_labels=True
)
ner_model.load_model_tokenizer()

# Create evaluator
evaluator = Evaluator(
    tasktype="sequence_classification",
    task="named_entity_recognition",
    name="ner_evaluator",
    model=ner_model
)

# Evaluate single sample
result = evaluator.evaluate_sample(
    text="Apple Inc. is headquartered in Cupertino, California",
    label=["B-ORG", "I-ORG", "O", "O", "B-LOC", "I-LOC"]
)

# Evaluate dataset
dataset_results = evaluator.evaluate_dataset(
    dataset=your_dataset,  # Your CodeMixDataset instance
    max_eval_samples=100
)
"""

# 4. Using UnicodeLIDtagger for language identification
"""
from codemixtoolkit.models.models import UnicodeLIDtagger
from codemixtoolkit.evaluation.evaluator import Evaluator

# Initialize LID model
lid_model = UnicodeLIDtagger()

# Create evaluator
evaluator = Evaluator(
    tasktype="sequence_classification",
    task="language_identification",
    name="lid_evaluator",
    model=lid_model
)

# Evaluate single sample
result = evaluator.evaluate_sample(
    text="नमस्ते दुनिया",  # Hindi text
    label=["hi", "hi"]  # Hindi labels
)

# Evaluate dataset
dataset_results = evaluator.evaluate_dataset(
    dataset=your_dataset,  # Your CodeMixDataset instance
    max_eval_samples=100
)
"""

# 5. Using CSNLILIDClient for language identification
"""
from codemixtoolkit.models.models import CSNLILIDClient
from codemixtoolkit.evaluation.evaluator import Evaluator

# Initialize CSNLI LID client
csnli_model = CSNLILIDClient(base_url="http://localhost:6000")

# Create evaluator
evaluator = Evaluator(
    tasktype="sequence_classification",
    task="language_identification",
    name="csnli_lid_evaluator",
    model=csnli_model
)

# Evaluate single sample
result = evaluator.evaluate_sample(
    text="Hello world",
    label="en"  # English label
)

# Evaluate dataset
dataset_results = evaluator.evaluate_dataset(
    dataset=your_dataset,  # Your CodeMixDataset instance
    max_eval_samples=100
)
"""
