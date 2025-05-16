from typing import Dict, List, Optional, Union, Any
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Base class for all models in the toolkit.

    This class provides common functionality for prediction and dataset processing
    that can be inherited by specific model implementations.
    """

    def __init__(self):
        """Initialize the base model."""
        pass

    @abstractmethod
    def predict(self, text: str, **kwargs) -> Dict[str, Any]:
        """Make a prediction for a single input text.

        Args:
            text: Input text to process
            **kwargs: Additional arguments for the prediction

        Returns:
            Dictionary containing the prediction results and metadata
        """
        pass

    def predict_dataset(
        self,
        dataset: Union[Dataset, DatasetDict, List],
        text_column: str = "text",
        label_column: str = "label",
        max_samples: Optional[int] = None,
        eval_split: Optional[str] = "test",
        **kwargs,
    ) -> Dict[str, Any]:
        """Helper method to process a dataset and make predictions.

        This is a protected method that can be used by subclasses to implement
        their predict_dataset method.

        Args:
            dataset: HuggingFace Dataset or DatasetDict to predict on
            text_column: Name of the column containing input text (default: "text")
            label_column: Name of the column containing labels (default: "label")
            max_samples: Maximum number of samples to predict (default: None)
            eval_split: Dataset split to use if dataset is DatasetDict (default: "test")
            **kwargs: Additional arguments for the predictions

        Returns:
            Dictionary containing:
            - results: List of prediction results for each sample
            - predictions: List of predictions
            - true_labels: List of true labels
            - num_samples: Number of samples processed
        """

        results = []
        predictions = []
        true_labels = []

        if isinstance(dataset, (DatasetDict, Dataset)):
            # Handle DatasetDict
            if isinstance(dataset, DatasetDict):
                if eval_split not in dataset:
                    raise ValueError(f"Split {eval_split} not found in dataset")
                dataset_data = dataset[eval_split]
            else:
                dataset_data = dataset

            # Limit samples if specified
            if max_samples is not None:
                dataset_data = dataset_data.select(range(max_samples))

            # Process each sample
            for item in tqdm(dataset_data):
                # Get text and label
                text = item.get(
                    text_column, item.get("input", item.get("sentence", ""))
                )
                label = item.get(label_column, item.get("labels", None))

                if not text:
                    continue

                # Get prediction
                result = self.predict(text, **kwargs)
                results.append(result)

                # Store prediction and label
                predictions.append(result.get("prediction"))
                true_labels.append(label)

        else:
            raise ValueError(
                f"Dataset type {type(dataset)} not supported. Dataset should be a HuggingFace Dataset or DatasetDict"
            )

        return {
            "results": results,
            "predictions": predictions,
            "true_labels": true_labels,
            "num_samples": len(results),
        }
