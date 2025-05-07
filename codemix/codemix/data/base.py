from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Union 
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset


class LanguagePair(Enum):
    """Enum for different language pairs"""

    EN_HI = "en_hi"
    EN_ES = "en_es"
    EN_TE = "en_te"
    EN_TA = "en_ta"
    EN_MA = "en_ma"
    EN_BN = "en_bn"


class TaskType(Enum):
    """Enum for different types of tasks"""

    SEQUENCE_CLASSIFICATION = "sequence_classification"
    SEQUENCE_REGRESSION = "sequence_regression"
    TOKEN_CLASSIFICATION = "token_classification"
    TRANSLATION = "translation"
    GENERATION = "generation"
    # Add more task types as needed


@dataclass
class DatasetInfo:
    """Class to store dataset information"""
    name: str
    source: str  # 'huggingface' or 'local'
    task_type: TaskType
    language_pair: LanguagePair
    input_fields: Optional[List[str]] = None
    label_fields: Optional[List[str]] = None
    metrics: Optional[List[str]] = None
    description: Optional[str] = None
    reference: Optional[str] = None


class CodeMixDataset:
    """Base class for dataset loading"""

    def __init__(self, dataset_info: Optional[DatasetInfo] = None):
        """
        Initialize dataset loader

        Args:
            dataset_info: Optional DatasetInfo object from registry
        """
        self.data = None
        self.dataset_info = dataset_info

    def load(self) -> Union[Dataset, DatasetDict, pd.DataFrame]:
        """Load the dataset"""
        raise NotImplementedError

    def get_info(self) -> Optional[DatasetInfo]:
        """Get dataset information if available"""
        return self.dataset_info


class HFDatasetsReader(CodeMixDataset):
    """Loader for HuggingFace datasets"""

    def __init__(
        self,
        dataset_info: DatasetInfo,
        dataset_config: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize HuggingFace dataset loader

        Args:
            dataset_name: Name of the dataset on HuggingFace Hub
            dataset_config: Optional dataset configuration
            split: Optional split to load (e.g. 'train', 'test')
            dataset_info: Optional DatasetInfo object from registry
            **kwargs: Additional arguments to pass to load_dataset
        """
        super().__init__(dataset_info)
        self.dataset_config = dataset_config
        self.kwargs = kwargs

    def __repr__(self) -> str:
        """String representation of the dataset"""
        return f"HFDatasetsReader(dataset_name={self.dataset_name}, \
        dataset_info={self.dataset_info},)"

    def load(self) -> Union[Dataset, DatasetDict]:
        """Load the dataset from HuggingFace"""
        self.data = load_dataset(
            self.dataset_info.source, 
            self.dataset_config, 
            **self.kwargs
        )
        return self.data
