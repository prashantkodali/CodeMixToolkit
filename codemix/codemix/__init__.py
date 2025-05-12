"""
CodeMix Toolkit - A toolkit for code-mixed language processing
"""

from .config import config
from .data import (
    DATASET_REGISTRY,
    CodeMixDataset,
    DatasetInfo,
    DatasetRegistry,
    HFDatasetsReader,
    LanguagePair,
    TaskType,
)

__version__ = "0.1.0"

__all__ = [
    "DatasetInfo",
    "TaskType",
    "LanguagePair",
    "CodeMixDataset",
    "HFDatasetsReader",
    "DatasetRegistry",
    "DATASET_REGISTRY",
    "config",  # Expose the config instance
    "set_env_file",  # Expose set_env_file function
]
