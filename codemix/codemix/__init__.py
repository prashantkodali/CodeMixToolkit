"""
CodeMix Toolkit - A toolkit for code-mixed language processing
"""

from .data import (
    DatasetInfo,
    TaskType,
    LanguagePair,
    CodeMixDataset,
    HFDatasetsReader,
    DatasetRegistry,
    DATASET_REGISTRY,
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
]

# __all__.extend(list(DATASET_REGISTRY.keys()))

# # Optionally bind them globally by name
# globals().update(DATASET_REGISTRY)
