from .base import (
    DatasetInfo, 
    TaskType, 
    LanguagePair,
    CodeMixDataset,
    HFDatasetsReader
)
from .registry import DatasetRegistry

import pkgutil
import importlib

_DATASET_CLASSES_MAP = {}

# Iterate through all modules in this package
for loader, module_name, is_pkg in pkgutil.iter_modules(__path__):
    module = importlib.import_module(f".{module_name}", __name__)
    if hasattr(module, "__DATASET_CLASSES_MAP"):
        _DATASET_CLASSES_MAP.update(module.__DATASET_CLASSES_MAP)
        
DATASET_REGISTRY = DatasetRegistry(_DATASET_CLASSES_MAP)
    
__all__ = [
    'DatasetInfo',
    'TaskType',
    'LanguagePair',
    'CodeMixDataset',
    'HFDatasetsReader',
    'DatasetRegistry',
]

__all__.extend(list(_DATASET_CLASSES_MAP.keys()))

globals().update(_DATASET_CLASSES_MAP)

