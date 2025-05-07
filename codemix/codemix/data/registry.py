from .base import CodeMixDataset, TaskType, LanguagePair
from typing import Dict, List, Type


class DatasetRegistry:
    """Registry for dataset classes"""

    def __init__(self, dataset_registry: Dict[str, Type[CodeMixDataset]]):
        self._dataset_classes: Dict[str, Type[CodeMixDataset]] = dataset_registry

    def __repr__(self):
        # ToDo: print number of datasets, and datasets names not class names
        # ToDo: print number of datasets for each task type
        # ToDo: print number of datasets for each language pair
        return f"DatasetRegistry({self._dataset_classes})"

    def get_dataset_class(self, name: str) -> Type[CodeMixDataset]:
        """Get a dataset class by name"""
        if name not in self._dataset_classes:
            raise KeyError(f"Dataset '{name}' not found in registry")
        return self._dataset_classes[name]

    def create_dataset(self, name: str, **kwargs) -> CodeMixDataset:
        """Create a dataset instance by name"""
        dataset_class = self.get_dataset_class(name)
        return dataset_class(**kwargs)

    def list_datasets(self) -> List[str]:
        """List all available datasets"""
        return list(self._dataset_classes.keys())

    def get_datasets_by_task(self, task_type: TaskType) -> List[str]:
        """Get all dataset names for a specific task type"""
        return [
            name
            for name, cls in self._dataset_classes.items()
            if cls().get_info().task_type == task_type
        ]

    def get_datasets_by_languagepair(self, language_pair: LanguagePair) -> List[str]:
        """Get all dataset names for a specific language pair"""
        return DatasetRegistry(
            {
                name: cls
                for name, cls in self._dataset_classes.items()
                if cls().get_info().language_pair == language_pair
            }
        )


# # Create a singleton instance
# registry = DatasetRegistry()
