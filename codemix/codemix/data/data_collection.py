from typing import Dict, Type
import inspect
import sys
from .base import HFDatasetsReader, DatasetInfo, TaskType, LanguagePair, CodeMixDataset


__DATASET_CLASSES_MAP = {}

def register_dataset(cls):
    __DATASET_CLASSES_MAP[cls.__name__] = cls
    return cls

# Sentiment Analysis Datasets
@register_dataset
class HateSpeechHIDataset(HFDatasetsReader):
    """Hate Speech Hindi Dataset"""
    def __init__(self, **kwargs):
        super().__init__(
            dataset_name="hate_speech_hindi",
            dataset_info=DatasetInfo(
                name="hate_speech_hindi",
                task_type=TaskType.CLASSIFICATION,
                language_pair=LanguagePair.HI_EN,
                description="Hate speech detection dataset in Hindi-English code-mixed text"
            ),
            **kwargs
        )

# class LocalSentimentDataset(CSVDatasetLoader):
#     """Local Sentiment Dataset"""
#     def __init__(self, **kwargs):
#         super().__init__(
#             file_path="data/local_sentiment.csv",
#             dataset_info=DatasetInfo(
#                 name="local_sentiment",
#                 task_type=TaskType.CLASSIFICATION,
#                 language_pair=LanguagePair.HI_EN,
#                 description="Sentiment analysis dataset in Hindi-English code-mixed text"
#             ),
#             **kwargs
#         )

# NER Datasets
@register_dataset
class WNUT17Dataset(HFDatasetsReader):
    """WNUT 2017 Dataset"""
    def __init__(self, **kwargs):
        super().__init__(
            dataset_name="wnut17",
            dataset_info=DatasetInfo(
                name="wnut17",
                task_type=TaskType.NER,
                language_pair=LanguagePair.EN,
                description="Named Entity Recognition dataset from WNUT 2017"
            ),
            **kwargs
        )

def get_dataset_classes() -> Dict[str, Type[CodeMixDataset]]:
    """Get all registered dataset classes from this module"""
    # Find all classes in this module that inherit from CodeMixDataset
    dataset_classes = {
        name: obj for name, obj in inspect.getmembers(sys.modules[__name__])
        if inspect.isclass(obj) and
        issubclass(obj, CodeMixDataset) and
        obj != CodeMixDataset and
        obj != HFDatasetsReader
    }
    
    return dataset_classes 