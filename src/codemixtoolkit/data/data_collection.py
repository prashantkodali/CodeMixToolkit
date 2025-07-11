from .base import CodeMixDataset, DatasetInfo, HFDatasetsReader, LanguagePair, TaskType

__DATASET_CLASSES_MAP = {}


def register_dataset(cls):
    __DATASET_CLASSES_MAP[cls.__name__] = cls
    return cls


@register_dataset
class AcceptabilityEnHiClineGCM(HFDatasetsReader):
    """Acceptability Dataset - Cline - GCM Set"""

    def __init__(self, **kwargs):
        super().__init__(
            dataset_info=DatasetInfo(
                name="cline-acceptability-gcm",
                source="prakod/cline_gcm",
                task_type=TaskType.SEQUENCE_REGRESSION,
                language_pair=LanguagePair.EN_HI,
                input_fields=["text"],
                label_fields=["average_rating", "sum_abs_diff"],
                metrics=["mae", "rmse"],
                description="Acceptability Dataset - Cline - GCM Set",
                reference="https://arxiv.org/abs/2405.05572",
            ),
            **kwargs,
        )


@register_dataset
class AcceptabilityEnHiClineOSN(HFDatasetsReader):
    """Acceptability Dataset - Cline - OSN Set"""

    def __init__(self, **kwargs):
        super().__init__(
            dataset_info=DatasetInfo(
                name="cline-acceptability-osn",
                source="prakod/cline_osn",
                task_type=TaskType.SEQUENCE_REGRESSION,
                language_pair=LanguagePair.EN_HI,
                input_fields=["text"],
                label_fields=["average_rating", "sum_abs_diff"],
                metrics=["mae", "rmse"],
                description="Acceptability Dataset - Cline - OSN Set",
                reference="https://arxiv.org/abs/2405.05572",
            ),
            **kwargs,
        )


@register_dataset
class ToDDialogXRISAWOZ(CodeMixDataset):
    """ToD Dialog XRISAWO Set"""

    def __init__(self, **kwargs):
        super().__init__(
            dataset_info=DatasetInfo(
                name="sentiment-en-hi-prabhu-et-al",
                source="prakod/sentiment_en_hi_prabhu_et_al",
                task_type=TaskType.SEQUENCE_CLASSIFICATION,
                language_pair=LanguagePair.EN_HI,
                input_fields=["text"],
                label_fields=["label"],
                metrics=["accuracy"],
                description="Sentiment Dataset - En-Hi - Prabhu et al.",
                reference="https://aclanthology.org/C16-1234/",
            ),
            **kwargs,
        )


@register_dataset
class SentimentEnHiPrabhuEtAl(HFDatasetsReader):
    """Sentiment Dataset - En-Hi - Prabhu et al."""

    def __init__(self, **kwargs):
        super().__init__(
            dataset_info=DatasetInfo(
                name="sentiment-en-hi-prabhu-et-al",
                source="prakod/sentiment_en_hi_prabhu_et_al",
                task_type=TaskType.SEQUENCE_CLASSIFICATION,
                language_pair=LanguagePair.EN_HI,
                input_fields=["text"],
                label_fields=["label"],
                metrics=["accuracy"],
                description="Sentiment Dataset - En-Hi - Prabhu et al.",
                reference="https://aclanthology.org/C16-1234/",
            ),
            **kwargs,
        )


@register_dataset
class SentimentEnHiGLUECOS(HFDatasetsReader):
    """Sentiment Dataset - En-Hi - GLUE COS Set"""

    def __init__(self, **kwargs):
        super().__init__(
            dataset_name="sentiment-en-hi-glue-cos",
            dataset_info=DatasetInfo(
                name="sentiment-en-hi-glue-cos",
                source="prakod/sentiment_en_hi_glue_cos",
                task_type=TaskType.SEQUENCE_CLASSIFICATION,
                language_pair=LanguagePair.EN_HI,
                input_fields=["text"],
                label_fields=["label"],
                metrics=["accuracy"],
                description="Sentiment Dataset - En-Hi - GLUE COS Set",
                reference="https://arxiv.org/abs/2405.05572",
            ),
            **kwargs,
        )


@register_dataset
class SentimentEnHiSentiMix(HFDatasetsReader):
    """Sentiment Dataset - En-Hi - SentiMix Set"""

    def __init__(self, **kwargs):
        super().__init__(
            dataset_name="sentiment-en-hi-senti-mix",
            dataset_info=DatasetInfo(
                name="sentiment-en-hi-senti-mix",
                source="prakod/sentiment_en_hi_senti_mix",
                task_type=TaskType.SEQUENCE_CLASSIFICATION,
                language_pair=LanguagePair.EN_HI,
                input_fields=["text"],
                label_fields=["label"],
                metrics=["accuracy"],
                description="Sentiment Dataset - En-Hi - SentiMix Set",
                reference="https://arxiv.org/abs/2405.05572",
            ),
            **kwargs,
        )


@register_dataset
class HateSpeechEnHiBohraEtAl(HFDatasetsReader):
    """Hate Speech Dataset - En-Hi - Bohra et al."""

    def __init__(self, **kwargs):
        super().__init__(
            dataset_name="hate-speech-en-hi-bohra-et-al",
            dataset_info=DatasetInfo(
                name="hate-speech-en-hi-bohra-et-al",
                source="prakod/hate_speech_enhi_bohraetal",
                task_type=TaskType.SEQUENCE_CLASSIFICATION,
                language_pair=LanguagePair.EN_HI,
                input_fields=["text"],
                label_fields=["label"],
                metrics=["accuracy"],
                description="Hate Speech Dataset - En-Hi - Bohra et al.",
                reference="https://arxiv.org/abs/2405.05572",
            ),
            **kwargs,
        )


# pun classification - https://github.com/Likhith-Asapu/Codemix-Pun-Generation/tree/main

# sarcasam

# humour

# pretraining corpus - acl paper

# l3cube


# # [Data Efficiency Commands](data_efficiency_ft/README.md)


# # Dataset Names/Paths
# - En-Hi
#     - Sentiment
#         - subword : prakod/subwordlstm_sentiment
#         - gluecos : prakod/gluecos_sentiment_enhi_with_testset || prakod/gluecos_sentiment
#         - sentimix : prakod/sentimix_semeval
#     - QA: prakod/gluecos_qa_enhi
#     - NLI: prakod/gluecos_nli_enhi
#     - Hate: prakod/hate_speech_enhi_bohraetal

# - En-Es
#     - Sentiment :
#         - Gluecos : prakod/sentiment_gluecos_enes_with_test_split
#         - Sentimix : prakod/sentimix_semeval_enes_with_test_split || prakod/sentiment_gluecos_enes


# - En-Ta
#     - Sentiment: prakod/sentiment_fire_enta
#     - Hate/Offense: prakod/offenseval_dravidian_tamil_binary

# - En-Ml
#     - Sentiment: prakod/sentiment_fire_enma
#     - Hate/Offense: prakod/offenseval_dravidian_malayalam_binary

# - En-Kn
#     - Hate/Offense: prakod/offenseval_dravidian_kannada_binary

# - En-Te
#     - Sentiment: prakod/sentiment_ente
#     - Hate/Offense: prakod/hate_ente


# # Model IDs
# - XLMR - FacebookAI/xlm-roberta-base
# - mbart - facebook/mbart-large-cc25
# - llama 1b - meta-llama/Llama-3.2-1B


# Sentiment Analysis Datasets
# @register_dataset
# class HateSpeechHIDataset(HFDatasetsReader):
#     """Hate Speech Hindi Dataset"""

#     def __init__(self, **kwargs):
#         super().__init__(
#             dataset_name="hate_speech_hindi",
#             dataset_info=DatasetInfo(
#                 name="hate_speech_hindi",
#                 task_type=TaskType.CLASSIFICATION,
#                 language_pair=LanguagePair.HI_EN,
#                 description="Hate speech detection dataset in Hindi-English code-mixed text",
#             ),
#             **kwargs,
#         )


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
# @register_dataset
# class WNUT17Dataset(HFDatasetsReader):
#     """WNUT 2017 Dataset"""

#     def __init__(self, **kwargs):
#         super().__init__(
#             dataset_name="wnut17",
#             dataset_info=DatasetInfo(
#                 name="wnut17",
#                 task_type=TaskType.NER,
#                 language_pair=LanguagePair.EN,
#                 description="Named Entity Recognition dataset from WNUT 2017",
#             ),
#             **kwargs,
#         )


# def get_dataset_classes() -> Dict[str, Type[CodeMixDataset]]:
#     """Get all registered dataset classes from this module"""
#     # Find all classes in this module that inherit from CodeMixDataset
#     dataset_classes = {
#         name: obj for name, obj in inspect.getmembers(sys.modules[__name__])
#         if inspect.isclass(obj) and
#         issubclass(obj, CodeMixDataset) and
#         obj != CodeMixDataset and
#         obj != HFDatasetsReader
#     }

#     return dataset_classes
