import os
import random
import numpy as np
import torch
from transformers import set_seed
from dataclasses import dataclass, field
from typing import Optional
import re
import emoji


def seed_everything(SEED):
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(SEED)
    print(f"SEED : {SEED}")


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    do_regression: bool = field(
        default=None,
        metadata={
            "help": "Whether to do regression instead of classification. If None, will be inferred from the dataset."
        },
    )
    text_column_names: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the text column in the input dataset or a CSV/JSON file. "
                'If not specified, will use the "sentence" column for single/multi-label classification task.'
            )
        },
    )
    text_column_delimiter: Optional[str] = field(
        default=" ",
        metadata={
            "help": "THe delimiter to use to join text columns into a single sentence."
        },
    )
    train_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the train split in the input dataset. If not specified, will use the "train" split when do_train is enabled'
        },
    )
    validation_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the validation split in the input dataset. If not specified, will use the "validation" split when do_eval is enabled'
        },
    )
    test_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the test split in the input dataset. If not specified, will use the "test" split when do_predict is enabled'
        },
    )
    remove_splits: Optional[str] = field(
        default=None,
        metadata={
            "help": "The splits to remove from the dataset. Multiple splits should be separated by commas."
        },
    )
    remove_columns: Optional[str] = field(
        default=None,
        metadata={
            "help": "The columns to remove from the dataset. Multiple columns should be separated by commas."
        },
    )
    label_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the label column in the input dataset or a CSV/JSON file. "
                'If not specified, will use the "label" column for single/multi-label classification task'
            )
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    shuffle_train_dataset: bool = field(
        default=False, metadata={"help": "Whether to shuffle the train dataset or not."}
    )
    shuffle_seed: int = field(
        default=42,
        metadata={
            "help": "Random seed that will be used to shuffle the train dataset."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    metric_name: Optional[str] = field(
        default=None, metadata={"help": "The metric to use for evaluation."}
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the training data."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the validation data."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the test data."},
    )

    clean_dataset: Optional[bool] = field(
        default=False,
        metadata={"help": "clean the datasets using clean_text method in utils"},
    )

    job_name: Optional[str] = field(default=None, metadata={"help": "log jobname"})

    early_stopping_callback: Optional[bool] = field(
        default=False,
        metadata={
            "help": "whether or not to include early stopping callback in trainer"
        },
    )

    early_stopping_patience: Optional[int] = field(
        default=3, metadata={"help": "early stopping patience"}
    )

    early_stopping_threshold: Optional[float] = field(
        default=0.005, metadata={"help": "early stopping threshold"}
    )

    percentage_training_data: Optional[float] = field(
        default=None, metadata={"help": "percentage of training data to use"}
    )

    results_logging_file: Optional[str] = field(
        default=None, metadata={"help": "results logging file"}
    )

    def __post_init__(self):
        if self.dataset_name is None:
            if self.train_file is None or self.validation_file is None:
                raise ValueError(" training/validation file or a dataset name.")

            train_extension = self.train_file.split(".")[-1]
            assert train_extension in [
                "csv",
                "json",
            ], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={
            "help": "Will enable to load a pretrained model whose head dimensions are different."
        },
    )

    quantize_model: bool = field(default=False, metadata={"help": "quantize the model"})

    add_lora_adapter: bool = field(
        default=False, metadata={"help": "add lora adapter to the model"}
    )

    train_only_classifier_layer: bool = field(
        default=False,
        metadata={"help": "Freeze all layers apart from the head e.g classifier head"},
    )


def freeze_augumenting_model(model):
    for name, param in model.named_parameters():
        param.requires_grad = False

    print("Aug Model Frozen")
    return model


def freeze_anchor_model(model):
    if "bert" in model.config._name_or_path:
        for name, param in model.bert.named_parameters():
            param.requires_grad = False
        print("Anchor Model Frozen")
        return model
    elif "Phi" in model.config._name_or_path:
        for name, param in model.model.named_parameters():
            param.requires_grad = False
        print("Anchor Model Frozen")
        return model
    else:
        raise RuntimeError(
            "model is neither bert of phi class - this code doesn't support other models"
        )


def freeze_layers_train_only_head(model):
    # breakpoint()

    if model.config.model_type.lower() == "xlm-roberta":
        for name, param in model.roberta.named_parameters():
            param.requires_grad = False

        print("xlm-roberta layers frozen!")

        for name, param in model.roberta.named_parameters():
            print(name, param.requires_grad)

        return model

    elif model.config.model_type.lower() == "bert":
        for name, param in model.bert.named_parameters():
            param.requires_grad = False

        print("bert layers frozen!")

        for name, param in model.bert.named_parameters():
            print(name, param.requires_grad)

        return model


def get_num_trainable_parameters(model):
    """
    Get the number of trainable parameters. Taken from HF Transformers trainer.py
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _clean_text(text):
    # Removing the emojis
    cleaned_text = emoji.replace_emoji(text, replace="")

    # Removing the specified symbols
    pattern = r"[#_^+:]"
    cleaned_text = re.sub(pattern, "", cleaned_text)

    # Removing mentions
    cleaned_text = re.sub(r"@\s+\w+", "", cleaned_text)

    # Removing malformed and standard URLs
    url_pattern1 = re.compile(r"https?://\S+|www\.\S+")
    url_pattern2 = re.compile(r"https\s*/\s*/\s*\S+|https\s*/\s*/\s*\s*\S+")

    # Additional pattern to remove fragments
    fragment_pattern = re.compile(r"\s*/\s*\S*")
    # fragment_pattern2 = re.compile(r'\s*\.\sco\s*\S*')

    # Applying the patterns
    cleaned_text = url_pattern2.sub("", cleaned_text)
    cleaned_text = url_pattern1.sub("", cleaned_text)
    cleaned_text = fragment_pattern.sub("", cleaned_text)
    # cleaned_text = fragment_pattern2.sub('', cleaned_text)

    # excessive punctuations
    punct = r"[!\"#$%&\'()*+,\-.\/:;<=>?@\[\\\]^_`{|}~ ]{2,}"
    cleaned_text = re.sub(punct, " ", cleaned_text)

    return cleaned_text


def clean_text(batch):
    batch["text"] = [_clean_text(text) for text in batch["text"]]
    return batch


# @dataclass
# class ModelArguments:
#     augmenting_model_name_or_path: str = field(
#         default="google-bert/bert-base-multilingual-cased",
#         metadata={"help": "Path to the augmenting mBERT model"},
#     )
#     anchor_model_name_or_path: str = field(
#         default="google-bert/bert-base-multilingual-cased",
#         metadata={"help": "Path to the anchor mBERT model"},
#     )
#     augmenting_layers_mapping: Optional[dict] = field(
#         default=None, metadata={"help": "Mapping of augmenting layers"}
#     )
#     ma_layer_ratio: int = field(
#         default=3, metadata={"help": "Layer ratio for augmenting model"}
#     )
#     mb_layer_ratio: int = field(
#         default=3, metadata={"help": "Layer ratio for anchor model"}
#     )


# @dataclass
# class DataTrainingArguments:
#     dataset_name: Optional[str] = field(
#         default="imdb",
#         metadata={"help": "The name of the dataset to use (via the datasets library)."},
#     )
#     max_seq_length: int = field(
#         default=128,
#         metadata={
#             "help": "The maximum total input sequence length after tokenization."
#         },
#     )
#     batch_size: int = field(
#         default=16,
#         metadata={"help": "Batch size for training and evaluation."},
#     )
#     max_train_samples: Optional[int] = field(
#         default=None,
#         metadata={
#             "help": "For debugging purposes or quicker training, truncate the number of training examples to this value if set."
#         },
#     )
#     max_eval_samples: Optional[int] = field(
#         default=None,
#         metadata={
#             "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set."
#         },
#     )


# @dataclass
# class CustomTrainingArguments(TrainingArguments):
#     output_dir: str = field(
#         default="./results",
#         metadata={
#             "help": "The output directory where the model predictions and checkpoints will be written."
#         },
#     )
#     num_train_epochs: int = field(
#         default=1,
#         metadata={"help": "Total number of training epochs to perform."},
#     )
#     save_steps: int = field(
#         default=500,
#         metadata={"help": "Save checkpoint every X updates steps."},
#     )
#     logging_dir: str = field(
#         default="./logs",
#         metadata={"help": "Directory for storing logs."},
#     )
#     logging_steps: int = field(
#         default=100,
#         metadata={"help": "Log every X updates steps."},
#     )


# @dataclass
# class ModelArguments:
#     """
#     Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
#     """

#     model_name_or_path: Optional[str] = field(
#         default=None,
#         metadata={
#             "help": "The model checkpoint for weights initialization."
#             "Don't set if you want to train a model from scratch."
#         },
#     )

#     augmenting_model_name_or_path: Optional[str] = field(
#         default=None,
#         metadata={
#             "help": "The model checkpoint for weights initialization for augmenting model (ma)."
#         },
#     )

#     quantize_model_a: bool = field(default=False, metadata={"help": "Quantize model A"})

#     anchor_model_name_or_path: Optional[str] = field(
#         default=None,
#         metadata={
#             "help": "The model checkpoint for weights initialization for anchor model (mb)."
#         },
#     )
#     quantize_model_b: bool = field(default=False, metadata={"help": "Quantize model B"})

#     augmenting_layers_mapping: Optional[str] = field(
#         default=None,
#         metadata={
#             "help": "Dictionary of m_a and m_b layer mappings. Ex : {6:6, 10:10}"
#         },
#     )

#     augmenting_layers_ratio: Optional[str] = field(
#         default=None,
#         metadata={
#             "help": "Ratio of layers mapping between m_a and m_b - i:j , every ith layer of m_a maps to jth layer of m_b. num_layers in m_a, m_b should be divisible by i and j, respectively. Ex: 1:2"
#         },
#     )

#     tokenizer_name: Optional[str] = field(
#         default=None,
#         metadata={
#             "help": "Pretrained tokenizer name or path if not the same as model_name"
#         },
#     )

#     augmenting_tokenizer_name: Optional[str] = field(
#         default=None,
#         metadata={
#             "help": "Pretrained tokenizer name or path if not the same as model_name"
#         },
#     )

#     anchor_tokenizer_name: Optional[str] = field(
#         default=None,
#         metadata={
#             "help": "Pretrained tokenizer name or path if not the same as model_name"
#         },
#     )

#     cache_dir: Optional[str] = field(
#         default=None,
#         metadata={
#             "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
#         },
#     )
#     use_fast_tokenizer: bool = field(
#         default=True,
#         metadata={
#             "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
#         },
#     )
#     log_dir: Optional[str] = field(
#         default=None,
#         metadata={"help": "Where do you want to store the log files"},
#     )

#     token: str = field(
#         default=None,
#         metadata={
#             "help": (
#                 "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
#                 "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
#             )
#         },
#     )

#     use_peft: bool = field(
#         default=False,
#         metadata={"help": "Use PEFT"},
#     )

#     use_qlora: bool = field(
#         default=False,
#         metadata={"help": "Use QLORA PEFT"},
#     )

#     batch_size: Optional[int] = field(
#         default=16, metadata={"help": "Batch size for training and evaluation."}
#     )

#     test_train_ratio: Optional[float] = field(
#         default=0.3, metadata={"help": "Ratio of test set size to train set size."}
#     )

#     freeze_ma: Optional[bool] = field(
#         default=True, metadata={"help": "if flag set to True freeze ma"}
#     )

#     freeze_mb: Optional[bool] = field(
#         default=True, metadata={"help": "if flag set to True freeze mb"}
#     )

#     combine_cross_attention: Optional[bool] = field(
#         default=True,
#         metadata={
#             "help": "if set to False cross attention from m_a will not be added to mb"
#         },
#     )

#     weight_mb_layer_output: Optional[float] = field(
#         default=1,
#         metadata={
#             "help": "weight for mb layer output while doing weighted sum of mb layer output and cross attention ouput "
#         },
#     )

#     weight_cross_attn: Optional[float] = field(
#         default=1,
#         metadata={
#             "help": "weight for ma layer output while doing weighted sum of ma layer output and cross attention ouput "
#         },
#     )

#     pre_norm: Optional[bool] = field(
#         default=False,
#         metadata={
#             "help": "If True, add additional norm to attn output before add and norm"
#         },
#     )

#     add_and_norm: Optional[bool] = field(
#         default=False,
#         metadata={"help": "If True, add_and_norm like the BertOutput layer"},
#     )

#     dense_layer_in_cross_attention: Optional[bool] = field(
#         default=False,
#         metadata={"help": "If True, add dense_layer in cross_attention"},
#     )


# @dataclass
# class DataTrainingArguments:
#     """
#     Arguments pertaining to what data we are going to input our model for training and eval.
#     """

#     dataset_name: Optional[str] = field(
#         default=None,
#         metadata={"help": "The name of the dataset to use (via the datasets library)."},
#     )
#     dataset_config_name: Optional[str] = field(
#         default=None,
#         metadata={
#             "help": "The configuration name of the dataset to use (via the datasets library)."
#         },
#     )
#     train_file: Optional[str] = field(
#         default=None, metadata={"help": "The input training data file (a text file)."}
#     )
#     validation_file: Optional[str] = field(
#         default=None,
#         metadata={
#             "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
#         },
#     )
#     test_file: Optional[str] = field(
#         default=None,
#         metadata={
#             "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
#         },
#     )
#     max_train_samples: Optional[int] = field(
#         default=None,
#         metadata={
#             "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
#             "value if set."
#         },
#     )
#     max_eval_samples: Optional[int] = field(
#         default=None,
#         metadata={
#             "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
#             "value if set."
#         },
#     )
#     max_seq_length: Optional[int] = field(
#         default=512,
#         metadata={
#             "help": "The maximum total input sequence length after tokenization. Sequences longer "
#             "than this will be truncated."
#         },
#     )

#     overwrite_cache: bool = field(
#         default=False,
#         metadata={"help": "Overwrite the cached training and evaluation sets"},
#     )
#     validation_split_percentage: Optional[int] = field(
#         default=5,
#         metadata={
#             "help": "The percentage of the train set used as validation set in case there's no validation split"
#         },
#     )
#     preprocessing_num_workers: Optional[int] = field(
#         default=None,
#         metadata={"help": "The number of processes to use for the preprocessing."},
#     )

#     task_type: str = field(
#         default=None,
#         metadata={
#             "help": "define Task type for appropriate model init - sequence_classification, token_classification, generation"
#         },
#     )

#     clean_dataset: Optional[bool] = field(
#         default=False,
#         metadata={"help": "clean the datasets using clean_text method in utils"},
#     )

#     job_name: Optional[str] = field(default=None, metadata={"help": "log jobname"})

#     early_stopping_callback: Optional[bool] = field(
#         default=False,
#         metadata={
#             "help": "whether or not to include early stopping callback in trainer"
#         },
#     )

#     early_stopping_patience: Optional[int] = field(
#         default=3, metadata={"help": "early stopping patience"}
#     )

#     early_stopping_threshold: Optional[float] = field(
#         default=0.005, metadata={"help": "early stopping threshold"}
#     )

#     label_column_name: Optional[str] = field(
#         default=None,
#         metadata={
#             "help": (
#                 "The name of the label column in the input dataset or a CSV/JSON file. "
#                 'If not specified, will use the "label" column for single/multi-label classification task'
#             )
#         },
#     )

#     text_column_names: Optional[str] = field(
#         default=None,
#         metadata={
#             "help": (
#                 "The name of the text column in the input dataset or a CSV/JSON file. "
#                 'If not specified, will use the "sentence" column for single/multi-label classification task.'
#             )
#         },
#     )
