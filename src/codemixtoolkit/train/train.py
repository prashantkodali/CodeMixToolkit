#!/usr/bin/env python
# coding=utf-8
"""
Main training script that supports multiple training types:
- clm (Causal Language Modeling)
- mlm (Masked Language Modeling)
- token_classification (Token Classification/NER)
- text_classification (Text Classification)
"""

import logging
import sys
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


@dataclass
class CommandArguments:
    """
    Arguments for selecting the training command type.
    """

    command: str = field(
        default=None,
        metadata={
            "help": "The type of training to perform",
            "choices": ["clm", "mlm", "token_classification", "text_classification"],
        },
    )

    def __post_init__(self):
        # Get allowed commands from the field's metadata
        allowed_commands = self.__dataclass_fields__["command"].metadata["choices"]
        if self.command not in allowed_commands:
            raise ValueError(
                f"Unknown value for --command: {self.command}. "
                f"Value for --command argument should be one of: {allowed_commands}"
            )


def main():
    command_arg = None
    command_index = -1
    for i, arg in enumerate(sys.argv):
        if arg == "--command" and i + 1 < len(sys.argv):
            command_arg = sys.argv[i + 1]
            command_index = i
            break

    if command_arg is None:
        raise ValueError(
            "No command argument provided. Please provide a command argument, e.g 'python train.py --command text_classification ...'"
        )

    # Remove the --command argument and its value from sys.argv
    if command_index != -1:
        sys.argv.pop(command_index)  # Remove --command
        sys.argv.pop(command_index)  # Remove the command value

    # Create CommandArguments object - this will validate the command against choices
    command_args = CommandArguments(command=command_arg)

    # Import the appropriate training script based on the command
    if command_args.command == "clm":
        raise NotImplementedError("CLM is not implemented yet")
        from run_clm import main as train_main
    elif command_args.command == "mlm":
        raise NotImplementedError("MLM is not implemented yet")
        from run_mlm_xfmer_4_49_0 import main as train_main

    elif command_args.command == "token_classification":
        raise NotImplementedError("Token classification is not implemented yet")
        from run_ner_xfmer_4_49_0 import main as train_main
    elif command_args.command == "text_classification":
        from run_classification import main as train_main
    else:
        raise ValueError(f"Unknown command: {command_args.command}")

    # Run the selected training script with all remaining arguments
    train_main()


if __name__ == "__main__":
    main()


# python train.py --command text_classification \
#         --model_name_or_path "FacebookAI/xlm-roberta-base" \
#         --dataset_name "prakod/hate_speech_enhi_bohraetal" \
#         --percentage_training_data 100 \
#         --run_name "test" \
#         --seed 42 \
#         --learning_rate 1e-5 \
#         --num_train_epochs 10 \
#         --report_to wandb \
#         --shuffle_train_dataset \
#         --text_column_name "text" \
#         --text_column_delimiter "\n" \
#         --label_column_name "label" \
#         --do_train \
#         --do_eval \
#         --max_seq_length 512 \
#         --per_device_train_batch_size 1 \
#         --gradient_accumulation_steps 2 \
#         --output_dir . \
#         --train_only_classifier_layer True \
#         --clean_dataset True \
#         --add_lora_adapter False \
#         --early_stopping_callback True \
#         --early_stopping_patience 3 \
#         --early_stopping_threshold 0.005 \
#         --load_best_model_at_end True \
#         --save_strategy steps --save_steps 10 \
#         --eval_strategy steps --eval_steps 10 \
#         --metric_for_best_model "f1" \
#         --overwrite_output_dir \
#         --results_logging_file test.log
