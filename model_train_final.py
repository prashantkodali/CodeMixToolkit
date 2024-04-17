# modules required - datasets==2.11 transformers==4.28.0 seqeval, apt install git-lfs accelerate==0.20.1 -U

import transformers
from huggingface_hub import login
from transformers.utils import send_example_telemetry
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import load_dataset, load_metric, ClassLabel, Sequence, load_from_disk
import random
import pandas as pd
import accelerate, torch
import os
import wandb

print("hello")

os.environ['TRANSFORMERS_CACHE'] = '/scratch/ashna.dua/cache'
os.environ['HF_DATASETS_CACHE'] = '/scratch/ashna.dua/cache'

login(token='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', add_to_git_credential=True)

send_example_telemetry("token_classification_notebook", framework="pytorch")

task = "lid"
model_checkpoint = "ai4bharat/indic-bert"
batch_size = 10

print("Downloading data")
datasets = load_dataset("ashnadua01/data_small")

labels = ['acro', 'en', 'univ', 'hi', 'ne', 'ml']
label_list = labels

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

example = datasets["train"][4]

tokenized_input = tokenizer(example["tokens"], is_split_into_words=True)
tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])

word_ids = tokenized_input.word_ids()
aligned_labels = []

for i in word_ids:
    if i is None:
        aligned_labels.append(-100)
    else:
        if i < len(example["lid_tags"]):
            aligned_labels.append(example["lid_tags"][i])
        else:
            aligned_labels.append(-100)

label_all_tokens = True

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",  # Add padding to ensure consistent lengths
        return_offsets_mapping=True  # Return offset mappings for aligning labels
    )

    labels = []
    for i, label in enumerate(examples["lid_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = [-100] * len(tokenized_inputs['input_ids'][i])  # Initialize with -100 for padding tokens

        for word_idx, (start, end) in enumerate(tokenized_inputs['offset_mapping'][i]):
            # Check if the word's start and end positions are within the sentence (excluding padding tokens)
            if start is not None and end is not None:
                # Map word position to label index
                if word_idx < len(label):
                    label_ids[word_idx] = label[word_idx]
                else:
                    label_ids[word_idx] = -100  # Assign -100 for tokens without labels

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenize_and_align_labels(datasets['train'][:5])
# print(tokenize_and_align_labels)
tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)

model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))
model_name = model_checkpoint.split("/")[-1]

args = TrainingArguments(
    "/scratch/ashna.dua/model_indicbert_small",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
    save_steps=1000,
    save_total_limit=5,
    report_to="wandb",
    run_name="initial"
)

data_collator = DataCollatorForTokenClassification(tokenizer)
metric = load_metric("seqeval")

labels = [label_list[i] for i in example["lid_tags"]]
metric.compute(predictions=[labels], references=[labels])

import numpy as np

import numpy as np

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] if p != -1 else 'ml' for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] if l != -1 else 'ml' for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
	"precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()

predictions, labels, _ = trainer.predict(tokenized_datasets["validation"])
predictions = np.argmax(predictions, axis=2)

# Remove ignored index (special tokens)
true_predictions = [
    [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]
true_labels = [
    [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

results = metric.compute(predictions=true_predictions, references=true_labels)

# not sure
results

trainer.push_to_hub()
