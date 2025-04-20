from datasets import load_from_disk, Dataset, DatasetDict
import random
import pandas as pd

original_dataset = load_from_disk("dataset")

def eliminate_vowels(word):
    vowels = 'aeiou'
    return ''.join(c for c in word if c not in vowels)

def remove_duplicate_consonants(word):
    cleaned_word = []
    for i, char in enumerate(word):
        if i == 0 or char != word[i - 1]:
            cleaned_word.append(char)
    return ''.join(cleaned_word)

def generate_noisy_form(word, rule):
    if rule == 1:
        noisy_word = eliminate_vowels(word)
    elif rule == 2:
        noisy_word = remove_duplicate_consonants(word)
    return noisy_word

noisy_datasets = {}

for split in ["train", "test", "validation"]:
    noisy_samples = []

    split_df = original_dataset[split].to_pandas()

    for index, row in split_df.iterrows():
        word = row["CM_candidates_transliterated_indictrans"]
        sentence_length = len(word)
        num_characters_to_modify = max(1, sentence_length // 10)

        positions_to_modify = random.sample(range(sentence_length), num_characters_to_modify)
        noisy_word = list(word)
        for pos in positions_to_modify:
            if pos < sentence_length - 1 and word[pos] not in 'aeiou' and word[pos+1] not in 'aeiou':
                random_rule = 2
            else:
                random_rule = random.choice([1, 2])
            noisy_word[pos] = generate_noisy_form(word[pos], random_rule)
        noisy_word = ''.join(noisy_word)
        noisy_example = {key: row[key] for key in row.keys()}
        noisy_example["noisy_data"] = noisy_word
        noisy_samples.append(noisy_example)

    noisy_datasets[split] = Dataset.from_pandas(pd.DataFrame(noisy_samples))

noisy_data = DatasetDict({
    "train": noisy_datasets["train"],
    "validation": noisy_datasets["validation"],
    "test": noisy_datasets["test"]})

noisy_data.save_to_disk("noisy_dataset")
