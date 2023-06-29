from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from compute import CodeMixAnalyzer

tokenizer_path = "ai4bharat/IndicNER"
model_path = "ai4bharat/IndicNER"

analyzer = CodeMixAnalyzer(model_path=model_path, tokenizer_path=tokenizer_path)
analyzer.load_model()

def get_lid_tags(sentence):
    cmi_value, combined_labels = analyzer.unicode_LID_get_sentence_cmi(sentence)
    return combined_labels

sentence = "मैं Hyderabaed मैं movie देखने जा रहा हूँ"
lid_tags = get_lid_tags(sentence)

print("LID tags:", lid_tags)