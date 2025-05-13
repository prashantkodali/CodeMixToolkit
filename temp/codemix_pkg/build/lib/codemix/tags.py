from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from codemix.compute import CodeMixAnalyzer

class Tags:
    def __init__(self):
        self.analyzer = CodeMixAnalyzer(model_path="ai4bharat/IndicNER", tokenizer_path="ai4bharat/IndicNER")
        self.analyzer.load_model()

    def get_lid_tags(self, sentence):
        tokens, cmi_value, combined_labels = self.analyzer.unicode_LID_get_sentence_cmi(sentence)
        return tokens, combined_labels
    
    def analyze(self, sentence):
        tokens, lid_tags = self.get_lid_tags(sentence)
        return tokens, lid_tags