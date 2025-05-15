from typing import Tuple, List, Any
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from codemix.compute import CodeMixAnalyzer


class Tags:
    """A class for analyzing and tagging code-mixed text using language identification.

    This class provides functionality to analyze text containing code-mixing (multiple languages)
    and identify the language of each token in the text using the IndicNER model.
    """

    def __init__(self) -> None:
        """Initialize the Tags analyzer with the IndicNER model.

        The model is loaded from the ai4bharat/IndicNER repository and is used for
        language identification in code-mixed text.
        """
        self.analyzer = CodeMixAnalyzer(
            model_path="ai4bharat/IndicNER", tokenizer_path="ai4bharat/IndicNER"
        )
        self.analyzer.load_model()

    def get_lid_tags(self, sentence: str) -> Tuple[List[str], List[Any]]:
        """Get language identification tags for each token in the sentence.

        Args:
            sentence: The input text to analyze for language identification.

        Returns:
            A tuple containing:
                - List of tokens from the input text
                - List of language identification labels for each token
        """
        tokens, cmi_value, combined_labels = self.analyzer.unicode_LID_get_sentence_cmi(
            sentence
        )
        return tokens, combined_labels

    def analyze(self, sentence: str) -> Tuple[List[str], List[Any]]:
        """Analyze the input text and return tokens with their language identification tags.

        Args:
            sentence: The input text to analyze.

        Returns:
            A tuple containing:
                - List of tokens from the input text
                - List of language identification labels for each token
        """
        tokens, lid_tags = self.get_lid_tags(sentence)
        return tokens, lid_tags
