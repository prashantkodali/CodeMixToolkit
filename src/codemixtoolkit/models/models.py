from typing import List, Optional, Union, Dict, Literal, Any
import torch
from torch import device as torch_device
import numpy as np
import re
import requests
import pandas as pd
from transformers import AutoModelForTokenClassification, AutoTokenizer
from alphabet_detector import AlphabetDetector
from ai4bharat.transliteration import XlitEngine


class PoSTagger:
    """A class for Part-of-Speech tagging using transformer models.

    This class provides functionality to load a pre-trained POS tagging model and
    use it to predict POS tags for input sentences.
    """

    def __init__(self, model_path: str, tokenizer_path: str) -> None:
        """Initialize the PoSTagger.

        Args:
            model_path: Path to the pre-trained model
            tokenizer_path: Path to the tokenizer
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.model: Optional[AutoModelForTokenClassification] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.device: torch_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def load_model_tokenizer(self) -> None:
        """Load the model and tokenizer from the specified paths."""
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.model = self.model.to(self.device)

    def predict_pos_sentence(self, sentence: str) -> List[str]:
        """Predict POS tags for a given sentence.

        Args:
            sentence: Input sentence to tag

        Returns:
            List of predicted POS tags for each token in the sentence
        """
        tokenized_sentence = self.tokenizer(sentence, return_tensors="pt")
        mask = []
        prev_id = None
        for ind, id in enumerate(tokenized_sentence.word_ids()):
            if id is None:
                mask.append(-100)
            elif id == prev_id:
                mask.append(-100)
            elif id != prev_id:
                mask.append(id)
            prev_id = id

        with torch.no_grad():
            outputs = self.model(**tokenized_sentence.to(self.device))
            preds = np.argmax(
                outputs["logits"].cpu().detach().numpy(), axis=2
            ).squeeze()
        true_preds = [
            self.model.config.id2label[p] for (p, l) in zip(preds, mask) if l != -100
        ]
        return true_preds


class NERtagger:
    """A class for Named Entity Recognition using transformer models.

    This class provides functionality to load a pre-trained NER model and
    use it to predict named entities in input sentences.
    """

    def __init__(
        self, model_path: str, tokenizer_path: str, finegrain_labels: bool = False
    ) -> None:
        """Initialize the NERtagger.

        Args:
            model_path: Path to the pre-trained model
            tokenizer_path: Path to the tokenizer
            finegrain_labels: Whether to return fine-grained NER labels or just binary NE/non-NE
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.model: Optional[AutoModelForTokenClassification] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.device: torch_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.finegrain_labels = finegrain_labels

    def load_model_tokenizer(self) -> None:
        """Load the model and tokenizer from the specified paths."""
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.model = self.model.to(self.device)

    def predict_ner_sentence(self, sentence: str) -> List[Union[str, bool]]:
        """Predict named entities for a given sentence.

        Args:
            sentence: Input sentence to analyze

        Returns:
            List of predicted named entity labels for each token in the sentence.
            If finegrain_labels is True, returns detailed NER labels.
            If False, returns binary labels (False for non-NE, "ne" for named entities).
        """
        # Let us first tokenize the sentence - split words into subwords
        tok_sentence = self.tokenizer(sentence, return_tensors="pt").to(self.device)

        with torch.no_grad():
            # we will send the tokenized sentence to the model to get predictions
            breakpoint()
            logits = self.model(**tok_sentence).logits.argmax(-1)

            # We will map the maximum predicted class id with the class label
            predicted_tokens_classes = [
                self.model.config.id2label[t.item()] for t in logits[0]
            ]

            predicted_labels = []

            previous_token_id = 0
            # we need to assign the named entity label to the head word and not the following sub-words
            word_ids = tok_sentence.word_ids()
            for word_index in range(len(word_ids)):
                if word_ids[word_index] is None:
                    previous_token_id = word_ids[word_index]
                elif word_ids[word_index] == previous_token_id:
                    previous_token_id = word_ids[word_index]
                else:
                    predicted_labels.append(predicted_tokens_classes[word_index])
                    previous_token_id = word_ids[word_index]

        if self.finegrain_labels:
            return predicted_labels
        else:
            ner_predictions = [
                False if item == "O" else "ne" for item in predicted_labels
            ]
            return ner_predictions


class UnicodeLIDtagger:
    """A class for Language Identification (LID) using Unicode-based detection.

    This class provides functionality to identify languages in text using Unicode
    character detection and acronym detection.
    """

    def __init__(self) -> None:
        """Initialize the UnicodeLIDtagger with regex pattern for acronym detection."""
        self.acro_regex_pattern: str = r"\b[A-Z][A-Z0-9\.]{2,}s?\b"

        # Mapping of alphabet names to language codes
        self.alphabet_language_mapping: Dict[str, str] = {
            "DEVANAGARI": "hi",
            "LATIN": "en",
            "TELUGU": "te",
            "TAMIL": "ta",
            "GUJARATI": "gu",
            "KANNADA": "ka",
            "MALAYALAM": "ml",
        }

        self.ad = AlphabetDetector()

    def combine_lid_ner_acro_labels(
        self,
        acros: List[Union[str, bool]],
        ner_predictions: List[Union[str, bool]],
        lids: List[str],
    ) -> List[str]:
        """Combine LID, NER, and acronym labels into a single label sequence.

        Args:
            acros: List of acronym labels
            ner_predictions: List of NER labels
            lids: List of language identification labels

        Returns:
            Combined list of labels where NER and acronym labels take precedence
            over LID labels
        """
        combined_labels = []

        for lid, ner, acr in zip(lids, ner_predictions, acros):
            if not ner and not acr:
                combined_labels.append(lid)
                continue
            elif ner:
                combined_labels.append(ner)
                continue
            elif acr:
                combined_labels.append(acr)

        return combined_labels

    def get_unicode_lid_predictions(
        self, sentence: str, ner_predictions: Optional[List[str]] = None
    ) -> tuple[List[str], List[str]]:
        """Get language identification predictions for a sentence.

        Args:
            sentence: Input sentence to analyze
            ner_predictions: Optional list of NER predictions for the sentence

        Returns:
            Tuple containing:
            - List of tokens from the sentence
            - List of combined labels (LID, NER, acronym) for each token
        """
        if ner_predictions is None:
            ner_predictions = self.get_predictions(sentence)
            ner_predictions = [
                False if item == "O" else "ne" for item in ner_predictions
            ]

        sentence = sentence.split(" ")
        lids = []
        acros = []
        tokens = []

        for token in sentence:
            tokens.append(token)
            if re.match(self.acro_regex_pattern, token):
                acros.append("acro")
            else:
                acros.append(False)

            detected = self.ad.detect_alphabet(token)

            if detected:
                lid = list(self.ad.detect_alphabet(token))[0]
                if lid in self.alphabet_language_mapping:
                    lids.append(self.alphabet_language_mapping[lid])
                else:
                    lids.append("univ")
            else:
                lids.append("univ")

        # if(len(lids) != len(tokens)):
        #     print("LID and tokens length mismatch")
        # else:
        #     print("LID and tokens length match")

        combined_labels = self.combine_lid_ner_acro_labels(acros, ner_predictions, lids)

        return tokens, combined_labels


class Romanizer:
    """
    A class for normalizing text using AI4Bharat's transliteration engine.

    This class provides functionality to transliterate text from one script to another,
    primarily focused on Indic scripts to English and vice versa.

    Attributes:
        tgt_script (str): Target script for transliteration (default: "en")
        src_script (str): Source script for transliteration (default: "hi")
        src_script_type (str): Type of source script (default: "indic")
        engine (XlitEngine): Instance of AI4Bharat's transliteration engine
    """

    def __init__(
        self,
        tgt_script: str = "en",
        src_script: str = "hi",
        src_script_type: Literal["indic", "roman"] = "indic",
    ) -> None:
        """
        Initialize the Normalizer with specified script parameters.

        Args:
            tgt_script (str): Target script for transliteration (default: "en")
            src_script (str): Source script for transliteration (default: "hi")
            src_script_type (Literal["indic", "roman"]): Type of source script (default: "indic")
        """
        self.tgt_script = tgt_script
        self.src_script = src_script
        self.src_script_type = src_script_type
        self.engine = XlitEngine(beam_width=10, src_script_type=src_script_type)

    def romanize_text(self, text: str) -> str:
        """
        Normalize the input text by transliterating it to the target script.

        Args:
            text (str): Input text to be normalized

        Returns:
            str: Normalized text in the target script
        """
        out = self.engine.translit_sentence(text, self.src_script)
        return out


class CSNLILIDClient:
    """Client for interacting with the CSNLI-LID API service.

    This class provides a convenient interface for making requests to the CSNLI-LID API
    for language identification and text normalization.
    """

    def __init__(self, base_url: str = "http://localhost:6000"):
        """Initialize the CSNLI-LID client.

        Args:
            base_url: Base URL of the CSNLI-LID API service. Defaults to "http://localhost:6000"
        """
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/csnli-lid"
        self.headers = {"Content-Type": "application/json"}

    def get_lid(self, text: str) -> Dict[str, Any]:
        """Process text through the CSNLI-LID API.

        Args:
            text: Input text to process

        Returns:
            Dictionary containing the processed text information including:
            - text_str: Original text
            - text_tokenized: Tokenized text
            - norm_text: Normalized text
            - lid: Language identification tags
        """
        data = {"text": text}

        try:
            response = requests.post(self.endpoint, headers=self.headers, json=data)
            response.raise_for_status()
            return response.json()["csnli_op"]
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error processing text through CSNLI-LID API: {str(e)}")

    def get_lid_and_print(self, text: str) -> pd.DataFrame:
        """Process text and return results as a pandas DataFrame.

        Args:
            text: Input text to process

        Returns:
            pandas DataFrame containing the processed text information
        """
        result = self.get_lid(text)
        df = pd.DataFrame(result)
        return df.drop(columns=["text_str"])

    def is_service_available(self) -> bool:
        """Check if the CSNLI-LID service is available.

        Returns:
            bool: True if service is available, False otherwise
        """
        try:
            response = requests.post(
                self.endpoint, headers=self.headers, json={"text": "hello test"}
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
