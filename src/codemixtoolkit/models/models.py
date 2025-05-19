from typing import List, Optional, Union, Dict, Literal, Any
import torch
from tqdm import tqdm
from torch import device as torch_device
import numpy as np
import re
import requests
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoModelForTokenClassification, AutoTokenizer
from alphabet_detector import AlphabetDetector
from ai4bharat.transliteration import XlitEngine
from litellm import completion

from ..config import config
from .base import BaseModel


class PoSTagger(BaseModel):
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
        super().__init__()
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

    def predict(self, text: str, **kwargs) -> Dict[str, Any]:
        """Predict POS tags for a given sentence.

        Args:
            text: Input sentence to tag
            **kwargs: Additional arguments (not used)

        Returns:
            Dictionary containing:
            - prediction: List of predicted POS tags
            - raw_response: Same as prediction
        """
        tokenized_sentence = self.tokenizer(text, return_tensors="pt")
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

        return {"sample": text, "prediction": true_preds, "raw_response": true_preds}


class NERtagger(BaseModel):
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
        super().__init__()
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

    def predict(self, text: str, **kwargs) -> Dict[str, Any]:
        """Predict named entities for a given sentence.

        Args:
            text: Input sentence to analyze
            **kwargs: Additional arguments (not used)

        Returns:
            Dictionary containing:
            - prediction: List of predicted named entity labels
            - raw_response: Same as prediction
        """
        tok_sentence = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            logits = self.model(**tok_sentence).logits.argmax(-1)
            predicted_tokens_classes = [
                self.model.config.id2label[t.item()] for t in logits[0]
            ]

            predicted_labels = []

            previous_token_id = 0
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
            return {"prediction": predicted_labels, "raw_response": predicted_labels}
        else:
            ner_predictions = [
                False if item == "O" else "ne" for item in predicted_labels
            ]
            return {
                "sample": text,
                "prediction": ner_predictions,
                "raw_response": ner_predictions,
            }


class UnicodeLIDtagger(BaseModel):
    """A class for Language Identification (LID) using Unicode-based detection.

    This class provides functionality to identify languages in text using Unicode
    character detection and acronym detection.
    """

    def __init__(self) -> None:
        """Initialize the UnicodeLIDtagger with regex pattern for acronym detection."""
        super().__init__()
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

    def predict(
        self, text: str, ner_predictions: Optional[List[str]] = None, **kwargs
    ) -> Dict[str, Any]:
        """Get language identification predictions for a sentence.

        Args:
            text: Input sentence to analyze
            ner_predictions: Optional list of NER predictions for the sentence
            **kwargs: Additional arguments (not used)

        Returns:
            Dictionary containing:
            - prediction: List of combined labels (LID, NER, acronym)
            - tokens: List of tokens from the sentence
            - raw_response: Same as prediction
        """
        if ner_predictions is None:
            ner_predictions = [False] * len(text.split())

        sentence = text.split(" ")
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

        combined_labels = self.combine_lid_ner_acro_labels(acros, ner_predictions, lids)

        return {
            "sample": text,
            "prediction": combined_labels,
            "tokens": tokens,
            "raw_response": combined_labels,
        }


class Romanizer(BaseModel):
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
        super().__init__()
        self.tgt_script = tgt_script
        self.src_script = src_script
        self.src_script_type = src_script_type
        self.engine = XlitEngine(beam_width=10, src_script_type=src_script_type)

    def predict(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Normalize the input text by transliterating it to the target script.

        Args:
            text (str): Input text to be normalized
            **kwargs: Additional arguments (not used)

        Returns:
            Dictionary containing:
            - prediction: Normalized text in the target script
            - raw_response: Same as prediction
        """
        out = self.engine.translit_sentence(text, self.src_script)
        return {"sample": text, "prediction": out, "raw_response": out}


class CSNLILIDClient(BaseModel):
    """Client for interacting with the CSNLI-LID API service.

    This class provides a convenient interface for making requests to the CSNLI-LID API
    for language identification and text normalization.
    """

    def __init__(self, base_url: str = "http://localhost:6000"):
        """Initialize the CSNLI-LID client.

        Args:
            base_url: Base URL of the CSNLI-LID API service. Defaults to "http://localhost:6000"
        """
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/csnli-lid"
        self.headers = {"Content-Type": "application/json"}

    def predict(self, text: str, **kwargs) -> Dict[str, Any]:
        """Process text through the CSNLI-LID API.

        Args:
            text: Input text to process
            **kwargs: Additional arguments (not used)

        Returns:
            Dictionary containing:
            - prediction: Processed text information
            - raw_response: Same as prediction
        """
        data = {"text": text}

        try:
            response = requests.post(self.endpoint, headers=self.headers, json=data)
            response.raise_for_status()
            result = response.json()["csnli_op"]
            return {"sample": text, "prediction": result, "raw_response": result}
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error processing text through CSNLI-LID API: {str(e)}")

    def get_lid_and_print(self, text: str) -> pd.DataFrame:
        """Process text and return results as a pandas DataFrame.

        Args:
            text: Input text to process

        Returns:
            pandas DataFrame containing the processed text information
        """
        result = self.predict(text)
        df = pd.DataFrame(result["prediction"])
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


def make_openrouter_api_call(
    messages: List[Dict[str, str]],
    model: str = "openai/gpt-3.5-turbo",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    api_key: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Make an API call to OpenRouter using LiteLLM.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model: Model to use (default: openai/gpt-3.5-turbo)
        temperature: Temperature for generation (default: 0.7)
        max_tokens: Maximum tokens to generate (default: 1000)
        api_key: OpenRouter API key. If not provided, will try to get from OPENROUTER_API_KEY env var
        **kwargs: Additional arguments to pass to the API call

    Returns:
        Dictionary containing the API response

    Raises:
        ValueError: If no API key is provided and OPENROUTER_API_KEY is not set
    """
    # Get API key from argument or environment variable
    # api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    api_key = config.OPENROUTER_API_KEY

    if not api_key:
        raise ValueError(
            "OpenRouter API key must be provided either as an argument or through OPENROUTER_API_KEY environment variable"
        )

    # Set up the API call parameters
    params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "api_key": api_key,
        **kwargs,
    }

    # Make the API call using LiteLLM
    response = completion(**params)

    return {
        "content": response.choices[0].message.content,
        "usage": response.usage,
        "model": model,
        "finish_reason": response.choices[0].finish_reason,
    }


class LLMPromptModel(BaseModel):
    """A class for handling LLM prompting using LiteLLM."""

    def __init__(
        self,
        model: str = "openrouter/meta-llama/llama-3.3-70b-instruct:free",
        temperature: float = 0.1,
        max_tokens: int = 10,
        instruction: Optional[str] = None,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        instruction_label_mapping: Optional[Dict[str, str]] = None,
        dataset_label_mapping: Optional[Dict[str, str]] = None,
    ):
        """Initialize the LLM prompt model.

        Args:
            model: Model to use for evaluation (default: openrouter/meta-llama/llama-3.3-70b-instruct:free)
            temperature: Temperature for generation (default: 0.1)
            max_tokens: Maximum tokens to generate (default: 10)
            instruction: Optional instruction to include in the prompt (default: None)
            few_shot_examples: List of few-shot examples (default: None)
            instruction_label_mapping: Mapping from model output strings to numeric labels
            dataset_label_mapping: Mapping from dataset label strings to numeric labels
        """
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.instruction = instruction
        self.few_shot_examples = few_shot_examples or []
        self.instruction_label_mapping = instruction_label_mapping or {}
        self.dataset_label_mapping = dataset_label_mapping or {}

    def _create_prompt(self, text: str) -> str:
        """Create the prompt for the LLM.

        Args:
            text: Input text

        Returns:
            Formatted prompt string
        """
        prompt = ""

        # Add instruction if provided
        if self.instruction:
            prompt += f"{self.instruction}\n\n"

        # Add few-shot examples if available
        if self.few_shot_examples:
            for example in self.few_shot_examples:
                prompt += f"Input: {example['input']}\n"
                prompt += f"Output: {example['output']}\n\n"

        # Add the current input
        prompt += f"Input: {text}\nOutput:"
        return prompt

    def predict(self, text: str, **kwargs) -> Dict[str, Any]:
        """Make a prediction using the LLM.

        Args:
            text: Input text
            **kwargs: Additional arguments for the LLM call

        Returns:
            Dictionary containing:
            - prediction: Numeric prediction label
            - prompt: The prompt used
            - raw_response: Raw response from the model
            - usage: Token usage information
        """
        prompt = self._create_prompt(text)

        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs,
        )

        if self.instruction_label_mapping:
            prediction = self._convert_prediction_to_label(
                response.choices[0].message.content
            )
        else:
            prediction = None

        return {
            "sample": text,
            "prediction": prediction,
            "prompt": prompt,
            "raw_response": response.choices[0].message.content,
            "usage": response.usage,
        }

    # def predict_dataset(
    #     self,
    #     dataset: Union[Dataset, DatasetDict],
    #     text_column: str = "text",
    #     label_column: str = "label",
    #     max_samples: Optional[int] = None,
    #     eval_split: Optional[str] = "test",
    #     **kwargs,
    # ) -> Dict[str, Any]:
    #     """Make predictions for a dataset using LLM prompting.

    #     Args:
    #         dataset: HuggingFace Dataset or DatasetDict to predict on
    #         text_column: Name of the column containing input text (default: "text")
    #         label_column: Name of the column containing labels (default: "label")
    #         max_samples: Maximum number of samples to predict (default: None)
    #         eval_split: Dataset split to use if dataset is DatasetDict (default: "test")
    #         **kwargs: Additional arguments for the predictions

    #     Returns:
    #         Dictionary containing prediction results and metadata
    #     """
    #     results = []
    #     predictions = []
    #     true_labels = []

    #     # Handle DatasetDict
    #     if isinstance(dataset, DatasetDict):
    #         if eval_split not in dataset:
    #             raise ValueError(f"Split {eval_split} not found in dataset")
    #         dataset_data = dataset[eval_split]
    #     else:
    #         dataset_data = dataset

    #     # Limit samples if specified
    #     if max_samples is not None:
    #         dataset_data = dataset_data.select(range(max_samples))

    #     # Process each sample
    #     for item in tqdm(dataset_data):
    #         # Get text and label
    #         text = item.get(text_column, item.get("input", item.get("sentence", "")))
    #         label = item.get(label_column, item.get("labels", None))

    #         if not text:
    #             continue

    #         # Get prediction
    #         result = self.predict(text, **kwargs)
    #         results.append(result)

    #         # Convert predictions and labels to numeric values
    #         pred_label = result["prediction"]

    #         if isinstance(label, str):
    #             true_label = self._convert_dataset_label(label)
    #         elif isinstance(label, int):
    #             true_label = label
    #         else:
    #             raise ValueError(f"Label type {type(label)} not supported")

    #         predictions.append(pred_label)
    #         true_labels.append(true_label)

    #     return {
    #         "results": results,
    #         "predictions": predictions,
    #         "true_labels": true_labels,
    #         "num_samples": len(results),
    #     }

    def _convert_prediction_to_label(self, prediction: str) -> int:
        """Convert model's string prediction to numeric label.

        Args:
            prediction: String prediction from the model

        Returns:
            Numeric label corresponding to the prediction
        """
        # Clean the prediction string
        prediction = prediction.strip().lower()

        # If the prediction is in the mapping, return the numeric label
        if prediction in self.instruction_label_mapping:
            return self.instruction_label_mapping[prediction]

        # Try to find the label in the mapping
        found_in = 0
        for label_str, label_num in self.instruction_label_mapping.items():
            if label_str.lower() in prediction:
                found_in += 1
        if found_in == 1:
            return int(label_num)
        elif found_in > 1:
            print(
                f"WARNING: Prediction {prediction} found in multiple labels: {self.instruction_label_mapping.keys()}"
            )
            return -100
        elif found_in == 0:
            print(
                f"WARNING: Prediction {prediction} not found in any of the labels: {self.instruction_label_mapping.keys()}"
            )
            return -100
        else:
            print(f"WARNING: Prediction {prediction} is not a valid label")
            return -100

    def _convert_dataset_label(self, label: str) -> int:
        """Convert dataset's string label to numeric label.

        Args:
            label: String label from the dataset

        Returns:
            Numeric label corresponding to the dataset label
        """
        # Clean the label string
        label = str(label).strip().lower()

        # Try to find the label in the mapping
        for label_str, label_num in self.dataset_label_mapping.items():
            if label_str.lower() == label:
                return int(label_num)

        # If no mapping found, return -1 to indicate error
        return -1
