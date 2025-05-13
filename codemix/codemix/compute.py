from typing import Dict, List, Optional, Union
from torch import device as torch_device

# from minicons import scorer
from codemix.cs_metrics import SyMCoMTemp  # , CodeMixMetricsTemp
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from alphabet_detector import AlphabetDetector
import numpy as np
import re

# import sys
# from collections import Counter
# from datetime import datetime
# import pandas as pd
from tqdm import tqdm
# from indictrans import Transliterator

tqdm.pandas()
np.random.seed(1234)
ad = AlphabetDetector()

# Initialize SyMCoMTemp with language and tagset configurations
symcom = SyMCoMTemp(
    L1="en",
    L2="hi",
    LID_tagset=["hi", "en", "ne", "univ", "acro"],
    PoS_tagset=[
        "NOUN",
        "ADV",
        "VERB",
        "AUX",
        "ADJ",
        "ADP",
        "PUNCT",
        "DET",
        "PRON",
        "PROPN",
        "PART",
        "CCONJ",
        "SCONJ",
        "INTJ",
        "NUM",
        "SYM",
        "X",
    ],
)

# Mapping of alphabet names to language codes
alphabet_language_mapping: Dict[str, str] = {
    "DEVANAGARI": "hi",
    "LATIN": "en",
    "TELUGU": "te",
    "TAMIL": "ta",
    "GUJARATI": "gu",
    "KANNADA": "ka",
    "MALAYALAM": "ml",
}


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

            detected = ad.detect_alphabet(token)

            if detected:
                lid = list(ad.detect_alphabet(token))[0]
                if lid in alphabet_language_mapping:
                    lids.append(alphabet_language_mapping[lid])
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


class LIDTaggerBhatetal:
    """A placeholder class for the LID tagger based on Bhat et al.'s work.

    This class is currently not implemented and serves as a placeholder for future
    implementation of the language identification approach described in Bhat et al.'s work.
    """

    def __init__(self, model_path: str, tokenizer_path: str) -> None:
        """Initialize the LIDTaggerBhatetal (not implemented).

        Args:
            model_path: Path to the pre-trained model
            tokenizer_path: Path to the tokenizer
        """
        raise NotImplementedError("Not implemented")

    def predict_lid_sentence(self, sentence: str) -> List[str]:
        """Predict language identification for a sentence (not implemented).

        Args:
            sentence: Input sentence to analyze

        Returns:
            List of language identification labels for each token
        """
        raise NotImplementedError("Not implemented")


# class CodeMixAnalyzer:
#     """A class for analyzing code-mixed text.

#     This class provides functionality to analyze code-mixed text by combining
#     various linguistic features including POS tagging, NER, and language identification.
#     """

#     def __init__(self, model_path: str, tokenizer_path: str) -> None:
#         """Initialize the CodeMixAnalyzer.

#         Args:
#             model_path: Path to the pre-trained model
#             tokenizer_path: Path to the tokenizer
#         """
#         self.model_path = model_path
#         self.tokenizer_path = tokenizer_path
#         self.model: Optional[AutoModelForTokenClassification] = None
#         self.tokenizer: Optional[AutoTokenizer] = None

#     def load_model(self) -> None:
#         """Load the model and tokenizer from the specified paths."""
#         self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
#         self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
#         self.device: torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = self.model.to(self.device)

#     def get_predictions(self, sentence: str) -> List[str]:
#         """Get predictions for a given sentence.

#         Args:
#             sentence: Input sentence to analyze

#         Returns:
#             List of predicted labels for each token in the sentence
#         """
#         # Let us first tokenize the sentence - split words into subwords
#         tok_sentence = self.tokenizer(sentence, return_tensors="pt").to(self.device)

#         with torch.no_grad():
#             # we will send the tokenized sentence to the model to get predictions
#             breakpoint()
#             logits = self.model(**tok_sentence).logits.argmax(-1)

#             # We will map the maximum predicted class id with the class label
#             predicted_tokens_classes = [
#                 self.model.config.id2label[t.item()] for t in logits[0]
#             ]

#             predicted_labels = []

#             previous_token_id = 0
#             # we need to assign the named entity label to the head word and not the following sub-words
#             word_ids = tok_sentence.word_ids()
#             for word_index in range(len(word_ids)):
#                 if word_ids[word_index] is None:
#                     previous_token_id = word_ids[word_index]
#                 elif word_ids[word_index] == previous_token_id:
#                     previous_token_id = word_ids[word_index]
#                 else:
#                     predicted_labels.append(predicted_tokens_classes[word_index])
#                     previous_token_id = word_ids[word_index]

#             return predicted_labels

#     def combine_lid_ner_acro_labels(
#         self,
#         acros: List[Union[str, bool]],
#         ner_predictions: List[Union[str, bool]],
#         lids: List[str]
#     ) -> List[str]:
#         """Combine LID, NER, and acronym labels into a single label sequence.

#         Args:
#             acros: List of acronym labels
#             ner_predictions: List of NER labels
#             lids: List of language identification labels

#         Returns:
#             Combined list of labels where NER and acronym labels take precedence
#             over LID labels
#         """
#         combined_labels = []

#         for lid, ner, acr in zip(lids, ner_predictions, acros):
#             if not ner and not acr:
#                 combined_labels.append(lid)
#                 continue
#             elif ner:
#                 combined_labels.append(ner)
#                 continue
#             elif acr:
#                 combined_labels.append(acr)

#         return combined_labels

#     def unicode_LID_get_sentence_cmi(
#         self,
#         sentence: str
#     ) -> tuple[List[str], float, List[str]]:
#         """Get code-mixing index (CMI) for a sentence using Unicode-based LID.

#         Args:
#             sentence: Input sentence to analyze

#         Returns:
#             Tuple containing:
#             - List of tokens from the sentence
#             - Code-mixing index (CMI) value
#             - List of combined labels for each token
#         """
#         self.acro_regex_pattern = r"\b[A-Z][A-Z0-9\.]{2,}s?\b"
#         ner_predictions = self.get_predictions(sentence)
#         ner_predictions = [False if item == "O" else "ne" for item in ner_predictions]
#         sentence = sentence.split(" ")
#         lids = []
#         acros = []
#         tokens = []

#         for token in sentence:
#             tokens.append(token)
#             if re.match(self.acro_regex_pattern, token):
#                 acros.append("acro")
#             else:
#                 acros.append(False)

#             detected = ad.detect_alphabet(token)

#             if detected:
#                 lid = list(ad.detect_alphabet(token))[0]
#                 if lid in alphabet_language_mapping:
#                     lids.append(alphabet_language_mapping[lid])
#                 else:
#                     lids.append("univ")
#             else:
#                 lids.append("univ")

#         # if(len(lids) != len(tokens)):
#         #     print("LID and tokens length mismatch")
#         # else:
#         #     print("LID and tokens length match")

#         combined_labels = self.combine_lid_ner_acro_labels(acros, ner_predictions, lids)

#         other = []
#         code_mix_metrics = CodeMixMetricsTemp(combined_labels, other)
#         cmi_value = code_mix_metrics.cmi(combined_labels)

#         return tokens, cmi_value, combined_labels

#     def predictposSent(self, sentence: str) -> List[str]:
#         """Predict POS tags for a given sentence.

#         Args:
#             sentence: Input sentence to tag

#         Returns:
#             List of predicted POS tags for each token in the sentence
#         """
#         tokenized_sentence = self.tokenizer(sentence, return_tensors="pt")

#         mask = []
#         prev_id = None
#         for ind, id in enumerate(tokenized_sentence.word_ids()):
#             if id is None:
#                 mask.append(-100)
#             elif id == prev_id:
#                 mask.append(-100)
#             elif id != prev_id:
#                 mask.append(id)
#             prev_id = id

#         outputs = self.model(**tokenized_sentence.to(self.device))

#         preds = np.argmax(outputs["logits"].cpu().detach().numpy(), axis=2).squeeze()

#         true_preds = [label_list[p] for (p, l) in zip(preds, mask) if l != -100]

#         return true_preds

#     def generate_symcom_count_features(self, row: pd.Series) -> Dict[str, int]:
#         """Generate count features for SyMCoM analysis.

#         Args:
#             row: Pandas Series containing symcom_pos_scores

#         Returns:
#             Dictionary containing various count features:
#             - zero_count: Number of zero scores
#             - one_count: Number of one scores
#             - neg_one_count: Number of negative one scores
#             - positives: Number of positive scores between 0 and 1
#             - negatives: Number of negative scores between -1 and 0
#             - count: Total number of scores
#         """
#         zero_count, one_count, neg_one_count, positives, negatives, count = (
#             0,
#             0,
#             0,
#             0,
#             0,
#             0,
#         )
#         symcom_pos_scores = row["symcom_pos_scores"]
#         count = len(symcom_pos_scores)
#         for k, v in symcom_pos_scores.items():
#             if v == 0:
#                 zero_count += 1
#             elif v == -1:
#                 neg_one_count += 1
#             elif v == 1:
#                 one_count += 1
#             elif -1 < v < 0:
#                 negatives += 1
#             elif 0 < v < 1:
#                 positives += 1

#         return {
#             "zero_count": zero_count,
#             "one_count": one_count,
#             "neg_one_count": neg_one_count,
#             "positives": positives,
#             "negatives": negatives,
#             "count": count,
#         }

#     def get_scores(self, lines: List[str], mlm_model: Any) -> List[float]:
#         """Get sequence scores for a list of lines using a masked language model.

#         Args:
#             lines: List of input lines to score
#             mlm_model: Masked language model to use for scoring

#         Returns:
#             List of scores for each input line
#         """
#         dl = DataLoader(lines, batch_size=1)
#         scores = []
#         for idx, batch in enumerate(tqdm(dl)):
#             scores.extend(
#                 mlm_model.sequence_score(batch, reduction=lambda x: -x.sum(0).item())
#             )
#         return scores


# if __name__ == "__main__":
#     tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicNER")
#     model = AutoModelForTokenClassification.from_pretrained("ai4bharat/IndicNER")
#     device = "cuda:0"

#     model = model.to(device)

#     df = pd.read_json("GCM1-annotations.json")

#     cmi_list, lid_list = [], []

#     print("Starting LID computation")
#     code_mix_analyzer = CodeMixAnalyzer(model, tokenizer)
#     for ind, row in tqdm(df.iterrows()):
#         cmi, lid = code_mix_analyzer.unicode_LID_get_sentence_cmi(row["data.CM_candidates"])

#         cmi_list.append(cmi)
#         lid_list.append(lid)

#     count = 0

#     for row, cmi in zip(df["data.CMI_unicode_based_LID"].tolist(), cmi_list):
#         try:
#             assert row == cmi, print(row, cmi)
#         except AssertionError:
#             count += 1
#             print(type(row), type(cmi))

#     df["CMI"], df["LID"] = cmi_list, lid_list
#     df["sum_abs_diff"] = df.apply(lambda row: code_mix_analyzer.get_abs_diff(row), axis=1)

#     cmi_list, spavg_list, burstiness_list = [], [], []

#     for ind, lids in tqdm(enumerate(df["LID"])):
#         code_mix_metrics = CodeMixMetrics()
#         cmi = code_mix_metrics.cmi(lids)
#         burstiness = code_mix_metrics.burstiness(lids)
#         spavg = code_mix_metrics.spavg(lids)

#         cmi_list.append(cmi)
#         burstiness_list.append(burstiness)
#         spavg_list.append(spavg)

#     df["CMI"] = cmi_list
#     df["spavg"] = spavg_list
#     df["burstiness"] = burstiness_list

#     del model

#     print("Starting PoS computation")


#     # check please
#     modelpath = r"/home2/anmol.goel/prashantk/en-hi-pos-tagger/lemma_final_model/2-xlmr-onlyUDTokensLemmas"
#     modelname = r"xlm-roberta-base"

#     tokenizer = AutoTokenizer.from_pretrained(modelname)
#     model = AutoModelForTokenClassification.from_pretrained(modelpath)
#     model.to("cuda")

#     datasets_UD = load_dataset(
#         "/home2/anmol.goel/prashantk/en-hi-pos-tagger/load_UD_enhics_mod (3).py",
#         "qhe_hiencs",
#     )
#     label_list = datasets_UD["train"].features["upos"].feature.names

#     tags = []
#     errors, no_errors = [], []
#     for ind, sample in tqdm(enumerate(df["data.CM_candidates"])):
#         try:
#             tags_normalised = code_mix_analyzer.predictposSent(model, sample)
#             tags.append(tags_normalised)
#             no_errors.append(ind)

#         except Exception as e:
#             print(e, sample)
#             tags.append(None)
#             errors.append((ind, e))

#     df["PoSTags"] = tags
#     symcom_pos_scores, symcom_sentence_scores = [], []
#     for ind, row in tqdm(df.iterrows()):
#         cm_sentence = CodeMixSentence(
#             sentence=None,
#             tokens=row["data.CM_candidates"],
#             LID_Tags=row["LID"],
#             PoS_Tags=row["PoSTags"],
#         )

#         symcom_pos_scores.append(symcom.symcom_pos_tags(cm_sentence))
#         symcom_sentence_scores.append(symcom.symcom_sentence(cm_sentence))

#     df["symcom_pos_scores"], df["symcom_sentence_scores"] = (
#         symcom_pos_scores,
#         symcom_sentence_scores,
#     )

#     pos_categories = {
#         "open-class": ["ADJ", "ADV", "INTJ", "NOUN", "PROPN", "VERB"],
#         "closed-class": ["ADP", "AUX", "CCONJ", "DET", "NUM", "PART", "PRON", "SCONJ"],
#         "other": ["PUNCT", "SYM", "X"],
#     }
#     symcom_feats = []
#     for ind, row in tqdm(df.iterrows()):
#         symcom_feats.append(code_mix_analyzer.generate_symcom_count_features(row))

#     symcom_temp_df = pd.DataFrame.from_dict(symcom_feats)
#     gcm_symcom_feat_concat = pd.concat([df, symcom_temp_df], axis=1)

#     del model

#     # PPL scores
#     device = torch.device("cuda:0")

#     models = {
#         "xlmr": "xlm-roberta-base",
#         "bernice": "jhu-clsp/bernice",
#     }

#     print("starting PPL scores computation")

#     # for modelname, modelcard in models.items():
#     #     print(f"starting PPL score computation using {modelname}")
#     #     model = scorer.MaskedLMScorer(modelcard, device)
#     #     lines = gcm_symcom_feat_concat["data.CM_candidates"].tolist()
#     #     scores = code_mix_analyzer.get_scores(lines, model)
#     #     gcm_symcom_feat_concat[f"{modelname}_ppl"] = scores
#     #     del model

#     gcm_symcom_feat_concat.to_json(
#         "../wip-annotations/GCM1-annotations-with-postags-symcom.json",
#         force_ascii=False,
#         indent=4,
#     )
