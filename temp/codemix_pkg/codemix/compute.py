from torch.utils.data import DataLoader
# from minicons import scorer
from codemix.cs_metrics import CodeMixSentence, SyMCoM, CodeMixMetrics
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import concatenate_datasets, load_dataset, load_metric
from alphabet_detector import AlphabetDetector
import numpy as np
import csv
import os
import pickle
import re
import sys
from collections import Counter
from datetime import datetime
import pandas as pd
from tqdm import tqdm
# from indictrans import Transliterator

tqdm.pandas()
np.random.seed(1234)
ad = AlphabetDetector()

symcom = SyMCoM(L1="en", L2="hi", LID_tagset=["hi", "en", "ne", "univ", "acro"],
                PoS_tagset=["NOUN", "ADV", "VERB", "AUX", "ADJ", "ADP", "PUNCT", "DET",
                            "PRON", "PROPN", "PART", "CCONJ", "SCONJ", "INTJ", "NUM", "SYM", "X",])

alphabet_language_mapping = {
    "DEVANAGARI": "hi",
    "LATIN": "en",
    "TELUGU": "te",
    "TAMIL": "ta",
    "GUJARATI": "gu",
    "KANNADA": "ka",
    "MALAYALAM": "ml",
}

class CodeMixAnalyzer:
    def __init__(self, model_path, tokenizer_path):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.model = None
        self.tokenizer = None

    def load_model(self):
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def combine_lid_ner_acro_labels(self, acros, ner_predictions, lids):
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

    def get_predictions(self, sentence):
        # Let us first tokenize the sentence - split words into subwords
        tok_sentence = self.tokenizer(sentence, return_tensors="pt").to(self.device)

        with torch.no_grad():
            # we will send the tokenized sentence to the model to get predictions
            breakpoint()
            logits = self.model(**tok_sentence).logits.argmax(-1)

            # We will map the maximum predicted class id with the class label
            predicted_tokens_classes = [
                self.model.config.id2label[t.item()] for t in logits[0]]

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
                    predicted_labels.append(
                        predicted_tokens_classes[word_index])
                    previous_token_id = word_ids[word_index]

            return predicted_labels

    def unicode_LID_get_sentence_cmi(self, sentence):
        acro_regex_pattern = r"\b[A-Z][A-Z0-9\.]{2,}s?\b"
        ner_predictions = self.get_predictions(sentence)
        ner_predictions = [False if item ==
                           "O" else "ne" for item in ner_predictions]
        sentence = sentence.split(" ")
        lids = []
        acros = []
        tokens = []

        for token in sentence:
            tokens.append(token)
            if re.match(acro_regex_pattern, token):
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
    
        combined_labels = self.combine_lid_ner_acro_labels(
            acros, ner_predictions, lids)
        
        other = []
        code_mix_metrics = CodeMixMetrics(combined_labels, other)
        cmi_value = code_mix_metrics.cmi(combined_labels)

        return tokens, cmi_value, combined_labels

    def get_abs_diff(self, row):
        # choice_col_names = ["choice_value", "choice_value_2", "choice_value_3"]
        if len(row["int_annotations"]) == 1:
            return 0

        elif len(row["int_annotations"]) == 2:
            cvs = []
            for cv in row["int_annotations"]:
                if cv in [1.0, 2.0, 3.0, 4.0, 5.0, "2", "4", "5", "3"]:
                    if not isinstance(cv, (int, float)):
                        cv = eval(cv)
                    cvs.append(cv)
            return abs(cvs[0] - cvs[1])

        elif len(row["int_annotations"]) == 3:
            cvs = []
            for cv in row["int_annotations"]:
                if cv in [1.0, 2.0, 3.0, 4.0, 5.0, "2", "4", "5", "3"]:
                    if not isinstance(cv, (int, float)):
                        cv = eval(cv)
                    cvs.append(cv)
            return abs(cvs[0] - cvs[1]) + abs(cvs[0] - cvs[2]) + abs(cvs[1] - cvs[2])

    def predictposSent(self, sentence):
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

        outputs = self.model(**tokenized_sentence.to(self.device))

        preds = np.argmax(outputs["logits"].cpu(
        ).detach().numpy(), axis=2).squeeze()

        true_preds = [label_list[p]
                      for (p, l) in zip(preds, mask) if l != -100]

        return true_preds

    def generate_symcom_count_features(self, row):
        zero_count, one_count, neg_one_count, positives, negatives, count = 0, 0, 0, 0, 0, 0
        symcom_pos_scores = row["symcom_pos_scores"]
        count = len(symcom_pos_scores)
        for k, v in symcom_pos_scores.items():
            if v == 0:
                zero_count += 1
            elif v == -1:
                neg_one_count += 1
            elif v == 1:
                one_count += 1
            elif -1 < v < 0:
                negatives += 1
            elif 0 < v < 1:
                positives += 1

        return {
            "zero_count": zero_count,
            "one_count": one_count,
            "neg_one_count": neg_one_count,
            "positives": positives,
            "negatives": negatives,
            "count": count
        }
    
    def get_scores(self, lines, mlm_model):
        dl = DataLoader(lines, batch_size=1)
        scores = []
        for idx, batch in enumerate(tqdm(dl)):
            scores.extend(
                mlm_model.sequence_score(
                    batch, reduction=lambda x: -x.sum(0).item())
            )
        return scores

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
