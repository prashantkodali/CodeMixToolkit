import os
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

from tqdm import tqdm

tqdm.pandas()

import stanza

# downloading stanza models for indian languages & english
# for i in ["en", "hi", "mr", "ta", "te"]:
#     stanza.download(i)

for i in ["en", "hi"]:
    stanza.download(i)


cwd = os.getcwd()

# TOKENIZE-POS CODE

# setting up awesome-align aligner
from .align_util import awesomealign

aligner = awesomealign(
    modelpath="bert-base-multilingual-cased",
    tokenizerpath="bert-base-multilingual-cased",
)


# setting up stanza pipelines
def get_stanza_info(text: str, language: str) -> Dict[str, List[List[str]]]:
    """
    Process text using Stanza NLP pipeline to get tokenization and POS tagging information.

    Args:
        text (str): Input text to process
        language (str): Language code (e.g., 'en', 'hi')

    Returns:
        Dict[str, List[List[str]]]: Dictionary containing:
            - sentences: List of sentences as strings
            - tokens: List of lists of tokens for each sentence
            - postags: List of lists of POS tags for each sentence
    """
    nlp_lang = stanza.Pipeline(lang=language, processors="tokenize, pos")
    doc = nlp_lang(text)

    sents, tokens, postags = [], [], []

    for sentence in doc.sentences:
        sents.append(" ".join([f"{token.text}" for token in sentence.tokens]))
        tokens.append([f"{token.text}" for token in sentence.words])
        postags.append([f"{token.upos}" for token in sentence.words])

    return {"sentences": sents, "tokens": tokens, "postags": postags}


# creating token alignment map
def create_alignments_token_map(
    sent_src: str, sent_tgt: str, alignments: str
) -> Optional[Dict[str, str]]:
    """
    Create a mapping between aligned tokens from source and target sentences.

    Args:
        sent_src (str): Source sentence
        sent_tgt (str): Target sentence
        alignments (str): Alignment string in format "src_idx-tgt_idx src_idx-tgt_idx ..."

    Returns:
        Optional[Dict[str, str]]: Dictionary mapping source tokens to target tokens and vice versa,
                                or None if alignment fails
    """
    token_map = {}
    sent_src = sent_src.split()
    sent_tgt = sent_tgt.split()

    for el in alignments.split():
        el = el.split("-")
        try:
            token_map[sent_src[int(el[0])]] = sent_tgt[int(el[1])]
            token_map[sent_tgt[int(el[1])]] = sent_src[int(el[0])]
        except IndexError:
            print("index error")
            print(sent_src, sent_tgt, alignments)
            print("-" * 20)
            token_map = None

    return token_map


# getting alignments and token map
def get_alignment_token_map(
    en_sent: str, hi_sent: str
) -> Tuple[str, Optional[Dict[str, str]]]:
    """
    Get alignments and token mapping between English and Hindi sentences.

    Args:
        en_sent (str): English sentence
        hi_sent (str): Hindi sentence

    Returns:
        Tuple[str, Optional[Dict[str, str]]]: Tuple containing:
            - alignments: Alignment string
            - token_alignment_map: Dictionary mapping aligned tokens
    """
    alignments = aligner.get_alignments_sentence_pair(en_sent, hi_sent)
    token_alignment_map = create_alignments_token_map(en_sent, hi_sent, alignments)
    return alignments, token_alignment_map


# generating alignments for POS tags
# heuristic - noun, adj, propn
def replace_noun_adj_single_aligned(
    sent: List[str], postags: List[str], token_map: Dict[str, str]
) -> str:
    """
    Generate a codemixed sentence by replacing nouns, adjectives, and proper nouns with their aligned counterparts.

    Args:
        sent (List[str]): List of tokens in the sentence
        postags (List[str]): List of POS tags corresponding to the tokens
        token_map (Dict[str, str]): Dictionary mapping tokens between languages

    Returns:
        str: Codemixed sentence with replaced tokens
    """
    codemixcandidate = ""

    for token, token_pos in zip(sent, postags):
        if "NOUN" in token_pos or "ADJ" in token_pos or "PROPN" in token_pos:
            if token in token_map:
                codemixcandidate += f" {token_map[token]}"
            else:
                codemixcandidate += f" {token}"
        else:
            codemixcandidate += f" {token}"

    return codemixcandidate


# generating codemix candidates for single row
def get_codemix_candidate(row: pd.Series) -> str:
    """
    Generate codemixed candidates for a single row of the dataframe.

    Args:
        row (pd.Series): Row from the dataframe containing tokenized sentences and alignments

    Returns:
        str: Generated codemixed sentence
    """
    sentence = ""
    for en_sent, en_pos, hi_sent, hi_pos, alignments, token_alignment_map in zip(
        row["lang1_tokens"],
        row["lang1_pos"],
        row["lang2_tokens"],
        row["lang2_pos"],
        row["alignments_awesomealign"],
        row["token_alignment_map_awesomealign"],
    ):
        sentence += replace_noun_adj_single_aligned(
            hi_sent, hi_pos, token_alignment_map
        )
    return sentence


def get_codemix_candidates_for_dataframe(df: pd.DataFrame) -> List[str]:
    """
    Generate codemixed candidates for all rows in the dataframe.

    Args:
        df (pd.DataFrame): DataFrame containing tokenized sentences and alignments

    Returns:
        List[str]: List of codemixed sentences
    """
    codemix_candidates = []

    for ind, row in tqdm(df.iterrows()):
        cm = get_codemix_candidate(row)
        codemix_candidates.append(cm)

    return codemix_candidates


def set_lang_tokens_postags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add tokenization and POS tagging information to the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe with source and target language columns

    Returns:
        pd.DataFrame: DataFrame with added tokenization and POS tagging columns
    """
    lang1_tokenized, lang1_pos, lang2_tokenized, lang2_pos = [], [], [], []
    for ind, row in tqdm(df.iterrows()):
        languages = row.keys()
        lang1, lang2 = [str(lang) for lang in languages]

        lang1_feats = get_stanza_info(row[lang1], lang1)
        lang2_feats = get_stanza_info(row[lang2], lang2)

        lang1_tokenized.append(lang1_feats["tokens"])
        lang2_tokenized.append(lang2_feats["tokens"])

        lang1_pos.append(lang1_feats["postags"])
        lang2_pos.append(lang2_feats["postags"])

    df["lang1"] = lang1
    df["lang1_tokens"] = lang1_tokenized
    df["lang1_pos"] = lang1_pos

    df["lang2"] = lang2
    df["lang2_tokens"] = lang2_tokenized
    df["lang2_pos"] = lang2_pos

    return df


def set_alignments_token_map(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add alignment and token mapping information to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with tokenized sentences

    Returns:
        pd.DataFrame: DataFrame with added alignment and token mapping columns
    """
    list_of_alignments, list_of_token_alignment_map = [], []

    for ind, row in tqdm(df.iterrows()):
        lang1_tokenized_sent = [
            " ".join(sent_list) for sent_list in row["lang1_tokens"]
        ]
        lang2_tokenized_sent = [
            " ".join(sent_list) for sent_list in row["lang2_tokens"]
        ]
        alignment_row, token_alignment_map_row = [], []
        try:
            assert len(lang1_tokenized_sent) == len(lang2_tokenized_sent)
            for lang1sent, lang2sent in zip(lang1_tokenized_sent, lang2_tokenized_sent):
                alignments, token_alignment_map = get_alignment_token_map(
                    lang1sent, lang2sent
                )
                alignment_row.append(alignments)
                token_alignment_map_row.append(token_alignment_map)
            list_of_alignments.append(alignment_row)
            list_of_token_alignment_map.append(token_alignment_map_row)

        except AssertionError:
            alignment_row, token_alignment_map_row = None, None
            list_of_alignments.append(None)
            list_of_token_alignment_map.append(None)

    df["alignments_awesomealign"] = list_of_alignments
    df["token_alignment_map_awesomealign"] = list_of_token_alignment_map

    return df


# main code
def get_codemix_candidates_for_file(filename: str) -> Optional[pd.DataFrame]:
    """
    Process a JSON file to generate codemixed sentences.

    Args:
        filename (str): Path to the input JSON file

    Returns:
        Optional[pd.DataFrame]: DataFrame with codemixed sentences, or None if processing fails
    """
    try:
        df_translations_set = pd.read_json(filename)
        df_translations_set = set_lang_tokens_postags(df_translations_set)
        df_translations_set = set_alignments_token_map(df_translations_set)

        codemix_candidates = get_codemix_candidates_for_dataframe(df_translations_set)
        df_translations_set["codemixed-sentences"] = codemix_candidates
        df_translations_set.to_json(
            "train_unique_utterances_en_hi_transltions_token_pos_alignments.json",
            force_ascii=False,
            orient="records",
            indent=4,
        )
    except:
        print("error")
        print(filename)
        df_translations_set = None
    print("done")
    return df_translations_set
