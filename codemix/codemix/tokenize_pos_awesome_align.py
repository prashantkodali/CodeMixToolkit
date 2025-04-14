import os
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

aligner = awesomealign(modelpath = 'bert-base-multilingual-cased',
                      tokenizerpath = 'bert-base-multilingual-cased')  

# setting up stanza pipelines
def get_stanza_info(text, language): #TO ADD - language parameter
    # language accepts - en, hi, ta, te
    nlp_lang = stanza.Pipeline(lang=language, processors='tokenize, pos')
    doc = nlp_lang(text)
    
    sents, tokens, postags = [], [], []
    
    for sentence in doc.sentences:
        sents.append(' '.join([f'{token.text}' for token in sentence.tokens]))
        tokens.append([f'{token.text}' for token in sentence.words])
        postags.append([f'{token.upos}' for token in sentence.words])

    return {"sentences" : sents,
           "tokens" : tokens,
           "postags" : postags}

# creating token alignment map
def create_alignments_token_map(sent_src, sent_tgt, alignments):
    
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
            print("-"*20)
            token_map = None
    
    return token_map

# getting alignments and token map
def get_alignment_token_map(en_sent, hi_sent):
    alignments = aligner.get_alignments_sentence_pair(en_sent, hi_sent)
    token_alignment_map = create_alignments_token_map(en_sent, hi_sent, alignments)
    return alignments, token_alignment_map

#generating alignments for POS tags
#heuristic - noun, adj, propn
def replace_noun_adj_single_aligned(sent, postags, token_map):
    
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

#generating codemix candidates for single row
def get_codemix_candidate(row):
    sentence = ""
    for en_sent, en_pos, hi_sent, hi_pos, alignments, token_alignment_map in zip(row["lang1_tokens"], row["lang1_pos"], row["lang2_tokens"], row["lang2_pos"], row["alignments_awesomealign"], row["token_alignment_map_awesomealign"]):
        sentence += replace_noun_adj_single_aligned(hi_sent, hi_pos, token_alignment_map)
    return sentence

            
def get_codemix_candidates_for_dataframe(df):
    codemix_candidates = []

    for ind, row in tqdm(df.iterrows()):
        cm = get_codemix_candidate(row)
        codemix_candidates.append(cm)
        
    return codemix_candidates

def set_lang_tokens_postags(df):
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

def set_alignments_token_map(df):
    list_of_alignments, list_of_token_alignment_map = [], []

    for ind, row in tqdm(df.iterrows()):
        lang1_tokenized_sent = [" ".join(sent_list) for sent_list in row["lang1_tokens"]]
        lang2_tokenized_sent = [" ".join(sent_list) for sent_list in row["lang2_tokens"]]
        alignment_row, token_alignment_map_row = [], []
        try:
            assert len(lang1_tokenized_sent) == len(lang2_tokenized_sent)
            for lang1sent, lang2sent in zip(lang1_tokenized_sent, lang2_tokenized_sent):
                alignments, token_alignment_map = get_alignment_token_map(lang1sent, lang2sent)
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
def get_codemix_candidates_for_file(filename):
    try:
        df_translations_set = pd.read_json(filename)
        df_translations_set = set_lang_tokens_postags(df_translations_set)
        df_translations_set = set_alignments_token_map(df_translations_set)

        codemix_candidates = get_codemix_candidates_for_dataframe(df_translations_set)
        df_translations_set["codemixed-sentences"] = codemix_candidates
        df_translations_set.to_json("train_unique_utterances_en_hi_transltions_token_pos_alignments.json", 
                                                force_ascii = False, 
                                                orient = "records",
                                                indent = 4)
    except:
        print("error")
        print(filename)
        df_translations_set = None
    print("done")
    return df_translations_set



