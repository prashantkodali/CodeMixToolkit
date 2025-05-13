from indicnlp.tokenize import indic_tokenize
import re
import torch
import transformers
import itertools
from tqdm import tqdm
from typing import List, Set, Tuple, Iterable


class awesomealign:
    """A class for performing word alignment between source and target sentences using AwesomeAlign model.

    This class uses BERT-based models to compute word alignments between pairs of sentences.
    It supports both single sentence pair alignment and batch processing of multiple sentence pairs.
    """

    def __init__(self, modelpath: str, tokenizerpath: str) -> None:
        """Initialize the AwesomeAlign model and tokenizer.

        Args:
            modelpath (str): Path to the pre-trained BERT model
            tokenizerpath (str): Path to the BERT tokenizer
        """
        self.model = transformers.BertModel.from_pretrained(modelpath)
        self.tokenizer = transformers.BertTokenizer.from_pretrained(tokenizerpath)

    def get_alignments_sentence_pair(self, src: str, tgt: str) -> str:
        """Get word alignments between a source and target sentence pair.

        Args:
            src (str): Source sentence
            tgt (str): Target sentence

        Returns:
            str: Space-separated string of word alignments in the format "src_idx-tgt_idx"
        """
        sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
        token_src, token_tgt = (
            [self.tokenizer.tokenize(word) for word in sent_src],
            [self.tokenizer.tokenize(word) for word in sent_tgt],
        )
        wid_src, wid_tgt = (
            [self.tokenizer.convert_tokens_to_ids(x) for x in token_src],
            [self.tokenizer.convert_tokens_to_ids(x) for x in token_tgt],
        )
        ids_src, ids_tgt = (
            self.tokenizer.prepare_for_model(
                list(itertools.chain(*wid_src)),
                return_tensors="pt",
                model_max_length=self.tokenizer.model_max_length,
                truncation=True,
            )["input_ids"],
            self.tokenizer.prepare_for_model(
                list(itertools.chain(*wid_tgt)),
                return_tensors="pt",
                truncation=True,
                model_max_length=self.tokenizer.model_max_length,
            )["input_ids"],
        )
        sub2word_map_src = []
        for i, word_list in enumerate(token_src):
            sub2word_map_src += [i for x in word_list]
        sub2word_map_tgt = []
        for i, word_list in enumerate(token_tgt):
            sub2word_map_tgt += [i for x in word_list]

        # alignment
        align_layer = 8
        threshold = 1e-3
        self.model.eval()
        with torch.no_grad():
            out_src = self.model(ids_src.unsqueeze(0), output_hidden_states=True)[2][
                align_layer
            ][0, 1:-1]
            out_tgt = self.model(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][
                align_layer
            ][0, 1:-1]

            dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))

            softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
            softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

            softmax_inter = (softmax_srctgt > threshold) * (softmax_tgtsrc > threshold)

        align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
        align_words: Set[Tuple[int, int]] = set()
        for i, j in align_subwords:
            align_words.add((sub2word_map_src[i], sub2word_map_tgt[j]))

        st = ""
        for el in sorted(align_words):
            st += f"{el[0]}-{el[1]} "

        return st

    def get_alignments_iter(
        self, src_iterable: Iterable[str], tgt_iterable: Iterable[str]
    ) -> List[str]:
        """Get word alignments for multiple sentence pairs.

        Args:
            src_iterable (Iterable[str]): Iterable of source sentences
            tgt_iterable (Iterable[str]): Iterable of target sentences

        Returns:
            List[str]: List of alignment strings for each sentence pair
        """
        align_words_align: List[str] = []
        for src, tgt in tqdm(zip(src_iterable, tgt_iterable)):
            align_words = self.get_alignments_sentence_pair(src, tgt)
            align_words_align.append(align_words)

        return align_words_align
