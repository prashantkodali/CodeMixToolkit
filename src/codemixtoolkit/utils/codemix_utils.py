"""
Standardized metrics for code-switching
[1]: https://www.isca-speech.org/archive/Interspeech_2017/pdfs/1429.PDF
[2]: http://amitavadas.com/Pub/CMI.pdf
"""

from collections import Counter
import math
from statistics import stdev, mean
from statistics import StatisticsError
from typing import List, Dict, Optional, Union, Any


class CodeMixMetrics:
    """A class representing code-mixing metrics.

    This class provides static methods to compute various code-mixing metrics
    such as CMI (Code-Mixing Index), burstiness, I-index, language entropy,
    M-index, SPavg, and SyMCoM scores.
    """

    @staticmethod
    def compute_cmi(lid_tags: List[str], lang_tagset: List[str]) -> Optional[float]:
        """Compute the Code-Mixing Index (CMI).

        CMI measures the degree of code-mixing in a sentence.

        Args:
            lid_tags: List of language identification tags
            lang_tagset: List of language tags to consider

        Returns:
            CMI value between 0 and 100, or None if computation fails.
        """
        try:
            c = Counter(lid_tags)
            if len(c.most_common()) == 1:  # only one language is present
                print("One")
                return 0
            if len(c.most_common()) == 0:  # no language tokens.
                print("Zero")
                return 0

            max_wi = c.most_common()[0][1]
            n = len(lid_tags)
            u = sum([v for k, v in c.items() if k not in lang_tagset])
            if n == u:
                return 0
            return 100 * (1 - (max_wi / (n - u)))
        except:
            print("Error in cmi")
            return None

    @staticmethod
    def compute_mindex(
        lid_tags: List[str], lang_tagset: List[str], k: int = 2
    ) -> float:
        """Compute the M-index, a measure of language diversity.

        Args:
            lid_tags: List of language identification tags
            lang_tagset: List of language tags to consider
            k: Number of languages (default: 2)

        Returns:
            M-index value
        """
        c = Counter(lid_tags)
        total = sum([v for k, v in c.items() if k in lang_tagset])
        term = sum([(v / total) ** 2 for k, v in c.items() if k in lang_tagset])
        return (1 - term) / ((k - 1) * term)

    @staticmethod
    def compute_lang_entropy(
        lid_tags: List[str], lang_tagset: List[str], k: int = 2
    ) -> float:
        """Compute the language entropy of the sentence.

        Args:
            lid_tags: List of language identification tags
            lang_tagset: List of language tags to consider
            k: Number of languages (default: 2)

        Returns:
            Language entropy value
        """
        c = Counter(lid_tags)
        total = sum([v for k, v in c.items() if k in lang_tagset])
        terms = [(v / total) for k, v in c.items() if k in lang_tagset]
        result = sum([(i * math.log2(i)) for i in terms])
        return -result

    @staticmethod
    def compute_spavg(
        lid_tags: List[str], other_tagset: List[str], _: int = 2
    ) -> Optional[float]:
        """Compute the average number of switches per word.

        Args:
            lid_tags: List of language identification tags
            other_tagset: List of other tags to consider
            _: Unused parameter (kept for compatibility)

        Returns:
            Average number of switches per word, or None if computation fails
        """
        try:
            x = [el.lower() for el in lid_tags]

            count = 0
            mem = None
            for l_i, l_j in zip(x, x[1:]):
                if l_i in other_tagset:
                    continue
                if l_i != l_j:
                    count += 1

            return count

        except TypeError:
            return None

    @staticmethod
    def compute_i_index(
        lid_tags: List[str], other_tagset: List[str], k: int = 2
    ) -> float:
        """Compute the I-index, a measure of language alternation.

        Args:
            lid_tags: List of language identification tags
            other_tagset: List of other tags to consider
            k: Number of languages (default: 2)

        Returns:
            I-index value
        """
        x = [el.lower() for el in lid_tags]

        count = 0
        mem = None
        for l_i, l_j in zip(x, x[1:]):
            if l_i in other_tagset:
                continue
            if l_i != l_j:
                count += 1

        return count / (len(x) - 1)

    @staticmethod
    def compute_burstiness(
        lid_tags: List[str], other_tagset: List[str]
    ) -> Optional[float]:
        """Compute the burstiness of language switches.

        Args:
            lid_tags: List of language identification tags
            other_tagset: List of other tags to consider

        Returns:
            Burstiness value between -1 and 1, or None if computation fails
        """
        try:
            x = list(filter(lambda a: a not in other_tagset, lid_tags))
            spans = []
            cnt = 0
            prev = x[0]
            for i in x[1:]:
                if i == prev:
                    cnt += 1
                else:
                    spans.append(cnt + 1)
                    cnt = 0
                prev = i
            if cnt != 0:
                spans.append(cnt + 1)
            span_std = stdev(spans)
            span_mean = mean(spans)
            return (span_std - span_mean) / (span_std + span_mean)
        except StatisticsError:
            return None
        except TypeError:
            return None

    @staticmethod
    def compute_symcom_pos_tags(
        poS_count_map: Dict[str, int],
        lid_pos_count_map: Dict[str, int],
        l1: str,
        l2: str,
    ) -> Dict[str, float]:
        """Compute SyMCoM scores for each part-of-speech tag.

        Args:
            poS_count_map: Dictionary mapping POS tags to their counts
            lid_pos_count_map: Dictionary mapping combined LID-POS tags to their counts
            l1: Primary language code
            l2: Secondary language code

        Returns:
            Dictionary mapping POS tags to their SyMCoM scores
        """
        symcom_scores_pos_tags = {}

        for postag in poS_count_map:
            if postag not in ["PUNCT", "SYM", "X"]:
                pos_l1 = lid_pos_count_map.get(postag + "_" + l1, 0)
                pos_l2 = lid_pos_count_map.get(postag + "_" + l2, 0)

                try:
                    symcom_scores_pos_tags[postag + "_symcom"] = (pos_l1 - pos_l2) / (
                        pos_l1 + pos_l2
                    )
                except ZeroDivisionError:
                    pass

        return symcom_scores_pos_tags

    @staticmethod
    def compute_symcom_sentence(
        poS_count_map: Dict[str, int],
        lid_pos_count_map: Dict[str, int],
        l1: str,
        l2: str,
        length: int,
    ) -> float:
        """Compute the overall SyMCoM score for the sentence.

        Args:
            poS_count_map: Dictionary mapping POS tags to their counts
            lid_pos_count_map: Dictionary mapping combined LID-POS tags to their counts
            l1: Primary language code
            l2: Secondary language code
            length: Length of the sentence

        Returns:
            Overall SyMCoM score
        """
        symcom_sentence = 0
        symcom_scores_pos_tags = CodeMixMetrics.compute_symcom_pos_tags(
            poS_count_map, lid_pos_count_map, l1, l2
        )

        for pos, score in symcom_scores_pos_tags.items():
            pos = pos.split("_")[0]
            symcom_sentence += abs(score) * (poS_count_map[pos] / length)

        return symcom_sentence


class CodeMixSentence:
    """A class representing a code-mixed sentence with various metrics.

    This class provides functionality to compute various code-mixing metrics
    such as CMI (Code-Mixing Index), burstiness, I-index, language entropy,
    M-index, SPavg, and SyMCoM scores.

    Attributes:
        lang_tagset: List of language tags to consider
        other_tagset: List of other tags to consider
        l1: Primary language code
        l2: Secondary language code
        sentence: Original sentence text
        tokens: List of tokens in the sentence
        LID_Tags: Language identification tags
        PoS_Tags: Part-of-speech tags
    """

    def __init__(
        self,
        lang_tagset: Optional[List[str]] = None,
        other_tagset: Optional[List[str]] = None,
        l1: Optional[str] = None,
        l2: Optional[str] = None,
        sentence: Optional[str] = None,
        tokens: Optional[List[str]] = None,
        LID_Tags: Optional[List[str]] = None,
        PoS_Tags: Optional[List[str]] = None,
    ) -> None:
        """Initialize a CodeMixSentence object.

        Args:
            lang_tagset: List of language tags to consider
            other_tagset: List of other tags to consider
            l1: Primary language code
            l2: Secondary language code
            sentence: Original sentence text
            tokens: List of tokens in the sentence
            LID_Tags: Language identification tags
            PoS_Tags: Part-of-speech tags
        """
        self.lang_tagset = lang_tagset
        self.other_tagset = other_tagset

        self.lang_tagset = [tag.lower() for tag in self.lang_tagset]
        self.other_tagset = [tag.lower() for tag in self.other_tagset]

        self.l1 = l1
        self.l2 = l2
        self.sentence = sentence
        self.sentence_alternatives = None
        self.tokens = tokens
        self.LID_Tags = LID_Tags
        self.PoS_Tags = PoS_Tags
        # has to changed to take length from number of token

        self.LID_Tags = self._preprocess_LID_Tags()

        if self.LID_Tags is not None and self.PoS_Tags is not None:
            self.LID_count_map = dict(Counter(LID_Tags).most_common())
            self.PoS_count_map = dict(Counter(PoS_Tags).most_common())

            lid_pos_combined = [pos + "_" + lid for lid, pos in zip(LID_Tags, PoS_Tags)]
            self.LID_POS_count_map = dict(Counter(lid_pos_combined).most_common())
        else:
            self.LID_count_map = None
            self.PoS_count_map = None
            self.LID_POS_count_map = None

        self.length = len(LID_Tags)
        self.cmi: Optional[float] = None
        self.burstiness: Optional[float] = None
        self.i_index: Optional[float] = None
        self.lang_entropy: Optional[float] = None
        self.mindex: Optional[float] = None
        self.spavg: Optional[float] = None
        self.symcom_sentence: Optional[float] = None

    def __repr__(self) -> str:
        """Return a string representation of the CodeMixSentence object."""
        return (
            f"CodeMixSentenceCombined(\n"
            f"    lang_tagset={self.lang_tagset},\n"
            f"    other_tagset={self.other_tagset},\n"
            f"    l1='{self.l1}',\n"
            f"    l2='{self.l2}',\n"
            f"    sentence='{self.sentence}',\n"
            f"    tokens={self.tokens},\n"
            f"    LID_Tags={self.LID_Tags},\n"
            f"    PoS_Tags={self.PoS_Tags},\n"
            f"    length={self.length},\n"
            f"    cmi={self.cmi},\n"
            f"    burstiness={self.burstiness},\n"
            f"    i_index={self.i_index},\n"
            f"    lang_entropy={self.lang_entropy},\n"
            f"    mindex={self.mindex},\n"
            f"    spavg={self.spavg},\n"
            f"    symcom_sentence={self.symcom_sentence}\n"
            f")"
        )

    def compute_all_metrics(self) -> None:
        """Compute all available metrics for the code-mixed sentence."""
        self.cmi = CodeMixMetrics.compute_cmi(self.LID_Tags, self.lang_tagset)
        self.burstiness = CodeMixMetrics.compute_burstiness(
            self.LID_Tags, self.other_tagset
        )
        self.i_index = CodeMixMetrics.compute_i_index(self.LID_Tags, self.other_tagset)
        self.lang_entropy = CodeMixMetrics.compute_lang_entropy(
            self.LID_Tags, self.lang_tagset
        )
        self.mindex = CodeMixMetrics.compute_mindex(self.LID_Tags, self.lang_tagset)
        self.spavg = CodeMixMetrics.compute_spavg(self.LID_Tags, self.other_tagset)
        self.symcom_sentence = CodeMixMetrics.compute_symcom_sentence(
            self.PoS_count_map, self.LID_POS_count_map, self.l1, self.l2, self.length
        )

    def _preprocess_LID_Tags(self) -> List[str]:
        """Preprocess language identification tags.

        Returns:
            List of processed language identification tags.
        """
        x = self.LID_Tags
        if isinstance(x, str):
            x = x.split()
        x = [i for i in x if i in self.lang_tagset]
        return x

    def compute_cmi(self) -> Optional[float]:
        """Compute the Code-Mixing Index (CMI).

        CMI measures the degree of code-mixing in a sentence.

        Returns:
            CMI value between 0 and 100, or None if computation fails.
        """
        x = self.LID_Tags
        try:
            c = Counter(x)
            if len(c.most_common()) == 1:  # only one language is present
                print("One")
                return 0
            if len(c.most_common()) == 0:  # no language tokens.
                print("Zero")
                return 0

            max_wi = c.most_common()[0][1]
            n = len(x)
            u = sum([v for k, v in c.items() if k not in self.lang_tagset])
            if n == u:
                return 0
            return 100 * (1 - (max_wi / (n - u)))
        except:
            print("Error in cmi")
            return None

    def compute_mindex(self, k: int = 2) -> float:
        """Compute the M-index, a measure of language diversity.

        Args:
            k: Number of languages (default: 2)

        Returns:
            M-index value
        """
        x = self.LID_Tags

        c = Counter(x)
        total = sum([v for k, v in c.items() if k in self.lang_tagset])
        term = sum([(v / total) ** 2 for k, v in c.items() if k in self.lang_tagset])
        return (1 - term) / ((k - 1) * term)

    def compute_lang_entropy(self, k: int = 2) -> float:
        """Compute the language entropy of the sentence.

        Args:
            k: Number of languages (default: 2)

        Returns:
            Language entropy value
        """
        x = self.LID_Tags

        c = Counter(x)
        total = sum([v for k, v in c.items() if k in self.lang_tagset])
        terms = [(v / total) for k, v in c.items() if k in self.lang_tagset]
        result = sum([(i * math.log2(i)) for i in terms])
        return -result

    def compute_spavg(self, _: int = 2) -> Optional[float]:
        """Compute the average number of switches per word.

        Args:
            _: Unused parameter (kept for compatibility)

        Returns:
            Average number of switches per word, or None if computation fails
        """
        x = self.LID_Tags
        try:
            x = [el.lower() for el in x]

            count = 0
            mem = None
            for l_i, l_j in zip(x, x[1:]):
                if l_i in self.other_tagset:
                    continue
                if l_i != l_j:
                    count += 1

            return count

        except TypeError:
            return None

    def compute_i_index(self, k: int = 2) -> float:
        """Compute the I-index, a measure of language alternation.

        Args:
            k: Number of languages (default: 2)

        Returns:
            I-index value
        """
        x = self.LID_Tags
        x = [el.lower() for el in x]

        count = 0
        mem = None
        for l_i, l_j in zip(x, x[1:]):
            if l_i in self.other_tagset:
                continue
            if l_i != l_j:
                count += 1

        return count / (len(x) - 1)

    def compute_burstiness(self) -> Optional[float]:
        """Compute the burstiness of language switches.

        Returns:
            Burstiness value between -1 and 1, or None if computation fails
        """
        x = self.LID_Tags
        try:
            x = list(filter(lambda a: a not in self.other_tagset, x))
            spans = []
            cnt = 0
            prev = x[0]
            for i in x[1:]:
                if i == prev:
                    cnt += 1
                else:
                    spans.append(cnt + 1)
                    cnt = 0
                prev = i
            if cnt != 0:
                spans.append(cnt + 1)
            span_std = stdev(spans)
            span_mean = mean(spans)
            return (span_std - span_mean) / (span_std + span_mean)
        except StatisticsError:
            return None
        except TypeError:
            return None

    def compute_symcom_pos_tags(self) -> Dict[str, float]:
        """Compute SyMCoM scores for each part-of-speech tag.

        Returns:
            Dictionary mapping POS tags to their SyMCoM scores
        """
        symcom_scores_pos_tags = {}

        for postag in self.PoS_count_map:
            if postag not in ["PUNCT", "SYM", "X"]:
                pos_l1 = self.LID_POS_count_map.get(postag + "_" + self.l1, 0)
                pos_l2 = self.LID_POS_count_map.get(postag + "_" + self.l2, 0)

                try:
                    symcom_scores_pos_tags[postag + "_symcom"] = (pos_l1 - pos_l2) / (
                        pos_l1 + pos_l2
                    )
                except ZeroDivisionError:
                    pass

        return symcom_scores_pos_tags

    def compute_symcom_sentence(self) -> float:
        """Compute the overall SyMCoM score for the sentence.

        Returns:
            Overall SyMCoM score
        """
        symcom_sentence = 0
        symcom_scores_pos_tags = self.compute_symcom_pos_tags()

        for pos, score in symcom_scores_pos_tags.items():
            pos = pos.split("_")[0]
            symcom_sentence += abs(score) * (self.PoS_count_map[pos] / self.length)

        return symcom_sentence
