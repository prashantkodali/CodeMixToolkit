"""
Standardized metrics for code-switching
[1]: https://www.isca-speech.org/archive/Interspeech_2017/pdfs/1429.PDF
[2]: http://amitavadas.com/Pub/CMI.pdf
"""

from collections import Counter
import math
from statistics import stdev, mean
from statistics import StatisticsError


class CodeMixSentence:
    def __init__(
        self,
        lang_tagset=None,
        other_tagset=None,
        l1=None,
        l2=None,
        sentence=None,
        tokens=None,
        LID_Tags=None,
        PoS_Tags=None,
    ):
        self.lang_tagset = lang_tagset
        self.other_tagset = other_tagset

        self.lang_tagset = [tag.lower() for tag in self.lang_tagset]
        self.other_tagset = [tag.lower() for tag in self.other_tagset]

        self.l1 = l1
        self.l2 = l2
        self.sentence = sentence
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

        self.length = len(LID_Tags)
        self.cmi = None
        self.burstiness = None
        self.i_index = None
        self.lang_entropy = None
        self.mindex = None
        self.spavg = None
        self.symcom_sentence = None

    def __repr__(self):
        # Create a concise representation of the object
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

    def compute_all_metrics(self):
        self.cmi = self.compute_cmi()
        self.burstiness = self.compute_burstiness()
        self.i_index = self.compute_i_index()
        self.lang_entropy = self.compute_lang_entropy()
        self.mindex = self.compute_mindex()
        self.spavg = self.compute_spavg()
        self.symcom_sentence = self.compute_symcom_sentence()

    def _preprocess_LID_Tags(self):
        x = self.LID_Tags
        if isinstance(x, str):
            x = x.split()
        x = [i for i in x if i in self.lang_tagset]
        return x

    def compute_cmi(self):
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

    def compute_mindex(self, k=2):
        x = self.LID_Tags

        c = Counter(x)
        total = sum([v for k, v in c.items() if k in self.lang_tagset])
        term = sum([(v / total) ** 2 for k, v in c.items() if k in self.lang_tagset])
        return (1 - term) / ((k - 1) * term)

    def compute_lang_entropy(self, k=2):
        x = self.LID_Tags

        c = Counter(x)
        total = sum([v for k, v in c.items() if k in self.lang_tagset])
        terms = [(v / total) for k, v in c.items() if k in self.lang_tagset]
        result = sum([(i * math.log2(i)) for i in terms])
        return -result

    def compute_spavg(self, _=2):
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

    def compute_i_index(self, k=2):
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

    def compute_burstiness(self):
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

    def compute_symcom_pos_tags(self):
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

    def compute_symcom_sentence(self):
        symcom_sentence = 0
        symcom_scores_pos_tags = self.compute_symcom_pos_tags()

        for pos, score in symcom_scores_pos_tags.items():
            pos = pos.split("_")[0]
            symcom_sentence += abs(score) * (self.PoS_count_map[pos] / self.length)

        return symcom_sentence


class CodeMixSentenceTemp:
    def __init__(self, sentence=None, tokens=None, LID_Tags=None, PoS_Tags=None):
        self.sentence = sentence
        self.tokens = tokens
        self.LID_Tags = LID_Tags
        self.PoS_Tags = PoS_Tags
        # has to changed to take length from number of token
        self.length = len(LID_Tags)

        self.LID_count_map = dict(Counter(LID_Tags).most_common())
        self.PoS_count_map = dict(Counter(PoS_Tags).most_common())

        lid_pos_combined = [pos + "_" + lid for lid, pos in zip(LID_Tags, PoS_Tags)]
        self.LID_POS_count_map = dict(Counter(lid_pos_combined).most_common())


class CodeMixMetricsTemp:
    def __init__(self, lang_tags, other_tags):
        self.lang_tags = [tag.lower() for tag in lang_tags]
        self.other_tags = [tag.lower() for tag in other_tags]

    def _preprocess_text(self, x):
        if isinstance(x, str):
            x = x.split()
        x = [i for i in x if i in self.lang_tags]
        return x

    def cmi(self, x):
        try:
            x = self._preprocess_text(x)
            c = Counter(x)
            if len(c.most_common()) == 1:  # only one language is present
                print("One")
                return 0
            if len(c.most_common()) == 0:  # no language tokens.
                print("Zero")
                return 0

            max_wi = c.most_common()[0][1]
            n = len(x)
            u = sum([v for k, v in c.items() if k not in self.lang_tags])
            if n == u:
                return 0
            return 100 * (1 - (max_wi / (n - u)))
        except:
            return None

    def mindex(self, x, k=2):
        x = self._preprocess_text(x)
        c = Counter(x)
        total = sum([v for k, v in c.items() if k in self.lang_tags])
        term = sum([(v / total) ** 2 for k, v in c.items() if k in self.lang_tags])
        return (1 - term) / ((k - 1) * term)

    def lang_entropy(self, x, k=2):
        x = self._preprocess_text(x)
        c = Counter(x)
        total = sum([v for k, v in c.items() if k in self.lang_tags])
        terms = [(v / total) for k, v in c.items() if k in self.lang_tags]
        result = sum([(i * math.log2(i)) for i in terms])
        return -result

    def spavg(self, x, _=2):
        try:
            x = [el.lower() for el in x]

            x = self._preprocess_text(x)
            count = 0
            mem = None
            for l_i, l_j in zip(x, x[1:]):
                if l_i in self.other_tags:
                    continue
                if l_i != l_j:
                    count += 1

            return count

        except TypeError:
            return None

    def i_index(self, x, k=2):
        x = [el.lower() for el in x]

        x = self._preprocess_text(x)
        count = 0
        mem = None
        for l_i, l_j in zip(x, x[1:]):
            if l_i in self.other_tags:
                continue
            if l_i != l_j:
                count += 1

        return count / (len(x) - 1)

    def burstiness(self, x):
        try:
            x = self._preprocess_text(x)
            x = list(filter(lambda a: a not in self.other_tags, x))
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


class SyMCoMTemp:
    def __init__(self, LID_tagset=None, PoS_tagset=None, L1=None, L2=None):
        self.LID_tagset = LID_tagset
        self.PoS_tagset = PoS_tagset
        self.l1 = L1
        self.l2 = L2

    def symcom_pos_tags(self, codemixsent_obj):
        symcom_scores_pos_tags = {}
        l1 = self.l1
        l2 = self.l2

        for postag in codemixsent_obj.PoS_count_map:
            if postag not in ["PUNCT", "SYM", "X"]:
                pos_l1 = codemixsent_obj.LID_POS_count_map.get(
                    postag + "_" + self.l1, 0
                )
                pos_l2 = codemixsent_obj.LID_POS_count_map.get(
                    postag + "_" + self.l2, 0
                )

                try:
                    symcom_scores_pos_tags[postag + "_symcom"] = (pos_l1 - pos_l2) / (
                        pos_l1 + pos_l2
                    )
                except ZeroDivisionError:
                    pass

        return symcom_scores_pos_tags

    def symcom_sentence(self, codemixsent_obj):
        symcom_sentence = 0
        symcom_scores_pos_tags = self.symcom_pos_tags(codemixsent_obj)

        for pos, score in symcom_scores_pos_tags.items():
            pos = pos.split("_")[0]
            symcom_sentence += abs(score) * (
                codemixsent_obj.PoS_count_map[pos] / codemixsent_obj.length
            )

        return symcom_sentence


"""
In [8]: symcom = cs_metrics.SyMCoM(L1 = 'en',
   ...:                 L2 = 'hi',
   ...:                 LID_tagset = ['hi', 'en', 'ne', 'univ', 'acro'],
   ...:                 PoS_tagset = ['NOUN', 'ADV', 'VERB', 'AUX', 'ADJ', 'ADP', 'PUNCT', 'DET', 'PRON', 'PROPN', 'PART', 'CCONJ', 'SCONJ', 'INTJ', 'NUM', 'SYM','X'])

In [9]: tokens = ['Gully', 'cricket', 'चल', 'रहा', 'हैं', 'यहां', '"', '(', 'Soniya', ')', 'Gandhi', '"']
   ...: LID_Tags = ['en', 'en', 'hi', 'hi', 'hi', 'hi', 'univ', 'univ', 'ne', 'univ', 'ne', 'univ']
   ...: PoS_Tags = ['ADJ', 'PROPN', 'VERB', 'AUX', 'AUX', 'ADV', 'PUNCT', 'PUNCT', 'PROPN', 'PUNCT', 'PROPN', 'PUNCT']
   ...: 
   ...: cm_sentence = cs_metrics.CodeMIxSentence(sentence = None,
   ...:                                      tokens = tokens,
   ...:                                      LID_Tags = LID_Tags,
   ...:                                      PoS_Tags = PoS_Tags)

In [10]: symcom.symcom_pos_tags(cm_sentence)
Out[10]: 
{'PROPN_symcom': 1.0,
 'AUX_symcom': -1.0,
 'ADJ_symcom': 1.0,
 'VERB_symcom': -1.0,
 'ADV_symcom': -1.0}

In [11]: symcom.symcom_sentence(cm_sentence)
Out[11]: 0.6666666666666666

In [12]: 


"""
