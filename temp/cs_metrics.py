'''
Standardized metrics for code-switching
[1]: https://www.isca-speech.org/archive/Interspeech_2017/pdfs/1429.PDF
[2]: http://amitavadas.com/Pub/CMI.pdf
'''

from collections import Counter
import math
from statistics import stdev, mean
from statistics import StatisticsError

LANG_TAGS = ['en', 'hi']
OTHER_TAGS = ['univ', 'ne','acro']

def cmi(x):
    try:
        if isinstance(x, str):
            x = x.split()
        x = [i for i in x if i in LANG_TAGS]
        c = Counter(x)
        if len(c.most_common()) == 1:  #only one language is present
            return 0
        if len(c.most_common()) == 0: #no language tokens. 
            return 0

        max_wi = c.most_common()[0][1]
        n = len(x)
        u = sum([v for k,v in c.items() if k not in LANG_TAGS])
        if n==u: return 0
        return 100 * (1 - (max_wi/(n-u)))

    except:
        return None

def mindex(x, k=2):
    x = x.split()
    x = [i for i in x if i in LANG_TAGS]
    c = Counter(x)
    total = sum([v for k,v in c.items() if k in LANG_TAGS])
    term = sum([(v/total)**2 for k,v in c.items() if k in LANG_TAGS])
    return (1-term)/((k-1)*term)

def lang_entropy(x, k=2):
    x = x.split()
    x = [i for i in x if i in LANG_TAGS]
    c = Counter(x)
    total = sum([v for k,v in c.items() if k in LANG_TAGS])
    terms = [(v/total) for k,v in c.items() if k in LANG_TAGS]
    result = sum([(i*math.log2(i)) for i in terms])
    return -result

def spavg(x, k= 2):
    LANG_TAGS = ['en', 'hi']
    OTHER_TAGS = ['univ', 'ne','acro']

    LANG_TAGS = [tag.lower() for tag in LANG_TAGS]
    OTHER_TAGS = [tag.lower() for tag in OTHER_TAGS]
    try:
      
      x = [el.lower() for el in x]

      if isinstance(x,str):
          x = x.split()

      x = [i for i in x if i in LANG_TAGS]


      count = 0 
      mem = None
      for l_i, l_j in zip(x,x[1:]):
          if l_i in OTHER_TAGS:
              continue
          if l_i != l_j:
              count+=1

      return count 
    
    except TypeError:
      return None


def i_index(x,k=2):

    LANG_TAGS = [tag.lower() for tag in LANG_TAGS]
    OTHER_TAGS = [tag.lower() for tag in OTHER_TAGS]

    x = [el.lower() for el in x]

    if isinstance(x,str):
        x = x.split()

    x = [i for i in x if i in LANG_TAGS]

    count = 0 
    mem = None
    for l_i, l_j in zip(x,x[1:]):
        if l_i in OTHER_TAGS:
            continue
        if l_i != l_j:
            count+=1

    return count/(len(x) - 1) 





def burstiness(x):
    try:
      if isinstance(x, str):
          x = x.split()
      x = [i for i in x if i in LANG_TAGS]
      x = list(filter(lambda a: a not in OTHER_TAGS, x))
      spans = []
      cnt = 0
      prev = x[0]
      for i in x[1:]:
          if i==prev: cnt+=1
          else:
              spans.append(cnt+1)
              cnt=0
          prev = i
      if cnt!=0: spans.append(cnt+1)
      span_std = stdev(spans)
      span_mean = mean(spans)
      return ((span_std - span_mean)/(span_std+span_mean))
    except StatisticsError:
      return None
    except TypeError:
      return None


class CodeMIxSentence():
    
    def __init__(self, sentence = None,
                         tokens = None,
                         LID_Tags = None,
                         PoS_Tags = None):
        
        self.sentence = sentence
        self.tokens = tokens
        self.LID_Tags = LID_Tags
        self.PoS_Tags = PoS_Tags
        self.length = len(LID_Tags) # has to changed to take length from number of token
        
        self.LID_count_map = dict(Counter(LID_Tags).most_common())
        self.PoS_count_map = dict(Counter(PoS_Tags).most_common())      
        
        lid_pos_combined = []
        
        for lid,pos in zip(self.LID_Tags, self.PoS_Tags):
            
            lid_pos_combined.append(pos+'_'+lid)
            
        self.LID_POS_count_map = dict(Counter(lid_pos_combined).most_common())


class SyMCoM:
    
    def __init__(self, 
                LID_tagset = None,
                PoS_tagset = None, 
                L1 = None, 
                L2 = None):
        
        self.LID_tagset = LID_tagset
        self.PoS_tagset = PoS_tagset
        self.l1 = L1
        self.l2 = L2
        
        
    def symcom_pos_tags(self, codemixsent_obj):
        
        symcom_scores_pos_tags = {}
        
        l1 = self.l1
        l2 = self.l2
        

            
                
        for postag in codemixsent_obj.PoS_count_map:
            
            if postag not in ['PUNCT','SYM', 'X']:
            
                pos_l1 = codemixsent_obj.LID_POS_count_map[postag+'_'+self.l1] if postag+'_'+self.l1 in codemixsent_obj.LID_POS_count_map else 0
                pos_l2 = codemixsent_obj.LID_POS_count_map[postag+'_'+self.l2] if postag+'_'+self.l2 in codemixsent_obj.LID_POS_count_map else 0
                
                try:
                    symcom_scores_pos_tags[postag+'_symcom'] = (pos_l1 - pos_l2) / (pos_l1 + pos_l2) 
                    
                except ZeroDivisionError: # to handle cases where there the LID for pos tag is not en or hi. 
                    pass
        return symcom_scores_pos_tags
    
    
        
        
    def symcom_sentence(self, codemixsent_obj):
        
        symcom_sentence = 0
        
        symcom_scores_pos_tags = self.symcom_pos_tags(codemixsent_obj)
        
        for pos, score in symcom_scores_pos_tags.items():
            pos = pos.split('_')[0]
            symcom_sentence += abs(score) * (codemixsent_obj.PoS_count_map[pos] / codemixsent_obj.length)     
            
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