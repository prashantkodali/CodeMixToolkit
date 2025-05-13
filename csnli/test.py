from three_step_decoding import *

tsd = ThreeStepDecoding(
    "lid_models/hinglish",
    htrans="nmt_models/rom2hin.pt",
    etrans="nmt_models/eng2eng.pt",
)


print(
    "\n".join(["\t".join(x) for x in tsd.tag_sent("i thght mosam dfrnt hoga bs fog h")])
)

def transliterate(text):
    try:
        op = tsd.tag_sent(text)
    except ValueError:
        print(f"Exception ValueError for : {text}")
        op = None
    except:
        print(f"Exception for : {text}")
        op = None

    if op:
        t, nt, l = [], [], []
        for el in op:
            t.append(el[0])
            nt.append(el[1])
            l.append(el[2])

    else:
        t = text.split()
        nt = (None,)
        l = None

    d = {"og_text": text, "text": t, "norm_text": nt, "lid": l}

    td = {"csnli_op": d}

    return td



text = "i thght mosam dfrnt hoga bs fog h"
response = transliterate(text)
print(response)



##########
#indictrans

from indictrans import Transliterator
trn = Transliterator(source='hin', target='eng', build_lookup=True)
hin = """कांग्रेस पार्टी अध्यक्ष सोनिया गांधी, तमिलनाडु की मुख्यमंत्री जयललिता और रिज़र्व बैंक के गवर्नर रघुराम राजन के बीच एक समानता है. ये सभी अलग-अलग कारणों से भारतीय जनता पार्टी के राज्यसभा सांसद सुब्रमण्यम स्वामी के निशाने पर हैं. उनके जयललिता और सोनिया गांधी के पीछे पड़ने का कारण कथित रष्टाचार है."""
eng = trn.transform(hin)
print(eng)

