from indicnlp.tokenize import indic_tokenize  
import re
import torch
import transformers
import itertools
import tqdm
import os
import subprocess
import sys
import threading

def popen_io(cmd):
#     p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1,
                         universal_newlines=True)
    def consume(s):
        for _ in s:
            pass
    threading.Thread(target=consume, args=(p.stderr,)).start()
    return p

# commit 23 - added statistical aligner
class statisticalign:

    def __init__(self, fwd_params, fwd_err, rev_params, rev_err, heuristic='grow-diag-final-and'):
        build_root = os.path.dirname(os.path.abspath(__file__))
        fast_align = os.path.join(build_root, "fast_align-master", "build", 'fast_align')
        atools = os.path.join(build_root, "fast_align-master","build", 'atools')

        (fwd_T, fwd_m) = self.read_err(fwd_err)
        (rev_T, rev_m) = self.read_err(rev_err)

        fwd_cmd = [fast_align, '-i', '-', '-d', '-T', fwd_T, '-m', fwd_m, '-f', fwd_params]
        rev_cmd = [fast_align, '-i', '-', '-d', '-T', rev_T, '-m', rev_m, '-f', rev_params, '-r']
        tools_cmd = [atools, '-i', '-', '-j', '-', '-c', heuristic]

        self.fwd_align = popen_io(fwd_cmd)
        self.rev_align = popen_io(rev_cmd)
        self.tools = popen_io(tools_cmd)

    def align(self, line):
        print(type(line))
        print(line)
        self.fwd_align.stdin.write('{}\n'.format(line))
        self.rev_align.stdin.write('{}\n'.format(line))
        # f words ||| e words ||| links ||| score
        fwd_line = self.fwd_align.stdout.readline().split('|||')[2].strip()
        rev_line = self.rev_align.stdout.readline().split('|||')[2].strip()
        self.tools.stdin.write('{}\n'.format(fwd_line))
        self.tools.stdin.write('{}\n'.format(rev_line))
        al_line = self.tools.stdout.readline().strip()
        return al_line
 
    def close(self):
        self.fwd_align.stdin.close()
        self.fwd_align.wait()
        self.rev_align.stdin.close()
        self.rev_align.wait()
        self.tools.stdin.close()
        self.tools.wait()

    def read_err(self, err):
        (T, m) = ('', '')
        for line in open(err):
            # expected target length = source length * N
            if 'expected target length' in line:
                m = line.split()[-1]
            # final tension: N
            elif 'final tension' in line:
                T = line.split()[-1]
        return (T, m)   



class awesomealign:    
    
    def __init__(self, modelpath, tokenizerpath):
        self.model = transformers.BertModel.from_pretrained(modelpath)
        self.tokenizer = transformers.BertTokenizer.from_pretrained(tokenizerpath) 

        
    def get_alignments_sentence_pair(self,src, tgt):
        sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
        token_src, token_tgt = [self.tokenizer.tokenize(word) for word in sent_src], [self.tokenizer.tokenize(word) for word in sent_tgt]
        wid_src, wid_tgt = [self.tokenizer.convert_tokens_to_ids(x) for x in token_src], [self.tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
        ids_src, ids_tgt = self.tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', model_max_length=self.tokenizer.model_max_length, truncation=True)['input_ids'], self.tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', truncation=True, model_max_length=self.tokenizer.model_max_length)['input_ids']
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
            out_src = self.model(ids_src.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
            out_tgt = self.model(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]

            dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))

            softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
            softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

            softmax_inter = (softmax_srctgt > threshold)*(softmax_tgtsrc > threshold)

        align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
        align_words = set()
        for i, j in align_subwords:
            align_words.add( (sub2word_map_src[i], sub2word_map_tgt[j]) )
            
        
        st = ""
        for el in sorted(align_words):
            st+=f"{el[0]}-{el[1]} "

            

        return st


    def get_alignments_iter(self, src_iterable, tgt_iterable):
        align_words_align = []
        for src, tgt in tqdm(zip(src_iterable, tgt_iterable)):
            align_words = self.get_alignments_sentence_pair(src, tgt)
            align_words_align.append(align_words)
            
        return align_words_align




