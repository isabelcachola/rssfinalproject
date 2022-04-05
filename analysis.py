'''
Script to do analysis
'''
import argparse
import logging
import time
import torch
import transformers
import itertools
from collections import defaultdict

from models import MTModel

# Use with care: logging error only while printing analysis for reading sanity
transformers.utils.logging.set_verbosity_error()


def output_diff(alignment, translation):
    pass

def get_out_token(src_idx, s2t, output):
    #get 1-best ali
    out_idx = list(s2t[src_idx])[0]
    #get token from idx
    tmp = output.split()
    out_token = tmp[out_idx]
    return out_token

# Align source and target word sequences with the awesome aligner (expects non-tokenized input) 
def align(src, tgt):
    model = transformers.BertModel.from_pretrained('bert-base-multilingual-cased')
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    # pre-processing
    sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
    token_src, token_tgt = [tokenizer.tokenize(word) for word in sent_src], [tokenizer.tokenize(word) for word in sent_tgt]
    wid_src, wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_src], [tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
    ids_src, ids_tgt = tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', model_max_length=tokenizer.model_max_length, truncation=True)['input_ids'], tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', truncation=True, model_max_length=tokenizer.model_max_length)['input_ids']
    sub2word_map_src = []
    for i, word_list in enumerate(token_src):
        sub2word_map_src += [i for x in word_list]
        sub2word_map_tgt = []
    for i, word_list in enumerate(token_tgt):
        sub2word_map_tgt += [i for x in word_list]

    # alignment
    align_layer = 8
    threshold = 1e-3
    model.eval()
    with torch.no_grad():
        out_src = model(ids_src.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
        out_tgt = model(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]

        dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))

        softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
        softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

        softmax_inter = (softmax_srctgt > threshold)*(softmax_tgtsrc > threshold)

    # src2tgt is a dict mapping src words to their set of aligned tgt words; align_words is the set of alignments for printing alis etc
    align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
    align_words = set()
    src2tgt = defaultdict(set)
    for i, j in align_subwords:
        align_words.add( (sub2word_map_src[i], sub2word_map_tgt[j]) )
        src2tgt[sub2word_map_src[i]].add(sub2word_map_tgt[j])

    return src2tgt, align_words

def print_alignments(align_words):
    for i, j in sorted(align_words):
        print(f'{color.BOLD}{color.BLUE}{sent_src[i]}{color.END}==={color.BOLD}{color.RED}{sent_tgt[j]}{color.END}')
    return
    
# printing
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

#example: python analysis.py --lang_pair en-es --src "this is a test" --swap_idx 3 --swap_val sentence
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang_pair')
    parser.add_argument('--src')
    parser.add_argument('--swap_idx', action='store', type=int)
    parser.add_argument('--swap_val')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    # -- swap analysis -- 
    start = time.time()

    #instantiate model
    model    = MTModel(args.lang_pair)
    src_idx  = args.swap_idx
    
    #standard setting
    src     = args.src
    out     = model.translation_from_string(src)
    s2t, _  = align(src,out)

    #noised source
    src_word = src.split()[ src_idx ]
    src_swap = args.swap_val
    src_cos  = model.compute_cos(model.get_embed_from_text(src_word), model.get_embed_from_text(src_swap))
    print("cossim between src (%s) and sub (%s) is: %f." % (src_word, src_swap, src_cos))

    #do swap
    tmp = src.split()
    tmp[src_idx] = src_swap
    swap_src = ' '.join(tmp)
    swap_out = model.translation_from_string(swap_src)
    swap_s2t, _  = align(swap_src,swap_out)
    
    #noised output
    out_word = get_out_token(src_idx, s2t, out)
    out_swap = get_out_token(src_idx, swap_s2t, swap_out)
    out_cos  = model.compute_cos(model.get_embed_from_text(out_word), model.get_embed_from_text(out_swap))
    print("cossim between output (%s) and sub (%s) is: %f." % (out_word, out_swap, out_cos))    

    print(out)
    print(swap_out)

    end = time.time()
    logging.info(f'Time to run script: {end-start} secs')
