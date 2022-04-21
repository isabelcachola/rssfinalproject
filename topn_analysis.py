'''
Script to do analysis
'''
import argparse
import logging
import time
import torch
import numpy as np
import transformers
import itertools
import tqdm
import pandas as pd
import os
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

#example: python topn_analysis.py --lang_pair en-es --src "this is a test" --swap_idx 3 --swap_n 3
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang_pair', required=True)
    parser.add_argument('--src', required=True)
    parser.add_argument('--swap_idx', type=int, required=True)
    parser.add_argument('--swap_n', type=int)
    parser.add_argument('--swap_percent', type=float)
    args = parser.parse_args()

    if args.swap_percent and not 0 <= args.swap_percent <= 1:
        parser.error("swap_percent should be a decimal percentage between 0 and 1")
        
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    # -- swap analysis -- 
    start = time.time()
    input_lang, output_lang = args.lang_pair.split('-')

    #instantiate model
    model    = MTModel(args.lang_pair)
    src_idx  = args.swap_idx
    
    #standard setting
    src     = args.src
    out     = model.translation_from_string(src)
    s2t, _  = align(src,out)

    src_list = src.split()
    src_token_idx = model.tokenizer([src_list[ src_idx ]], return_tensors="pt", padding=True)
    src_token_idx = src_token_idx['input_ids'][0, 0].item()
    src_embed = model.get_embed_from_text(src)

    tgt = model.translation_from_string(src)
    tgt_embed = model.get_embed_from_text(tgt)


    # Get top N subwords 
    src_token_cos_sim = np.load(f'precomputed_cos_sims/{input_lang}/{input_lang}-{src_token_idx}.npz')['cos']
    if args.swap_n:
        swap_n = args.swap_n + 1 # Accounts for same word
    elif args.swap_percent:
        swap_n = int(src_token_cos_sim.shape[0] * args.swap_percent) + 1
        print(f"Swapping %0.3f%% yields %d swaps" % (args.swap_percent,swap_n))
    else:
        raise ValueError('Either --swap_n or --swap_percent required.')

    top_N_sim = ind = np.argpartition(src_token_cos_sim, -(swap_n))[-(swap_n):]
    swaps_tried = pd.DataFrame(columns=['swap_token_idx','swap_val', 'cos_input', 'cos_output', 'cos_diff'])
    for swap_token_idx in tqdm.tqdm(top_N_sim):
        if swap_token_idx != src_token_idx:
            src_swap = src_list
            swap_val = model.tokenizer.decode([swap_token_idx])
            src_swap[ src_idx ] = swap_val
            src_swap = ' '.join(src_swap)

            src_cos  = model.compute_cos(src_embed, model.get_embed_from_text(src_swap)).detach()
            # print("cossim between src (%s) and sub (%s) is: %f." % (src, src_swap, src_cos))

            #do swap
            tgt_swap = model.translation_from_string(src_swap)
            
            # swap_s2t, _  = align(src_swap,swap_out)
            
            #noised output
            # out_word = get_out_token(src_idx, s2t, out)
            # out_swap = get_out_token(src_idx, swap_s2t, swap_out)
            out_cos  = model.compute_cos(tgt_embed, model.get_embed_from_text(tgt_swap)).detach()
            # print("cossim between output (%s) and sub (%s) is: %f." % (tgt, tgt_swap, out_cos))    

            cos_diff = np.abs(out_cos - src_cos)
            # print(f'cossim diff = {cos_diff}')
            swaps_tried = pd.concat([swaps_tried, 
                                    pd.DataFrame.from_dict({
                                        'swap_token_idx': [swap_token_idx],
                                        'swap_val': [swap_val],
                                        'src_swap': [src_swap],
                                        'tgt_swap': [tgt_swap],
                                        'cos_input': [src_cos.item()], 
                                        'cos_output': [out_cos.item()],
                                        'cos_diff': [np.abs(out_cos - src_cos).item()]})
                                    ], 
                                    ignore_index=True
                                    )
    swaps_tried = swaps_tried.sort_values('cos_diff', ascending=False)
    print(swaps_tried.head(10))
    if not os.path.exists('output'):
        os.makedirs('output')
    swaps_tried.to_csv(f'output/{args.lang_pair}_{src_list[ src_idx ]}.csv', index=False)


    end = time.time()
    logging.info(f'Time to run script: {(end-start)/60} mins')
