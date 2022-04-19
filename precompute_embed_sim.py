'''
This is a simple example of multiprocessing
'''
import argparse
import logging
import tqdm
import time
import itertools
import torch
import multiprocessing
import os
import numpy as np
from models import MTModel



def compute_cosines(model, lang):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    input_embeddings = model.model.get_input_embeddings().weight.detach()
    vocab_size = input_embeddings.shape[0]
    print(vocab_size)

    for i in tqdm.trange(vocab_size):
        file_name = f'precomputed_cos_sims/{lang}/{lang}-{i}'
        if not os.path.exists(file_name):
            embed_i = model.get_embed_from_ind(i).detach() 
            cos_sims = cos(embed_i.unsqueeze(0), input_embeddings)
            np.savez_compressed(file_name, cos=cos_sims)
                

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('lang_pair')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,format='%(levelname)s - %(message)s')


    start = time.time()
    model = MTModel(args.lang_pair)
    compute_cosines(model, args.lang_pair.split('-')[0])
    end = time.time()
    logging.info(f'Time to run script: {end-start} secs')