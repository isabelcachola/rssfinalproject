'''
Functions to load and use models
'''
import argparse
import logging
import time
#from transformers import AutoTokenizer, AutoModel
from transformers import MarianTokenizer, MarianMTModel

import torch

class MTModel:
    def __init__(self, lang_pair) -> None:
#        self.tokenizer = AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{lang_pair}")
#        self.model = AutoModel.from_pretrained(f"Helsinki-NLP/opus-mt-{lang_pair}")
        self.tokenizer = MarianTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{lang_pair}")
        self.model = MarianMTModel.from_pretrained(f"Helsinki-NLP/opus-mt-{lang_pair}")
        
    def get_embed_from_ind(self, ind):
        return self.model.get_input_embeddings().weight[ind]

    def get_embed_from_text(self, text):
        tokens = self.tokenizer([text], return_tensors="pt", padding=True)
        weight = self.model.get_input_embeddings().weight
        embed = []
        for token_id, mask in zip(tokens['input_ids'][0], tokens['attention_mask'][0]):
            this_embed = weight[token_id] * mask
            embed.append(this_embed)
        embed.pop() #last idx is eos padding (id=0) -- will be added seemingly regardless of tokenization args, not needed for analysis, so pop it off here
        return torch.stack(embed)

    def compute_cos(self, embed1, embed2):
        #first, average subword embeddings to get 1 embed per word
        embed1_avg = torch.mean(embed1, dim=0).unsqueeze(0)
        embed2_avg = torch.mean(embed2, dim=0).unsqueeze(0)
        #second, compute cosine similarity
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        output = cos(embed1_avg, embed2_avg)
        return output[0]

    # Given an input sentence string, return translation sentence string
    def translation_from_string(self, source):
        batch = self.tokenizer([source], return_tensors="pt")
        generated_ids = self.model.generate(**batch)
        output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return output

    # Given a seq of embeddings, return translated ids
    def translation_from_embed(self, embed):
        output = self.model(inputs_embeds=embed.unsqueeze(0), 
                            decoder_input_ids=torch.tensor([[0]]))
        last_hidden = output['encoder_last_hidden_state'] # batch X seqlen X embeddim
        #requires modifying model methods
        pass

    # Given a seq of embeddings, swap tokens out for similar embeddings
    def swap_embed(self, embed):
        pass

    # Given an embeddings, add noise 
    def noise_embed(self, embed):
        pass

def test_model(args):
    model = MTModel(args.lang_pair)
    out   = model.translation_from_string(args.src)
    print(out)
    return
#    model.get_embed_from_text('this is an example')
#    import ipdb;ipdb.set_trace()
#    print()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang_pair')
    parser.add_argument('--src')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    start = time.time()
    test_model(args)
    end = time.time()
    logging.info(f'Time to run script: {end-start} secs')
