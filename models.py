'''
Functions to load and use models
'''
import argparse
import logging
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class MTModel:
    def __init__(self, lang_pair) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{lang_pair}")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(f"Helsinki-NLP/opus-mt-{lang_pair}")
        
    def get_embed_from_ind(self, ind):
        return self.model.get_input_embeddings().weight[ind]

    def get_embed_from_text(self, text):
        tokens = self.tokenizer([text], return_tensors="pt", padding=True)
        weight = self.model.get_input_embeddings().weight
        embed = []
        for token_id, mask in zip(tokens['input_ids'][0], tokens['attention_mask'][0]):
            this_embed = weight[token_id] * mask
            embed.append(this_embed)
        return torch.stack(embed)

    # Given a seq of embeddings, return translated ids
    def translation_from_embed(self, embed):
        output = self.model(inputs_embeds=embed.unsqueeze(0), 
                            decoder_input_ids=torch.tensor([[0]]))
        last_hidden = output['encoder_last_hidden_state'] # batch X seqlen X embeddim

        pass

    # Given a seq of embeddings, swap tokens out for similar embeddings
    def swap_embed(self, embed):
        pass

    # Given an embeddings, add noise 
    def noise_embed(self, embed):
        pass

def test_model(args):
    model = MTModel(args.lang_pair)
    model.get_embed_from_text('this is an example')
    import ipdb;ipdb.set_trace()
    print()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('lang_pair')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    start = time.time()
    test_model(args)
    end = time.time()
    logging.info(f'Time to run script: {end-start} secs')