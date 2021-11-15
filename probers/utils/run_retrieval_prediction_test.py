import os
import sys
import csv
import glob
import math
from tqdm.auto import tqdm
import collections
import numpy as np
import pandas as pd
import argparse
import faiss
from transformers import AutoTokenizer, AutoModel


parser = argparse.ArgumentParser(description='Retrieval-based LAMA evaluation')

# configs
parser.add_argument('--vocab_dir', type=str, help="can be a folder containing csv files or a csv file")
parser.add_argument("--model_path")
parser.add_argument("--query", type=str)
parser.add_argument('--use_layer', default=-1, type=int)

args = parser.parse_args()


# read vocab
with open(args.vocab_dir, "r") as f:
    vocab = f.readlines()
vocab = [x.strip() for x in vocab] 

# load model
print ("[model loading]")
model_path = args.model_path
#model_path = "/home/newpapa/repos/sapbert_dev/contrastive_probe/tmp/pubmedbert_fulltext_test_100k_start_end_random_mask/"
#model_path = "/home/newpapa/repos/sapbert_dev/contrastive_probe/tmp/pubmedbert_fulltext_test_100k_start_end_random_mask_v2/"
#model_path = "/home/newpapa/repos/sapbert_dev/contrastive_probe/tmp/roberta_base_test_100k_start_end_random_mask/"
#model_path = "/home/newpapa/repos/sapbert_dev/contrastive_probe/tmp/bert_base_test_100k_start_end_random_mask/"
#model_path = "/home/newpapa/repos/sapbert_dev/contrastive_probe/tmp/scibert_test_100k_start_end_random_mask/"
#model_path = "/home/newpapa/repos/sapbert_dev/contrastive_probe/tmp/biobert_test_100k_start_end_random_mask/"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path).half().cuda() # 
print ("[model loaded]")

# encode query
toks = tokenizer.batch_encode_plus([args.query], 
                                padding="max_length", 
                                max_length=50, 
                                truncation=True,
                                return_tensors="pt")
toks_cuda = {}
for k,v in toks.items():
    toks_cuda[k] = v.cuda()
output = model(**toks_cuda, return_dict=True, output_hidden_states=True)
    
query_cls_rep = output.hidden_states[args.use_layer][:,0,:] # cls
query_cls_rep = query_cls_rep.cpu().detach().numpy().astype("float32")


# encode all vocab
bs = 128
all_reps = []
for i in tqdm(np.arange(0, len(vocab), bs)):
    toks = tokenizer.batch_encode_plus(vocab[i:i+bs], 
                                padding="max_length", 
                                max_length=25, 
                                truncation=True,
                                return_tensors="pt")
    toks_cuda = {}
    for k,v in toks.items():
        toks_cuda[k] = v.cuda()
    output = model(**toks_cuda, return_dict=True, output_hidden_states=True)
        
    cls_rep = output.hidden_states[args.use_layer][:,0,:] # cls
    
    all_reps.append(cls_rep.cpu().detach().numpy())
all_reps_emb = np.concatenate(all_reps, axis=0)
#print (all_reps_emb.shape)

print (query_cls_rep.shape, all_reps_emb.shape)

# config faiss
index = faiss.IndexFlatL2(768)   # build the index
#print(index.is_trained)
index.add(all_reps_emb.astype("float32")) # add vectors to the index
#print(index.ntotal)

# NN search
D, nn_indices = index.search(query_cls_rep, 20)    
print (nn_indices)
for n in nn_indices[0]:
    print (vocab[n])