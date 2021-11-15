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
parser.add_argument('--test_dir', type=str, help="can be a folder containing csv files or a csv file")
parser.add_argument('--prompt_dir', default=None)
parser.add_argument("--model_path")
parser.add_argument('--use_common_vocab', action="store_true")
parser.add_argument('--log', action="store_true")
parser.add_argument("--log_path", default="logs/")
parser.add_argument("--log_identifier", default="")
parser.add_argument('--use_layer', default=-1, type=int)

args = parser.parse_args()

if args.log:
    os.system("mkdir "+args.log_path)

if os.path.isdir(args.test_dir): 
    TEST_FILES = glob.glob(args.test_dir+"/*.csv")
else:
    TEST_FILES = [args.test_dir] # a csv file

print ("[files loaded]")

if args.prompt_dir is None:
    PROMPT_TABLE = {}
else:
    hp = pd.read_csv(args.prompt_dir)
    hp = hp[["rel", "human_prompt"]]
    hp.dropna(inplace=True)
    PROMPT_TABLE =  {row[1]["rel"]: row[1]["human_prompt"] for row in hp.iterrows()}
#print (PROMPT_TABLE)


# use common vocab
use_common_vocab = True
#ALL_CUIS, ALL_NAMES, NAME2CUI, DF_ALL  = None, None, None, None
ALL_NAMES = None
if use_common_vocab:
    ALL_NAMES = []
    for test_file in TEST_FILES:
        df_test = pd.read_csv(test_file, encoding='latin-1')
    
        for row in df_test.iterrows():
            #NAME2CUI[str(row[1]["head_name"])] = row[1]["head_cui"]
            #NAME2CUI[str(row[1]["tail_name"])] = row[1]["tail_cui"]
            ALL_NAMES += row[1]["tail_names"].split(" || ")
            ALL_NAMES += [row[1]["head_name"]]
        
    ALL_NAMES = list(set(ALL_NAMES))
    print ("vocab size:", len(ALL_NAMES))


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


overall_r1, overall_r10 = 0,0

file_r1_dict = {}
file_r10_dict = {}
global_count = 0
global_correct = 0
global_correct_at_k = 0
global_correct_indices = []

for test_file in TEST_FILES:
    file_name = test_file.split("/")[-1].split(".")[0]
    print ("query file:", file_name)
    df_test = pd.read_csv(test_file)

           
   # create queries
    name2cui = {}
    query_answers = []
    #for row in tqdm(df_test.iterrows(), total=len(df_test)):
    all_names_in_df = []
    for row in df_test.iterrows():
        #name2cui[str(row[1]["head_name"])] = row[1]["head_cui"]
        #name2cui[str(row[1]["tail_name"])] = row[1]["tail_cui"]

        rel_name = row[1]["rel"]
        #print (rel_name, PROMPT_TABLE[rel_name])
        if (rel_name in PROMPT_TABLE.keys()) and (len(PROMPT_TABLE[rel_name]) != 0):
            #print ("use template:", PROMPT_TABLE[row[1]["rel"]])
            query = PROMPT_TABLE[row[1]["rel"]].replace("[X]", row[1]["head_name"])
            query = query.replace("[Y]", "[MASK]")
        else:
            #query = row[1]["query"]
            query = row[1]["head_name"] + " " +row[1]["rel"].replace("_"," ") +" [MASK]."
        #print (query)
        #exit()
        #if DF_ALL is None:
        #    answers = df_test[(df_test["head_cui"]==row[1]["head_cui"])]["tail_cui"].tolist()
        #else:
        #    answers = DF_ALL[(DF_ALL["head_cui"]==row[1]["head_cui"]) & (DF_ALL["rel"]==row[1]["rel"])]["tail_cui"].tolist()
        answers = row[1]["tail_names"].split(" || ")
        all_names_in_df += answers
        all_names_in_df += [row[1]["head_name"]]
        query_answers.append((query,answers))
    if ALL_NAMES is None:
        all_names = list(set(all_names_in_df))
    else:
        all_names = ALL_NAMES
    
    print ("#queries:", len(query_answers))
    print ("search space size:", len(set(all_names)))

    # encode all target surface forms
    bs = 128
    all_reps = []
    #for i in tqdm(np.arange(0, len(all_names), bs)):
    for i in np.arange(0, len(all_names), bs):
        toks = tokenizer.batch_encode_plus(all_names[i:i+bs], 
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

    # config faiss
    index = faiss.IndexFlatL2(768)   # build the index
    #print(index.is_trained)
    index.add(all_reps_emb.astype("float32")) # add vectors to the index
    #print(index.ntotal)

    
    # let's search
    correct, correct_at_k = 0, 0
    bs = 128
    correct_indices = []
    preds = []
    hit_or_not = []
    hit_10_or_not = []

    #for i in tqdm(np.arange(0, len(query_answers), bs)):
    for i in np.arange(0, len(query_answers), bs):
        qa_batch = query_answers[i:i+bs]
        queries = [p[0] for p in qa_batch]
        answers = [p[1] for p in qa_batch]
        query_toks = tokenizer.batch_encode_plus(queries, 
                                        padding="max_length", 
                                        max_length=50, 
                                        truncation=True,
                                        return_tensors="pt")
        toks_cuda = {}
        for k,v in query_toks.items():
            toks_cuda[k] = v.cuda()
        query_output = model(**toks_cuda, return_dict=True, output_hidden_states=True)
        query_cls_rep = query_output.hidden_states[args.use_layer][:,0,:]
        
        D, nn_indices = index.search(query_cls_rep.cpu().detach().numpy().astype("float32"), 10)     # actual search

        for j in range(len(nn_indices)):
            if all_names[nn_indices[j][0]] in answers[j]:
                correct+=1
                global_correct += 1
                correct_indices.append(i+j)
                global_correct_indices.append(global_count)
                hit_or_not.append("correct")
            else:
                hit_or_not.append("wrong")
            preds.append(all_names[nn_indices[j][0]])
            global_count += 1
            hitat10 = "miss"
            for jj in range(len(nn_indices[j])):
                if  all_names[nn_indices[j][jj]] in answers[j]:
                    correct_at_k += 1
                    global_correct_at_k += 1
                    hitat10 = "hit"
                    break
            hit_10_or_not.append(hitat10)
    df_test["pred_correctness"] = hit_or_not
    df_test["pred_hit_at_10"] = hit_10_or_not
    df_test["pred"] = preds
    if args.log:
        df_test.to_csv( os.path.join(args.log_path, test_file.split("/")[-1]), sep=",", index=False)

    print ("correct indices:", correct_indices)
    print ("R@1: %.4f, R@10: %.4f" % (correct/len(df_test), correct_at_k/len(df_test)))
    print ("----------------------------------------------------")
    overall_r1 += correct/len(df_test)
    overall_r10 += correct_at_k/len(df_test)

    file_r1_dict[file_name] = correct/len(df_test)
    file_r10_dict[file_name] = correct_at_k/len(df_test)

file_r1_dict["average"] = overall_r1/len(TEST_FILES)
file_r10_dict["average"] = overall_r10/len(TEST_FILES)
print ("macro acc@1: %.4f, macro acc@10: %.4f" % (overall_r1/len(TEST_FILES), overall_r10/len(TEST_FILES)))
print ("micro acc@1: %.4f, micro acc@10: %.4f" % (global_correct/global_count, global_correct_at_k/global_count))

"""
if args.log:
    with open("LOG_by_category_r1_r10_"+args.log_identifier+".csv", 'w') as csv_file:  
        writer = csv.writer(csv_file)
        writer.writerow(["relation", "R@1", "R@10"])
        for key, value in file_r1_dict.items():
            writer.writerow([key, value, file_r10_dict[key]])

    with open("LOG_global_correct_indices_"+args.log_identifier+".txt", 'w') as f:  
        for ind in global_correct_indices:
            f.write(str(ind)+"\n")
"""
