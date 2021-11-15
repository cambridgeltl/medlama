import argparse
import collections
import csv
import datetime
import glob
import math
import os
import sys

import faiss
import numpy as np
import pandas as pd
import wandb
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

parser = argparse.ArgumentParser(description="Retrieval-based LAMA evaluation")

# configs
parser.add_argument(
    "--test_dir", type=str, help="can be a folder containing csv files or a csv file"
)
parser.add_argument("--prompt_dir", default=None)
parser.add_argument("--prompt_type", default="default_prompt")
parser.add_argument("--model_path")
parser.add_argument("--set")
parser.add_argument("--mask_ratio")
parser.add_argument("--model_short", default=None)
parser.add_argument("--use_common_vocab", action="store_true")
parser.add_argument("--prob_hard", action="store_true")
parser.add_argument("--epoch", default=None)
parser.add_argument("--log", action="store_true")
parser.add_argument("--log_path", default="logs/")
parser.add_argument("--log_identifier", default="")
parser.add_argument("--use_layer", default=-1, type=int)

args = parser.parse_args()
print(args)
if args.log:
    os.system("mkdir " + args.log_path)
if args.model_short is None:
    args.model_short = args.model_path.split("_")[0]
if not args.prob_hard:
    TEST_FILES = glob.glob(args.test_dir + "/*1000.csv")
    TEST_FILES.extend(glob.glob(args.test_dir + "/*_full_*.csv"))
    # TEST_FILES.extend(glob.glob(args.test_dir + "/*.csv"))
else:
    TEST_FILES = glob.glob(args.test_dir + "/*_hard.csv")


print(f"{args.test_dir} [{len(TEST_FILES)} files loaded]")

if args.prompt_dir is None:
    PROMPT_TABLE = {}
else:
    hp = pd.read_csv(args.prompt_dir)
    hp = hp[["pid", args.prompt_type]]
    hp.dropna(inplace=True)
    PROMPT_TABLE = {row[1]["pid"]: row[1][args.prompt_type] for row in hp.iterrows()}
# print (PROMPT_TABLE)


# use common vocab
use_common_vocab = True
# ALL_CUIS, ALL_NAMES, NAME2CUI, DF_ALL  = None, None, None, None
ALL_NAMES = None
if use_common_vocab:
    ALL_NAMES = []
    for test_file in TEST_FILES:
        df_test = pd.read_csv(test_file, encoding="latin-1")

        for row in df_test.iterrows():
            # NAME2CUI[str(row[1]["head_name"])] = row[1]["head_cui"]
            # NAME2CUI[str(row[1]["tail_name"])] = row[1]["tail_cui"]
            ALL_NAMES += row[1]["tail_names"].split(" || ")
            ALL_NAMES += [row[1]["head_name"]]

    ALL_NAMES = list(set(ALL_NAMES))
    print("triple files:", len(TEST_FILES), "vocab size:", len(ALL_NAMES))

if not args.prob_hard:
    TEST_FILES = glob.glob(args.test_dir + "/*1000.csv")
    TEST_FILES.extend(glob.glob(args.test_dir + "/*_full_*.csv"))
    # TEST_FILES.extend(glob.glob(args.test_dir + "/*.csv"))
else:
    TEST_FILES = glob.glob(args.test_dir + "/*_hard.csv")

# load model
print("[model loading]")
model_path = args.model_path
if args.epoch:
    model_path = f"{model_path}/checkpoint_iter_{args.epoch}/"
# model_path = "/home/newpapa/repos/sapbert_dev/contrastive_probe/tmp/pubmedbert_fulltext_test_100k_start_end_random_mask/"
# model_path = "/home/newpapa/repos/sapbert_dev/contrastive_probe/tmp/pubmedbert_fulltext_test_100k_start_end_random_mask_v2/"
# model_path = "/home/newpapa/repos/sapbert_dev/contrastive_probe/tmp/roberta_base_test_100k_start_end_random_mask/"
# model_path = "/home/newpapa/repos/sapbert_dev/contrastive_probe/tmp/bert_base_test_100k_start_end_random_mask/"
# model_path = "/home/newpapa/repos/sapbert_dev/contrastive_probe/tmp/scibert_test_100k_start_end_random_mask/"
# model_path = "/home/newpapa/repos/sapbert_dev/contrastive_probe/tmp/biobert_test_100k_start_end_random_mask/"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path).half().cuda()  #
print("[model loaded]")


overall_r1, overall_r5, overall_r10 = 0, 0, 0
overall_h1, overall_h5, overall_h10 = 0, 0, 0

global_count = 0
global_correct = 0
global_correct_at_k = 0
global_correct_indices = []
time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M-")
for test_file in TEST_FILES:
    result = {}
    wandb.init(project="Bio_LAMA_Mirror")
    wandb.config.update(args)
    file_name = test_file.split("/")[-1].split(".")[0]
    result["file_name"] = file_name
    print("query file:", file_name)
    df_test = pd.read_csv(test_file)

    # create queries
    name2cui = {}
    query_answers = []
    # for row in tqdm(df_test.iterrows(), total=len(df_test)):
    all_names_in_df = []
    for row in df_test.iterrows():
        # name2cui[str(row[1]["head_name"])] = row[1]["head_cui"]
        # name2cui[str(row[1]["tail_name"])] = row[1]["tail_cui"]

        rel_name = row[1]["rel"]
        # print (rel_name, PROMPT_TABLE[rel_name])
        if (rel_name in PROMPT_TABLE.keys()) and (len(PROMPT_TABLE[rel_name]) != 0):
            # print ("use template:", PROMPT_TABLE[row[1]["rel"]])
            query = PROMPT_TABLE[row[1]["rel"]].replace("[X]", row[1]["head_name"])
            query = query.replace("[Y]", "[MASK]")
        else:
            # query = row[1]["query"]
            query = (
                row[1]["head_name"] + " " + row[1]["rel"].replace("_", " ") + " [MASK]."
            )
        # print (query)
        # exit()
        # if DF_ALL is None:
        #    answers = df_test[(df_test["head_cui"]==row[1]["head_cui"])]["tail_cui"].tolist()
        # else:
        #    answers = DF_ALL[(DF_ALL["head_cui"]==row[1]["head_cui"]) & (DF_ALL["rel"]==row[1]["rel"])]["tail_cui"].tolist()
        answers = row[1]["tail_names"].split(" || ")
        all_names_in_df += answers
        all_names_in_df += [row[1]["head_name"]]
        query_answers.append((query, answers))
    if ALL_NAMES is None:
        all_names = list(set(all_names_in_df))
    else:
        all_names = ALL_NAMES

    print("#queries:", len(query_answers))
    print("search space size:", len(set(all_names)))

    # encode all target surface forms
    bs = 128
    all_reps = []
    # for i in tqdm(np.arange(0, len(all_names), bs)):
    for i in np.arange(0, len(all_names), bs):
        toks = tokenizer.batch_encode_plus(
            all_names[i : i + bs],
            padding="max_length",
            max_length=25,
            truncation=True,
            return_tensors="pt",
        )
        toks_cuda = {}
        for k, v in toks.items():
            toks_cuda[k] = v.cuda()
        output = model(**toks_cuda, return_dict=True, output_hidden_states=True)

        cls_rep = output.hidden_states[args.use_layer][:, 0, :]  # cls

        all_reps.append(cls_rep.cpu().detach().numpy())
    all_reps_emb = np.concatenate(all_reps, axis=0)
    # print (all_reps_emb.shape)

    # config faiss
    index = faiss.IndexFlatL2(768)  # build the index
    # print(index.is_trained)
    index.add(all_reps_emb.astype("float32"))  # add vectors to the index
    # print(index.ntotal)

    # let's search
    correct, correct_at_k = 0, 0
    bs = 128
    correct_indices = []
    preds = []
    hit_or_not = []
    hit_5_or_not = []
    hit_10_or_not = []
    # for i in tqdm(np.arange(0, len(query_answers), bs)):
    for i in np.arange(0, len(query_answers), bs):
        qa_batch = query_answers[i : i + bs]
        queries = [p[0] for p in qa_batch]
        answers = [p[1] for p in qa_batch]
        query_toks = tokenizer.batch_encode_plus(
            queries,
            padding="max_length",
            max_length=50,
            truncation=True,
            return_tensors="pt",
        )
        toks_cuda = {}
        for k, v in query_toks.items():
            toks_cuda[k] = v.cuda()
        query_output = model(**toks_cuda, return_dict=True, output_hidden_states=True)
        query_cls_rep = query_output.hidden_states[args.use_layer][:, 0, :]

        D, nn_indices = index.search(
            query_cls_rep.cpu().detach().numpy().astype("float32"), 10
        )  # actual search

        for j in range(len(nn_indices)):
            if all_names[nn_indices[j][0]] in answers[j]:
                correct += 1
                global_correct += 1
                correct_indices.append(i + j)
                global_correct_indices.append(global_count)
                hit_or_not.append(1)
            else:
                hit_or_not.append(0)
            # preds.append(all_names[nn_indices[j][0]])
            global_count += 1
            hitat10 = 0
            hitat5 = 0
            top10preds = []
            hitat5_count = 0
            for jj in range(len(nn_indices[j])):
                top10preds.append(all_names[nn_indices[j][jj]])
                if all_names[nn_indices[j][jj]] in answers[j]:
                    hitat10 += 1
                    if hitat5_count < 5:
                        hitat5 += 1
                hitat5_count += 1

            preds.append(" || ".join(top10preds))
            hit_10_or_not.append(hitat10)
            hit_5_or_not.append(hitat5)
    df_test["pred_hit_at_1"] = hit_or_not
    df_test["pred_hit_at_5"] = hit_5_or_not
    df_test["pred_hit_at_10"] = hit_10_or_not
    df_test["pred"] = preds
    if args.log:
        df_test.to_csv(
            os.path.join(args.log_path, time_str + test_file.split("/")[-1]),
            sep=",",
            index=False,
        )

    # print ("correct indices:", correct_indices)
    print(
        "R@1: %.4f, R@5: %.4f, R@10: %.4f"
        % (
            sum(df_test["pred_hit_at_1"]) / len(df_test),
            sum(df_test["pred_hit_at_5"]) / len(df_test),
            sum(df_test["pred_hit_at_10"]) / len(df_test),
        )
    )
    print("----------------------------------------------------")
    overall_r1 += sum(df_test["pred_hit_at_1"]) / len(df_test)
    overall_r5 += sum(df_test["pred_hit_at_5"]) / len(df_test)
    overall_r10 += sum(df_test["pred_hit_at_10"]) / len(df_test)
    overall_h1 += sum(df_test["pred_hit_at_1"])
    overall_h5 += sum(df_test["pred_hit_at_5"])
    overall_h10 += sum(df_test["pred_hit_at_10"])
    result["recall@1"] = sum(df_test["pred_hit_at_1"]) / len(df_test)
    result["recall@5"] = sum(df_test["pred_hit_at_5"]) / len(df_test)
    result["recall@10"] = sum(df_test["pred_hit_at_10"]) / len(df_test)
    wandb.config.update(result)
    wandb.finish()
wandb.init(project="Bio_LAMA_Mirror_summary")

summary = {}
summary["macro recall@1"] = overall_r1 / len(TEST_FILES)
summary["macro recall@5"] = overall_r5 / len(TEST_FILES)
summary["macro recall@10"] = overall_r10 / len(TEST_FILES)
summary["micro recall@1"] = overall_h1 / global_count
summary["micro recall@5"] = overall_h5 / global_count
summary["micro recall@10"] = overall_h10 / global_count
wandb.config.update(args)
wandb.config.update(summary)
print(summary)
