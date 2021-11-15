import argparse
import csv
import json
import os
import re
import time
import traceback
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Union

import faiss
import numpy as np
import pandas as pd
import torch
from tabulate import tabulate
from tqdm.auto import tqdm
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

import wandb
from prompt import Prompt

SUB_LABEL = "##"


def print_dict_as_table(dic, tag=None, columns=["keys", "values"]):
    """Print a dictionary as table.
    Args:
        dic (dict): dict object to be formatted.
        tag (str): A name for this dictionary.
        columns ([str,str]):  default ["keys", "values"]. columns name for keys and values.
    Returns:
        None
    """
    print("-" * 80)
    if tag is not None:
        print(tag)
    df = pd.DataFrame(dic.items(), columns=columns)
    print(tabulate(df, headers=columns, tablefmt="psql"))
    print("-" * 80)
    return tabulate(df, headers=columns, tablefmt="psql")


def print_args_as_table(args, tag=None, columns=["keys", "values"]):
    """Print a ArgumentParser as table.

    Args:
        dic (dict): dict object to be formatted.
        tag (str): A name for this dictionary.
        columns ([str,str]):  default ["keys", "values"]. columns name for keys and values.

    Returns:
        None
    """
    return print_dict_as_table(vars(args), tag, columns)


_tie_breaking: Dict[int, torch.Tensor] = {}


def get_tie_breaking(dim: int):
    if dim not in _tie_breaking:
        _tie_breaking[dim] = torch.zeros(dim).uniform_(0, 1e-5)
    return _tie_breaking[dim]


def model_prediction_wrap(model, inp_tensor, attention_mask):
    output = model(inp_tensor, attention_mask=attention_mask, return_dict=True)
    return output.logits


class EvalContext(object):
    def __init__(self, args):
        self.norm: bool = args.norm
        self.use_alias: bool = True
        self.uncase: bool = True
        self.use_multi_rel: bool = True
        self.use_period: bool = True
        self.multi_lang: str = args.multi_lang
        self.skip_cate: bool = args.skip_cate
        self.lang: str = args.prompt
        self.gold_len: bool = args.gold_len

        self.lm = LM_NAME[args.model] if args.model in LM_NAME else args.model

        self.entity2iscate = None  #
        self.entity2gender = None
        # self.entity2gender = load_entity_gender(self.entity_gender_path)
        self.entity2instance = None
        # entity2instance = {entityid: string of all the alias }
        # self.entity2iscate = load_entity_is_cate(self.is_cate)
        self.tokenizer = AutoTokenizer.from_pretrained(self.lm)


class CsvLogFileContext:
    def __init__(self, filename: str = None, headers: List[str] = None):
        self.filename = filename
        self.headers = headers

    def __enter__(self):
        if self.filename:
            self.file = open(self.filename, "w")
            self.file.write(",".join(self.headers) + "\n")
            csv_file = csv.writer(self.file)
            return csv_file
        return None

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.filename:
            self.file.close()


class LamaPredictions(object):
    greek_unstress = str.maketrans("άόίέύώή", "αοιευωη")

    def __init__(self, result: Dict, pid: str = None):
        self.result = result
        self.pid = pid

    def __str__(self):
        return json.dumps(self.result)

    @classmethod
    def from_str(cls, str, pid: str = None):
        return cls(json.loads(str), pid)

    @staticmethod
    def prettify_tokens(tokens: List[str], tokenizer):
        return tokenizer.convert_tokens_to_string(tokens)

    @staticmethod
    def is_y_followed_by_at_end(prompt: str, last_char: str) -> bool:
        assert len(last_char) == 1, "should be a char"
        if prompt[-1] == last_char and re.match("\[.*Y.*\]$", prompt[:-1].rstrip()):
            return True
        return False

    def eval(self, eval: EvalContext) -> bool:
        scores: List[float] = []
        for p in self.result["pred_log_prob"]:
            scores.append(np.mean(p) if eval.norm else np.sum(p))
        best = int(np.argmax(scores))
        correct, pred, pred_log_prob, golds = self.match_with_gold(
            self.result["pred"],
            best,
            self.pid,
            self.result,
            use_alias=eval.use_alias,
            lang=eval.lang,
            uncase=eval.uncase,
            use_multi_rel=eval.use_multi_rel,
            use_period=eval.use_period,
            multi_lang=eval.multi_lang,
            gold_len=eval.gold_len,
            tokenizer=eval.tokenizer,
            eval=eval,
        )
        self.pred = pred
        self.pred_log_prob = pred_log_prob
        self.correct = correct
        self.golds = golds
        return correct

    @property
    def is_single_word(self) -> bool:
        return len(self.result["tokenized_tail_name_inflection"]) <= 1

    @property
    def single_word_pred(self) -> Tuple[str, float]:
        return (self.result["pred"][0][0], self.result["pred_log_prob"][0][0])

    @property
    def num_tokens(self) -> int:
        return len(self.pred)

    @property
    def confidence(self) -> float:
        return np.exp(np.mean(self.pred_log_prob))

    @property
    def is_use_single_word_pred(self) -> bool:
        return self.num_tokens == 1

    def is_cate(self, entity2iscate: Dict[str, bool]) -> bool:
        if entity2iscate[self.result["head_cui"]]:
            print(self.result["head_name"])
        return entity2iscate[self.result["head_cui"]]

    def add_prediction(self, pred: List[str], correct: bool):
        self.pred2 = pred
        self.correct2 = correct

    def prettify(self, csv_file: CsvLogFileContext, eval: EvalContext):
        csv_file.writerow(
            [self.prettify_tokens(self.result["sentence"], eval.tokenizer)]
            + [self.prettify_tokens(self.pred, eval.tokenizer)]
            + (
                [self.prettify_tokens(self.pred2, eval.tokenizer)]
                if hasattr(self, "pred2")
                else []
            )
            + [
                " | ".join(
                    [self.prettify_tokens(g, eval.tokenizer) for g in self.golds]
                )
            ]
            + [self.correct]
            + ([self.correct2] if hasattr(self, "correct2") else [])
            + [self.confidence]
            + ([self.confidence2] if hasattr(self, "confidence2") else [])
            + [  # TODO: confidence2
                self.is_single_word,
                self.result["head_cui"],
                self.result["tail_cui"],
            ]
        )

    @staticmethod
    def match_with_gold(
        pred: List[List[str]],
        best: int,
        pid: str,
        result: Dict,
        use_alias: bool = False,
        lang: str = None,
        uncase: bool = False,
        use_multi_rel: bool = False,
        use_period: bool = False,
        multi_lang: str = None,
        gold_len: bool = False,
        tokenizer=None,
        eval: EvalContext = None,
    ) -> Tuple[bool, List[str], List[float], List[List[str]]]:
        if use_multi_rel and not use_alias:
            raise NotImplementedError
        if multi_lang and not use_alias:
            raise NotImplementedError

        raw_lang = lang
        if multi_lang and lang != multi_lang:  # TODO: use all languages?
            langs = [lang, multi_lang]
        else:
            langs = [lang]

        all_golds: List[List[str]] = []
        for lang in langs:
            casify = lambda x: x.lower() if uncase else x
            unstress = (
                lambda x: x.translate(LamaPredictions.greek_unstress)
                if lang == "el"
                else x
            )
            golds: List[List[str]] = [result["tokenized_tail_name"]]
            all_golds.extend(golds)

            for gold in golds:
                _gold: str = unstress(casify(tokenizer.convert_tokens_to_string(gold)))
                if gold_len and len(gold) <= len(pred):
                    choices = [len(gold) - 1]
                    if lang == "en" and len(gold) > 1:
                        choices.append(len(gold) - 2)  # the period issue
                else:
                    choices = [best]
                for choice in choices:
                    _pred: str = unstress(
                        casify(tokenizer.convert_tokens_to_string(pred[choice]))
                    )
                    if _pred == _gold:
                        return (
                            True,
                            pred[choice],
                            result["pred_log_prob"][choice],
                            all_golds,
                        )
                    if (
                        use_period
                        and lang == "en"
                        and LamaPredictions.is_y_followed_by_at_end(
                            result["prompt"], "."
                        )
                    ):
                        if (
                            len(_gold) > 0
                            and _gold[-1] == "."
                            and _pred.rstrip() == _gold[:-1].rstrip()
                        ):
                            return (
                                True,
                                pred[choice],
                                result["pred_log_prob"][choice],
                                all_golds,
                            )
        return False, pred[best], result["pred_log_prob"][best], all_golds


class JsonLogFileContext:
    def __init__(self, filename: str = None):
        self.filename = filename

    def __enter__(self):
        if self.filename:
            self.file = open(self.filename, "w")
            return self.file
        return None

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.filename:
            self.file.close()


class ProbeIterator(object):
    def __init__(self, args: argparse.Namespace):
        self.args = args
        print("Load tokenizer:", args.model)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        print("Load model:", args.model)
        self.model = AutoModelForMaskedLM.from_pretrained(args.model)
        self.args.model_short = self.get_short_model_name(self.model)
        self.model.eval()
        if torch.cuda.is_available() and not args.no_cuda:
            self.model.to("cuda")

        # get prompt
        self.prompts = {}
        prompt_df = pd.read_csv(args.prompt_file)
        for _, item in prompt_df.iterrows():
            self.prompts[item["pid"]] = item[args.prompt]

        self.get_vocab_token_ids(args)
        if self.args.use_knn:
            self.init_embedding_index(args)

        self.restrict_vocab = []
        if self.args.use_restrict_vocab:
            self.get_restrict_vocab(args)

        self.mask_label = self.tokenizer.mask_token
        self.unk_label = self.tokenizer.unk_token
        self.pad_label = self.tokenizer.pad_token
        self.mask = self.tokenizer.convert_tokens_to_ids(self.mask_label)
        self.unk = self.tokenizer.convert_tokens_to_ids(self.unk_label)
        self.pad = self.tokenizer.convert_tokens_to_ids(self.pad_label)

        # load data
        self.entity2lang = None
        self.entity2gender = None
        self.entity2instance = None

        # log
        if args.log_dir and not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

        # prompt model
        self.prompt_model = Prompt.instantiate(args.prompt)

        # summary
        self.summary = {
            "num_max_mask": 0,  # number of facts where the object has more tokens than the max number of masks
            "numtoken2count": defaultdict(lambda: 0),  # number of token (gold) to count
        }

    def get_restrict_vocab(self, args):
        """Get a token_ids that are not in the entity names

        Args:
            args ([type]): [description]
        """
        toks = self.tokenizer.batch_encode_plus(
            self.all_names,
            padding=True,
            return_tensors="pt",
        )

        token_ids = set(toks["input_ids"].view(-1).numpy())
        self.restrict_vocab = torch.tensor(
            list(set([i for i in range(self.tokenizer.vocab_size)]) - token_ids),
            dtype=torch.long,
            device=self.model.device,
        )
        print(
            f"Get restrict vocab size: {len(self.restrict_vocab)}/{self.tokenizer.vocab_size}"
        )

    def init_embedding_index(self, args):
        """Initialize embedding index for searching

        Args:
            args ([type]): [description]
        """
        self.model_base = AutoModel.from_pretrained(args.model)
        self.model_base.eval()
        if torch.cuda.is_available() and not args.no_cuda:
            self.model_base.to("cuda")
        if args.use_knn:
            # encode all target surface forms
            bs = 128
            all_reps = []
            # for i in tqdm(np.arange(0, len(all_names), bs)):
            for i in tqdm(
                np.arange(0, len(self.all_names), bs),
                desc=f"Inferring embeddings for all-names ({len(self.all_names)})",
            ):
                try:
                    toks = self.tokenizer.batch_encode_plus(
                        self.all_names[i : i + bs],
                        padding="max_length",
                        max_length=args.max_entity_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                except:
                    print(self.all_names[i : i + bs])
                toks_cuda = {}
                for k, v in toks.items():
                    toks_cuda[k] = v.cuda()
                output = self.model_base(**toks_cuda)
                cls_rep = output[0][:, 0, :]  # cls

                all_reps.append(cls_rep.cpu().detach().numpy())
            all_reps_emb = np.concatenate(all_reps, axis=0)
            # print (all_reps_emb.shape)
            # config faiss
            self.index = faiss.IndexFlatL2(768)  # build the index
            # print(index.is_trained)
            self.index.add(all_reps_emb.astype("float32"))  # add vectors to the index
            # print(index.ntotal)

    def get_queries(self, fact_path: str) -> Tuple[List[Dict], List[Union[int, float]]]:

        queries: List[Dict] = []
        fact_df = pd.read_csv(fact_path, header=0)
        for _, item in fact_df.iterrows():
            queries.append(item)
        return queries

    def batcher(self, queries: List[Dict], prompts: str) -> Tuple[List, Tuple, Tuple]:
        NUM_MASK = self.args.num_mask

        for b in tqdm(range(0, len(queries), self.args.batch_size), disable=True):
            query_batch = queries[b : b + self.args.batch_size]

            inp_tensor: List[torch.Tensor] = []
            tail_name_list = []
            instance_xys: List[str] = []
            for query in query_batch:
                tail_name_list.append(query["tail_names"].split(" || "))
                # fill in subjects
                instance_x, _ = self.prompt_model.fill_x(
                    prompts[query["rel"]],
                    query["head_cui"],
                    query["head_name"].replace("[X]", ""),
                )
                for nm in range(NUM_MASK):
                    instance_xy, tail_name = self.prompt_model.fill_y(
                        instance_x,
                        query["tail_names"].split(" || ")[0],
                        num_mask=nm + 1,
                        mask_sym=self.mask_label,
                    )
                    instance_xys.append(instance_xy)

                # tokenize sentences
                for instance_xy in instance_xys:
                    # TODO: greek BERT does not seem to need this
                    """
                    if self.args.model == 'el_bert_base':
                        instance_xy = self.prompt_model.normalize(instance_xy, mask_sym=self.mask_label)
                        tail_name = self.prompt_model.normalize(tail_name)
                    """
                    if re.match("\[.*X.*\]", instance_xy) or re.match(
                        "\[.*Y.*\]", instance_xy
                    ):
                        raise Exception(
                            'inflection missing from "{}"'.format(instance_xy)
                        )
                    if instance_xy.find(self.mask_label) == -1:
                        raise Exception(
                            'not contain mask tokens "{}"'.format(instance_xy)
                        )
            out_feature = self.tokenizer.batch_encode_plus(
                instance_xys,
                padding="max_length",
                max_length=args.max_query_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            # print(out_feature["input_ids"].size())
            inp_tensor = out_feature["input_ids"]
            attention_mask = out_feature["attention_mask"]
            # print("=" * 20)
            # print(instance_xys[0])
            # print(self.tokenizer.convert_ids_to_tokens(inp_tensor[0]))
            # tokenize gold object

            mask_ind: torch.Tensor = inp_tensor.eq(self.mask).long()

            if torch.cuda.is_available() and not self.args.no_cuda:
                inp_tensor = inp_tensor.cuda()
                attention_mask = attention_mask.cuda()
                mask_ind = mask_ind.cuda()

            yield query_batch, (inp_tensor, attention_mask, mask_ind)

    def get_vocab_token_ids(self, args):
        """Get the vocab in the same sets of triples

        Args:
            fact_dir ([type]): [description]

        Returns:
            [type]: [description]
        """
        fact_dir = args.fact_dir
        entity_str_to_cui = {}
        for fact_path in tqdm(os.listdir(fact_dir), desc="Load vocab"):
            if ".csv" not in fact_path:  # skip non-facts files
                continue
            if not args.prob_hard and "_hard" in fact_path:
                continue
            # if args.prob_hard and "_hard" not in fact_path:
            #     continue
            fact_file = os.path.join(fact_dir, fact_path)
            fact_df = pd.read_csv(fact_file)
            for col in fact_df.columns:
                if sum(fact_df[col].isnull()):
                    print(
                        f"Find nan value in col [{col}] of {fact_path}, skip these rows."
                    )
                    fact_df = fact_df[~fact_df[col].isnull()]
            for _, item in fact_df.iterrows():
                if item["rel"] not in self.prompts:
                    print(f"{item['rel']} is missing in prompts.csv")
                    self.prompts[item["rel"]] = (
                        "[X] " + item["rel"].replace("_", " ") + " [Y] ."
                    )
                    print(f"{item['rel']}, {self.prompts[item['rel']]}")
                entity_str_to_cui[item["head_name"]] = item["head_cui"]
                tail_names = item["tail_names"].split(" || ")
                tail_cuis = item["tail_cuis"].split(" || ")
                for tail_name, tail_cui in zip(tail_names, tail_cuis):
                    entity_str_to_cui[tail_name.strip()] = tail_cui.strip()
        print(f"Total vocab size: {len(entity_str_to_cui)}")
        self.all_names = list(entity_str_to_cui.keys())
        self.all_cuis = list(entity_str_to_cui.values())

    def get_short_model_name(self, model):
        model_name = model.config._name_or_path
        model_name = model_name.replace("/", "_").replace("-", "_")
        for split in model_name.split("_"):
            if "bert" in split.lower():
                model_name = split
        return model_name

    def iter(self):
        NUM_MASK = self.args.num_mask

        num_fact = 0
        num_correct_fact = 0
        acc_li: List[float] = []
        iters: List[int] = []
        overall_r1 = 0
        global_count = 0
        global_correct = 0
        for fact_path in tqdm(
            os.listdir(args.fact_dir),
            desc="Processing fact files",
        ):
            if ".csv" not in fact_path:  # skip non-facts files
                continue
            if not self.args.prob_hard and "_hard" in fact_path:
                continue
            # if self.args.prob_hard and "_hard" not in fact_path:
            #     continue
            try:
                # assume fact_dir path  will be ../data/xxx/round_x/
                wandb.init(
                    project=f"xfactr_{args.fact_dir.split('data/')[-1].replace('/','_')}"
                )
                args.fact_table = fact_path
                wandb.config.update(args)
                result_metrics = {}
                print_args_as_table(args)
                start_time = time.time()

                # get queries
                fact_file = os.path.join(args.fact_dir, fact_path)
                queries = self.get_queries(fact_file)

                total_correct_num = 0

                acc = []
                queries_to_save = []
                predict_answers = []
                gold_cuis = []
                gold_names = []
                gold_tokens = []
                predicts_to_save = []
                knn_predicts_to_save = []
                recalls_to_save = []
                knn_recalls_to_save = []
                probilities_to_save = []
                for _, batch in tqdm(
                    enumerate(self.batcher(queries, self.prompts)),
                    desc="Loading batch queries",
                    total=len(queries),
                ):  ## Note that now only support batch_size 1
                    (
                        query_batch,
                        (inp_tensor, attention_mask, mask_ind),
                    ) = batch
                    for query in query_batch:
                        gold_cuis.append(query["tail_cuis"].split(" || "))
                        tail_names = query["tail_names"].split(" || ")
                        tail_names_tokens = self.tokenizer.batch_encode_plus(
                            tail_names, add_special_tokens=False, return_tensors="np"
                        )["input_ids"]
                        gold_tokens.append(tail_names_tokens)
                        gold_names.append(tail_names)
                    batch_size = len(query_batch)
                    inp_tensor = inp_tensor.view(batch_size, NUM_MASK, -1)
                    attention_mask = attention_mask.view(batch_size, NUM_MASK, -1)
                    mask_ind = mask_ind.view(batch_size, NUM_MASK, -1)

                    out_tensors: List[torch.LongTensor] = []
                    logprobs: List[torch.Tensor] = []
                    for nm in range(NUM_MASK):
                        # decoding
                        # SHAPE: (batch_size * num_mask, seq_len)
                        out_tensor, logprob, iter = iter_decode_beam_search(
                            self.model,
                            inp_tensor[:, nm, :],
                            mask_ind[:, nm, :],
                            attention_mask[:, nm, :],
                            restrict_vocab=self.restrict_vocab,
                            mask_value=self.mask,
                            max_iter=self.args.max_iter,
                            init_method=self.args.init_method,
                            iter_method=self.args.iter_method,
                            reprob=self.args.reprob,
                            beam_size=args.beam_size,
                        )
                        out_tensors.append(out_tensor)
                        logprobs.append(logprob)
                        iters.append(iter)
                        if False:
                            print("=== In Put #mask {} ===".format(nm + 1))
                            print(
                                self.tokenizer.convert_ids_to_tokens(
                                    inp_tensor[0, nm, :]
                                )
                            )
                            print("=== #mask {} ===".format(nm + 1))
                            print(self.tokenizer.convert_ids_to_tokens(out_tensor[0]))
                            print((logprob[0] * mask_ind[0, nm].float()).sum())

                    # SHAPE: (batch_size, num_mask, seq_len)
                    mask_ind = mask_ind.float()
                    logprob = torch.stack(logprobs, 1)
                    out_tensor = torch.stack(out_tensors, 1)

                    # mask len norm
                    mask_len = mask_ind.sum(-1)
                    mask_len_norm = 1.0 if self.args.no_len_norm else mask_len

                    # find the best setting
                    for i, avg_log in enumerate(
                        (logprob * mask_ind).sum(-1) / mask_len_norm
                    ):
                        lp, best_num_mask = avg_log.max(0)
                        pred: np.ndarray = (
                            out_tensor[i, best_num_mask]
                            .masked_select(mask_ind[i, best_num_mask].eq(1))
                            .detach()
                            .cpu()
                            .numpy()
                            .reshape(-1)
                        )
                        inp: np.ndarray = (
                            inp_tensor[i, best_num_mask].detach().cpu().numpy()
                        )
                        is_correct = 0
                        for tail_name in tail_names_tokens:
                            if list(pred) == list(tail_name):
                                is_correct = 1
                                total_correct_num += 1
                                print(f"total_correct_num:{total_correct_num}")

                        predict_answers.append(
                            load_word_ids(pred, self.tokenizer, self.pad_label)
                        )

                        acc.append(is_correct)

                        """
                        print('===', tokenizer.convert_ids_to_tokens(obj), is_correct, '===')
                        for j in range(NUM_MASK):
                            print(tokenizer.convert_ids_to_tokens(inp_tensor[i, j].detach().cpu().numpy()))
                            tpred = out_tensor[i, j].masked_select(mask_ind[i, j].eq(1)).detach().cpu().numpy().reshape(-1)
                            print(tokenizer.convert_ids_to_tokens(tpred), avg_log[j])
                        input()
                        """
                        queries_to_save.append(
                            load_word_ids(inp, self.tokenizer, self.pad_label)
                        )
                        predicts_to_save.append(
                            load_word_ids(pred, self.tokenizer, self.pad_label)
                        )
                        recalls_to_save.append(is_correct)
                        probilities_to_save.append("{:.5f}".format(lp.item()))

                        def get_all_pred_score():
                            results: List[str] = []
                            for nm in range(NUM_MASK):
                                pred = (
                                    logprob[i, nm]
                                    .masked_select(mask_ind[i, nm].eq(1))
                                    .detach()
                                    .cpu()
                                    .numpy()
                                    .reshape(-1)
                                )
                                results.append(pred.tolist())
                            return results

                        def get_all_pred():
                            results: List[str] = []
                            for nm in range(NUM_MASK):
                                pred = (
                                    out_tensor[i, nm]
                                    .masked_select(mask_ind[i, nm].eq(1))
                                    .detach()
                                    .cpu()
                                    .numpy()
                                    .reshape(-1)
                                )
                                results.append(
                                    merge_subwords(pred, self.tokenizer, merge=False)
                                )
                            return results

                num_fact += len(queries)
                num_correct_fact += total_correct_num
                acc_for_rel = total_correct_num / (len(queries))
                result_metrics["recall@1"] = acc_for_rel
                overall_r1 += acc_for_rel
                acc_li.append(acc_for_rel)
                end_predict_time = time.time()
                print(
                    f"[w/o KNN] fact_path:{fact_path} \t #fact {len(queries)} \t acc: {acc_for_rel:.4f} \t time: { end_predict_time- start_time}"
                )
                # -----------------------KNN  matching

                assert len(predict_answers) == len(gold_cuis)
                if args.use_knn:
                    correct, correct_at_5, correct_at_10 = 0, 0, 0
                    bs = 128
                    for i in np.arange(0, len(predict_answers), bs):
                        batch_answers = predict_answers[i : i + bs]
                        batch_gold_cui = gold_cuis[i : i + bs]
                        batch_gold_name = gold_names[i : i + bs]
                        query_toks = self.tokenizer.batch_encode_plus(
                            batch_answers,
                            padding="max_length",
                            max_length=args.max_entity_length,
                            truncation=True,
                            return_tensors="pt",
                        )
                        toks_cuda = {}
                        for k, v in query_toks.items():
                            toks_cuda[k] = v.cuda()
                        query_output = self.model_base(**toks_cuda)
                        query_cls_rep = query_output[0][:, 0, :]  # cls

                        D, nn_indices = self.index.search(
                            query_cls_rep.cpu().detach().numpy().astype("float32"),
                            10,
                        )  # actual search
                        # print(f"nn_indices: {nn_indices}")
                        for j in range(len(nn_indices)):  # batch
                            knn_predicts_to_save.append(
                                self.all_names[nn_indices[j][0]]
                            )
                            if (
                                self.all_cuis[nn_indices[j][0]] in batch_gold_cui[j]
                            ):  # the first result hit gold tail cuid
                                correct += 1
                                knn_recalls_to_save.append(1)
                                print(
                                    f"predict answer:{batch_answers[j]} \t cui:{self.all_cuis[nn_indices[j][0]]} \t name: {self.all_names[nn_indices[j][0]]} \t gold_cui: {batch_gold_cui[j]} \t gold_name: {batch_gold_name[j]}"
                                )
                            else:
                                knn_recalls_to_save.append(0)
                            for jj in range(5):
                                if (
                                    self.all_cuis[nn_indices[j][jj]]
                                    in batch_gold_cui[j]
                                ):
                                    correct_at_5 += 1
                                    break

                            for jj in range(len(nn_indices[j])):
                                if (
                                    self.all_cuis[nn_indices[j][jj]]
                                    in batch_gold_cui[j]
                                ):
                                    correct_at_10 += 1
                                    break
                    result_metrics["knn_R@1"] = correct / len(predict_answers)
                    result_metrics["knn_R@5"] = correct_at_5 / len(predict_answers)
                    result_metrics["knn_R@10"] = correct_at_10 / len(predict_answers)
                    end_knn_time = time.time()
                    print(
                        f"[KNN] fact_path:{fact_path} \t #fact {len(queries)} \t R@1: {result_metrics['knn_R@1']:.4f} \t R@5: {result_metrics['knn_R@5']:.4f} \t R@10: {result_metrics['knn_R@10']:.4f} \t time: {end_knn_time-end_predict_time}"
                    )
                from datetime import datetime

                if self.args.log_dir:
                    log_filename = os.path.join(
                        self.args.log_dir,
                        args.fact_dir.split("data/")[-1].replace("/", "_")
                        + fact_path
                        + f"_{datetime.now().strftime('%Y%m%d')}_{args.model_short}_{args.init_method}_{args.iter_method}{'_use_res_vocab' if self.args.use_restrict_vocab else ''}.csv",
                    )
                    log_data = {
                        "query": queries_to_save,
                        "gold": gold_names,
                        "predict": predicts_to_save,
                        "pred_hit": recalls_to_save,
                        "log_prob": probilities_to_save,
                    }
                    if args.use_knn:
                        log_data["knn_predict"] = knn_predicts_to_save
                        log_data["knn_predict_hit"] = knn_recalls_to_save
                    log_df = pd.DataFrame(log_data)
                    log_df.to_csv(log_filename)
                    print(f"Saved predicted result to {log_filename}.")

                wandb.config.update(result_metrics)
                wandb.finish()
            except Exception as e:
                print("bug for pid {}".format(fact_path))
                print(e)
                traceback.print_exc()
                wandb.finish()
                raise e
        wandb.init(project="xfactr_summary")
        summary = {}
        summary["macro recall@1"] = overall_r1 / len(acc_li)
        summary["micro recall@1"] = num_correct_fact / num_fact
        wandb.config.update(self.args)
        wandb.config.update(summary)
        print(
            "macro acc@1: %.4f, micro acc@1: %.4f"
            % (overall_r1 / len(acc_li), num_correct_fact / num_fact)
        )
        print(
            "acc per fact {}/{}={:.4f}\tacc per relation {}\tavg iter {}\tnum_max_mask {}".format(
                num_correct_fact,
                num_fact,
                num_correct_fact / (num_fact + 1e-10),
                np.mean(acc_li),
                np.mean(iters),
                self.summary["num_max_mask"],
            )
        )


def load_entity_lang(filename: str) -> Dict[str, Dict[str, str]]:
    entity2lang = defaultdict(lambda: {})
    with open(filename, "r") as fin:
        for l in fin:
            l = l.strip().split("\t")
            entity = l[0]
            for lang in l[1:]:
                label, lang = lang.rsplit("@", 1)
                entity2lang[entity][lang] = label.strip('"')
    return entity2lang


def load_word_ids(ids: Union[np.ndarray, List[int]], tokenizer, pad_label: str) -> str:
    tokens: List[Tuple[str, int]] = []
    for t in tokenizer.convert_ids_to_tokens(ids):
        if t == pad_label:
            continue
        if t.startswith(SUB_LABEL) and len(tokens) > 0:
            tokens[-1][0] += t[len(SUB_LABEL) :]
            tokens[-1][1] += 1
        else:
            tokens.append([t, 1])
    # return " ".join(map(lambda t: "{}:{}".format(*t) if t[1] > 1 else t[0], tokens))
    return " ".join(map(lambda t: t[0], tokens))


def merge_subwords(
    ids: Union[np.ndarray, List[int]], tokenizer, merge: bool = False
) -> str:
    if not merge:
        return list(tokenizer.convert_ids_to_tokens(ids))
    return NotImplementedError


def iter_decode_beam_search(
    model,
    inp_tensor: torch.LongTensor,  # SHAPE: (batch_size, seq_len)
    raw_mask: torch.LongTensor,  # SHAPE: (batch_size, seq_len)
    attention_mask: torch.LongTensor,  # SHAPE: (batch_size, seq_len)
    restrict_vocab: List[int] = None,
    mask_value: int = 0,  # indicate which value is used for mask
    max_iter: int = None,  # max number of iteration
    init_method: str = "all",
    iter_method: str = "none",
    reprob: bool = False,  # recompute the prob finally
    beam_size: int = 5,
) -> Tuple[torch.LongTensor, torch.Tensor, int]:  # HAPE: (batch_size, seq_len)
    """
    Masks must be consecutive.
    """
    assert init_method in {"all", "left", "confidence"}
    assert iter_method in {"none", "left", "confidence", "confidence-multi"}
    bs, sl = inp_tensor.size(0), inp_tensor.size(1)
    init_mask = inp_tensor.eq(mask_value).long()  # SHAPE: (batch_size, seq_len)
    init_has_mask = init_mask.sum().item() > 0

    if iter_method == "confidence-multi":
        number_to_mask = torch.unique(init_mask.sum(-1))
        assert (
            number_to_mask.size(0) == 1
        ), "this batch has different numbers of mask tokens"
        number_to_mask = number_to_mask[0].item() - 1
        assert max_iter == 0, "do not need to set max_iter in confidence-multi setting"
    elif iter_method == "left":
        leftmost_mask = (
            init_mask
            * torch.cat([init_mask.new_ones((bs, 1)), 1 - init_mask], 1)[:, :-1]
        )
        number_to_mask = torch.unique(init_mask.sum(-1))
        assert (
            number_to_mask.size(0) == 1
        ), "this batch has different numbers of mask tokens"
        number_to_mask: int = number_to_mask[0].item()
        mask_offset: int = 0
        has_modified: bool = (
            False  # track wether modification happens during a left-to-right pass
        )

    # SHAPE: (<=beam_size, batch_size, seq_len)
    out_tensors: List[torch.LongTensor] = inp_tensor.unsqueeze(0)
    # tokens not considered have log prob of zero
    out_logprobs: List[torch.Tensor] = torch.zeros_like(inp_tensor).float().unsqueeze(0)
    iter: int = 0
    stop: bool = False
    while True and init_has_mask:  # skip when there is not mask initially
        next_out_tensors = []
        next_out_logprobs = []

        # enumerate over all previous result
        for out_tensor, out_logprob in zip(out_tensors, out_logprobs):
            # print(tokenizer.convert_ids_to_tokens(out_tensor[0].cpu().numpy()))

            # get input
            if iter > 0:
                if iter_method == "none":
                    inp_tensor = out_tensor
                    if inp_tensor.eq(mask_value).long().sum().item() == 0:  # no mask
                        stop = True
                        break
                elif iter_method == "confidence":
                    has_mask = (
                        out_tensor.eq(mask_value).any(-1).unsqueeze(-1).long()
                    )  # SHAPE: (batch_size, 1)
                    inp_tensor = out_tensor.scatter(
                        1, out_logprob.min(-1)[1].unsqueeze(-1), mask_value
                    )
                    # no need to insert mask when there are masks
                    inp_tensor = out_tensor * has_mask + inp_tensor * (1 - has_mask)
                elif iter_method == "confidence-multi":
                    has_mask = (
                        out_tensor.eq(mask_value).any(-1).unsqueeze(-1)
                    )  # SHAPE: (batch_size, 1)
                    all_has_mask = has_mask.all().item()
                    assert (
                        all_has_mask == has_mask.any().item()
                    ), "some samples have masks while the others do not"
                    if not all_has_mask:
                        if number_to_mask <= 0:
                            stop = True
                            break
                        inp_tensor = out_tensor.scatter(
                            1,
                            (-out_logprob).topk(number_to_mask, dim=-1)[1],
                            mask_value,
                        )
                        init_method = "all"
                        number_to_mask -= 1
                    else:
                        inp_tensor = out_tensor
                elif iter_method == "left":
                    has_mask = (
                        out_tensor.eq(mask_value).any(-1).unsqueeze(-1)
                    )  # SHAPE: (batch_size, 1)
                    all_has_mask = has_mask.all().item()
                    any_has_mask = has_mask.any().item()
                    assert (
                        all_has_mask == any_has_mask
                    ), "some samples have masks while the others do not"
                    if not all_has_mask:  # no mask, should do refinement
                        if mask_offset >= number_to_mask:
                            mask_offset = 0
                        if mask_offset == 0:  # restart when starting from the beginning
                            has_modified = False
                        cur_mask = torch.cat(
                            [leftmost_mask.new_zeros((bs, mask_offset)), leftmost_mask],
                            1,
                        )[:, :sl]
                        cur_mask = cur_mask * init_mask
                        inp_tensor = out_tensor * (1 - cur_mask) + mask_value * cur_mask
                        mask_offset += 1
                    else:
                        inp_tensor = out_tensor
                else:
                    raise NotImplementedError

            # predict
            # SHAPE: (batch_size, seq_len)
            mask_mask = inp_tensor.eq(mask_value).long()
            logit = model_prediction_wrap(model, inp_tensor, attention_mask)
            if restrict_vocab is not None:
                logit[:, :, restrict_vocab] = float("-inf")
            # SHAPE: (batch_size, seq_len, beam_size)
            new_out_logprobs, new_out_tensors = logit.log_softmax(-1).topk(
                beam_size, dim=-1
            )

            if init_method == "confidence":
                # mask out non-mask positions
                new_out_logprobs = (
                    new_out_logprobs + mask_mask.unsqueeze(-1).float().log()
                )
                new_out_logprobs = new_out_logprobs.view(-1, sl * beam_size)
                new_out_tensors = new_out_tensors.view(-1, sl * beam_size)

            for b in range(beam_size):
                if init_method == "all":
                    new_out_logprob = new_out_logprobs[:, :, b]
                    new_out_tensor = new_out_tensors[:, :, b]
                    # SHAPE: (batch_size, seq_len)
                    changes = (out_tensor * mask_mask).ne(new_out_tensor * mask_mask)
                elif init_method == "left":  # only modify the left-most one.
                    new_out_logprob = new_out_logprobs[:, :, b]
                    new_out_tensor = new_out_tensors[:, :, b]
                    # SHAPE: (batch_size, seq_len)
                    changes = (out_tensor * mask_mask).ne(new_out_tensor * mask_mask)
                    changes = (
                        changes
                        & torch.cat([changes.new_ones((bs, 1)), ~changes], 1)[:, :-1]
                    )
                elif init_method == "confidence":  # only modify the most confident one.
                    # SHAPE: (batch_size,)
                    raw_lp, raw_ind = new_out_logprobs.max(-1)
                    # SHAPE: (batch_size, 1)
                    raw_lp, raw_ind = raw_lp.unsqueeze(-1), raw_ind.unsqueeze(-1)
                    seq_ind = raw_ind // beam_size
                    changes = mask_mask & torch.zeros_like(mask_mask).scatter(
                        1, seq_ind, True
                    )
                    new_out_tensor = torch.zeros_like(out_tensor).scatter(
                        1, seq_ind, new_out_tensors.gather(1, raw_ind)
                    )
                    new_out_logprob = torch.zeros_like(out_logprob).scatter(
                        1, seq_ind, raw_lp
                    )
                    changes = (out_tensor * changes.long()).ne(
                        new_out_tensor * changes.long()
                    )
                    # max for the next max in beam search
                    new_out_logprobs = new_out_logprobs.scatter(
                        1, raw_ind, float("-inf")
                    )
                else:
                    raise NotImplementedError

                # only modify tokens that have changes
                changes = changes.long()
                _out_tensor = out_tensor * (1 - changes) + new_out_tensor * changes
                _out_logprob = (
                    out_logprob * (1 - changes.float())
                    + new_out_logprob.detach() * changes.float()
                )

                # involves heavy computation, where we re-compute probabilities for beam_size * beam_size samples
                if reprob:
                    _out_logprob = compute_likelihood(
                        model,
                        _out_tensor,
                        _out_logprob,
                        init_mask,
                        attention_mask,
                        restrict_vocab,
                        mask_value=mask_value,
                    )
                    _out_logprob = _out_logprob * (
                        1 - _out_tensor.eq(mask_value).float()
                    )  # skip mask tokens

                next_out_tensors.append(_out_tensor)
                next_out_logprobs.append(_out_logprob)

                """
                for i in range(bs):
                    print(tokenizer.convert_ids_to_tokens(inp_tensor[i].cpu().numpy()))
                    print(tokenizer.convert_ids_to_tokens(_out_tensor[i].cpu().numpy()))
                input()
                """

        if stop:
            break

        next_out_tensors = torch.stack(next_out_tensors, 0)
        next_out_logprobs = torch.stack(next_out_logprobs, 0)
        # tie breaking
        next_out_logprobs = next_out_logprobs + get_tie_breaking(
            int(next_out_logprobs.size(0))
        ).view(-1, 1, 1).to(next_out_logprobs.device)

        # dedup
        not_dups = []
        for i in range(bs):
            abs = next_out_tensors.size(0)
            # SHAPE: (all_beam_size, seq_len)
            one_sample = next_out_tensors[:, i, :]
            # SHAPE: (all_beam_size,)
            inv = torch.unique(one_sample, dim=0, return_inverse=True)[1]
            # SHAPE: (all_beam_size, all_beam_size)
            not_dup = inv.unsqueeze(-1).ne(inv.unsqueeze(0)) | (
                torch.arange(abs).unsqueeze(-1) <= torch.arange(abs).unsqueeze(0)
            ).to(inv.device)
            # SHAPE: (all_beam_size,)
            not_dup = not_dup.all(-1)
            not_dups.append(not_dup)
        # SHAPE: (all_beam_size, batch_size)
        not_dups = torch.stack(not_dups, -1)

        # select top
        # SHAPE: (all_beam_size, batch_size)
        beam_score = (
            next_out_logprobs * init_mask.unsqueeze(0).float()
            + not_dups.unsqueeze(-1).float().log()
        ).sum(-1)
        # SHAPE: (beam_size, batch_size, seq_len)
        beam_top = beam_score.topk(beam_size, dim=0)[1].view(-1, bs, 1).repeat(1, 1, sl)
        next_out_logprobs = torch.gather(next_out_logprobs, 0, beam_top)
        next_out_tensors = torch.gather(next_out_tensors, 0, beam_top)

        # stop condition for other type of iter
        if (
            next_out_tensors.size(0) == out_tensors.size(0)
            and next_out_tensors.eq(out_tensors).all()
        ):
            if iter_method != "left":
                stop = True
        else:
            if iter_method == "left":
                has_modified = True
        # stop condition for 'left' iter
        if iter_method == "left" and not has_modified and mask_offset == number_to_mask:
            # reach the last position and no modification happens during this iteration
            stop = True

        # print(next_out_tensors.ne(out_tensors).any(-1).any(0).nonzero())

        out_tensors = next_out_tensors
        out_logprobs = next_out_logprobs

        iter += 1
        if max_iter and iter >= max_iter:  # max_iter can be zero
            stop = True
        if stop:
            break

    out_tensor = out_tensors[0]
    final_out_logprob = out_logprobs[0]

    return out_tensor, final_out_logprob, iter


def compute_likelihood(
    model,
    inp_tensor: torch.LongTensor,  # SHAPE: (batch_size, seq_len)
    lp_tensor: torch.Tensor,  # SHAPE: (batch_size, seq_len)
    mask_tensor: torch.LongTensor,  # SHAPE: (batch_size, seq_len)
    attention_mask: torch.LongTensor,  # SHAPE: (batch_size, seq_len))
    restrict_vocab: List[int] = None,
    mask_value: int = 0,  # indicate which value is used for mask
) -> torch.Tensor:  # SHAPE: (batch_size, seq_len)
    """
    Masks must be consecutive.
    """
    bs, seq_len = inp_tensor.size(0), inp_tensor.size(1)
    max_num_masks = mask_tensor.sum(-1).max().item()
    leftmost_mask = (
        mask_tensor
        * torch.cat([mask_tensor.new_ones((bs, 1)), 1 - mask_tensor], 1)[:, :-1]
    )
    logits = None
    for i in range(max_num_masks):
        # SHAPE: (batch_size, seq_len)
        cur_mask = (
            torch.cat([leftmost_mask.new_zeros((bs, i)), leftmost_mask], 1)[:, :seq_len]
            * mask_tensor
        )
        inp_tensor_ = (1 - cur_mask) * inp_tensor + cur_mask * mask_value
        logit = model_prediction_wrap(model, inp_tensor_, attention_mask)
        cur_mask = cur_mask.unsqueeze(-1).float()
        if logits is None:
            logits = (logit * cur_mask).detach()
        else:
            logits = (logits * (1 - cur_mask) + logit * cur_mask).detach()
    if restrict_vocab is not None:
        logits[:, :, restrict_vocab] = float("-inf")
    lp = logits.log_softmax(-1)
    lp = torch.gather(
        lp.view(-1, lp.size(-1)), 1, inp_tensor.view(-1).unsqueeze(-1)
    ).view(bs, seq_len)
    lp_tensor = (1 - mask_tensor).float() * lp_tensor + mask_tensor.float() * lp
    return lp_tensor.detach()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="probe LMs with multilingual LAMA")
    parser.add_argument(
        "--model", type=str, help="LM to probe file", default="bert-base-uncased"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="language to probe",
        choices=["default_prompt", "human_prompt", "en"],
        default="default_prompt",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        help="file path of the prompts",
        default="../data/prompt_all.csv",
    )
    # decoding-related flags
    parser.add_argument(
        "--num_mask", type=int, help="the maximum number of masks to insert", default=5
    )
    parser.add_argument(
        "--max_entity_length", type=int, help="max_entity_length", default=20
    )
    parser.add_argument(
        "--max_query_length", type=int, help="max_query_length", default=64
    )

    parser.add_argument(
        "--max_iter",
        type=int,
        help="the maximum number of iteration in decoding",
        default=10,
    )
    parser.add_argument(
        "--init_method", type=str, help="iteration method", default="left"
    )
    parser.add_argument(
        "--iter_method", type=str, help="iteration method", default="none"
    )
    parser.add_argument(
        "--no_len_norm", action="store_true", help="not use length normalization"
    )
    parser.add_argument(
        "--reprob", action="store_true", help="recompute the prob finally"
    )
    parser.add_argument(
        "--prob_hard",
        action="store_true",
    )
    parser.add_argument(
        "--use_knn", action="store_true", help="use knn to match the predicted entity"
    )
    parser.add_argument(
        "--use_restrict_vocab",
        action="store_true",
        help="only use the tokens in the eneity vocab",
    )
    parser.add_argument("--beam_size", type=int, help="beam search size", default=1)

    # others
    parser.add_argument(
        "--fact_dir", type=str, help="directory to the csv facts files", default=None
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        help="directory to vis prediction results",
        default="./output",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="the real batch size is this times num_mask",
        default=20,
    )
    parser.add_argument("--no_cuda", action="store_true", help="not use cuda")
    args = parser.parse_args()

    if (args.init_method != "all" or args.iter_method != "none") and args.max_iter:
        assert args.max_iter >= args.num_mask, "the results will contain mask"
    # load data

    if args.iter_method == "confidence-multi":
        args.max_iter = 0
    print("Init ProbeIterator")
    print(args)
    probe_iter = ProbeIterator(args)
    probe_iter.iter()
