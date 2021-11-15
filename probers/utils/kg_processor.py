import os
import pickle

import pandas as pd
import torch
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm


class KGProcessor(object):
    def __init__(
        self,
        triples_dir,
        concepts_dir=None,
        prompt_file=None,
        prompt="default_prompt",
    ):
        self.name = os.path.basename(triples_dir.split(".csv")[0])
        self.model_name = ""
        self.triples_dir = triples_dir
        self.concepts_dir = concepts_dir
        # columns should be ['head_cui', 'head_name', 'rel', 'tail_cuis', 'num_tail_cuis', 'tail_names', 'num_tail_names']
        self.triple_df = pd.read_csv(self.triples_dir)
        # self.entities should be a dict with {entity_name:cui}
        self.entities, self.num_entity = self.get_entities()
        self.init_queries()
        # get prompt
        self.prompts = {}
        if prompt_file:
            prompt_df = pd.read_csv(prompt_file)
            for _, item in prompt_df.iterrows():
                self.prompts[item["pid"]] = item[prompt]
        self.num_relation = self.triple_df["rel"].nunique()

    def get_entities(self):
        """read queries from dataframe

        Returns:
            [type]: [description]
        """
        entity2cui = {}
        self.entity_type_vocab = {}  # to collect the entity type
        for _, item in self.triple_df.iterrows():
            rel_str = item["rel"]
            head_name, head_cui = item["head_name"], item["head_cui"]
            if head_name not in entity2cui:
                entity2cui[head_name] = head_cui
            tail_names, tail_cuis = item["tail_names"], item["tail_cuis"]

            if rel_str not in self.entity_type_vocab:
                self.entity_type_vocab[rel_str] = [head_name]
            else:
                if head_name not in self.entity_type_vocab[rel_str]:
                    self.entity_type_vocab[rel_str].append(head_name)

            for tail_name, tail_cui in zip(
                tail_names.split("||"), tail_cuis.split("||")
            ):
                if tail_name not in entity2cui:
                    entity2cui[tail_name] = tail_cui

                if tail_name not in self.entity_type_vocab[rel_str]:
                    self.entity_type_vocab[rel_str].append(tail_name)
        print(f"Get {len(entity2cui)} entities.")
        return entity2cui, len(entity2cui)

    def init_queries(self):
        """read queries from dataframe

        Returns:
            [type]: [description]
        """
        self.queries = []
        self.str_labels = []
        self.cui_labels = []
        self.num_query = 0
        for _, item in self.triple_df.iterrows():
            rel_str = item["rel"]
            tail_names, tail_cuis = item["tail_names"], item["tail_cuis"]
            query = (item["head_name"], rel_str)
            if query not in self.queries:
                self.queries.append(query)
                self.num_query += 1
            str_labels = []
            cui_labels = []
            for tail_name, tail_cui in zip(
                tail_names.split("||"), tail_cuis.split("||")
            ):
                str_labels.append(tail_name)
                cui_labels.append(tail_cui)
            self.str_labels.append(str_labels)
            self.cui_labels.append(cui_labels)
        print(f"Get {self.num_query} queries.")

    def get_query_types(self):
        query_types = []
        for query in self.queries:
            _, rel = query
            query_types.append(rel)
        print(f"get_query_types {len(query_types)}.")
        return query_types

    def get_type_vocab_token_ids(self, tokenizer):
        type_vocab_token_ids = {}
        for entity_type, entity_names in self.entity_type_vocab.items():
            text_features = tokenizer.batch_encode_plus(
                entity_names,
                padding=True,  # First sentence will have some PADDED tokens to match second sequence length
                return_tensors="pt",
                truncation=True,
                add_special_tokens=False,
            )
            type_vocab_token_ids[entity_type] = text_features
        return self.get_query_types(), type_vocab_token_ids

    def get_full_vocab_token_ids(self, tokenizer):
        vocab = list(self.entities.keys())
        text_features = tokenizer.batch_encode_plus(
            vocab,
            padding=True,  # First sentence will have some PADDED tokens to match second sequence length
            return_tensors="pt",
            truncation=True,
            add_special_tokens=False,
        )
        return vocab, text_features

    def load_features(self, tokenizer, max_seq_length, add_special_tokens=True):
        self.model_name = os.path.basename(tokenizer.name_or_path)
        examples = []
        for head_name, rel in self.queries:
            prompt = self.prompts[rel]
            query = prompt.replace("[X]", head_name)
            query = query.replace("[Y]", "[MASK]")
            examples.append(query)
        text_features = tokenizer.batch_encode_plus(
            examples,
            padding="max_length",  # First sentence will have some PADDED tokens to match second sequence length
            max_length=max_seq_length,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=add_special_tokens,
        )
        tokenized_features = TensorDataset(
            text_features.input_ids,
            text_features.attention_mask,
            text_features.token_type_ids,
        )
        return tokenized_features
