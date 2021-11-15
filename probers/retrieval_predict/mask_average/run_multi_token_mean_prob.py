from argparse import ArgumentParser

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, BertForMaskedLM

import wandb
from common_utils import print_args_as_table
from kg_processor import KGProcessor

wandb.init(project="Bio_LAMA_multi_token_mean_prob_type")


def get_args():
    parser = ArgumentParser(description="Run Multi Token Probing via Mean Probability")
    parser.add_argument("--model_name", default=None, type=str, required=True)
    parser.add_argument("--cuda", action="store_true", help="to use gpu")
    parser.add_argument(
        "--mask_target", default="tail", type=str, help="target entity to be masked"
    )
    parser.add_argument("--triples_dir", default=None)
    parser.add_argument("--concepts_dir", default=None)
    parser.add_argument("--max_seq_length", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    print(args)
    device = torch.device(
        "cuda" if (torch.cuda.is_available() and args.cuda) else "cpu"
    )
    n_gpu = torch.cuda.device_count()
    args.device = device
    args.n_gpu = n_gpu
    wandb.config.update(args)
    print_args_as_table(args)

    model = BertForMaskedLM.from_pretrained(args.model_name)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    mask_token_id = tokenizer(tokenizer.mask_token, add_special_tokens=False)[
        "input_ids"
    ][0]
    special_token_ids = tokenizer(
        [tokenizer.cls_token, tokenizer.sep_token], add_special_tokens=False
    )["input_ids"]
    special_token_ids = sum(special_token_ids, [])

    kg_processor = KGProcessor(args.triples_dir, args.concepts_dir)
    tokenized_examples, examples, labels = kg_processor.load_and_cache_features(
        tokenizer, args.max_seq_length, mask_target=args.mask_target
    )
    # tokenized_vocab, vocab, _ = kg_processor.load_and_cache_features(
    #     tokenizer, args.max_seq_length, mask_target="full"
    # )  # dataset vocab, still contains special tokens
    # kg_vocab_token_ids = torch.stack([entity[0] for entity in tokenized_vocab]).cuda()
    # kg_vocab_attention_mask = torch.stack(
    #     [entity[1] for entity in tokenized_vocab]
    # ).cuda()
    # kg_vocab_attention_mask[:, 0] = 0  # set the [CLS] attention to 0
    # for entity_attention in kg_vocab_attention_mask:  # set the [SEP] attention to 0
    #     entity_attention[entity_attention.nonzero(as_tuple=False)[-1]] = 0
    type_vocab_feature, example_types = kg_processor.get_type_vocab_token_ids(tokenizer)
    type_vocab_token_ids = {}
    type_vocab_attention_mask = {}

    for k, vacab_ids in type_vocab_feature.items():
        type_vocab_token_ids[k] = vacab_ids["input_ids"].cuda()
        type_vocab_attention_mask[k] = vacab_ids["attention_mask"].cuda()

    entity_type_dict = kg_processor.entity_type_dict  # {rel:[entity, entity]}
    from torch.utils.data import DataLoader

    model.eval()
    predicts = []
    hit1 = 0
    hit5 = 0
    data_loader = DataLoader(tokenized_examples, batch_size=args.batch_size)
    for step, batch in enumerate(tqdm(data_loader, desc="Infering")):
        low_idx = step * args.batch_size
        high_idx = (
            (step + 1) * args.batch_size
            if (step + 1) * args.batch_size < len(labels) - 1
            else len(labels) - 1
        )
        batch_labels = labels[low_idx:high_idx]
        batch_types = example_types[low_idx:high_idx]
        input_ids, attention_mask, token_type_ids = [
            data.to(model.device) for data in batch[:3]
        ]
        mask_token_idx = (input_ids == mask_token_id).nonzero(as_tuple=False)
        outputs = model(input_ids, attention_mask, token_type_ids)
        batch_logits = [
            outputs.logits[token_idx[0], token_idx[1]] for token_idx in mask_token_idx
        ]  # batch_size*model_vocab_size
        for logits, label, example_type in zip(batch_logits, batch_labels, batch_types):
            scores = torch.mean(
                logits[type_vocab_token_ids[example_type]]
                * type_vocab_attention_mask[example_type],
                dim=1,
            )
            if scores.size(0) < 5:
                top_k_prediction = entity_type_dict[example_type]
            else:
                top_k = torch.topk(scores, k=5).indices
                top_k_prediction = [entity_type_dict[example_type][i] for i in top_k]
            if label == top_k_prediction[0]:
                hit1 += 1
            if label in top_k_prediction[:5]:
                hit5 += 1
            predicts.append(top_k)
    result = {}
    print(f"recall@1: {hit1/len(labels):.4f}")
    print(f"recall@5: {hit5/len(labels):.4f}")
    result["recall@1"] = hit1 / len(labels)
    result["recall@5"] = hit5 / len(labels)
    wandb.config.update(result)
