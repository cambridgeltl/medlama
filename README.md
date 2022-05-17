## Rewire-then-Probe: A Contrastive Recipe for Probing Biomedical Knowledge of Pre-trained Language Models
**Authors:** Zaiqiao Meng, Fangyu Liu, Ehsan Shareghi, Yixuan Su, Charlotte Collins, Nigel Collier

Code of our paper: [Rewire-then-Probe: A Contrastive Recipe for Probing Biomedical Knowledge of Pre-trained Language Models](https://arxiv.org/abs/2110.08173)

----
### News:
-  [24 Fen 2022] - Our paper has been accepted to appear at the [**ACL 2022**](https://www.2022.aclweb.org/) Main conference.
-  [15 Nov 2021] - The first version of MedLAMA is released.
----
### Introduction
Knowledge probing is crucial for understanding the knowledge transfer mechanism behind the pre-trained language models (PLMs). Despite the growing progress of probing knowledge for PLMs in the general domain, specialised areas such as the biomedical domain are vastly under-explored. To facilitate this, we release a well-curated biomedical knowledge probing benchmark, MedLAMA, constructed based on the Unified Medical Language System (UMLS) Metathesaurus. We test a wide spectrum of state-of-the-art PLMs and probing approaches on our benchmark, reaching at most 3\% of acc@10. While highlighting various sources of domain-specific challenges that amount to this underwhelming performance, we illustrate that the underlying PLMs have a higher potential for probing tasks. To achieve this, we propose Contrastive-Probe, a novel self-supervised contrastive probing approach, that adjusts the underlying PLMs without using any probing data. While Contrastive-Probe pushes the acc@10 to 28\%, the performance gap still remains notable. Our human expert evaluation suggests that the probing performance of our Contrastive-Probe is still under-estimated as UMLS  still does not include the full spectrum of factual knowledge. We hope MedLAMA and Contrastive-Probe facilitate further developments of more suited probing techniques for this domain.
 ![front-page-graph](/imgs/probing_approaches.jpg)

------
### Repo structure

- `data`: data used in the probing experiments.
  - `elicit_corpora`: sampled random sentence from Wikipedia or PubMed articles.
  - `medlama`: our collected biomedical knowledge probing benchmark, *MedLAMA*, where the `2021AA` contains the Full set & Hard set of knowledge queries, and the `prompts.csv` is file containing the query templates.
- `probers`: source code for different categories of probing approaches.
- `notebooks`: notebook for some testing codes, data preprocessing or result (e.g. figures of the paper) output. 

### MedLAMA
 MedLAMA is a well-curated biomedical knowledge probing benchmark that consists of 19 thoroughly selected relations. Each relation contains 1k queries (19k queries in total with at most 10 answers each), which are extracted from the large UMLS biomedical knowledge graph and verified by domain experts. We use automatic metrics to identify the hard examples based on the hardness of exposing  answers from their query tokens.
| Benchmark | # Relations | # Queries | Avg. # answer | Avg. # Char | % Single-Tokens |
| :---: | :---: | :---: | :---: | :---: | :---: |
| LAMA | 41 | 41k | 1 | 7.11 | 100% |
| Bio-LAMA | 36 | 49k | 1 | 18.40 | 2.2% |
| **MedLAMA** | 19 | 19k | 2.3 | 20.88 | 2.6% |

You can find the full MedLAMA dataset in [here](https://github.com/cambridgeltl/medlama/tree/master/data/medlama).
### Contrastive-Probe
You can find the full Contrastive-Probe codes in [here](https://github.com/cambridgeltl/medlama/tree/master/src/probers/retrieval_predict/contrastive_probe/).
Two steps to probe a pretrained BERT model: 

1. **Rewire:** Pre-trains the PLM on a small number of sentences sampled from the PLM's *original pre-training corpora* with a contrastive self-supervising objective.
<details>
  <summary>Click to expand!</summary>
  
```shell
OUTPUT_DIR=../../../models/mirror_bert/PubMedBERT_rewired_model
MODEL_DIR=microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
TRAIN_DIR=../../../data/elicit_corpora/pubmed_sents_no_punkt_10k_0
	
CUDA_VISIBLE_DEVICES=$1 python3 train.py \
	--train_dir $TRAIN_DIR \
	--output_dir $OUTPUT_DIR \
	--use_cuda \
	--epoch 10 \
	--train_batch_size 192 \
	--learning_rate 2e-5 \
	--max_length 48 \
	--checkpoint_step 50 \
	--parallel \
	--amp \
	--pairwise \
	--random_seed 33 \
	--mask_ratio 0.5 \
	--loss "infoNCE" \
	--infoNCE_tau 0.04 \
	--dropout_rate 0.1 \
	--agg_mode "cls" \
	--use_layer -1 \
	--model_dir $MODEL_DIR
```
</details>


2.  **Probe:** Ask the PLM to retrieve a most relevant answer from the answer set based on the [CLS] embeddings of the query and answers.
<details>
<summary>Click to expand!</summary>
	
```shell

MASK_RATIO=0.5
EPOCH=150
SET=1

TEST_DIR=../../../data/medlama/
DATASET=2021AA

CUDA_VISIBLE_DEVICES=$1 python3 run_retrieval_prediction.py \
	--test_dir $TEST_DIR$DATASET \
	--prompt_dir $TEST_DIR"/prompts.csv" \
	--prompt_type human_prompt \
	--model_path $OUTPUT_DIR \
	--epoch $EPOCH \
	--use_common_vocab \
	--log
```
</details>


### If you find our paper and resources useful, please kindly cite our paper:
```latex
@article{meng2021rewire,
  title={Rewire-then-Probe: A Contrastive Recipe for Probing Biomedical Knowledge of Pre-trained Language Models},
  author={Meng, Zaiqiao and Liu, Fangyu and Shareghi, Ehsan and Su, Yixuan and Collins, Charlotte and Collier, Nigel},
  journal={arXiv preprint arXiv:2110.08173},
  year={2021}
}
```
### Contact
If you have any questions, feel free to contact me via (zm324@cam.ac.uk).
