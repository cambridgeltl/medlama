CUDA_VISIBLE_DEVICES=$1 python3 train.py \
	--train_dir ../../../data/elicit_corpora/pubmed_sents_no_punkt_10k_0 \
	--output_dir ../../../models/mirror_bert/PubMedBERT_rewired_model \
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
	--model_dir "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" \
#   --model_dir "bert-base-uncased" \	
# 	--save_checkpoint_all \
#	--model_dir "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12" \
#	--model_dir "dmis-lab/biobert-v1.1" \
