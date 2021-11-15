MODEL=../../../models/mirror_bert/PubMedBERT_rewired_model

MASK_RATIO=0.5
EPOCH=150
SET=1

TEST_DIR=../../../data/medlama/
DATASET=2021AA

CUDA_VISIBLE_DEVICES='0' python3 run_retrieval_prediction.py \
	--test_dir $TEST_DIR$DATASET \
	--prompt_dir $TEST_DIR"/prompts.csv" \
	--prompt_type human_prompt \
	--model_path $MODEL \
	--epoch $EPOCH \
	--use_common_vocab \
	--log
