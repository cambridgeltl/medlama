
INIT_METHOD_LIST=("all" "left" "confidence")
ITER_METHOD_LIST=("none" "left" "confidence" )
OUTPUT_DIR=./output/biomedlama/
FACT_DIR=../../../data/biomedlama/
prompt_dir=../../../data/biomedlama/prompts.csv
INIT_METHOD_LIST=("left")
ITER_METHOD=("left")

MODEL_NAMES=(
    "bert-base-uncased" \
    "dmis-lab/biobert-base-cased-v1.2" \
    "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12"
)

VARIABLES=("by_relation_1k")
DEVICE="1"

for MODEL in ${MODEL_NAMES[*]}; do
    for INIT_METHOD in ${INIT_METHOD_LIST[*]}; do
        for VARIABLE in ${VARIABLES[*]}; do
            CUDA_VISIBLE_DEVICES=$DEVICE, python probe.py \
                --model $MODEL \
                --prompt_file $prompt_dir \
                --prompt human_prompt \
                --fact_dir $FACT_DIR$VARIABLE \
                --log_dir $OUTPUT_DIR \
                --use_knn \
                --beam_size 5 \
                --init_method $INIT_METHOD \
                --iter_method $ITER_METHOD \
                --batch_size 1 
        done
    done
done