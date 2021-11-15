CONCEPTS_DIR="/mnt/epinas/workspace/data/kgs/umls/raw/META/MRCONSO.RRF"
MAX_SEQ_LENGTH=50
BATCH_SIZE=128
GENERATE_LENGTH=10

MASK_TARGET="tail"
TRIPLES_DIR="../data/"
MODEL_NAME="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

KG_NAMES=(MSH_14970_10_core.csv RXNORM_20000_uniform.csv SNOMEDCT_US_22221_10_core.csv MSH_14970_10_core.csv MSH_20000_uniform.csv SNOMEDCT_US_20000_uniform.csv RXNORM_16496_10_core.csv)

MODEL_NAMES=(
    "dmis-lab/biobert-v1.1" \
)
for KG_NAME in ${KG_NAMES[*]}; do
    for MODEL_NAME in ${MODEL_NAMES[*]}; do
        CUDA_VISIBLE_DEVICES="1", python run_multi_token_mean_prob.py \
        --triples_dir $TRIPLES_DIR$KG_NAME \
        --concepts_dir $CONCEPTS_DIR \
        --model_name $MODEL_NAME \
        --mask_target $MASK_TARGET \
        --max_seq_length $MAX_SEQ_LENGTH \
        --batch_size $BATCH_SIZE \
        --cuda
    done
done

BATCH_SIZE=16

MODEL_NAMES=(
    "bert-large-uncased" \
)

for KG_NAME in ${KG_NAMES[*]}; do
    for MODEL_NAME in ${MODEL_NAMES[*]}; do
        CUDA_VISIBLE_DEVICES="1", python run_multi_token_mean_prob.py \
        --triples_dir $TRIPLES_DIR$KG_NAME \
        --concepts_dir $CONCEPTS_DIR \
        --model_name $MODEL_NAME \
        --mask_target $MASK_TARGET \
        --max_seq_length $MAX_SEQ_LENGTH \
        --batch_size $BATCH_SIZE \
        --cuda
    done
done