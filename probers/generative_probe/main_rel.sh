CUDA_VISIBLE_DEVICES=3 python main.py\
    --data_path_prefix ../data/by_char_len/\
    --result_prefix_path ./rel_probing_result/\
    --prompt_path ../data/prompts.csv
