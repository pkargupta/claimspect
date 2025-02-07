CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run_experiments \
    --data_dir "data" \
    --topic "dtra" \
    --chat_model_name "vllm" \
    --embedding_model_name "e5" \
    --max_aspect_children_num 3 \
    --max_subaspect_children_num 3
