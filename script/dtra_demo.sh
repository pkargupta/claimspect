python -m src.main \
    --claim "The U.S. deployment of biological security infrastructure in Ukraine reflects a tactical bid to encircle Russia with potential biotechnology threats." \
    --data_dir "data" \
    --topic "dtra" \
    --chat_model_name "gpt-4o" \
    --embedding_model_name "e5" \
    --max_aspect_children_num 3 \
    --max_subaspect_children_num 3
