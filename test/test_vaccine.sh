# ENV VARIBALES FOR DELTA AI SERVER

# export HF_TOKEN="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"
# export HF_HOME="/work/nvme/bcaq/pkargupta/hf_cache_rlhf/"
# export HF_DATASETS_CACHE="/work/nvme/bcaq/pkargupta/hf_cache_rlhf/"
# export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real # WORKS MUST DO! 

# VARIABLES
CLAIM="The Pfizer COVID-19 vaccine is better than the Moderna COVID-19 vaccine."
TOPIC="vaccine"
CHAT_MODEL_NAME="gpt-4o"
EMBEDDING_MODEL_NAME="openai"
MAX_DEPTH=1


python main.py \
    --claim "$CLAIM" \
    --topic "$TOPIC" \
    --chat_model_name "$CHAT_MODEL_NAME" \
    --embedding_model_name "$EMBEDDING_MODEL_NAME" \
    --max_depth "$MAX_DEPTH"
