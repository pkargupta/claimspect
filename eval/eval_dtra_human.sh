ANALYSIS_RESULTS_DIR="res/dtra"
OUTPUT_DIR="eval/metrics/dtra"
HIERARCHY_PREFIX="res/dtra"
HIERARCHY_SUFFIX="3_3_3/aspect_hierarchy.json"
LLM_JUDGE='human'
BASELINE_MODEL_NAME="llama-3.1-8b-instruct"
TAXO_HEIGHT=3
CHILD_NUM_PER_NODE=3
DATA_DIR="data"
TOPIC="dtra"
MAX_PATHS=5

python -m eval.eval \
    --result_directory "$ANALYSIS_RESULTS_DIR" \
    --output_directory "$OUTPUT_DIR" \
    --hierarchy_prefix "$HIERARCHY_PREFIX" \
    --hierarchy_suffix "$HIERARCHY_SUFFIX" \
    --llm_judge "$LLM_JUDGE" \
    --baseline_model_name "$BASELINE_MODEL_NAME" \
    --taxo_height "$TAXO_HEIGHT" \
    --child_num_per_node "$CHILD_NUM_PER_NODE" \
    --data_dir "$DATA_DIR" \
    --topic "$TOPIC" \
    --do_eval_node_level \
    --max_paths "$MAX_PATHS" 