ANALYSIS_RESULTS_DIR="res/vaccine"
OUTPUT_DIR="eval/metrics/vaccine"
HIERARCHY_PREFIX="res/vaccine"
HIERARCHY_SUFFIX="3_3_3/aspect_hierarchy.json"
LLM_JUDGE='gpt-4o-mini'
BASELINE_MODEL_NAME="llama-3.1-8b-instruct"
TAXO_HEIGHT=3
CHILD_NUM_PER_NODE=3
DATA_DIR="data"
TOPIC="vaccine"
ABLATION_PATH="res/vaccine_nodisc"

python -m eval.eval \
    --result_directory "$ANALYSIS_RESULTS_DIR" \
    --ablation_directory "$ABLATION_PATH" \
    --output_directory "$OUTPUT_DIR" \
    --hierarchy_prefix "$HIERARCHY_PREFIX" \
    --hierarchy_suffix "$HIERARCHY_SUFFIX" \
    --llm_judge "$LLM_JUDGE" \
    --baseline_model_name "$BASELINE_MODEL_NAME" \
    --taxo_height "$TAXO_HEIGHT" \
    --child_num_per_node "$CHILD_NUM_PER_NODE" \
    --data_dir "$DATA_DIR" \
    --topic "$TOPIC" \
    --do_eval_ablation
