ANALYSIS_RESULTS_DIR="res/dtra"
OUTPUT_DIR="eval/metrics/dtra"
HIERARCHY_PREFIX="res/dtra"
HIERARCHY_SUFFIX="3_3_3/aspect_hierarchy.json"
LLM_JUDGE='gpt-4o-mini'

python -m eval.eval \
    --result_directory "$ANALYSIS_RESULTS_DIR" \
    --output_directory "$OUTPUT_DIR" \
    --hierarchy_prefix "$HIERARCHY_PREFIX" \
    --hierarchy_suffix "$HIERARCHY_SUFFIX" \
    --llm_judge "$LLM_JUDGE" \
    --do_eval_taxonomy_level \
    --do_eval_node_level
