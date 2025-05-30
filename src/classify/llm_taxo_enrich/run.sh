LABEL_PATH="classify/1_llm_taxo_enrich/example/labels.txt"
LABEL_HIERARCHY_PATH="classify/1_llm_taxo_enrich/example/label_hierarchy.txt"
CLAIM_PATH="classify/1_llm_taxo_enrich/example/claim.txt"
OUTPUT_PATH="classify/1_llm_taxo_enrich/example/llm_enrichment.txt"

python classify/1_llm_taxo_enrich/LLM_taxonomy_enrichment.py \
	--label_path $LABEL_PATH \
	--label_hierarchy_path $LABEL_HIERARCHY_PATH \
	--claim_path $CLAIM_PATH \
	--output_path $OUTPUT_PATH