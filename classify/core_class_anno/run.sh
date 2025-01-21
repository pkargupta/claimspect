python -m classify.2_core_class_anno.core_class_annotation \
    --llm_enrichment_path classify/1_llm_taxo_enrich/example/llm_enrichment.txt \
    --corpus_path classify/2_core_class_anno/example/corpus.txt \
    --gpu 0 \
    --label_path classify/1_llm_taxo_enrich/example/labels.txt \
    --label_hierarchy_path classify/1_llm_taxo_enrich/example/label_hierarchy.txt \
    --output_path classify/2_core_class_anno/example/init_core_classes_core.json