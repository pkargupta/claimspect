from filter.keyword_ensemble_embedding_llm_judge import KeywordEnsembleEmbeddingLLMFilter

def filter_segments(args, tree):
    
    # extract segments
    segments = tree.root.get_all_segments()
    segment_str_list = [seg.content for seg in segments]
    major_aspects = [aspect.name for aspect in tree.root.sub_aspects]
    
    # init filter
    filter_obj = KeywordEnsembleEmbeddingLLMFilter(args, aspect_list=major_aspects)
    # filter segments
    new_segment_str_list = filter_obj.filter(args.claim, segment_str_list, major_aspects)
    
    # change the results 
    filtered_papers_dict = {}
    for paper_id, paper in tree.root.related_papers.items():
        original_segments = paper["relevant_segments"]
        filtered_segments = []
        for segment in original_segments:
            if segment.content in new_segment_str_list:
                filtered_segments.append(segment)
        
        # if no filtered segments, remove the paper
        if len(filtered_segments) == 0:
            continue
        
        # else, reduce the relevant segments to the filtered ones
        paper["relevant_segments"] = filtered_segments
        
        # update the paper in the dict
        filtered_papers_dict[paper_id] = paper
    
    tree.root.related_papers = filtered_papers_dict