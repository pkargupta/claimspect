import argparse
from vllm import LLM
from vllm import SamplingParams
from outlines.serve.vllm import JSONLogitsProcessor
import json
from tqdm import tqdm
import numpy as np
from collections import deque

from api.local.e5_model import E5
from api.local.e5_model import e5_embed
from api.openai.embed import openai_embed

from hierarchy import Paper, AspectNode, Tree, Segment
from keyword_generation.keyword_extractor import extract_keyword, stage1_retrieve_top_k_corpus_segments
from prompts import aspect_list_schema, aspect_prompt
from segment_ranking import aspect_segment_ranking
from discovery import subaspect_discovery

def load_data(args, chunk_size=3):
    with open(f'{args.data_dir}/{args.topic}/{args.topic}_text.txt', 'r') as f:
        corpus = []
        paper_id = 0
        global_id = 0

        for line in f:
            # chunk input paper content
            sents = line.strip().lower().split('. ')
            local_id = 0
            paper_segments = []
            for i in np.arange(0, len(sents), chunk_size):
                paper_segments.append(Segment(global_id, local_id, paper_id, sents[i:i+chunk_size]))
                local_id += 1
                global_id += 1

            corpus.append(Paper(paper_id=paper_id, segments=paper_segments))
            paper_id += 1
    
    return corpus

def coarse_grained_aspect_discovery(args, claim, temperature=0.3, top_p=0.99):
    """Step 1: Generate coarse-grained aspects for the claim."""
    """Step 2: Generate specific keywords for each aspect."""
    
    logits_processor = JSONLogitsProcessor(schema=aspect_list_schema, llm=args.chat_model.llm_engine)
    sampling_params = SamplingParams(max_tokens=2000, logits_processors=[logits_processor], temperature=temperature, top_p=top_p)

    output = args.chat_model.generate(aspect_prompt(claim), sampling_params=sampling_params)[0].outputs[0].text

    aspects = json.loads(output)['aspect_list']
    
    print(f"Generated coarse-grained aspects for claim '{claim}': {aspects}")
    return aspects

def hierarchical_segment_classification(claim, aspect_hierarchy):
    """Step 5: Classify segments into the aspect hierarchy."""
    tree = Tree(AspectNode(name=claim, description="Root of the aspect hierarchy"))
    for aspect, sub_aspects in aspect_hierarchy.items():
        aspect_node = AspectNode(name=aspect, description=f"Aspect node for {aspect}")
        for sub_aspect in sub_aspects:
            aspect_node.add_sub_aspect(AspectNode(name=sub_aspect, description=f"Sub-aspect node for {sub_aspect}"))
        tree.add_aspect(tree.root.name, aspect_node)
    tree.display_tree()
    return tree

def perspective_discovery(tree, claim):
    """Step 6: Identify perspectives/stances in the corpus."""
    for aspect in tree.root.sub_aspects:
        for sub_aspect in aspect.sub_aspects:
            print(f"Identified perspective for sub-aspect '{sub_aspect.name}' under aspect '{aspect.name}'")
    print(f"Perspectives discovered for claim '{claim}'.")

def main(args):
    # input corpus -> @Runchu is writing the dataset loader; we assume we have a corpus of Paper classes where the Paper class has an attribute called segments
    # corpus: dict of Paper objects
    # paper.segments: list of Segment objects where
    
    claim = args.claim
    id2node = []
    corpus = load_data(args)
    
    # Initialize tree
    root_node = AspectNode(idx=0, name=claim)
    for paper in corpus:
        # for the root_node, we naively assume that all segments in the corpus are relevant to the root
        root_node.add_related_paper(paper, paper.segments)
    
    tree = Tree(root_node)
    id2node.append(root_node)

    # Coarse-grained aspect discovery + keyword generation
    aspects = coarse_grained_aspect_discovery(args, claim)
    
    # Expand tree with first level
    all_segments = root_node.get_all_segments()
    for aspect in aspects:
        # Refine keywords
        refined_aspect_keywords = extract_keyword(args, 
                                                  claim=claim, 
                                                  aspect_name=aspect["aspect_label"],
                                                  aspect_keywords_from_llm=aspect["aspect_keywords"],
                                                  corpus_segments=all_segments,
                                                  retrieved_corpus_num=5,
                                                  min_keyword_num=5,
                                                  max_keyword_num=10,
                                                  iteration_num=1)
        
        aspect_node = AspectNode(idx=len(id2node), name=aspect["aspect_label"], parent=root_node, keywords=refined_aspect_keywords)
        tree.add_aspect(root_node, aspect_node)
        id2node.append(aspect_node)

    # Initialize a queue where for each aspect node, we:
    queue = deque(root_node.subaspects)

    while queue:
        current_node:AspectNode = queue.popleft()
        print(f"Current Node: {current_node.name}")
        
        ## (1) retrieve relevant segments based on refined keywords
        top_k_segments = stage1_retrieve_top_k_corpus_segments(args=args,
                                              claim=claim,
                                              aspect_name=current_node.name,
                                              corpus_segments=all_segments,
                                              retrieved_corpus_num=100,
                                              current_keyword_group=current_node.keywords)
        top_k_seg_contents = [seg.content for seg in top_k_segments]

        ## (2) rank the segments
        rank2id, id2rank = aspect_segment_ranking(args=args,
                               segments=top_k_seg_contents, 
                               target_aspect=current_node, 
                               neg_aspects=current_node.get_siblings())
        
        ## (3) identify the relevant subaspects from the top-k ranked segments
        subaspects = subaspect_discovery(args=args,
                            segments=top_k_seg_contents,
                            rank2id=rank2id,
                            parent_aspect=current_node,
                            top_k=args.top_k, temperature=0.7, top_p=0.99)
        
        ## (4) perform depth expansion using these subaspects
        for subaspect in tqdm(subaspects):
            # Refine keywords
            refined_aspect_keywords = extract_keyword(args, 
                                                    claim=claim, 
                                                    aspect_name=subaspect["subaspect_label"],
                                                    aspect_keywords_from_llm=subaspect["subaspect_keywords"],
                                                    corpus_segments=all_segments,
                                                    retrieved_corpus_num=5,
                                                    min_keyword_num=5,
                                                    max_keyword_num=10,
                                                    iteration_num=1)
            
            subaspect_node = AspectNode(idx=len(id2node), name=subaspect["subaspect_label"], parent=current_node, keywords=refined_aspect_keywords)
            tree.add_aspect(current_node, subaspect_node)
            id2node.append(subaspect_node)
            subaspect_node.ranked_segments = top_k_segments
            
            if subaspect_node.depth < args.max_depth:
                queue.append(subaspect_node)


    # # Hierarchical segment classification
    # tree = hierarchical_segment_classification(claim, aspect_hierarchy)

    # # Perspective discovery
    # perspective_discovery(tree, claim)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim", default="The Pfizer COVID-19 vaccine is better than the Moderna COVID-19 vaccine.")
    parser.add_argument("--data_dir", default="datasets")
    parser.add_argument("--topic", default="vaccine")
    parser.add_argument("--chat_model_name", default="vllm")
    parser.add_argument("--embedding_model_name", default="e5")
    parser.add_argument("--top_k", type=float, default=5)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--gamma", type=float, default=2)
    parser.add_argument("--max_depth", type=int, default=3)
    args = parser.parse_args()

    if args.embedding_model_name == "e5":
        args.embed_model = E5()
        args.embed_func = e5_embed
    else:
        args.embed_func = openai_embed

    if args.chat_model_name == "vllm":
        args.chat_model = LLM(model="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF", tensor_parallel_size=4, max_num_seqs=100, enable_prefix_caching=True)

    main(args)