import argparse
from vllm import LLM
from vllm import SamplingParams
from outlines.serve.vllm import JSONLogitsProcessor
import json
from tqdm import tqdm
import numpy as np
from collections import deque
from contextlib import redirect_stdout

from api.local.e5_model import E5
from api.local.e5_model import e5_embed
from api.openai.embed import embed as openai_embed

from hierarchy import Paper, AspectNode, Tree, Segment
from keyword_generation.keyword_extractor import extract_keyword, extract_keywords, stage1_retrieve_top_k_corpus_segments
from prompts import aspect_list_schema, aspect_prompt
from segment_ranking import aspect_segment_ranking
from discovery import subaspect_discovery, perspective_discovery
from unidecode import unidecode

def load_data(args, chunk_size=3):
    with open(f'{args.data_dir}/{args.topic}/{args.topic}_text.txt', 'r', encoding='utf-8', errors='ignore') as f:
        corpus = []
        paper_id = 0
        global_id = 0

        for line in f:
            # chunk input paper content
            sents = line.strip().lower().split('. ')
            local_id = 0
            paper_segments = []
            for i in np.arange(0, len(sents), chunk_size):
                seg_content = unidecode(". ".join(sents[i:i+chunk_size]))
                paper_segments.append(Segment(global_id, local_id, paper_id, seg_content))
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

def main(args):
    # input corpus -> @Runchu is writing the dataset loader; we assume we have a corpus of Paper classes where the Paper class has an attribute called segments
    # corpus: dict of Paper objects
    # paper.segments: list of Segment objects where

    print("######## LOADING DATA ########")
    claim = args.claim
    id2node = []
    corpus = load_data(args)
    
    # Initialize tree
    print("######## INITIALIZE TREE ########")
    root_node = AspectNode(idx=0, name=claim, description="")
    for paper in corpus:
        # for the root_node, we naively assume that all segments in the corpus are relevant to the root
        root_node.add_related_paper(paper, paper.segments)
    
    tree = Tree(root_node)
    id2node.append(root_node)

    # Coarse-grained aspect discovery + keyword generation
    aspects = coarse_grained_aspect_discovery(args, claim)
    
    # Expand tree with first level
    print("######## EXPAND AND ENRICH TREE WITH ASPECTS ########")
    
    all_segments = root_node.get_all_segments()
    seg_contents = [seg.content for seg in all_segments]
    corpus_embs = args.embed_func(args.embed_model, seg_contents)

    refined_keywords = extract_keywords(args=args, claim=claim, aspects=aspects, corpus_segments=all_segments,
                                        retrieved_corpus_num=5, min_keyword_num=5, max_keyword_num=15, iteration_num=1,
                                        corpus_embs=corpus_embs)
    
    for a_idx, aspect in enumerate(aspects):
        # Refine keywords
        aspect_node = AspectNode(idx=len(id2node), name=aspect["aspect_label"], description=aspect["aspect_description"], parent=root_node, keywords=refined_keywords[a_idx])
        print(f'{aspect_node.name} keywords: {str(aspect_node.keywords)}')
        tree.add_aspect(root_node, aspect_node)
        id2node.append(aspect_node)

    # Initialize a queue where for each aspect node, we:
    print("######## SUBASPECT EXPANSION BEGINS ########")
    queue = deque(root_node.sub_aspects)

    while queue:
        current_node:AspectNode = queue.popleft()
        print(f"Current Node: {current_node.name}; {len(queue)} nodes left in the queue!")
        
        ## (1) retrieve relevant segments based on refined keywords
        top_k_segments = stage1_retrieve_top_k_corpus_segments(args=args,
                                                               claim=claim,
                                                               aspect_name=current_node.name,
                                                               aspect_description=current_node.description,
                                                               corpus_segments=all_segments,
                                                               retrieved_corpus_num=100,
                                                               current_keyword_group=current_node.keywords,
                                                               corpus_embs=corpus_embs)
        top_k_seg_contents = [seg.content for seg in top_k_segments]

        ## (2) rank the segments
        rank2id, id2score = aspect_segment_ranking(args=args,
                               segments=top_k_seg_contents, 
                               target_aspect=current_node, 
                               neg_aspects=current_node.get_siblings())

        current_node.ranked_segments = {i:(top_k_segments[rank2id[i]], id2score[rank2id[i]]) for i in np.arange(len(top_k_segments))}

        if current_node.depth < args.max_depth:
        
            ## (3) identify the relevant subaspects from the top-k ranked segments
            subaspects = subaspect_discovery(args=args,
                                segments=top_k_seg_contents,
                                rank2id=rank2id,
                                parent_aspect=current_node,
                                top_k=args.top_k, temperature=0.7, top_p=0.99)
            
            ## (4) perform depth expansion using these subaspects
            refined_keywords = extract_keywords(args=args, claim=claim, aspects=subaspects, corpus_segments=all_segments,
                                                retrieved_corpus_num=5, min_keyword_num=5, max_keyword_num=15, iteration_num=1,
                                                corpus_embs=corpus_embs, is_subaspect=True)
            
            for s_idx, subaspect in tqdm(enumerate(subaspects), total=len(subaspects)):
                # Refine keywords
                subaspect_node = AspectNode(idx=len(id2node), name=subaspect["subaspect_label"], description=subaspect["subaspect_description"], parent=current_node, keywords=refined_keywords[s_idx])
                tree.add_aspect(current_node, subaspect_node)
                print(f'{subaspect_node.name} keywords: {str(subaspect_node.keywords)}')
                id2node.append(subaspect_node)
                
                queue.append(subaspect_node)

    print("######## DISCOVERING PERSPECTIVES ########")
    perspective_discovery(args, id2node)
    
    print("######## OUTPUT ASPECT HIERARCHY ########")
    with open(f'{args.data_dir}/{args.topic}/aspect_hierarchy.txt', 'w') as f:
        with redirect_stdout(f):
            hierarchy_dict = root_node.display(indent_multiplier=5, visited=None, corpus_len=len(corpus))

    with open(f'{args.data_dir}/{args.topic}/hierarchy.json', 'w', encoding='utf-8') as f:
        json.dump(hierarchy_dict, f, ensure_ascii=False, indent=4)


    # # Hierarchical segment classification
    # tree = hierarchical_segment_classification(claim, aspect_hierarchy)

    # # Perspective discovery
    # perspective_discovery(tree, claim)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim", default="The Pfizer COVID-19 vaccine is better than the Moderna COVID-19 vaccine.")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--topic", default="vaccine")
    parser.add_argument("--chat_model_name", default="vllm")
    parser.add_argument("--embedding_model_name", default="e5")
    parser.add_argument("--top_k", type=float, default=5)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--gamma", type=float, default=2)
    parser.add_argument("--max_depth", type=int, default=2)
    args = parser.parse_args()

    if args.embedding_model_name == "e5":
        args.embed_model = E5()
        args.embed_func = e5_embed
    else:
        args.embed_model = args.embedding_model_name
        args.embed_func = openai_embed

    if args.chat_model_name == "vllm":
        args.chat_model = LLM(model="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF", tensor_parallel_size=4, max_num_seqs=100, enable_prefix_caching=True)

    main(args)