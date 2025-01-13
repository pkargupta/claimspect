import argparse
from vllm import LLM
from api.local.e5_model import E5
from vllm import SamplingParams
from outlines.serve.vllm import JSONLogitsProcessor
import json

from hierarchy import Paper, AspectNode, Tree
from prompts import aspect_list_schema, aspect_prompt

def coarse_grained_aspect_discovery(args, claim, temperature=0.3, top_p=0.99):
    """Step 1: Generate coarse-grained aspects for the claim."""
    """Step 2: Generate specific keywords for each aspect."""
    
    logits_processor = JSONLogitsProcessor(schema=aspect_list_schema, llm=args.chat_model.llm_engine)
    sampling_params = SamplingParams(max_tokens=2000, logits_processors=[logits_processor], temperature=temperature, top_p=top_p)

    output = args.chat_model.generate(aspect_prompt(claim), sampling_params=sampling_params)[0].outputs[0].text

    aspects = json.loads(output)['aspect_list']
    
    print(f"Generated coarse-grained aspects for claim '{claim}': {aspects}")
    return aspects

def corpus_segment_ranking(aspect, keywords):
    """Step 3: Rank corpus segments based on relevance to keywords."""
    # Simulated ranked segments.
    ranked_segments = [
        f"Segment 1 discussing {aspect} with keywords {keywords}",
        f"Segment 2 focusing on {aspect} with keywords {keywords}"
    ]
    print(f"Ranked segments for aspect '{aspect}': {ranked_segments}")
    return ranked_segments

def sub_aspect_discovery(aspect, ranked_segments):
    """Step 4: Discover sub-aspects based on ranked segments."""
    sub_aspects = [
        f"Sub-aspect 1 for {aspect}",
        f"Sub-aspect 2 for {aspect}"
    ]
    print(f"Discovered sub-aspects for aspect '{aspect}': {sub_aspects}")
    return sub_aspects

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
    all_segments = root_node.get_all_segments(as_str=True)
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
                                                  iteration_num=2)
        
        aspect_node = AspectNode(idx=len(id2node), name=aspect["aspect_label"], keywords=refined_aspect_keywords)
        tree.add_aspect(self, root_node, aspect_node)
        id2node.append(aspect_node)

    

    # Keyword generation and corpus segment ranking
    # aspect_hierarchy = {}
    # for aspect in aspects:
    #     keywords = keyword_generation(aspect)
    #     ranked_segments = corpus_segment_ranking(aspect, keywords)

    #     # Sub-aspect discovery
    #     sub_aspects = sub_aspect_discovery(aspect, ranked_segments)
    #     aspect_hierarchy[aspect] = sub_aspects

    # # Hierarchical segment classification
    # tree = hierarchical_segment_classification(claim, aspect_hierarchy)

    # # Perspective discovery
    # perspective_discovery(tree, claim)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim", default="The United States' efforts to study and detect dangerous biological pathogens internationally risks misuse by allies and/or targeted biological attacks by adversaries.")
    parser.add_argument("--data_dir", default="datasets/")
    parser.add_argument("--topic", default="biosecurity")
    parser.add_argument("--chat_model_name", default="vllm")
    parser.add_argument("--embedding_model_name", default="e5")
    args = parser.parse_args()

    if args.embedding_model_name == "e5":
        args.embed_model = E5()
    if args.chat_model_name == "vllm":
        args.chat_model = LLM(model="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF", tensor_parallel_size=4, max_num_seqs=100, enable_prefix_caching=True)

    main(args)