import argparse
from vllm import LLM
from vllm import SamplingParams
from outlines.serve.vllm import JSONLogitsProcessor
import json

from hierarchy import Paper, AspectNode, Tree
from prompts import aspect_list_schema, aspect_prompt

def coarse_grained_aspect_discovery(args, claim, temperature=0.3, top_p=0.99):
    """Step 1: Generate coarse-grained aspects for the claim."""
    """Step 2: Generate specific keywords for each aspect."""
    
    logits_processor = JSONLogitsProcessor(schema=aspect_list_schema, llm=args.model.llm_engine)
    sampling_params = SamplingParams(max_tokens=2000, logits_processors=[logits_processor], temperature=temperature, top_p=top_p)

    output = args.model.generate(aspect_prompt(claim), sampling_params=sampling_params)[0].outputs[0].text

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
    claim = args.claim
    id2node = []

    # Initialize tree
    root_node = AspectNode(idx=0, name=claim)
    tree = Tree(root_node)
    id2node.append(root_node)

    # Coarse-grained aspect discovery + keyword generation
    aspects = coarse_grained_aspect_discovery(args, claim)

    for aspect in aspects:
        aspect_node = AspectNode(idx=len(id2node), name=aspect["aspect_label"], keywords=aspect["aspect_keywords"])
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
    args = parser.parse_args()

    args.model = LLM(model="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF", tensor_parallel_size=4, max_num_seqs=100, enable_prefix_caching=True)

    main(args)