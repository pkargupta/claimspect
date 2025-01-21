import os
import json
import time

from classify.core_class_anno.core_class_annotation import run_annotation

def find_node_by_name(tree, node_name):
    """
    Finds a node in the tree by its name.

    Args:
        tree (Tree): The tree object to search in.
        node_name (str): The name of the node to find.

    Returns:
        AspectNode: The node with the specified name, or None if not found.
    """
    def traverse(node):
        if node.name == node_name:
            return node
        for sub_aspect in node.sub_aspects:
            result = traverse(sub_aspect)
            if result:
                return result
        return None

    return traverse(tree.root)

# Iterate through the original data
def group_by_ancestor(data):
    result = {}
    for corpus_id, details in data.items():
        ancestors = details['with ancestors']
        for ancestor_id in ancestors:
            if ancestor_id not in result:
                result[ancestor_id] = []
            result[ancestor_id].append(corpus_id)

    return result

def convert_tree_to_format(tree):
    """
    Converts a tree hierarchy into the specified format.
    
    Args:
        tree (Tree): The tree object representing the hierarchy.

    Returns:
        tuple: A tuple containing:
            - idx_to_name (dict): A dictionary mapping indices to names.
            - hierarchy (list): A list of parent-child index relationships.
    """
    idx_to_name = {}
    hierarchy = []

    def traverse(node, idx_counter):
        """
        Traverses the tree recursively to populate idx_to_name and hierarchy.

        Args:
            node (AspectNode): The current node being traversed.
            idx_counter (list): A single-element list to maintain the current index across recursive calls.

        Returns:
            int: The index of the current node.
        """
        # Assign current node an index
        current_idx = idx_counter[0]
        idx_to_name[current_idx] = node.name
        idx_counter[0] += 1

        # Process sub-aspects
        for sub_aspect in node.sub_aspects:
            child_idx = traverse(sub_aspect, idx_counter)
            hierarchy.append((current_idx, child_idx))

        return current_idx

    # Start traversal from the root
    traverse(tree.root, [0])

    return idx_to_name, hierarchy

def get_keywords_dict(tree):
    """
    Traverses the tree and generates a dictionary mapping each aspect's name to its keywords list.

    Args:
        tree (Tree): The tree object representing the hierarchy.

    Returns:
        dict: A dictionary where keys are aspect names and values are lists of keywords.
    """
    keywords_dict = {}

    def traverse(node):
        """
        Recursively traverses the tree to populate the keywords dictionary.

        Args:
            node (AspectNode): The current node being traversed.
        """
        # Add the node's name and its keywords to the dictionary
        keywords_dict[node.name] = node.keywords if node.keywords else []

        # Traverse sub-aspects
        for sub_aspect in node.sub_aspects:
            traverse(sub_aspect)

    # Start traversal from the root
    traverse(tree.root)

    return keywords_dict

def write_label_hierarchy_file(tree, cache_dir):
    """Write the label file."""
    idx_to_name, hierarchy = convert_tree_to_format(tree)
    time_stamp = int(time.time()*1000)
    label_file_path = os.path.join(cache_dir, f'label_file_{time_stamp}.txt')
    label_hierarchy_path = os.path.join(cache_dir, f'label_hierarchy_file_{time_stamp}.txt')
    
    with open(label_file_path, 'w') as f:
        for idx, name in idx_to_name.items():
            f.write(f'{idx}\t{name.replace(' ', '_')}\n')
    with open(label_hierarchy_path, 'w') as f:
        for parent_idx, child_idx in hierarchy:
            f.write(f'{parent_idx}\t{child_idx}\n')
    
    return label_file_path, label_hierarchy_path, idx_to_name
    
def write_corpus_file(tree, cache_dir):
    """Write the corpus file."""
    segments = tree.root.get_all_segments()[:3]  # TODO: debug use
    segment_str_list = [seg.content for seg in segments]
    time_stamp = int(time.time()*1000)
    corpus_path = os.path.join(cache_dir, f'corpus_file_{time_stamp}.txt')
    
    with open(corpus_path, 'w') as f:
        for idx, segment_str in enumerate(segment_str_list):
            f.write(f'{idx}\t{segment_str}\n')
    
    idx2str = {idx: segment_str for idx, segment_str in enumerate(segment_str_list)}
    return corpus_path, idx2str
    
def write_enrich_file(tree, cache_dir):
    """Write the enrichment file."""
    keyword_dict = get_keywords_dict(tree)
    time_stamp = int(time.time()*1000)
    enrichment_path = os.path.join(cache_dir, f'enrichment_file_{time_stamp}.txt')
    with open(enrichment_path, 'w') as f:
        for aspect, keywords in keyword_dict.items():
            f.write(f'{aspect.replace(' ', '_')}:{",".join(keywords)}\n')
    return enrichment_path
    
def get_output_path(cache_dir: str):
    """Get the output path."""
    time_stamp = int(time.time()*1000)
    return os.path.join(cache_dir, f'output_{time_stamp}.json')
    
def load_results(output_path):
    """Load the results."""
    with open(output_path, 'r') as f:
        results = json.load(f)
    return results
    
def convert_results_to_tree(tree, results, node_idx2node_name, corpus_idx2corpus_str):
    
    """Convert the results to a tree."""
    node_idx2corpus_idx = group_by_ancestor(results)
    # deep copy the original paper info
    original_paper_info = tree.root.related_papers.items()
    
    for node_idx, corpus_idx_list in node_idx2corpus_idx.items():
        
        if node_idx == '0': continue  # skip the root
        breakpoint()
        
        node_name = node_idx2node_name[int(node_idx)]
        corpus_str_list = [corpus_idx2corpus_str[int(corpus_idx)] for corpus_idx in corpus_idx_list]
        
        node = find_node_by_name(tree, node_name)
        if not node:
            raise ValueError(f"Node with name '{node_name}' not found in the tree.")

        classified_papers_dict = {}
        for paper_id, paper in original_paper_info:
            original_segments = paper["relevant_segments"]
            classified_segments = []
            for segment in original_segments:
                if segment.content in corpus_str_list:
                    classified_segments.append(segment)
                    
            if len(classified_segments) == 0:
                continue
            
            paper["relevant_segments"] = classified_segments
            classified_papers_dict[paper_id] = paper
        
        node.related_papers = classified_papers_dict
        breakpoint()
        # assert [seg.content for seg in node.get_all_segments()] == corpus_str_list
        print([seg.content for seg in node.get_all_segments()] == corpus_str_list)
    
def get_cache_dir(
    cache_dir: str = '.cache',
):
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def hierarchical_segment_classification(claim, tree):
    
    """Step 5: Classify segments into the aspect hierarchy."""
    
    # ensure cache dir
    cache_dir = get_cache_dir()
    
    label_file_path, label_hierarchy_path, node_idx2node_name = write_label_hierarchy_file(tree, cache_dir)
    corpus_path, corpus_idx2corpus_str= write_corpus_file(tree, cache_dir)
    enrichment_path = write_enrich_file(tree, cache_dir)
    output_path = get_output_path(cache_dir)

    # run annotation
    run_annotation(claim, enrichment_path, corpus_path, label_file_path, label_hierarchy_path, output_path)
    results = load_results(output_path)
    convert_results_to_tree(tree, results, node_idx2node_name, corpus_idx2corpus_str)
    