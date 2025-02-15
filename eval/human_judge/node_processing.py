import random

def flatten_tree(tree: dict) -> list:
    """
    Flatten a hierarchical tree structure into a list of nodes.
    
    Each node is represented as a dictionary containing:
      - aspect_name
      - aspect_description
      - perspectives
      - mapped_segs

    :param tree: Hierarchical JSON tree.
    :return: List of flattened node dictionaries.
    """
    flat_list = []

    def traverse(node: dict):
        flat_list.append({
            "aspect_name": node["aspect_name"],
            "aspect_description": node.get("aspect_description", ""),
            "perspectives": node.get("perspectives", {}),
            "mapped_segs": node.get("mapped_segs", []),
            "children": node.get("children", [])
        })
        for child in node.get("children", []):
            traverse(child)

    traverse(tree)
    return flat_list

def check_qualification(node: dict, segment_per_perspective: int) -> bool:
    """
    Check if a node qualifies based on its description, perspectives, and segment counts.
    
    :param node: A node dictionary.
    :param segment_per_perspective: Required minimum number of segments per perspective.
    :return: True if the node qualifies, False otherwise.
    """
    if not node["aspect_description"]:
        return False
    if not node["perspectives"]:
        return False
    if not node["mapped_segs"]:
        return False

    perspectives = node["perspectives"]
    supports = perspectives.get('supports_claim', {}).get('perspective_segments', [])
    neutral = perspectives.get('neutral_to_claim', {}).get('perspective_segments', [])
    opposes = perspectives.get('opposes_claim', {}).get('perspective_segments', [])

    if len(supports) < segment_per_perspective or len(neutral) < segment_per_perspective or len(opposes) < segment_per_perspective:
        return False

    return True

def sample_nodes(node_objects: dict, segment_per_perspective: int, sample_per_split: int) -> dict:
    """
    Sample a fixed number of qualified nodes per topic.
    
    :param node_objects: Dictionary with topics as keys and list of nodes as value.
    :param segment_per_perspective: Minimum segments required per perspective.
    :param sample_per_split: Number of nodes to sample for evaluation.
    :return: Dictionary with topics as keys and list of sampled nodes as value.
    """
    results = {}
    for topic, nodes in node_objects.items():
        qualified_nodes = [node for node in nodes if check_qualification(node, segment_per_perspective)]
        # Randomly sample nodes
        sampled_nodes = random.sample(qualified_nodes, sample_per_split)
        results[topic] = sampled_nodes
        print(f"Sampled {len(sampled_nodes)} nodes from {topic}")
    return results

def organize_nodes(sampled_nodes: dict, segment_per_perspective: int) -> dict:
    """
    For each node, randomly sample segments from each perspective and organize them.
    
    :param sampled_nodes: Dictionary with topics as keys and list of sampled nodes as value.
    :param segment_per_perspective: Number of segments to sample per perspective.
    :return: Dictionary with topics as keys and organized node data.
    """
    results = {}
    for topic, nodes in sampled_nodes.items():
        results[topic] = []
        for node in nodes:
            aspect_name = node["aspect_name"]
            aspect_description = node["aspect_description"]
            perspectives = node["perspectives"]

            # Randomly sample indices for segments for each perspective
            supports_idx = random.sample(perspectives["supports_claim"]["perspective_segments"], segment_per_perspective)
            neutral_idx = random.sample(perspectives["neutral_to_claim"]["perspective_segments"], segment_per_perspective)
            opposes_idx = random.sample(perspectives["opposes_claim"]["perspective_segments"], segment_per_perspective)

            # Map indices to actual segments
            supports_segments = [node["mapped_segs"][idx] for idx in supports_idx]
            neutral_segments = [node["mapped_segs"][idx] for idx in neutral_idx]
            opposes_segments = [node["mapped_segs"][idx] for idx in opposes_idx]

            # Prepare segments with an associated LLM label
            sampled_segments = (
                [{"segment": seg, "llm_label": "support"} for seg in supports_segments] +
                [{"segment": seg, "llm_label": "neutral"} for seg in neutral_segments] +
                [{"segment": seg, "llm_label": "oppose"} for seg in opposes_segments]
            )
            random.shuffle(sampled_segments)
            results[topic].append({
                "aspect_name": aspect_name,
                "aspect_description": aspect_description,
                "segments": sampled_segments
            })
    return results
