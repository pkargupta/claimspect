import json
import argparse
from eval.llm.io import llm_chat

def get_json_files(args):
    with open(args.eval_json_path, 'r') as f:
        method_json = json.load(f)
    return method_json

def get_claim(eval_json):
    return eval_json['aspect_name']

def get_taxonomy(taxonomy_json, level=0):
    expression = "  " * level + f"- {taxonomy_json['aspect_name']}\n"

    if 'children' in taxonomy_json:
        for child in taxonomy_json['children']:
            expression += present_taxonomy(child, level + 1)
    return expression

def get_paths(node, current_path=None):
    """
    Recursively traverses the tree and returns a list of paths for each leaf node.
    Each path is a string of aspect names separated by ' - '.
    
    :param node: A dictionary representing the current node in the tree.
    :param current_path: A list of aspect names collected so far along the path.
    :return: A list of path strings from the root to each leaf node.
    """
    if current_path is None:
        current_path = []

    # Append current node's aspect name to the current path
    new_path = current_path + [node["aspect_name"]]

    # If the node does not have children, it's a leaf node.
    if "children" not in node or not node["children"]:
        # Return the path as a single string
        return [" - ".join(new_path)]
    
    # Otherwise, traverse each child and collect all paths.
    paths = []
    for child in node["children"]:
        paths.extend(get_paths(child, new_path))
    return paths
    
def get_levels(node):
    """
    Recursively traverse the tree and collect lists of sibling 'aspect_name' values
    for each occurrence of a 'children' key.
    
    Parameters:
        node (dict or list): The current node in the tree.
        
    Returns:
        list: A list of lists, where each inner list contains the 'aspect_name'
              values of sibling nodes found under a 'children' key.
    """
    sibling_lists = []
    
    # Check if the current node is a dictionary.
    if isinstance(node, dict):
        # If the node has a 'children' key and it is a list, process its children.
        if 'children' in node and isinstance(node['children'], list):
            # Extract the 'aspect_name' for each child in the current children list.
            siblings = [child.get('aspect_name') for child in node['children'] if 'aspect_name' in child]
            sibling_lists.append(siblings)
            
            # Recursively process each child node.
            for child in node['children']:
                sibling_lists.extend(get_levels(child))
        else:
            # If the current dict does not have a 'children' key, check its values.
            for value in node.values():
                if isinstance(value, (dict, list)):
                    sibling_lists.extend(get_levels(value))
    
    # If the current node is a list, process each element.
    elif isinstance(node, list):
        for item in node:
            sibling_lists.extend(get_levels(item))
    
    return sibling_lists

def get_all_nodes(tree):

    nodes = []
    def traverse(node):
        nodes.append(node)
        if "children" in node:
            for child in node["children"]:
                traverse(child)
    traverse(tree)
    return nodes

def node_name2segments(data: dict) -> dict:
    """
    Given a dictionary with an aspect and its associated perspectives and mapped segments,
    this function collects all the valid segment indices from the various perspectives.
    If the number of collected indices is equal to the length of mapped_segs,
    the function returns mapped_segs directly. Otherwise, it uses the collected indices
    to filter the mapped_segs list.

    Parameters:
        data (dict): A dictionary containing:
            - "aspect_name" (str): The aspect name.
            - "mapped_segs" (list of str): A list of segment strings.
            - "perspectives" (dict): A dictionary where each key (e.g., "supports_claim",
              "neutral_to_claim", "opposes_claim", "irrelevant_to_claim") has a nested dictionary
              with a key "perspective_segments" (a list of indices).

    Returns:
        dict: A dictionary with a single key-value pair where the key is the aspect_name and
              the value is a list of segment strings selected from mapped_segs.
    """
    # Extract the aspect name and the list of mapped segments
    aspect_name = data.get("aspect_name", "")
    mapped_segs = data.get("mapped_segs", [])

    # Initialize a list to collect all perspective segment indices
    collected_indices = []

    perspectives = data.get("perspectives", {})
    for key, value in perspectives.items():
        # Only process entries that are dictionaries and have a "perspective_segments" key
        if isinstance(value, dict) and "perspective_segments" in value:
            segments = value.get("perspective_segments", [])
            if isinstance(segments, list):
                collected_indices.extend(segments)

    # Remove duplicate indices and sort them in ascending order
    unique_indices = sorted(set(collected_indices))

    # If the number of collected indices equals the length of mapped_segs,
    # return mapped_segs directly.
    if len(unique_indices) == len(mapped_segs):
        selected_segments = mapped_segs
    else:
        # Otherwise, filter mapped_segs using the collected indices.
        # Only include indices that are within the range of mapped_segs.
        print(f"In aspect {aspect_name}, only {len(unique_indices)} out of {len(mapped_segs)} segments are selected.")
        selected_segments = [mapped_segs[i] for i in unique_indices if 0 <= i < len(mapped_segs)]

    return {aspect_name: selected_segments}

def analyze_json(eval_json):
    
    claim = get_claim(eval_json)
    print("Get 1 claim.")
    
    taxonomy = get_taxonomy(eval_json)
    print("Get 1 taxonomy.")
    
    paths = get_paths(eval_json)
    # remove the claim
    for i in range(len(paths)):
        paths[i] = paths[i].replace(claim + " - ", "")
    print(f"Get {len(paths)} path.")
    
    levels = get_levels(eval_json)
    print(f"Get {len(levels)} levels.")

    raw_nodes = get_all_nodes(eval_json)
    print(f"Get {len(raw_nodes)} nodes.")
    
    nodes = [node_name2segments(node) for node in raw_nodes]
    
    return claim, taxonomy, paths, levels, nodes

def get_prompt(claim, taxo_str1, taxo_str2) -> str:
    """Construct the LLM prompt based on the claim and taxonomy height."""
    return (
        "Claims made by individuals or entities are often nuanced and cannot always be strictly categorized as entirely 'true' or 'false'. "
        "Particularly in scientific and political contexts. Instead, a claim can be broken down "
        "into its core aspects and sub-aspects, which are easier to evaluate individually.\n\n"
        f"Given the claim: '{claim}', decide which of the following taxonomies are better:\n\n"
        "taxonomy 1:\n"
        f"{taxo_str1}\n\n"
        "taxonomy 2:\n"
        f"{taxo_str2}\n\n"
        "Choose the taxonomy that is more accurate and informative. If both taxonomies are equally informative, choose 'tie'. "
        "Output options: 'taxonomy 1 wins', 'taxonomy 2 wins', or 'tie'. Do some simple rationalization if possible."
    )

def present_taxonomy(taxonomy_json, level=0):
    expression = "  " * level + f"- {taxonomy_json['aspect_name']}\n"

    if 'children' in taxonomy_json:
        for child in taxonomy_json['children']:
            expression += present_taxonomy(child, level + 1)

    return expression

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_json_path", type=str, default="eval/example/hierarchy.json")
    parser.add_argument("--output_path", type=str, default="eval/example/node_llm_judge.json")
    parser.add_argument("--model_judge_name", type=str, default="gpt-4o", help="Name of the LLM model")
    args = parser.parse_args()

    # Load JSON files
    eval_json = get_json_files(args)
    claim, taxonomy, paths, levels, nodes = analyze_json(eval_json)
    
    


if __name__ == "__main__":
    main()
