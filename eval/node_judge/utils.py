# utils.py
import json

def get_json_files(eval_json_path: str) -> dict:
    """Load and return the JSON content from the specified file."""
    with open(eval_json_path, 'r') as f:
        return json.load(f)

# taxonomy.py

def get_claim(eval_json: dict) -> str:
    return eval_json['aspect_name']

def process_name(node_name:str, name2note=None) -> str:
    if name2note is None:
        return node_name
    if node_name in name2note:
        if name2note[node_name]:
            return f"{node_name} ({name2note[node_name]})"
        else:
            return node_name

def get_taxonomy(taxonomy_json: dict, level: int = 0, name2note=None) -> str:
    
    expression = "  " * level + f"- {process_name(taxonomy_json['aspect_name'], name2note)}\n"
    if 'children' in taxonomy_json:
        for child in taxonomy_json['children']:
            expression += get_taxonomy(child, level + 1, name2note)
    return expression

def present_taxonomy(taxonomy_json: dict, level: int = 0) -> str:
    """Recursively returns a string presentation of the taxonomy tree."""
    expression = "  " * level + f"- {taxonomy_json['aspect_name']}\n"
    if 'children' in taxonomy_json:
        for child in taxonomy_json['children']:
            expression += present_taxonomy(child, level + 1)
    return expression

def get_paths(node: dict, current_path: list = None, name2note=None) -> list:
    """
    Recursively traverse the tree and return a list of paths (strings) for each leaf node.
    """
    if current_path is None:
        current_path = []
    new_path = current_path + [process_name(node["aspect_name"], name2note)]
    if "children" not in node or not node["children"]:
        return [" -> ".join(new_path)]
    paths = []
    for child in node["children"]:
        paths.extend(get_paths(child, new_path, name2note))
    return paths

def get_levels(node, name2note=None) -> list:
    """
    Recursively traverse the tree and collect a list of dictionaries, each containing:
      - 'parent': the parent node’s aspect_name
      - 'siblings': a list of aspect_names of all children of that parent.
    """
    result = []
    if isinstance(node, dict):
        if 'children' in node and isinstance(node['children'], list):
            parent_name = node.get('aspect_name')
            siblings = [process_name(child.get('aspect_name'), name2note) for child in node['children'] if 'aspect_name' in child]
            result.append({"parent": process_name(parent_name, name2note), "siblings": siblings})
            for child in node['children']:
                result.extend(get_levels(child, name2note))
        else:
            for value in node.values():
                if isinstance(value, (dict, list)):
                    result.extend(get_levels(value, name2note))
    elif isinstance(node, list):
        for item in node:
            result.extend(get_levels(item, name2note))
    return result

def get_all_nodes(tree: dict) -> list:
    """Return a list of all nodes in the taxonomy tree."""
    nodes = []
    def traverse(node):
        nodes.append(node)
        if "children" in node:
            for child in node["children"]:
                traverse(child)
    traverse(tree)
    return nodes

def node_name2segments(data: dict, name2note, top_k=5) -> dict:
    """
    Given a node dictionary with an aspect and its mapped segments,
    return a dictionary with the aspect_name and the list of selected segments.
    """

    aspect_name = data.get("aspect_name", "")
    mapped_segs = data.get("top_10_segments", [])[:top_k]
    if not mapped_segs:
        mapped_segs = data.get("mapped_segs", [])[:top_k]  # Only for root node
    
    return {"aspect_name": process_name(aspect_name, name2note), "segments": mapped_segs}

def get_node_name_2_description(data: dict) -> dict:
    """
    Given a taxonomy tree, return a dictionary mapping each node's aspect_name to its aspect_description.
    """
    node_descriptions = {}
    
    def traverse(node):
        if "aspect_name" in node and "aspect_description" in node:
            node_descriptions[node["aspect_name"]] = node["aspect_description"]
        if "children" in node:
            for child in node["children"]:
                traverse(child)
    
    traverse(data)
    return node_descriptions
    