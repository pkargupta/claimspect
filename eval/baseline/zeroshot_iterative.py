import argparse
import json
import logging
import os
import pickle

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from unidecode import unidecode

from api.local.e5_model import E5
from eval.llm.io import llm_chat

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def get_embedding_model(model_name: str = 'e5'):
    """
    Return an embedding function for the specified model name.
    Currently only supports the 'e5' model.
    """
    if model_name == 'e5':
        e5 = E5()

        def embed(text: list[str]):
            return e5(text)

        return embed
    else:
        raise ValueError(f"Model {model_name} not supported.")


def load_claim(json_path):
    """
    Load the claim from a JSON file.
    
    Args:
        json_path (str): Path to the JSON file containing the claim.

    Returns:
        str or None: The 'aspect_name' value if present, otherwise None.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data.get('aspect_name', None)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading claim from {json_path}: {e}")
        return None


def build_prompt(claim, max_aspects_per_node=5):
    """
    Construct the LLM prompt based on the claim and taxonomy height.
    
    Args:
        claim (str): The current claim.
        literature (str): The top relevant literature segments.
        max_aspects_per_node (int): The maximum number of aspects to generate.

    Returns:
        str: The prompt to be passed to the language model.
    """
    return (
        "Claims made by individuals or entities are often nuanced and cannot always be strictly categorized "
        "as entirely 'true' or 'false'—particularly in scientific and political contexts. Instead, a claim can "
        "be broken down into its core aspects, which are easier to evaluate individually.\n\n"
        f"Given the claim: '{claim}', generate a list of up to {max_aspects_per_node} aspects (short phrases) of the claim. "
        "They should be the node of the same level\n\n"
        "The aspects should be structured as a list, formatted as follows:\n"
        '["aspect 1", "aspect 2", "aspect 3"]\n'
        "directly output the list and do not include any additional text in the response."
    )


def clean_taxonomy(node):
    """
    Recursively remove empty 'children' keys from the taxonomy.
    
    Args:
        node (dict): A node in the taxonomy tree.

    Returns:
        dict: A cleaned version of the node with no empty children lists.
    """
    if not isinstance(node, dict):
        return node
    node['children'] = [clean_taxonomy(child) for child in node.get('children', []) if child]
    if not node['children']:
        node.pop('children', None)
    return node


def save_taxonomy(output, output_path):
    """
    Save the cleaned taxonomy to a file.
    
    Args:
        output (dict): The taxonomy data to be saved.
        output_path (str): The path where the taxonomy will be saved.
    """
    try:
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=4)
        logging.info(f"Taxonomy saved successfully at {output_path}")
    except IOError as e:
        logging.error(f"Error saving taxonomy to {output_path}: {e}")


def load_segments(data_dir, topic, claim_id):
    """
    Load text segments from a topic-specific text file, chunked by a specified size.
    """
    json_path = os.path.join(data_dir, topic, f"segments.json")
    with open(json_path, 'r') as f:
        data = json.load(f)
    segment_list = data[str(claim_id)]
    segment_str_list = [seg['segment'] for seg in segment_list]
    return segment_str_list


def iterative_zeroshot(args, current_claim, original_claim, current_height=0):
    """
    Recursively build a taxonomy for a claim by splitting it into aspects at each level,
    guided by the most relevant literature segments.

    Args:
        args: Parsed command-line arguments.
        current_claim (str): The current claim or sub-claim being processed.
        original_claim (str): The main/initial claim.
        segment_embeddings (dict): Precomputed embeddings for literature segments.
        embedding_func (callable): Function to produce embeddings for a given text.
        current_height (int): Current depth in the taxonomy tree.

    Returns:
        dict: A dictionary representing the node at this level and its children.
    """
    print("Current claim: ", current_claim)
    print("Current height: ", current_height)

    # Determine the aspect name for the node
    if current_claim == original_claim:
        aspect_name = current_claim
    else:
        aspect_name = current_claim.split("With regard to ")[1].split(", ")[0]

    # Base case: if we have reached the maximum height
    if current_height == args.height:
        return {"aspect_name": aspect_name}
    
    # Build the prompt and call the LLM
    input_prompt = build_prompt(current_claim, args.max_aspects_per_node)
    response = llm_chat([input_prompt], model_name=args.model_name)[0]

    # Attempt to parse the LLM response as JSON
    try:
        if '```json' in response:
            response = response.strip()[7:-3]
        aspects = json.loads(response)
    except Exception as e:
        logging.error(f"Error decoding JSON response: {e}")
        raise e

    # Recursively process each aspect
    children = [
        iterative_zeroshot(
            args,
            f"With regard to {aspect}, {original_claim}",
            original_claim,
            current_height + 1
        )
        for aspect in aspects
    ]

    return {"aspect_name": aspect_name, "children": children}

def generate_description(claim: str, taxonomy: dict, model_name: str):
    """
    Generate descriptions for the taxonomy nodes using the LLM model in batched mode.
    """
    
    def collect_node_names(node: dict, collected_nodes: list):
        """
        Recursively collect all node names from the taxonomy.
        """
        collected_nodes.append(node["aspect_name"])
        if "children" in node:
            for child in node["children"]:
                collect_node_names(child, collected_nodes)
    
    # Collect all node names
    node_names = []
    collect_node_names(taxonomy, node_names)
    
    # Generate batched descriptions
    prompts = [
        f"Based on the claim: '{claim}', provide one-sentence for what does this aspect name mean: '{node_name}'. Do not include any additional comments."
        for node_name in node_names
    ]
    
    responses = llm_chat(prompts, model_name)  # Call LLM once with batched prompts
    
    def assign_descriptions(node: dict, response_iterator):
        """
        Recursively assign descriptions to the taxonomy nodes.
        """
        node["aspect_description"] = next(response_iterator)
        if "children" in node:
            for child in node["children"]:
                assign_descriptions(child, response_iterator)
    
    # Assign descriptions using an iterator over the responses
    response_iterator = iter(responses)
    assign_descriptions(taxonomy, response_iterator)
    
    return taxonomy

def main():
    """
    Main function to handle command-line execution:
      1. Parse arguments.
      2. Load the claim.
      3. Load or create and cache embeddings for literature segments.
      4. Generate a taxonomy from the claim using a recursive approach.
      5. Clean and save the final taxonomy.
    """
    parser = argparse.ArgumentParser(description="Generate a taxonomy from a claim using GPT model.")
    parser.add_argument("--model_name", type=str, default="llama-3.1-8b-instruct", help="Name of the LLM model")
    parser.add_argument("--height", type=int, default=3, help="Height of the taxonomy tree")
    parser.add_argument("--output_path", type=str, default="eval/example/zeroshot_iterative_taxonomy.json", help="Output file path")
    parser.add_argument("--input_path", type=str, default="eval/example/hierarchy.json", help="Input JSON file with claim")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--topic", default="vaccine")
    parser.add_argument("--max_aspects_per_node", default=3)
    parser.add_argument("--claim_id", default=1)
    args = parser.parse_args()

    # Load the claim
    claim = load_claim(args.input_path)
    if not claim:
        logging.error("No valid claim found. Exiting.")
        return
    
    # Generate the taxonomy
    taxonomy = iterative_zeroshot(args, claim, claim)

    # Clean and save the taxonomy
    cleaned_taxonomy = clean_taxonomy(taxonomy)
    
    # generate the description for the keywords
    enriched_taxonomy = generate_description(claim, cleaned_taxonomy, args.model_name)
    
    save_taxonomy(enriched_taxonomy, args.output_path)


if __name__ == "__main__":
    main()
