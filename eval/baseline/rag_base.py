import argparse
import json
import logging
import os
import pickle
import re
import json

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from unidecode import unidecode

from src.api.local.e5_model import E5
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


def build_prompt(claim, literature, height, node_num_per_level):
    """Construct the LLM prompt based on the claim and taxonomy height."""
    return (
        "Claims made by individuals or entities are often nuanced and cannot always be strictly categorized as entirely 'true' or 'false'—"
        "particularly in scientific and political contexts. Instead, a claim can be broken down "
        "into its core aspects and sub-aspects, which are easier to evaluate individually.\n\n"
        f"Given the claim: '{claim}', generate a taxonomy of the claim with a specified height of {height}.\n\n"
	f"Here are some literautre segments to help you generate the taxonomy: {literature}\n"
        f'Generate up to {node_num_per_level} subnodes per node in the taxonomy.\n'
        "The taxonomy should be structured as a dictionary, formatted as follows:\n\n"
        "{\n"
        '    "aspect_name": "{claim}",\n'
        '    "children": [\n'
        '        {\n'
        '            "aspect_name": "Sub-aspect 1",\n'
        '            "children": [\n'
        '                { "aspect_name": "Sub-sub-aspect 1.1",\n'
        '                  "children": [\n'
        '                      { "aspect_name": "Sub-sub-sub-aspect 1.1.1" },\n'
        '                      { "aspect_name": "Sub-sub-sub-aspect 1.1.2" },\n'
        '                      { "aspect_name": "Sub-sub-sub-aspect 1.1.3" }\n'
        '                  ]\n'
        '                },\n'
        '                { "aspect_name": "Sub-sub-aspect 1.2",\n'
        '                  "children": [\n'
        '                      { "aspect_name": "Sub-sub-sub-aspect 1.2.1" },\n'
        '                      { "aspect_name": "Sub-sub-sub-aspect 1.2.2" },\n'
        '                      { "aspect_name": "Sub-sub-sub-aspect 1.2.3" }\n'
        '                  ]\n'
        '                },\n'
        '                { "aspect_name": "Sub-sub-aspect 1.3",\n'
        '                  "children": [\n'
        '                      { "aspect_name": "Sub-sub-sub-aspect 1.3.1" },\n'
        '                      { "aspect_name": "Sub-sub-sub-aspect 1.3.2" },\n'
        '                      { "aspect_name": "Sub-sub-sub-aspect 1.3.3" }\n'
        '                  ]\n'
        '                }\n'
        '            ]\n'
        '        },\n'
        '        {\n'
        '            "aspect_name": "Sub-aspect 2",\n'
        '            "children": [\n'
        '                { "aspect_name": "Sub-sub-aspect 2.1",\n'
        '                  "children": [\n'
        '                      { "aspect_name": "Sub-sub-sub-aspect 2.1.1" },\n'
        '                      { "aspect_name": "Sub-sub-sub-aspect 2.1.2" },\n'
        '                      { "aspect_name": "Sub-sub-sub-aspect 2.1.3" }\n'
        '                  ]\n'
        '                },\n'
        '                { "aspect_name": "Sub-sub-aspect 2.2",\n'
        '                  "children": [\n'
        '                      { "aspect_name": "Sub-sub-sub-aspect 2.2.1" },\n'
        '                      { "aspect_name": "Sub-sub-sub-aspect 2.2.2" },\n'
        '                      { "aspect_name": "Sub-sub-sub-aspect 2.2.3" }\n'
        '                  ]\n'
        '                },\n'
        '                { "aspect_name": "Sub-sub-aspect 2.3",\n'
        '                  "children": [\n'
        '                      { "aspect_name": "Sub-sub-sub-aspect 2.3.1" },\n'
        '                      { "aspect_name": "Sub-sub-sub-aspect 2.3.2" },\n'
        '                      { "aspect_name": "Sub-sub-sub-aspect 2.3.3" }\n'
        '                  ]\n'
        '                }\n'
        '            ]\n'
        '        },\n'
        '        {\n'
        '            "aspect_name": "Sub-aspect 3",\n'
        '            "children": [\n'
        '                { "aspect_name": "Sub-sub-aspect 3.1",\n'
        '                  "children": [\n'
        '                      { "aspect_name": "Sub-sub-sub-aspect 3.1.1" },\n'
        '                      { "aspect_name": "Sub-sub-sub-aspect 3.1.2" },\n'
        '                      { "aspect_name": "Sub-sub-sub-aspect 3.1.3" }\n'
        '                  ]\n'
        '                },\n'
        '                { "aspect_name": "Sub-sub-aspect 3.2",\n'
        '                  "children": [\n'
        '                      { "aspect_name": "Sub-sub-sub-aspect 3.2.1" },\n'
        '                      { "aspect_name": "Sub-sub-sub-aspect 3.2.2" },\n'
        '                      { "aspect_name": "Sub-sub-sub-aspect 3.2.3" }\n'
        '                  ]\n'
        '                },\n'
        '                { "aspect_name": "Sub-sub-aspect 3.3",\n'
        '                  "children": [\n'
        '                      { "aspect_name": "Sub-sub-sub-aspect 3.3.1" },\n'
        '                      { "aspect_name": "Sub-sub-sub-aspect 3.3.2" },\n'
        '                      { "aspect_name": "Sub-sub-sub-aspect 3.3.3" }\n'
        '                  ]\n'
        '                }\n'
        '            ]\n'
        '        }\n'
        '    ]\n'
        "}\n\n"
        'directly output the dict and do not include any additional text in the response.'
    )


def clean_taxonomy(node):
    """
    Recursively remove empty 'children' keys from the taxonomy.

    Args:
        node (dict): A node in the taxonomy tree.

    Returns:
        dict: A cleaned version of the node, without empty children lists.
    """
    if not isinstance(node, dict):
        return node
    node['children'] = [clean_taxonomy(child) for child in node.get('children', []) if child]
    if not node['children']:
        node.pop('children', None)
    return node


def save_taxonomy(output, output_path):
    """
    Save the cleaned taxonomy to a JSON file.

    Args:
        output (dict): The taxonomy data to be saved.
        output_path (str): File path for saving the output JSON.
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

def generate_taxonomy(prompt, model_name):
    """Send the prompt to LLM and get taxonomy response."""
    response = llm_chat([prompt], model_name)
    cleaned_response = response[0]
    
    # Remove possible ```json or other code block markers
    cleaned_response = re.sub(r'^```json\n?|```$', '', cleaned_response, flags=re.MULTILINE)
    result = json.loads(cleaned_response)
    assert 'aspect_name' in result, "Aspect name not found in the response."
    return result


def retry_generate_taxonomy(prompt, model_name, max_retries=16):
    """Retry generating taxonomy if the response is empty."""
    for i in range(max_retries):
        try: 
            return generate_taxonomy(prompt, model_name)
        except Exception as e:
            logging.warning(f"Failed to generate taxonomy. Retrying attempt {i+1}...")
    return None

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
      3. Load or compute embeddings for literature segments.
      4. Select the most relevant segments.
      5. Build and send a prompt to the LLM.
      6. Clean and save the resulting taxonomy.
    """
    parser = argparse.ArgumentParser(description="Generate a taxonomy from a claim using GPT model.")
    parser.add_argument("--model_name", type=str, default="llama-3.1-8b-instruct", help="Name of the LLM model")
    parser.add_argument("--height", type=int, default=3, help="Height of the taxonomy tree")
    parser.add_argument("--output_path", type=str, default="eval/example/rag_base_taxonomy.json", help="Output file path")
    parser.add_argument("--input_path", type=str, default="eval/example/hierarchy.json", help="Input JSON file with claim")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--topic", default="vaccine")
    parser.add_argument("--rag_segment_num", default=10)
    parser.add_argument("--node_num_per_level", default=3)
    parser.add_argument("--claim_id", default=1)
    args = parser.parse_args()

    # Load the claim
    claim = load_claim(args.input_path)
    if not claim:
        logging.error("No valid claim found. Exiting.")
        return

    # Load text segments
    segments = load_segments(data_dir=args.data_dir, topic=args.topic, claim_id=args.claim_id)

    # Define an embedding function
    embedding_func = get_embedding_model()

    # Load or compute segment embeddings
    segment_embeddings = embedding_func(segments)

    # Embed the main claim
    claim_embedding = embedding_func([claim])[claim]

    # Find and select the most relevant segments
    segment_similarity = []
    for segment_str, segment_embed in segment_embeddings.items():
        similarity = cosine_similarity([claim_embedding], [segment_embed])[0][0]
        segment_similarity.append((segment_str, similarity))

    segment_similarity = sorted(segment_similarity, key=lambda x: x[1], reverse=True)
    selected_segments = [seg for seg, sim in segment_similarity[:args.rag_segment_num]]
    literature = "\n".join(selected_segments)

    # Build the prompt and generate the taxonomy
    prompt = build_prompt(claim, args.height, literature, args.node_num_per_level)
    taxonomy = retry_generate_taxonomy(prompt, args.model_name)

    # generate the description for the keywords
    enriched_taxonomy = generate_description(claim, taxonomy, args.model_name)

    save_taxonomy(enriched_taxonomy, args.output_path)

if __name__ == "__main__":
    main()
