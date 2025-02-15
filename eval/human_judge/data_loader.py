import os
import json
from typing import Dict, Any, List
import random

from eval.human_judge.utils import TOPIC_LIST

random.seed(42)

def load_json(file_path: str, json_prefix: str) -> Dict[str, List[Any]]:
    """
    Load JSON objects from topic-specific directories.

    :param file_path: Base directory containing topic folders.
    :param json_prefix: Relative path to the JSON file within each directory.
    :return: A dictionary with topic as key and list of loaded JSON objects as value.
    """
    results = {}
    # Build file paths for each topic
    topic_specific_file_paths = [os.path.join(file_path, topic) for topic in TOPIC_LIST]
    for topic, topic_file_path in zip(TOPIC_LIST, topic_specific_file_paths):
        # List all directories under the topic folder
        directories = [os.path.join(topic_file_path, directory) for directory in os.listdir(topic_file_path)]
        # Collect JSON files if they exist
        json_files = [os.path.join(directory, json_prefix) for directory in directories if os.path.exists(os.path.join(directory, json_prefix))]
        # Load JSON objects from each file
        loaded_json_objects = [json.load(open(json_file)) for json_file in json_files]
        results[topic] = loaded_json_objects
    return results

def load_node(json_objects: Dict[str, List[Any]], flatten_func) -> Dict[str, List[dict]]:
    """
    Process loaded JSON objects into a flat list of node dictionaries for each topic.

    :param json_objects: Dictionary mapping topic to list of JSON objects.
    :param flatten_func: Function used to flatten the JSON tree structure.
    :return: Dictionary with topic as key and list of node dictionaries as value.
    """
    nodes = {}
    for topic, json_object_list in json_objects.items():
        nodes[topic] = []
        for json_object in json_object_list:
            nodes[topic].extend(flatten_func(json_object))
    return nodes

def load_data(data_path: str, json_prefix: str, segment_per_perspective: int, sample_per_split: int, flatten_func, sample_func, organize_func) -> dict:
    """
    Complete pipeline to load, process, sample, and organize data.

    :param data_path: Base directory where data is stored.
    :param json_prefix: Relative path to the JSON file in each directory.
    :param segment_per_perspective: Number of segments to sample per perspective.
    :param sample_per_split: Number of nodes to sample per topic.
    :param flatten_func: Function to flatten JSON trees.
    :param sample_func: Function to sample nodes.
    :param organize_func: Function to organize sampled nodes.
    :return: Processed data organized by topic.
    """
    print(f"Loading data from {data_path}")
    json_objects = load_json(data_path, json_prefix)
    nodes = load_node(json_objects, flatten_func)
    sampled_nodes = sample_func(nodes, segment_per_perspective, sample_per_split)
    organized_nodes = organize_func(sampled_nodes, segment_per_perspective)
    return organized_nodes
