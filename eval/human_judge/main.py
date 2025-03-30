import argparse
import os
import json
import random
from enum import Enum
from collections import defaultdict

TOPIC_LIST = ['dtra', 'vaccine']

random.seed(42)

def load_json(file_path, json_prefix):
    results = {}
    topic_specfici_file_paths = [os.path.join(file_path, topic) for topic in TOPIC_LIST]
    for topic, topic_file_path in zip(TOPIC_LIST, topic_specfici_file_paths):
        # find all the directories in the topic_file_path
        directories = [os.path.join(topic_file_path, directory) for directory in os.listdir(topic_file_path)]
        json_files = [os.path.join(directory, json_prefix) for directory in directories if os.path.exists(os.path.join(directory, json_prefix))]
        loaded_json_object = [json.load(open(json_file)) for json_file in json_files]
        results[topic] = loaded_json_object
    return results
    
def flatten_tree(tree):
    """
    Flattens a hierarchical tree structure into a list of dictionaries containing only aspect_name and aspect_description.
    """
    
    flat_list = []
    
    def traverse(node):
        flat_list.append({
            "aspect_name": node["aspect_name"],
            "aspect_description": node.get("aspect_description", ""),
            "perspectives": node.get("perspectives", {}),
            "mapped_segs": node.get("mapped_segs", [])
        })
        
        for child in node.get("children", []):
            traverse(child)
    
    traverse(tree)
    
    return flat_list
    

def load_node(json_objects):
    
    nodes = {}
    for topic, json_object_list in json_objects.items():
        nodes[topic] = []
        for json_object in json_object_list:
            nodes[topic].extend(flatten_tree(json_object))
    return nodes

def check_qualification(node, segment_per_perspective):
    
    if not node["aspect_description"]:
        return False

    if not node["perspectives"]:
        return False
    
    if not node["mapped_segs"]:
        return False
    
    perspective = node["perspectives"]
    supports_num = len(perspective['supports_claim']['perspective_segments'])
    neutral_num = len(perspective['neutral_to_claim']['perspective_segments'])
    opposes_num = len(perspective['opposes_claim']['perspective_segments'])
    
    if supports_num < segment_per_perspective or neutral_num < segment_per_perspective or opposes_num < segment_per_perspective:
        return False

    return True    

def sample_nodes(node_objects, segment_per_perspective, sample_per_split):
    results = {}
    for topic, nodes in node_objects.items():
        qualified_nodes = [node for node in nodes if check_qualification(node, segment_per_perspective)]
        sampled_nodes = random.sample(qualified_nodes, min(sample_per_split, len(qualified_nodes)))
        results[topic] = sampled_nodes
        print(f"Sampled {len(sampled_nodes)} nodes from {topic}")
    return results

def organize_nodes(sampled_nodes, segment_per_perspective):
    
    results = {}
    for topic, nodes in sampled_nodes.items():
        results[topic] = []
        for node in nodes:
            
            aspect_name = node["aspect_name"]
            aspect_description = node["aspect_description"]
            # randomly pick one perspective from support claim, neutral to claim, and opposes claim
            perspective = {0: 'supports_claim', 1: 'neutral_to_claim', 2: 'opposes_claim'}[random.randint(0, 2)]
            perspective_description = node["perspectives"][perspective]["perspective_description"]
            perspective_segments_idx = random.sample(node["perspectives"][perspective]["perspective_segments"], segment_per_perspective)
            perspective_segments = [node["mapped_segs"][idx] for idx in perspective_segments_idx]
            
            results[topic].append({
                "aspect_name": aspect_name,
                "aspect_description": aspect_description,
                "perspective": perspective,
                "perspective_description": perspective_description,
                "segments": perspective_segments
            })
    return results

def load_data(data_path, json_prefix, segment_per_perspective, sample_per_split):
    
    # Logic to load data
    print(f"Loading data from {data_path}")
    # load all the json files from the data_path
    json_objects = load_json(data_path, json_prefix)
    # load json into nodes
    node_objects = load_node(json_objects)
    
    # sample nodes of complete data
    sampled_node_objects = sample_nodes(node_objects, segment_per_perspective, sample_per_split)
    # oragnize the sampled nodes
    organized_nodes =  organize_nodes(sampled_node_objects, segment_per_perspective)
    return organized_nodes

def robust_input(prompt: str, valid_responses: list[str]):
    response = input(prompt+'\n')
    while response not in valid_responses:
        print(f"Invalid response. Please choose from {valid_responses}")
        response = input(prompt)
    return response

def get_user_id(user_num):
    return robust_input(f"If you are Priyanka, input 0. If you are Runchu, input 1.", [str(i) for i in range(user_num)])
    

def user_interface(processed_data, user_num, results_path):
    
    user_id = get_user_id(user_num)

    for topic, nodes in processed_data.items():
        topic_results_path = os.path.join(results_path, topic)
        
        if not os.path.exists(topic_results_path):
            os.makedirs(topic_results_path)
        
        print(f"Evaluating {topic}")

        for idx, node in enumerate(nodes):
            
            # skip according to user_id
            if idx % user_num != int(user_id):
                continue
            
            print(f"Aspect {idx+1}/{len(nodes)}")
            aspect_save_path = os.path.join(topic_results_path, f"aspect_{idx}.json")
            if os.path.exists(aspect_save_path):
                print(f"Aspect {idx} already evaluated. Skipping.")
                continue
            
            print("############################################")
            print(f"Aspect Name: {node['aspect_name']}")
            print('############################################')
            print(f"Aspect Description: {node['aspect_description']}")
            print('############################################')
            print("Perspective: ", node["perspective"])
            print('############################################')
            print(f"Perspective Description: {node['perspective_description']}")
            for j, seg in enumerate(node["segments"]):
                print(f"Segment {j+1}: {seg}")
                print()
            print('############################################')
            user_feedback = robust_input("Is there any segments providing background knowledge to support the perspective? 0=No, 1=Yes", ["0", "1"])
            node['human_label'] = user_feedback
            print('############################################')
            with open(aspect_save_path, "w") as f:
                json.dump(node, f, indent=4)
            print(f"Aspect {idx} saved to {aspect_save_path}")


def result_processor(results_path):
    topic_labels = defaultdict(list)
    
    for topic in TOPIC_LIST:
        topic_results_path = os.path.join(results_path, topic)
        if not os.path.exists(topic_results_path):
            print(f"No results found for topic {topic}. Skipping.")
            continue
        
        for file_name in os.listdir(topic_results_path):
            if file_name.endswith(".json"):
                file_path = os.path.join(topic_results_path, file_name)
                with open(file_path, "r") as f:
                    node = json.load(f)
                    if 'human_label' in node:
                        topic_labels[topic].append(node['human_label'])
    
    for topic, labels in topic_labels.items():
        label_counts = defaultdict(int)
        for label in labels:
            label_counts[label] += 1
        
        total_labels = len(labels)
        print(f"Results for topic {topic}:")
        for label, count in label_counts.items():
            probability = count / total_labels
            print(f"Label {label}: {probability:.2f}")

def main():
    
    TOP_K = 15
    
    parser = argparse.ArgumentParser(description="Human Evaluation Script")
    parser.add_argument('--data_path', type=str, default='res', help='Path to the data file')
    parser.add_argument('--results_path', type=str, default=f'eval/human_judge/res/top_{TOP_K}', help='Path to save the results')
    parser.add_argument('--json_prefix', type=str, default='3_3_3/aspect_hierarchy.json', help='Prefix for JSON files')
    parser.add_argument('--segment_per_perspective', type=int, default=TOP_K, help='Number of segments per perspective')
    parser.add_argument('--sample_per_split', type=int, default=20, help='Number of samples per split')
    parser.add_argument('--user_num', type=int, default=2, help='Number of users')

    args = parser.parse_args()
    
    processed_data = load_data(args.data_path, args.json_prefix, args.segment_per_perspective, args.sample_per_split)
    user_interface(processed_data, args.user_num, args.results_path)
    result_processor(args.results_path)

if __name__ == "__main__":
    main()