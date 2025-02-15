import os
import json
from eval.human_judge.utils import robust_input, get_user_id

def user_interface(processed_data: dict, user_num: int, results_path: str, topic) -> dict:
    """
    Present data to the user for evaluation, collect human feedback, and save results.
    
    :param processed_data: Data organized by topic ready for human evaluation.
    :param user_num: Total number of evaluators.
    :param results_path: Base path where evaluation results will be stored.
    :return: A dictionary containing evaluation results.
    """
    user_id = get_user_id(user_num)
    results = {}

    nodes = processed_data[topic]
    topic_results_path = os.path.join(results_path, topic)
    os.makedirs(topic_results_path, exist_ok=True)
    
    print(f"Evaluating topic: {topic}")
    results[topic] = []
    for idx, node in enumerate(nodes):
        if idx % user_num != int(user_id):
            continue
        
        print(f"Evaluating Aspect {idx+1}/{len(nodes)}")
        aspect_save_path = os.path.join(topic_results_path, f"aspect_{idx}.json")
        if os.path.exists(aspect_save_path):
            print(f"Aspect {idx} already evaluated. Skipping.")
            results[topic].append(json.load(open(aspect_save_path)))
            continue

        aspect_results = []
        for seg in node["segments"]:
            print("############################################")
            print(f"Aspect Name: {node['aspect_name']}")
            print("############################################")
            print(f"Aspect Description: {node['aspect_description']}")
            print("############################################")
            print("Please evaluate the following segment:")
            print("############################################")
            print(seg["segment"])
            
            user_feedback = robust_input("Please provide feedback (0:support, 1:neutral, 2:oppose, quit:quit): ", ["0", "1", "2", "quit"])
            if user_feedback == "quit":
                return

            seg["human_label"] = "support" if user_feedback == "0" else "neutral" if user_feedback == "1" else "oppose"
            aspect_results.append(seg)
            print("############################################")
        
        with open(aspect_save_path, "w") as f:
            json.dump(aspect_results, f, indent=4)
        print(f"Aspect {idx} saved to {aspect_save_path}")
        results[topic].append(aspect_results)
    
    return