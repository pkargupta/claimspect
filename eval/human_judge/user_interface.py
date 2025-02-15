import os
import json
from eval.human_judge.utils import robust_input, get_user_id

def user_interface(processed_data: dict, user_num: int, results_path: str) -> dict:
    """
    Present data to the user for evaluation, collect human feedback, and save results.
    
    :param processed_data: Data organized by topic ready for human evaluation.
    :param user_num: Total number of evaluators.
    :param results_path: Base path where evaluation results will be stored.
    :return: A dictionary containing evaluation results.
    """
    user_id = get_user_id(user_num)
    results = {}

    for topic, nodes in processed_data.items():
        topic_results_path = os.path.join(results_path, topic)
        os.makedirs(topic_results_path, exist_ok=True)
        
        print(f"Evaluating topic: {topic}")
        results[topic] = []
        for idx, node in enumerate(nodes):
            # Allow split evaluation among multiple users
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
                feedback = robust_input("Input your feedback (0=support, 1=neutral, 2=oppose):", ["0", "1", "2"])
                seg["human_label"] = "support" if feedback == "0" else "neutral" if feedback == "1" else "oppose"
                aspect_results.append(seg)
                print("############################################")
            with open(aspect_save_path, "w") as f:
                json.dump(aspect_results, f, indent=4)
            print(f"Aspect {idx} saved to {aspect_save_path}")
            results[topic].append(aspect_results)
    
    return results
