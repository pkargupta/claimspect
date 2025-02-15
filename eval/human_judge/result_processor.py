import os
import json
from collections import defaultdict

def result_processor(results_path: str) -> dict:
    """
    Process evaluation results and compute the accuracy of LLM labels by topic.

    :param results_path: Base path where evaluation results are stored.
    :return: Dictionary containing accuracy results per topic.
    """
    llm_correct = defaultdict(int)
    llm_total = defaultdict(int)
    
    for topic in os.listdir(results_path):
        topic_path = os.path.join(results_path, topic)
        if not os.path.isdir(topic_path):
            continue
        
        for file_name in os.listdir(topic_path):
            if not file_name.endswith(".json"):
                continue
            file_path = os.path.join(topic_path, file_name)
            with open(file_path, "r") as f:
                aspects = json.load(f)
                for seg in aspects:
                    llm_label = seg.get("llm_label")
                    human_label = seg.get("human_label")
                    if llm_label and human_label:
                        llm_total[topic] += 1
                        if llm_label == human_label:
                            llm_correct[topic] += 1

    accuracy_results = {}
    for topic in llm_total:
        accuracy = llm_correct[topic] / llm_total[topic] if llm_total[topic] > 0 else 0
        accuracy_results[topic] = accuracy
        print(f"{topic}: {accuracy:.2%} accuracy")
    
    print("LLM Label Accuracy by Topic:")
    for topic, acc in accuracy_results.items():
        print(f"{topic}: {acc:.2%}")
    
    return accuracy_results
