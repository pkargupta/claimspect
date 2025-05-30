import os
import sys
import json

def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, "r") as f:
        return json.load(f)

def calculate_metrics(true_list, false_list, pred_list):
    """Calculate precision and recall metrics."""
    true_positives = len(set(pred_list) & set(true_list))
    false_positives = len(set(pred_list) & set(false_list))
    true_negatives = len(set(false_list) - set(pred_list))
    false_negatives = len(set(true_list) - set(pred_list))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return precision, recall

def save_metrics(metrics, output_path):
    """Save metrics to a JSON file."""
    with open(output_path, "w") as f:
        json.dump(metrics, f)

def main(pred_json_path):
    # Load the full input JSON file
    full_json_path = "filter/example_data/filter_input.json"
    full_list = load_json(full_json_path)

    # Split the full list into true and false segments
    true_list = full_list[:32]
    false_list = full_list[32:]

    # Load the prediction JSON file
    pred_list = load_json(pred_json_path)

    # Calculate precision and recall metrics
    precision, recall = calculate_metrics(true_list, false_list, pred_list)

    # Print metrics to the console
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")

    # Save metrics to a file
    output_dir = "filter/example_data/metric_results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(pred_json_path))
    save_metrics({"Precision": precision, "Recall": recall}, output_path)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <pred_json_path>")
        sys.exit(1)

    prediction_json_path = sys.argv[1]
    main(prediction_json_path)
