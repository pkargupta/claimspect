import argparse
from eval.human_judge.data_loader import load_data
from eval.human_judge.node_processing import flatten_tree, sample_nodes, organize_nodes
from eval.human_judge.user_interface import user_interface
from eval.human_judge.result_processor import result_processor

def main():
    parser = argparse.ArgumentParser(description="Human Evaluation Script")
    parser.add_argument('--data_path', type=str, default='res', help='Path to the data directory')
    parser.add_argument('--results_path', type=str, default='eval/human_judge/res', help='Path to save evaluation results')
    parser.add_argument('--json_prefix', type=str, default='3_3_3/aspect_hierarchy.json', help='Prefix for JSON files')
    parser.add_argument('--segment_per_perspective', type=int, default=50, help='Segments per perspective')
    parser.add_argument('--sample_per_split', type=int, default=2, help='Number of node samples per topic')
    parser.add_argument('--user_num', type=int, default=2, help='Total number of evaluators')

    args = parser.parse_args()

    # Load and process the data
    processed_data = load_data(
        data_path=args.data_path,
        json_prefix=args.json_prefix,
        segment_per_perspective=args.segment_per_perspective,
        sample_per_split=args.sample_per_split,
        flatten_func=flatten_tree,
        sample_func=sample_nodes,
        organize_func=organize_nodes
    )

    # Run the user interface for human evaluation
    user_interface(processed_data, args.user_num, args.results_path)
    
    # Process and display the results
    result_processor(args.results_path)

if __name__ == "__main__":
    main()
