import os
import argparse
from typing import List
from tqdm import tqdm
from eval.utils import perform_evaluation

def generate_paths(root_dir: str, suffix: str) -> List[str]:
    """
    Generate a list of paths by combining root_dir, each number in numbers, and suffix.
    """
    # find all the directories in the root_dir
    directories = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    # give the full path to the directories
    paths = [os.path.join(root_dir, d, suffix) for d in directories]
    # check if they exist, get the existing paths, report the missing ones
    paths = [(p, d) for p, d in zip(paths, directories) if os.path.exists(p)]
    print(f"Found {len(paths)} existing paths.")
    print(f"Missing {len(directories) - len(paths)} paths.")
    return paths

def main():
    
    """ Step I: Load the data """
    parser = argparse.ArgumentParser(description='Process directories.')
    parser.add_argument('--result_directory', type=str, required=True, help='Path to the result directory')
    parser.add_argument('--output_directory', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--hierarchy_prefix', type=str, required=True, help='Prefix for the hierarchy path')
    parser.add_argument('--hierarchy_suffix', type=str, required=True, help='Suffix for the hierarchy path')
    parser.add_argument('--llm_judge', type=str, required=True, help='Evaluation model')
    parser.add_argument('--do_eval_node_level', action='store_true', help='Evaluate node level')
    parser.add_argument('--do_eval_taxonomy_level', action='store_true', help='Evaluate taxononmy level')
    parser.add_argument('--do_eval_ablation', action='store_true', help='Evaluate ablation')
    parser.add_argument('--baseline_model_name', type=str, required=True, help='Baseline model name')
    parser.add_argument('--taxo_height', type=int, required=True, help='Taxonomy height')
    parser.add_argument('--child_num_per_node', type=int, required=True, help='Child number per node')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--topic', type=str, required=True, help='Topic')
    parser.add_argument('--ablation_directory', type=str, default=None, help='Path to the ablation directory')
    
    args = parser.parse_args()
    # get the hierarchy paths
    hierarchy_paths = generate_paths(args.hierarchy_prefix,args.hierarchy_suffix)
    
    """ Step II: Process the data """
    for hierarchy_path, idx in tqdm(hierarchy_paths, desc="Processing hierarchy paths", total=len(hierarchy_paths)):
        
        local_output_path = os.path.join(args.output_directory, f"claim_{idx}")
        print("Processing hierarchy path:", hierarchy_path)
        perform_evaluation(hierarchy_path, 
                           local_output_path, 
                           args.llm_judge, 
                           args.do_eval_node_level,
                           args.do_eval_taxonomy_level,
                           args.do_eval_ablation,
                           args.baseline_model_name,
                           args.taxo_height,
                           args.child_num_per_node,
                           args.data_dir,
                           args.topic,
                           int(idx),
                           args)

if __name__ == "__main__":
    main()
