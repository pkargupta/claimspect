import os
import argparse
from typing import List
from tqdm import tqdm
from eval.utils import perform_evaluation

def generate_paths(root_dir: str, numbers: List[int], suffix: str) -> List[str]:
    """
    Generate a list of paths by combining root_dir, each number in numbers, and suffix.
    """
    paths = []
    for number in numbers:
        path = os.path.join(root_dir, str(number), suffix)
        paths.append(path)
    return paths

def extract_completed_numbers(file_path: str) -> List[int]:
    """
    Extract completed numbers from a file. Each line starting with 'Completed:' is parsed to extract the number.
    """
    completed_numbers = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("Completed:"):
                number = int(line.split(":")[1].strip())
                completed_numbers.append(number)
    return completed_numbers

def main():
    
    """ Step I: Load the data """
    parser = argparse.ArgumentParser(description='Process directories.')
    parser.add_argument('--result_directory', type=str, required=True, help='Path to the result directory')
    parser.add_argument('--index_path', type=str, required=True, help='Path to the index file')
    parser.add_argument('--output_directory', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--hierarchy_prefix', type=str, required=True, help='Prefix for the hierarchy path')
    parser.add_argument('--hierarchy_suffix', type=str, required=True, help='Suffix for the hierarchy path')
    parser.add_argument('--llm_judge', type=str, required=True, help='Evaluation model')
    
    args = parser.parse_args()
    
    # Extract completed numbers from the index file
    completed_numbers = extract_completed_numbers(args.index_path)
    # get the hierarchy paths
    hierarchy_paths = generate_paths(args.hierarchy_prefix, completed_numbers, args.hierarchy_suffix)
    
    """ Step II: Process the data """
    for hierarchy_path, idx in tqdm(zip(hierarchy_paths, completed_numbers), desc="Processing hierarchy paths", total=len(hierarchy_paths)):
        
        local_output_path = os.path.join(args.output_directory, f"claim_{idx}")
        if os.path.exists(local_output_path):
            continue
        perform_evaluation(hierarchy_path, local_output_path, args.llm_judge)
    

if __name__ == "__main__":
    main()
