# First we load all the data
import os
import json
import pandas as pd
import random
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

VACCINE_DIR = Path("eval/metrics/vaccine")
DTRA_DIR = Path("eval/metrics/dtra")
VACCINE_RES_DIR = Path("res/vaccine")
DTRA_RES_DIR = Path("res/dtra")

@dataclass
class NodeRelevanceInput:
    claim: str
    path: str
    human_result: Optional[float] = None
    llm_result: Optional[float] = None
    llm_reasoning: Optional[str] = None
    
    def get_prompt(self) -> str:
        return (
            "Claims made by individuals or entities are often nuanced and cannot always be strictly categorized as entirely 'true' or 'false'. "
            "Particularly in scientific and political contexts. Instead, a claim can be broken down "
            "into its core aspects and sub-aspects, which are easier to evaluate individually.\n\n"
            f"Given the claim: '{self.claim}', decide whether this path from the aspect tree is relevant to the analysis of the claim: '{self.path}'\n\n"
            "Output options: '<relevant>' or '<irelevant>'. Do some simple rationalization before giving the output if possible."
        )
    
    def is_aligned(self) -> bool:
        return self.human_result == self.llm_result

@dataclass
class PathGranularityInput:
    claim: str
    path: str
    human_result: Optional[str] = None
    llm_result: Optional[str] = None
    llm_reasoning: Optional[str] = None
    
    def get_prompt(self) -> str:
        return (
            "Claims made by individuals or entities are often nuanced and cannot always be strictly categorized as entirely 'true' or 'false'. "
            "Particularly in scientific and political contexts. Instead, a claim can be broken down "
            "into its core aspects and sub-aspects, which are easier to evaluate individually.\n\n"
            f"Given the claim: '{self.claim}', decide whether this path from the aspect tree has good granularity: '{self.path}' Check whether the child node is a more specific subaspect of the parent node. \n\n"
            "Output options: '<good granularity>' or '<bad granularity>'. Do some simple rationalization before giving the output if possible."
        )
    
    def is_aligned(self) -> bool:
        return self.human_result == self.llm_result

@dataclass
class LevelGranularityInput:
    claim: str
    parent: str
    siblings: List[str]
    human_result: Optional[str] = None
    llm_result: Optional[str] = None
    llm_reasoning: Optional[str] = None
    
    def get_prompt(self) -> str:
        return (
            "Claims made by individuals or entities are often nuanced and cannot always be strictly categorized as entirely 'true' or 'false'. "
            "Particularly in scientific and political contexts. Instead, a claim can be broken down "
            "into its core aspects and sub-aspects, which are easier to evaluate individually.\n\n"
            f"Given the claim: '{self.claim}', decide whether these siblings from parent node '{self.parent}', have good granularity: '{', '.join(self.siblings)}' Check whether they have similar specificity level. \n\n"
            "Output options: '<all not granular>' or '<majority not granular>' or "
            "'<majority granular>' or '<all granular>'. Do some simple rationalization before giving the output if possible."
        )
    
    def is_aligned(self) -> bool:
        return self.human_result == self.llm_result

@dataclass
class UniquenessInput:
    claim: str
    taxonomy: str
    human_result: Optional[str] = None
    llm_result: Optional[str] = None
    llm_reasoning: Optional[str] = None
    
    def get_prompt(self) -> str:
        return (
            "Claims made by individuals or entities are often nuanced and cannot always be strictly categorized as entirely 'true' or 'false'. "
            "Particularly in scientific and political contexts. Instead, a claim can be broken down "
            "into its core aspects and sub-aspects, which are easier to evaluate individually.\n\n"
            "Normally, we want the aspects and sub-aspects to be unique in the taxonomy. "
            f"Given the claim: '{self.claim}', count how many nodes in this taxonomy are largely overlapping or almost equivalent: {self.taxonomy}\n\n"
            "Output options: '<overlap_num>0</overlap_num>', '<overlap_num>1</overlap_num>'... or other possible numbers. Do some simple rationalization before giving the output if possible."
        )
    
    def is_aligned(self) -> bool:
        return self.human_result != -1 and self.llm_result != -1

@dataclass
class SegmentQualityInput:
    claim: str
    aspect_name: str
    segments: List[str]
    human_result: Optional[str] = None
    llm_result: Optional[str] = None
    llm_reasoning: Optional[str] = None
    
    def get_prompt(self) -> str:
        return (
            "Claims made by individuals or entities are often nuanced and cannot always be strictly categorized as entirely 'true' or 'false'. "
            "Particularly in scientific and political contexts. Instead, a claim can be broken down "
            "into its core aspects and sub-aspects, which are easier to evaluate individually.\n\n"
            f"Given the claim: '{self.claim}', evaluate the quality of these segments for aspect '{self.aspect_name}':\n"
            f"{chr(10).join(f'- {segment}' for segment in self.segments)}\n\n"
            "Output options: '<rel_seg_num> ... (int) </rel_seg_num>'. Do some rationalization before outputting the number of relevant segments."
        )
    
    def is_aligned(self) -> bool:
        return self.human_result != -1 and self.llm_result != -1

def get_topic_from_claim(claim_name: str, dataset_name: str) -> str:
    """
    Get the topic from the original aspect hierarchy file.
    
    Args:
        claim_name (str): Name of the claim (e.g., 'claim_6')
        dataset_name (str): Name of the dataset ('vaccine' or 'dtra')
    
    Returns:
        str: The topic from the aspect hierarchy file, or None if not found
    """
    # Extract claim number from claim_name
    claim_num = claim_name.split('_')[1]
    
    # Determine the correct directory based on dataset
    base_dir = VACCINE_RES_DIR if dataset_name == 'vaccine' else DTRA_RES_DIR
    
    # Construct the path to the aspect hierarchy file
    hierarchy_file = base_dir / claim_num / "3_3_3" / "aspect_hierarchy.json"
    
    if not hierarchy_file.exists():
        print(f"Warning: Could not find hierarchy file for {claim_name} at {hierarchy_file}")
        print(f"Available claims in {base_dir}: {sorted([d.name for d in base_dir.iterdir() if d.is_dir()])}")
        return None
        
    # Read and extract the topic
    try:
        with open(hierarchy_file, 'r') as f:
            data = json.load(f)
            topic = data.get('aspect_name', None)
            if not topic:
                print(f"Warning: No topic found in hierarchy file for {claim_name}")
                print(f"File contents: {data}")
            return topic
    except Exception as e:
        print(f"Error reading hierarchy file for {claim_name}: {str(e)}")
        return None

def load_evaluation_data(base_dir: Path, dataset_name: str) -> dict:
    """
    Load evaluation data from the specified directory for a given dataset.
    Only includes non-empty evaluation data.
    
    Args:
        base_dir (Path): Directory containing claim subdirectories
        dataset_name (str): Name of the dataset ('vaccine' or 'dtra')
    
    Returns:
        dict: Dictionary containing evaluation data for each claim and model
    """
    eval_data = {}
    
    # Get list of available claims
    available_claims = sorted([d.name for d in base_dir.iterdir() if d.is_dir()])
    print(f"Available claims in {base_dir}: {available_claims}")
    
    # Iterate through claim directories
    for claim_dir in base_dir.iterdir():
        if not claim_dir.is_dir():
            continue
            
        claim_data = {}
        # Iterate through model directories in each claim
        for model_dir in claim_dir.iterdir():
            if not model_dir.is_dir():
                continue
                
            node_eval_file = model_dir / "node_level_eval.json"
            if not node_eval_file.exists():
                continue
                
            # Load and validate evaluation data
            with open(node_eval_file, 'r') as f:
                data = json.load(f)
                if data and any(data.values()):  # Only include non-empty data
                    claim_data[model_dir.name] = data
        
        # Only include claims with valid evaluations
        if claim_data:
            # Get topic for this claim
            topic = get_topic_from_claim(claim_dir.name, dataset_name)
            if topic is not None:  # Only include claims with valid topics
                eval_data[claim_dir.name] = {
                    'topic': topic,
                    'evaluations': claim_data
                }
    
    print(f"Loaded {len(eval_data)} {dataset_name} claims")
    return eval_data

# Load data for both datasets
all_eval_data = {
    'vaccine': load_evaluation_data(VACCINE_DIR, 'vaccine'),
    'dtra': load_evaluation_data(DTRA_DIR, 'dtra')
}

# then we classify them into 5 categories
node_relevance_instances = []
path_granularity_instances = []
sibling_granularity_instances = []
uniqueness_instances = []
segment_quality_instances = []

for dataset_name in ['vaccine', 'dtra']:
    for claim_name, claim_data in all_eval_data[dataset_name].items():
        topic = claim_data['topic']
        if topic is None:  # Skip if topic is None
            continue
            
        for model_name, model_data in claim_data['evaluations'].items():
            
            # first deal with node relevance
            node_relevance_meta_data = model_data['path_wise_relevance']
            for data in node_relevance_meta_data:
                path = data['path']
                claim = topic
                machine_result = data['score']
                llm_reasoning = data['reasoning']
                instance = NodeRelevanceInput(
                    claim=claim,
                    path=path,
                    llm_result=machine_result,
                    llm_reasoning=llm_reasoning
                )
                node_relevance_instances.append(instance)
            
            # then deal with path granularity
            path_granularity_meta_data = model_data['path_wise_granularity']
            for data in path_granularity_meta_data:
                path = data['path']
                claim = topic
                machine_result = data['score']
                llm_reasoning = data['reasoning']
                instance = PathGranularityInput(
                    claim=claim,
                    path=path,
                    llm_result=machine_result,
                    llm_reasoning=llm_reasoning
                )
                path_granularity_instances.append(instance)
            
            # then deal with sibling granularity
            sibling_granularity_meta_data = model_data['level_wise_granularity']
            for data in sibling_granularity_meta_data:
                parent = data['path']['parent']
                siblings = data['path']['siblings']
                claim = topic
                machine_result = data['score']
                llm_reasoning = data['reasoning']
                instance = LevelGranularityInput(
                    claim=claim,
                    parent=parent,
                    siblings=siblings,
                    llm_result=machine_result,
                    llm_reasoning=llm_reasoning
                )
                sibling_granularity_instances.append(instance)
            
            # then deal with uniqueness
            uniqueness_meta_data = model_data['taxonomy_wise_uniqueness']
            data = uniqueness_meta_data
            taxonomy = data['taxonomy']
            claim = topic
            machine_result = data['score']
            llm_reasoning = data['reasoning']
            instance = UniquenessInput(
                claim=claim,
                taxonomy=taxonomy,
                llm_result=machine_result,
                llm_reasoning=llm_reasoning
            ) 
            uniqueness_instances.append(instance)
        
            # then deal with segment quality
            segment_quality_meta_data = model_data['node_wise_segment_quality']
            for data in segment_quality_meta_data:
                aspect_name = data['node']
                segments = data['segments']
                machine_result = data['score']
                llm_reasoning = data['reasoning']
                instance = SegmentQualityInput(
                    claim=topic,
                    aspect_name=aspect_name,
                    segments=segments,
                    llm_result=machine_result,  
                    llm_reasoning=llm_reasoning
                )
                segment_quality_instances.append(instance)

# Report the number of instances for each type
print(f"Number of node relevance instances: {len(node_relevance_instances)}")
print(f"Number of path granularity instances: {len(path_granularity_instances)}")
print(f"Number of sibling granularity instances: {len(sibling_granularity_instances)}")
print(f"Number of uniqueness instances: {len(uniqueness_instances)}")
print(f"Number of segment quality instances: {len(segment_quality_instances)}")

def sample_instances(instances: List[Any], k: int, seed: int) -> List[Any]:
    """Sample k instances from the list with a given seed, excluding instances where llm_result is -1."""
    # Filter out instances where llm_result is -1
    valid_instances = [instance for instance in instances if instance.llm_result != -1]
    if not valid_instances:
        return []
        
    random.seed(seed)
    return random.sample(valid_instances, min(k, len(valid_instances)))

def parse_human_result(output: str, eval_type: str) -> float:
    """Parse human input based on evaluation type."""
    output = output.lower().strip()
    
    if eval_type == "node_relevance":
        if "<relevant>" in output:
            return 1.0
        elif "<irelevant>" in output:
            return 0.0
        else:
            return -1.0
            
    elif eval_type == "path_granularity":
        if "<good granularity>" in output:
            return 1.0
        elif "<bad granularity>" in output:
            return 0.0
        else:
            return -1.0
            
    elif eval_type == "level_granularity":
        if "<all not granular>" in output:
            return 1.0
        elif "<majority not granular>" in output:
            return 2.0
        elif "<majority granular>" in output:
            return 3.0
        elif "<all granular>" in output:
            return 4.0
        else:
            return -1.0
            
    elif eval_type == "uniqueness":
        try:
            output_int = output.split('<overlap_num>')[1].split('</overlap_num>')[0].strip()
            return 1 - float(output_int/40)
        except:
            return -1.0
            
    elif eval_type == "segment_quality":
        try:
            output_int = output.split('<rel_seg_num>')[1].split('</rel_seg_num>')[0]
            return float(output_int) / 5
        except:
            return -1.0
            
    return -1.0

def calculate_metrics(instances: List[Any], eval_type: str) -> Dict[str, float]:
    """Calculate metrics for a list of instances."""
    total = len(instances)
    if total == 0:
        return {
            "avg_score_llm": 0,
            "avg_score_human": 0,
            "alignment_rate": 0
        }
    
    # For uniqueness and segment quality, alignment rate is based on valid scores
    if eval_type in ["level_granularity", "uniqueness", "segment_quality"]:
        alignment_rate = None
    else:
        aligned = sum(1 for instance in instances if instance.is_aligned())
        alignment_rate = aligned / total if total > 0 else 0
    
    # Calculate average scores based on evaluation type
    if eval_type == "node_relevance":
        avg_score_llm = sum(instance.llm_result for instance in instances) / total
        avg_score_human = sum(instance.human_result for instance in instances) / total
    elif eval_type == "path_granularity":
        avg_score_llm = sum(instance.llm_result for instance in instances) / total
        avg_score_human = sum(instance.human_result for instance in instances) / total
    elif eval_type == "level_granularity":
        avg_score_llm = sum(instance.llm_result for instance in instances) / total
        avg_score_human = sum(instance.human_result for instance in instances) / total
    elif eval_type == "uniqueness":
        # For uniqueness, we use the actual numbers from the results
        avg_score_llm = sum(instance.llm_result for instance in instances) / total
        avg_score_human = sum(instance.human_result for instance in instances) / total
    else:  # segment_quality
        # For segment quality, we use the actual numbers from the results
        avg_score_llm = sum(instance.llm_result for instance in instances) / total
        avg_score_human = sum(instance.human_result for instance in instances) / total
    
    return {
        "avg_score_llm": avg_score_llm,
        "avg_score_human": avg_score_human,
        "alignment_rate": alignment_rate
    }

def evaluate_instances(instances: List[Any], eval_type: str, k: int, seed: int, output_dir: Path):
    """Evaluate a list of instances with human input."""
    # Sample instances
    sampled_instances = sample_instances(instances, k, seed)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate each instance
    for i, instance in enumerate(sampled_instances, 1):
        while True:
            print(f"\n{'='*80}")
            print(f"Instance {i}/{len(sampled_instances)}")
            print(f"{'='*80}")
            print(instance.get_prompt())
            print(f"{'='*80}\n")
            
            response = input("Please enter your response: ")
            human_result = parse_human_result(response, eval_type)
            
            if human_result != -1:
                instance.human_result = human_result
                break
            else:
                print("\nInvalid input. Please try again with a valid response format.")
                print("Valid formats are:")
                if eval_type == "node_relevance":
                    print("- <relevant> or <irelevant>")
                elif eval_type == "path_granularity":
                    print("- <good granularity> or <bad granularity>")
                elif eval_type == "level_granularity":
                    print("- <all not granular>, <majority not granular>, <majority granular>, or <all granular>")
                elif eval_type == "uniqueness":
                    print("- <overlap_num>X</overlap_num> where X is a number")
                elif eval_type == "segment_quality":
                    print("- <rel_seg_num>X</rel_seg_num> where X is a number")
                print("\n")
    
    # Calculate metrics
    metrics = calculate_metrics(sampled_instances, eval_type)
    
    # Save results
    output_file = output_dir / f"{eval_type}_k{k}_seed{seed}.json"
    results = {
        "metrics": metrics,
        "instances": [asdict(instance) for instance in sampled_instances]
    }
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return metrics

def main(k: int = 5, seed: int = 42, output_dir: Path = Path("eval/human_align")):
    """Main function to run the evaluation."""
    eval_types = {
        "node_relevance": node_relevance_instances,
        "path_granularity": path_granularity_instances,
        "level_granularity": sibling_granularity_instances,
        "uniqueness": uniqueness_instances,
        "segment_quality": segment_quality_instances
    }
    
    results = {}
    for eval_type, instances in eval_types.items():
        print(f"\nEvaluating {eval_type}...")
        metrics = evaluate_instances(instances, eval_type, k, seed, output_dir)
        results[eval_type] = metrics
        
        print(f"\nResults for {eval_type}:")
        print(f"Average LLM score: {metrics['avg_score_llm']:.3f}")
        print(f"Average human score: {metrics['avg_score_human']:.3f}")
        print(f"Alignment rate: {metrics['alignment_rate']:.3f}") if metrics['alignment_rate'] is not None else print("Alignment rate: N/A")
    
    # Save overall results
    overall_results_file = output_dir / f"overall_results_k{k}_seed{seed}.json"
    with open(overall_results_file, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10, help="Number of instances to sample for each type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--output_dir", type=Path, default=Path("eval/human_align"), help="Output directory for results")
    args = parser.parse_args()
    
    main(args.k, args.seed, args.output_dir)
