import os
import re
import json
import argparse
from dataclasses import dataclass
from typing import Literal, List

# Constants
NODE_SCORE_NAME = "node_level_eval.json"
ZEROSHOT_SCORE_NAME = "zeroshot_baseline_node_level_eval.json"
RAG_SCORE_NAME = "rag_baseline_node_level_eval.json"
BASELINE_COMPARE_NAME = "preference_comparison_rag.json"
METHOD_COMPARE_NAME = "preference_comparison_our_method.json"

@dataclass
class Path:
    zeroshot_baseline_score_path: str
    rag_baseline_score_path: str
    method_score_path: str
    baseline_compare_path: str
    method_compare_path: str

@dataclass
class Score:
    node_relevance: float
    vertical_granularity: float
    horizontal_granularity: float
    uniqueness: float
    segment_quality: float

@dataclass
class GroupScore:
    zeroshot_score: Score
    rag_score: Score
    method_score: Score

@dataclass
class Comparison:
    zeroshot_vs_rag: Literal['zeroshot_wins', 'rag_wins', 'tie', 'inconsistent']
    zeroshot_vs_method: Literal['zeroshot_wins', 'method_wins', 'tie', 'inconsistent']
    rag_vs_method: Literal['rag_wins', 'method_wins', 'tie', 'inconsistent']

@dataclass
class Instance:
    claim_id: str
    group_score: GroupScore
    comparison: Comparison

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', type=str, default="dtra")
    parser.add_argument('--metric_dir', type=str, default="eval/metrics")
    parser.add_argument('--judge_name', type=str, default="gpt-4o-mini")
    return parser.parse_args()

def get_paths(args) -> List[Path]:
    tgt_dir = f"{args.metric_dir}/{args.topic}"
    claim_dirs = [f"{tgt_dir}/{d}" for d in os.listdir(tgt_dir) if os.path.isdir(f"{tgt_dir}/{d}")]
    claim_judge_dirs = [f"{d}/{args.judge_name}" for d in claim_dirs if os.path.isdir(f"{d}/{args.judge_name}")]
    
    path_results = []
    for d in claim_judge_dirs:
        paths = Path(
            f"{d}/{ZEROSHOT_SCORE_NAME}",
            f"{d}/{RAG_SCORE_NAME}",
            f"{d}/{NODE_SCORE_NAME}",
            f"{d}/{BASELINE_COMPARE_NAME}",
            f"{d}/{METHOD_COMPARE_NAME}"
        )
        if all(os.path.exists(p) for p in paths.__dict__.values()):
            path_results.append(paths)
        else:
            print(f"Missing files for {d}, skipped")
    
    return path_results

def load_score(path: str) -> Score:
    with open(path, 'r') as f:
        json_file = json.load(f)
    return Score(
        json_file['avg_relevance_score'],
        json_file['avg_granularity_score'],
        json_file['avg_level_granularity_scpre'],
        json_file['taxonomy_wise_uniqueness_score'],
        json_file['avg_segment_quality_scpre']
    )

def parse_comparison(result: str) -> str:
    mapping = {
        "SingleResult.INVALID": "inconsistent",
        "SingleResult.BASELINE_WINS": "zeroshot_wins",
        "SingleResult.METHOD_WINS": "method_wins",
        "SingleResult.TIE": "tie"
    }
    return mapping.get(result, "inconsistent")

def get_comparison(baseline_compare_path: str, method_compare_path: str) -> Comparison:
    with open(baseline_compare_path, 'r') as f:
        baseline_compare = json.load(f)
    with open(method_compare_path, 'r') as f:
        method_compare = json.load(f)
    
    return Comparison(
        parse_comparison(baseline_compare[0]['result']),
        parse_comparison(method_compare[0]['result']),
        parse_comparison(method_compare[1]['result'])
    )

def get_instance(path: Path) -> Instance:
    match = re.search(r'claim_(\d+)', path.zeroshot_baseline_score_path)
    assert match, "Failed to parse claim ID"
    
    return Instance(
        claim_id=match.group(0),
        group_score=GroupScore(
            load_score(path.zeroshot_baseline_score_path),
            load_score(path.rag_baseline_score_path),
            load_score(path.method_score_path)
        ),
        comparison=get_comparison(path.baseline_compare_path, path.method_compare_path)
    )

def get_instances(paths: List[Path]) -> List[Instance]:
    return [get_instance(path) for path in paths]

def get_results(instances: List[Instance]) -> dict:
    total_instances = len(instances)
    score_fields = ['node_relevance', 'vertical_granularity', 'horizontal_granularity', 'uniqueness', 'segment_quality']
    
    def avg_scores(score_list):
        return [sum(score.__dict__[field] for score in score_list) / total_instances for field in score_fields]
    
    zeroshot_scores = avg_scores([instance.group_score.zeroshot_score for instance in instances])
    rag_scores = avg_scores([instance.group_score.rag_score for instance in instances])
    method_scores = avg_scores([instance.group_score.method_score for instance in instances])
    
    comparison_counts = {key: {res: 0 for res in ['zeroshot_wins', 'rag_wins', 'method_wins', 'tie', 'inconsistent']} for key in ['zeroshot_vs_rag', 'zeroshot_vs_method', 'rag_vs_method']}
    
    for instance in instances:
        for key in comparison_counts:
            comparison_counts[key][getattr(instance.comparison, key)] += 1
    
    comparison_distributions = {key: {res: count / total_instances for res, count in value.items()} for key, value in comparison_counts.items()}
    
    return {
        'avg_zeroshot_scores': zeroshot_scores,
        'avg_rag_scores': rag_scores,
        'avg_method_scores': method_scores,
        'comparison_distributions': comparison_distributions
    }

def show_results(results: dict, output_dir: str):
    score_titles = [
        "Node Relevance",
        "Vertical Granularity",
        "Horizontal Granularity",
        "Uniqueness",
        "Segment Quality"
    ]
    
    print("Average Scores:")
    for label, scores in zip(["Zeroshot", "RAG", "Method"], [results['avg_zeroshot_scores'], results['avg_rag_scores'], results['avg_method_scores']]):
        print(f"{label}:")
        for title, score in zip(score_titles, scores):
            print(f"  {title}: {score:.4f}")
    
    print("\nComparison Distributions:")
    for comparison, distribution in results['comparison_distributions'].items():
        print(f"{comparison}:")
        for result, percentage in distribution.items():
            print(f"  {result}: {percentage:.2%}")
    
    # Convert average scores to dictionaries
    avg_zeroshot_scores = {title: score for title, score in zip(score_titles, results['avg_zeroshot_scores'])}
    avg_rag_scores = {title: score for title, score in zip(score_titles, results['avg_rag_scores'])}
    avg_method_scores = {title: score for title, score in zip(score_titles, results['avg_method_scores'])}
    
    # Save results to a file with headers
    results_with_headers = {
        'score_titles': score_titles,
        'avg_zeroshot_scores': avg_zeroshot_scores,
        'avg_rag_scores': avg_rag_scores,
        'avg_method_scores': avg_method_scores,
        'comparison_distributions': results['comparison_distributions']
    }
    
    with open(f"{output_dir}/overall_statistics.json", 'w') as f:
        json.dump(results_with_headers, f, indent=4)

def main():
    args = parse_args()
    instances = get_instances(get_paths(args))
    results = get_results(instances)
    
    output_dir = f"{args.metric_dir}/statistics/{args.topic}_{args.judge_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    show_results(results, output_dir)
    
    # Save each instance to a single text file
    instances_file = f"{output_dir}/instances.txt"
    with open(instances_file, 'w') as f:
        for instance in instances:
            f.write(json.dumps(instance.__dict__, indent=4, default=lambda o: o.__dict__) + "\n")

if __name__ == '__main__':
    main()
