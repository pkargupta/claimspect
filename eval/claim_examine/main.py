import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

random.seed(42)

def load_claim_data(file_path: str) -> Dict:
    """Load claim data from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_user_input(prompt: str) -> str:
    """Get input from user."""
    return input(prompt).strip()

def evaluate_claim(claim: str, topic: str) -> Tuple[int, int]:
    """Evaluate a single claim for nuance and relevancy."""
    print(f"\nClaim: {claim}")
    
    while True:
        nuance = input("Nuance (0/1): ").strip()
        if nuance in ['0', '1']:
            break
        print("Please enter 0 or 1")
    
    while True:
        relevancy = input("Relevancy (0/1): ").strip()
        if relevancy in ['0', '1']:
            break
        print("Please enter 0 or 1")
    
    return int(nuance), int(relevancy)

def evaluate_papers(papers: List[Dict], topic: str) -> Tuple[bool, bool]:
    """Evaluate papers for corpus alignment."""
    print("\nPaper titles:")
    for i, paper in enumerate(papers[:100], 1):
        print(f"{i}. {paper.get('title', 'No title')}")
    
    if len(papers) > 100:
        print(f"\nNote: Only showing first 100 papers out of {len(papers)} total papers")
    
    while True:
        has_5 = input("\nAre there at least 5 relevant papers? (y/n): ").strip().lower()
        if has_5 in ['y', 'n']:
            break
        print("Please enter y or n")
    
    while True:
        has_10 = input("Are there at least 10 relevant papers? (y/n): ").strip().lower()
        if has_10 in ['y', 'n']:
            break
        print("Please enter y or n")
    
    return has_5 == 'y', has_10 == 'y'

def main():
    # Get input file path
    file_path = get_user_input("Enter the path to the claim data JSON file: ")
    
    # Get topic
    topic = get_user_input("Enter the topic (e.g., vaccine, dtra): ")
    
    # Get number of claims to sample
    while True:
        try:
            num_claims = int(get_user_input("Enter the number of claims to sample: "))
            if num_claims > 0:
                break
            print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")
    
    # Load data
    data = load_claim_data(file_path)
    
    # Sample claims
    claims = random.sample(list(data.keys()), min(num_claims, len(data)))
    
    # Initialize results
    log_data = []
    results = {
        'nuance_scores': [],
        'relevancy_scores': [],
        'has_5_papers': [],
        'has_10_papers': []
    }
    
    # Evaluate each claim
    for claim in claims:
        print("\n" + "="*80)
        print(f"Evaluating claim {len(log_data) + 1}/{len(claims)}")
        
        # Evaluate claim
        nuance, relevancy = evaluate_claim(claim, topic)
        
        # Evaluate papers
        has_5, has_10 = evaluate_papers(data[claim], topic)
        
        # Record results
        log_entry = {
            'claim': claim,
            'nuance': nuance,
            'relevancy': relevancy,
            'has_5_papers': has_5,
            'has_10_papers': has_10
        }
        log_data.append(log_entry)
        
        results['nuance_scores'].append(nuance)
        results['relevancy_scores'].append(relevancy)
        results['has_5_papers'].append(1 if has_5 else 0)
        results['has_10_papers'].append(1 if has_10 else 0)
    
    # Calculate averages
    averages = {
        'nuance_avg': sum(results['nuance_scores']) / len(results['nuance_scores']),
        'relevancy_avg': sum(results['relevancy_scores']) / len(results['relevancy_scores']),
        'has_5_papers_avg': sum(results['has_5_papers']) / len(results['has_5_papers']),
        'has_10_papers_avg': sum(results['has_10_papers']) / len(results['has_10_papers'])
    }
    
    # Save results
    output_dir = Path(file_path).parent
    topic = topic.lower().replace(' ', '_')
    
    # Save log
    log_file = output_dir / f"{topic}_log.json"
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    # Save averages
    results_file = output_dir / f"{topic}_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(averages, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to:")
    print(f"Log file: {log_file}")
    print(f"Results file: {results_file}")

if __name__ == "__main__":
    main() 