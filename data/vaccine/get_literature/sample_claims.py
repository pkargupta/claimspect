import os
import json
import random

SAMPLED_CLAIMS_FILE = 'data/vaccine/claims.json'
UNFILTERED_CLAIMS_FILE = 'data/vaccine/unfiltered_claims.json'
OUTPUT_FILE = 'data/vaccine/claims.json'

def get_claims(tgt_dir: str="data/vaccine/raw_claims") -> list[str]:
    claims = []
    for file in os.listdir(tgt_dir):
        if file.endswith('.json'):
            with open(os.path.join(tgt_dir, file), 'r') as f:
                data = json.load(f)
                claims.extend(data)
    return claims

def main():
    claims = get_claims()
    
    # Randomly sample 150 claims
    random.seed(42)
    sampled_claims = random.sample(claims, 150)
    
    # Load unfiltered claims
    with open(UNFILTERED_CLAIMS_FILE, 'r') as f:
        unfiltered_claims = json.load(f)
    
    # Filter unfiltered claims based on sampled claims' bodies
    filtered_claims = [claim for claim in unfiltered_claims if claim['body'] in sampled_claims]
    
    # Re-number the filtered claims from 1 to 150
    for i, claim in enumerate(filtered_claims, start=1):
        claim['id'] = i
    
    # Save filtered claims to a new JSON file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(filtered_claims, f, indent=4)
    
if __name__ == '__main__':
    main()