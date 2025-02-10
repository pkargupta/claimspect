import os
import json
import random
from tqdm import tqdm
from itertools import combinations
from api.openai.chat import chat
from api.scholar.search_by_plain_str import search_literature_by_plain_text
from tenacity import RetryError

DECOMPOSITION_PROMPT = """
Task: You are an expert in analyzing and deconstructing claims into their core themes and underlying components. Given a specific claim, your goal is to:

Identify the main entities or subjects in the claim.
Break down the claim into its primary themes and related subtopics.
List the components in a concise, bullet-point format.
Format:
Input: <Claim>
Output: <Decomposed Components>

Example 1
Input:
The Cooperative Threat Reduction Program's focus on partner nations may inadvertently lead to decreased domestic investment in the U.S. public health infrastructure.

Output:
- Cooperative Threat Reduction (CTR) Program
- U.S. Public Health Infrastructure
- Potential Resource Trade-offs
- Opportunity Costs of International Assistance
- Policy Critiques and Evaluations

Example 2
Input: {claim}
Output:"""

CACHE_FILE = '.cache/fine_components_cache.json'

def get_claims(
    tgt_dir: str="data/vaccine/raw_claims"
    ) -> list[str]:
    # get all the json file under tgt_dir
    # read the json file and get the claim
    # return a list of claims
    claims = []
    for file in os.listdir(tgt_dir):
        if file.endswith('.json'):
            with open(os.path.join(tgt_dir, file), 'r') as f:
                data = json.load(f)
                claims.extend(data)
    return claims

def parse_components(raw_components: str) -> list[str]:
    split_components = raw_components.split('\n')
    stripped_components = [component.replace('-', '').strip() for component in split_components if component]
    return stripped_components

def generate_unique_pairs(strings_list):
    """
    Generate all unique pairs from a list of strings.

    :param strings_list: List of strings to generate pairs from
    :return: List of unique pairs as tuples
    """
    return list(combinations(strings_list, 2))

def main():
    claims = get_claims()
    
    # Randomly sample 400 claims
    random.seed(42)
    claims = random.sample(claims, 150)  # otherwise it will take too long to run
    
    # Check if cache file exists
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            fine_components = json.load(f)
    else:
        # for each claim, we want llm to extract the major component of it.
        raw_components = chat([DECOMPOSITION_PROMPT.format(claim=claim) for claim in claims], model_name='gpt-4o', seed=42)
        fine_components = [parse_components(raw_component) for raw_component in raw_components]
        
        # Save fine_components to cache file
        with open(CACHE_FILE, 'w') as f:
            json.dump(fine_components, f, indent=4)

    claim2paper = {}
    for claim, fine_component in tqdm(zip(claims, fine_components), total=len(claims)):
        claim2paper[claim] = []
        # gather the search word by combining the fine components
        pairs = generate_unique_pairs(fine_component)
        for i, pair in enumerate(pairs, start=1):
            search_word = '+'.join([f"{pair[0]}", f"{pair[1]}"])
            try:
                results = search_literature_by_plain_text(search_word, max_num=100)
            except RetryError as e:
                print(f"Error: {e}")
                results = []
            claim2paper[claim].extend(results)
            print(f"Completed {i}/{len(pairs)} searches for claim: {claim}")

    with open('data/vaccine/get_literature/claim2paper_meta_info.json', 'w') as f:
        json.dump(claim2paper, f, indent=4)

if __name__ == '__main__':
    main()