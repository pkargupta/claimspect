import os
import time
import json
import random
from tqdm import tqdm
from preprocessing.pdf2txt import parse_paper

SAVE_PATH = "data/dtra/get_literature/literature_body"
MAX_PAPER_NUM = 100

random.seed(42)

def load_meta_info(meta_info_path: str="data/dtra/get_literature/claim2paper_meta_info.json"):
    print(f"Loading meta info from {meta_info_path}...")
    start_time = time.time()
    with open(meta_info_path, 'r') as f:
        meta_info = json.load(f)
    print(f"Loaded meta info in {time.time() - start_time:.2f} seconds.")
    return meta_info

class Claim2idConverter:
    
    def __init__(self, 
                 claim_json_path: str="data/dtra/claims.json"):
        with open(claim_json_path, 'r') as f:
            claims = json.load(f)
        self.claim2id = {claim['body']: claim['id'] for claim in claims}
    
    def __call__(self, claim: str):
        return self.claim2id[claim]

def save_results_for_claim(claim_id, results):
    """Save results for a given claim_id."""
    claim_dir = f"{SAVE_PATH}/{claim_id}"
    os.makedirs(claim_dir, exist_ok=True)
    for instance in results:
        paper_id = instance['paperId']
        body = instance['body']
        file_path = f"{claim_dir}/{paper_id}.txt"
        if os.path.exists(file_path):
            continue
        with open(file_path, 'w') as f:
            f.write(body)

def main():
    claim2id = Claim2idConverter()
    meta_info = load_meta_info()

    # iterate over 140 different claims with tqdm
    for claim, meta_info_list in tqdm(list(meta_info.items()), desc="Processing claims"):
        claim_id = claim2id(claim)
        claim_dir = f"{SAVE_PATH}/{claim_id}"

        # Skip processing if claim directory already exists and is non-empty
        if os.path.exists(claim_dir) and os.listdir(claim_dir):
            print(f"Skipping claim_id {claim_id}, already processed.")
            continue

        results = []

        # shuffle the meta_info_list
        random.shuffle(meta_info_list)

        # iterate over meta_info_list with tqdm
        for meta_info in tqdm(meta_info_list, desc=f"Processing meta info for claim_id {claim_id}"):
            paper_id = meta_info['paperId']
            open_access_pdf = meta_info['openAccessPdf']
            if not open_access_pdf:
                continue
            pdf_url = open_access_pdf['url']

            # put the paper into the results
            try:
                paper_content = parse_paper(pdf_url)
                instance = {
                    "paperId": paper_id,
                    "body": paper_content
                }
                results.append(instance)

                # Save incrementally
                save_results_for_claim(claim_id, [instance])

                if len(results) >= MAX_PAPER_NUM:
                    break

            # skip this case if the url is not working
            except Exception as e:
                continue

if __name__ == '__main__':
    main()
