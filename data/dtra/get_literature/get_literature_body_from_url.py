import os
import time
import json
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
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

def save_results_for_claim(claim_id, paper_id, body):
    claim_dir = os.path.join(SAVE_PATH, str(claim_id))
    os.makedirs(claim_dir, exist_ok=True)
    file_path = os.path.join(claim_dir, f"{paper_id}.txt")
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write(body)

def process_pdf(args):
    
    paper_id, pdf_url, claim_id = args
    claim_dir = os.path.join(SAVE_PATH, str(claim_id))
    os.makedirs(claim_dir, exist_ok=True)

    file_path = os.path.join(claim_dir, f"{paper_id}.txt")

    if os.path.exists(file_path):
        return None
    
    try:
        paper_content = parse_paper(pdf_url)
        with open(file_path, 'w') as f:
            f.write(paper_content)
        return paper_id
    except Exception as e:

        return None

def main():
    claim2id = Claim2idConverter()
    meta_info = load_meta_info()

    all_claims = list(meta_info.items())

    for claim, meta_info_list in tqdm(all_claims, desc="Processing claims"):
        claim_id = claim2id(claim)
        claim_dir = os.path.join(SAVE_PATH, str(claim_id))

        if os.path.exists(claim_dir) and os.listdir(claim_dir):
            print(f"Skipping claim_id {claim_id}, already processed.")
            continue

        random.shuffle(meta_info_list)

        paper_args_list = []
        for info in meta_info_list:
            paper_id = info['paperId']
            open_access_pdf = info['openAccessPdf']
            if not open_access_pdf:
                continue
            pdf_url = open_access_pdf['url']
            paper_args_list.append((paper_id, pdf_url, claim_id))

        if not paper_args_list:
            continue

        results = []
        with Pool(processes=cpu_count()) as pool:
            for paper_id in tqdm(pool.imap_unordered(process_pdf, paper_args_list),
                                 total=len(paper_args_list),
                                 desc=f"Processing meta info for claim_id {claim_id}"):
                if paper_id is not None:
                    results.append(paper_id)
                if len(results) >= MAX_PAPER_NUM:
                    pool.terminate()
                    break

        print(f"Claim ID {claim_id} processed: {len(results)} papers saved.")

if __name__ == '__main__':
    main()
