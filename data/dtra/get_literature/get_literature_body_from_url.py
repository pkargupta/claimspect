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

def main():
    
    claim2id = Claim2idConverter()
    meta_info = load_meta_info()
    results = {}
    
    # iterate over 140 different claims with tqdm
    for claim, meta_info_list in tqdm(list(meta_info.items()), desc="retrieving for claim ..."):
        claim_id = claim2id(claim)
        results[claim_id] = []
        
        # shuffle the meta_info_list
        random.shuffle(meta_info_list)
        
        # iterative over meta_info_list with tqdm
        for meta_info in tqdm(meta_info_list, desc="retrieving for meta info ..."):
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
                results[claim_id].append(instance)
                if len(results[claim_id]) >= MAX_PAPER_NUM:
                    break
            
            # skip this case if the url is not working
            except Exception as e:
                continue
    
    # save the results
    for claim_id in results:
        os.makedirs(f"{SAVE_PATH}/{claim_id}", exist_ok=True)
        for instance in results[claim_id]:
            paper_id = instance['paperId']
            body = instance['body']
            with open(f"{SAVE_PATH}/{claim_id}/{paper_id}.txt", 'w') as f:
                f.write(body)
        
    

    

if __name__ == '__main__':
    main()