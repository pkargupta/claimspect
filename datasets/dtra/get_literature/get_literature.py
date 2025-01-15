import os
import tqdm
import time
import json
from api.openai.chat import chat
from api.scholar.search_by_plain_str import search_literature

def get_claims(
    tgt_dir: str="datasets/dtra/raw_claims"
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


def main():
    
    claims = get_claims()
    

if __name__ in '__main__':
    main()