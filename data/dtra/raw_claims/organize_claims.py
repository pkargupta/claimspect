import os
import json

save_path = "data/dtra/claims.json"

def get_organized_claims(
    tgt_dir: str="data/dtra/raw_claims"
    ) -> list[str]:
    # get all the json file under tgt_dir
    # read the json file and get the claim
    # return a list of claims
    id = 0
    claims = []
    for file in os.listdir(tgt_dir):
        if file.endswith('.json'):
            with open(os.path.join(tgt_dir, file), 'r') as f:
                data = json.load(f)
                for claim in data:
                    instance = {}
                    instance['id'] = id
                    instance['body'] = claim
                    instance['source'] = file.split('.')[0]
                    claims.append(instance)
                    id += 1
    return claims

def main():
    claims = get_organized_claims()
    with open(save_path, 'w') as f:
        json.dump(claims, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()