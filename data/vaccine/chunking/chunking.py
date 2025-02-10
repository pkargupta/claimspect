import os
import sys
import json
import tqdm
from C99 import TopicSegmentor

PARALLEL_SIZE = 16

def get_passages(
    idx: int,
    tgt_dir='data/vaccine/get_literature/literature_body',
):
    # take only a part of the claims
    batch_size = len(list(os.listdir(tgt_dir))) // PARALLEL_SIZE + 1
    start_idx = idx * batch_size
    end_idx = start_idx + batch_size

    print(f"Reading passages from {tgt_dir} ...")
    results = {}
    # dir name is the claim id, the key
    list_dir = list(os.listdir(tgt_dir))[start_idx:end_idx]
    for dir_name in list_dir:
        # the value is a list of passages
        results[dir_name] = {}
        for file_name in os.listdir(os.path.join(tgt_dir, dir_name)):
            paper_id = file_name.split('.')[0]
            with open(os.path.join(tgt_dir, dir_name, file_name), 'r') as f:
                text = f.read()
                results[dir_name][paper_id] = text
    
    print(f"Done! {len(results)} claims found.")
    return results

def connect_sentences(inputs: list[str]):
    return ' '.join(inputs)

def main():
    # read in index as cmd argument
    idx = int(sys.argv[1])
    
    segmentor = TopicSegmentor()
    results = {}
    passages = get_passages(idx)
    
    for claim_id in tqdm.tqdm(passages, desc="Claims"):
        results[claim_id] = []
        
        for paper_id in tqdm.tqdm(passages[claim_id], desc=f"Papers in {claim_id}"):
            try:
                raw_segments = segmentor.segment([passages[claim_id][paper_id]], enable_tqdm=False)
            except:
                print(f"Error in {claim_id} - {paper_id}")
                continue
            segments = [connect_sentences(segment) for segment in raw_segments]
            for segment in segments:
                instance = {'segment': segment, 'paper_id': paper_id}
                results[claim_id].append(instance)
    
    with open(f'data/vaccine/chunking/shards/corpus_segments_{idx}.json', 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()