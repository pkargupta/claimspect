import random
import json
from src.api.openai.embed import embed as openai_embed
# import cosine similarity function
from sklearn.metrics.pairwise import cosine_similarity

embed_model_name = "text-embedding-3-large"

# load the claim json
claim_json_path = "data/dtra/claims.json"
with open(claim_json_path, "r") as f:
    claim_json = json.load(f)
    
# load the segment json
segment_json_path = "data/dtra/segments.json"
with open(segment_json_path, "r") as f:
    segment_json = json.load(f)
    
# random sample 3 claims
random.seed(42)
sampled_claims = random.sample(claim_json, 3)

# get the corresponding segments
segment_sets = []
for claim in sampled_claims:
    claim_id = claim["id"]
    segments = segment_json[str(claim_id)]
    segment_sets.append(segments)

results = []
for claim, segments in zip(sampled_claims, segment_sets):
    
    claim_str = claim["body"]
    segments_str_list = [segment["segment"] for segment in segments]
    claim_embedding = openai_embed([claim_str], embed_model_name)[claim_str]
    segment_embeddings = openai_embed(segments_str_list, embed_model_name)
    # the results are dicts {str: vector}
    
    top_20_segments = []
    for segment_str, segment_embedding in segment_embeddings.items():

        # use cosine similarity to calculate the similarity with list of float
        similarity = cosine_similarity([claim_embedding], [segment_embedding])[0][0]
        top_20_segments.append((segment_str, similarity))
    top_20_segments = sorted(top_20_segments, key=lambda x: x[1], reverse=True)[:20]
    results.append((claim_str, top_20_segments))
    
# save the results
save_path = "data/dtra/chunking/top_embed_sim_segments_example.json"
with open(save_path, "w") as f:
    json.dump(results, f, indent=4)


