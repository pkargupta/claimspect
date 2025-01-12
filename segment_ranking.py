from api.local.e5_model import e5_embed
from api.openai.embed import openai_embed
from scipy.spatial.distance import cosine
import numpy as np

# calcualte the cosine similarity between two embeddings (np.array)
def cosine_similarity(embed1, embed2):
    return 1 - cosine(embed1, embed2)

def positive_rank(target_aspect, segment_embs, embed_func):
    # Query: We assume that the segment is relevant to the target_aspect's parent aspect, so we do not need to include this within the query
    keyword_embs = []
    for keyword in target_aspect.keywords:
        query = f"{keyword} with respect to {target_aspect.name}"
        keyword_embs.append(embed_func([query])[query])

    keyword_embs = np.array(keyword_embs)
    target_similarity = cosine_similarity(segment_embs, keyword_embs) # S x K

    # the more number of subaspects/keywords discussed, the better
    mean_pos = target_similarity.mean(axis=1) # S x 1
    # S x 1
    return mean_pos

def negative_rank(neg_aspects, segment_embs, embed_func, breadth_weight=0.5):
    # Query: We assume that the segment is relevant to the neg_aspects' parent aspect, so we do not need to include this within the query
    # penalize both breadth of neg_aspects discussed (mean of mean) AND depth of neg_aspects discussed (max of mean overall)

    aspect_sims = []
    for aspect in neg_aspects:
        keyword_embs = []
        for keyword in aspect.keywords:
            query = f"{keyword} with respect to {aspect.name}"
            keyword_embs.append(embed_func([query])[query])
    
        keyword_embs = np.array(keyword_embs)
        neg_aspect_sim = cosine_similarity(segment_embs, keyword_embs) # S x K
    
        # the more number of subaspects/keywords discussed, the worse -> 
        mean_neg = neg_aspect_sim.mean(axis=1) # S x 1
        aspect_sims.append(mean_neg)
    
    aspect_sims = np.array(aspect_sims) # S x A
    
    breadth_rank = aspect_sims.mean(axis=1)
    depth_rank = aspect_sims.max(axis=1)
    
    neg_rank = (breadth_weight * breadth_rank) + ((1-breadth_weight) * depth_rank)
    # S x 1
    return neg_rank

def discriminative_rank(target_aspect, neg_aspects, segment_embs, embed_func, beta=1, gamma=2):
    
    # Reward chunks that discuss the target aspect (S x 1)
    print(f"Computing positive rank for {target_aspect.name}")
    pos_r = positive_rank(target_aspect, segment_embs, embed_func)
    
    # Penalize chunks that discuss the distractor aspects (S x 1)
    print(f"Computing negative rank for {target_aspect.name}")
    neg_r = negative_rank(neg_aspects, segment_embs, embed_func)
    
    return (pos_r * beta)/(neg_r * gamma)

def aspect_segment_ranking(segments, target_aspect, neg_aspects, embed_func=e5_embed):
    """Step 3: Rank corpus segments based on relevance to keywords."""
    
    # we are by default, provided segments that are relevant to a given aspect
    # however, which are most likely to contain all relevant subaspects? we assume that the keywords cover many subaspects
    # we also want to penalize the segments which discuss a multitude of other aspects, given that they may distract during subaspect discovery + perspective

    segment_embs = embed_func([segments])[segments]
    segment_scores = discriminative_rank(target_aspect, neg_aspects, segment_embs, embed_func)

    segment_ranks = sorted(np.arange(len(segments)), key=lambda x: -segment_scores[x])

    id2rank = {}
    rank2id = {}
    for rank, seg_id in enumerate(segment_ranks):
        id2rank[seg_id] = rank
        rank2id[rank] = seg_id
    
    return rank2id, id2rank
    