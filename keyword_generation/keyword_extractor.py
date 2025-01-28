from sklearn.metrics.pairwise import cosine_similarity
from api.openai.embed import embed as openai_embed
from api.openai.chat import chat
from api.local.e5_model import e5_embed
from hierarchy import Segment
from prompts import keyword_extraction_prompt, keyword_filter_prompt

from vllm import SamplingParams
from outlines.serve.vllm import JSONLogitsProcessor
from pydantic import BaseModel, StringConstraints, conlist
from typing_extensions import Annotated
import json
import numpy as np

""" 
Step II: Keyword Generation
"""

"""
Given the claim, the aspect and all the corpus segments, 
we want to get the keywords under this aspect as a refinement, 
so that we can do good corpus segment ranking.
"""



def get_embedding_function(args):
    """
    Get the embedding function based on the model name.
    """
    if args.embedding_model_name == "text-embedding-3-large":
        return openai_embed
    elif args.embedding_model_name == "e5":
        return e5_embed
    else:
        raise ValueError(f"Embedding model {args.embedding_model_name} not supported.")

class keyword_schema(BaseModel):
    output_keywords : Annotated[str, StringConstraints(strip_whitespace=True)]

def get_chat_function(args):
    """
    Get the chat function based on the model name.
    """
    if args.chat_model_name == "gpt-4o":
        def chat_gpt_4o(prompts: list[str], batch=True) -> list[str]:
            return chat(prompts, model_name='gpt-4o', seed=42, temperature=0.7)
        return chat_gpt_4o

    elif args.chat_model_name == "gpt-4o-mini":
        def chat_gpt_4o_mini(prompts: list[str], batch=True) -> list[str]:
            return chat(prompts, model_name='gpt-4o-mini', seed=42, temperature=0.7)
        return chat_gpt_4o_mini
    
    elif args.chat_model_name == "vllm":
        def vllm(prompts: list[str], batch=False) -> list[str]:
            logits_processor = JSONLogitsProcessor(schema=keyword_schema, llm=args.chat_model.llm_engine)
            sampling_params = SamplingParams(max_tokens=2000, logits_processors=[logits_processor], seed=42, temperature=0.7)
            if batch:
                outputs = args.chat_model.generate(prompts, sampling_params=sampling_params)
                return [json.loads(o.outputs[0].text)['output_keywords'] for o in outputs]
            else:
                outputs = args.chat_model.generate(prompts, sampling_params=sampling_params)[0].outputs[0].text
                keywords = json.loads(outputs)['output_keywords']
                return [keywords]
            
        return vllm
    else:
        raise ValueError(f"Chat model {args.chat_model_name} not supported.")


def stage1_retrieve_top_k_corpus_segments(args, 
                                          claim: str, 
                                          aspect_name: str,
                                          aspect_description: str,
                                          corpus_segments: list[Segment], 
                                          retrieved_corpus_num: int, 
                                          current_keyword_group: list[str],
                                          corpus_embs=None) -> list[str]:
    """
    Retrieve top-k the corpus segments relevant to a given aspect of the claim.
    """
    
    # get the embedding of query and corpus segments
    embedding_func = args.embed_func
    retrieval_query = f"Claim: {claim} Aspect: {aspect_name}: {aspect_description}; Aspect Keywords: {','.join(current_keyword_group)}"
    query_emb = embedding_func(text=[retrieval_query], model=args.embed_model)[retrieval_query]
    
    if corpus_embs is None:
        seg_contents = [seg.content for seg in corpus_segments]
        corpus_segments_embeddings_dict = embedding_func(text=seg_contents, model=args.embed_model)
    else:
        corpus_segments_embeddings_dict = corpus_embs

    seg_embs = np.stack([corpus_segments_embeddings_dict[seg.content] for seg in corpus_segments], axis=0)
    seg_embs = seg_embs.reshape((len(corpus_segments), -1))
    
    cosine_sim = cosine_similarity(seg_embs, [query_emb]).reshape((len(seg_embs), 1))
    
    segment_ranks = sorted(np.arange(len(corpus_segments)), key=lambda x: -cosine_sim[x][0])
    top_k_segments = [corpus_segments[i] for i in segment_ranks[:retrieved_corpus_num]]
    
    return top_k_segments


def extract_keyword(args, claim: str, aspect_name: str, aspect_description: str, aspect_keywords_from_llm: list[str],
                    corpus_segments: list[str], retrieved_corpus_num: int, min_keyword_num: int, max_keyword_num: int,
                    iteration_num: int=1, corpus_embs=None) -> list[str]:
    """
    Generate keywords for the given aspect in the claim.
    """
    current_keyword_group = aspect_keywords_from_llm
    
    for i in range(iteration_num):
        
        """ Stage 1: Retrieve top-k the corpus segments relevant to a given aspect of the claim """
        top_k_corpus_segments = stage1_retrieve_top_k_corpus_segments(args, claim, aspect_name, aspect_description, corpus_segments, retrieved_corpus_num, current_keyword_group, corpus_embs)
        seg_contents = [seg.content for seg in top_k_corpus_segments]

        """ Stage 2: Extract keywords from the top-k corpus segments """
        chat_func = get_chat_function(args)
        prompt = keyword_extraction_prompt(claim, aspect_name, aspect_description, max_keyword_num, seg_contents)
        chat_response = chat_func([prompt])
        
        if type(chat_response[0]) == str:
            keyword_candidates = [kw.strip().lower() for kw in chat_response[0].split(", ")]
        else:
            keyword_candidates = [kw.strip().lower() for kw in chat_response[0]]
        
        """ Stage 3: Fusion and Filtering """
        prompt = keyword_filter_prompt(claim, aspect_name, aspect_description, 
                                       min_keyword_num, max_keyword_num, keyword_candidates)    
        chat_response = chat_func([prompt])
        if type(chat_response[0]) == str:
            current_keyword_group = [kw.strip().lower() for kw in chat_response[0].split(", ")]
        else:
            current_keyword_group = [kw.strip().lower() for kw in chat_response[0]]
    
    return current_keyword_group
    
def extract_keywords(args, claim: str, aspects: list[dict], corpus_segments: list[str], retrieved_corpus_num: int,
                    min_keyword_num: int, max_keyword_num: int, iteration_num: int=1, corpus_embs=None, is_subaspect=False) -> list[str]:
    
    current_keyword_group = [aspect["subaspect_keywords"] if is_subaspect else aspect["aspect_keywords"] for aspect in aspects]

    for i in range(iteration_num):

        aspect_prompts = []
        aspect_names = []
        aspect_descriptions = []

        for idx, aspect in enumerate(aspects):
            aspect_names.append(aspect["subaspect_label"] if is_subaspect else aspect["aspect_label"])
            aspect_descriptions.append(aspect["subaspect_description"] if is_subaspect else aspect["aspect_description"])
            
            """ Stage 1: Retrieve top-k the corpus segments relevant to a given aspect of the claim """
            top_k_segments = stage1_retrieve_top_k_corpus_segments(args, claim, aspect_names[idx], aspect_descriptions[idx],
                                                                   corpus_segments, retrieved_corpus_num,
                                                                   current_keyword_group[idx], corpus_embs)
            seg_contents = [seg.content for seg in top_k_segments]
            prompt = keyword_extraction_prompt(claim, aspect_names[idx], aspect_descriptions[idx], max_keyword_num, seg_contents)
            aspect_prompts.append(prompt)

        """ Stage 2: Extract keywords from the top-k corpus segments """
        chat_func = get_chat_function(args)
        chat_responses = chat_func(aspect_prompts, batch=True)

        aspect_keyword_prompts = []
        for idx, chat_response in enumerate(chat_responses):
            
            if '```json' in chat_response:
                # remove the prefix and suffix
                chat_response = chat_response[7:-3]
                chat_response = json.loads(chat_response)['output_keywords']
                keyword_candidates = [kw.strip().lower() for kw in chat_response]
            
            else:  
                if type(chat_response) == str:
                    keyword_candidates = [kw.strip().lower() for kw in chat_response.split(", ")]
                else:
                    keyword_candidates = [kw.strip().lower() for kw in chat_response]

            aspect_keyword_prompts.append(keyword_filter_prompt(claim, aspect_names[idx], aspect_descriptions[idx],
                                                                min_keyword_num, max_keyword_num, keyword_candidates))  

        """ Stage 3: Fusion and Filtering """
        chat_responses = chat_func(aspect_keyword_prompts, batch=True)

        current_keyword_group = []
        for idx, chat_response in enumerate(chat_responses):
            
            if '```json' in chat_response:
                # remove the prefix and suffix
                chat_response = chat_response[7:-3]
                chat_response = json.loads(chat_response)['output_keywords']
                current_keyword_group.append([kw.strip().lower() for kw in chat_response])
            
            else:
                if type(chat_response) == str:
                    current_keyword_group.append([kw.strip().lower() for kw in chat_response.split(", ")])
                else:
                    current_keyword_group.append([kw.strip().lower() for kw in chat_response])
    
    return current_keyword_group


if __name__ == "__main__":
    
    claim = "Pfizer vaccine is better than Moderna."
    aspect_name = "Safety"
    aspect_keywords_from_llm = ["vaccine side effects", "safety profile", "adverse reactions"]
    corpus_segments = [
        "Pfizer's mRNA vaccine (Comirnaty) and Moderna's mRNA vaccine (Spikevax) have both undergone rigorous clinical trials demonstrating high levels of efficacy and safety. However, subtle differences exist. For example, Moderna's vaccine typically uses a higher mRNA dose (100 µg compared to Pfizer’s 30 µg), which may contribute to reports of slightly stronger immune responses but also a marginally higher incidence of transient side effects like fever, fatigue, and muscle pain.",
        "Rare adverse events, such as myocarditis and pericarditis, have been reported for both vaccines, primarily in younger males. Studies suggest the incidence may be slightly higher with Moderna, possibly due to the higher mRNA content, though both vaccines remain safe and effective according to regulatory bodies.",
        "Both vaccines have robust safety monitoring systems in place. As of recent data, no long-term safety concerns unique to Pfizer or Moderna have been definitively identified, but ongoing monitoring is critical.",
        "Pfizer initially required ultra-cold storage (-70°C), which posed logistical challenges, potentially impacting the vaccine's safety if storage guidelines were not strictly followed. Moderna's vaccine is more stable at standard freezer temperatures (-20°C), making it easier to handle in diverse settings.",
        "Pfizer’s vaccine is generally priced slightly lower per dose than Moderna’s in many markets, which can make it a more cost-effective choice for governments and health organizations purchasing at scale. However, prices vary by country, contracts, and purchase agreements.",
        "Both vaccines require two doses for the primary series, and booster recommendations are similar. Pfizer’s lower dose formulation might lead to lower material costs per dose, offering an economic advantage, especially in large vaccination campaigns.",
        "Moderna’s vaccine has more lenient storage requirements, reducing logistical and distribution costs in some settings. This could make Moderna more cost-efficient in areas with limited access to ultra-cold storage facilities."
    ]

    aspect_keywords_from_corpus = extract_keyword(
        claim=claim, 
        aspect_name=aspect_name,
        aspect_keywords_from_llm=aspect_keywords_from_llm, 
        corpus_segments=corpus_segments,
        retrieved_corpus_num=4, 
        embedding_model_name="text-embedding-3-large",
        chat_model_name="gpt-4o",
        min_keyword_num=5,
        max_keyword_num=8,
        iteration_num=1)
    
    print(aspect_keywords_from_corpus)
    # ['Safety monitoring systems', 'long-term safety concerns', 'adverse events', 'myocarditis', 'transient side effects']