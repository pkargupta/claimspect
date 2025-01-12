from sklearn.metrics.pairwise import cosine_similarity
from api.openai.embed import openai_embed
from api.openai.chat import chat


""" 
Step II: Keyword Generation
"""

"""
Given the claim, the aspect and all the corpus segments, 
we want to get the keywords under this aspect as a refinement, 
so that we can do good corpus segment ranking.
"""

def get_embedding_function(embedding_model_name: str):
    """
    Get the embedding function based on the model name.
    """
    if embedding_model_name == "text-embedding-3-large":
        return openai_embed
    else:
        raise ValueError(f"Embedding model {embedding_model_name} not supported.")


def get_chat_function(chat_model_name: str):
    """
    Get the chat function based on the model name.
    """
    if chat_model_name == "gpt-4o":
        def chat_gpt_4o(prompts: list[str]) -> list[str]:
            return chat(prompts, model_name='gpt-4o', seed=42, temperature=0.7)
        return chat_gpt_4o
    elif chat_model_name == "gpt-4o-mini":
        def chat_gpt_4o_mini(prompts: list[str]) -> list[str]:
            return chat(prompts, model_name='gpt-4o-mini', seed=42, temperature=0.7)
        return chat_gpt_4o_mini
    else:
        raise ValueError(f"Chat model {chat_model_name} not supported.")


def stage1_retrieve_top_k_corpus_segments(claim: str, 
                                         aspect_name: str, 
                                         corpus_segments: list[str], 
                                         retrieved_corpus_num: int, 
                                         embedding_model_name: str,
                                         current_keyword_group: list[str],) -> list[str]:
    """
    Retrieve top-k the corpus segments relevant to a given aspect of the claim.
    """
    
    # get the embedding of query and corpus segments
    embedding_func = get_embedding_function(embedding_model_name)
    retrieval_query = f"Claim: {claim} Aspect: {aspect_name} Aspect Keywords: {', '.join(current_keyword_group)}"
    retrieval_query_embedding_dict = embedding_func([retrieval_query])
    corpus_segments_embeddings_dict = embedding_func(corpus_segments)
    
    # pick up the top-k corpus segments from cosine similarity
    corpus_segments_scores = {}
    retrieval_query_embedding = retrieval_query_embedding_dict[retrieval_query]
    for segment, embedding in corpus_segments_embeddings_dict.items():
        similarity_score = cosine_similarity([retrieval_query_embedding], [embedding])[0][0]
        corpus_segments_scores[segment] = similarity_score
    
    top_k_corpus_segments = sorted(corpus_segments_scores, key=corpus_segments_scores.get, reverse=True)[:retrieved_corpus_num]
    return top_k_corpus_segments


def extract_keyword(claim: str, 
                    aspect_name: str, 
                    aspect_keywords_from_llm: list[str],
                    corpus_segments: list[str],
                    retrieved_corpus_num: int,
                    embedding_model_name: str,
                    chat_model_name: str,
                    min_keyword_num: int,
                    max_keyword_num: int,
                    iteration_num: int=1) -> list[str]:
    """
    Generate keywords for the given aspect in the claim.
    """
    current_keyword_group = aspect_keywords_from_llm
    
    for i in range(iteration_num):
        
        """ Stage 1: Retrieve top-k the corpus segments relevant to a given aspect of the claim """
        top_k_corpus_segments = stage1_retrieve_top_k_corpus_segments(claim, aspect_name, corpus_segments, retrieved_corpus_num, embedding_model_name, current_keyword_group=current_keyword_group)

        """ Stage 2: Extract keywords from the top-k corpus segments """
        chat_func = get_chat_function(chat_model_name)
        prompt = f"We are discussing the claim {claim} with a focus on the aspect {aspect_name}. Please extract up to {2*max_keyword_num} keywords related to the aspect {aspect_name} from the following documents: {'\n\n'.join(top_k_corpus_segments)}. Ensure that the extracted keywords are diverse, specific, and highly relevant to the given aspect. Only output the keywords and seperate them with comma. "
        chat_response = chat_func([prompt])
        keyword_candidates = [kw.strip() for kw in chat_response[0].split(",")]
        
        """ Stage 3: Fusion and Filtering """
        prompt = f"Based on the claim '{claim}' and the target aspect '{aspect_name}', identify {min_keyword_num} to {max_keyword_num} relevant keywords from the provided list: {keyword_candidates}. Merge terms with similar meanings, exclude relatively irrelevant ones, and output only the final keywords separated by commas."    
        chat_response = chat_func([prompt])
        current_keyword_group = [kw.strip() for kw in chat_response[0].split(",")]
    
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