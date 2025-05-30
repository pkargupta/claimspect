from src.filter.abstract import AbstractFilter
from src.api.local.e5_model import E5
import re
import numpy as np
from src.api.local.e5_model import e5_embed
from src.api.openai.embed import embed as openai_embed
from sklearn.metrics.pairwise import cosine_similarity

class KeywordEnsembleEmbeddingFilter(AbstractFilter):
    
    def __init__(self, 
                 minimal_occurrences: int = 3,  # Minimal number of occurrences of a keyword for a segment to be kept
                 embedding_top_k: int = 100,
                 char_length_threshold: int = 500,
                 embedding_model_name: str = 'e5',
                 weight_keywords: float = 0.5):

        self.weight_keywords = weight_keywords
        self.weight_claim = 1 - self.weight_keywords
        
        # Initialize filter parameters
        self.embedding_top_k = embedding_top_k
        self.minimal_occurrences = minimal_occurrences
        self.char_length_threshold = char_length_threshold
        # Initialize the embedding model based on the given name
        self.embedding_model = self.get_embedding_model(embedding_model_name)
        # Initialize the embedding function based on the given name
        self.embedding_func = self.get_embedding_func(embedding_model_name)
    
    def get_embedding_model(self, model_name: str):
        # Return the embedding model based on the model name
        if model_name == 'e5':
            return E5()  # Use the E5 model
        elif model_name == 'openai':
            return None  # OpenAI model is not defined here
        else:
            # Raise an error for invalid model names
            raise ValueError(f'Invalid embedding model name: {model_name}')
    
    def calculate_cosine_similarity(self, embed1, embed2):
        # Calculate cosine similarity between two embeddings
        return cosine_similarity([embed1], [embed2])[0][0]
    
    def get_embedding_func(self, model_name: str):
        # Return the embedding function based on the model name
        if model_name == 'e5':
            # Define a function that uses e5_embed with the E5 model
            def new_e5_embed(input_strs: list[str]): 
                return e5_embed(self.embedding_model, input_strs)
            return new_e5_embed
        elif model_name == 'openai':
            return openai_embed  # Use the OpenAI embedding function
        else:
            # Raise an error for invalid model names
            raise ValueError(f'Invalid embedding model name: {model_name}')
    
    def length_sub_filter(self, list_of_segments: list[str]) -> list[str]:
        # Filter segments based on the character length threshold
        return [segment for segment in list_of_segments if len(segment) > self.char_length_threshold]


    def keyword_ensemble_embedding_sub_filter(self, claim: str, keyword_list: list[str], list_of_segments: list[str]) -> list[str]:
        
        # Filter segments based on their embedding similarity to the claim
        keyword_query_list = [f"{keyword} with respect to {claim}" for keyword in keyword_list]
        keyword_embeddings_dict = self.embedding_func(keyword_query_list)
        average_keyword_embedding = np.array(sum(keyword_embeddings_dict.values()) / len(keyword_embeddings_dict))
        claim_embedding = np.array(self.embedding_func([claim])[claim])  # Embed the claim
        weighted_claim_embedding = self.weight_claim * claim_embedding + self.weight_keywords * average_keyword_embedding
        
        segment_embeddings = self.embedding_func(list_of_segments)  # Embed the segments
        top_k_segments = []
        # Calculate similarity for each segment embedding
        for segment, segment_embedding in segment_embeddings.items():
            similarity = self.calculate_cosine_similarity(weighted_claim_embedding, segment_embedding)
            top_k_segments.append((segment, similarity))
        # Sort segments by similarity and take the top K
        top_k_segments = sorted(top_k_segments, key=lambda x: x[1], reverse=True)[:self.embedding_top_k]
        return [segment for segment, similarity in top_k_segments]
    
    def filter(self, 
               claim: str, 
               list_of_segments: list[str], 
               keyword_list: list[str],
               **kwargs) -> list[str]:
        # Perform length-based filtering
        length_filtered_segments = self.length_sub_filter(list_of_segments)
        # Perform embedding-based filtering
        embedding_filtered_segments = self.keyword_ensemble_embedding_sub_filter(claim, keyword_list, length_filtered_segments)
        # Return the final filtered list of segments
        return embedding_filtered_segments

if __name__ == "__main__":
    
    # Example usage of the KeywordPlusEmbeddingFilter
    import json
    claim = "Pfizer COVID-19 vaccine is better than Moderna COVID-19 vaccine."
    
    # load the keywords
    keyword_path = "filter/example_data/vaccine_keyword_prompt_4o.json"
    with open(keyword_path, "r") as f:
        keyword_list = json.load(f)
    
    # load the segments from example_filter_input.json
    with open("filter/example_data/filter_input.json", "r") as f:
        segments = json.load(f)
    
    # Initialize the KeywordPlusEmbeddingFilter
    # filter_obj = KeywordEnsembleEmbeddingFilter(embedding_top_k=50)
    filter_obj = KeywordEnsembleEmbeddingFilter(embedding_top_k=100)
    
    # Filter the segments based on the claim
    filtered_segments = filter_obj.filter(claim, segments, keyword_list)
    
    # save the filtered segments to example_filter_output.json
    # output_path = "filter/example_data/filter_output_keyword_ensemble_embedding_top50.json"
    output_path = "filter/example_data/filter_output_keyword_ensemble_embedding_top100.json"
    
    with open(output_path, "w") as f:
        json.dump(filtered_segments, f, indent=4)