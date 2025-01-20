from filter.abstract import AbstractFilter
from api.local.e5_model import E5
import re
import numpy as np
from api.local.e5_model import e5_embed
from api.openai.chat import chat
from api.openai.embed import embed as openai_embed
from sklearn.metrics.pairwise import cosine_similarity

class KeywordEnsembleEmbeddingLLMFilter(AbstractFilter):
    
    def __init__(self, 
                 aspect_list: list[str],
                 minimal_occurrences: int = 3,  # Minimal number of occurrences of a keyword for a segment to be kept
                 char_length_threshold: int = 500,
                 embedding_model_name: str = 'e5',
                 chat_model_name: str = 'gpt-4o',
                 weight_keywords: float = 0.5,
                 boundary_density: float = 0.2,
                 density_interval: int = 9,
                 threshold_upper_bound: int = 200):

        self.weight_keywords = weight_keywords
        self.weight_claim = 1 - self.weight_keywords
        self.boundary_density = boundary_density
        self.threshold_upper_bound = threshold_upper_bound
        self.aspect_list = aspect_list
        self.density_interval = density_interval
        
        # Initialize filter parameters
        self.minimal_occurrences = minimal_occurrences
        self.char_length_threshold = char_length_threshold
        # Initialize the embedding model based on the given name
        self.embedding_model = self.get_embedding_model(embedding_model_name)
        self.chat_model_name = chat_model_name
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
    
    def llm_judge_relevance(self, segments: list[str], claim: str) -> list[bool]:
        aspects = self.aspect_list
        """ llm judge relevance of segments to claim """
        prompt_template = """I am currently analyzing a claim based on a segment from the literature from several different aspects.

The segment is: {segment}

The claim is: {claim}

The aspects are: {aspects}

Please help me determine whether this segment is related to the claim so that I can analyze this claim based on it from at least one of these aspects. Your output should be 'Yes' or 'No'. """
        prompts = [prompt_template.format(segment=segment, claim=claim, aspects=', '.join(aspects)) for segment in segments]
        if self.chat_model_name == 'gpt-4o' or self.chat_model_name == 'gpt-4o-mini':
            responses = chat(prompts,temperature=0.01, model_name=self.chat_model_name)
        else:
            raise ValueError(f'Invalid chat model name: {self.chat_model_name} not implemented yet.')

        bool_list = ['yes' in response.lower() for response in responses]
        return bool_list
    
    def get_density(self, ordered_segments, claim, ratio: float):
        
        total_length = len(ordered_segments)
        middle_index = int(total_length * ratio)
        radius = int((self.density_interval-1)/2)
        right_index = min(middle_index + radius + 1, total_length - 1)
        left_index = max(middle_index - radius, 0)
        
        target_segment_interval = ordered_segments[left_index:right_index]
        interval_relevance = self.llm_judge_relevance(target_segment_interval, claim)
        density = sum(interval_relevance) / len(interval_relevance)
        return density
        
    
    def pick_boundary(self, ordered_segments, claim):
        """
        This function performs a binary search for the ratio within [0, 1].
        We want to find the ratio for which the density is >= target_density,
        and among all valid ratios, we pick the rightmost one.
        If no valid ratio is found, we return a fallback value (left or 0, depending on your needs).
        """

        # The target density we want to reach
        target_density = self.boundary_density
        
        # Define the search boundaries
        left, right = 0.0, 1.0
        # Track the best ratio that satisfies density >= target_density
        best_ratio = None

        # max_iter = int(log(len(ordered_segments)))
        max_iter = int(np.log(len(ordered_segments))) + 1
        for _ in range(max_iter):
            mid = (left + right) / 2.0
            density = self.get_density(ordered_segments, claim, mid)
            
            # If current mid achieves or exceeds the target density,
            # record this ratio and search further to the right.
            if density >= target_density:
                best_ratio = mid
                left = mid
            else:
                # Otherwise, we need to search on the left side (lower ratios).
                right = mid
        
        # If best_ratio is still None, it means we never found a density >= target_density.
        # Decide a fallback return value based on your business requirement.
        if best_ratio is None:
            best_ratio = left

        return int(best_ratio * len(ordered_segments))
        


    def keyword_ensemble_embedding_sub_filter(self, claim: str, keyword_list: list[str], list_of_segments: list[str]) -> list[str]:
        
        # Filter segments based on their embedding similarity to the claim
        keyword_query_list = [f"{keyword} with respect to {claim}" for keyword in keyword_list]
        keyword_embeddings_dict = self.embedding_func(keyword_query_list)
        average_keyword_embedding = np.array(sum(keyword_embeddings_dict.values()) / len(keyword_embeddings_dict))
        claim_embedding = np.array(self.embedding_func([claim])[claim])  # Embed the claim
        weighted_claim_embedding = self.weight_claim * claim_embedding + self.weight_keywords * average_keyword_embedding
        
        segment_embeddings = self.embedding_func(list_of_segments)  # Embed the segments
        ordered_segments = []
        # Calculate similarity for each segment embedding
        for segment, segment_embedding in segment_embeddings.items():
            similarity = self.calculate_cosine_similarity(weighted_claim_embedding, segment_embedding)
            ordered_segments.append((segment, similarity))
        # Sort segments by similarity and take the top K
        ordered_segments = sorted(ordered_segments, key=lambda x: x[1], reverse=True)
        ordered_segments = ordered_segments[:self.threshold_upper_bound]
        
        dynamic_top_k = self.pick_boundary(ordered_segments, claim)
        print(f"Dynamic top k: {dynamic_top_k}")
        return [segment for segment, similarity in ordered_segments[:dynamic_top_k]]
    
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
    major_aspects = ['Efficacy', 'Safety', 'Cost & Availability', "Manufacturing & Transportation"]
    
    # load the keywords
    keyword_path = "filter/example_data/vaccine_keyword_prompt_4o.json"
    with open(keyword_path, "r") as f:
        keyword_list = json.load(f)
    
    # load the segments from example_filter_input.json
    with open("filter/example_data/filter_input.json", "r") as f:
        segments = json.load(f)
    
    # Initialize the KeywordPlusEmbeddingFilter
    # filter_obj = KeywordEnsembleEmbeddingFilter(boundary_density=0.5, aspect_list = major_aspects)
    # filter_obj = KeywordEnsembleEmbeddingFilter(boundary_density=0.4, aspect_list = major_aspects)
    # filter_obj = KeywordEnsembleEmbeddingFilter(boundary_density=0.3, aspect_list = major_aspects)
    # filter_obj = KeywordEnsembleEmbeddingFilter(boundary_density=0.2, aspect_list = major_aspects)
    # filter_obj = KeywordEnsembleEmbeddingFilter(boundary_density=0.1, aspect_list = major_aspects)
    filter_obj = KeywordEnsembleEmbeddingLLMFilter(boundary_density=0.05, aspect_list = major_aspects)
    
    # Filter the segments based on the claim
    filtered_segments = filter_obj.filter(claim, segments, keyword_list)
    # density = filter_obj.llm_judge_relevance(segments[:3], claim)
    
    # save the filtered segments to example_filter_output.json
    # output_path = "filter/example_data/filter_output_keyword_ensemble_embedding_llm_judge_d05.json"
    # output_path = "filter/example_data/filter_output_keyword_ensemble_embedding_llm_judge_d04.json"
    # output_path = "filter/example_data/filter_output_keyword_ensemble_embedding_llm_judge_d03.json"
    # output_path = "filter/example_data/filter_output_keyword_ensemble_embedding_llm_judge_d02.json"
    # output_path = "filter/example_data/filter_output_keyword_ensemble_embedding_llm_judge_d01.json"
    output_path = "filter/example_data/filter_output_keyword_ensemble_embedding_llm_judge_d005.json"

    with open(output_path, "w") as f:
        json.dump(filtered_segments, f, indent=4)