from filter.abstract import AbstractFilter
from api.local.e5_model import E5
import re
from api.local.e5_model import e5_embed
from api.openai.embed import embed as openai_embed
from sklearn.metrics.pairwise import cosine_similarity

class KeywordPlusEmbeddingFilter(AbstractFilter):
    
    def __init__(self, 
                 minimal_occurrences: int = 3,  # Minimal number of occurrences of a keyword for a segment to be kept
                 embedding_top_k: int = 100,
                 char_length_threshold: int = 500,
                 embedding_model_name: str = 'e5'):

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
        
    def judge_occurance(self, segment: str, keyword: str, maximal_distance: int=1) -> bool:
        # 1. Normalize case
        segment = segment.lower()
        keyword = keyword.lower().strip()
        
        # 2. Split keyword into individual words
        words = keyword.split()
        if not words:
            # Return False if the keyword is empty or contains only whitespace
            return False
        
        # 3. Construct the regular expression
        #    Example: For ["harry", "potter"] and maximal_distance = 1,
        #    the result would look like: \bharry\b(?:\s+\S+){0,1}\s+\bpotter\b
        pattern_parts = []
        for i, w in enumerate(words):
            # \b denotes a word boundary to avoid matching inside other words
            part = r"\b" + re.escape(w) + r"\b"
            pattern_parts.append(part)
            if i < len(words) - 1:
                # Allow up to maximal_distance words between adjacent words
                # (?:\s+\S+){0,maximal_distance} matches 0 to maximal_distance occurrences 
                # of "non-whitespace character groups", followed by \s+ for at least one space
                pattern_parts.append(r"(?:\s+\S+){0," + str(maximal_distance) + r"}\s+")
        
        pattern = "".join(pattern_parts)
    
        # 4. Match the constructed pattern in the segment
        return re.search(pattern, segment) is not None
        
    def judge_segment(self, segment: str, keyword_list: list[str]) -> bool:
        occurrences = 0
        # Count the number of occurrences of each keyword in the segment
        for keyword in keyword_list:
            if self.judge_occurance(segment, keyword):
                occurrences += 1
        # Return True if the number of occurrences is above the threshold
        return occurrences >= self.minimal_occurrences
    
    def length_sub_filter(self, list_of_segments: list[str]) -> list[str]:
        # Filter segments based on the character length threshold
        return [segment for segment in list_of_segments if len(segment) > self.char_length_threshold]

    def keyword_sub_filter(self, list_of_segments: list[str], keyword_list: list[str]) -> list[str]:

        filtered_segments = []
        # Iterate through each segment
        for segment in list_of_segments:
            if self.judge_segment(segment, keyword_list):
                filtered_segments.append(segment)
        return filtered_segments
            
    def calculate_cosine_similarity(self, embed1, embed2):
        # Calculate cosine similarity between two embeddings
        return cosine_similarity([embed1], [embed2])[0][0]

    def embedding_sub_filter(self, claim: str, list_of_segments: list[str]) -> list[str]:
        # if the list of segments is smaller than the top k, return the list
        if len(list_of_segments) <= self.embedding_top_k:
            print(f"Number of segments is smaller than the top k ({len(list_of_segments)} <= {self.embedding_top_k}).")
            print("No embedding filtering will be performed.")
            return list_of_segments
        
        # Filter segments based on their embedding similarity to the claim
        claim_embedding = self.embedding_func([claim])[claim]  # Embed the claim
        segment_embeddings = self.embedding_func(list_of_segments)  # Embed the segments
        top_k_segments = []
        # Calculate similarity for each segment embedding
        for segment, segment_embedding in segment_embeddings.items():
            similarity = self.calculate_cosine_similarity(claim_embedding, segment_embedding)
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
        # Perform keyword-based filtering
        keyword_filtered_segments = self.keyword_sub_filter(length_filtered_segments, keyword_list)
        # Perform embedding-based filtering
        embedding_filtered_segments = self.embedding_sub_filter(claim, keyword_filtered_segments)
        # Return the final filtered list of segments
        return embedding_filtered_segments

if __name__ == "__main__":
    
    # Example usage of the KeywordPlusEmbeddingFilter
    import json
    claim = "Pfizer vaccine is better than Moderna vaccine."
    
    # load the keywords
    keyword_path = "filter/example_data/vaccine_keyword_prompt_4o.json"
    with open(keyword_path, "r") as f:
        keyword_list = json.load(f)
    
    # load the segments from example_filter_input.json
    with open("filter/example_data/filter_input.json", "r") as f:
        segments = json.load(f)
    
    # Initialize the KeywordPlusEmbeddingFilter
    filter_obj = KeywordPlusEmbeddingFilter()
    
    # Filter the segments based on the claim
    filtered_segments = filter_obj.filter(claim, segments, keyword_list)
    
    # save the filtered segments to example_filter_output.json
    output_path = "filter/example_data/filter_output_keyword_plus_embedding.json"
    with open(output_path, "w") as f:
        json.dump(filtered_segments, f, indent=4)