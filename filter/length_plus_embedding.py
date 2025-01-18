from filter.abstract import AbstractFilter
from api.local.e5_model import E5
from api.local.e5_model import e5_embed
from api.openai.embed import embed as openai_embed
from sklearn.metrics.pairwise import cosine_similarity

class LengthPlusEmbeddingFilter(AbstractFilter):
    
    def __init__(self, 
                 embedding_top_k: int = 100,
                 char_length_threshold: int = 500,
                 embedding_model_name: str = 'e5'):
        # Initialize filter parameters
        self.embedding_top_k = embedding_top_k
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

    def length_sub_filter(self, list_of_segments: list[str]) -> list[str]:
        # Filter segments based on the character length threshold
        return [segment for segment in list_of_segments if len(segment) > self.char_length_threshold]
    
    def calculate_cosine_similarity(self, embed1, embed2):
        # Calculate cosine similarity between two embeddings
        return cosine_similarity([embed1], [embed2])[0][0]

    def embedding_sub_filter(self, claim: str, list_of_segments: list[str]) -> list[str]:
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
               **kwargs) -> list[str]:
        # Perform length-based filtering
        length_filtered_segments = self.length_sub_filter(list_of_segments)
        # Perform embedding-based filtering
        embedding_filtered_segments = self.embedding_sub_filter(claim, length_filtered_segments)
        # Return the final filtered list of segments
        return embedding_filtered_segments

if __name__ == "__main__":
    
    # Example usage of the LengthPlusEmbeddingFilter
    import json
    claim = "Pfizer vaccine is better than Moderna vaccine."
    
    # load the segments from example_filter_input.json
    with open("filter/example_data/filter_input.json", "r") as f:
        segments = json.load(f)
    
    # Initialize the LengthPlusEmbeddingFilter
    filter_obj = LengthPlusEmbeddingFilter()
    
    # Filter the segments based on the claim
    filtered_segments = filter_obj.filter(claim, segments)
    
    # save the filtered segments to example_filter_output.json
    output_path = "filter/example_data/filter_output_length_plus_embedding.json"
    with open(output_path, "w") as f:
        json.dump(filtered_segments, f, indent=4)