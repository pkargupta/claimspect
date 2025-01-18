from abstract import AbstractFilter

class LengthPlusEmbeddingFilter(AbstractFilter):
    
    def __init__(self, 
                 embedding_top_k: int = 100,
                 char_length_threshold: int = 500,
                 embedding_model: str = 'bert-base-uncased'):
        
        self.embedding_top_k = embedding_top_k
        self.char_length_threshold = char_length_threshold

    def length_sub_filter(self, list_of_segments: list[str]) -> list[str]:
        return [segment for segment in list_of_segments if len(segment) > self.char_length_threshold]
    
    def 