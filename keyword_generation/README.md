# Code Description

- This script generates keywords relevant to a specific aspect of a claim by analyzing a corpus of text segments. It uses embedding models to compute similarities between a query (combining the claim, aspect, and aspect keywords from common sense) and corpus segments, retrieving the most relevant ones. These segments are then processed using chat models to extract and refine aspect-specific keywords, ensuring diversity and relevance.

- To test the code, first fill in your `OPENAI_API_KEY` to the environment and then run `python -m keyword_generation.keyword_extractor`