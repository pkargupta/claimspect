import os
from openai import OpenAI
from joblib import Memory

if "OPENAI_API_KEY" in os.environ:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
else:
    client = None

memory = Memory(".cache", verbose=0)

@memory.cache
def openai_embed(embedding_model_name, text_list: list[str]):
    res = {}
    for text in text_list:
        text = text.replace("\n", " ")
        res[text] = client.embeddings.create(input = [text], model=embedding_model_name).data[0].embedding
    return res