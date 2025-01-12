import os
from openai import OpenAI
from joblib import Memory

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
memory = Memory(".cache", verbose=0)

@memory.cache
def openai_embed(text_list: list[str], model="text-embedding-3-large"):
    res = {}
    for text in text_list:
        text = text.replace("\n", " ")
        res[text] = client.embeddings.create(input = [text], model=model).data[0].embedding
    return res