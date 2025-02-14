from openai import OpenAI
from tqdm import tqdm

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def llama_chat(prompt_list: list[str]) -> list[str]:
    
    results = []
    for prompt in tqdm(prompt_list, leave=False):
        completion = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            messages=[
                {"role": "developer", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        temperature=0.7)
        completion_text = completion.choices[0].message.content
        results.append(completion_text)
    return results

if __name__ == '__main__':
    
    prompt_list = ["Translate this into English: 'Je suis un garçon.' Only output the English translation. And do not output anything else", "Tell me a interesting joke."]
    print(llama_chat(prompt_list))
    