import os
import json
from api.openai.chat import chat

CLAIM_GENERATION_PROMPT = """
Scientific or political claims are often nuanced and multifaceted, rarely lending themselves to simple “yes” or “no” answers. To answer such questions effectively, claims must be broken into specific aspects for in-depth analysis, with evidence drawn from relevant scientific literature. We are currently studying such claims using this corpus: 
{context}
Task: Generate 20 nuanced and diverse claims based on this corpus. The claims should adhere to the following criteria:
	1.	Diversity: The claims should be sufficiently varied: (i) they should involve diverse sub-topics in the context; (ii) they should come from different perspectives (not always from the author's side).
	2.	Complexity: The claims should be complex and controversial (and not necessarity true), requiring multi-aspect analysis rather than simplistic treatment. Avoid overly straightforward or simplistic claims.
	3.	Research Feasibility: The claims should not be too specific and should pertain to topics with a likely body of existing literature to support evidence-based exploration.
    4.  Concision: The claims should be concise and focused in one short sentence.
    5.  Completeness: The claims should be complete and not require additional context to understand.
Output: Provide the claims as a list."""

ORGANIZE_LIST_PROMPT = """
Please organize the following list of claims into a python list (without the index) like ["claim1", "claim2", "claim3", ...]. You must not generate other content like ```python sign.
{claim_list}"""

def get_txt_paths(
    target_dir: str = "datasets/dtra/factsheet_txt"
):
    return [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith('.txt')]

def load_txt_files(txt_paths):
    txt_list = []
    for txt_path in txt_paths:
        with open(txt_path, 'r') as f:
            txt_list.append(f.read())
    return txt_list

def get_json_paths(txt_paths):
    results = []
    for txt_path in txt_paths:
        results.append(txt_path.replace('factsheet_txt', 'raw_claims').replace('.txt', '.json'))
    return results

def main():
    
    txt_paths = get_txt_paths()
    output_paths = get_json_paths(txt_paths)
    
    txt_list = load_txt_files(txt_paths)
    raw_responses = chat([CLAIM_GENERATION_PROMPT.format(context=txt_str) for txt_str in txt_list], model_name='gpt-4o',seed=42)
    
    organized_responses = chat([ORGANIZE_LIST_PROMPT.format(claim_list=raw_response) for raw_response in raw_responses], model_name='gpt-4o-mini',seed=42)
    # convert organized_responses to json
    organized_json = []
    for organized_response in organized_responses:
        organized_json.append(json.loads(organized_response))
    
    for output_path, organized_response in zip(output_paths, organized_json):
        with open(output_path, 'w') as f:
            json.dump(organized_response, f, indent=4)

if __name__ == '__main__':
    main()