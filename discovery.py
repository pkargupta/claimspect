from vllm import LLM
from vllm import SamplingParams
from outlines.serve.vllm import JSONLogitsProcessor
import json

class gen_subaspect_schema(BaseModel):
    subaspect_label: Annotated[str, StringConstraints(strip_whitespace=True)]
    subaspect_keywords: Annotated[str, StringConstraints(strip_whitespace=True)]
    
class subaspect_list_schema(BaseModel):
    subaspect_list : conlist(gen_subaspect_schema, min_length=2, max_length=5)

subaspect_instruction = lambda topic, k=5, n=10: f"You are an analyst that identifies the different aspects (a minimum of 2 and a maximum of {k} aspects) that scientific research papers would consider when determining the answer to the following topic: \"{topic}\". For each aspect, provide {n} additional keywords that would typically be used to describe that aspect for the topic, {topic}.\n\n"

def subaspect_prompt(topic, k=5, n=10): 
    prompt = aspect_candidate_init_prompt(topic, k, n)
    prompt += f"""For the topic, {topic}, output the list of up to {k} aspects in the following JSON format:
---
{{
    "aspect_list":
        [
            {{
                "aspect_label": <should be a brief, 5-10 word string where the value is an all-lowercase label of the aspect (phrase-length)>,
                "aspect_keywords": <list of {n} unique, all-lowercase, and comma-separated keywords (always a space after the comma) commonly used to describe the aspect_label (e.g., keyword_1, keyword_2, ..., keyword_{n})>
            }}
        ]
}}
---
"""

    return prompt

def subaspect_discovery(args, segments, rank2id, parent_aspect, top_k=10):

    subset = [segments[rank2id[i]] for i in np.arange(top_k)]

    logits_processor = JSONLogitsProcessor(schema=subaspect_list_schema, llm=args.model.llm_engine)
    sampling_params = SamplingParams(max_tokens=2000, logits_processors=[logits_processor], temperature=temperature, top_p=top_p)

    output = args.model.generate(subaspect_prompt(claim), sampling_params=sampling_params)[0].outputs[0].text

    aspects = json.loads(output)['subaspect_list']
    
    print(f"Generated coarse-grained aspects for claim '{claim}': {aspects}")
    
    return


def perspective_discovery():
    return