from pydantic import BaseModel, StringConstraints, conlist
from typing_extensions import Annotated

class gen_aspect_schema(BaseModel):
    aspect_label: Annotated[str, StringConstraints(strip_whitespace=True)]
    aspect_keywords: Annotated[str, StringConstraints(strip_whitespace=True)]
    
class aspect_list_schema(BaseModel):
    aspect_list : conlist(gen_aspect_schema, min_length=2,max_length=5)


aspect_candidate_init_prompt = lambda topic, k=5, n=10: f"You are an analyst that identifies the different aspects (a minimum of 2 and a maximum of {k} aspects) that scientific research papers would consider when determining the answer to the following topic: \"{topic}\". For each aspect, provide {n} additional keywords that would typically be used to describe that aspect for the topic, {topic}.\n\n"

def aspect_prompt(topic, k=5, n=10): 
    prompt = aspect_candidate_init_prompt(topic, k, n)
    prompt += f"""For the topic, {topic}, output the list of up to {k} aspects in the following JSON format:
---
{{
    "aspect_list":
        [
            {{
                "aspect_label": <should be a brief, 5-10 word string where the value is the label of the aspect (phrase-length)>,
                "aspect_keywords": <list of {n} unique, comma-separated keywords (always a space after the comma) commonly used to describe the aspect_label (e.g., keyword_1, keyword_2, ..., keyword_{n})>
            }}
        ]
}}
---
"""

    return prompt