from pydantic import BaseModel, StringConstraints, conlist
from typing_extensions import Annotated

############ COARSE-GRAINED ASPECT DISCOVERY ############

class gen_aspect_schema(BaseModel):
    aspect_label: Annotated[str, StringConstraints(strip_whitespace=True)]
    aspect_keywords: conlist(str, min_length=2, max_length=20)
    
class aspect_list_schema(BaseModel):
    aspect_list : conlist(gen_aspect_schema, min_length=2, max_length=5)


aspect_candidate_init_prompt = lambda topic, k=5, n=10: f"You are an analyst that identifies the different aspects (a minimum of 2 and a maximum of {k} aspects) that scientific research papers would consider when determining the answer to the following topic: \"{topic}\". For each aspect, provide {n} additional keywords that would typically be used to describe that aspect for the topic, {topic}.\n\n"

def aspect_prompt(topic, k=5, n=10): 
    prompt = aspect_candidate_init_prompt(topic, k, n)
    prompt += f"""For the topic, {topic}, output the list of up to {k} aspects in the following JSON format:
---
{{
    "aspect_list":
        [
            {{
                "aspect_label": <should be a brief, 5-10 word string where the value is an all-lowercase label of the aspect (phrase-length)>,
                "aspect_keywords": <list of {n} unique, all-lowercase, and comma-separated string keywords (always a space after the comma) commonly used to describe the aspect_label (e.g., ["keyword_1", "keyword_2", ..., "keyword_{n}"])>
            }}
        ]
}}
---
"""

    return prompt

############ SUBASPECT DISCOVERY ############

class gen_subaspect_schema(BaseModel):
    subaspect_label: Annotated[str, StringConstraints(strip_whitespace=True)]
    subaspect_keywords: Annotated[str, StringConstraints(strip_whitespace=True)]
    
class subaspect_list_schema(BaseModel):
    subaspect_list : conlist(gen_subaspect_schema, min_length=2, max_length=5)

subaspect_instruction = lambda aspect, topic, k, n: f"You are an analyst that identifies different subaspects of parent aspect, {aspect}, to consider when evaluating whether or not {topic}. You will identify a minimum of 2 and a maximum of {k} subaspects that the provided set of scientific research paper excerpts are considering when determining the answer to the following topic: {aspect} for \"{topic}\". For each subaspect, provide {n} additional keywords that would typically be used to describe that subaspect, using the research excerpts we provide you as reference.\n\n"

def subaspect_prompt(aspect, topic, segments, k=5, n=10): 
    prompt = subaspect_instruction(aspect, topic, k, n)

    for idx, segment in enumerate(segments):
        prompt += f"Research Paper Excerpt #{idx+1}: {segment}\n\n"
    
    prompt += f"""For the topic, {topic}, output the list of up to {k} subaspects of parent aspect {aspect} in the following JSON format:
---
{{
    "subaspect_list":
        [
            {{
                "subaspect_label": <should be a brief, 5-10 word string where the value is an all-lowercase label of the subaspect (phrase-length) that falls under parent aspect {aspect}>,
                "subaspect_keywords": <list of {n} unique, all-lowercase, and comma-separated string keywords (always a space after the comma) used to describe the subaspect_label (e.g., ["keyword_1", "keyword_2", ..., "keyword_{n}"]) based on the input excerpts>
            }}
        ]
}}
---
"""

    return prompt