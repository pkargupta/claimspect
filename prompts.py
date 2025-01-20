from pydantic import BaseModel, StringConstraints, conlist
from typing_extensions import Annotated

############ COARSE-GRAINED ASPECT DISCOVERY ############

class gen_aspect_schema(BaseModel):
    aspect_label: Annotated[str, StringConstraints(strip_whitespace=True)]
    aspect_description: Annotated[str, StringConstraints(strip_whitespace=True)]
    aspect_keywords: conlist(str, min_length=2, max_length=20)
    
class aspect_list_schema(BaseModel):
    aspect_list : conlist(gen_aspect_schema, min_length=2, max_length=5)


aspect_candidate_init_prompt = lambda topic, k=5, n=10: f"You are an analyst that identifies the different aspects (a minimum of 2 and a maximum of {k} aspects) that scientific research papers would consider when determining the answer to the following topic: \"{topic}\". For each aspect, provide a description and {n} additional keywords that would typically be used to describe that aspect for the topic, {topic}.\n\n"

def aspect_prompt(topic, k=5, n=10): 
    prompt = aspect_candidate_init_prompt(topic, k, n)
    prompt += f"""For the topic, {topic}, output the list of up to {k} aspects in the following JSON format:
---
{{
    "aspect_list":
        [
            {{
                "aspect_label": <should be a brief, 5-10 word string where the value is an all-lowercase label of the aspect (phrase-length)>,
                "aspect_description": <a string where the value is a  brief, sentence-long description of the aspect and why it is significant for addressing the topic>,
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
    subaspect_description: Annotated[str, StringConstraints(strip_whitespace=True)]
    subaspect_keywords: conlist(str, min_length=2, max_length=20)
    
class subaspect_list_schema(BaseModel):
    subaspect_list : conlist(gen_subaspect_schema, min_length=2, max_length=5)

subaspect_instruction = lambda aspect, topic, k, n: f"You are an analyst that identifies different subaspects of parent aspect, {aspect}, to consider when evaluating the claim: {topic}. You will identify a minimum of 2 and a maximum of {k} subaspects under aspect {aspect} that the provided set of scientific research paper excerpts are considering when determining the answer to the following topic: {aspect} for \"{topic}\". For each subaspect, provide a description and {n} additional keywords that would typically be used to describe that subaspect, using the research excerpts we provide you as reference. We define subaspect as a more specific point underneath the parent aspect that would considered when specifically addressing the claim.\n\n"

def subaspect_prompt(aspect, aspect_description, topic, segments, k=5, n=10): 
    prompt = subaspect_instruction(aspect, topic, k, n)

    for idx, segment in enumerate(segments):
        prompt += f"Research Paper Excerpt #{idx+1}: {segment}\n\n"
    
    prompt += f"""Output the list of up to {k} subaspects of parent aspect {aspect} that would be considered when evaluating the claim, {topic}.

claim: {topic}
parent_aspect: {aspect}; {aspect_description}

Provide your output in the following JSON format:
---
{{
    "subaspect_list":
        [
            {{
                "subaspect_label": <should be a brief, 5-10 word string where the value is an all-lowercase label of the subaspect (phrase-length) that falls under parent aspect {aspect} and is specifically relevant to the claim>,
                "subaspect_description": <a string where the value is a brief, sentence-long description of the subaspect and why it is significant for addressing specifically both the claim and the parent aspect>,
                "subaspect_keywords": <list of {n} unique, all-lowercase, and comma-separated string keywords (always a space after the comma) used to describe the subaspect_label (e.g., ["keyword_1", "keyword_2", ..., "keyword_{n}"]) based on the input excerpts>
            }}
        ]
}}
---
"""

    return prompt


def keyword_extraction_prompt(claim, aspect_name, aspect_description, max_keyword_num, seg_contents):
    contents = ""
    for idx, seg in enumerate(seg_contents):
        contents += f"Document #{idx}: {seg}\n\n"
    
    return f"""The claim is: {claim}. You are analyzing it with a focus on the aspect {aspect_name}. The aspect, {aspect_name}, can be described as the following: {aspect_description}

Please extract at most {2*max_keyword_num} keywords related to the aspect {aspect_name} from the following documents: 

{contents}

Ensure that the extracted keywords are diverse, specific, and highly relevant to the given aspect. Only output the keywords and seperate them with comma.
Your output should be in the following JSON format:
---
{{
    "output_keywords": <list of strings, where each string is a unique, lowercase keyword relevant to the aspect and prominent within the input documents>
}}
---
"""

def keyword_filter_prompt(claim, aspect_name, aspect_description, min_keyword_num, max_keyword_num, keyword_candidates):
    return f"""Our claim is '{claim}'. With respective to the target aspect '{aspect_name}', identify {min_keyword_num} to {max_keyword_num} relevant keywords from the provided list: {keyword_candidates}.

{aspect_name}: {aspect_description}

Merge terms with similar meanings, exclude relatively irrelevant ones, and output only the final keywords separated by commas.

Your output should be in the following JSON format:
---
{{
    "output_keywords": <FILTERED list of strings, where each string is a unique, lowercase keyword relevant to the target aspect>
}}
---
"""