
aspect_candidate_init_prompt = lambda topic, k=5, n=10: f"You are an analyst that identifies the different {k} aspects that scientific research papers would consider when determining the answer to the following topic: \"{topic}\". For each aspect, provide {n} additional keywords that would typically be used to describe that aspect for the topic, {topic}."

def aspect_candidate_prompt(topic, k=5, n=10): 
    
    prompt = f"""For the topic, {topic}, output the list of up-to {k} aspects in the following YAML format:

---
"""
    for i in range(k):
        prompt += f"aspect_{i}: <1-2 word label (string format) of aspect_{i}>\n"
        prompt += f"aspect_{i}_keywords: <list of {n} comma-separated keywords (always a space after the comma) commonly used to describe aspect_{i} (e.g., keyword_1, keyword_2, ..., keyword_{n})>\n"

    prompt += "---\n"

    return prompt