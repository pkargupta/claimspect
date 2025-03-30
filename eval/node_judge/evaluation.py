# evaluation.py

from eval.llm.io import llm_chat  # make sure this import path is correct in your project

def human_chat(input_str_list: list[str]) -> list[str]:
    results = []
    for prompt in input_str_list:
        print("\n" + "="*80)
        print(prompt)
        print("="*80 + "\n")
        response = input("Please enter your response: ")
        results.append(response)
    return results

def get_path_relevance(claim: str, paths: list, judge_name, max_paths: int = None) -> list:
    def get_prompt(claim, path):
        return (
            "Claims made by individuals or entities are often nuanced and cannot always be strictly categorized as entirely 'true' or 'false'. "
            "Particularly in scientific and political contexts. Instead, a claim can be broken down "
            "into its core aspects and sub-aspects, which are easier to evaluate individually.\n\n"
            f"Given the claim: '{claim}', decide whether this path from the aspect tree is relevant to the analysis of the claim: '{path}'\n\n"
            "Output options: '<relevant>' or '<irerelevant>'. Do some simple rationalization before giving the output if possible."
        )
    
    # Limit paths if max_paths is specified
    if max_paths is not None:
        paths = paths[:max_paths]
    
    input_strs = [get_prompt(claim, path) for path in paths]
    outputs = human_chat(input_strs) if judge_name == "human" else llm_chat(input_strs, model_name=judge_name)
    
    results = []
    for path, output in zip(paths, outputs):
        result = {"path": path}
        if "<relevant>" in output.lower():
            result['score'] = 1
        elif "<irelevant>" in output.lower():
            result['score'] = 0
        else:
            result['score'] = -1
        result['reasoning'] = output
        results.append(result)
    
    return results

def get_path_granularity(claim: str, paths: list, judge_name, max_paths: int = None) -> list:
    def get_prompt(claim, path):
        return (
            "Claims made by individuals or entities are often nuanced and cannot always be strictly categorized as entirely 'true' or 'false'. "
            "Particularly in scientific and political contexts. Instead, a claim can be broken down "
            "into its core aspects and sub-aspects, which are easier to evaluate individually.\n\n"
            f"Given the claim: '{claim}', decide whether this path from the aspect tree has good granularity: '{path}' Check whether the child node is a more specific subaspect of the parent node. \n\n"
            "Output options: '<good granularity>' or '<bad granularity>'. Do some simple rationalization before giving the output if possible."
        )
    
    # Limit paths if max_paths is specified
    if max_paths is not None:
        paths = paths[:max_paths]
        
    input_strs = [get_prompt(claim, path) for path in paths]
    outputs = human_chat(input_strs) if judge_name == "human" else llm_chat(input_strs, model_name=judge_name)
    
    results = []
    for path, output in zip(paths, outputs):
        result = {"path": path}
        if "<good granularity>" in output.lower():
            result['score'] = 1
        elif "<bad granularity>" in output.lower():
            result['score'] = 0
        else:
            result['score'] = -1
        result['reasoning'] = output
        results.append(result)
    
    return results

def get_level_granularity(claim: str, levels: list, judge_name, max_levels: int = None) -> list:
    def get_prompt(claim, level_instance):
        parent = level_instance['parent']
        siblings = level_instance['siblings']
        return (
            "Claims made by individuals or entities are often nuanced and cannot always be strictly categorized as entirely 'true' or 'false'. "
            "Particularly in scientific and political contexts. Instead, a claim can be broken down "
            "into its core aspects and sub-aspects, which are easier to evaluate individually.\n\n"
            f"Given the claim: '{claim}', decide whether these siblings from parent node '{parent}', have good granularity: '{', '.join(siblings)}' Check whether they have similar specificity level. \n\n"
            "Output options: '<all not granular>' or '<majority not granular>' or "
            "'<majority granular>' or '<all granular>'. Do some simple rationalization before giving the output if possible."
        )
    
    # Limit levels if max_levels is specified
    if max_levels is not None:
        levels = levels[:max_levels]
        
    input_strs = [get_prompt(claim, level) for level in levels]
    outputs = human_chat(input_strs) if judge_name == "human" else llm_chat(input_strs, model_name=judge_name)
    
    results = []
    for level_instance, output in zip(levels, outputs):
        result = {"path": level_instance}
        if "<all not granular>" in output.lower():
            result['score'] = 1
        elif "<majority not granular>" in output.lower():
            result['score'] = 2
        elif "<majority granular>" in output.lower():
            result['score'] = 3
        elif "<all granular>" in output.lower():
            result['score'] = 4
        else:
            result['score'] = -1
        result['reasoning'] = output
        results.append(result)
    
    return results

def retry_taxonomy_wise_uniqueness(claim: str, taxonomy: str, node_num: int, retry_times: int = 3, judge_name=None) -> dict:
    for _ in range(retry_times):
        result = get_taxonomy_wise_uniqueness(claim, taxonomy, node_num, judge_name)
        if result['score'] is not None:
            return result
    return result

def get_taxonomy_wise_uniqueness(claim: str, taxonomy: str, node_num: int, judge_name) -> dict:
    def get_prompt(claim, taxonomy):
        return (
            "Claims made by individuals or entities are often nuanced and cannot always be strictly categorized as entirely 'true' or 'false'. "
            "Particularly in scientific and political contexts. Instead, a claim can be broken down "
            "into its core aspects and sub-aspects, which are easier to evaluate individually.\n\n"
            "Normally, we want the aspects and sub-aspects to be unique in the taxonomy. "
            f"Given the claim: '{claim}', count how many nodes in this taxonomy are largely overlapping or almost equivalent: {taxonomy}\n\n"
            "Output options: '<overlap_num>0</overlap_num>', '<overlap_num>1</overlap_num>'... or other possible numbers. Do some simple rationalization before giving the output if possible."
        )
    prompt = get_prompt(claim, taxonomy)
    outputs = human_chat([prompt])[0] if judge_name == "human" else llm_chat([prompt], model_name=judge_name)[0]
    
    # extract the number from the output
    result = {"taxonomy": taxonomy}
    try:
        output_int = outputs.split('<overlap_num>')[1].split('</overlap_num>')[0].strip()
        result['score'] = 1 - int(output_int) / node_num
    except Exception:
        print(Exception)
        result['score'] = None
    result['reasoning'] = outputs
    return result

def get_node_wise_segment_quality(claim: str, nodes: list, judge_name, max_nodes: int = None) -> list:
    def get_prompt(claim, node):
        aspect_name = node['aspect_name']
        segments = node['segments']
        index_segments = "\n".join([f"{i+1}. {seg}" for i, seg in enumerate(segments)])
        return (
            "Claims made by individuals or entities are often nuanced and cannot always be strictly categorized as entirely 'true' or 'false'. "
            "Particularly in scientific and political contexts. Instead, a claim can be broken down "
            "into its core aspects and sub-aspects, which are easier to evaluate individually.\n\n"
            f"Given the claim: '{claim}', Here is one aspect to analyze this claim: {aspect_name} and these are some segments: {index_segments}\n\n"
            "Count how many of them are relevant to this specific aspect.\n\n"
            "Output options: '<rel_seg_num> ... (int) </rel_seg_num>'. Do some rationalization before outputting the number of relevant segments."
        )
    
    # Limit nodes if max_nodes is specified
    if max_nodes is not None:
        nodes = nodes[:max_nodes]
        
    input_strs = [get_prompt(claim, node) for node in nodes]
    outputs = human_chat(input_strs) if judge_name == "human" else llm_chat(input_strs, model_name=judge_name)
    results = []
    for node, output in zip(nodes, outputs):
        if len(node['segments']) == 0:
            continue
        result = {"node": node, "segments": node['segments']}
        try:
            output_int = output.split('<rel_seg_num>')[1].split('</rel_seg_num>')[0]
            result['score'] = int(output_int) / len(node['segments'])
        except Exception:
            result['score'] = -1
        result['reasoning'] = output
        results.append(result)
    return results
