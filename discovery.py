from vllm import LLM
from vllm import SamplingParams
from outlines.serve.vllm import JSONLogitsProcessor
import json
import numpy as np

from prompts import subaspect_list_schema, subaspect_prompt, perspective_prompt, perspective_schema

def subaspect_discovery(args, segments, rank2id, parent_aspect, top_k=10, temperature=0.7, top_p=0.99):

    subset = [segments[rank2id[i]] for i in np.arange(top_k)]

    logits_processor = JSONLogitsProcessor(schema=subaspect_list_schema, llm=args.chat_model.llm_engine)
    sampling_params = SamplingParams(max_tokens=2000, logits_processors=[logits_processor], temperature=temperature, top_p=top_p)

    output = args.chat_model.generate(subaspect_prompt(parent_aspect.name, parent_aspect.description, args.claim, subset), sampling_params=sampling_params)[0].outputs[0].text

    subaspects = json.loads(output)['subaspect_list']
    
    print(f"Generated subaspects for parent aspect '{parent_aspect.name}': {str([aspect['subaspect_label'] for aspect in subaspects])}")
    
    return subaspects


def perspective_discovery(args, id2node, temperature=0.1, top_p=0.99):
    # construct prompts
    perspective_prompts = {}
    for node in id2node:
        if len(node.ranked_segments):
            prompt = perspective_prompt(args.claim, node.name, node.description, node.ranked_segments)
            perspective_prompts[node.id] = prompt

    # generate perspectives & classify
    logits_processor = JSONLogitsProcessor(schema=perspective_schema, llm=args.chat_model.llm_engine)
    sampling_params = SamplingParams(max_tokens=2000, logits_processors=[logits_processor], temperature=temperature, top_p=top_p)

    outputs = args.chat_model.generate(list(perspective_prompts.values()), sampling_params=sampling_params)

    # add perspectives into nodes
    for n_id, output in zip(perspective_prompts, outputs):
        o = output.outputs[0].text
        perspective = json.loads(o)
        id2node[n_id].perspectives = perspective

    return