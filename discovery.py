from vllm import LLM
from vllm import SamplingParams
from outlines.serve.vllm import JSONLogitsProcessor
import json
import numpy as np
from prompts import subaspect_list_schema, subaspect_prompt

def subaspect_discovery(args, segments, rank2id, parent_aspect, top_k=10, temperature=0.7, top_p=0.99):

    if args.chat_model_name == "vllm":
        # OPTION 1: Use VLLM
        subset = [segments[rank2id[i]] for i in np.arange(top_k)]
        logits_processor = JSONLogitsProcessor(schema=subaspect_list_schema, llm=args.chat_model.llm_engine)
        sampling_params = SamplingParams(max_tokens=2000, logits_processors=[logits_processor], temperature=temperature, top_p=top_p)
        output = args.chat_model.generate(subaspect_prompt(parent_aspect.name, args.claim, subset), sampling_params=sampling_params)[0].outputs[0].text
        subaspects = json.loads(output)['subaspect_list']
        print(f"Generated subaspects for parent aspect '{parent_aspect.name}': {str([aspect['subaspect_label'] for aspect in subaspects])}")
    
    elif args.chat_model_name == "gpt-4o" or "gpt-4o-mini":
        # OPTION 2: Use GPT-4o
        subset = [segments[rank2id[i]] for i in np.arange(top_k)]
        response = args.chat_model([subaspect_prompt(parent_aspect.name, args.claim, subset)])[0]
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        subaspects = json.loads(response)['subaspect_list']
        print(f"Generated subaspects for parent aspect '{parent_aspect.name}': {str([aspect['subaspect_label'] for aspect in subaspects])}")
    
    return subaspects


def perspective_discovery():
    return