from vllm import LLM
from vllm import SamplingParams
from vllm.sampling_params import GuidedDecodingParams
import json
import numpy as np
from tqdm import tqdm

from src.prompts import subaspect_list_schema, subaspect_prompt, perspective_prompt, perspective_schema
from src.prompts import stance_schema, stance_prompt, perspective_desc_schema, perspective_desc_prompt

def subaspect_discovery(args, segments, rank2id, parent_aspect, top_k=10, temperature=0.7, top_p=0.99):
    parent_path = " -> ".join(parent_aspect.get_ancestors(as_str=True))

    if args.chat_model_name == "vllm":
        subset = [segments[rank2id[i]] for i in np.arange(top_k)]
        guided_decoding_params = GuidedDecodingParams(json=subaspect_list_schema.model_json_schema())
        sampling_params = SamplingParams(max_tokens=2000, guided_decoding=guided_decoding_params, temperature=temperature, top_p=top_p)
        output = args.chat_model.generate(subaspect_prompt(parent_aspect.name, parent_aspect.description, parent_path, args.claim, subset, k=args.max_subaspect_children_num), sampling_params=sampling_params)[0].outputs[0].text
        subaspects = json.loads(output)['subaspect_list']
    
    elif (args.chat_model_name == "gpt-4o") or (args.chat_model_name == "gpt-4o-mini"):
        # OPTION 2: Use GPT-4o
        subset = [segments[rank2id[i]] for i in np.arange(top_k)]
        response = args.chat_model([subaspect_prompt(parent_aspect.name, parent_aspect.description, args.claim, subset, k=args.max_subaspect_children_num)])[0]
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        subaspects = json.loads(response)['subaspect_list']
        print(f"Generated subaspects for parent aspect '{parent_aspect.name}': {str([aspect['subaspect_label'] for aspect in subaspects])}")
    
    return subaspects


def perspective_discovery(args, id2node, temperature=0.1, top_p=0.99, top_k=20):
    for node in tqdm(id2node):
        # STAGE 1: for each node, get all segments
        segments = node.get_all_segments()
        node.mapped_segs = segments
        node_path = " -> ".join(node.get_ancestors(as_str=True))

        # STAGE 2: classify the stance of each segment
        if len(segments):
            stance_prompts = [stance_prompt(args.claim, node.name, node.description, node_path, seg) for seg in segments]
            if args.chat_model_name == "vllm":
                # stance detection
                guided_decoding_params = GuidedDecodingParams(json=stance_schema.model_json_schema())
                sampling_params = SamplingParams(max_tokens=1000, guided_decoding=guided_decoding_params, temperature=temperature, top_p=top_p)
                outputs = args.chat_model.generate(stance_prompts, sampling_params=sampling_params)
            
            elif (args.chat_model_name == "gpt-4o") or (args.chat_model_name == "gpt-4o-mini"):
                # OPTION 2: Use GPT-4o
                responses = args.chat_model(stance_prompts)
                outputs = responses

            stance_clusters = {"supports_claim":[],
                               "neutral_to_claim":[],
                               "opposes_claim":[],
                               "irrelevant_to_claim":[]}
            # cluster the stances
            for seg_id, output in enumerate(outputs):
                o = output.outputs[0].text if args.chat_model_name == "vllm" else output
                if "```json" in o:
                    o = o.split("```json")[1].split("```")[0].strip()
                segment_stance = json.loads(o)
                if not (segment_stance["irrelevant_to_claim"]):
                    if segment_stance["supports_claim"]:
                        stance_clusters["supports_claim"].append((seg_id, segments[seg_id], segment_stance["explanation"]))
                    elif segment_stance["neutral_to_claim"]:
                        stance_clusters["neutral_to_claim"].append((seg_id, segments[seg_id], segment_stance["explanation"]))
                    else:
                        stance_clusters["opposes_claim"].append((seg_id, segments[seg_id], segment_stance["explanation"]))
                else:
                    stance_clusters["irrelevant_to_claim"].append((seg_id, segments[seg_id], segment_stance["explanation"]))

            # STAGE 3: for each stance cluster, determine a description
            perspective_prompts = [perspective_desc_prompt(args.claim, node.name, node.description, node_path, "supportive", stance_clusters["supports_claim"]),
                                   perspective_desc_prompt(args.claim, node.name, node.description, node_path, "neutral", stance_clusters["neutral_to_claim"]),
                                   perspective_desc_prompt(args.claim, node.name, node.description, node_path, "in opposition", stance_clusters["opposes_claim"])]
            if args.chat_model_name == "vllm":
                # generate perspectives & classify
                guided_decoding_params = GuidedDecodingParams(json=perspective_desc_schema.model_json_schema())
                sampling_params = SamplingParams(max_tokens=2000, guided_decoding=guided_decoding_params, temperature=temperature, top_p=top_p)
                outputs = args.chat_model.generate(perspective_prompts, sampling_params=sampling_params)
            
            elif (args.chat_model_name == "gpt-4o") or (args.chat_model_name == "gpt-4o-mini"):
                # OPTION 2: Use GPT-4o
                responses = args.chat_model(perspective_prompts)
                outputs = responses
            
            perspective = {"supports_claim":{}, "neutral_to_claim":{}, "opposes_claim":{}}
            for p, output in zip(["supports_claim", "neutral_to_claim", "opposes_claim"], outputs):
                o = output.outputs[0].text if args.chat_model_name == "vllm" else output
                if "```json" in o:
                    o = o.split("```json")[1].split("```")[0].strip()
                
                perspective_output = json.loads(o)
                
                perspective[p] = {"perspective_segments": [seg_id for (seg_id, seg, exp) in stance_clusters[p]] 
                                  if len(stance_clusters[p]) else [],
                                  "perspective_description": perspective_output["perspective_description"]}
            
            node.perspectives = perspective

    return