from main import main
import argparse
from api.local.e5_model import E5
from api.local.e5_model import e5_embed
from api.openai.embed import embed as openai_embed
from api.openai.chat import chat
from vllm import LLM
import os
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim", default="The Pfizer COVID-19 vaccine is better than the Moderna COVID-19 vaccine.")
    parser.add_argument("--claim_id", type=int, default=0)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--topic", default="vaccine")
    parser.add_argument("--chat_model_name", default="vllm")
    parser.add_argument("--embedding_model_name", default="e5")
    parser.add_argument("--top_k", type=float, default=5)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--gamma", type=float, default=2)
    parser.add_argument("--max_depth", type=int, default=2)
    parser.add_argument("--max_aspect_children_num", type=int, default=5)
    parser.add_argument("--max_subaspect_children_num", type=int, default=5)
    args = parser.parse_args()

    if args.embedding_model_name == "e5":
        args.embed_model = E5()
        args.embed_func = lambda text, model: e5_embed(embed_model=model, text_list=text)
    else:
        args.embed_model = args.embedding_model_name
        args.embed_func = lambda text, model: openai_embed(inputs=text, model_name=model)

    if args.chat_model_name == "vllm":
        # args.chat_model = LLM(model="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF", tensor_parallel_size=4, max_num_seqs=100, enable_prefix_caching=True)
        args.chat_model = LLM(model="meta-llama/Llama-3.1-8B-Instruct", tensor_parallel_size=4, max_num_seqs=100, enable_prefix_caching=True, gpu_memory_utilization=0.8)
    
    elif (args.chat_model_name == "gpt-4o") or (args.chat_model_name == "gpt-4o-mini"):
        def openai_model_specific_chat(prompts: list[str], temperature=0.3, top_p=0.99) -> list[str]:
            return chat(prompts, model_name=args.chat_model_name, seed=42, temperature=temperature, top_p=top_p)
        args.chat_model = openai_model_specific_chat

    with open(f"{args.data_dir}/{args.topic}/claims.json", "r") as f:
        claims = json.load(f)

    for c in claims[:1]:
        args.claim_id = c["id"]
        args.claim = c["body"]
        dir = f'{args.data_dir}/{args.topic}/{args.claim_id}/'

        assert os.path.exists(dir)

        main(args)
