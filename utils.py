from prompts import aspect_candidate_init_prompt, aspect_candidate_prompt
from model_definitions import llama_8b_model
import re


def aspect_candidate_gen(topic, k=5):
    messages = [
            {"role": "system", "content": aspect_candidate_init_prompt(topic, k)},
            {"role": "user", "content": aspect_candidate_prompt(topic, k)}]
        
    model_prompt = llama_8b_model.tokenizer.apply_chat_template(messages, 
                                                    tokenize=False, 
                                                    add_generation_prompt=True)

    terminators = [
        llama_8b_model.tokenizer.eos_token_id,
        llama_8b_model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = llama_8b_model(
        model_prompt,
        max_new_tokens=1024,
        eos_token_id=terminators,
        do_sample=False,
        pad_token_id=llama_8b_model.tokenizer.eos_token_id
    )
    message = outputs[0]["generated_text"][len(model_prompt):]

    print(message)

    gen_aspects = re.findall(r'aspect_\d+:\s*\[*(.*)\]*', message, re.IGNORECASE)
    gen_keywords = re.findall(r'aspect_\d+_keywords:\s*\[*(.*)\]*', message, re.IGNORECASE)

    gen_aspects = [a.lower().replace(" ", "_").replace("-", "_") for a in gen_aspects]
    gen_keywords = [[w.lower().replace(" ", "_").replace("-", "_") for w in a.split(", ")] for a in gen_keywords]
    
    return gen_aspects, gen_keywords