

def llm_chat(input_str_list: list[str], model_name: str) -> list[str]:
    if model_name == 'gpt-4o':
        from eval.llm.gpt import gpt4o_chat
        return gpt4o_chat(input_str_list)
    if model_name == 'gpt-4o-mini':
        from eval.llm.gpt import gpt4o_mini_chat
        return gpt4o_mini_chat(input_str_list)
    if model_name == 'llama-3.1-8b-instrcut':
        from eval.llm.llama import llama_chat
        return llama_chat(input_str_list)