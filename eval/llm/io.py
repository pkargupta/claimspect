from eval.llm.gpt import gpt4o_chat, gpt4o_mini_chat

def llm_chat(input_str_list: list[str], model_name: str) -> list[str]:
    if model_name == 'gpt-4o':
        return gpt4o_chat(input_str_list)
    if model_name == 'gpt-4o-mini':
        return gpt4o_mini_chat(input_str_list)
