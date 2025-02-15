import os

# List of topics to process
TOPIC_LIST = ['dtra', 'vaccine']

def robust_input(prompt: str, valid_responses: list[str]) -> str:
    """
    Prompt the user until a valid response is given.

    :param prompt: The input prompt to display.
    :param valid_responses: List of valid responses.
    :return: A valid response from the user.
    """
    response = input(prompt + '\n')
    while response not in valid_responses:
        print(f"Invalid response. Please choose from {valid_responses}")
        response = input(prompt + '\n')
    return response

def get_user_id(user_num: int) -> str:
    """
    Retrieve the user id based on a simple prompt.

    :param user_num: Number of users.
    :return: User id as a string.
    """
    prompt = f"If you are Priyanka, input 0. If you are Runchu, input 1."
    return robust_input(prompt, [str(i) for i in range(user_num)])
