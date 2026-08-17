import os
import dotenv
import json
import requests
from openai import OpenAI


def calculator(expression: str) -> float:
    return eval(expression)


def call_calculator_tool(expression: str) -> str:
    try:
        result = calculator(expression)
        return f"The result of the expression '{expression}': {result}"
    except Exception as e:
        return f"Error in calculation: {str(e)}"


def calc(model_name, client):
    messages = [
        {
            "role": "user",
            "content": "Calculate ((1+1)-(1+1)*2)/2"
        }
    ]
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=200,
    )
    print(completion.choices[0].message.content)

    calculator_tool = {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Calculates math expression and returns the result",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression for calculation, e.g. '2+2*3'"
                    }
                },
                "required": ["expression"],
                "additionalProperties": False
            }
        }
    }
    available_functions = {
        "calculator": call_calculator_tool,
    }
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=[calculator_tool],
        tool_choice="auto",
        max_completion_tokens=200,
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    print(tool_calls)
    messages.append(response_message)

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_to_call = available_functions[function_name]
        function_args = json.loads(tool_call.function.arguments)

        if function_name == "calculator":
            function_response = function_to_call(
                expression=function_args.get("expression")
            )
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
        )
    second_response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=[calculator_tool],
        tool_choice="auto",
        max_completion_tokens=200,
    )
    print(second_response.choices[0].message.content)

def count_letters(word: str) -> int:
    return len(word)


def call_count_letters_tool(word: str) -> str:
    try:
        result = count_letters(word)
        return f"A number of letters in the word '{word}': {result}"
    except Exception as e:
        return f"Error in calculation: {str(e)}"

def lenc(model_name, client):
    count_letters_tool = {
        "type": "function",
        "function": {
            "name": "count_letters",
            "description": "Calculates a number of letters in a word and returns the result",
            "parameters": {
                "type": "object",
                "properties": {
                    "word": {
                        "type": "string",
                        "description": "A word for a letter calculation, e.g. 'hello'"
                    }
                },
                "required": ["word"],
                "additionalProperties": False
            }
        }
    }
    available_functions = {"count_letters": call_count_letters_tool}
    messages = [
        {
            "role": "user",
            "content": "How many letters in a word 'programming'?"
        }
    ]
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=[count_letters_tool],
        tool_choice="auto",
        max_completion_tokens=200,
    )

    response_message = response.choices[0].message
    print(response_message)
    tool_calls = response_message.tool_calls
    print(tool_calls)
    messages.append(response_message)
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_to_call = available_functions[function_name]
        function_args = json.loads(tool_call.function.arguments)

        if function_name == "count_letters":
            function_response = function_to_call(
                word=function_args.get("word")
            )
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )
    final_response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=200,
    )
    assistant_response = final_response.choices[0].message.content
    print(assistant_response)


def main(**kwargs):

    dotenv.load_dotenv()
    API_KEY = os.getenv('OPENROUTE_API_KEY', '')
    model_name = "nvidia/nemotron-3.5-lightning:free"
    client = OpenAI(base_url="https://openrouter.ai/api/v1",api_key=API_KEY)
    if kwargs['calc']:
        calc(model_name, client)
    if kwargs['lenc']:
        lenc(model_name, client)


if __name__ == "__main__":
    params = {'calc': False, 'lenc': True}
    main(**params)