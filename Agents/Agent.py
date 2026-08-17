import os
import dotenv
import json
import datetime
from openai import OpenAI
from typing import List, Dict, Any
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI


def format_scratchpad(steps: List[Dict[str, Any]]) -> str:
    """Formats action history for ReAct prompts"""
    if not steps:
        return "I'm obliged to help the user."

    scratchpad = ""
    for step in steps:
        if "thought" in step:
            scratchpad += f"Thought: {step['thought']}\n"
        if "action" in step:
            scratchpad += f"Action: {step['action']}\n"
        if "action_input" in step:
            scratchpad += f"Action Input: {step['action_input']}\n"
        if "observation" in step:
            scratchpad += f"Observation: {step['observation']}\n"
    return scratchpad


def parse_react_response(content: str) -> Dict[str, Any]:
    """ReAct answer parsing"""
    result = {}

    if "Thought:" in content:
        thought_part = content.split("Thought:")[1]
        if "\n" in thought_part:
            result["thought"] = thought_part.split("\n")[0].strip()
        else:
            result["thought"] = thought_part.strip()

    if "Action:" in content:
        action_part = content.split("Action:")[1]
        if "\n" in action_part:
            result["action"] = action_part.split("\n")[0].strip()
        else:
            result["action"] = action_part.strip()

    if "Action Input:" in content:
        action_input_part = content.split("Action Input:")[1]
        if "\n" in action_input_part:
            result["action_input"] = action_input_part.split("\n")[0].strip()
        else:
            result["action_input"] = action_input_part.strip()

    if "Final Answer:" in content:
        final_part = content.split("Final Answer:")[1]
        result["final_answer"] = final_part.strip()

    return result


def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> str:

    if tool_name == "get_time":
        return get_time()
    elif tool_name == "count_letters":
        return count_letters(arguments.get("word", ""))
    else:
        return f"Unknown tool: {tool_name}"


def format_tools_description(tools) -> str:
    """Description for ReAct"""

    tools_desc = []
    for tool in tools:
        name = tool["function"]["name"]
        desc = tool["function"]["description"]
        params = tool["function"].get("parameters", {})

        if params.get("properties"):
            param_desc = ", ".join([f"{k}: {v.get('type', 'string')}" for k, v in params["properties"].items()])
            tools_desc.append(f"- {name}: {desc} (properties: {param_desc})")
        else:
            tools_desc.append(f"- {name}: {desc}")

    return "\n".join(tools_desc)


def create_react_prompt(user_input: str, scratchpad: str, tools: Any) -> str:
    """Creates ReAct prompts"""

    tools_desc = format_tools_description(tools)
    return f"""You are an intellectual assistant, which uses ReAct approach (Reasoning + Acting).
    Avialable tools:
    {tools_desc}

    Response format:
    Thought: You need to think about what needs to be done
    Action: Tool name (if needed) or answer
    Action Input: Input data for the tool (if needed)
    Observation: Result of the tool execution
    ... (this Thought/Action/Action Input/Observation cycle can be repeated)
    Thought: I now know the final answer
    Final Answer: The final answer to the question
    
    Get started!    
    Question: {user_input}
    Thought: {scratchpad}"""


def generate_search_queries(user_query, model_name, client, system_message, search_query_schema):
    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "system", "content": system_message}, {"role": "user", "content": user_query}],
        response_format=search_query_schema,
        max_tokens=1000
    )
    return completion.choices[0].message.content


def get_time():
    """Returns actual time"""
    return f"Time: {datetime.datetime.now().strftime('%H:%M:%S')}"


def count_letters(word):
    """Counts letters in a word"""
    count = len(word)
    return f"The word '{word}' consist of {count} letters"


def get_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "Learn current time",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "count_letters",
                "description": "Count the number of letters in a word",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "word": {
                            "type": "string",
                            "description": "Word for letter counting"
                        }
                    },
                    "required": ["word"]
                }
            }
        },
    ]


def run_agent(model_name: str, client: Any, user_input: str, chat_history: List, calls_num: int = 5, max_tokens: int = 500):

    chat_history.append({"role": "user", "content": user_input})
    for _ in range(calls_num):
        response = client.chat.completions.create(
            model=model_name,
            messages=chat_history,
            max_tokens=max_tokens,
            tools=get_tools()
        )

        msg = response.choices[0].message
        chat_history.append(msg)

        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for call in msg.tool_calls:
                name = call.function.name
                args = json.loads(call.function.arguments)

                if name == "get_time":
                    result = get_time()
                elif name == "count_letters":
                    result = count_letters(args.get("word", ""))
                else:
                    result = "Неизвестная команда"

                chat_history.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result
                })
        else:
            return msg.content

    return "Sorry, too much instrument were being requested for the answer, can't do that."


def run_react_agent(model_name, client, user_input: str, chat_history: List[Dict], max_tokens=500, steps=5) -> str:

    chat_history.append({"role": "user", "content": user_input})
    steps_history = []
    for step in range(steps):
        scratchpad = format_scratchpad(steps_history)
        tools_desc = format_tools_description(get_tools())
        prompt = f"""Use the ReAct approach: think, then act with tools if needed.
            Tools:
            {tools_desc}
            Question: {user_input}
            {scratchpad}"""

        messages = [
            {"role": "system", "content": "Follow the format: Thought: ... Action: ... Action Input: ..."},
            {"role": "user", "content": prompt}
        ]

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            tools=get_tools()
        )
        msg = response.choices[0].message
        chat_history.append(msg)

        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            current_step = {}

            if msg.content and "Thought:" in msg.content:
                thought = msg.content.split("Thought:")[1].split("\n")[0].strip()
                current_step["thought"] = thought

            for call in msg.tool_calls:
                tool_name = call.function.name
                args = json.loads(call.function.arguments)

                current_step["action"] = tool_name
                current_step["action_input"] = args

                result = execute_tool(tool_name, args)
                current_step["observation"] = result

                chat_history.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result
                })

                steps_history.append(current_step)
        else:
            if msg.content and ("Final Answer:" in msg.content or "Thought:" in msg.content):
                return msg.content
            else:
                return msg.content or "Couldn't get the answer"

    return "Max step number has been exceeded"


def main(**kwargs):
    dotenv.load_dotenv()
    OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')
    model_name = "nvidia/nemotron-3-ultra-550b-a55b:free"
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

    if kwargs['run_agent']:
        history = [
            {"role": "system", "content": "You are a helpful assistant. Use tools when it needs."}
        ]
        print("FC/TC Agent with Tool Calling")
        print("'Break' for exit\\n")

        while True:
            question = input("Your question: ")
            if question == "Break":
                break
            else:
                answer = run_agent(model_name, client, question, history)
            print("Answer:", answer)
            print()

    elif kwargs['run_react_agent']:

        history = [{"role": "system", "content": "You are a helpful assistant. Use the ReAct approach with tools when needed."}]
        print("ReAct Agent with Tool Calling")
        print("'Break' for exit\\n")
        while True:
            question = input("You question: ")
            if question == "Break":
                break
            else:
                answer = run_react_agent(model_name, client, question, history)
                print("Answer:", answer)
                print()

    elif kwargs['run_lg_react_agent']:

        model_name = "openai/gpt-oss-20b:free"
        os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
        os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
        reasoning_efforts = ["low", "medium", "high"]
        llm = ChatOpenAI(
            model=model_name,
            # stream_usage=True,
            # temperature=None,
            max_tokens=200,
            # timeout=None,
            reasoning_effort=reasoning_efforts[1],
            max_retries=3,
        )
        agent = create_agent(
            model=llm,
            tools=[get_time, count_letters],
            system_prompt="You are a helpful assistant",
        )
        state = {"messages": []}
        print("LangGraph ReAct Agent with Tool Calling")
        print("'Break' for exit\\n")
        while True:
            question = input("You question: ")
            if question == "Break":
                break
            else:
                state["messages"].append({"role": "user", "content": question})
                state = agent.invoke(state)
                print("Answer: ", state["messages"][-1].content)


if __name__ == "__main__":
    params = {'run_agent': False, 'run_react_agent': False}
    main(**params)
