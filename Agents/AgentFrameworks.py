import os
import asyncio
import dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo
from autogen_agentchat.tools import AgentTool
from autogen_agentchat.ui import Console
from smolagents import CodeAgent, OpenAIServerModel, ToolCallingAgent, tool


async def autogen(client) -> None:
    agent = AssistantAgent("assistant", model_client=client)
    response = await agent.run(task="Say Hello To Tomorrow!")
    print(response.messages[-1].content)
    await client.close()


async def autogen_science(client) -> None:
    math_agent = AssistantAgent(
        "math_expert",
        model_client=client,
        system_message="You know math very well and explain the solution step by step.",
        description="Math Expert",
        model_client_stream=True,
    )
    math_agent_tool = AgentTool(math_agent, return_value_as_last_message=True)

    physics_agent = AssistantAgent(
        "physics_expert",
        model_client=client,
        system_message="You know physics very well and explain the solution with jokes.",
        description="Physics Expert",
        model_client_stream=True,
    )
    physics_agent_tool = AgentTool(physics_agent, return_value_as_last_message=True)

    agent = AssistantAgent(
        "assistant",
        system_message="You are an agent manager and select agents to solve problems.",
        model_client=client,
        model_client_stream=True,
        tools=[math_agent_tool, physics_agent_tool],
        max_tool_iterations=10,
    )
    # await Console(agent.run_stream(task="Find the integral of x^3"))
    await Console(agent.run_stream(task="What is happened when an apple falls on a head?"))


@tool
def multiple_tool(a: int, b: int) -> str:
    """Multiplies two numbers.

    Args:
        a: first number.
        b: second number.
    """
    return str(a * b)


def run_smolagents(model, tools=False):
    if tools:
        agent = ToolCallingAgent(tools=[multiple_tool], model=model)
        agent.run("Evaluate 221 * 5831")
    else:
        agent = CodeAgent(tools=[], model=model, stream_outputs=True)
        agent.run("Evaluate 221 + 5831")


def main(**kwargs):
    dotenv.load_dotenv()
    model_name = "nvidia/nemotron-3-ultra-550b-a55b:free" # "openai/gpt-oss-20b:free"
    os.environ["OPENAI_API_KEY"] = os.getenv('OPENROUTER_API_KEY', '')
    os.environ["OPENAI_BASE_URL"] = os.getenv('OPENROUTER_BASE_URL', '')
    client = OpenAIChatCompletionClient(model=model_name,
                                        model_info=ModelInfo(
                                            vision=False,
                                            function_calling=True,
                                            json_output=True,
                                            family='gpt-5',
                                            structured_output=True
                                        ))
    if kwargs['autogen_demo']:
        asyncio.run(autogen(client))
    if kwargs['autogen_science']:
        asyncio.run(autogen_science(client))
    if kwargs['run_smolagents']:
        model = OpenAIServerModel(model_id=model_name)
        run_smolagents(model, True)


    stop = True

if __name__ == "__main__":
    params = {'run_smolagents': True, 'autogen_demo': False, 'autogen_science': False}
    main(**params)
