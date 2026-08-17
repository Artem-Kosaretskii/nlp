import os
import dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from typing import Annotated, Any, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain.agents import create_agent
from langgraph.graph import MessagesState, END
from langgraph.types import Command
from langgraph.graph import StateGraph, START
from IPython.display import Image, display


@tool
def python_perl_tool(code: Annotated[str, "Python code to run to plot the graph."]):
    """Use this function to execute python code. If you want to see the value output,
        you need to print it using `print(...)`. The user will see this output."""
    pyperl = PythonREPL()
    try:
        result = pyperl.run(code)
    except BaseException as e:
        return f"Couldn't execute it. Error: {repr(e)}"
    result_str = f"Successfully completed:\n\`\`\`python\n{code}\n\`\`\`\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, answer with FINAL ANSWER."
    )


def make_system_prompt(suffix: str) -> str:
    return (
        "You are a helpful AI assistant, working in collaboration with other assistants."
        "Use the tools provided to move toward answering the question."
        "If you can't answer completely, that's okay - another assistant with different tools"
        " will help you pick up where you left off. Do what you can to make progress."
        "If you or any of the other assistants have received the final answer or result,"
        "begin your answer with FINAL ANSWER so the team knows to stop."
        f"\n{suffix}"
    )


def main(**kwargs):

    dotenv.load_dotenv()
    OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')

    if kwargs['run_langgraph_multi_agent']:

        os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
        os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
        reasoning_efforts = ["low", "medium", "high"]
        model_name = "openai/gpt-oss-20b:free" # "nvidia/nemotron-3-ultra-550b-a55b:free" #

        llm = ChatOpenAI(
            model=model_name,
            # stream_usage=True,
            # temperature=None,
            max_tokens=200,
            # timeout=None,
            reasoning_effort=reasoning_efforts[0],
            max_retries=2,
            # rate_limiter=rate_limiter,
        )
        tavily_tool = TavilySearch(max_results=5)
        research_agent = create_agent(
            model=llm,
            tools=[tavily_tool],
            system_prompt=make_system_prompt(
                "You can only do research. You are working with a fellow designer. It generates graphs."
            ),
        )
        chart_agent = create_agent(
            model=llm,
            tools=[python_perl_tool],
            system_prompt=make_system_prompt(
                "You can only generate graphs. You are working with a fellow researcher."
            ),
        )

        def get_next_node(last_message: BaseMessage, goto: str):
            if "FINAL ANSWER" in last_message.content:
                return END
            return goto

        def research_node(state: MessagesState) -> Command[Literal["chart_generator", END]]:
            result = research_agent.invoke(state)
            goto = get_next_node(result["messages"][-1], "chart_generator")
            result["messages"][-1] = HumanMessage(
                content=result["messages"][-1].content, name="researcher"
            )
            return Command(update={"messages": result["messages"]}, goto=goto)

        def chart_node(state: MessagesState) -> Command[Literal["researcher", END]]:
            result = chart_agent.invoke(state)
            goto = get_next_node(result["messages"][-1], "researcher")

            result["messages"][-1] = HumanMessage(
                content=result["messages"][-1].content, name="chart_generator"
            )
            return Command(update={"messages": result["messages"]}, goto=goto)

        workflow = StateGraph(MessagesState)
        workflow.add_node("researcher", research_node)
        workflow.add_node("chart_generator", chart_node)
        workflow.add_edge(START, "researcher")
        graph = workflow.compile()
        # display(Image(graph.get_graph().draw_mermaid_png()))
        print(graph.get_graph().draw_ascii())

        events = graph.stream(
            {
                "messages": [
                    (
                        "user",
                        "First get the US GDP for the last 5 years, then plot a line graph. "
                        "After creating the graph, finish the job.",
                    )
                ],
            },
            {"recursion_limit": 150},
        )
        for s in events:
            print(s)
            print("----")


if __name__ == "__main__":
    params = {'run_langgraph_multi_agent': True}
    main(**params)
