import os
import dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI


def add_numbers(a: float, b: float) -> str:
    """Adds two numbers a and b"""
    result = a + b
    return f"Addition result: {a} + {b} = {result}"


def subtract_numbers(a: float, b: float) -> str:
    """Subtracts number b from number a"""
    result = a - b
    return f"Subtraction result: {a} - {b} = {result}"


def multiply_numbers(a: float, b: float) -> str:
    """Multiplies two numbers a and b"""
    result = a * b
    return f"Multiplication result: {a} × {b} = {result}"


def divide_numbers(a: float, b: float) -> str:
    """Divides number a by number b"""
    if b == 0:
        return "Error: dividing by zero is impossible"
    result = a / b
    return f"Division result: {a} ÷ {b} = {result}"


def main(**kwargs):

    dotenv.load_dotenv()
    OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')

    if kwargs['run_langgraph_mt_agent']:

        os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
        os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
        reasoning_efforts = ["low", "medium", "high"]
        model_name = "nvidia/nemotron-3-ultra-550b-a55b:free" # "openai/gpt-oss-20b:free"
        llm = ChatOpenAI(
            model=model_name,
            # stream_usage=True,
            # temperature=None,
            max_tokens=200,
            # timeout=None,
            reasoning_effort=reasoning_efforts[0],
            max_retries=3,
        )
        agent = create_agent(
            model=llm,
            tools=[add_numbers, subtract_numbers, multiply_numbers, divide_numbers],
            system_prompt="You are a helpful assistant"
        )

        test_cases = [
            "Evaluate: (15.7 + 8.3) × 2.5 - 10 ÷ 4",
            "Find the result: (2^4 × 3) ÷ (6 - 2) + 7.5",
            "Solve the chain: 100 ÷ 5 × 3 + 12 - 8 ÷ 2",
            "Evaluate the expression: (4.5 + 3.5)^2 ÷ 2 × 1.5",
            "Find the value: (20 - 3 × 4) ÷ 2 + 5^2 × 0.4"
        ]

        correct_results = [
            (15.7 + 8.3) * 2.5 - 10 / 4,
            (2 ** 4 * 3) / (6 - 2) + 7.5,
            100 / 5 * 3 + 12 - 8 / 2,
            (4.5 + 3.5) ** 2 / 2 * 1.5,
            (20 - 3 * 4) / 2 + 5 ** 2 * 0.4
        ]

        state = {"messages": []}
        print("LangGraph ReAct Agent with Tool Calling")
        print("'Break' for exit\\n")
        count = 0
        while True:
            question = input("You question: ")
            if question == "Break":
                break
            else:
                if question == 'test':
                    question = test_cases[count]
                    state["messages"].append({"role": "user", "content": question})
                    state = agent.invoke(state)
                    print("Answer: ", state["messages"][-1].content)
                    print(f"Correct result: {correct_results[count]}")
                    count += 1
                else:
                    state["messages"].append({"role": "user", "content": question})
                    state = agent.invoke(state)
                    print("Answer: ", state["messages"][-1].content)


if __name__ == "__main__":
    params = {'run_langgraph_mt_agent': True}
    main(**params)
