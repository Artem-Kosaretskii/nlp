import os
import dotenv
from bs4 import BeautifulSoup
from ddgs import DDGS
from typing import Dict
import requests
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.tools import AgentTool
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo
from pydantic import BaseModel
from typing import List


class PlannerOutput(BaseModel):
    queries: List[str]



async def get_summary(query: str, model_client: OpenAIChatCompletionClient) -> str:
    researcher = AssistantAgent("researcher",
                            model_client=model_client,
                            tools=[ddg_top_result, fetch_readable_text],
                            max_tool_iterations=3,
                            reflect_on_tool_use=True,
                            tool_call_summary_format="{result}",
                            system_message=(
                                "Jesteś badaczem internetu. Algorytm:\n"
                                "1) Wywołaj ddg_top_result z zapytaniem wyszukiwania i pobierz link.\n"
                                "2) Następnie wywołaj fetch_readable_text za pomocą tego linku, aby pobrać tekst.\n"
                                "3) Podsumuj treść w 5–8 zdaniach, dodając tytuł. Treść powinna być jak najbardziej adekwatna do zapytania.\n"
                                "Jeśli strona się nie ładuje, zgłoś to szczerze. Zachowaj zwięzłość i konkrety."
                            ),
                        )
    summary = await researcher.run(task=query)
    return summary.messages[-1].content


def _ua_headers() -> Dict[str, str]:
    return {
        "User-Agent": "Mozilla/5.0 (compatible; AutoGenBot/1.0; +https://github.com/microsoft/autogen)"
    }


def _is_wikipedia(url: str) -> bool:
    from urllib.parse import urlparse
    try:
        host = urlparse(url).hostname or ""
    except Exception:
        return False
    host = host.lower()
    return host == "wikipedia.org" or host.endswith(".wikipedia.org")


async def ddg_top_result(query: str) -> Dict[str, str]:
    """
    Only Wikipedia search:
      1) appends site:wikipedia.org to request;
      2) selects the first link *.wikipedia.org from results.
    Returns: {"title": str, "url": str, "snippet": str}
    """
    q = f"{query} site:wikipedia.org"
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(q, max_results=10))
        if not results:
            return {"title": "", "url": "", "snippet": ""}

        wiki_results = []
        for r in results:
            url = r.get("href") or r.get("link", "")
            if url and _is_wikipedia(url):
                wiki_results.append(r)

        if not wiki_results:
            return {
                "title": "",
                "url": "",
                "snippet": "There is no satisfying results on Wikipedia."
            }

        top = wiki_results[0]
        return {
            "title": top.get("title", "") or top.get("heading", ""),
            "url": top.get("href", "") or top.get("link", ""),
            "snippet": top.get("body", ""),
        }
    except Exception as e:
        return {"title": "", "url": "", "snippet": f"DDG error: {e}"}


async def fetch_readable_text(url: str) -> str:
    """
    Downloads HTML and returns page text
    """
    if not url:
        return ""
    try:
        resp = requests.get(url, headers=_ua_headers(), timeout=15)
        resp.raise_for_status()
        text = resp.text
        soup = BeautifulSoup(text, "html.parser")
        paragraphs = []
        for paragraph in soup.find_all("p"):
            paragraphs.append(paragraph.text)
        text = "\n".join(paragraphs)
        return text
    except Exception as e:
        return f"[fetch error] {e}"


async def main():
    query = "Biografia Jana Pawła II"
    dotenv.load_dotenv()
    model_name = "openai/gpt-oss-20b:free" # 'nvidia/nemotron-3-ultra-550b-a55b:free'
    search_model = "nvidia/nemotron-nano-9b-v2:free"
    os.environ["OPENAI_API_KEY"] = os.getenv('OPENROUTER_API_KEY', '')
    os.environ["OPENAI_BASE_URL"] = os.getenv('OPENROUTER_BASE_URL', '')
    # result_search = await ddg_top_result(query)
    # print(result_search)
    # text = await fetch_readable_text(result_search["url"])
    # print(text)

    search_client = OpenAIChatCompletionClient(model=search_model,
                                              parallel_tool_calls=False,
                                              model_info=ModelInfo(
                                                  vision=False,
                                                  function_calling=True,
                                                  json_output=True,
                                                  family='gpt-5',
                                                  structured_output=True
                                              ))
    general_client = OpenAIChatCompletionClient(model=model_name,
                                                parallel_tool_calls=False,
                                                model_info=ModelInfo(
                                                    vision=False,
                                                    function_calling=True,
                                                    json_output=True,
                                                    family='gpt-5',
                                                    structured_output=True
                                                ))
    planner = AssistantAgent(
        "planner",
        model_client=search_client,
        system_message="Napisz 5 zapytań internetowych, które pomogą Ci znaleźć informacje potrzebne do odpowiedzi na pytanie użytkownika. Odpowiedz natychmiast w formacie JSON.",
        output_content_type=PlannerOutput,
    )
    planner_response = await planner.run(task=query)
    queries = planner_response.messages[-1].content
    print(queries)

    summaries = await asyncio.gather(*(get_summary(q, general_client) for q in queries.queries))
    print(summaries)

    prompt = (
        "Na podstawie materiałów znalezionych w internecie, odpowiedz na pytanie użytkownika."
        "Materiał: \n{summary}"
        "Pytanie użytkownika: \n{query}"
    )
    prompt = prompt.format(summary='\n\n'.join(summaries), query=query)

    answer_agent = AssistantAgent(
        name="answer_agent",
        model_client=general_client,
        system_message="Odpowiadasz na pytania użytkownika w oparciu o materiały znalezione w internecie. Jeśli materiał nie zawiera informacji, odpowiedz, że nie wiesz."
    )
    answer = await answer_agent.run(task=prompt)
    print(answer.messages[-1].content)

asyncio.run(main())
