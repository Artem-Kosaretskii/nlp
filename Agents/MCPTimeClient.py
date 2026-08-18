import asyncio
import json
import os
import dotenv
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI


class MCPClient:
    def __init__(self, model_name):
        self.session = None
        self.exit_stack = AsyncExitStack()
        self.openai = OpenAI()
        self.messages = []
        self.model_name = model_name

    async def connect(self, server_config):
        params = StdioServerParameters(**server_config)
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()
        tools_resp = await self.session.list_tools()
        print("MCP tools:", [t.name for t in tools_resp.tools])

    async def ask(self, query: str) -> str:
        self.messages.append({"role": "user", "content": query})
        tools_resp = await self.session.list_tools()
        oa_tools = [convert_tool_format(t) for t in tools_resp.tools]

        resp = self.openai.chat.completions.create(
            model=self.model_name,
            tools=oa_tools,
            messages=self.messages,
        )
        msg = resp.choices[0].message
        self.messages.append(msg.model_dump())

        if msg.tool_calls:
            call = msg.tool_calls[0]
            args = json.loads(call.function.arguments or "{}")
            result = await self.session.call_tool(call.function.name, args)
            self.messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.function.name,
                "content": result.content,
            })
            follow = self.openai.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
            )
            return follow.choices[0].message.content
        return msg.content or ""


def convert_tool_format(tool):
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": tool.inputSchema["properties"],
                "required": tool.inputSchema.get("required", []),
            },
        },
    }


async def main():
    dotenv.load_dotenv()
    model_name = "nvidia/nemotron-3-ultra-550b-a55b:free" # "openai/gpt-oss-20b:free"
    os.environ["OPENAI_API_KEY"] = os.getenv('OPENROUTER_API_KEY', '')
    os.environ["OPENAI_BASE_URL"] = os.getenv('OPENROUTER_BASE_URL', '')
    SERVER_CONFIG = {
        "command": "python",
        "args": ["-m", "mcp_server_time"],
        "env": None,
    }
    client = MCPClient(model_name)
    await client.connect(SERVER_CONFIG)
    print(await client.ask("Give me a current time in Europe/London timezone"))


if __name__ == "__main__":
    asyncio.run(main())
