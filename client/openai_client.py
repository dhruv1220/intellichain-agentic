import asyncio
import json
import os
import base64
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional
from datetime import datetime

import nest_asyncio
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncAzureOpenAI
from openai.types.chat import ChatCompletionMessageParam
from memory.session_memory import MemoryStore

nest_asyncio.apply()
load_dotenv()

def log_tool_usage(tool_name: str, arguments: Dict, response: str, user_query: str, reasoning: str = ""):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "tool_name": tool_name,
        "arguments": arguments,
        "response": response,
        "user_query": user_query,
        "reasoning": reasoning
    }

    os.makedirs("logs", exist_ok=True)
    with open("logs/tool_usage_logs.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

class MCPOpenAIClient:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.sessions = {}  # key: server name, value: (session, tool names)
        self.memory = MemoryStore()

        # Azure OpenAI configuration
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment = os.getenv("AZURE_OPENAI_MODEL") or os.getenv("AZURE_OPENAI_DEPLOYMENT")

        self.openai_client = AsyncAzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
        )

    async def connect_to_servers(self, server_map: Dict[str, str]):
        """
        server_map = {
            "SupplyChainServer": "server/supply_data_server.py",
            "ForecastAgent": "server/forecast_agent_server.py"
        }
        """
        for server_name, path in server_map.items():
            params = StdioServerParameters(command="python", args=[path])
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(params))
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            tools_result = await session.list_tools()
            tool_names = [tool.name for tool in tools_result.tools]
            self.sessions[server_name] = (session, tool_names)

    async def get_mcp_tools(self) -> List[Dict[str, Any]]:
        all_tools = []
        for session, _ in self.sessions.values():
            tools_result = await session.list_tools()
            for tool in tools_result.tools:
                all_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    },
                })
        return all_tools

    async def process_query(self, query: str, user_id: str = "default") -> dict:
        tools = await self.get_mcp_tools()
        trace = []  # for reasoning trace

        # Inject memory into prompt
        memory_context = self.memory.get(user_id)
        system_prompt = f"You are a helpful supply chain assistant.\n\nUser preferences:\n{json.dumps(memory_context, indent=2)}"

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}]

        # Initial request
        response = await self.openai_client.chat.completions.create(
            messages=messages,
            model=self.deployment,
            tools=tools,
            tool_choice="auto",
        )

        assistant_message = response.choices[0].message
        messages = [{"role": "user", "content": query}, assistant_message]

        # First round tool call
        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                # Fallback to memory if some arguments are missing
                if user_id:
                    memory_data = self.memory.get(user_id)
                    memory_args = memory_data.get("last_tool_args", {})

                    for key, value in memory_args.items():
                        if key not in tool_args or not tool_args[key]:
                            tool_args[key] = value

                # Determine which session owns the tool
                session = None
                for _, (sess, tool_names) in self.sessions.items():
                    if tool_name in tool_names:
                        session = sess
                        break

                if session is None:
                    raise ValueError(f"Tool '{tool_name}' not found in any connected MCP server")

                result = await session.call_tool(
                    tool_name, arguments=tool_args
                )

                tool_output = result.content[0].text if result.content else "⚠️ Tool returned no output"

                # Add to reasoning trace
                trace.append({
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "tool_response": tool_output,
                })

                # Feed result back to GPT
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_output,
                })

                self.memory.update(user_id, "last_tool_args", tool_args)

            self.memory.append_to_list(user_id, "recent_queries", query)
        
            # Final GPT response incorporating tool results
            final_response = await self.openai_client.chat.completions.create(
                messages=messages,
                model=self.deployment,
                tools=tools,
                tool_choice="none",
            )
            final_text = final_response.choices[0].message.content

        else:
            final_text = assistant_message.content

        # === Intelligent Logging ===
        log = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "reasoning_trace": trace,
            "final_response": final_text,
        }
        os.makedirs("logs", exist_ok=True)
        with open("logs/agent_trace_logs.jsonl", "a") as f:
            f.write(json.dumps(log) + "\n")

            return {
                "response": final_text,
                "trace": trace
            }


    async def analyze_image(self, image_bytes: bytes, question: str) -> str:
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": "You are a supply chain expert."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": question},
                ],
            },
        ]

        response = await self.openai_client.chat.completions.create(
            model=self.deployment,
            messages=messages,
        )

        return response.choices[0].message.content

    async def cleanup(self):
        await self.exit_stack.aclose()
