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
        self.session: Optional[ClientSession] = None

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

    async def connect_to_server(self, server_script_path: str = "server/supply_data_server.py"):
        server_params = StdioServerParameters(command="python", args=[server_script_path])
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()

        # Log available tools
        tools_result = await self.session.list_tools()
        print("\nConnected to MCP server with tools:")
        for tool in tools_result.tools:
            print(f"- {tool.name}: {tool.description}")

    async def get_mcp_tools(self) -> List[Dict[str, Any]]:
        tools_result = await self.session.list_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in tools_result.tools
        ]

    async def process_query(self, query: str) -> str:
        tools = await self.get_mcp_tools()
        trace = []  # for reasoning trace

        # Initial request
        response = await self.openai_client.chat.completions.create(
            messages=[{"role": "user", "content": query}],
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

                # Execute MCP tool
                result = await self.session.call_tool(tool_name, arguments=tool_args)

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
