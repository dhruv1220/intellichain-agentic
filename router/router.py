import asyncio
import json
import os
import time
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv

load_dotenv()

REGION_MAP = {
    "asia pacific": ["Southeast Asia", "South Asia", "Eastern Asia"],
    "europe": ["Europe"],
    "north america": ["North America"],
    "south america": ["South America"],
    "oceania": ["Oceania"],
    "africa": ["Africa"]
}

GPT = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

MODEL = os.getenv("AZURE_OPENAI_MODEL")

def load_agent_cards():
    cards = {}
    for fname in os.listdir("agents"):
        if fname.endswith(".json"):
            with open(f"agents/{fname}") as f:
                card = json.load(f)
                cards[card["name"]] = card
    return cards

async def call_agent(query: str):
    agent_cards = load_agent_cards()
    trace_log = []

    # Step 1: Initialize with GPT
    tools = []
    for agent in agent_cards.values():
        server_params = StdioServerParameters(command=agent["endpoint"], args=agent["args"])
        async with AsyncExitStack() as stack:
            read, write = await stack.enter_async_context(stdio_client(server_params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            tools_result = await session.list_tools()
            for tool in tools_result.tools:
                tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    }
                })

    # Step 2: GPT selects tools & arguments
    initial_response = await GPT.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": query}],
        tools=tools,
        tool_choice="auto"
    )

    assistant_msg = initial_response.choices[0].message
    messages = [{"role": "user", "content": query}, assistant_msg]

    if not assistant_msg.tool_calls:
        return {"response": assistant_msg.content, "trace": []}

    # Step 3: Execute tool calls
    for tool_call in assistant_msg.tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        trace_entry = {
            "step": len(trace_log) + 1,
            "tool": tool_name,
            "args": arguments,
            "start": time.time()
        }

        # Find which agent has this tool
        target_card = None
        for card in agent_cards.values():
            if tool_name in card["tools"]:
                target_card = card
                break

        if not target_card:
            trace_entry["error"] = f"Tool '{tool_name}' not found in agent registry"
            trace_log.append(trace_entry)
            continue

        # Connect and call tool
        server_params = StdioServerParameters(
            command=target_card["endpoint"], args=target_card["args"]
        )
        async with AsyncExitStack() as stack:
            read, write = await stack.enter_async_context(stdio_client(server_params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)

        trace_entry["end"] = time.time()
        trace_entry["duration"] = trace_entry["end"] - trace_entry["start"]
        trace_entry["result"] = result.content[0].text

        # Add tool result to chat for possible chaining
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result.content[0].text,
        })

        trace_log.append(trace_entry)

    # Step 4: Final GPT response (optional)
    final_response = await GPT.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="none"
    )

    return {
        "response": final_response.choices[0].message.content,
        "trace": trace_log
    }
