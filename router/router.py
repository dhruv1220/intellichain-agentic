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
            with open(f"agents/" + fname) as f:
                card = json.load(f)
                cards[card["name"]] = card
    return cards

async def call_agent(query: str):
    agent_cards = load_agent_cards()
    trace_log = []
    tool_to_agent = {}
    agent_descriptions = {}

    # Step 1: Discover tools and build tool-agent map
    tools = []
    for agent in agent_cards.values():
        agent_descriptions[agent["name"]] = agent["description"]
        server_params = StdioServerParameters(command=agent["endpoint"], args=agent["args"])
        async with AsyncExitStack() as stack:
            read, write = await stack.enter_async_context(stdio_client(server_params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            tools_result = await session.list_tools()
            for tool in tools_result.tools:
                tool_to_agent[tool.name] = agent
                tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    }
                })

    # Step 2: System message to describe available agents and domains
    agent_guide = "\n".join([
        f"- {name}: {desc}" for name, desc in agent_descriptions.items()
    ])
    system_prompt = (
        "You are a multi-agent AI assistant.\n"
        "Available agents and their specialties:\n"
        f"{agent_guide}\n\n"
        "Use tools from the most relevant agent(s) to help answer the user's query."
    )

    # Step 3: Ask GPT to pick tools
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]

    initial_response = await GPT.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    assistant_msg = initial_response.choices[0].message
    messages.append(assistant_msg)

    if not assistant_msg.tool_calls:
        return {"response": assistant_msg.content, "trace": []}

    # Step 4: Execute tool calls and add result back to GPT
    for tool_call in assistant_msg.tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        trace_entry = {
            "step": len(trace_log) + 1,
            "tool": tool_name,
            "args": arguments, 
            "start": time.time()
        }

        agent_card = tool_to_agent.get(tool_name)
        if not agent_card:
            trace_entry["error"] = f"Tool '{tool_name}' not found in any agent"
            trace_log.append(trace_entry)
            continue

        server_params = StdioServerParameters(
            command=agent_card["endpoint"],
            args=agent_card["args"]
        )

        # Adding Agent Name to Trace Entry
        trace_entry["agent"] = agent_card["name"] 

        async with AsyncExitStack() as stack:
            read, write = await stack.enter_async_context(stdio_client(server_params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)

        tool_output = result.content[0].text if result.content else "⚠️ No output"
        trace_entry["end"] = time.time()
        trace_entry["duration"] = trace_entry["end"] - trace_entry["start"]
        trace_entry["result"] = tool_output

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_output,
        })

        trace_log.append(trace_entry)

    # Step 5: Final GPT response with tool results
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
