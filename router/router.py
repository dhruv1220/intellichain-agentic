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
            with open(f"agents/{fname}") as f:
                card = json.load(f)
                cards[card["name"]] = card
    return cards


def trace_to_messages(trace):
    msgs = []
    for entry in trace:
        if entry["type"] == "reasoning":
            msgs.append({
                "role": "assistant",
                "content": f"[Thought] {entry['reasoning']}"
            })
        else:  # tool entry
            payload = {
                "tool": entry["tool"],
                "args": entry["args"],
                "result": entry["result"]
            }
            msgs.append({
                "role": "assistant",
                "content": f"[Tool] {json.dumps(payload)}"
            })
    return msgs


async def call_agent(query: str):
    # 1) Discover all tools + schemas
    agent_cards = load_agent_cards()
    tool_to_agent = {}
    tool_schemas = {}
    for agent in agent_cards.values():
        params = StdioServerParameters(command=agent["endpoint"], args=agent["args"])
        async with AsyncExitStack() as stack:
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            tools_list = (await session.list_tools()).tools
            for t in tools_list:
                tool_to_agent[t.name] = agent
                tool_schemas[t.name] = t.inputSchema

    # 2) Build OpenAI-compatible `tools` array
    tools = []
    for name, schema in tool_schemas.items():
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": f"Auto-generated for {name}",
                "parameters": schema
            }
        })

    # 3) Build human-readable bullet list of tool signatures
    tool_lines = []
    for name, schema in tool_schemas.items():
        props = schema.get("properties", {})
        params = ", ".join(f"{p}: {props[p]['type']}" for p in props)
        tool_lines.append(f"- {name}({params})")
    tool_guide = "\n".join(tool_lines)

    # 4) System & user messages enforcing JSON-only schema
    system_msg = {
        "role": "system",
        "content": (
            "You are a multi-agent supply chain assistant.\n"
            "Only use the tools listed below. Do not invent or assume any other tools exist.\n\n"
            "Available tools:\n"
            f"{tool_guide}\n\n"
            "When responding, strictly use one of the following JSON formats:\n"
            "1) { \"reasoning\": \"...\", \"next_tool\": \"tool_name\", \"args\": { ... } }\n"
            "2) { \"reasoning\": \"...\", \"final_response\": \"...\" }\n\n"
            "Do not wrap the response in markdown. Do not include anything else. Only return a valid JSON object."
        )
    }
    user_msg = {"role": "user", "content": query}

    # 5) Trace & step counter
    trace = []
    step = 1

    # 6) First GPT turn: pick first action
    resp = await GPT.chat.completions.create(
        model=MODEL,
        messages=[system_msg, user_msg]
    )
    choice = json.loads(resp.choices[0].message.content)
    trace.append({"step": step, "type": "reasoning", "reasoning": choice["reasoning"]})
    step += 1

    # 7) Loop until we see `final_response`
    while "next_tool" in choice:
        tool_name = choice["next_tool"]
        args      = choice["args"]
        agent     = tool_to_agent.get(tool_name)
        if not agent:
            raise ValueError(f"Unknown tool requested: {tool_name}")

        # 7a) execute the tool
        params = StdioServerParameters(command=agent["endpoint"], args=agent["args"])
        async with AsyncExitStack() as stack:
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            t0 = time.time()
            res = await session.call_tool(tool_name, arguments=args)
            t1 = time.time()

        output = res.content[0].text if res.content else ""

        # 7b) log the tool call
        trace.append({
            "step": step,
            "type":    "tool",
            "agent":   agent["name"],
            "tool":    tool_name,
            "args":    args,
            "result":  output,
            "duration": round(t1 - t0, 3),
        })
        step += 1

        # 7c) Ask GPT what to do next, feeding in the tool result
        prompt_tool = {
            "role": "assistant",
            "content": f"[Tool Output] {output}"
        }


        prompt_reflect = {
            "role":    "system",
            "content": (
                "Given the above tool result, reply with JSON in one of these forms:\n"
                "1) { \"reasoning\": \"…\", \"next_tool\": \"tool_name\", \"args\": { … } }\n"
                "2) { \"reasoning\": \"…\", \"final_response\": \"…\" }"
            )
        }

        resp2 = await GPT.chat.completions.create(
            model=MODEL,
            messages=[
                system_msg,
                user_msg,
                *trace_to_messages(trace),
                prompt_tool,         # use assistant or user here
                prompt_reflect
            ]
        )
        choice = json.loads(resp2.choices[0].message.content)

        trace.append({"step": step, "type": "reasoning", "reasoning": choice["reasoning"]})
        step += 1

    # 8) Done!
    return {"response": choice["final_response"], "trace": trace}
