import asyncio
import json
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
import re

REGION_MAP = {
    "asia pacific": ["Southeast Asia", "South Asia", "Eastern Asia"],
    "europe": ["Europe"],
    "north america": ["North America"],
    "south america": ["South America"],
    "oceania": ["Oceania"],
    "africa": ["Africa"]
}

def resolve_to_valid_regions(query: str) -> list:
    q = query.lower()
    for key, regions in REGION_MAP.items():
        if key in q:
            return regions
    return ["Southeast Asia"]  # fallback

def load_agent_cards():
    cards = {}
    for fname in os.listdir("agents"):
        if fname.endswith(".json"):
            with open(f"agents/{fname}") as f:
                card = json.load(f)
                cards[card["name"]] = card
    return cards

def extract_region_from_query(query: str) -> str:
    # Define common region aliases
    known_regions = [
        "Asia Pacific", "Europe", "North America", "South America", "Africa", "Australia"
    ]
    query_lower = query.lower()

    for region in known_regions:
        if region.lower() in query_lower:
            return region

    # Fallback: look for last word after 'in' or 'for'
    match = re.search(r"(in|for)\s+([\w\s]+)", query_lower)
    if match:
        return match.group(2).strip().title()

    return "Asia Pacific"  # default fallback

async def call_agent(query: str):
    cards = load_agent_cards()

    if any(word in query.lower() for word in ["forecast", "demand", "trend"]):
        agent = cards["ForecastAgent"]
        tool = "forecast_demand"
        region_list = resolve_to_valid_regions(query)
        tool_args = {"regions": region_list}
    else:
        agent = cards["DelayStatsAgent"]
        tool = "get_delay_stats"
        tool_args = {}

    server_params = StdioServerParameters(
        command=agent["endpoint"],
        args=agent["args"]
    )

    async with AsyncExitStack() as stack:
        read_stream, write_stream = await stack.enter_async_context(stdio_client(server_params))
        session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()
        result = await session.call_tool(tool, arguments=tool_args)
        return result.content[0].text
