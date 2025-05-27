import asyncio
from client.openai_client import MCPOpenAIClient
import json
import os
import uuid
from datetime import datetime

# Initialize one instance for reuse
client = MCPOpenAIClient()

async def call_agent(query: str) -> str:
    # Connect to supply_data_server if not already connected
    if not client.session:
        await client.connect_to_server("server/supply_data_server.py")

    # Call OpenAI with GPT-based tool selection
    response = await client.process_query(query)

    # Log the interaction (optional, good for audit/debugging)
    log = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "response": response
    }
    os.makedirs("logs", exist_ok=True)
    with open("logs/agent_logs.jsonl", "a") as f:
        f.write(json.dumps(log) + "\n")

    return response
