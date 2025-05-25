import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client

async def main():
    async with sse_client("http://localhost:8050/sse") as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            print("Tools:", [tool.name for tool in tools_result.tools])

            result = await session.call_tool("get_delay_stats")
            print("Stats:\n", result.content[0].text)

if __name__ == "__main__":
    asyncio.run(main())