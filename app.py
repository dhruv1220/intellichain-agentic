from flask import Flask, request, jsonify
import asyncio
from client.openai_client import MCPOpenAIClient

app = Flask(__name__)
client = MCPOpenAIClient()
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

@app.route("/ask", methods=["POST"])
def ask():
    query = request.json.get("query", "")
    response = loop.run_until_complete(client.process_query(query))
    return jsonify({"response": response})

@app.route("/")
def hello():
    return "MCP + OpenAI Supply Chain Assistant is live."

if __name__ == "__main__":
    loop.run_until_complete(client.connect_to_server("server/supply_data_server.py"))
    app.run(port=5001)
