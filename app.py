from flask import Flask, request, jsonify
import asyncio
from client.openai_client import MCPOpenAIClient
from router.router import call_agent

import os
import uuid
from datetime import datetime
import json

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

@app.route("/analyze-image", methods=["POST"])
def analyze_image_route():
    if "image" not in request.files or "question" not in request.form:
        return jsonify({"error": "Missing 'image' or 'question'"}), 400

    image_file = request.files["image"]
    question = request.form["question"]
    image_bytes = image_file.read()

    # Save image locally
    image_id = str(uuid.uuid4())
    image_ext = os.path.splitext(image_file.filename)[-1]
    image_path = f"logs/{image_id}{image_ext}"
    with open(image_path, "wb") as f:
        f.write(image_bytes)

    # Call GPT-4o
    response_text = loop.run_until_complete(client.analyze_image(image_bytes, question))

    # Log metadata
    log = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "image_path": image_path,
        "response": response_text,
    }

    with open("logs/image_logs.jsonl", "a") as f:
        f.write(json.dumps(log) + "\n")

    return jsonify({"response": response_text})

@app.route("/multi-agent", methods=["POST"])
def multi_agent():
    query = request.json.get("query", "")
    user_id = request.json.get("user_id", "default")
    result = loop.run_until_complete(client.process_query(query, user_id))

    return jsonify({
        "response": result["response"],
        "trace": result["trace"]  # contains tool_name, args, tool_response
    })

if __name__ == "__main__":
    loop.run_until_complete(client.connect_to_servers({
        "SupplyChainServer": "server/supply_data_server.py",
        "ForecastAgent": "server/forecast_agent_server.py"
    }))
    app.run(port=5001)
