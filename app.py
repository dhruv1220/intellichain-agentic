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

@app.route("/analyze-image", methods=["POST"])
def analyze_image_route():
    if "image" not in request.files or "question" not in request.form:
        return jsonify({"error": "Missing 'image' or 'question'"}), 400

    image = request.files["image"].read()
    question = request.form["question"]
    response = loop.run_until_complete(client.analyze_image(image, question))
    return jsonify({"response": response})

if __name__ == "__main__":
    loop.run_until_complete(client.connect_to_server("server/supply_data_server.py"))
    app.run(port=5001)
