# 🧠 Intellichain Agentic

**Intellichain Agentic** is a multi-agent, multimodal Supply Chain Assistant built with OpenAI, MCP (Multi-agent Communication Protocol), and Streamlit UI. It supports dynamic tool chaining, agent-to-agent reasoning, and image-based querying — all with explainable trace logs.

---

## 🚀 Features

- 🔗 **Dynamic Tool Chaining**: GPT autonomously selects tools from connected MCP agents and invokes them with relevant arguments.
- 🧠 **Chain-of-Thought + A2A**: Agents share intermediate reasoning before and between tool calls.
- 🧰 **Multi-Agent Support**: Agent servers expose specialized tools via the MCP protocol.
- 🖼️ **Multimodal Queries**: Upload images and ask questions via `/analyze-image`, powered by GPT-4o.
- 📜 **Trace Log Viewer**: UI shows each reasoning step and tool execution in expandable trace logs.
- 🧠 **Short-Term Memory**: Retains last-used arguments for smoother tool re-use.
- 🧪 **Streamlit Chat UI**: Unified interface for querying tools, agents, and image-based tasks.

---

## 📦 Project Structure

```
intellichain-agentic/
│
├── app.py                     # Flask API
├── ui.py                      # Streamlit frontend
├── router/router.py           # Multi-agent orchestration logic
├── client/openai\_client.py    # Tool chaining + image analysis logic
├── server/                    # MCP-compatible agent servers
├── memory/session\_memory.py   # In-memory user session store
├── logs/                      # JSON logs of queries and tool calls
├── requirements.txt
└── README.md
```

---

## 📡 API Endpoints

| Endpoint             | Description                               |
|----------------------|-------------------------------------------|
| `/tool-chaining`     | Executes tool chain using GPT + MCP       |
| `/multi-agent`       | Multi-step reasoning with trace logging   |
| `/analyze-image`     | GPT-4o image question-answering           |
| `/`                  | Health check                              |

---

## 🛠️ Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/intellichain-agentic.git
cd intellichain-agentic

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add .env file with your Azure OpenAI creds
touch .env
# Add AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_MODEL, AZURE_OPENAI_API_VERSION

# 5. Launch backend
python app.py

# 6. In a new terminal, launch the frontend
streamlit run ui.py
````

---

## 🧠 Example Trace (Multi-Agent)

```json
{
  "response": "It is advisable to prioritize air shipping...",
  "trace": [
    {
      "reasoning": "To analyze delivery performance...",
      "type": "reasoning"
    },
    {
      "tool": "get_delay_stats",
      "agent": "DelayStatsAgent",
      "args": {},
      "result": "...",
      "type": "tool"
    },
    ...
  ]
}
```

---

## 💡 Inspiration

* [LangChain Agents](https://docs.langchain.com/docs/components/agents)
* [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
* [Multimodal GPT-4o](https://openai.com/index/gpt-4o/)

---

## 📜 License

MIT License

---

## 🙌 Acknowledgments

Built by Dhruv Arora with a focus on **transparent, multi-step decision-making in enterprise AI systems**.
