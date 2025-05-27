import streamlit as st
import requests
import uuid
import base64

# Set the backend URL
BACKEND_URL = "http://localhost:5001"  # Update if different

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = "multi-agent"  # Options: multi-agent, tool-chaining, analyze-image

# Sidebar for mode selection
st.sidebar.title("🛠️ Chat Mode")
st.session_state.chat_mode = st.sidebar.radio(
    "Select Mode:",
    ("multi-agent", "tool-chaining", "analyze-image"),
    format_func=lambda x: {
        "multi-agent": "Multi-Agent Reasoning",
        "tool-chaining": "Tool Chaining",
        "analyze-image": "Analyze Image"
    }[x]
)

st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using Streamlit")

# Main title
st.title("🤖 MCP Supply Chain Assistant")

# Display chat history
for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(entry["user"])

    with st.chat_message("assistant"):
        st.markdown(entry["response"])

        # Display trace in an expander if available
        if "trace" in entry:
            with st.expander("🔍 View Reasoning Trace"):
                for step in entry["trace"]:
                    if "type" in step:
                        if step["type"] == "reasoning":
                            st.markdown(f"**Step {step['step']} - Reasoning:** {step['reasoning']}")
                        elif step["type"] == "tool":
                            st.markdown(f"**Step {step['step']} - Tool Used:** `{step['tool']}`")
                            st.markdown(f"- **Agent:** {step['agent']}")
                            st.markdown(f"- **Arguments:** `{step['args']}`")
                            st.markdown(f"- **Result:** `{step['result']}`")
                            st.markdown(f"- **Duration:** {step['duration']} seconds")
                    else:
                        st.markdown(f"**Tool:** `{step.get('tool_name')}`")
                        st.markdown(f"- **Arguments:** `{step.get('tool_args')}`")
                        st.markdown(f"- **Response:** `{step.get('tool_response')}`")


# Chat input
if st.session_state.chat_mode in ["multi-agent", "tool-chaining"]:
    prompt = st.chat_input("Type your query here...")
    if prompt:
        with st.spinner("Processing..."):
            endpoint = "/multi-agent" if st.session_state.chat_mode == "multi-agent" else "/tool-chaining"
            try:
                response = requests.post(
                    f"{BACKEND_URL}{endpoint}",
                    json={"query": prompt}
                )
                response.raise_for_status()
                data = response.json()
                st.session_state.chat_history.append({
                    "user": prompt,
                    "response": data["response"],
                    "trace": data.get("trace", [])
                })
                st.rerun()
            except requests.exceptions.RequestException as e:
                st.error(f"Error: {e}")

# Image analysis input
elif st.session_state.chat_mode == "analyze-image":
    with st.form(key="image_form"):
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        question = st.text_input("Ask a question about the image:")
        submit_button = st.form_submit_button(label="Analyze Image")

    if submit_button and uploaded_file and question:
        with st.spinner("Analyzing image..."):
            files = {"image": uploaded_file.getvalue()}
            data = {"question": question}
            try:
                response = requests.post(
                    f"{BACKEND_URL}/analyze-image",
                    files={"image": (uploaded_file.name, uploaded_file.getvalue())},
                    data={"question": question}
                )
                response.raise_for_status()
                result = response.json()
                st.session_state.chat_history.append({
                    "user": f"**Question:** {question}\n\n![Uploaded Image](data:image/jpeg;base64,{base64.b64encode(uploaded_file.getvalue()).decode()})",
                    "response": result["response"]
                })
                st.rerun()
            except requests.exceptions.RequestException as e:
                st.error(f"Error: {e}")
