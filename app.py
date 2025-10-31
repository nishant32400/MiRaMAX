# app.py
import asyncio
import json
import streamlit as st
from dotenv import load_dotenv
from client import FlightOpsMCPClient

load_dotenv()

st.set_page_config(page_title="FlightOps Smart Agent (Groq MCP)", layout="wide")

st.title("✈️ FlightOps — Groq + MCP Chatbot")
st.caption("Ask any flight operations question. The LLM plans tool calls → MCP server executes → Groq summarizes.")

# Create global event loop if not exists
if "event_loop" not in st.session_state:
    st.session_state.event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(st.session_state.event_loop)

loop = st.session_state.event_loop

# Initialize MCP client once per session
if "mcp_client" not in st.session_state:
    st.session_state.mcp_client = FlightOpsMCPClient()
mcp_client = st.session_state.mcp_client

# Connect once per app session
if "mcp_connected" not in st.session_state:
    try:
        loop.run_until_complete(mcp_client.connect())
        st.session_state.mcp_connected = True
        st.success("✅ Connected to MCP server")
    except Exception as e:
        st.error(f"❌ Could not connect to MCP server.\n\n{e}")
        st.stop()

with st.sidebar:
    st.markdown("## Server / LLM Info")
    st.write("**MCP Server:**", mcp_client.base_url)
    st.write("**LLM Model:**", "Groq - llama3-70b-8192")

st.markdown("### 💬 Example questions")
st.write("- Why was flight **6E215** delayed on **June 23, 2024**?")
st.write("- Show **aircraft** and **delay info** for **6E215**.")
st.write("- What were **operation times** for **6E215 on 2024-06-23**?")
st.write("---")

user_query = st.text_area("Your question:", height=100, key="query_box")

if st.button("Ask"):
    if not user_query.strip():
        st.warning("Please enter a question.")
        st.stop()

    st.info("🧠 Thinking with Groq LLM to plan the query...")
    with st.spinner("Generating tool plan and fetching results..."):
        try:
            # ✅ Use the same event loop, don't recreate
            result = loop.run_until_complete(mcp_client.run_query(user_query))
        except Exception as e:
            st.error(f"❌ Error during query:\n{e}")
            st.stop()

    plan = result.get("plan", [])
    if not plan:
        st.warning("LLM did not produce a valid tool plan.")
        st.json(result)
        st.stop()

    st.subheader("🗂️ LLM Tool Plan")
    st.json(plan)

    results = result.get("results", [])
    if results:
        st.subheader("🔧 MCP Tool Results")
        for step in results:
            tool_name = list(step.keys())[0]
            tool_result = step[tool_name]
            st.markdown(f"**Tool:** `{tool_name}`")
            st.json(tool_result)

    summary = result.get("summary", {}).get("summary", "")
    if summary:
        st.subheader("📝 Final Summary")
        st.write(summary)
    else:
        st.warning("No summary returned by Groq.")

    st.session_state.last_result = result

# Show previous result
if "last_result" in st.session_state:
    with st.expander("📦 Previous Results"):
        st.json(st.session_state.last_result)
