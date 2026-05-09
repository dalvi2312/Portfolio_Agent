"""
streamlit_app.py
────────────────
Optional Streamlit graphical interface for the Portfolio Agent.

Run with:
    streamlit run streamlit_app.py
"""

import os
import urllib.request

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Portfolio Intelligence Agent",
    page_icon="🏦",
    layout="wide",
)

# Sidebar
with st.sidebar:
    st.title("🏦 Portfolio Agent")
    st.caption(f"Model: {os.getenv('OLLAMA_MODEL', 'llama3:8b')}")
    st.markdown("---")
    st.markdown("**Example questions**")
    examples = [
        "How many portfolios do we have in total?",
        "What are the names of all active portfolios?",
        "Which securities are in the Technology sector?",
        "What is the total AUM for high risk portfolios?",
        "Show top 5 holdings by cost basis in the Growth Equity Fund",
        "What are the sector exposures for the Tech Innovation Fund?",
        "Calculate the sector exposure breakdown for international equity",
        "Find portfolios with holdings in more than 5 different sectors",
        "What is the average current price of securities in each sector?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=ex):
            st.session_state["prefill"] = ex

    st.markdown("---")
    st.caption("Powered by Ollama + LangGraph")

# Pre-flight: database
db_path = os.getenv("DB_PATH", "portfolio_database.db")
if not os.path.exists(db_path):
    st.error(
        f"Database not found at `{db_path}`. "
        "Run `python setup_database.py` first."
    )
    st.stop()

# Pre-flight: Ollama connectivity
base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
try:
    urllib.request.urlopen(f"{base_url}/api/tags", timeout=3)
except Exception:
    st.error(
        f"Cannot reach Ollama at `{base_url}`.\n\n"
        "Make sure it is running:\n"
        "```\nollama serve\nollama pull llama3:8b\n```"
    )
    st.stop()

# Agent (cached per session)
@st.cache_resource(show_spinner="Loading agent...")
def load_agent():
    from agent.agent import PortfolioAgent
    return PortfolioAgent()

agent = load_agent()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Portfolio Intelligence Agent 🏦")

# Render existing history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle sidebar example-button pre-fill
prefill = st.session_state.pop("prefill", None)

# Chat input
user_input = st.chat_input("Ask a question about your portfolio data...") or prefill

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = agent.answer_question(user_input)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
