import json
import re
from typing import Any
import base64
from PIL import Image

import pandas as pd
import requests
import snowflake.connector
import sseclient
import streamlit as st

# --- CONFIG ---
DATABASE = "OMEGA"
SCHEMA = "PROD"
SEMANTIC_VIEW = "OMEGA.PROD.CORTEX_TEST_V1"

# --- Setup page + logos ---
omega_icon = Image.open("assets/Omega logo v1.png").resize((64, 64))
st.set_page_config(page_title="OMEGA ChatBot", page_icon=omega_icon)

with open("assets/Omega logo v1.png", "rb") as f:
    omega_logo_base64 = base64.b64encode(f.read()).decode()
with open("assets/emersongroup_logo.png", "rb") as f:
    emerson_logo_base64 = base64.b64encode(f.read()).decode()

st.markdown(f"""
    <style>
        .emerson-logo {{ position: fixed; top: 60px; right: 20px; width: 90px; z-index: 1000; }}
        .omega-center {{ display: flex; flex-direction: column; align-items: center; margin-top: 70px; }}
        .omega-center img {{ width: 180px !important; }}
        .omega-center h1 {{ font-size: 28px; margin: 12px 0 6px 0; }}
    </style>
    <img src="data:image/png;base64,{emerson_logo_base64}" class="emerson-logo">
""", unsafe_allow_html=True)

st.markdown(f"""
    <div class="omega-center">
        <img src="data:image/png;base64,{omega_logo_base64}">
        <h1>Ask. Explore. Discover. — with OMEGA</h1>
    </div>
""", unsafe_allow_html=True)

# --- Init connection ---
if "CONN" not in st.session_state:
    cfg = st.secrets["sf"]
    st.session_state.CONN = snowflake.connector.connect(
        account=cfg["account"], user=cfg["user"], password=cfg["password"],
        role=cfg["role"], warehouse=cfg["warehouse"], authenticator="snowflake")

# --- Init state ---
for key in ["pending_prompt", "messages", "status", "error", "suggestions"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "messages" else None
st.session_state.status = "idle"

# --- Helper: Fetch starter suggestions ---
def fetch_initial_suggestions():
    request_body = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        "semantic_view": SEMANTIC_VIEW, "stream": True,
    }
    resp = requests.post(
        url=f"https://{st.session_state.CONN.host}/api/v2/cortex/analyst/message",
        json=request_body,
        headers={
            "Authorization": f'Snowflake Token="{st.session_state.CONN.rest.token}"',
            "Content-Type": "application/json",
        },
        stream=True,
    )
    suggestions = {}
    for event in sseclient.SSEClient(resp).events():
        if event.event == "message.content.delta":
            data = json.loads(event.data)
            if data["type"] == "suggestions":
                idx = data["suggestions_delta"]["index"]
                delta = data["suggestions_delta"]["suggestion_delta"]
                suggestions[idx] = suggestions.get(idx, "") + delta
    return [s.strip() for _, s in sorted(suggestions.items())]

if not st.session_state.suggestions:
    st.session_state.suggestions = fetch_initial_suggestions()

# --- Helper: Minimal content history ---
def get_conversation_history():
    messages = []
    for msg in st.session_state.messages:
        text = "\n".join([c for c in msg["content"] if isinstance(c, str)])
        messages.append({"role": "user" if msg["role"] == "user" else "analyst", "content": [{"type": "text", "text": text}]})
    return messages

# --- Helper: Stream and display message ---
def send_message():
    body = {"messages": get_conversation_history(), "semantic_view": SEMANTIC_VIEW, "stream": True}
    return requests.post(
        f"https://{st.session_state.CONN.host}/api/v2/cortex/analyst/message",
        json=body,
        headers={
            "Authorization": f'Snowflake Token="{st.session_state.CONN.rest.token}"',
            "Content-Type": "application/json",
        },
        stream=True,
    )

# --- Display df helper ---
def display_df(df):
    st.dataframe(df) if not df.empty else st.info("No data returned.")

# --- Save message ---
def append_message(role, content):
    if not content: return
    if st.session_state.messages and st.session_state.messages[-1]["role"] == role:
        st.session_state.messages[-1]["content"].extend(content)
    else:
        st.session_state.messages.append({"role": role, "content": content})

# --- Core processor ---
def process_message(prompt):
    with st.chat_message("user"):
        st.markdown(prompt)

    text_buffer, sql_buffer, output_content = [], [], []

    with st.chat_message("assistant"):
        with st.spinner("Omega is thinking..."):
            st.session_state.status = "starting"
            response = send_message()
            events = sseclient.SSEClient(response).events()

            for event in events:
                if event.event == "message.content.delta":
                    data = json.loads(event.data)
                    if data["type"] == "text":
                        text_buffer.append(data["text_delta"])
                    elif data["type"] == "sql":
                        sql_buffer.append(data["statement_delta"])
                elif event.event == "status":
                    st.session_state.status = json.loads(event.data).get("status_message", "")
                    if st.session_state.status.lower() == "done": break
                elif event.event == "error":
                    st.error("Error: " + json.loads(event.data), icon="🚨")
                    return

            interpretation = " ".join(text_buffer).strip()
            sql_query = "".join(sql_buffer).strip()

            if interpretation:
                st.info(interpretation)
                output_content.append(interpretation)

            if sql_query:
                with st.spinner("Executing query..."):
                    df = pd.read_sql(sql_query, st.session_state.CONN)
                    output_content.append(df)
                    display_df(df)
                with st.expander("Show SQL query"):
                    st.code(sql_query, language="sql")
                output_content.append({"_omega_sql": sql_query})

    append_message("analyst", output_content)
    st.session_state.status = "idle"
    st.rerun()

# --- Replay history ---
def show_conversation_history():
    for msg in st.session_state.messages:
        role = "assistant" if msg["role"] == "analyst" else "user"
        with st.chat_message(role):
            for c in msg["content"]:
                if isinstance(c, pd.DataFrame):
                    display_df(c)
                elif isinstance(c, dict) and "_omega_sql" in c:
                    with st.expander("Show SQL query"):
                        st.code(c["_omega_sql"], language="sql")
                elif isinstance(c, Exception):
                    st.error(f"Error while processing request:\n {c}", icon="🚨")
                else:
                    st.write(c)

# --- Style ---
st.markdown("""
<style>
    .stChatInputContainer { margin-top: 0px !important; margin-bottom: 20px !important; }
    .sample-title { font-weight: 600; font-size: 15px; margin-bottom: 6px; color: #333; }
    .stButton > button {
        border-radius: 18px; border: 1px solid rgba(0,0,0,0.1);
        background-color: #f9f9f9; color: rgba(0,0,0,0.75);
        font-size: 14px; font-weight: 400;
        padding: 8px 14px; margin-bottom: 4px;
        text-align: left; white-space: normal;
    }
    .stButton > button:hover {
        background-color: rgba(0,75,135,0.12);
        border-color: #004b87; color: #004b87;
    }
</style>
""", unsafe_allow_html=True)

# --- Chat input + suggestions ---
sample_container = st.empty()
user_input = st.chat_input("What insight would you like to see?")

if user_input:
    st.session_state.pending_prompt = user_input
    st.session_state.suggestions = []
    sample_container.empty()
    st.rerun()

if not st.session_state.messages and st.session_state.suggestions:
    with sample_container.container():
        st.markdown('<div class="sample-title">💡 Sample Questions</div>', unsafe_allow_html=True)
        for s in st.session_state.suggestions:
            if st.button(s, key=f"sample_{s}", use_container_width=True):
                st.session_state.pending_prompt = s
                st.session_state.suggestions = []
                sample_container.empty()
                st.rerun()

# --- On rerun, process pending ---
if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
    append_message("user", [prompt])
    process_message(prompt)

# --- Show full chat ---
if not st.session_state.pending_prompt:
    show_conversation_history()
