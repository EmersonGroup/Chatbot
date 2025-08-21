import json
from typing import Any, Generator, Iterator
import base64
from PIL import Image

import pandas as pd
import requests
import snowflake.connector
import sseclient
import streamlit as st

# DATABASE = "OMEGA"
# SCHEMA = "PROD"
# STAGE = "CORTEX_STAGE"
# FILE = "cortex_test_v2.yaml"
DATABASE = "OMEGA"
SCHEMA = "PROD"
SEMANTIC_VIEW = "OMEGA.PROD.CORTEX_TEST_V1"

# --- Page Config with Omega logo as favicon (resized for clarity) ---
omega_icon = Image.open("assets/Omega logo v1.png").resize((64, 64))
st.set_page_config(
    page_title="OMEGA ChatBot",
    page_icon=omega_icon,
)

# =========================
# LOGOS (read first, then use in CSS)
# =========================
with open("assets/Omega logo v1.png", "rb") as f:
    omega_logo_base64 = base64.b64encode(f.read()).decode()

with open("assets/emersongroup_logo.png", "rb") as f:
    emerson_logo_base64 = base64.b64encode(f.read()).decode()

# =========================
# CSS
# =========================
st.markdown(
    f"""
    <style>
        .emerson-logo {{
            position: fixed;
            top: 60px;
            right: 20px;
            width: 90px;
            z-index: 1000;
        }}
        .omega-center {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            margin-top: 70px;
        }}
        .omega-center img {{
            width: 180px !important;
        }}
        .omega-center h1 {{
            font-size: 28px !important;
            margin: 12px 0 6px 0;
        }}
        .omega-center p {{
            font-size: 14px !important;
            color: gray !important;
            margin: 0;
        }}
        .stChatInputContainer {{
            margin-top: 0px !important;
            margin-bottom: 20px !important;
        }}
        .sample-title {{
            font-weight: 600;
            font-size: 15px;
            color: #333;
            margin-bottom: 6px;
        }}
        .stButton > button {{
            border-radius: 18px !important;
            border: 1px solid rgba(0, 0, 0, 0.1) !important;
            background-color: #f9f9f9 !important;
            backdrop-filter: blur(6px);
            color: rgba(0, 0, 0, 0.75) !important;
            font-size: 14px !important;
            font-weight: 400 !important;
            padding: 8px 14px !important;
            margin-bottom: 4px !important;
            text-align: left !important;
            white-space: normal !important;
        }}
        .stButton > button:hover {{
            background-color: rgba(0, 75, 135, 0.12) !important;
            border-color: #004b87 !important;
            color: #004b87 !important;
        }}
    </style>
    <img src="data:image/png;base64,{emerson_logo_base64}" class="emerson-logo">
    """,
    unsafe_allow_html=True,
)

# =========================
# OMEGA HEADER
# =========================
st.markdown(
    f"""
    <div class="omega-center">
        <img src="data:image/png;base64,{omega_logo_base64}">
        <h1>Ask. Explore. Discover. — with OMEGA</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# Single, fixed connection via service account
if "CONN" not in st.session_state or st.session_state.CONN is None:
    try:
        cfg = st.secrets["sf"]
        st.session_state.CONN = snowflake.connector.connect(
            account=cfg["account"],
            user=cfg["user"],
            password=cfg["password"],
            role=cfg["role"],
            warehouse=cfg["warehouse"],
            authenticator="snowflake",
        )
        st.success("Service account connected.")
    except Exception as e:
        st.error(f"Failed to connect to Snowflake: {e}")
        st.stop()

# ---- Session state init ----
if "messages" not in st.session_state:
    st.session_state.messages = []   # [{role: "user"/"analyst", content: [str | DataFrame | {'_omega_sql': str}]}]
    st.session_state.status = "Interpreting question"
    st.session_state.error = None

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# ---- Helpers ----
def fetch_initial_suggestions() -> list[str]:
    request_body = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        "semantic_view": SEMANTIC_VIEW,
        "stream": True,
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
    if resp.status_code >= 400:
        return []

    suggestions: dict[int, str] = {}
    events = sseclient.SSEClient(resp).events()
    for event in events:
        if event.event == "message.content.delta":
            data = json.loads(event.data)
            if data["type"] == "suggestions":
                idx = data["suggestions_delta"]["index"]
                delta = data["suggestions_delta"]["suggestion_delta"]
                suggestions[idx] = suggestions.get(idx, "") + delta
        elif event.event == "error":
            break
    return [s.strip() for _, s in sorted(suggestions.items())]

if "suggestions" not in st.session_state:
    st.session_state.suggestions = fetch_initial_suggestions()

def append_message(role: str, content: list[Any]) -> None:
    clean_content = [c for c in content if not (isinstance(c, str) and c.strip() == "")]
    if not clean_content:
        return
    if st.session_state.messages and st.session_state.messages[-1]["role"] == role:
        st.session_state.messages[-1]["content"].extend(clean_content)
    else:
        st.session_state.messages.append({"role": role, "content": clean_content})

def get_conversation_history() -> list[dict[str, Any]]:
    messages = []
    for msg in st.session_state.messages:
        m: dict[str, Any] = {}
        m["role"] = "user" if msg["role"] == "user" else "analyst"
        # Only send text content to Cortex
        text_content = "\n".join([c for c in msg["content"] if isinstance(c, str)])
        m["content"] = [{"type": "text", "text": text_content}]
        messages.append(m)
    return messages

def send_message() -> requests.Response:
    request_body = {
        "messages": get_conversation_history(),
        "semantic_view": SEMANTIC_VIEW,
        "stream": True,
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
    if resp.status_code < 400:
        return resp  # type: ignore
    else:
        raise Exception(f"Failed request with status {resp.status_code}: {resp.text}")

def display_df(df: pd.DataFrame) -> None:
    if len(df) == 0:
        st.info("No data returned.")
    else:
        st.dataframe(df)

# ---------- CORE CHANGE ----------
# Do NOT render any assistant content here. Just compute and append.
def process_message(prompt: str) -> None:
    text_buffer = []
    sql_buffer = []

    # Keep spinner visible while we compute, but don't render results here
    with st.chat_message("assistant"):
        with st.spinner("Omega is thinking..."):
            response = send_message()
            events = sseclient.SSEClient(response).events()  # type: ignore

            while st.session_state.status.lower() != "done":
                event = next(events, None)
                if not event:
                    break

                data = json.loads(event.data)
                if event.event == "message.content.delta":
                    if data["type"] == "text":
                        text_buffer.append(data["text_delta"])
                    elif data["type"] == "sql":
                        sql_buffer.append(data["statement_delta"])
                elif event.event == "status":
                    # Cortex marks completion with "done"
                    st.session_state.status = data["status_message"]
                elif event.event == "error":
                    st.session_state.error = data
                    st.error(f"Error: {data}", icon="🚨")
                    return

    # After stream completes:
    final_sql = "".join(sql_buffer).strip()
    interpretation = " ".join(text_buffer).strip()

    accumulated_content: list[Any] = []
    if interpretation:
        accumulated_content.append(interpretation)

    if final_sql:
        # Use Snowflake cursor to avoid pandas SQLAlchemy warning
        cur = st.session_state.CONN.cursor()
        try:
            cur.execute(final_sql)
            df = cur.fetch_pandas_all()
        finally:
            cur.close()

        accumulated_content.append(df)
        accumulated_content.append({"_omega_sql": final_sql})  # store SQL for history renderer

    st.session_state.status = "Interpreting question"
    append_message("analyst", accumulated_content)
# ---------------------------------

def show_conversation_history() -> None:
    for message in st.session_state.messages:
        chat_role = "assistant" if message["role"] == "analyst" else "user"
        with st.chat_message(chat_role):
            for content in message["content"]:
                if isinstance(content, pd.DataFrame):
                    display_df(content)
                elif isinstance(content, dict) and "_omega_sql" in content:
                    with st.expander("Show SQL query"):
                        st.code(content["_omega_sql"], language="sql")
                elif isinstance(content, Exception):
                    st.error(f"Error while processing request:\n {content}", icon="🚨")
                else:
                    st.write(content)

# --- Spacer pushes sample questions down to input ---
st.markdown("<div style='flex:1;'></div>", unsafe_allow_html=True)

# --- Placeholder for suggestions (so we can clear them before rerun) ---
sample_container = st.empty()

# --- Chat input (always bottom) ---
user_input = st.chat_input("What insight would you like to see?")

# --- Handle manual chat input ---
if user_input:
    # Append the user question immediately so it shows up right away
    append_message("user", [user_input])

    # Hide suggestions and process
    st.session_state.chat_started = True
    st.session_state.suggestions = []
    sample_container.empty()

    # Process and then re-run to show the new assistant message in history
    process_message(user_input)
    st.rerun()

# --- Render sample questions above input (only before first prompt) ---
if not st.session_state.get("chat_started") and st.session_state.get("suggestions"):
    with sample_container.container():
        st.markdown('<div class="sample-title">💡 Sample Questions</div>', unsafe_allow_html=True)
        for s in st.session_state.suggestions:
            if st.button(s, key=f"sample_{s}", use_container_width=True):
                # Append the user question immediately so it shows up right away
                append_message("user", [s])

                st.session_state.chat_started = True
                st.session_state.suggestions = []
                sample_container.empty()

                # Process and re-run (same as text input)
                process_message(s)
                st.rerun()

# --- Finally, render the full conversation (each item exactly once) ---
show_conversation_history()
