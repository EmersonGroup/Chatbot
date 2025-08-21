import json
import re
from typing import Any, Generator, Iterator
import base64
from PIL import Image

import pandas
import pandas as pd
import requests
import snowflake.connector
import sseclient
import streamlit as st
import altair as alt

# DATABASE / VIEW
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
        /* Emerson logo pinned top-right (under Share/GitHub) */
        .emerson-logo {{
            position: fixed;
            top: 60px;
            right: 20px;
            width: 90px;
            z-index: 1000;
        }}

        /* Omega logo + title centered */
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
    </style>

    <!-- Emerson logo pinned -->
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
            account       = cfg["account"],
            user          = cfg["user"],
            password      = cfg["password"],
            role          = cfg["role"],
            warehouse     = cfg["warehouse"],
            authenticator = "snowflake",   # service account (no SSO)
        )
        st.success("Service account connected.")
    except Exception as e:
        st.error(f"Failed to connect to Snowflake: {e}")
        st.stop()

# -----------------------
# Session state inits
# -----------------------
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.status = "idle"
    st.session_state.error = None



def fetch_initial_suggestions() -> list[str]:
    """Fetch default recommendations from Cortex Analyst."""
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

# Initialize suggestions once
if "suggestions" not in st.session_state:
    st.session_state.suggestions = fetch_initial_suggestions()

def get_conversation_history() -> list[dict[str, Any]]:
    """Only send plain text content to Cortex."""
    messages = []
    for msg in st.session_state.messages:
        m: dict[str, Any] = {}
        m["role"] = "user" if msg["role"] == "user" else "analyst"
        text_content = "\n".join([c for c in msg["content"] if isinstance(c, str)])
        m["content"] = [{"type": "text", "text": text_content}]
        messages.append(m)
    return messages

def send_message() -> requests.Response:
    """Calls the REST API and returns a streaming client."""
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

def append_message(role: str, content: list[Any]) -> None:
    """Append one message to our persisted history."""
    clean_content = [c for c in content if not (isinstance(c, str) and c.strip() == "")]
    if not clean_content:
        return
    if st.session_state.messages and st.session_state.messages[-1]["role"] == role:
        st.session_state.messages[-1]["content"].extend(clean_content)
    else:
        st.session_state.messages.append({"role": role, "content": clean_content})

# =========================
# LIVE RENDER this turn, then append and rerun
# =========================
def process_message(prompt: str) -> None:
    """Live-render this turn (user + spinner + results), then append to history and rerun."""

    # Show user bubble
    with st.chat_message("user"):
        st.markdown(prompt)

    # Initialize buffers
    text_buffer: list[str] = []
    sql_buffer: list[str] = []
    accumulated_content: list[Any] = []

    # Assistant bubble: we stream everything inside here
    with st.chat_message("assistant"):
        with st.spinner("Omega is thinking..."):

            st.session_state.status = "starting"
            response = send_message()
            events = sseclient.SSEClient(response).events()  # type: ignore

            for event in events:
                if not event:
                    break

                if event.event == "message.content.delta":
                    data = json.loads(event.data)
                    if data["type"] == "text":
                        text_buffer.append(data["text_delta"])
                    elif data["type"] == "sql":
                        sql_buffer.append(data["statement_delta"])

                elif event.event == "status":
                    data = json.loads(event.data)
                    st.session_state.status = data.get("status_message", "")
                    if st.session_state.status.lower() == "done":
                        break

                elif event.event == "error":
                    data = json.loads(event.data)
                    st.error(f"Error: {data}", icon="🚨")
                    return

            final_sql = "".join(sql_buffer).strip()
            interpretation = " ".join(text_buffer).strip()

            # Show interpretation first
            if interpretation:
                st.info(interpretation)
                accumulated_content.append(interpretation)

            # Then run and show results
            if final_sql:
                with st.spinner("Executing query..."):
                    df = pd.read_sql(final_sql, st.session_state.CONN)
                    accumulated_content.append(df)
                    display_df(df)

                with st.expander("Show SQL query"):
                    st.code(final_sql, language="sql")
                accumulated_content.append({"_omega_sql": final_sql})

    # Save assistant message to history
    append_message("analyst", accumulated_content)
    st.session_state.status = "idle"
    st.rerun()


# =========================
# HISTORY (simple, no skip-tail)
# =========================
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

# --- Placeholder for sample questions (just above chat input) ---
sample_container = st.container()

# --- CSS for professional style ---
st.markdown(
    """
    <style>
        .stChatInputContainer {
            margin-top: 0px !important;
            margin-bottom: 20px !important;
        }
        .sample-title {
            font-weight: 600;
            font-size: 15px;
            color: #333;
            margin-bottom: 6px;
        }
        .stButton > button {
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
        }
        .stButton > button:hover {
            background-color: rgba(0, 75, 135, 0.12) !important;
            border-color: #004b87 !important;
            color: #004b87 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Placeholder we can clear before rerun ---
sample_container = st.empty()

# --- Chat input (always bottom) ---
user_input = st.chat_input("What insight would you like to see?")

# --- Handle manual chat input ---
if user_input:
    st.session_state.chat_started = True
    st.session_state.pending_prompt = user_input
    st.session_state.suggestions = []
    sample_container.empty()
    st.rerun()

# --- Render sample questions above input (only before first prompt) ---
if not st.session_state.get("chat_started") and st.session_state.get("suggestions"):
    with sample_container.container():
        st.markdown('<div class="sample-title">💡 Try This..</div>', unsafe_allow_html=True)
        for s in st.session_state.suggestions:
            if st.button(s, key=f"sample_{s}", use_container_width=True):
                st.session_state.chat_started = True
                st.session_state.pending_prompt = s
                st.session_state.suggestions = []
                sample_container.empty()
                st.rerun()


# --- Show history when not actively processing ---
show_conversation_history()


# --- Process pending prompt (typed or button) ---
if st.session_state.get("pending_prompt"):
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

    # Append user to history once (persisted)
    append_message("user", [prompt])

    # Live-render this turn; will append assistant and rerun internally
    process_message(prompt)



    