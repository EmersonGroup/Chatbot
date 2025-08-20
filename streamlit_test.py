
import json
import re
from typing import Any, Generator, Iterator
import base64

import pandas
import pandas as pd
import requests
import snowflake.connector
import sseclient
import streamlit as st

#DATABASE = "OMEGA"
#SCHEMA = "PROD"
#STAGE = "CORTEX_STAGE"
#FILE = "cortex_test_v2.yaml"
DATABASE = "OMEGA"
SCHEMA = "PROD"
SEMANTIC_VIEW = "OMEGA.PROD.CORTEX_TEST_V1"


st.set_page_config(page_title="OMEGA ChatBot", page_icon=":bar_chart:", layout="wide")

# Read logo as base64 (so it works on Streamlit Cloud too)
with open("assets/Omega logo v1.png", "rb") as f:   # <-- adjust path if needed
    omega_logo_base64 = base64.b64encode(f.read()).decode()

with open("assets/emersongroup_logo.png", "rb") as f:
    emerson_logo_base64 = base64.b64encode(f.read()).decode()

# --- CSS for Logos ---
st.markdown(
    f"""
    <style>
        /* Emerson logo pinned top-right under Share */
        .emerson-logo {{
            position: fixed;
            top: 8px;
            right: 20px;
            width: 90px;
            z-index: 1000;
        }}

        /* Omega centered (initial) */
        .omega-center {{
            text-align: center;
            margin-top: 70px;
        }}
        .omega-center img {{
            width: 140px !important;   /* moderate size */
        }}
        .omega-center h1 {{
            font-size: 26px !important;
            margin: 8px 0 4px 0;
        }}
        .omega-center p {{
            font-size: 14px !important;
            color: gray !important;
            margin: 0;
        }}

        /* Omega shrinks top-left (after chat starts) */
        .omega-top-left {{
            position: fixed;
            top: 8px;
            left: 20px;
            z-index: 999;
            display: flex;
            align-items: center;
        }}
        .omega-top-left img {{
            width: 45px !important;
            margin-right: 8px;
        }}
        .omega-top-left h1 {{
            font-size: 18px !important;
            margin: 0;
        }}
        .omega-top-left p {{
            font-size: 11px !important;
            margin: 0 0 0 10px;
            color: gray !important;
        }}
    </style>

    <!-- Emerson logo always -->
    <img src="data:image/png;base64,{emerson_logo_base64}" class="emerson-logo">
    """,
    unsafe_allow_html=True,
)

# --- Dynamic Omega header (only once) ---
omega_class = "omega-top-left" if st.session_state.get("chat_started", False) else "omega-center"

st.markdown(
    f"""
    <div id="omega-header" class="{omega_class}">
        <img src="data:image/png;base64,{omega_logo_base64}">
        <div>
            <h1>Omega ChatBot</h1>
            <p>Semantic View: <b>{SEMANTIC_VIEW}</b></p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)



st.markdown(f"Semantic View: `{SEMANTIC_VIEW}`")

# Single, fixed connection via service account
if "CONN" not in st.session_state or st.session_state.CONN is None:
    try:
        cfg = st.secrets["sf"]
        st.session_state.CONN = snowflake.connector.connect(
            account     = cfg["account"],
            user        = cfg["user"],
            password    = cfg["password"],
            role        = cfg["role"],
            warehouse   = cfg["warehouse"],
            authenticator = "snowflake",   # <-- service account (no SSO)
        )
        st.success("Service account connected.")
    except Exception as e:
        st.error(f"Failed to connect to Snowflake: {e}")
        st.stop()


def get_conversation_history() -> list[dict[str, Any]]:
    messages = []
    for msg in st.session_state.messages:
        m: dict[str, Any] = {}
        if msg["role"] == "user":
            m["role"] = "user"
        else:
            m["role"] = "analyst"
        text_content = "\n".join([c for c in msg["content"] if isinstance(c, str)])
        m["content"] = [{"type": "text", "text": text_content}]
        messages.append(m)
    return messages


def send_message() -> requests.Response:
    """Calls the REST API and returns a streaming client."""
    request_body = {
        "messages": get_conversation_history(),
        #"semantic_model_file": f"@{DATABASE}.{SCHEMA}.{STAGE}/{FILE}",
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


def stream(events: Iterator[sseclient.Event]) -> Generator[Any, Any, Any]:
    prev_index = -1
    prev_type = ""
    prev_suggestion_index = -1
    while True:
        event = next(events, None)
        if not event:
            return
        data = json.loads(event.data)
        new_content_block = event.event != "message.content.delta" or data["index"] != prev_index

        if prev_type == "sql" and new_content_block:
            # Close sql markdown once sql section finishes.
            yield "\n```\n\n"
        match event.event:
            case "message.content.delta":
                match data["type"]:
                    case "sql":
                        if new_content_block:
                            # Add sql markdown when we enter a new sql block.
                            yield "```sql\n"
                        yield data["statement_delta"]
                    case "text":
                        yield data["text_delta"]
                    case "suggestions":
                        if new_content_block:
                            # Add a suggestions header when we enter a new suggestions block.
                            yield "\nHere are some example questions you could ask:\n\n"
                            yield "\n- "
                        elif (
                            prev_suggestion_index != data["suggestions_delta"]["index"]
                        ):
                            yield "\n- "
                        yield data["suggestions_delta"]["suggestion_delta"]
                        prev_suggestion_index = data["suggestions_delta"]["index"]
                prev_index = data["index"]
                prev_type = data["type"]
            case "status":
                st.session_state.status = data["status_message"]
                # We return here to allow the spinner to update with the latest status, but this method will be
                #  called again for the next iteration
                return
            case "error":
                st.session_state.error = data
                return


def display_df(df: pandas.DataFrame) -> None:
    if len(df.index) > 1:
        data_tab, line_tab, bar_tab = st.tabs(["Data", "Line Chart", "Bar Chart"])
        data_tab.dataframe(df)
        if len(df.columns) > 1:
            df = df.set_index(df.columns[0])
        with line_tab:
            st.line_chart(df)
        with bar_tab:
            st.bar_chart(df)
    else:
        st.dataframe(df)

def append_message(role: str, content: list[Any]) -> None:
    """
    Cortex requires roles to alternate (user -> analyst -> user ...).
    This helper ensures we don't accidentally push two of the same in a row.
    """
    if st.session_state.messages and st.session_state.messages[-1]["role"] == role:
        # If the last message has the same role, extend it instead of appending
        st.session_state.messages[-1]["content"].extend(content)
    else:
        st.session_state.messages.append({"role": role, "content": content})


def process_message(prompt: str) -> None:
    """Processes a message and adds the response to the chat."""
    append_message("user", [prompt])
    with st.chat_message("user"):
        st.markdown(prompt)

    accumulated_content = []
    with st.chat_message("assistant"):
        with st.spinner("Sending request..."):
            response = send_message()
        st.markdown(
            f"```request_id: {response.headers.get('X-Snowflake-Request-Id')}```"
        )
        events = sseclient.SSEClient(response).events()  # type: ignore
        while st.session_state.status.lower() != "done":
            with st.spinner(st.session_state.status):
                written_content = st.write_stream(stream(events))
                accumulated_content.append(written_content)
            if st.session_state.error:
                st.error(
                    f"Error while processing request:\n {st.session_state.error}",
                    icon="🚨",
                )
                accumulated_content.append(Exception(st.session_state.error))
                st.session_state.error = None
                st.session_state.status = "Interpreting question"
                st.session_state.messages.pop()
                return
            pattern = r"```sql\s*(.*?)\s*```"
            sql_blocks = re.findall(pattern, written_content, re.DOTALL | re.IGNORECASE)
            if sql_blocks:
                for sql_query in sql_blocks:
                    with st.spinner("Executing Query"):
                        df = pd.read_sql(sql_query, st.session_state.CONN)
                        accumulated_content.append(df)
                        display_df(df)
    st.session_state.status = "Interpreting question"
    append_message("analyst", accumulated_content)


def show_conversation_history() -> None:
    for message in st.session_state.messages:
        chat_role = "assistant" if message["role"] == "analyst" else "user"
        with st.chat_message(chat_role):
            for content in message["content"]:
                if isinstance(content, pd.DataFrame):
                    display_df(content)
                elif isinstance(content, Exception):
                    st.error(f"Error while processing request:\n {content}", icon="🚨")
                else:
                    st.write(content)


st.title("Omega ChatBot")
#st.markdown(f"Semantic Model: `{FILE}`")
st.markdown(f"Semantic View: `{SEMANTIC_VIEW}`")


if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.status = "Interpreting question"
    st.session_state.error = None

show_conversation_history()

# --- Chat input (only once in the whole app) ---
user_input = st.chat_input("What is your question?")
if user_input:
    st.session_state.chat_started = True
    process_message(prompt=user_input)
