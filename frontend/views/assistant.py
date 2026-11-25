"""Chat assistant: one chat for the project documents (RAG) and your uploaded data (tools), via POST /v1/chat."""
import json

import streamlit as st

from churn_ui import theme
from churn_ui.client import ApiError

ROUTE_LABELS = {"knowledge": "the project documents", "data": "your uploaded data",
                "both": "your uploaded data and the project documents"}
DATA_PREFIX = "data tool:"

theme.require_backend()
client = theme.get_client()


def clean(text: str) -> str:
    """Stop Streamlit from turning text between two $ signs into a math formula."""
    return text.replace("$", "\\$")


def source_line(msg: dict) -> str:
    """'Answered from ... · Documents: a.md · Data lookup: get_segment_breakdown'"""
    sources = msg.get("sources") or []
    parts = []
    if msg.get("route") in ROUTE_LABELS:
        parts.append(f"Answered from {ROUTE_LABELS[msg['route']]}")
    docs = [x for x in sources if not x.startswith(DATA_PREFIX)]
    lookups = [x[len(DATA_PREFIX):].strip() for x in sources if x.startswith(DATA_PREFIX)]
    if docs:
        parts.append("Documents: " + ", ".join(docs))
    if lookups:
        parts.append("Data lookup: " + ", ".join(lookups))
    return " · ".join(parts)


st.title("Assistant")
st.caption("Ask about the model, the data preparation, the app and the retention playbooks, "
           "or about the customers in your uploaded data.")

# Which dataset do data questions use? The one selected on the Portfolio page, else the newest upload.
try:
    datasets = client.datasets()
except ApiError:
    datasets = []
current = st.session_state.get("dataset_id")
active = next((d for d in datasets if d["id"] == current), datasets[0] if datasets else None)
if active:
    scored = (active.get("rows") or {}).get("scored")
    size = f" ({scored:,} customers)" if scored else ""
    st.caption(f"Data questions use **{active['name']}**{size}. Change it on the Portfolio page.")
else:
    st.caption("No dataset uploaded yet. Questions about your customers need one; project questions work without it.")
    st.page_link("views/upload.py", label="Upload a dataset", icon="📤")

st.session_state.setdefault("chat", [])
for msg in st.session_state["chat"]:
    with st.chat_message(msg["role"]):
        st.write(clean(msg["content"]))
        if msg["role"] == "assistant" and source_line(msg):
            st.caption(source_line(msg))

if not st.session_state["chat"]:
    st.caption("Project: “Why was XGBoost selected?” · “What does the policy say about eligibility?”")
    st.caption("Your data: “Compare churn risk by country in my uploaded file” · "
               "“Show the top 5 highest-risk customers in Germany” · “Why is customer <ID> likely to leave?”")

if prompt := st.chat_input("Ask about the project or your customers…"):
    st.session_state["chat"].append({"role": "user", "content": prompt})
    sources, route = [], None
    try:
        with st.spinner("Working on it… data questions can take up to a minute on a local model."):
            data = client.chat(st.session_state["chat"], active["id"] if active else None)
        reply, sources, route = data["reply"], data.get("sources", []), data.get("route")
    except ApiError as exc:
        reply = f"Error: {exc.message}"
    st.session_state["chat"].append({"role": "assistant", "content": reply, "sources": sources, "route": route})
    st.rerun()

with st.expander("Tool explorer (run a data tool directly)"):
    tools = client.chat_tools()
    tool = st.selectbox("Tool", [t["name"] for t in tools])
    spec = next(t for t in tools if t["name"] == tool)
    st.caption(spec["description"])
    st.json(spec["input_schema"], expanded=False)
    args = st.text_area("Arguments (JSON)", value="{}", height=90)
    if st.button("Run tool"):
        try:
            st.json(client.call_tool(tool, json.loads(args or "{}")))
        except json.JSONDecodeError:
            st.error("Arguments must be valid JSON.")
        except ApiError as exc:
            theme.show_error(exc)