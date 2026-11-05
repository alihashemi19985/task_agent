"""Streamlit UI for the Persian Task Management Agent.

Run the FastAPI backend first, then:
    streamlit run app.py
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("TASK_AGENT_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="Task Agent", page_icon="💬", layout="centered")
st.markdown(
    """<style>
    [data-testid="stChatMessage"] p { direction: rtl; text-align: right; }
    [data-testid="stSidebar"] { direction: rtl; }
    </style>""",
    unsafe_allow_html=True,
)


def api_request(method: str, path: str, *, allow_error: bool = False, **kwargs):
    try:
        response = requests.request(
            method, f"{API_URL}{path}", timeout=180, **kwargs
        )
        data = response.json()
    except requests.RequestException as exc:
        raise RuntimeError("ارتباط با FastAPI برقرار نشد. ابتدا سرور API را اجرا کن.") from exc
    except ValueError as exc:
        raise RuntimeError("پاسخ دریافتی از API معتبر نیست.") from exc

    if not response.ok and not allow_error:
        raise RuntimeError(data.get("message") or data.get("detail") or "خطای API")
    return data


if "active_conversation" not in st.session_state:
    st.session_state.active_conversation = None

try:
    conversations = api_request("GET", "/conversations")["conversations"]
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.header("گفتگوها")
    if st.button("➕ گفتگوی جدید", use_container_width=True, type="primary"):
        try:
            result = api_request("POST", "/conversations")
            st.session_state.active_conversation = result["conversation_id"]
            st.rerun()
        except RuntimeError as exc:
            st.error(str(exc))

    st.divider()
    if not conversations:
        st.caption("هنوز گفتگویی ذخیره نشده است.")

    for conv in conversations:
        cid = conv["conversation_id"]
        title = conv["title"]
        if st.button(
            title,
            key=f"conversation_{cid}",
            use_container_width=True,
            type="primary" if cid == st.session_state.active_conversation else "secondary",
        ):
            st.session_state.active_conversation = cid
            st.rerun()

if st.session_state.active_conversation is None and conversations:
    st.session_state.active_conversation = conversations[0]["conversation_id"]

cid = st.session_state.active_conversation
st.title("💬 دستیار مدیریت تسک‌ها")

messages = []
pending = None
if cid:
    try:
        messages = api_request("GET", f"/conversations/{cid}/messages")["messages"]
        pending_data = api_request("GET", f"/conversations/{cid}/pending")
        if pending_data["pending"]:
            pending = pending_data["approval"]
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

for item in messages:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        if item["role"] == "assistant" and item["status"] == "error":
            st.caption(f"HTTP {item['http_status']}")

if pending:
    st.warning("این عملیات منتظر تأیید شماست. هنوز تغییری انجام نشده است.")
    with st.expander("جزئیات عملیات", expanded=True):
        st.write("**ابزار:**", pending.get("tool", "نامشخص"))
        st.json(pending.get("args", {}), expanded=False)

    approve_col, cancel_col = st.columns(2)
    decision = None
    if approve_col.button("✅ تأیید می‌کنم", use_container_width=True, type="primary"):
        decision = True
    if cancel_col.button("❌ لغو", use_container_width=True):
        decision = False

    if decision is not None:
        try:
            with st.spinner("در حال ثبت تصمیم و دریافت نتیجه..."):
                api_request(
                    "POST", "/chat", allow_error=True,
                    json={"conversation_id": cid, "approved": decision},
                )
            st.rerun()
        except RuntimeError as exc:
            st.error(str(exc))

prompt = st.chat_input(
    "پیام خود را درباره تسک‌ها بنویسید...",
    disabled=pending is not None,
)

if prompt:
    try:
        if cid is None:
            created = api_request("POST", "/conversations")
            cid = created["conversation_id"]
            st.session_state.active_conversation = cid

        with st.spinner("در حال دریافت پاسخ..."):
            api_request(
                "POST", "/chat", allow_error=True,
                json={"conversation_id": cid, "message": prompt},
            )
        st.rerun()
    except RuntimeError as exc:
        st.error(str(exc))