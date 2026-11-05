"""SQLite chat history and conversation endpoints, separate from tasks.db.

History is for the UI; LangGraph's checkpoints.db remains authoritative
for agent memory and pending human approvals.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path as FilePath
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Path as APIPath, Request

DB_PATH = FilePath(__file__).resolve().with_name("chat_history.db")
router = APIRouter(prefix="/conversations", tags=["Conversations"])
logger = logging.getLogger("task_agent.agent")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_chat_db() -> None:
    """Call once during FastAPI startup; safe for repeated startup."""
    with closing(_connect()) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT 'گفتگوی جدید',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    status TEXT,
                    approval_json TEXT,
                    http_status INTEGER,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations(id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation
                ON chat_messages(conversation_id, id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_updated
                ON conversations(updated_at)
            """)


def create_conversation() -> dict:
    conversation_id = uuid4().hex
    now = _now()
    with closing(_connect()) as conn:
        with conn:
            conn.execute(
                """INSERT INTO conversations(id, title, created_at, updated_at)
                   VALUES (?, ?, ?, ?)""",
                (conversation_id, "گفتگوی جدید", now, now),
            )
    return {"conversation_id": conversation_id, "title": "گفتگوی جدید"}


def conversation_exists(conversation_id: str) -> bool:
    with closing(_connect()) as conn:
        return conn.execute(
            "SELECT 1 FROM conversations WHERE id=?", (conversation_id,)
        ).fetchone() is not None


def save_exchange(
    *,
    conversation_id: str,
    message: str | None,
    approved: bool | None,
    assistant_text: str,
    status: str,
    approval: dict | None,
    http_status: int,
) -> None:
    """Atomically store the user's input and the actual API response.

    If this is an approval, store the explicit user decision as its user message.
    Never calls the agent or modifies tasks.db.
    """
    now = _now()
    user_text = (
        message.strip()
        if message is not None
        else ("تأیید می‌کنم" if approved is True else "لغو")
    )
    with closing(_connect()) as conn:
        with conn:
            conn.execute(
                """INSERT OR IGNORE INTO conversations
                   (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)""",
                (conversation_id, "گفتگوی جدید", now, now),
            )

            if message is not None:
                previous_users = conn.execute(
                    """SELECT 1 FROM chat_messages
                       WHERE conversation_id=? AND role='user' LIMIT 1""",
                    (conversation_id,),
                ).fetchone()
                if previous_users is None:
                    title = " ".join(user_text.split())[:55] or "گفتگوی جدید"
                    conn.execute(
                        "UPDATE conversations SET title=? WHERE id=?",
                        (title, conversation_id),
                    )

            conn.execute(
                """INSERT INTO chat_messages
                   (conversation_id, role, content, created_at)
                   VALUES (?, 'user', ?, ?)""",
                (conversation_id, user_text, now),
            )
            conn.execute(
                """INSERT INTO chat_messages
                   (conversation_id, role, content, status,
                    approval_json, http_status, created_at)
                   VALUES (?, 'assistant', ?, ?, ?, ?, ?)""",
                (
                    conversation_id,
                    assistant_text,
                    status,
                    json.dumps(approval, ensure_ascii=False) if approval else None,
                    http_status,
                    now,
                ),
            )
            conn.execute(
                "UPDATE conversations SET updated_at=? WHERE id=?",
                (now, conversation_id),
            )


@router.post("", status_code=201)
def new_conversation():
    return create_conversation()


@router.get("")
def list_conversations():
    with closing(_connect()) as conn:
        rows = conn.execute(
            """SELECT id AS conversation_id, title, created_at, updated_at
               FROM conversations ORDER BY updated_at DESC LIMIT 200"""
        ).fetchall()
    return {"conversations": [dict(row) for row in rows]}


@router.get("/{conversation_id}/messages")
def get_conversation_messages(
    conversation_id: str = APIPath(
        min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"
    ),
):
    with closing(_connect()) as conn:
        exists = conn.execute(
            "SELECT 1 FROM conversations WHERE id=?", (conversation_id,)
        ).fetchone()
        if exists is None:
            raise HTTPException(status_code=404, detail="گفتگو پیدا نشد.")
        rows = conn.execute(
            """SELECT id, role, content, status, approval_json,
                      http_status, created_at
               FROM chat_messages WHERE conversation_id=? ORDER BY id ASC""",
            (conversation_id,),
        ).fetchall()

    messages = []
    for row in rows:
        entry = dict(row)
        value = entry.pop("approval_json")
        entry["approval"] = json.loads(value) if value else None
        messages.append(entry)
    return {"conversation_id": conversation_id, "messages": messages}


@router.get("/{conversation_id}/pending")
def get_pending_approval(
    request: Request,
    conversation_id: str = APIPath(
        min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"
    ),
):
    """Use the LangGraph checkpoint, not the UI message history, as truth."""
    if not conversation_exists(conversation_id):
        raise HTTPException(status_code=404, detail="گفتگو پیدا نشد.")

    config = {"configurable": {"thread_id": conversation_id}}
    try:
        with request.app.state.graph_lock:
            snapshot = request.app.state.graph.get_state(config)
            values = [
                item.value
                for task in getattr(snapshot, "tasks", ())
                for item in (getattr(task, "interrupts", ()) or ())
            ]
    except Exception as exc:
        logger.exception("Failed to read pending approval | conversation_id=%s", conversation_id)
        raise HTTPException(
            status_code=500, detail="بازیابی وضعیت تأیید گفتگو ناموفق بود."
        ) from exc

    if values and (len(values) != 1 or not isinstance(values[0], dict)):
        raise HTTPException(status_code=409, detail="وضعیت تأیید گفتگو نامعتبر است.")

    approval = values[0] if values else None
    return {
        "conversation_id": conversation_id,
        "pending": approval is not None,
        "approval": approval,
    }