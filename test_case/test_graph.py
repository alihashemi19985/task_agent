"""Milestone 2: interact with a persistent, read-only LangGraph conversation.

Example:
    python test_graph_memory.py --thread-id demo-1
    # Exit and repeat the same command to resume that conversation.

The original tasks.db is read-only; checkpoints are stored in a separate file.
"""

import argparse
import json
import os
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphRecursionError

from agent.graph import build_graph

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_DB = PROJECT_ROOT / "checkpoints.db"


def get_config(thread_id: str) -> dict:
    """Isolate messages/checkpoints by conversation identifier."""
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 12,
    }



def ask(graph, question: str, thread_id: str) -> None:

    config = get_config(thread_id)

    # خواندن پیام‌های قبلی
    previous = graph.get_state(config).values.get("messages", [])

    # شناسه پیام‌های قبلی
    previous_ids = {
        msg.id for msg in previous
        if msg.id is not None
    }

    try:
        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(content=question)
                ]
            },
            config=config,
        )

    except GraphRecursionError:
        print("خطا: تعداد مراحل مجاز اجرای Agent تمام شد.")
        return

    except Exception as exc:
        print(
            f"خطا در اجرای Graph: "
            f"{type(exc).__name__}: {exc}"
        )
        return

    # تشخیص پیام‌های جدید براساس ID
    fresh_messages = [
        msg
        for msg in result.get("messages", [])
        if msg.id not in previous_ids
    ]

    # نمایش Tool Call و Tool Result در حالت Debug
    if os.getenv("DEBUG_TOOLS", "0") == "1":

        for message in fresh_messages:

            if isinstance(message, AIMessage):
                for call in message.tool_calls:
                    print(f"[TOOL CALL] {call['name']}")
                    print(
                        json.dumps(
                            call["args"],
                            ensure_ascii=False
                        )
                    )

            elif isinstance(message, ToolMessage):
                print(
                    f"[TOOL RESULT] "
                    f"{message.name or message.tool_call_id}"
                )
                print(str(message.content)[:1500])

    # استخراج پاسخ نهایی فقط از پیام‌های جدید
    final_answers = [
        message.content
        for message in fresh_messages
        if (
            isinstance(message, AIMessage)
            and message.content
            and not message.tool_calls
        )
    ]

    if final_answers:
        print("\nدستیار:", final_answers[-1])
    else:
        print(
            "پاسخ نهایی دریافت نشد. "
            "برای بررسی، DEBUG_TOOLS=1 را فعال کن."
        )

def main():
    parser = argparse.ArgumentParser(
        description="Persistent LangGraph read-only CLI (milestone 2)"
    )
    parser.add_argument(
        "--thread-id",
        default="demo-memory-2",
        help="Unique conversation identifier. Reuse it to resume a past chat.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_CHECKPOINT_DB,
        help="Path to the separate SQLite checkpoint database.",
    )
    args = parser.parse_args()

    if not args.thread_id.strip():
        parser.error("--thread-id cannot be empty")

    db = args.db.expanduser().resolve()
    db.parent.mkdir(parents=True, exist_ok=True)

    # Keep the context manager OPEN across all graph.invoke() calls.
    with SqliteSaver.from_conn_string(str(db)) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        config = get_config(args.thread_id)
        stored_messages = graph.get_state(config).values.get("messages", [])

        print("LangGraph | Step 2: Persistent Memory (Read-only)")
        print(f"Thread ID: {args.thread_id}")
        print(f"Checkpoint DB: {db}")
        print(f"Restored messages: {len(stored_messages)}")
        print("برای خروج exit را وارد کن.")

        while True:
            try:
                question = input("\nشما: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nخروج")
                break

            if question.lower() in {"exit", "quit"}:
                break
            if question:
                ask(graph, question, args.thread_id)


if __name__ == "__main__":
    main()