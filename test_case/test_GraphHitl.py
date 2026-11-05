"""Milestone 3.3 interactive CLI: writes only after human approval.

Uses a persistent COPY of tasks.db for safe testing, never the original.
Run from project root: python test_GraphHitl.py --thread-id hitl-step33-1
"""

import argparse
import json
import os
import sqlite3
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from agent.graph import build_graph
from logging_config import setup_logging

setup_logging()



ROOT = Path(__file__).resolve().parent
CHECKPOINT_DB = ROOT / "checkpoints_step33.db"
ORIGINAL_DB = ROOT / "tasks.db"
TEST_TASK_DB = ROOT / "test_tasks_step33.db"


def configure_test_database():
    """Copy source once, then route ALL tools to the isolated test DB."""
    import tools
    import search_tools
    import analytics_tools
    import operation_tools

    source = ORIGINAL_DB.resolve()
    destination = TEST_TASK_DB.resolve()
    if source == destination:
        raise RuntimeError("Refusing to test against the original tasks.db")
    if not source.is_file():
        raise FileNotFoundError(f"Original tasks.db does not exist: {source}")
    if not destination.exists():
        with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as src:
            with sqlite3.connect(destination) as dst:
                src.backup(dst)
        print(f"Created isolated test database: {destination}")

    for module in (tools, search_tools, analytics_tools, operation_tools):
        module.DB_PATH = destination
    return destination


def get_config(thread_id):
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 15,
    }


def current_interrupts(graph, config):
    """Find interrupted task payloads in the current checkpoint, if any."""
    snapshot = graph.get_state(config)
    return [
        it
        for task in snapshot.tasks
        for it in getattr(task, "interrupts", ())
    ]


def print_new_messages(graph, config, old_ids):
    messages = graph.get_state(config).values.get("messages", [])
    new = [message for message in messages if message.id not in old_ids]

    if os.getenv("DEBUG_TOOLS", "0") == "1":
        for message in new:
            if isinstance(message, AIMessage):
                for call in message.tool_calls:
                    print(f"[TOOL CALL] {call['name']}")
                    print(json.dumps(call["args"], ensure_ascii=False, indent=2))
            elif isinstance(message, ToolMessage):
                print(f"[TOOL RESULT] {message.name or message.tool_call_id}")
                print(str(message.content)[:1000])

    final = [
        message.content
        for message in new
        if isinstance(message, AIMessage)
        and message.content
        and not message.tool_calls
    ]
    if final:
        print("\nدستیار:", final[-1])
    else:
        print("\nپاسخ نهایی دریافت نشد. برای بررسی، DEBUG_TOOLS=1 کن.")


def handle_approval(graph, config, old_ids=None):
    """Resume only when there is an outstanding interrupt."""
    while True:
        interrupts = current_interrupts(graph, config)
        if not interrupts:
            return
        if len(interrupts) != 1:
            raise RuntimeError("More than one pending approval; manual review needed")

        print("\n⚠️ درخواست تأیید عملیات (هنوز هیچ تغییری انجام نشده):")
        print(json.dumps(interrupts[0].value, ensure_ascii=False, indent=2))
        answer = input("تأیید می‌کنی؟ (yes/no؛ exit برای توقف): ").strip().lower()
        if answer in {"exit", "quit"}:
            print("درخواست تأیید در حافظه باقی ماند؛ بعداً با همان thread-id برگرد.")
            return
        if answer not in {"yes", "y", "بله", "no", "n", "خیر", "نه"}:
            print("فقط yes یا no را وارد کن.")
            continue

        # Only a trusted, explicitly selected boolean is sent to the graph.
        approved = answer in {"yes", "y", "بله"}
        old_ids = {
            msg.id for msg in graph.get_state(config).values.get("messages", [])
            if msg.id
        }
        result = graph.invoke(Command(resume={"approved": approved}), config=config)
        if "__interrupt__" not in result:
            print_new_messages(graph, config, old_ids)
        # If a future graph can interrupt more than once, handle the next one.


def ask(graph, question, thread_id):
    config = get_config(thread_id)
    if current_interrupts(graph, config):
        print("این گفتگو یک عملیات منتظر تأیید دارد. ابتدا آن را تعیین‌تکلیف کن.")
        handle_approval(graph, config)
        if current_interrupts(graph, config):
            return

    old_ids = {
        msg.id for msg in graph.get_state(config).values.get("messages", [])
        if msg.id
    }
    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=question)]},
            config=config,
        )
        if result.get("__interrupt__"):
            handle_approval(graph, config)
        else:
            print_new_messages(graph, config, old_ids)
    except GraphRecursionError:
        print("خطا: تعداد مراحل اجرای Graph تمام شد.")
    except Exception as exc:
        print(f"خطا: {type(exc).__name__}: {exc}")
        print("در صورت وجود عملیات معلق، با همین thread-id دوباره اجرا کن.")


def main():
    parser = argparse.ArgumentParser(description="LangGraph milestone 3.3 approved writes on test DB")
    parser.add_argument("--thread-id", default="hitl-demo-8")
    parser.add_argument("--db", type=Path, default=CHECKPOINT_DB)
    args = parser.parse_args()
    if not args.thread_id.strip():
        parser.error("thread-id must not be empty")

    task_db = configure_test_database()
    db = args.db.expanduser().resolve()
    db.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(db)) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        config = get_config(args.thread_id)
        print("LangGraph Step 3.3 — Approved writes on ISOLATED test DB")
        print(f"Task DB (test copy): {task_db}")
        print(f"Original DB (unchanged): {ORIGINAL_DB.resolve()}")
        print(f"Thread ID: {args.thread_id}")
        print(f"Checkpoint DB: {db}")
        if current_interrupts(graph, config):
            print("تأیید معلق از اجرای قبلی پیدا شد.")
            handle_approval(graph, config)

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