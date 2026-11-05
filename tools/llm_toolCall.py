"""Interactive AvalAI + LangChain tool-calling demo (without LangGraph).

Five tools:
    - count_tasks            (read only)
    - search_tasks           (read only)
    - analyze_tasks          (read only)
    - create_task            (writes to tasks.db, requires terminal approval)
    - update_task_status     (writes to tasks.db, requires terminal approval)

Environment (.env):
    AVALAI_API_KEY=...       Required; NEVER hardcode a key.
    AVALAI_MODEL=gpt-5.4     Optional.
    TASK_TIMEZONE=Asia/Tehran  Optional; align with operation_tools.py.
    DEBUG_TOOLS=1           Optional; print tool calls and results.

This CLI maintains memory until the process exits. It is a demo, not a
production authorization or idempotency system.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from tools import count_tasks
from search_tools import search_tasks
from analytics_tools import analyze_tasks
from operation_tools import create_task, update_task_status


# ---------------------------------------------------------
# 1. Configuration and LLM
# ---------------------------------------------------------

load_dotenv()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("task_agent")

API_KEY = os.getenv("AVALAI_API_KEY")
if not API_KEY:
    raise RuntimeError("AVALAI_API_KEY is missing. Add it to your .env file.")

llm = ChatOpenAI(
    model=os.getenv("AVALAI_MODEL", "gpt-5.4"),
    api_key=API_KEY,
    base_url="https://api.avalai.ir/v1",
    timeout=45,
    max_retries=2,
)


# ---------------------------------------------------------
# 2. Register all tools
# ---------------------------------------------------------

TOOLS = [
    count_tasks,
    search_tasks,
    analyze_tasks,
    create_task,
    update_task_status,
]
TOOLS_BY_NAME = {item.name: item for item in TOOLS}
WRITE_TOOL_NAMES = {"create_task", "update_task_status"}

if len(TOOLS_BY_NAME) != len(TOOLS):
    raise RuntimeError("Duplicate tool names were registered.")

llm_with_tools = llm.bind_tools(
    TOOLS,
    tool_choice="auto",
    parallel_tool_calls=False,
)

DEBUG = os.getenv("DEBUG_TOOLS", "0") == "1"
MAX_ROUNDS_PER_TURN = 5

# Your local count_tasks may have been upgraded with last_n_months.
HAS_RELATIVE_MONTHS = (
    "last_n_months" in count_tasks.args_schema.model_fields
)

SYSTEM_PROMPT = """
You are a reliable Persian-speaking assistant for organizational task data.

DATA RELIABILITY:
- Respond in Persian, concisely and accurately.
- Never invent task counts, statuses, employee names, dates or IDs.
- Use a database tool for every factual claim about tasks; chat history
  is NOT a substitute for current database evidence.
- Employee names and departments are dynamic, never hardcoded.
- Status values: Open, In Progress, Review, Done.
- Priority values: Low, Medium, High, Critical; High is not Critical.

TOOL ROUTING:
1. count_tasks: count matching tasks ("چند", "تعداد").
2. search_tasks: list, find, or display matching individual tasks
   ("نشان بده", "نمایش بده", "لیست"). Use employee/department
   names from the user's request. Do not guess assignee_id.
3. analyze_tasks: rankings (most OR fewest tasks, by assignee or
   department), average completion duration, or count overdue tasks.
   For ranking use analysis_type=workload_ranking, group_by=assignee
   or department, sort_order=desc (most) or asc (fewest).
   For average use analysis_type=average_completion_time.
   For overdue use analysis_type=overdue_count.
4. create_task: create a new task only if user provided a task title
   and an identifiable employee. Ask for missing information first.
   Do not invent a title or description. Default priority is Medium
   and new tasks start Open.
5. update_task_status: change an existing task to a specified status;
   require an explicit, known task_id and new_status.

TIME / BUSINESS RULES:
- For Done tasks, due_time is their completion date.
- For unfinished tasks, due_time is the scheduled deadline.
- A task is overdue when status is not Done and due_time < today.
- Completion duration = due_time - create_time for Done tasks.
- Do not use create_time as if it were the completion date.

WRITE SAFETY:
- The Python application asks the user for explicit terminal approval
  before executing create_task or update_task_status.
- Do not claim an operation succeeded before its tool result confirms it.
- If a write was cancelled, do not try it again.
- If a write failed, do not retry it automatically.

RESPONSES:
- If the tool returns ambiguous_person or person_not_found, ask the
  user for a valid full name or employee ID, without guessing.
- If a tool reports partial_data or a warning, mention the limitation.
- For paginated results, state how many were shown and whether more exist.
- Do not offer features or analysis unsupported by these tools.
- When the user's request is ambiguous or outside the tools' scope,
  ask one clarifying question or explain the limitation.
"""

MONTH_RULE = (
    "For 'last N months' completion questions, call count_tasks with "
    "status='Done' and last_n_months=N. NEVER manually invent date bounds."
    if HAS_RELATIVE_MONTHS else
    "The count_tasks tool does NOT support last_n_months in this installed "
    "version. Do not invent dates for relative-month requests; ask for "
    "an explicit date range or explain the limitation."
)


def _today() -> str:
    tz = os.getenv("TASK_TIMEZONE", "Asia/Tehran")
    try:
        return datetime.now(ZoneInfo(tz)).date().isoformat()
    except ZoneInfoNotFoundError as exc:
        raise RuntimeError(f"Invalid TASK_TIMEZONE: {tz}") from exc


def _system_message() -> SystemMessage:
    return SystemMessage(
        content=f"{SYSTEM_PROMPT}\n{MONTH_RULE}\nCurrent date: {_today()}."
    )


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _content_text(content: Any) -> str:
    """Handle plain-text and block-style model responses."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = [
            x.get("text", "") for x in content
            if isinstance(x, dict) and x.get("type") == "text"
        ]
        return "\n".join(chunk for chunk in chunks if chunk)
    return str(content) if content is not None else ""


# ---------------------------------------------------------
# 3. Explicit approval for real database changes
# ---------------------------------------------------------

def ask_write_confirmation(name: str, args: dict[str, Any]) -> bool:
    print("\n⚠️  درخواست تغییر دیتابیس اصلی tasks.db")
    print(f"عملیات: {name}")
    print(json.dumps(args, ensure_ascii=False, indent=2))
    if name == "update_task_status" and args.get("new_status") == "Done":
        print("توجه: طبق قرارداد پروژه، due_time تاریخ تکمیل امروز می‌شود.")
    if name == "update_task_status" and args.get("new_status") != "Done":
        print("توجه: بازگشایی تسک Done ممکن است تاریخ تکمیل قبلی را پاک کند.")
    try:
        answer = input("اجازه اجرای این تغییر را می‌دهی؟ (yes/no): ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print()
        return False
    # Only an explicit affirmative counts as approval.
    return answer in {"yes", "y", "بله"}


def execute_tool(
    call: dict[str, Any], *, write_already_attempted: bool
) -> tuple[dict[str, Any], bool]:
    """Validate, authorize and execute exactly one proposed tool call.

    Returns (result, write_was_attempted_or_denied). The second value is
    true even on validation failure or refusal, preventing further writes
    during this user turn.
    """
    name = call["name"]
    is_write = name in WRITE_TOOL_NAMES
    tool = TOOLS_BY_NAME.get(name)

    if tool is None:
        return {"status": "error", "message": f"Unknown tool: {name}"}, False

    if is_write and write_already_attempted:
        return {
            "status": "blocked",
            "message": "Only one write is allowed per user message.",
        }, False

    try:
        schema = tool.args_schema
        validated = schema.model_validate(call["args"])
        args = validated.model_dump(mode="json", exclude_none=True)
    except (ValidationError, TypeError, ValueError) as exc:
        return {
            "status": "invalid_arguments",
            "message": "Missing or invalid parameters. Ask for clarification.",
            "details": str(exc),
        }, is_write

    if DEBUG:
        print(f"\n[TOOL CALL] {name}: {_json(args)}")

    if is_write and not ask_write_confirmation(name, args):
        return {
            "status": "cancelled",
            "message": "The user explicitly declined this database modification.",
        }, True

    try:
        result = tool.invoke(args)
    except Exception:
        logger.exception("Tool execution error for %s", name)
        result = {
            "status": "error",
            "message": (
                "Tool execution failed. Do not invent a result or "
                "automatically retry a database write."
            ),
        }

    if not isinstance(result, dict):
        result = {"result": result}

    if DEBUG:
        print(f"[TOOL RESULT] {name}: {_json(result)}")

    return result, is_write



# ---------------------------------------------------------
# 4. One user turn; preserves conversation history
# ---------------------------------------------------------

def answer_turn(messages: list) -> str:
    """Execute read tools as needed, but never perform unapproved writes."""
    write_attempted = False

    for _ in range(MAX_ROUNDS_PER_TURN):
        response = llm_with_tools.invoke(messages)

        if response.invalid_tool_calls:
            # Malformed provider tool calls must not be executed.
            return "مدل آرگومان‌های ابزار را نامعتبر تولید کرد. لطفاً دوباره تلاش کن."

        messages.append(response)
        calls = response.tool_calls

        if not calls:
            return _content_text(response.content) or "پاسخی دریافت نشد."

        just_cancelled = False
        any_write_requested = False

        # Every tool call ID must have exactly one matching ToolMessage.
        for call in calls:
            result, attempted = execute_tool(
                call, write_already_attempted=write_attempted
            )
            write_attempted = write_attempted or attempted
            any_write_requested = any_write_requested or (
                call["name"] in WRITE_TOOL_NAMES
            )
            just_cancelled = just_cancelled or result.get("status") == "cancelled"

            messages.append(
                ToolMessage(
                    content=_json(result),
                    tool_call_id=call["id"],
                    name=call["name"],
                )
            )

        if just_cancelled:
            text = "عملیات به درخواست تو لغو شد و تغییری اعمال نشد."
            messages.append(AIMessage(content=text))
            return text

        if any_write_requested:
            # After a write attempt, use model WITHOUT tools to ensure no
            # second write is requested/executed in this user turn.
            final = llm.invoke(messages)
            messages.append(final)
            return _content_text(final.content) or "نتیجه عملیات ثبت شد."

    return "تعداد مراحل اجرای ابزارها از حد مجاز گذشت؛ لطفاً درخواست را ساده‌تر کن."


# ---------------------------------------------------------
# 5. Interactive CLI (no LangGraph yet)
# ---------------------------------------------------------

def main() -> None:
    messages = [_system_message()]
    print("دستیار مدیریت تسک | برای خروج: exit")
    print("داده‌های تسک از SQLite خوانده می‌شوند.")
    if not HAS_RELATIVE_MONTHS:
        print("هشدار: count_tasks محلی، پارامتر last_n_months ندارد.")

    while True:
        try:
            question = input("\nشما: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nخروج")
            return

        if question.lower() in {"exit", "quit"}:
            print("خروج")
            return
        if not question:
            continue

        # Refresh the date on every user turn (important across midnight).
        messages[0] = _system_message()
        turn_start = len(messages)
        messages.append(HumanMessage(content=question))

        try:
            answer = answer_turn(messages)
            print("\nدستیار:", answer)
        except Exception:
            logger.exception("LLM call failed")
            # Do not leave a partially completed tool-call protocol in history.
            del messages[turn_start:]
            print("\nارتباط با مدل ناموفق بود. وضعیت اجرای آخرین عملیات را "
                  "قبل از تلاش دوباره بررسی کن.")



if __name__ == "__main__":
    main()