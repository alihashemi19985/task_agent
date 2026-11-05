"""Stage 3.3: route read/write calls and execute writes only after approval."""

from __future__ import annotations

import json
import logging
from typing import Literal
import sqlite3

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import interrupt
from pydantic import ValidationError

from tools.count_tools import count_tasks
from tools.search_tools import search_tasks
from tools.analytics_tools import analyze_tasks
from tools.operation_tools import create_task, update_task_status 
from pathlib import Path
from contextlib import closing

from tools import operation_tools






READ_TOOLS = [count_tasks, search_tasks, analyze_tasks]
WRITE_TOOLS = [create_task, update_task_status]
ALL_TOOLS = READ_TOOLS + WRITE_TOOLS

_READ_NAMES = {tool.name for tool in READ_TOOLS}
_WRITE_BY_NAME = {tool.name: tool for tool in WRITE_TOOLS}
logger = logging.getLogger("task_agent.agent")





def route_after_agent(state: dict) -> Literal["read_tools", "validate_write", "summarize"]:
    calls = getattr(state["messages"][-1], "tool_calls", None) or []
    if not calls:
        return "summarize"
    if all(call.get("name") in _READ_NAMES for call in calls):
        return "read_tools"
    return "validate_write"




def _result_messages(calls: list[dict], status: str, reason: str, **extra) -> list:
    """Balance each tool call with a ToolMessage, then add a final AIMessage."""
    result = [
        ToolMessage(
            content=json.dumps(
                {"status": status, "reason": reason, "tool": call.get("name"), **extra},
                ensure_ascii=False,
                default=str,
            ),
            tool_call_id=call["id"],
            name=call.get("name") or "unknown_tool",
        )
        for call in calls
    ]
    return result




def validate_write_node(state: dict) -> dict:
    """
    Validate a write request before human approval.

    Checks:
    - Exactly one known write tool is requested
    - Arguments satisfy the tool's Pydantic schema
    - Target task exists for update_task_status

    This node never modifies the database.
    """

    calls = getattr(
        state["messages"][-1],
        "tool_calls",
        None,
    ) or []

    # Helper: Return a structured failure to LangGraph.
    def reject(
        status: str,
        reason: str,
        user_message: str,
    ) -> dict:

        messages = _result_messages(
            calls,
            status=status,
            reason=reason,
        )

        messages.append(
            AIMessage(content=user_message)
        )

        return {
            "messages": messages,
            "pending_write": None,
            "approved": False,
        }

    # ----------------------------------------
    # 1. Validate tool calls
    # ----------------------------------------

    if (
        len(calls) != 1
        or calls[0].get("name") not in _WRITE_BY_NAME
    ):

        logger.warning(
            "Write blocked | invalid tool calls | count=%d",
            len(calls),
        )

        return reject(
            status="blocked",
            reason="Invalid, unknown or multiple write calls.",
            user_message=(
                "درخواست عملیات نامعتبر است. "
                "هیچ تغییری در پایگاه داده انجام نشد."
            ),
        )

    call = calls[0]

    tool_name = call["name"]
    call_id = call["id"]

    tool = _WRITE_BY_NAME[tool_name]

    logger.info(
        "Write validation started | tool=%s | call_id=%s",
        tool_name,
        call_id,
    )

    # ----------------------------------------
    # 2. Validate arguments using Pydantic
    # ----------------------------------------

    try:

        validated = tool.args_schema.model_validate(
            call.get("args", {})
        )

        args = validated.model_dump(
            mode="json",
            exclude_none=True,
        )

    except (ValidationError, ValueError, TypeError) as exc:

        logger.warning(
            "Write validation failed | tool=%s | error=%s",
            tool_name,
            str(exc),
        )

        return reject(
            status="invalid_arguments",
            reason="Invalid tool arguments.",
            user_message=(
                "پارامترهای درخواست معتبر نیستند. "
                "لطفاً شناسه تسک و وضعیت موردنظر "
                "یا اطلاعات ساخت تسک را بررسی کنید."
            ),
        )

    # ----------------------------------------
    # 3. Check task existence before approval
    # ----------------------------------------

    if tool_name == "update_task_status":

        task_id = args["task_id"]

        logger.info(
            "Checking task existence | task_id=%s",
            task_id,
        )

        try:

            # Use the same DB as the actual write tool.
            db_path = Path(
                operation_tools.DB_PATH
            ).resolve()

            db_uri = db_path.as_uri() + "?mode=ro"

            with closing(
                sqlite3.connect(
                    db_uri,
                    uri=True,
                    timeout=5,
                )
            ) as conn:

                row = conn.execute(
                    """
                    SELECT 1
                    FROM tasks
                    WHERE id = ?
                    LIMIT 1
                    """,
                    (task_id,),
                ).fetchone()

                task_exists = row is not None

        except (sqlite3.Error, OSError, ValueError):

            logger.exception(
                "Task existence check failed | task_id=%s",
                task_id,
            )

            return reject(
                status="database_error",
                reason="Failed to verify task existence.",
                user_message=(
                    "در بررسی پایگاه داده خطایی رخ داد. "
                    "امکان تأیید عملیات وجود ندارد. "
                    "هیچ تغییری انجام نشد."
                ),
            )

        # ------------------------------------
        # Task does not exist
        # ------------------------------------

        if not task_exists:

            logger.warning(
                "Write rejected | task not found | task_id=%s",
                task_id,
            )

            return reject(
                status="task_not_found",
                reason=f"Task ID {task_id} does not exist.",
                user_message=(
                    f"تسکی با شناسه {task_id} "
                    "در پایگاه داده پیدا نشد. "
                    "بنابراین عملیات تغییر وضعیت انجام نمی‌شود."
                ),
            )

        logger.info(
            "Task existence verified | task_id=%s",
            task_id,
        )

    # ----------------------------------------
    # 4. Validation successful
    # ----------------------------------------

    logger.info(
        "Write validation succeeded | tool=%s | call_id=%s",
        tool_name,
        call_id,
    )

    # Save the validated proposal in LangGraph State.
    # Human approval is handled by request_approval_node.

    return {
        "pending_write": {
            "name": tool_name,
            "args": args,
            "tool_call_id": call_id,
        },
        "approved": False,
    }



def route_after_validation(state: dict) -> Literal["request_approval", "summarize"]:
    return "request_approval" if state.get("pending_write") else "summarize"


def request_approval_node(state: dict) -> dict:
    """Pause before mutation. Resume only with an explicit boolean decision.

    WARNING: this entire node restarts on resume. Nothing before interrupt()
    should mutate the database or invoke either write tool.
    """
    proposal = state.get("pending_write")
    if not proposal:
        raise RuntimeError("Missing pending write proposal")

    decision = interrupt(
        {
            "type": "write_approval",
            "message": "آیا اجرای عملیات زیر را تأیید می‌کنید؟",
            "tool": proposal["name"],
            "args": proposal["args"],
        }
    )
    # A truthy string such as 'yes' is NOT authorization.
    approved = (
        isinstance(decision, dict)
        and type(decision.get("approved")) is bool
        and decision["approved"] is True
    )
    return {"approved": approved}


def _describe_write_result(name: str, result: dict) -> str:
    """Use a deterministic final response; do not ask LLM to retry writes."""
    status = result.get("status")
    task = result.get("task") or {}

    if status == "created":
        return (
            f"تسک «{task.get('title', '')}» با شناسه {task.get('id')} "
            f"برای {task.get('assignee_name', 'کارمند انتخاب‌شده')} "
            f"با اولویت {task.get('priority')} ایجاد شد."
        )
    if status == "updated":
        return (
            f"وضعیت تسک {task.get('id')} از «{result.get('previous_status')}» "
            f"به «{task.get('status')}» تغییر کرد."
        )
    if status == "no_change":
        return f"تسک {task.get('id')} از قبل در وضعیت درخواستی بود؛ تغییری انجام نشد."
    if status == "task_not_found":
        return "تسکی با شناسه درخواستی پیدا نشد؛ تغییری انجام نشد."
    if status == "person_not_found":
        return "کارمند موردنظر در پایگاه داده یافت نشد؛ تسکی ایجاد نشد."
    if status == "ambiguous_person":
        return "چند کارمند با این مشخصات یافت شدند؛ برای ایجاد تسک، شناسه کارمند را مشخص کن."
    return (
        "اجرای عملیات با خطا مواجه شد. پیش از تلاش دوباره وضعیت دیتابیس "
        "را بررسی کن؛ نمی‌توان نتیجه عملیات را قطعی اعلام کرد."
    )


def finish_approval_node(state: dict) -> dict:
    """Perform exactly the approved single write (no LLM/tool retry)."""
    proposal = state.get("pending_write")
    if not proposal:
        raise RuntimeError("Missing pending write proposal")

    name = proposal["name"]
    call_id = proposal["tool_call_id"]
    if state.get("approved") is not True:
        result = {"status": "cancelled", "tool": name}
        text = "عملیات لغو شد؛ هیچ تغییری در دیتابیس انجام نشد."
    else:
        tool = _WRITE_BY_NAME.get(name)
        if tool is None:
            # Fail closed. Never execute a tool not on the write allow-list.
            result = {"status": "blocked", "reason": "Unknown write tool"}
            text = "عملیات ناشناخته است و اجرا نشد."
        else:
            try:
                # Validate a second time before real execution. The approved
                # args in State, not fresh model output, are the only input.
                args = tool.args_schema.model_validate(
                    proposal["args"]
                ).model_dump(mode="json", exclude_none=True)
                result = tool.invoke(args)
                if not isinstance(result, dict):
                    raise TypeError("Write tool must return a dictionary")
            except Exception:
                logger.exception("Approved write failed: %s", name)
                # Never auto-retry an unknown outcome after a write exception.
                result = {
                    "status": "error",
                    "message": "Write failed or outcome uncertain; inspect DB before retry",
                }
            text = _describe_write_result(name, result)

    return {
        "messages": [
            ToolMessage(
                content=json.dumps(result, ensure_ascii=False, default=str),
                tool_call_id=call_id,
                name=name,
            ),
            AIMessage(content=text),
        ],
        "pending_write": None,
        "approved": False,
    }