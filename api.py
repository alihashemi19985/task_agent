
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    BadRequestError,
    RateLimitError,
)
from pydantic import BaseModel, Field, StrictBool, model_validator
from chat_history import (
    router as history_router,
    init_chat_db,
    save_exchange,
)


from agent.graph import build_graph
from logging_config import setup_logging

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DB = BASE_DIR / "checkpoints.db"
logger = logging.getLogger("task_agent.agent")  # Match logging_config.py
WRITE_TOOL_NAMES = {"create_task", "update_task_status"}
init_chat_db()
# HTTP status and public message for failed write-tool results.
WRITE_ERRORS = {
    "invalid_arguments": (422, "پارامترهای عملیات نامعتبر هستند."),
    "task_not_found": (404, "تسک موردنظر پیدا نشد."),
    "person_not_found": (404, "کارمند موردنظر پیدا نشد."),
    "ambiguous_person": (409, "چند کارمند با این مشخصات پیدا شدند؛ اطلاعات دقیق‌تری لازم است."),
    "blocked": (409, "اجرای عملیات به دلایل ایمنی مسدود شد."),
    "database_error": (503, "بررسی پایگاه داده با خطا مواجه شد."),
    "error": (500, "نتیجه عملیات نامطمئن است؛ قبل از تلاش مجدد دیتابیس را بررسی کنید."),
}
WRITE_SUCCESS = {"created", "updated", "no_change", "cancelled"}


class ChatRequest(BaseModel):
    conversation_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
    message: str | None = None
    approved: StrictBool | None = None

    @model_validator(mode="after")
    def validate_chat_input(self):
        # Exactly one mode: a new message OR an approval decision.
        if (self.message is None) == (self.approved is None):
            raise ValueError("فقط یکی از message یا approved باید ارسال شود.")
        if self.message is not None and not self.message.strip():
            raise ValueError("message نباید خالی باشد.")
        if self.approved is not None and self.conversation_id is None:
            raise ValueError("برای تأیید یا رد، conversation_id الزامی است.")
        return self


class ChatResponse(BaseModel):
    conversation_id: str | None
    status: Literal["completed", "pending_confirmation", "cancelled", "error"]
    message: str
    approval: dict[str, Any] | None = None
    error: dict[str, str] | None = None


def error_response(
    http_status: int,
    code: str,
    message: str,
    conversation_id: str | None = None,
) -> JSONResponse:
    body = ChatResponse(
        conversation_id=conversation_id,
        status="error",
        message=message,
        error={"code": code, "message": message},
    )
    return JSONResponse(status_code=http_status, content=body.model_dump(mode="json"))


def interrupt_values(snapshot) -> list[Any]:
    """Read pending interrupts from a persisted LangGraph snapshot."""
    return [
        item.value
        for task in getattr(snapshot, "tasks", ())
        for item in (getattr(task, "interrupts", ()) or ())
    ]


def final_ai_text(messages: list) -> str | None:
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        if isinstance(msg.content, str) and msg.content.strip():
            return msg.content.strip()
        if isinstance(msg.content, list):
            text = " ".join(
                item.get("text", "")
                for item in msg.content
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            ).strip()
            if text:
                return text
    return None


def latest_new_write_result(messages: list, old_tool_call_ids: set[str]) -> dict | None:
    """Avoid mapping an error from a *previous* turn to this HTTP request."""
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage) or msg.name not in WRITE_TOOL_NAMES:
            continue
        if msg.tool_call_id in old_tool_call_ids:
            continue
        try:
            data = json.loads(msg.content) if isinstance(msg.content, str) else None
        except (ValueError, TypeError):
            continue
        if isinstance(data, dict):
            return data
    return None


def describe_write_result(result: dict) -> str:
    """Generate a reliable message from the write tool result."""

    status = result.get("status")
    task = result.get("task") or {}

    if status == "created":
        return (
            f"تسک «{task.get('title', '')}» "
            f"با شناسه {task.get('id')} "
            "با موفقیت ایجاد شد."
        )

    if status == "updated":
        return (
            f"وضعیت تسک {task.get('id')} "
            f"از «{result.get('previous_status')}» "
            f"به «{task.get('status')}» تغییر کرد."
        )

    if status == "no_change":
        return (
            f"تسک {task.get('id')} از قبل "
            "در وضعیت درخواستی قرار داشت؛ "
            "هیچ تغییری انجام نشد."
        )

    if status == "cancelled":
        return "عملیات لغو شد؛ هیچ تغییری انجام نشد."

    return "نتیجه عملیات قابل تشخیص نیست."




@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    init_chat_db()

    with SqliteSaver.from_conn_string(
        str(CHECKPOINT_DB)
    ) as checkpointer:

        app.state.graph = build_graph(
            checkpointer=checkpointer
        )

        app.state.graph_lock = Lock()

        logger.info("FastAPI started")
        yield
        logger.info("FastAPI stopped")



app = FastAPI(
    title="Task Management Agent API",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(history_router)

@app.exception_handler(RequestValidationError)
async def invalid_request_handler(request: Request, exc: RequestValidationError):
    logger.warning("Invalid API request | path=%s | errors=%d", request.url.path, len(exc.errors()))
    return error_response(422, "invalid_request", "ساختار درخواست نامعتبر است؛ message یا approved را بررسی کنید.")


#@app.post("/chat", response_model=ChatResponse)
def run_agent_chat(
    body: ChatRequest,
    request: Request,
) -> ChatResponse | JSONResponse:

    conversation_id = body.conversation_id or uuid4().hex

    config = {
        "configurable": {
            "thread_id": conversation_id
        }
    }

    graph = request.app.state.graph

    logger.info(
        "Chat received | conversation_id=%s | mode=%s",
        conversation_id,
        "approval" if body.approved is not None else "message",
    )

    try:
        # Prevent simultaneous graph executions on SQLite.
        with request.app.state.graph_lock:

            # -----------------------------------
            # 1. Read current conversation state
            # -----------------------------------

            before = graph.get_state(config)

            pending_before = interrupt_values(before)

            old_tool_call_ids = {
                msg.tool_call_id
                for msg in before.values.get("messages", [])
                if isinstance(msg, ToolMessage)
            }

            # -----------------------------------
            # 2. Process a new message
            # -----------------------------------

            if body.approved is None:

                if pending_before:

                    logger.warning(
                        "Message rejected | approval pending | "
                        "conversation_id=%s",
                        conversation_id,
                    )

                    return error_response(
                        409,
                        "approval_pending",
                        "یک عملیات منتظر تأیید است. "
                        "ابتدا approved را ارسال کنید.",
                        conversation_id,
                    )

                if before.next:

                    logger.warning(
                        "Message rejected | incomplete graph | "
                        "conversation_id=%s",
                        conversation_id,
                    )

                    return error_response(
                        409,
                        "conversation_incomplete",
                        "اجرای قبلی این گفتگو ناتمام است.",
                        conversation_id,
                    )

                result = graph.invoke(
                    {
                        "messages": [
                            HumanMessage(
                                content=body.message.strip()
                            )
                        ]
                    },
                    config=config,
                )

            # -----------------------------------
            # 3. Resume Human Approval
            # -----------------------------------

            else:

                if len(pending_before) != 1:

                    logger.warning(
                        "Approval rejected | no pending operation | "
                        "conversation_id=%s",
                        conversation_id,
                    )

                    return error_response(
                        409,
                        "no_pending_approval",
                        "برای این گفتگو درخواست تأیید "
                        "معتبری وجود ندارد.",
                        conversation_id,
                    )

                logger.info(
                    "Approval received | "
                    "conversation_id=%s | approved=%s",
                    conversation_id,
                    body.approved,
                )

                result = graph.invoke(
                    Command(
                        resume={
                            "approved": body.approved
                        }
                    ),
                    config=config,
                )

            # -----------------------------------
            # 4. Read new graph state
            # -----------------------------------

            after = graph.get_state(config)

            interrupts = list(
                result.get("__interrupt__", ()) or ()
            )

            approval_values = (
                [item.value for item in interrupts]
                if interrupts
                else interrupt_values(after)
            )

            # -----------------------------------
            # 5. Handle pending approval
            # -----------------------------------

            if approval_values:

                if (
                    len(approval_values) != 1
                    or not isinstance(
                        approval_values[0], dict
                    )
                ):

                    logger.error(
                        "Invalid interrupt | "
                        "conversation_id=%s",
                        conversation_id,
                    )

                    return error_response(
                        500,
                        "unexpected_interrupt",
                        "وضعیت تأیید عملیات قابل تشخیص نیست.",
                        conversation_id,
                    )

                approval = approval_values[0]

                logger.info(
                    "Write approval pending | "
                    "conversation_id=%s | tool=%s",
                    conversation_id,
                    approval.get("tool"),
                )

                return ChatResponse(
                    conversation_id=conversation_id,
                    status="pending_confirmation",
                    message=approval.get(
                        "message",
                        "این عملیات نیازمند تأیید شماست.",
                    ),
                    approval=approval,
                )

            # -----------------------------------
            # 6. Protect incomplete executions
            # -----------------------------------

            if after.next:

                logger.warning(
                    "Graph execution incomplete | "
                    "conversation_id=%s",
                    conversation_id,
                )

                return error_response(
                    409,
                    "conversation_incomplete",
                    "اجرای گفتگو ناتمام مانده است.",
                    conversation_id,
                )

            # -----------------------------------
            # 7. Collect results
            # -----------------------------------

            messages = result.get("messages", [])

            new_tool_messages = [
                msg
                for msg in messages
                if isinstance(msg, ToolMessage)
                and msg.tool_call_id
                not in old_tool_call_ids
            ]

            # Handle exceptions converted to ToolMessages.
            tool_errors = [
                msg
                for msg in new_tool_messages
                if getattr(msg, "status", None) == "error"
            ]

            if tool_errors:

                logger.error(
                    "Tool execution error | "
                    "conversation_id=%s | tool=%s",
                    conversation_id,
                    tool_errors[-1].name,
                )

                return error_response(
                    500,
                    "tool_execution_error",
                    "اجرای ابزار با خطا مواجه شد.",
                    conversation_id,
                )

            write_result = latest_new_write_result(
                messages,
                old_tool_call_ids,
            )

            assistant_text = final_ai_text(messages)

            # -----------------------------------
            # 8. Check result after Human Approval
            # -----------------------------------

            if (
                body.approved is not None
                and write_result is None
            ):

                logger.error(
                    "Missing write result after approval | "
                    "conversation_id=%s",
                    conversation_id,
                )

                return error_response(
                    500,
                    "missing_write_result",
                    "نتیجه عملیات تأییدشده مشخص نیست. "
                    "قبل از تلاش مجدد دیتابیس را بررسی کنید.",
                    conversation_id,
                )

            # -----------------------------------
            # 9. Handle actual write-tool results
            # -----------------------------------

            if write_result:

                tool_status = write_result.get("status")

                logger.info(
                    "Write result received | "
                    "conversation_id=%s | status=%s",
                    conversation_id,
                    tool_status,
                )

                # A. Known write errors
                if tool_status in WRITE_ERRORS:

                    http_status, error_message = (
                        WRITE_ERRORS[tool_status]
                    )

                    logger.warning(
                        "Write failed | "
                        "conversation_id=%s | status=%s",
                        conversation_id,
                        tool_status,
                    )

                    return error_response(
                        http_status,
                        tool_status,
                        error_message,
                        conversation_id,
                    )

                # B. Operation cancelled
                if tool_status == "cancelled":

                    logger.info(
                        "Write cancelled | "
                        "conversation_id=%s",
                        conversation_id,
                    )

                    return ChatResponse(
                        conversation_id=conversation_id,
                        status="cancelled",
                        message=describe_write_result(
                            write_result
                        ),
                    )

                # C. Successful write
                if tool_status in {
                    "created",
                    "updated",
                    "no_change",
                }:

                    logger.info(
                        "Write completed | "
                        "conversation_id=%s | status=%s",
                        conversation_id,
                        tool_status,
                    )

                    # IMPORTANT:
                    # Use the real tool result, not LLM text.
                    return ChatResponse(
                        conversation_id=conversation_id,
                        status="completed",
                        message=describe_write_result(
                            write_result
                        ),
                    )

                # D. Unknown write status
                logger.error(
                    "Unexpected write status | "
                    "conversation_id=%s | status=%s",
                    conversation_id,
                    tool_status,
                )

                return error_response(
                    500,
                    "unexpected_tool_result",
                    "نتیجه ابزار نوشتنی قابل تشخیص نیست.",
                    conversation_id,
                )

            # -----------------------------------
            # 10. Handle normal read/chat responses
            # -----------------------------------

            if not assistant_text:

                logger.error(
                    "Agent returned no final message | "
                    "conversation_id=%s",
                    conversation_id,
                )

                return error_response(
                    500,
                    "missing_agent_response",
                    "Agent پاسخ نهایی معتبری برنگرداند.",
                    conversation_id,
                )

            logger.info(
                "Chat completed | "
                "conversation_id=%s",
                conversation_id,
            )

            return ChatResponse(
                conversation_id=conversation_id,
                status="completed",
                message=assistant_text,
            )

    # ---------------------------------------
    # 11. Error Handling
    # Never automatically retry whole Graph.
    # ---------------------------------------

    except RateLimitError:

        logger.warning(
            "LLM rate limit | conversation_id=%s",
            conversation_id,
        )

        return error_response(
            429,
            "llm_rate_limit",
            "محدودیت تعداد درخواست‌های مدل فعال شده است.",
            conversation_id,
        )

    except (APITimeoutError, TimeoutError):

        logger.exception(
            "LLM timeout | conversation_id=%s",
            conversation_id,
        )

        return error_response(
            504,
            "llm_timeout",
            "مهلت پاسخ‌گویی مدل به پایان رسید.",
            conversation_id,
        )

    except APIConnectionError:

        logger.exception(
            "LLM connection error | conversation_id=%s",
            conversation_id,
        )

        return error_response(
            503,
            "llm_unavailable",
            "اتصال به سرویس مدل برقرار نشد.",
            conversation_id,
        )

    except BadRequestError:

        logger.exception(
            "LLM bad request | conversation_id=%s",
            conversation_id,
        )

        return error_response(
            502,
            "llm_invalid_request",
            "سرویس مدل درخواست را رد کرد. "
            "تاریخچه گفتگو باید بررسی شود.",
            conversation_id,
        )

    except APIStatusError as exc:

        logger.exception(
            "LLM service error | "
            "conversation_id=%s | upstream_status=%s",
            conversation_id,
            exc.status_code,
        )

        http_status = (
            503 if exc.status_code >= 500 else 502
        )

        return error_response(
            http_status,
            "llm_service_error",
            "سرویس مدل با خطا مواجه شد.",
            conversation_id,
        )

    except sqlite3.Error:

        logger.exception(
            "SQLite error | conversation_id=%s",
            conversation_id,
        )

        return error_response(
            500,
            "database_error",
            "خطای داخلی پایگاه داده رخ داد.",
            conversation_id,
        )

    except Exception:

        logger.exception(
            "Unexpected API error | conversation_id=%s",
            conversation_id,
        )

        return error_response(
            500,
            "internal_error",
            "خطای داخلی رخ داد. "
            "پیش از تلاش مجدد وضعیت گفتگو را بررسی کنید.",
            conversation_id,
        )

    # Retry policy is owned by the LLM client. NEVER auto-retry the graph:
    # a previous approved write might have already committed.
    except RateLimitError:
        logger.warning("LLM rate limited | conversation_id=%s", conversation_id)
        return error_response(429, "llm_rate_limit", "محدودیت تعداد درخواست‌های مدل فعال شده است.", conversation_id)
    except (APITimeoutError, TimeoutError):
        logger.exception("LLM timeout | conversation_id=%s", conversation_id)
        return error_response(504, "llm_timeout", "مهلت پاسخ‌گویی مدل به پایان رسید.", conversation_id)
    except APIConnectionError:
        logger.exception("LLM connection failed | conversation_id=%s", conversation_id)
        return error_response(503, "llm_unavailable", "اتصال به سرویس مدل برقرار نشد.", conversation_id)
    except BadRequestError:
        logger.exception("LLM rejected request | conversation_id=%s", conversation_id)
        return error_response(502, "llm_invalid_request", "سرویس مدل درخواست را رد کرد؛ وضعیت تاریخچه گفتگو نیاز به بررسی دارد.", conversation_id)
    except APIStatusError as exc:
        logger.exception("LLM HTTP error | conversation_id=%s | upstream_status=%s", conversation_id, exc.status_code)
        code = 503 if exc.status_code >= 500 else 502
        return error_response(code, "llm_service_error", "سرویس مدل با خطا مواجه شد.", conversation_id)
    except sqlite3.Error:
        logger.exception("SQLite error | conversation_id=%s", conversation_id)
        return error_response(500, "database_error", "خطای داخلی پایگاه داده رخ داد.", conversation_id)
    except Exception:
        logger.exception("Unexpected API error | conversation_id=%s", conversation_id)
        return error_response(500, "internal_error", "خطای داخلی رخ داد؛ وضعیت گفتگو پیش از تلاش دوباره باید بررسی شود.", conversation_id)





@app.post("/chat", response_model=ChatResponse)
def chat(
    body: ChatRequest,
    request: Request,
) -> ChatResponse | JSONResponse:
    """
    Execute the existing agent and persist the conversation.
    """

    # 1. Resolve conversation ID.
    conversation_id = (
        body.conversation_id or uuid4().hex
    )

    # Ensure the agent uses the same conversation ID
    # that will be stored in chat_history.db.
    agent_body = body.model_copy(
        update={"conversation_id": conversation_id}
    )

    # 2. Execute existing LangGraph logic.
    response = run_agent_chat(
        agent_body,
        request,
    )

    # 3. Extract the real API response.
    if isinstance(response, JSONResponse):
        response_data = json.loads(response.body)
        http_status = response.status_code

    else:
        response_data = response.model_dump(
            mode="json"
        )
        http_status = 200

    # 4. Save user input and assistant response.
    try:
        save_exchange(
            conversation_id=conversation_id,
            message=body.message,
            approved=body.approved,
            assistant_text=response_data["message"],
            status=response_data["status"],
            approval=response_data.get("approval"),
            http_status=http_status,
        )

        logger.info(
            "Chat history saved | conversation_id=%s",
            conversation_id,
        )

    except (sqlite3.Error, OSError):
        logger.exception(
            "Failed to save chat history | "
            "conversation_id=%s",
            conversation_id,
        )

        # Do not retry the Graph here.
        # A write operation might already have succeeded.

    # 5. Return the actual agent response.
    return response
