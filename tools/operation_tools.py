
from __future__ import annotations

import logging
import os
import sqlite3
import unicodedata
from contextlib import closing
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import ToolException, tool
from pydantic import BaseModel, ConfigDict, Field, model_validator, ValidationError



from tools.count_tools import DB_PATH, TaskPriority, TaskStatus


create_logger = logging.getLogger("task_agent.create_task")
update_logger = logging.getLogger("task_agent.update_task_status")


def _normalize_name(text: str) -> str:
    """Normalize common Persian/Arabic variants for exact full-name matching."""
    text = unicodedata.normalize("NFKC", text)
    text = (
        text.replace("ي", "ی")
        .replace("ى", "ی")
        .replace("ك", "ک")
        .replace("\u200c", " ")
        .replace("\u200d", " ")
    )
    return " ".join(text.split()).casefold()


def _today_iso() -> str:
    timezone = os.getenv("TASK_TIMEZONE", "Asia/Tehran")

    try:
        return datetime.now(ZoneInfo(timezone)).date().isoformat()

    except ZoneInfoNotFoundError as exc:
        create_logger.error(
            "Invalid TASK_TIMEZONE: %s",
            timezone
        )
        raise ToolException(
            f"Invalid TASK_TIMEZONE: {timezone}"
        ) from exc

def _connect() -> sqlite3.Connection:
    if not DB_PATH.is_file():
        create_logger.error("search_tasks database file not found: %s", DB_PATH)
        raise ToolException("Database tasks.db not found. Run init_db.py first.")

    connection = sqlite3.connect(DB_PATH, timeout=5, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.create_function("normalize_fa", 1, _normalize_name, deterministic=True)
    create_logger.info("Database tasks.db found")
    return connection


def _get_task(conn: sqlite3.Connection, task_id: int) -> dict[str, Any]:
    row = conn.execute(
        """SELECT t.id, t.title, t.description, t.create_time,
                  t.status, t.assignee_id, t.priority, t.due_time,
                  u.fullname AS assignee_name, u.department
           FROM tasks AS t
           LEFT JOIN users AS u ON u.id = t.assignee_id
           WHERE t.id = ?""",
        (task_id,),
    ).fetchone()
    if row is None:
        create_logger.error("Task was not found after a database write.")
        raise ToolException("Task was not found after a database write.")

    create_logger.info("Task was found")    
    return dict(row)


# ------------------------------------------------------------
# CREATE TASK
# ------------------------------------------------------------

class CreateTaskInput(BaseModel):
    """Validated fields required to create a new task."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(
        min_length=3,
        max_length=200,
        description=(
            "Required, user-provided task title. Do not invent a title. "
            "Ask the user if a title is missing."
        ),
    )
    assignee_name: str | None = Field(
        default=None,
        min_length=2,
        max_length=120,
        description=(
            "Employee's full name from the user's request. Look it up "
            "dynamically in the users table; never hardcode names. "
            "Provide exactly one of assignee_name and assignee_id."
        ),
    )
    assignee_id: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Employee ID only if the ID was explicitly provided or reliably "
            "resolved from database data. Never guess an ID. "
            "Provide exactly one of assignee_name and assignee_id."
        ),
    )
    description: str = Field(
        default="",
        max_length=5000,
        description=(
            "Optional user-provided task description. Leave empty if absent. "
            "Do not invent or infer task details."
        ),
    )
    priority: TaskPriority = Field(
        default=TaskPriority.MEDIUM,
        description=(
            "Task priority: 'Low', 'Medium', 'High', or 'Critical'. "
            "Defaults to 'Medium' when the user does not specify priority."
        ),
    )

    @model_validator(mode="after")
    def validate_assignee(self) -> "CreateTaskInput":
        if (self.assignee_name is None) == (self.assignee_id is None):
            raise ValueError(
                "Provide exactly one of assignee_name or assignee_id."
            )
        return self


@tool("create_task", args_schema=CreateTaskInput)
def create_task(
    title: str,
    assignee_name: str | None = None,
    assignee_id: int | None = None,
    description: str = "",
    priority: TaskPriority = TaskPriority.MEDIUM,
) -> dict[str, Any]:
    """Create a new task for an existing employee after user confirmation.

    Use only for requests to CREATE a task. The user must provide a clear title
    and identify the employee by full name or verified numeric ID; otherwise
    ask a clarifying question BEFORE calling this tool. Resolve employees
    dynamically from the users table. If a name matches multiple employees,
    return ambiguity instead of choosing one.

    Task ID and create_time are generated by the database/application.
    New tasks start as Open. Priority defaults to Medium. Description is
    optional. due_time is empty until status changes to Done, according to
    this project's simplified completion-date rule.

    IMPORTANT: This action writes to the database. The application must first
    collect the user's explicit approval of the task details. Do not call
    this tool for counting, searching, or status changes.

    Returns a structured result with status='created' and the created task,
    or an explanation if the assignee cannot be resolved uniquely.
    """
    try:
        args = CreateTaskInput.model_validate({
            "title": title,
            "assignee_name": assignee_name,
            "assignee_id": assignee_id,
            "description": description,
            "priority": priority,
        })

    except ValidationError as exc:
        create_logger.warning(
            "create_task input validation failed: %s",
            exc.errors()
        )
        raise ToolException(
            "Invalid create_task parameters."
        ) from exc

    else:
        create_logger.info(
            "create_task input validation succeeded"
        )

    try:
        with closing(_connect()) as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")

                if args.assignee_id is not None:
                    people = [
                        dict(row)
                        for row in conn.execute(
                            "SELECT id, fullname, department FROM users WHERE id = ?",
                            (args.assignee_id,),
                        )
                    ]
                else:
                    people = [
                        dict(row)
                        for row in conn.execute(
                            """SELECT id, fullname, department FROM users
                               WHERE normalize_fa(fullname) = ?""",
                            (_normalize_name(args.assignee_name),),
                        )
                    ]

                if not people:
                    create_logger.warning(
                        "create_task employee not found | assignee=%s",
                        args.assignee_name or args.assignee_id
                    )
                    conn.rollback()
                    return {
                        "status": "person_not_found",
                        "message": "No matching employee exists in users.",
                        "requested_assignee": args.assignee_name or args.assignee_id,
                    }

                if len(people) > 1:
                    create_logger.warning(
                    "create_task ambiguous employee | name=%s | matches=%d",
                    args.assignee_name,
                    len(people)
                    )
                    conn.rollback()
                    return {
                        "status": "ambiguous_person",
                        "message": "Multiple employees match; ask the user for an ID.",
                        "matching_people": people,
                    }

                employee = people[0]
                cursor = conn.execute(
                    """INSERT INTO tasks
                       (title, description, create_time, status,
                        assignee_id, priority, due_time)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        args.title,
                        args.description,
                        _today_iso(),
                        TaskStatus.OPEN.value,
                        employee["id"],
                        args.priority.value,
                        "",  # current DB: due_time NOT NULL; Open has no completion date
                    ),
                )
                task = _get_task(conn, cursor.lastrowid)
                conn.commit()
                create_logger.info("Created task id=%s", task["id"])
                return {"status": "created", "task": task}

            except Exception:
                conn.rollback()
                raise
    except sqlite3.Error as exc:
        create_logger.exception("create_task failed")
        raise ToolException("Failed to create task in SQLite.") from exc


# ------------------------------------------------------------
# UPDATE TASK STATUS
# ------------------------------------------------------------

class UpdateTaskStatusInput(BaseModel):
    """Validated input to modify the status of an existing task."""

    model_config = ConfigDict(extra="forbid")

    task_id: int = Field(
        ge=1,
        description=(
            "Required positive task ID. Use the exact ID provided by the "
            "user or retrieved from the database; never guess."
        ),
    )
    new_status: TaskStatus = Field(
        description=(
            "New status: 'Open' (باز), 'In Progress' (در حال اجرا), "
            "'Review' (در حال مرور), or 'Done' (بسته‌شده)."
        ),
    )


@tool("update_task_status", args_schema=UpdateTaskStatusInput)
def update_task_status(
    task_id: int,
    new_status: TaskStatus,
) -> dict[str, Any]:
    """Change the status of one existing task after user confirmation.

    Use only when the user explicitly requests a status change and the task
    ID and target status are known. Ask for missing information instead of
    guessing. The application must verify explicit user approval before
    invoking this write tool.

    BUSINESS RULE: Changing a task to Done saves today's date in due_time
    (completion date for this project). Reopening a Done task clears its
    completion date. Other status changes leave due_time unchanged.
    If the status already equals the requested status, make no changes.

    Returns status='updated', 'no_change', or 'task_not_found', including
    the task data when available. Do not use for task creation or reading.
    """
    try:
        args = UpdateTaskStatusInput.model_validate({
            "task_id": task_id,
            "new_status": new_status,
        })

    except ValidationError as exc:
        update_logger.warning(
            "update_task_status input validation failed: %s",
            exc.errors()
        )
        raise ToolException(
            "Invalid update_task_status parameters."
        ) from exc

    else:
        update_logger.info(
            "update_task_status input validation succeeded | task_id=%s",
            args.task_id
        )

    try:
        with closing(_connect()) as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")

                row = conn.execute(
                    "SELECT status FROM tasks WHERE id = ?",
                    (args.task_id,),
                ).fetchone()

                if row is None:
                    update_logger.warning(
                    "update_task_status task not found | task_id=%s",
                    args.task_id
                    )
                    conn.rollback()
                    return {
                        "status": "task_not_found",
                        "task_id": args.task_id,
                        "message": "No task exists with the requested ID.",
                    }

                previous_status = row["status"]
                if previous_status == args.new_status.value:
                    update_logger.info(
                            "update_task_status no change | task_id=%s | status=%s",
                            args.task_id,
                            previous_status
                        )
                    task = _get_task(conn, args.task_id)
                    conn.rollback()
                    return {"status": "no_change", "task": task}

                if args.new_status == TaskStatus.DONE:
                    conn.execute(
                        "UPDATE tasks SET status = ?, due_time = ? WHERE id = ?",
                        (args.new_status.value, _today_iso(), args.task_id),
                    )
                elif previous_status == TaskStatus.DONE.value:
                    # Reopened: no longer has a completion date.
                    conn.execute(
                        "UPDATE tasks SET status = ?, due_time = ? WHERE id = ?",
                        (args.new_status.value, "", args.task_id),
                    )
                else:
                    conn.execute(
                        "UPDATE tasks SET status = ? WHERE id = ?",
                        (args.new_status.value, args.task_id),
                    )

                task = _get_task(conn, args.task_id)
                conn.commit()
                update_logger.info(
                    "Updated task id=%s status=%s->%s",
                    args.task_id,
                    previous_status,
                    args.new_status.value,
                )
                return {
                    "status": "updated",
                    "previous_status": previous_status,
                    "task": task,
                }

            except Exception:
                conn.rollback()
                raise
    except sqlite3.Error as exc:
        update_logger.exception("update_task_status failed")
        raise ToolException("Failed to update task status in SQLite.") from exc