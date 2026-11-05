# ---------------------------------------------------
# Search Tasks: append this block to your existing tools.py
# Reuses TaskStatus, TaskPriority, DB_PATH, logger, and imports
# already present in tools.py.
# ---------------------------------------------------

import unicodedata
import logging
import sqlite3
from contextlib import closing
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any

from langchain_core.tools import ToolException, tool
from pydantic import BaseModel, ConfigDict, Field, model_validator,ValidationError

from dateutil.relativedelta import relativedelta
from tools.count_tools import TaskStatus, TaskPriority, DB_PATH
logger = logging.getLogger("task_agent.search_tasks")


def _normalize_persian(value: str | None) -> str:
    """Normalize common Arabic/Persian keyboard differences for exact matching."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", value)
    text = (
        text.replace("ي", "ی")
        .replace("ى", "ی")
        .replace("ك", "ک")
        .replace("\u200c", " ")  # zero-width non-joiner
        .replace("\u200d", " ")
    )
    return " ".join(text.split()).casefold()


class SearchTasksInput(BaseModel):
    """Validated search filters for retrieving task records."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    assignee_name: str | None = Field(
        default=None,
        min_length=2,
        max_length=120,
        description=(
            "Full name of the assigned employee, e.g. 'علی صابری'. "
            "Use when the user names a person; do not invent a numeric ID. "
            "Names are matched exactly after Persian character and whitespace "
            "normalization. If the name is incomplete, ask for clarification."
        ),
    )

    assignee_id: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Numeric employee ID, only when explicitly known. "
            "Do not guess IDs from names. Cannot be combined with assignee_name."
        ),
    )

    department: str | None = Field(
        default=None,
        min_length=2,
        max_length=100,
        description=(
            "Exact department name in Persian, e.g. 'پشتیبانی' or 'فنی'. "
            "Matched after Persian character and whitespace normalization. "
            "Omit to include every department."
        ),
    )

    status: TaskStatus | None = Field(
        default=None,
        description=(
            "Current task status: 'Open' (باز), 'In Progress' "
            "(در حال اجرا), 'Review' (در حال مرور), 'Done' (بسته‌شده). "
            "Omit to include all statuses."
        ),
    )

    priority: TaskPriority | None = Field(
        default=None,
        description=(
            "Task priority: 'Low', 'Medium', 'High', or 'Critical'. "
            "When the user says 'اولویت بالا', use 'High', not 'Critical'. "
            "Omit to include all priorities."
        ),
    )

    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description=(
            "Maximum number of task records returned in one call (1-100). "
            "Defaults to 20. Use pagination rather than requesting excessive output."
        ),
    )

    offset: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of matching records to skip for pagination. "
            "Defaults to 0. For the next page, increase offset by limit."
        ),
    )

    @model_validator(mode="after")
    def validate_identity_filter(self) -> "SearchTasksInput":
        if self.assignee_name is not None and self.assignee_id is not None:
            raise ValueError(
                "Use either assignee_name or assignee_id, not both."
            )
        return self


@tool("search_tasks", args_schema=SearchTasksInput)
def search_tasks(
    assignee_name: str | None = None,
    assignee_id: int | None = None,
    department: str | None = None,
    status: TaskStatus | None = None,
    priority: TaskPriority | None = None,
    limit: int = 20,
    offset: int = 0,
) -> dict[str, Any]:
    """Search and LIST task records using optional, structured filters.

    Use this tool when the user asks to show tasks assigned to a named
    employee, tasks in a department, or tasks with a particular status
    or priority. Multiple filters are combined with AND.

    This tool retrieves task records, including their titles, descriptions,
    status, priority, dates, and available assignee/department information.
    A LEFT JOIN retains tasks whose user record is missing. Records are
    sorted by creation date (newest first), then task ID (descending).

    Use full employee names; never guess assignee IDs. If a name matches
    multiple employees, the tool returns an ambiguity result; ask the user
    to clarify. If personnel data is incomplete, department and name
    searches only cover employees currently present in the users table.

    Do NOT use this tool to count tasks only (use count_tasks), perform
    grouped statistics, create tasks, or update task status.

    Returns a structured dictionary containing status, total_count,
    returned_count, has_more, applied_filters, and tasks.
    """
    try:
        args = SearchTasksInput.model_validate({
            "assignee_name": assignee_name,
            "assignee_id": assignee_id,
            "department": department,
            "status": status,
            "priority": priority,
            "limit": limit,
            "offset": offset,
        })
        logger.info("search_tasks input validation succeeded")
    except ValidationError as exc:
        logger.warning(
            "search_tasks invalid arguments: %s",
            exc.errors()
        )

        raise ToolException(
            "Invalid search parameters. Please correct the filters."
        ) from exc

    if not DB_PATH.is_file():
        logger.error("search_tasks database file not found: %s", DB_PATH)
        raise ToolException("Database tasks.db was not found beside tools.py.")

    conditions: list[str] = []
    parameters: list[Any] = []
    applied_filters: dict[str, Any] = {}

    if args.status is not None:
        conditions.append("t.status = ?")
        parameters.append(args.status.value)
        applied_filters["status"] = args.status.value

    if args.priority is not None:
        conditions.append("t.priority = ?")
        parameters.append(args.priority.value)
        applied_filters["priority"] = args.priority.value

    if args.assignee_id is not None:
        conditions.append("t.assignee_id = ?")
        parameters.append(args.assignee_id)
        applied_filters["assignee_id"] = args.assignee_id

    if args.department is not None:
        conditions.append("normalize_fa(u.department) = ?")
        parameters.append(_normalize_persian(args.department))
        applied_filters["department"] = args.department

    if args.assignee_name is not None:
        applied_filters["assignee_name"] = args.assignee_name

    try:
        with closing(
            sqlite3.connect(
                DB_PATH.as_uri() + "?mode=ro", uri=True, timeout=5
            )
        ) as conn:
            conn.row_factory = sqlite3.Row
            conn.create_function(
                "normalize_fa", 1, _normalize_persian, deterministic=True
            )
            conn.execute("BEGIN")  # Consistent snapshot for count and page.


            if args.department is not None:

                department_exists = conn.execute(
                    """
                    SELECT 1
                    FROM users
                    WHERE normalize_fa(department) = ?
                    LIMIT 1
                    """,
                    (_normalize_persian(args.department),)
                ).fetchone()

                if department_exists is None:
                    logger.warning(
                        "search_tasks department not found: %s",
                        args.department
                    )
                    return {
                        "status": "department_not_found",
                        "total_count": 0,
                        "returned_count": 0,
                        "has_more": False,
                        "applied_filters": applied_filters,
                        "tasks": [],
                        "message": (
                            "No matching department was found "
                            "in the users table."
                        ),
                    }

            if args.assignee_name is not None:
                # Resolve person first to prevent returning combined results
                # for different people who share the same full name.
                person_sql = """
                    SELECT id, fullname, department
                    FROM users
                    WHERE normalize_fa(fullname) = ?
                """
                person_params: list[Any] = [
                    _normalize_persian(args.assignee_name)
                ]
                if args.department is not None:
                    person_sql += " AND normalize_fa(department) = ?"
                    person_params.append(
                        _normalize_persian(args.department)
                    )
                people = [
                    dict(row)
                    for row in conn.execute(person_sql, person_params)
                ]

                if not people:
                    logger.warning(
                        "search_tasks employee not found: %s",
                        args.assignee_name
                    )
                    return {
                        "status": "person_not_found",
                        "total_count": 0,
                        "returned_count": 0,
                        "has_more": False,
                        "applied_filters": applied_filters,
                        "tasks": [],
                        "message": (
                            "Employee not found in the users table. "
                            "The personnel dataset may be incomplete."
                        ),
                    }

                if len(people) > 1:
                    logger.warning(
                        "search_tasks ambiguous employee: %s | matches=%d",
                        args.assignee_name,
                        len(people)
                    )
                    return {
                        "status": "ambiguous_person",
                        "total_count": None,
                        "returned_count": 0,
                        "has_more": False,
                        "applied_filters": applied_filters,
                        "tasks": [],
                        "matching_people": people,
                        "message": "Ask the user to specify the employee.",
                    }

                conditions.append("t.assignee_id = ?")
                parameters.append(people[0]["id"])
                applied_filters["resolved_assignee_id"] = people[0]["id"]

            from_clause = "FROM tasks AS t LEFT JOIN users AS u ON u.id = t.assignee_id"
            where_clause = (
                " WHERE " + " AND ".join(conditions) if conditions else ""
            )
            total_count = conn.execute(
                "SELECT COUNT(*) " + from_clause + where_clause,
                parameters,
            ).fetchone()[0]

            select_sql = (
                "SELECT t.id, t.title, t.description, t.create_time, "
                "t.status, t.assignee_id, t.priority, t.due_time, "
                "u.fullname AS assignee_name, u.department "
                + from_clause
                + where_clause
                + " ORDER BY t.create_time DESC, t.id DESC LIMIT ? OFFSET ?"
            )
            tasks = [
                dict(row)
                for row in conn.execute(
                    select_sql, [*parameters, args.limit, args.offset]
                )
            ]

            result: dict[str, Any] = {
                "status": "success",
                "total_count": int(total_count),
                "returned_count": len(tasks),
                "has_more": args.offset + len(tasks) < total_count,
                "limit": args.limit,
                "offset": args.offset,
                "applied_filters": applied_filters,
                "tasks": tasks,
            }

            # Without complete personnel data, filtered employee/department
            # searches cannot be guaranteed to cover all organizational tasks.
            if args.department is not None or args.assignee_name is not None:
                missing_user_count = conn.execute(
                    """SELECT COUNT(*) FROM tasks t
                       LEFT JOIN users u ON u.id = t.assignee_id
                       WHERE t.assignee_id IS NOT NULL AND u.id IS NULL"""
                ).fetchone()[0]
                if missing_user_count:
                    result["warning"] = (
                        f"{missing_user_count} tasks have no matching users row. "
                        "Results filtered by employee or department may be incomplete."
                    )


            logger.info(
                    "search_tasks completed | total_count=%d | returned_count=%d | filters=%s",
                    total_count,
                    len(tasks),
                    applied_filters
                )
            return result

    except sqlite3.Error as exc:
        logger.exception("search_tasks database query failed")
        raise ToolException("Could not search tasks in SQLite.") from exc