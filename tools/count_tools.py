
from __future__ import annotations

import logging
import sqlite3
import unicodedata
from contextlib import closing
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any

from dateutil.relativedelta import relativedelta
from langchain_core.tools import ToolException, tool
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger("task_agent.count_tasks")
DB_PATH = Path(__file__).resolve().parents[1] / "tasks.db"


class TaskStatus(str, Enum):
    OPEN = "Open"
    IN_PROGRESS = "In Progress"
    REVIEW = "Review"
    DONE = "Done"


class TaskPriority(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


def _normalize_persian(text: str | None) -> str:
    """Normalize Persian/Arabic keyboard variants and spacing for exact matches."""
    if text is None:
        return ""
    value = unicodedata.normalize("NFKC", text)
    value = (
        value.replace("ي", "ی")
        .replace("ى", "ی")
        .replace("ك", "ک")
        .replace("\u200c", " ")
        .replace("\u200d", " ")
    )
    return " ".join(value.split()).casefold()


class CountTasksInput(BaseModel):
    """Validated optional filters for counting task records."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    status: TaskStatus | None = Field(
        default=None,
        description=(
            "Filter by status: 'Open' (باز), 'In Progress' (در حال اجرا), "
            "'Review' (در حال مرور), or 'Done' (بسته شده). "
            "Omit to count all statuses."
        ),
    )
    priority: TaskPriority | None = Field(
        default=None,
        description=(
            "Filter by 'Low', 'Medium', 'High', or 'Critical'. "
            "'High' and 'Critical' are distinct. Omit to include all priorities."
        ),
    )
    assignee_name: str | None = Field(
        default=None,
        min_length=2,
        max_length=120,
        description=(
            "Full name of the assigned employee, as provided by the user. "
            "Match dynamically against users.fullname after Persian normalization. "
            "Do not invent an employee ID. Cannot be combined with assignee_id. "
            "If the name is incomplete, ask for the full name."
        ),
    )
    assignee_id: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Positive employee ID only when explicitly known. "
            "Never guess IDs from names. Cannot be combined with assignee_name."
        ),
    )
    department: str | None = Field(
        default=None,
        min_length=2,
        max_length=100,
        description=(
            "Exact department name from the user's question, such as 'فنی'. "
            "Match dynamically against users.department after Persian normalization. "
            "Can be combined with status, priority, and employee filters."
        ),
    )
    created_from: date | None = Field(
        default=None,
        description="Inclusive creation-date lower bound, YYYY-MM-DD.",
    )
    created_to: date | None = Field(
        default=None,
        description="Inclusive creation-date upper bound, YYYY-MM-DD.",
    )
    due_time_from: date | None = Field(
        default=None,
        description=(
            "Inclusive due_time lower bound, YYYY-MM-DD. "
            "For Done tasks, due_time is the completion date."
        ),
    )
    due_time_to: date | None = Field(
        default=None,
        description="Inclusive due_time upper bound, YYYY-MM-DD.",
    )
    last_n_months: int | None = Field(
        default=None,
        ge=1,
        le=120,
        description=(
            "Number of calendar months back from today for completed-task "
            "counts. Requires status='Done'. Do not calculate dates in the LLM. "
            "Cannot be combined with due_time_from or due_time_to."
        ),
    )

    @model_validator(mode="after")
    def validate_filters(self) -> "CountTasksInput":
        if self.assignee_name is not None and self.assignee_id is not None:
            raise ValueError("Use assignee_name or assignee_id, not both")
        if (
            self.created_from is not None
            and self.created_to is not None
            and self.created_from > self.created_to
        ):
            raise ValueError("created_from must be on or before created_to")
        if (
            self.due_time_from is not None
            and self.due_time_to is not None
            and self.due_time_from > self.due_time_to
        ):
            raise ValueError("due_time_from must be on or before due_time_to")
        if self.last_n_months is not None:
            if self.status != TaskStatus.DONE:
                raise ValueError("last_n_months requires status='Done'")
            if self.due_time_from is not None or self.due_time_to is not None:
                raise ValueError(
                    "last_n_months cannot be combined with explicit due_time dates"
                )
        return self


@tool("count_tasks", args_schema=CountTasksInput)
def count_tasks(
    status: TaskStatus | None = None,
    priority: TaskPriority | None = None,
    assignee_name: str | None = None,
    assignee_id: int | None = None,
    department: str | None = None,
    created_from: date | None = None,
    created_to: date | None = None,
    due_time_from: date | None = None,
    due_time_to: date | None = None,
    last_n_months: int | None = None,
) -> dict[str, Any]:
    """COUNT tasks with optional combined employee, department and task filters.

    Use for HOW MANY questions such as 'How many Open tasks does Ali Saberi
    have?' or 'How many High-priority tasks are in the technical department?'.
    All filters are combined with AND. Use search_tasks to LIST individual tasks,
    and analyze_tasks for rankings, averages or overdue analytics.

    Names and departments are resolved from the database, not a fixed list.
    Unknown or ambiguous employee names return structured results so the
    assistant can ask for clarification instead of fabricating a count.
    For Done tasks, due_time represents completion date in this project.
    """
    try:
        args = CountTasksInput.model_validate(
            {
                "status": status,
                "priority": priority,
                "assignee_name": assignee_name,
                "assignee_id": assignee_id,
                "department": department,
                "created_from": created_from,
                "created_to": created_to,
                "due_time_from": due_time_from,
                "due_time_to": due_time_to,
                "last_n_months": last_n_months,
            }
        )

    except ValidationError as exc:
        logger.warning(
            "count_tasks input validation failed: %s",
            exc.errors()
        )

        raise ToolException(
            "Invalid count_tasks parameters."
        ) from exc

    else:
        logger.info(
            "count_tasks input validation succeeded"
        )

    if not DB_PATH.is_file():
        logger.error("count_tasks database file not found: %s", DB_PATH)
        raise ToolException("Database tasks.db was not found. Run init_db.py first.")

    conditions: list[str] = []
    params: list[Any] = []
    applied_filters: dict[str, Any] = {}

    if args.status is not None:
        conditions.append("t.status = ?")
        params.append(args.status.value)
        applied_filters["status"] = args.status.value

    if args.priority is not None:
        conditions.append("t.priority = ?")
        params.append(args.priority.value)
        applied_filters["priority"] = args.priority.value

    if args.assignee_id is not None:
        conditions.append("t.assignee_id = ?")
        params.append(args.assignee_id)
        applied_filters["assignee_id"] = args.assignee_id

    if args.department is not None:
        conditions.append("normalize_fa(u.department) = ?")
        params.append(_normalize_persian(args.department))
        applied_filters["department"] = args.department

    if args.assignee_name is not None:
        applied_filters["assignee_name"] = args.assignee_name

    date_filters = {
        "created_from": ("t.create_time >= ?", args.created_from),
        "created_to": ("t.create_time <= ?", args.created_to),
        "due_time_from": ("t.due_time >= ?", args.due_time_from),
        "due_time_to": ("t.due_time <= ?", args.due_time_to),
    }
    for key, (condition, value) in date_filters.items():
        if value is not None:
            conditions.append(condition)
            params.append(value.isoformat())
            applied_filters[key] = value.isoformat()

    if args.last_n_months is not None:
        today = date.today()
        start = today - relativedelta(months=args.last_n_months)
        conditions.extend(("t.due_time >= ?", "t.due_time <= ?"))
        params.extend((start.isoformat(), today.isoformat()))
        applied_filters.update(
            {
                "last_n_months": args.last_n_months,
                "due_time_from": start.isoformat(),
                "due_time_to": today.isoformat(),
            }
        )

    try:
        with closing(
            sqlite3.connect(DB_PATH.as_uri() + "?mode=ro", uri=True, timeout=5)
        ) as conn:
            conn.row_factory = sqlite3.Row
            conn.create_function(
                "normalize_fa", 1, _normalize_persian, deterministic=True
            )

            if args.assignee_name is not None:
                # Resolve a unique employee first; never aggregate two people
                # with the same full name into a single person's task count.
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
                    person_params.append(_normalize_persian(args.department))
                people = [
                    dict(row)
                    for row in conn.execute(person_sql, person_params).fetchall()
                ]

                if not people:

                    logger.warning(
                            "count_tasks employee not found: %s",
                            args.assignee_name
                                )
                    return {
                        "status": "person_not_found",
                        "count": None,
                        "applied_filters": applied_filters,
                        "message": (
                            "Employee not found in users. Ask for the correct "
                            "full name or department; do not invent a count."
                        ),
                    }
                if len(people) > 1:
                    logger.warning(
                            "count_tasks ambiguous employee: %s, matches=%d",
                            args.assignee_name,
                            len(people)
                        )
                    return {
                        "status": "ambiguous_person",
                        "count": None,
                        "applied_filters": applied_filters,
                        "matching_people": people,
                        "message": "Ask the user which employee they mean.",
                    }

                conditions.append("t.assignee_id = ?")
                params.append(people[0]["id"])
                applied_filters["resolved_assignee_id"] = people[0]["id"]

            from_sql = "FROM tasks AS t LEFT JOIN users AS u ON u.id = t.assignee_id"
            where_sql = " WHERE " + " AND ".join(conditions) if conditions else ""
            sql = "SELECT COUNT(*) " + from_sql + where_sql
            count = conn.execute(sql, params).fetchone()[0]

            result: dict[str, Any] = {
                "count": int(count),
                "applied_filters": applied_filters,
            }

            if args.department is not None or args.assignee_name is not None:
                missing_people = conn.execute(
                    """SELECT COUNT(*) FROM tasks AS t
                       LEFT JOIN users AS u ON u.id = t.assignee_id
                       WHERE t.assignee_id IS NOT NULL AND u.id IS NULL"""
                ).fetchone()[0]
                if missing_people:
                    result["warning"] = (
                        f"{missing_people} tasks have no matching users row. "
                        "Counts filtered by employee or department may be incomplete."
                    )
            logger.info("count_tasks completed | count=%s", count)
            return result

    except sqlite3.Error as exc:
        logger.exception("count_tasks database query failed")
        raise ToolException("Could not retrieve task count from SQLite.") from exc