
from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import closing
from datetime import datetime
from enum import Enum
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import ToolException, tool
from pydantic import BaseModel, ConfigDict, Field, model_validator, ValidationError
from tools.count_tools import DB_PATH, TaskPriority, TaskStatus

logger = logging.getLogger("task_agent.analyze_tasks")


class AnalysisType(str, Enum):
    WORKLOAD_RANKING = "workload_ranking"
    AVERAGE_COMPLETION_TIME = "average_completion_time"
    OVERDUE_COUNT = "overdue_count"


class GroupBy(str, Enum):
    ASSIGNEE = "assignee"
    DEPARTMENT = "department"



class SortOrder(str, Enum):
    DESC = "desc"  # Most tasks first
    ASC = "asc"    # Fewest tasks first



class AnalyzeTasksInput(BaseModel):
    """Strict and documented arguments for organizational task analytics."""

    model_config = ConfigDict(extra="forbid")

    analysis_type: AnalysisType = Field(
        description=(
            "Required analysis: 'workload_ranking' for who/which department "
            "has most or fewest tasks; 'average_completion_time' for the "
            "mean number of days from creation to completion; "
            "'overdue_count' for how many non-Done tasks are past due."
        )
    )
    group_by: GroupBy | None = Field(
        default=None,
        description=(
            "Required only for workload_ranking. 'assignee' ranks employees "
            "using users.id; 'department' ranks departments using "
            "users.department. Names and departments are fetched from the DB."
        ),
    )
    sort_order: SortOrder | None = Field(
        default=None,
        description=(
            "Only for workload_ranking. 'desc' means most tasks; 'asc' means "
            "fewest tasks. Defaults to 'desc' for ranking."
        ),
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description=(
            "Only for workload_ranking. Number of ranked groups to return "
            "(1 to 20; default 1). All groups tied at the last position "
            "are also returned."
        ),
    )
    status: TaskStatus | None = Field(
        default=None,
        description=(
            "Optional task-status filter. For workload_ranking, filter by "
            "'Open', 'In Progress', 'Review', or 'Done'. For overdue_count, "
            "only non-Done values are allowed; omit to include all unfinished "
            "statuses. For average_completion_time, only 'Done' or omitted "
            "is valid because the calculation always uses Done tasks."
        ),
    )
    priority: TaskPriority | None = Field(
        default=None,
        description=(
            "Optional task-priority filter: 'Low', 'Medium', 'High', "
            "'Critical'. Omit to include every priority. 'High' and "
            "'Critical' are distinct levels."
        ),
    )

    @model_validator(mode="after")
    def validate_for_analysis(self) -> "AnalyzeTasksInput":
        if self.analysis_type == AnalysisType.WORKLOAD_RANKING:
            if self.group_by is None:
                raise ValueError("group_by is required for workload_ranking")
        else:
            if any(x is not None for x in (self.group_by, self.sort_order, self.limit)):
                raise ValueError(
                    "group_by, sort_order and limit apply only to workload_ranking"
                )

        if (
            self.analysis_type == AnalysisType.AVERAGE_COMPLETION_TIME
            and self.status not in (None, TaskStatus.DONE)
        ):
            raise ValueError("average_completion_time always uses Done tasks")

        if (
            self.analysis_type == AnalysisType.OVERDUE_COUNT
            and self.status == TaskStatus.DONE
        ):
            raise ValueError("Done tasks cannot be overdue in this project")

        return self


def _today_iso() -> str:
    """Current calendar date in the business's configurable time zone."""
    time_zone = os.getenv("TASK_TIMEZONE", "Asia/Tehran")
    try:
        return datetime.now(ZoneInfo(time_zone)).date().isoformat()
    except ZoneInfoNotFoundError as exc:
        logger.error(
            "analyze_tasks invalid timezone: %s",
            time_zone
        )
        raise ToolException(f"Invalid TASK_TIMEZONE: {time_zone}") from exc


def _connect_readonly() -> sqlite3.Connection:
    if not DB_PATH.is_file():
        logger.error(
            "analyze_tasks database file not found: %s",
            DB_PATH
        )
        raise ToolException("tasks.db not found. Run init_db.py first.")
    conn = sqlite3.connect(
        DB_PATH.as_uri() + "?mode=ro", uri=True, timeout=5
    )
    conn.row_factory = sqlite3.Row
    return conn


def _task_filters(args: AnalyzeTasksInput, alias: str = "t") -> tuple[list[str], list[Any]]:
    """Build approved, parameterized status/priority predicates."""
    conditions: list[str] = []
    values: list[Any] = []
    if args.status is not None:
        conditions.append(f"{alias}.status = ?")
        values.append(args.status.value)
    if args.priority is not None:
        conditions.append(f"{alias}.priority = ?")
        values.append(args.priority.value)
    return conditions, values


def _workload_ranking(conn: sqlite3.Connection, args: AnalyzeTasksInput) -> dict[str, Any]:
    # Keep filters in LEFT JOIN's ON clause. Otherwise users with zero
    # matching tasks disappear, which makes 'fewest' ranking incorrect.
    conditions, values = _task_filters(args)
    extra_on = " AND " + " AND ".join(conditions) if conditions else ""
    direction = (args.sort_order or SortOrder.DESC).value
    limit = args.limit or 1

    if args.group_by == GroupBy.ASSIGNEE:
        sql = f"""
            SELECT u.id AS assignee_id, u.fullname AS assignee_name,
                   u.department, COUNT(t.id) AS task_count
            FROM users AS u
            LEFT JOIN tasks AS t ON t.assignee_id = u.id{extra_on}
            GROUP BY u.id, u.fullname, u.department
            ORDER BY task_count {direction.upper()}, u.id ASC
        """
    else:
        sql = f"""
            SELECT u.department, COUNT(t.id) AS task_count
            FROM users AS u
            LEFT JOIN tasks AS t ON t.assignee_id = u.id{extra_on}
            GROUP BY u.department
            ORDER BY task_count {direction.upper()}, u.department ASC
        """

    all_groups = [dict(row) for row in conn.execute(sql, values).fetchall()]

    # Preserve everyone tied with the Nth result, including limit=1.
    chosen = all_groups[:limit]
    if chosen:
        cutoff = chosen[-1]["task_count"]
        for row in all_groups[limit:]:
            if row["task_count"] != cutoff:
                break
            chosen.append(row)

    # A missing users record means its tasks cannot be assigned to an
    # employee name or department. Explicitly report this limitation.
    missing_sql = """
        SELECT COUNT(*) FROM tasks AS t
        LEFT JOIN users AS u ON u.id = t.assignee_id
        WHERE u.id IS NULL
    """
    if conditions:
        missing_sql += " AND " + " AND ".join(conditions)
    missing_count = int(conn.execute(missing_sql, values).fetchone()[0])

    return {
        "status": "partial_data" if missing_count else ("success" if chosen else "no_data"),
        "analysis_type": args.analysis_type.value,
        "group_by": args.group_by.value,
        "sort_order": direction,
        "requested_limit": limit,
        "returned_count": len(chosen),
        "total_groups": len(all_groups),
        "results": chosen,
        "unresolved_assignee_task_count": missing_count,
        "warning": (
            "Some task assignee IDs are missing from users. Ranking only "
            "covers known employees/departments and may not be organization-wide."
            if missing_count else None
      
        ),
        "zero_task_groups_included": True,
    }


def _average_completion_time(
    conn: sqlite3.Connection, args: AnalyzeTasksInput
) -> dict[str, Any]:
    # Status is always Done. Dates represent calendar days, not hours.
    priority_sql = " AND t.priority = ?" if args.priority is not None else ""
    params: list[Any] = [args.priority.value] if args.priority is not None else []

    row = conn.execute(
        f"""
        SELECT COUNT(*) AS total_done,
               COUNT(CASE WHEN
                    date(t.create_time) IS NOT NULL
                    AND date(t.due_time) IS NOT NULL
                    AND julianday(t.due_time) >= julianday(t.create_time)
                    THEN 1 END) AS valid_count,
               AVG(CASE WHEN
                    date(t.create_time) IS NOT NULL
                    AND date(t.due_time) IS NOT NULL
                    AND julianday(t.due_time) >= julianday(t.create_time)
                    THEN julianday(t.due_time) - julianday(t.create_time)
                    END) AS avg_days
        FROM tasks AS t
        WHERE t.status = 'Done'{priority_sql}
        """,
        params,
    ).fetchone()

    total = int(row["total_done"])
    valid = int(row["valid_count"])
    return {
        "status": "success" if valid else "no_data",
        "analysis_type": args.analysis_type.value,
        "average_days": round(float(row["avg_days"]), 2) if valid else None,
        "unit": "calendar_days",
        "evaluated_task_count": valid,
        "total_done_tasks": total,
        "excluded_invalid_dates": total - valid,
        "calculation": "average(due_time - create_time) for Done tasks",
    }



def _overdue_count(conn: sqlite3.Connection, args: AnalyzeTasksInput) -> dict[str, Any]:
    today = _today_iso()
    filters, values = _task_filters(args)
    conditions = ["t.status != 'Done'", *filters]
    where = " AND ".join(conditions)

    row = conn.execute(
        f"""
        SELECT
          COUNT(CASE WHEN date(t.due_time) < date(?) THEN 1 END) AS overdue,
          COUNT(CASE WHEN date(t.due_time) IS NULL THEN 1 END) AS unknown_deadline
        FROM tasks AS t
        WHERE {where}
        """,
        [today, *values],
    ).fetchone()

    return {
        "status": "success",
        "analysis_type": args.analysis_type.value,
        "count": int(row["overdue"]),
        "as_of_date": today,
        "unassessable_missing_or_invalid_due_time": int(row["unknown_deadline"]),
        "rule": "status != Done AND due_time < as_of_date",
    }


@tool("analyze_tasks", args_schema=AnalyzeTasksInput)
def analyze_tasks(
    analysis_type: AnalysisType,
    group_by: GroupBy | None = None,
    sort_order: SortOrder | None = None,
    limit: int | None = None,
    status: TaskStatus | None = None,
    priority: TaskPriority | None = None,
) -> dict[str, Any]:
    """Analyze task workloads, completion durations, or overdue tasks.

    Use this tool when the user asks:
    - Which person or department has the most/fewest tasks? Use
      analysis_type='workload_ranking', group_by='assignee' or 'department',
      sort_order='desc' (most) or 'asc' (fewest), limit=1 by default.
      Includes zero-task groups and all ties at the cutoff.
    - What is the average time to close tasks? Use
      analysis_type='average_completion_time'. Average days between
      create_time and due_time for Done tasks with valid dates.
    - How many tasks are overdue? Use analysis_type='overdue_count'.
      Overdue means not Done and due_time earlier than today's date.

    Business rule: due_time means actual completion date for Done tasks,
    but scheduled deadline for tasks that are not Done. All names and
    departments come from the database; do not assume hardcoded values.
    This is a read-only tool. It does not list tasks or modify records.
    """
    try:
        args = AnalyzeTasksInput.model_validate({
            "analysis_type": analysis_type,
            "group_by": group_by,
            "sort_order": sort_order,
            "limit": limit,
            "status": status,
            "priority": priority,
        })

    except ValidationError as exc:
        logger.warning(
            "analyze_tasks input validation failed: %s",
            exc.errors()
        )
        raise ToolException(
            "Invalid analyze_tasks parameters."
        ) from exc

    else:
        logger.info(
            "analyze_tasks input validation succeeded | analysis_type=%s",
            args.analysis_type.value
        )

    try:
        with closing(_connect_readonly()) as conn:
            if args.analysis_type == AnalysisType.WORKLOAD_RANKING:
                result = _workload_ranking(conn, args)
            elif args.analysis_type == AnalysisType.AVERAGE_COMPLETION_TIME:
                result = _average_completion_time(conn, args)
            else:
                result = _overdue_count(conn, args)

    except sqlite3.Error as exc:
        logger.exception("Task analytics database query failed")
        raise ToolException("Could not retrieve task analytics from SQLite.") from exc

    applied_filters: dict[str, str] = {}
    if args.status is not None:
        applied_filters["status"] = args.status.value
    if args.priority is not None:
        applied_filters["priority"] = args.priority.value
    result["applied_filters"] = applied_filters

    logger.info(
            "analyze_tasks completed | analysis_type=%s | status=%s",
            args.analysis_type.value,
            result["status"]
        )
    return result