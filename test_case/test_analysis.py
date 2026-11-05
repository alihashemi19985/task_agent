"""Read-only smoke tests for analyze_tasks; do not modify tasks.db."""

from pprint import pprint

from pydantic import ValidationError
from analytics_tools import AnalyzeTasksInput, analyze_tasks



TEST_CASES = [
    (
        "Most tasks by employee",
        {"analysis_type": "workload_ranking", "group_by": "assignee", "sort_order": "desc"},
    ),
    (
        "Fewest tasks by employee (including zero)",
        {"analysis_type": "workload_ranking", "group_by": "assignee", "sort_order": "asc"},
    ),
    (
        "Most tasks by department",
        {"analysis_type": "workload_ranking", "group_by": "department", "sort_order": "desc"},
    ),
    (
        "Fewest tasks by department",
        {"analysis_type": "workload_ranking", "group_by": "department", "sort_order": "asc"},
    ),
    (
        "Most open/high-priority tasks by employee",
        {
            "analysis_type": "workload_ranking", "group_by": "assignee",
            "sort_order": "desc", "status": "Open", "priority": "High",
        },
    ),
    (
        "Average completion duration",
        {"analysis_type": "average_completion_time"},
    ),
    (
        "Overdue tasks",
        {"analysis_type": "overdue_count"},
    ),
    (
        "Overdue open tasks",
        {"analysis_type": "overdue_count", "status": "Open"},
    ),
]


def run():
    for title, arguments in TEST_CASES:
        result = analyze_tasks.invoke(arguments)
        print(f"\n--- {title} ---")
        pprint(result, sort_dicts=False)
        assert result["analysis_type"] == arguments["analysis_type"]
        assert result["status"] in {"success", "no_data", "partial_data"}
        if arguments["analysis_type"] == "workload_ranking":
            counts = [item["task_count"] for item in result["results"]]
            assert all(count >= 0 for count in counts)
            if counts:
                is_ascending = arguments["sort_order"] == "asc"
                assert counts == sorted(counts, reverse=not is_ascending)
        elif arguments["analysis_type"] == "overdue_count":
            assert result["count"] >= 0
        else:
            assert result["average_days"] is None or result["average_days"] >= 0

    # Input schema rejection: invalid combinations must not reach SQL.
    for invalid in [
        {"analysis_type": "workload_ranking"},
        {"analysis_type": "overdue_count", "status": "Done"},
        {"analysis_type": "average_completion_time", "sort_order": "asc"},
    ]:
        try:
            AnalyzeTasksInput.model_validate(invalid)
        except ValidationError:
            pass
        else:
            raise AssertionError(f"Unexpectedly accepted: {invalid}")

    print("\nAll analytics tests passed. tasks.db was not modified.")


if __name__ == "__main__":
    run()