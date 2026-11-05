
from tools import count_tasks




from tools import count_tasks

result = count_tasks.invoke({
    "status": "Done",
    "last_n_months": 10
})

print(result)


def test_count_tasks():
    test_cases = [
        ("All tasks", {}),
        ("Open tasks", {"status": "Open"}),
        ("Done tasks", {"status": "Done"}),
        ("High priority", {"priority": "High"}),
        (
            "Open and High",
            {"status": "Open", "priority": "High"},
        ),
        (
            "Completed in last month",
            {
                "status": "Done",
                "due_time_from": "2026-09-30",
                "due_time_to": "2026-10-31",
            },
        ),
    ]

    for name, filters in test_cases:
        result = count_tasks.invoke(filters)

        print(f"\nTest: {name}")
        print(f"Result: {result}")


if __name__ == "__main__":
    test_count_tasks()
