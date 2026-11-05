
import sqlite3
import tempfile
from pathlib import Path
from pprint import pprint

import operation_tools as op
from tools import DB_PATH


def test_operation_tools():

    if not DB_PATH.is_file():
        raise FileNotFoundError("tasks.db not found!")

    # Create a temporary test database
    with tempfile.TemporaryDirectory() as tmp_dir:

        test_db = Path(tmp_dir) / "tasks.db"

        # Make a safe SQLite backup
        with sqlite3.connect(DB_PATH) as source:
            with sqlite3.connect(test_db) as target:
                source.backup(target)

        # Redirect operation tools to test database
        op.DB_PATH = test_db

        with sqlite3.connect(test_db) as conn:
            user = conn.execute(
                "SELECT id, fullname FROM users LIMIT 1"
            ).fetchone()

            initial_count = conn.execute(
                "SELECT COUNT(*) FROM tasks"
            ).fetchone()[0]

        assert user is not None, "No users found!"

        user_id, user_name = user

        # ---------------------------------------
        # TEST 1: Create Task
        # ---------------------------------------

        print("\n--- TEST 1: CREATE TASK ---")

        result = op.create_task.invoke({
            "title": "بررسی عملکرد سیستم",
            "assignee_name": user_name,
            "description": "تست ایجاد تسک جدید",
            "priority": "High",
        })

        pprint(result)

        assert result["status"] == "created"
        assert result["task"]["assignee_id"] == user_id
        assert result["task"]["status"] == "Open"

        new_task_id = result["task"]["id"]

        print("CREATE TASK PASSED")

        # ---------------------------------------
        # TEST 2: Update Task Status
        # ---------------------------------------

        print("\n--- TEST 2: UPDATE TO DONE ---")

        result = op.update_task_status.invoke({
            "task_id": new_task_id,
            "new_status": "Done",
        })

        pprint(result)

        assert result["status"] == "updated"
        assert result["task"]["status"] == "Done"
        assert result["task"]["due_time"] == op._today_iso()

        print("UPDATE STATUS PASSED")

        # ---------------------------------------
        # TEST 3: Same Status
        # ---------------------------------------

        print("\n--- TEST 3: NO CHANGE ---")

        result = op.update_task_status.invoke({
            "task_id": new_task_id,
            "new_status": "Done",
        })

        pprint(result)

        assert result["status"] == "no_change"

        print("NO CHANGE PASSED")

        # ---------------------------------------
        # TEST 4: Reopen Task
        # ---------------------------------------

        print("\n--- TEST 4: REOPEN TASK ---")

        result = op.update_task_status.invoke({
            "task_id": new_task_id,
            "new_status": "Open",
        })

        pprint(result)

        assert result["status"] == "updated"
        assert result["task"]["status"] == "Open"
        assert result["task"]["due_time"] == ""

        print("REOPEN TASK PASSED")

        # ---------------------------------------
        # TEST 5: Invalid Task ID
        # ---------------------------------------

        print("\n--- TEST 5: TASK NOT FOUND ---")

        result = op.update_task_status.invoke({
            "task_id": 99999999,
            "new_status": "Done",
        })

        pprint(result)

        assert result["status"] == "task_not_found"

        print("TASK NOT FOUND PASSED")

        # ---------------------------------------
        # TEST 6: Invalid Employee
        # ---------------------------------------

        print("\n--- TEST 6: PERSON NOT FOUND ---")

        result = op.create_task.invoke({
            "title": "تسک آزمایشی",
            "assignee_name": "کاربر ناموجود آزمایشی",
        })

        pprint(result)

        assert result["status"] == "person_not_found"

        print("PERSON NOT FOUND PASSED")

        # ---------------------------------------
        # TEST 7: Verify Database Changes
        # ---------------------------------------

        print("\n--- TEST 7: DATABASE VALIDATION ---")

        with sqlite3.connect(test_db) as conn:
            final_count = conn.execute(
                "SELECT COUNT(*) FROM tasks"
            ).fetchone()[0]

        assert final_count == initial_count + 1

        print("DATABASE VALIDATION PASSED")

    print("\n" + "=" * 45)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("Original tasks.db was NOT modified.")
    print("=" * 45)


if __name__ == "__main__":
    test_operation_tools()
