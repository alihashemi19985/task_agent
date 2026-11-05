
import csv
import sqlite3
from datetime import datetime

DB_PATH = "tasks.db"
USERS_CSV = "users.csv"
TASKS_CSV = "tasks.csv"


def normalize_date(value):
    if not value or not value.strip():
        return None

    value = value.strip().replace("/", "-")
    return datetime.strptime(value, "%Y-%m-%d").date().isoformat()


def import_data():
    conn = sqlite3.connect(DB_PATH)

    try:
        conn.execute("PRAGMA foreign_keys = ON")

        # Create tables
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                fullname TEXT NOT NULL,
                department TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                create_time TEXT,
                status TEXT NOT NULL,
                assignee_id INTEGER,
                priority TEXT,
                due_time TEXT,
                FOREIGN KEY (assignee_id) REFERENCES users(id)
            );

            CREATE INDEX IF NOT EXISTS idx_tasks_status
                ON tasks(status);

            CREATE INDEX IF NOT EXISTS idx_tasks_assignee
                ON tasks(assignee_id);
        """)

        # Import users first
        with open(USERS_CSV, encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)

            for row in reader:
                conn.execute(
                    """
                    INSERT INTO users (id, fullname, department)
                    VALUES (?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        fullname = excluded.fullname,
                        department = excluded.department
                    """,
                    (
                        int(row["id"]),
                        row["fullname"].strip(),
                        row["department"].strip(),
                    ),
                )

        # Import tasks
        with open(TASKS_CSV, encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)

            for row in reader:
                assignee = row["assignee_id"].strip()

                conn.execute(
                    """
                    INSERT INTO tasks (
                        id, title, description, create_time,
                        status, assignee_id, priority, due_time
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        title = excluded.title,
                        description = excluded.description,
                        create_time = excluded.create_time,
                        status = excluded.status,
                        assignee_id = excluded.assignee_id,
                        priority = excluded.priority,
                        due_time = excluded.due_time
                    """,
                    (
                        int(row["id"]),
                        row["title"].strip(),
                        row["description"].strip(),
                        normalize_date(row["create_time"]),
                        row["status"].strip(),
                        int(assignee) if assignee else None,
                        row["priority"].strip(),
                        normalize_date(row["due_time"]),
                    ),
                )

        conn.commit()

        users_count = conn.execute(
            "SELECT COUNT(*) FROM users"
        ).fetchone()[0]

        tasks_count = conn.execute(
            "SELECT COUNT(*) FROM tasks"
        ).fetchone()[0]

        print("Database initialized successfully!")
        print(f"Users: {users_count}")
        print(f"Tasks: {tasks_count}")

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    import_data()
