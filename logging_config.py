
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"

LOG_NAMES = [
    "agent",
    "count_tasks",
    "search_tasks",
    "analyze_tasks",
    "create_task",
    "update_task_status",
]


def setup_logging():

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    )

    for name in LOG_NAMES:

        logger = logging.getLogger(
            f"task_agent.{name}"
        )

        logger.setLevel(logging.INFO)
        logger.propagate = False

        # Prevent duplicate handlers
        if logger.handlers:
            continue

        file_handler = RotatingFileHandler(
            filename=LOG_DIR / f"{name}.log",
            maxBytes=2 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )

        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
