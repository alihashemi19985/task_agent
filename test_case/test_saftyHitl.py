"""Offline test of approval interrupts without AvalAI and without database writes.
Run: python test_hitl_safety.py
"""

import hashlib
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agent.state import AgentState
from agent.write_routing import (
    finish_approval_node,
    request_approval_node,
    route_after_validation,
    validate_write_node,
)


def make_graph():
    builder = StateGraph(AgentState)
    builder.add_node("validate_write", validate_write_node)
    builder.add_node("request_approval", request_approval_node)
    builder.add_node("finish_approval", finish_approval_node)
    builder.add_edge(START, "validate_write")
    builder.add_conditional_edges(
        "validate_write",
        route_after_validation,
        {"request_approval": "request_approval", "summarize": END},
    )
    builder.add_edge("request_approval", "finish_approval")
    builder.add_edge("finish_approval", END)
    return builder.compile(checkpointer=InMemorySaver())


def call(name, args, id_):
    return {"name": name, "args": args, "id": id_, "type": "tool_call"}


def start(graph, thread_id, *calls):
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {"messages": [AIMessage(content="", tool_calls=list(calls))]},
        config=config,
    )
    return result, config


def latest_status(result):
    import json
    messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    return json.loads(messages[-1].content)["status"]


def run_tests():
    root = Path(__file__).resolve().parent
    original_db = root / "tasks.db"
    before = hashlib.sha256(original_db.read_bytes()).hexdigest() if original_db.exists() else None
    graph = make_graph()

    result, config = start(
        graph, "offline-approve",
        call("create_task", {"title": "بررسی API", "assignee_name": "علی صابری"}, "w1"),
    )
    assert result.get("__interrupt__"), "Approval must interrupt before execution"
    assert result["__interrupt__"][0].value["tool"] == "create_task"
    assert graph.get_state(config).values["pending_write"]["name"] == "create_task"
    approved = graph.invoke(Command(resume={"approved": True}), config=config)
    assert latest_status(approved) == "approved_not_executed"
    assert graph.get_state(config).values.get("pending_write") is None

    result, config = start(
        graph, "offline-reject",
        call("update_task_status", {"task_id": 120, "new_status": "Done"}, "w2"),
    )
    assert result.get("__interrupt__")
    rejected = graph.invoke(Command(resume={"approved": False}), config=config)
    assert latest_status(rejected) == "cancelled"

    result, config = start(
        graph, "offline-no-truthy-string",
        call("create_task", {"title": "بررسی API", "assignee_name": "علی صابری"}, "w3"),
    )
    assert result.get("__interrupt__")
    rejected = graph.invoke(Command(resume={"approved": "yes"}), config=config)
    assert latest_status(rejected) == "cancelled", "String 'yes' must not authorize a write"

    result, _ = start(
        graph, "offline-invalid",
        call("create_task", {"assignee_name": "علی صابری"}, "w4"),
    )
    assert not result.get("__interrupt__")
    assert latest_status(result) == "invalid_arguments"

    result, _ = start(
        graph, "offline-mixed",
        call("count_tasks", {}, "r1"),
        call("create_task", {"title": "بررسی API", "assignee_name": "علی صابری"}, "w5"),
    )
    assert not result.get("__interrupt__")
    assert latest_status(result) == "blocked"

    after = hashlib.sha256(original_db.read_bytes()).hexdigest() if before else None
    assert before == after, "The original tasks.db must never be modified"
    print("ALL HITL APPROVAL TESTS PASSED. NO DATABASE WRITES EXECUTED.")


if __name__ == "__main__":
    run_tests()