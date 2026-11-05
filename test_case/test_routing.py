"""Fast offline safety tests; they do not call AvalAI or modify tasks.db.
Run: python test_write_routing.py
"""

import json
from langchain_core.messages import AIMessage

from agent.write_routing import route_after_agent, write_preview_node


def state_with_calls(*calls):
    return {"messages": [AIMessage(content="", tool_calls=list(calls))]}


def call(name, args, id_):
    return {"name": name, "args": args, "id": id_, "type": "tool_call"}


def run_tests():
    assert route_after_agent({"messages": [AIMessage(content="سلام")]}) == "summarize"
    assert route_after_agent(state_with_calls(call("count_tasks", {}, "r1"))) == "read_tools"
    assert route_after_agent(state_with_calls(call("analyze_tasks", {"analysis_type": "overdue_count"}, "r2"))) == "read_tools"

    task_args = {"title": "بررسی API", "assignee_name": "علی صابری"}
    create_state = state_with_calls(call("create_task", task_args, "w1"))
    assert route_after_agent(create_state) == "write_preview"

    result = write_preview_node(create_state)
    payload = json.loads(result["messages"][0].content)
    assert payload["status"] == "not_executed"
    assert payload["tool"] == "create_task"
    assert "هیچ تغییری" in result["messages"][-1].content

    update_state = state_with_calls(
        call("update_task_status", {"task_id": 120, "new_status": "Done"}, "w2")
    )
    assert route_after_agent(update_state) == "write_preview"
    assert json.loads(write_preview_node(update_state)["messages"][0].content)["status"] == "not_executed"

    mixed = state_with_calls(
        call("count_tasks", {}, "r3"),
        call("create_task", task_args, "w3"),
    )
    assert route_after_agent(mixed) == "write_preview"
    result = write_preview_node(mixed)
    assert len(result["messages"]) == 3  # 2 ToolMessages + 1 AIMessage
    assert json.loads(result["messages"][0].content)["reason"] == "mixed_tool_calls_blocked_for_safety"

    invalid = state_with_calls(call("create_task", {"assignee_name": "علی صابری"}, "w4"))
    assert json.loads(write_preview_node(invalid)["messages"][0].content)["status"] == "invalid_arguments"

    print("ALL ROUTING SAFETY TESTS PASSED. NO DATABASE WRITES EXECUTED.")


if __name__ == "__main__":
    run_tests()