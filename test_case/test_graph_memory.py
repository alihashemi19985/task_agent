from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage
from agent.graph import build_graph

with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    graph = build_graph(checkpointer=checkpointer)

    config = {
        "configurable": {
            "thread_id": "test-memory-1"
        }
    }

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content="چند تسک باز داریم؟")
            ]
        },
        config=config,
    )