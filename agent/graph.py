
import os
import logging 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pathlib import Path
import json 
from tools.count_tools import count_tasks
from tools.search_tools import search_tasks
from tools.analytics_tools import analyze_tasks
from agent.prompts import build_system_message
from agent.state import AgentState
from agent.summary_node import * 
from langchain_core.messages import SystemMessage,HumanMessage, ToolMessage, AIMessage
from agent.write_routing import (
    ALL_TOOLS,
    READ_TOOLS,
    finish_approval_node,
    request_approval_node,
    route_after_agent,
    route_after_validation,
    validate_write_node,
)

logger = logging.getLogger("task_agent.agent")



def build_graph(*, checkpointer=None):
  
    load_dotenv()
    api_key = os.getenv("AVALAI_API_KEY")
    if not api_key:
        logger.error("Graph initialization failed: AVALAI_API_KEY is missing")
        raise RuntimeError("Missing AVALAI_API_KEY. Set it in .env.")

    llm = ChatOpenAI(
        model=os.getenv("AVALAI_MODEL", "gpt-4.1-nano"),
        api_key=api_key,
        base_url="https://api.avalai.ir/v1",
        timeout=45,
        max_retries=2,
    )
    llm_with_tools = llm.bind_tools(
        ALL_TOOLS,
        tool_choice="auto",
        parallel_tool_calls=False,
    )

    def agent_node(state: AgentState) -> dict:



        last_message = state["messages"][-1] if state["messages"] else None

        if isinstance(last_message, HumanMessage):
            logger.info("User input | %s", last_message.content)

        if isinstance(last_message, ToolMessage):
            logger.info(
                "Tool result | name=%s | content=%s",
                last_message.name or last_message.tool_call_id,
                str(last_message.content),
            )

        messages = [build_system_message()]

        summary = state.get("summary", "")

        if summary:
            messages.append(
                SystemMessage(
                    content=f"Previous conversation summary:\n{summary}"
                )
            )
        logger.info(
            "Agent prepared messages | count=%d | has_summary=%s",
            len(messages),
            bool(summary)
        )
        messages.extend(state["messages"])

        try:
             response = llm_with_tools.invoke(messages)

        except Exception:
            logger.exception("Agent LLM invocation failed")
            raise

        logger.info(
            "Agent LLM response received | tool_calls=%d | invalid_tool_calls=%d",
            len(response.tool_calls),
            len(response.invalid_tool_calls)
        )

        if response.tool_calls:
            for call in response.tool_calls:
                logger.info(
                    "Tool call | name=%s | args=%s",
                    call.get("name"),
                    call.get("args"),
                )

        if response.content and not response.tool_calls:
            logger.info("Final answer | %s", response.content)
        if response.invalid_tool_calls:
            logger.error(
                "Agent returned invalid tool calls | count=%d",
                len(response.invalid_tool_calls)
            )

            raise ValueError(
                "Model returned invalid tool-call arguments."
            )

        logger.info("Agent node completed successfully")

        return {"messages": [response]}

    def logged_route_after_agent(state: AgentState):
        try:
            route = route_after_agent(state)

            logger.info(
                "Routing after agent | destination=%s",
                route
            )
            return route

        except Exception:
            logger.exception("Routing after agent failed")
            raise


    def logged_route_after_validation(state: AgentState):
        try:
            route = route_after_validation(state)

            logger.info(
                "Routing after write validation | destination=%s",
                route
            )
            return route

        except Exception:
            logger.exception("Routing after write validation failed")
            raise

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("read_tools", ToolNode(READ_TOOLS, handle_tool_errors=True))
    builder.add_node("validate_write", validate_write_node)
    builder.add_node("request_approval", request_approval_node)
    builder.add_node("finish_approval", finish_approval_node)
    builder.add_node("summarize", create_summary_node(llm))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        logged_route_after_agent,
        {
            "read_tools": "read_tools",
            "validate_write": "validate_write",
            "summarize": "summarize",
        },
    )
    builder.add_edge("read_tools", "agent")
    builder.add_conditional_edges(
        "validate_write",
        logged_route_after_validation,
        {
            "request_approval": "request_approval",
            "summarize": "summarize",
        },
    )
    builder.add_edge("request_approval", "finish_approval")
    builder.add_edge("finish_approval", "agent")
    builder.add_edge("summarize", END)

    return builder.compile(checkpointer=checkpointer)