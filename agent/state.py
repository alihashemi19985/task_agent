"""Graph state for milestone 1: read-only tool calling."""

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    summary: str
    pending_write: dict | None
    approved: bool