from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    RemoveMessage,
)

KEEP_RECENT_TURNS = 3


def create_summary_node(llm):

    def summarize_node(state):

        messages = state["messages"]

        human_indices = [
            i for i, msg in enumerate(messages)
            if isinstance(msg, HumanMessage)
        ]

        if len(human_indices) <= KEEP_RECENT_TURNS:
            return {}

        split_index = human_indices[-KEEP_RECENT_TURNS]
        old_messages = messages[:split_index]

        previous_summary = state.get("summary", "")

        summary_prompt = SystemMessage(
            content=f"""
You are a conversation memory summarizer.

Previous summary:
{previous_summary or "None"}

Update the conversation summary using the
provided older messages.

Preserve:
- Important conversation context
- Active task filters
- Task IDs and project IDs
- Important decisions
- Unresolved questions

Treat tool results as historical observations.
Do not invent information.
Keep the summary concise.
"""
        )

        response = llm.invoke(
            [summary_prompt, *old_messages]
        )

        return {
            "summary": response.content,
            "messages": [
                RemoveMessage(id=msg.id)
                for msg in old_messages
            ],
        }

    return summarize_node