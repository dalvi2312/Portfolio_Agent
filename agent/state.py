"""
agent/state.py
──────────────
LangGraph shared state for the portfolio agent.

Fields added beyond the original:
  route       – "sql" or "exposure", set by route_node
  tool_result – raw string output from whichever tool was executed
"""

from typing import Annotated
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    messages:    Annotated[list, add_messages]
    question:    str
    answer:      str
    tool_used:   str
    route:       str   # "sql" | "exposure"
    tool_result: str   # raw output from the executed tool
