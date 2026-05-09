"""
agent/agent.py
──────────────
Portfolio agent – 3-stage deterministic pipeline.

WHY the old bind_tools() ReAct architecture was replaced
─────────────────────────────────────────────────────────
bind_tools() requires the LLM to emit structured JSON tool-call messages.
Local models like llama3.1:8b do this unreliably — they often narrate
("I will use sql_query_tool...") instead of emitting the required JSON,
causing the graph loop to exit without ever executing a tool.

New graph topology
──────────────────
  START
    │
    ▼
  route_node          ← classifies question: "sql" or "exposure"
    │                   (keyword match first, LLM fallback)
    ├─── "sql" ──────► sql_node        ← calls _run_sql() directly
    │                      │
    └─── "exposure" ─► exposure_node   ← extracts portfolio name, calls _run_exposure()
                           │
                    (both converge)
                           ▼
                     respond_node      ← LLM formats raw data into natural answer
                           │
                          END

The LLM is now only used for:
  1. Routing (one-word output: "sql" or "exposure")
  2. Portfolio name extraction (one-line output)
  3. Response formatting (natural language from raw data)

None of these require JSON tool-call emission, making the pipeline
reliable with local models.
"""

import logging
import os
import re

from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from agent.prompts import (
    PORTFOLIO_EXTRACTION_PROMPT,
    RESPONDER_PROMPT,
    ROUTER_PROMPT,
)
from agent.state import AgentState
from db.database import get_db
from tools.exposure_calculator import _run_exposure
from tools.sql_tool import _content_to_str, _run_sql

logger = logging.getLogger(__name__)

# ── Keywords that deterministically route to the exposure tool ────────────────
_EXPOSURE_KEYWORDS = [
    "sector exposure",
    "sector breakdown",
    "sector weight",
    "sector allocation",
    "sector composition",
    "sector distribution",
    "sector percentage",
    "exposure for",
    "exposure breakdown",
    "allocation breakdown",
]


# ── Shared LLM factory (no bind_tools – plain ChatOllama) ─────────────────────
def _get_llm() -> ChatOllama:
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – Route node
# ─────────────────────────────────────────────────────────────────────────────

def route_node(state: AgentState) -> dict:
    """
    Classify the question as 'sql' or 'exposure'.

    Strategy (in priority order):
      1. Keyword match – instant, deterministic, no LLM call needed.
      2. LLM classification – for edge cases the keywords don't cover.
         The LLM is asked for a single word output, minimising hallucination risk.
    """
    question = state["question"]
    q_lower  = question.lower()

    # Priority 1: keyword match
    if any(kw in q_lower for kw in _EXPOSURE_KEYWORDS):
        logger.info("Routed to 'exposure' via keyword match.")
        return {"route": "exposure"}

    # Priority 2: LLM classification
    llm    = _get_llm()
    prompt = ROUTER_PROMPT.format(question=question)
    try:
        response = llm.invoke(prompt)
        content  = _content_to_str(response.content).strip().lower()
        # Accept the first word to guard against verbose responses
        first_word = content.split()[0] if content.split() else "sql"
        route = "exposure" if first_word == "exposure" else "sql"
        logger.info("Routed to '%s' via LLM classification.", route)
    except Exception as exc:
        logger.warning("Router LLM failed (%s), defaulting to 'sql'.", exc)
        route = "sql"

    return {"route": route}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2a – SQL execute node
# ─────────────────────────────────────────────────────────────────────────────

def sql_node(state: AgentState) -> dict:
    """
    Call _run_sql() directly with the user's question.
    No tool-calling API involved – plain Python function call.
    """
    logger.info("Executing sql_node for: %s", state["question"])
    result = _run_sql(state["question"])
    return {
        "tool_result": result,
        "tool_used":   "sql_query_tool",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2b – Exposure execute node
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_portfolio_name(question: str) -> str:
    """
    Extract the portfolio name from the question.

    Strategy (in priority order):
      1. Exact substring match (e.g. question contains full portfolio name).
      2. Best-score word match — count how many words of each portfolio name
         appear in the question, pick the highest score. This prevents
         single-word false matches (e.g. "equity" matching "Growth Equity Fund"
         instead of "International Equity Fund" when "international" is present).
      3. LLM extraction as final fallback.
    """
    db = get_db()
    portfolios = db.execute_query("SELECT portfolio_name FROM portfolios")
    names      = [p["portfolio_name"] for p in portfolios]
    q_lower    = question.lower()

    # 1. Exact substring match (case-insensitive)
    for name in names:
        if name.lower() in q_lower:
            logger.info("Portfolio name resolved via exact match: %s", name)
            return name

    # 2. Best-score word match
    #    Score = number of portfolio-name words found in the question.
    STOPWORDS = {"the", "and", "for", "with", "fund", "portfolio", "index",
                 "total", "from", "that", "this", "what", "show", "list",
                 "give", "sector", "are", "all"}
    best_name  = None
    best_score = 0

    for name in names:
        words   = name.lower().split()
        matches = sum(1 for w in words if w in q_lower)
        # Require at least 2 matches OR the name is a single word
        min_required = 1 if len(words) == 1 else 2
        if matches >= min_required and matches > best_score:
            best_score = matches
            best_name  = name

    # Special case: single distinctive keyword that uniquely identifies one portfolio
    # e.g. "ESG" in question uniquely matches "ESG Sustainable Fund"
    if not best_name:
        q_words = [w for w in re.findall(r"[a-z0-9]+", q_lower)
                   if len(w) >= 3 and w not in STOPWORDS]
        for q_word in q_words:
            matched = [n for n in names if q_word in n.lower().split()]
            if len(matched) == 1:
                best_name = matched[0]
                logger.info(
                    "Portfolio resolved via unique keyword '%s': %s", q_word, best_name
                )
                break

    if best_name:
        logger.info(
            "Portfolio name resolved via word-score match (%d words): %s",
            best_score, best_name,
        )
        return best_name

    # 3. LLM extraction fallback
    llm    = _get_llm()
    prompt = PORTFOLIO_EXTRACTION_PROMPT.format(question=question)
    try:
        response       = llm.invoke(prompt)
        extracted_name = _content_to_str(response.content).strip()
        logger.info("Portfolio name extracted by LLM: %s", extracted_name)
        return extracted_name
    except Exception as exc:
        logger.warning("Portfolio name extraction failed (%s).", exc)
        return question   # last resort: pass the whole question, let the tool handle it


def exposure_node(state: AgentState) -> dict:
    """
    Resolve the portfolio name then call _run_exposure() directly.
    No tool-calling API involved – plain Python function call.
    """
    portfolio_name = _resolve_portfolio_name(state["question"])
    logger.info("Executing exposure_node for portfolio: %s", portfolio_name)
    result = _run_exposure(portfolio_name)
    return {
        "tool_result": result,
        "tool_used":   "exposure_calculator_tool",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 – Respond node
# ─────────────────────────────────────────────────────────────────────────────

def respond_node(state: AgentState) -> dict:
    """
    Use the LLM to turn raw tool output into a concise, professional answer.

    If the tool returned an error, pass it through without LLM formatting
    to avoid the model hallucinating a 'successful' answer over an error message.
    """
    tool_result = state.get("tool_result", "")

    # Pass errors through directly — don't let LLM mask them
    if tool_result.startswith("Error"):
        return {
            "answer":   tool_result,
            "messages": [AIMessage(content=tool_result)],
        }

    llm    = _get_llm()
    prompt = RESPONDER_PROMPT.format(
        question    = state["question"],
        tool_result = tool_result,
    )
    try:
        response = llm.invoke(prompt)
        answer   = _content_to_str(response.content).strip()
    except Exception as exc:
        logger.error("Responder LLM failed: %s", exc)
        # Fallback: return raw tool result so the user still gets data
        answer = tool_result

    return {
        "answer":   answer,
        "messages": [AIMessage(content=answer)],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routing helper for conditional edge
# ─────────────────────────────────────────────────────────────────────────────

def _pick_tool(state: AgentState) -> str:
    """Read the route set by route_node and dispatch to the correct node."""
    return state.get("route", "sql")


# ─────────────────────────────────────────────────────────────────────────────
# Graph compilation
# ─────────────────────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("route",    route_node)
    graph.add_node("sql",      sql_node)
    graph.add_node("exposure", exposure_node)
    graph.add_node("respond",  respond_node)

    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        _pick_tool,
        {"sql": "sql", "exposure": "exposure"},
    )
    graph.add_edge("sql",      "respond")
    graph.add_edge("exposure", "respond")
    graph.add_edge("respond",  END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

class PortfolioAgent:
    """
    High-level wrapper around the compiled LangGraph pipeline.

    Usage:
        agent = PortfolioAgent()
        print(agent.answer_question("How many portfolios do we have?"))
    """

    def __init__(self):
        self.graph = build_graph()

    def answer_question(self, question: str) -> str:
        initial_state: AgentState = {
            "messages":    [HumanMessage(content=question)],
            "question":    question,
            "answer":      "",
            "tool_used":   "none",
            "route":       "",
            "tool_result": "",
        }
        try:
            final_state = self.graph.invoke(initial_state)
            answer = final_state.get("answer", "").strip()
            logger.info(
                "Answered via '%s' (tool: %s)",
                final_state.get("route", "?"),
                final_state.get("tool_used", "?"),
            )
            return answer or "I was unable to produce an answer."
        except Exception as exc:
            logger.error("Agent error for '%s': %s", question, exc)
            return f"Error: {exc}"
