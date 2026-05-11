import logging
import os
import re

from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

# from pydantic import BaseModel, Field

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

def _get_llm() -> ChatOllama:
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
    )

def route_node(state: AgentState) -> dict:

    question = state["question"]
    q_lower  = question.lower()

    if any(kw in q_lower for kw in _EXPOSURE_KEYWORDS):
        logger.info("Routed to 'exposure' via keyword match.")
        return {"route": "exposure"}

    llm    = _get_llm()
    prompt = ROUTER_PROMPT.format(question=question)
    try:
        response = llm.invoke(prompt)
        content  = _content_to_str(response.content).strip().lower()
        first_word = content.split()[0] if content.split() else "sql"
        route = "exposure" if first_word == "exposure" else "sql"
        logger.info("Routed to '%s' via LLM classification.", route)
    except Exception as exc:
        logger.warning("Router LLM failed (%s), defaulting to 'sql'.", exc)
        route = "sql"

    return {"route": route}

def sql_node(state: AgentState) -> dict:
    logger.info("Executing sql_node for: %s", state["question"])
    result = _run_sql(state["question"])
    return {
        "tool_result": result,
        "tool_used":   "sql_query_tool",
    }

def _resolve_portfolio_name(question: str) -> str:
    db = get_db()
    portfolios = db.execute_query("SELECT portfolio_name FROM portfolios")
    names      = [p["portfolio_name"] for p in portfolios]
    q_lower    = question.lower()

    for name in names:
        if name.lower() in q_lower:
            logger.info("Portfolio name resolved via exact match: %s", name)
            return name

    STOPWORDS = {"the", "and", "for", "with", "fund", "portfolio", "index",
                 "total", "from", "that", "this", "what", "show", "list",
                 "give", "sector", "are", "all"}
    best_name  = None
    best_score = 0

    for name in names:
        words   = name.lower().split()
        matches = sum(1 for w in words if w in q_lower)
        min_required = 1 if len(words) == 1 else 2
        if matches >= min_required and matches > best_score:
            best_score = matches
            best_name  = name

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

    llm    = _get_llm()
    prompt = PORTFOLIO_EXTRACTION_PROMPT.format(question=question)
    try:
        response       = llm.invoke(prompt)
        extracted_name = _content_to_str(response.content).strip()
        logger.info("Portfolio name extracted by LLM: %s", extracted_name)
        return extracted_name
    except Exception as exc:
        logger.warning("Portfolio name extraction failed (%s).", exc)
        return question


def exposure_node(state: AgentState) -> dict:
    portfolio_name = _resolve_portfolio_name(state["question"])
    logger.info("Executing exposure_node for portfolio: %s", portfolio_name)
    result = _run_exposure(portfolio_name)
    return {
        "tool_result": result,
        "tool_used":   "exposure_calculator_tool",
    }

def respond_node(state: AgentState) -> dict:
    tool_result = state.get("tool_result", "")

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
        answer = tool_result

    return {
        "answer":   answer,
        "messages": [AIMessage(content=answer)],
    }

def _pick_tool(state: AgentState) -> str:
    return state.get("route", "sql")

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


class PortfolioAgent:

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
