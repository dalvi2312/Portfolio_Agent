"""
tools/sql_tool.py
─────────────────
Text-to-SQL — LangChain StructuredTool.

Uses explicit name/description/args_schema so the tool definition never
depends on docstring parsing (which is brittle across LangChain versions).

Bug fix included: _content_to_str() normalises response.content which can
be either a str or a list of content-block dicts depending on the model.
"""

import logging
import os
import re

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama
from db.database import get_db

logger = logging.getLogger(__name__)

_llm: ChatOllama | None = None


def _get_llm() -> ChatOllama:
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0,
        )
    return _llm


# Pydantic input schema
class SQLInput(BaseModel):
    question: str = Field(description="A natural-language question about portfolio data.")


def _content_to_str(content) -> str:
    """Normalise LangChain response.content — may be str or list of dicts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
            else:
                parts.append(str(block))
        return "\n".join(p for p in parts if p)
    return str(content)


def _extract_sql(raw_content) -> str:
    text = _content_to_str(raw_content)
    fence = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    inline = re.search(r"((?:SELECT|WITH)\s+.+)", text, re.DOTALL | re.IGNORECASE)
    if inline:
        return inline.group(1).strip()
    return text.strip()


def _format_results(rows: list[dict]) -> str:
    if not rows:
        return "Query returned no results."
    if len(rows) == 1 and len(rows[0]) == 1:
        return str(list(rows[0].values())[0])
    columns = list(rows[0].keys())
    header  = " | ".join(columns)
    lines   = [header, "-" * len(header)]
    for row in rows:
        lines.append(" | ".join(str(v) if v is not None else "NULL" for v in row.values()))
    return "\n".join(lines)


def _run_sql(question: str) -> str:
    db  = get_db()
    llm = _get_llm()

    prompt = f"""You are an expert SQLite query writer for a portfolio management database.

{db.get_schema_summary()}

Rules:
- Write ONLY a SELECT statement. No INSERT, UPDATE, DELETE, DROP.
- Use table aliases in JOINs.
- Use ROUND(..., 2) for percentages.
- Output ONLY the SQL inside a ```sql ... ``` block. No explanation.

Question: {question}

SQL Query:"""

    try:
        response = llm.invoke(prompt)
        sql      = _extract_sql(response.content)
        logger.info("Generated SQL: %s", sql)
        rows = db.execute_query(sql)
        logger.info("SQL returned %d row(s).", len(rows))
        return _format_results(rows)
    except ValueError as exc:
        logger.error("SQL safety check failed: %s", exc)
        return f"Error: {exc}"
    except Exception as exc:
        logger.error("SQL tool error: %s", exc)
        return f"Error executing query: {exc}"


# StructuredTool — explicit name/description, no docstring parsing
sql_query_tool = StructuredTool.from_function(
    func=_run_sql,
    name="sql_query_tool",
    description=(
        "Answer questions about portfolio data by generating and executing a SQL query. "
        "Use for: portfolio counts/names, holdings, securities, transactions, "
        "risk metrics, performance, AUM figures, sector membership, prices, benchmarks. "
        "Any question answerable from the relational database."
    ),
    args_schema=SQLInput,
)
