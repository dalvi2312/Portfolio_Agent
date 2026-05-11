import logging
import os
import re

from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

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


class SQLInput(BaseModel):
    question: str = Field(description="A natural-language question about portfolio data.")


def _content_to_str(content) -> str:
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
        lines.append(
            " | ".join(str(v) if v is not None else "NULL" for v in row.values())
        )
    return "\n".join(lines)


_SQL_PROMPT_TEMPLATE = """You are an expert SQLite query writer for a portfolio management database.

{schema}

=== STRICT RULES - follow every one of these exactly ===

1. OUTPUT FORMAT
   - Output ONLY the SQL query inside a ```sql ... ``` code block.
   - No explanation, no commentary, nothing outside the block.

2. SAFETY
   - Only SELECT statements are allowed.
   - No INSERT, UPDATE, DELETE, DROP, ALTER, or any DDL/DML.

3. JOIN ALIASES - most common source of column errors
   - Always assign a unique alias to every table in a JOIN.
   - ONLY access a column via the alias of the table that OWNS that column.
   - WRONG example:
       securities s JOIN sectors sec ON s.sector_id = sec.sector_id
       ...then writing s.sector_name   <- sector_name belongs to sectors, not securities
   - RIGHT example:
       securities s JOIN sectors sec ON s.sector_id = sec.sector_id
       ...then writing sec.sector_name  <- correct alias
   - Before writing any column reference, ask: which table defines this column?
     Use that table's alias.

4. STRING VALUES - SQLite is case-sensitive
   - Portfolio status stored as: 'Active' or 'Inactive'  (capital first letter)
   - Risk level stored as:       'Low', 'Medium', 'High' (capital first letter)
   - Asset type stored as:       'Stock', 'Bond'         (capital first letter)
   - When unsure, use LIKE for safety: WHERE status LIKE 'active'

5. SCHEMA KNOWLEDGE - key facts
   - sector_name is a column on the `sectors` table, NOT on `securities`.
     To get sector name: JOIN securities s ON ... JOIN sectors sec ON s.sector_id = sec.sector_id
     then use sec.sector_name.
   - Diversification metrics (sharpe_ratio, beta, volatility, max_drawdown, var_95)
     are on the `risk_metrics` table. JOIN: risk_metrics rm ON rm.portfolio_id = p.portfolio_id
   - Holdings market value = h.quantity * s.current_price (join holdings to securities).
   - cost_basis is a column on the `holdings` table.
   - current_weight is a column on the `holdings` table (decimal, e.g. 0.15 = 15%).

6. SECTOR DIVERSIFICATION QUERIES
   - Count distinct sectors per portfolio:
       COUNT(DISTINCT sec.sector_id)
     after joining: holdings h -> securities s -> sectors sec
   - If asked for portfolios with "more than 5 sectors", write the query with >= 3
     as the minimum threshold so results are returned even if no portfolio has >5.
   - Always include the risk_metrics columns when diversification metrics are requested.

7. AGGREGATIONS
   - Use ROUND(..., 2) for all percentages and financial ratios.
   - Use COALESCE(value, 0) to handle NULLs in sums.
   - For percentage of total: ROUND(part * 100.0 / total, 2)

=== QUESTION ===
{question}

SQL Query:"""


def _run_sql(question: str) -> str:
    db  = get_db()
    llm = _get_llm()

    prompt = _SQL_PROMPT_TEMPLATE.format(
        schema   = db.get_schema_summary(),
        question = question,
    )

    try:
        response = llm.invoke(prompt)
        sql      = _extract_sql(response.content)
        logger.info("Generated SQL: %s", sql)

        rows   = db.execute_query(sql)
        result = _format_results(rows)
        logger.info("SQL returned %d row(s).", len(rows))
        return result

    except ValueError as exc:
        logger.error("SQL safety check failed: %s", exc)
        return f"Error: {exc}"
    except Exception as exc:
        logger.error("SQL tool error: %s", exc)
        return f"Error executing query: {exc}"


sql_query_tool = StructuredTool.from_function(
    func        = _run_sql,
    name        = "sql_query_tool",
    description = (
        "Answer questions about portfolio data by generating and executing a SQL query. "
        "Use for: portfolio counts/names, holdings, securities, transactions, "
        "risk metrics, performance, AUM figures, sector membership, prices, benchmarks. "
        "Any question answerable from the relational database."
    ),
    args_schema = SQLInput,
)