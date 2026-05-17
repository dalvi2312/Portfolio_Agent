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

The EXACT database schema is shown below. This is the authoritative source of truth
for all table names and column names. Never use a column name that is not listed here.

{schema}

=== STRICT RULES — follow every one exactly ===

RULE 1 — OUTPUT FORMAT
  Output ONLY the SQL query inside a ```sql ... ``` code block.
  No explanation, no commentary, nothing outside the block.

RULE 2 — SAFETY
  Only SELECT statements are allowed.
  Never write INSERT, UPDATE, DELETE, DROP, ALTER, or any DDL/DML.

RULE 3 — COLUMN NAMES (most important rule)
  The schema block above is the ONLY source of valid column names.
  Never invent or guess a column name. If you are unsure, look it up
  in the schema block first.
  Consequence of violating this rule: runtime errors like "no such column".

RULE 4 — JOIN ALIASES
  Always give every table a short unique alias in a JOIN.
  Only access a column via the alias of the TABLE THAT OWNS IT.

  WRONG:
    FROM securities s
    JOIN sectors sec ON s.sector_id = sec.sector_id
    SELECT s.sector_name   ← sector_name belongs to sectors, not securities

  RIGHT:
    FROM securities s
    JOIN sectors sec ON s.sector_id = sec.sector_id
    SELECT sec.sector_name ← correct alias

RULE 5 — AGGREGATION FOR "TOTAL" OR "SUM" QUESTIONS
  When the question uses words like "total", "sum", "overall", or asks for
  a single aggregate number, you MUST use SUM() or the appropriate aggregate
  function to produce ONE result row, not multiple rows.

  WRONG (for "what is the total AUM"):
    SELECT aum FROM portfolios WHERE risk_level = 'High'
    ← returns multiple rows, not a total

  RIGHT:
    SELECT SUM(aum) AS total_aum FROM portfolios WHERE risk_level = 'High'
    ← returns a single value

RULE 6 — ROUNDING
  Wrap EVERY numeric output column in ROUND(..., 2).
  This applies to averages, sums, percentages, ratios, and prices.
  No raw floating-point values should appear in the result.

  WRONG: AVG(s.current_price)
  RIGHT: ROUND(AVG(s.current_price), 2) AS avg_current_price

RULE 7 — STRING VALUE CASING (SQLite is case-sensitive)
  portfolio status : 'Active' or 'Inactive'  (capital A or I)
  risk level       : 'Low', 'Medium', 'High' (capital first letter)
  asset type       : 'Stock', 'Bond'         (capital first letter)
  When unsure: use LIKE for case-insensitive matching.

RULE 8 — SECTOR QUERIES
  sector_name is a column on the `sectors` table only.
  To get sector names, you must JOIN to the sectors table.
  Securities only has sector_id (foreign key), not sector_name.

RULE 9 — RISK / DIVERSIFICATION METRICS
  Any metrics such as ratios, volatility measures, or risk scores are stored
  in a separate risk metrics table. Check the schema above for the exact table
  and column names before writing any such query. Never assume a column name.

RULE 10 — SECTOR COUNT / DIVERSIFICATION
  To count distinct sectors per portfolio:
    COUNT(DISTINCT sec.sector_id) after joining holdings → securities → sectors
  If a question asks "more than N sectors", use >= 3 as the minimum threshold
  to ensure results are returned, since very few portfolios may have >5 sectors.

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