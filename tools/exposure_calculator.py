"""
tools/exposure_calculator.py
────────────────────────────
Sector Exposure Calculator — LangChain StructuredTool.

Uses explicit name/description/args_schema so the tool definition never
depends on docstring parsing (which is brittle across LangChain versions).

Logic:
  - Fetch all equity (asset_type = 'Stock') holdings for the portfolio.
  - Ignore bonds entirely.
  - Sum current_weight per sector.
  - Normalise to 100% on an equity-only basis.
"""

import logging
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from db.database import get_db

logger = logging.getLogger(__name__)


# Pydantic input schema
class ExposureInput(BaseModel):
    portfolio_name: str = Field(
        description="The exact or approximate name of the portfolio."
    )


def _compute_sector_exposures(portfolio_name: str) -> dict:
    db = get_db()

    # Resolve portfolio
    rows = db.execute_query(
        "SELECT portfolio_id, portfolio_name FROM portfolios WHERE portfolio_name = ?",
        (portfolio_name,),
    )
    if not rows:
        all_p = db.execute_query("SELECT portfolio_id, portfolio_name FROM portfolios")
        match = next(
            (p for p in all_p if portfolio_name.lower() in p["portfolio_name"].lower()),
            None,
        )
        if not match:
            names = [p["portfolio_name"] for p in all_p]
            raise ValueError(f"Portfolio '{portfolio_name}' not found. Available: {names}")
        rows = [match]

    portfolio_id  = rows[0]["portfolio_id"]
    resolved_name = rows[0]["portfolio_name"]

    # Equity holdings with sector info
    equity = db.execute_query(
        """
        SELECT s.symbol, sec.sector_name, h.current_weight
        FROM   holdings h
        JOIN   securities s   ON h.security_id = s.security_id
        JOIN   sectors    sec ON s.sector_id    = sec.sector_id
        WHERE  h.portfolio_id = ?
          AND  s.asset_type   = 'Stock'
        """,
        (portfolio_id,),
    )

    if not equity:
        raise ValueError(f"No equity holdings found for '{resolved_name}'.")

    sector_weights: dict[str, float] = {}
    total = 0.0
    for row in equity:
        w = float(row["current_weight"] or 0.0)
        sector_weights[row["sector_name"]] = sector_weights.get(row["sector_name"], 0.0) + w
        total += w

    if total == 0:
        raise ValueError("Total equity weight is zero.")

    exposures = {
        s: round((w / total) * 100, 2)
        for s, w in sorted(sector_weights.items(), key=lambda x: -x[1])
    }

    return {
        "portfolio_name":        resolved_name,
        "exposures":             exposures,
        "equity_holdings_count": len(equity),
        "total_equity_weight":   round(total, 4),
    }


def _format_exposure(result: dict) -> str:
    lines = [
        f"Sector Exposure Breakdown - {result['portfolio_name']}",
        f"(Equity holdings: {result['equity_holdings_count']} | "
        f"Total equity weight: {result['total_equity_weight']:.4f})",
        "",
        f"{'Sector':<30} {'Exposure (%)':>12}",
        "-" * 44,
    ]
    for sector, pct in result["exposures"].items():
        lines.append(f"{sector:<30} {pct:>11.2f}%")
    lines.append("-" * 44)
    lines.append(f"{'TOTAL':<30} {'100.00%':>12}")
    return "\n".join(lines)


def _run_exposure(portfolio_name: str) -> str:
    try:
        return _format_exposure(_compute_sector_exposures(portfolio_name))
    except ValueError as exc:
        logger.warning("Exposure calc input error: %s", exc)
        return f"Error: {exc}"
    except Exception as exc:
        logger.error("Exposure calc error: %s", exc)
        return f"Error computing sector exposures: {exc}"


# StructuredTool — explicit name/description, no docstring parsing
exposure_calculator_tool = StructuredTool.from_function(
    func=_run_exposure,
    name="exposure_calculator_tool",
    description=(
        "Calculate sector exposures for a given portfolio (equity holdings only). "
        "Use for questions like: 'What are the sector exposures for X?', "
        "'Show sector breakdown for X', 'What percentage is in Technology for X?'. "
        "Returns a percentage breakdown table sorted by largest exposure first."
    ),
    args_schema=ExposureInput,
)
