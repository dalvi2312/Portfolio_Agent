"""
agent/prompts.py
────────────────
All prompt templates for the portfolio agent.

Three prompts are used in the 3-stage pipeline:
  ROUTER_PROMPT   – classifies a question as "sql" or "exposure"
  PORTFOLIO_EXTRACTION_PROMPT – extracts a portfolio name from an exposure question
  RESPONDER_PROMPT – turns raw tool output into a natural-language answer
"""

# ── Stage 1: Router ───────────────────────────────────────────────────────────
ROUTER_PROMPT = """You are a query classifier for a portfolio management system.
Classify the question below into exactly ONE category.

Categories:
- sql        : Any question about counts, lists, names, rankings, aggregations,
               financial metrics, prices, AUM, transactions, risk, performance,
               or any general database lookup.
- exposure   : Questions that specifically ask for sector exposure percentages
               or sector weight breakdown for a named portfolio.

Respond with ONLY the single word "sql" or "exposure". No punctuation. No explanation.

Question: {question}
Category:"""


# ── Stage 1b: Portfolio name extraction (for exposure questions) ───────────────
PORTFOLIO_EXTRACTION_PROMPT = """Extract the portfolio name from the question below.
Reply with ONLY the portfolio name — no extra words, no punctuation.

Question: {question}
Portfolio name:"""


# ── Stage 3: Responder ────────────────────────────────────────────────────────
RESPONDER_PROMPT = """You are a professional portfolio analysis assistant.

User question:
{question}

Data retrieved from the database:
{tool_result}

Instructions:
- Write a clear, direct, professional answer using ONLY the data above.
- Preserve every number exactly as given — do not round or estimate.
- If the data is a table, present it cleanly as part of your answer.
- If the data is a single value, state it in one sentence.
- Do not mention tools, SQL, databases, or internal systems.
- Do not add information that is not in the retrieved data.
- Keep the answer concise.

Answer:"""
