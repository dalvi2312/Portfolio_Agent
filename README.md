# Portfolio Intelligence Agent – Solution Documentation

## Overview

An AI-powered agent that answers natural-language questions about portfolio data.  
Built with **LangGraph** (agentic workflow), **LangChain** (tool abstractions), and **Google Gemini** (LLM), backed by a **SQLite** database populated from the provided CSV data.

---

## Architecture

```
                        ┌─────────────────────────────────────────┐
                        │           LangGraph Agent Graph          │
                        │                                          │
  User Question  ──────►│  START → llm_node ──tool_call?──► tools_node
                        │              ▲                       │
                        │              └───────────────────────┘
                        │          no tool_call → END          │
                        └─────────────────────────────────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         │                             │
                  sql_query_tool            exposure_calculator_tool
                         │                             │
                  Gemini generates SQL        Queries DB directly
                  → SQLite executes          → Aggregates weights
                  → Formats results          → Normalises to 100%
```

### Components

| File / Module | Role |
|---|---|
| `setup_database.py` | One-time script: loads CSVs → SQLite |
| `db/database.py` | SQLite connection manager + schema introspection |
| `tools/sql_tool.py` | LangChain tool: NL → Gemini SQL → SQLite |
| `tools/exposure_calculator.py` | LangChain tool: portfolio name → sector % |
| `agent/state.py` | LangGraph `TypedDict` state schema |
| `agent/prompts.py` | All LLM prompt templates |
| `agent/agent.py` | LangGraph graph + `PortfolioAgent` class |
| `main.py` | CLI entry point (REPL + single-question mode) |
| `evaluator.py` | Ground-truth evaluation script |
| `streamlit_app.py` | Optional web UI |

---

## Quick Start

### 1 – Clone / copy project files

```bash
# Your project directory should contain:
portfolio_agent/
├── data/                    ← CSV files (provided)
├── database_schema.sql      ← SQL schema (provided)
├── ground_truth_dataset.json
├── requirements.txt
├── .env.example
├── setup_database.py
├── db/
├── tools/
├── agent/
├── main.py
├── evaluator.py
└── streamlit_app.py
```

### 2 – Create a virtual environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3 – Install dependencies

```bash
pip install -r requirements.txt
```

### 4 – Set up your API key

```bash
cp .env.example .env
# Open .env and set GEMINI_API_KEY=<your key>
# Get a free key at: https://ai.google.dev/gemini-api/docs
```

### 5 – Build the database

```bash
python setup_database.py
```

Expected output:
```
INFO: Schema created successfully.
INFO: Loaded 10 rows into table 'sectors'.
INFO: Loaded 29 rows into table 'securities'.
...
── Database verification ──────────────────────────
  sectors                          10 rows
  securities                       29 rows
  portfolios                       13 rows
  holdings                        ~80 rows
  ...
INFO: Database setup complete: portfolio_database.db
```

---

## Running the Agent

### Interactive REPL

```bash
python main.py
```

```
╔══════════════════════════════════════════════════════╗
║          Portfolio Intelligence Agent  🏦             ║
║  Type a question or 'quit' / 'exit' to stop.         ║
╚══════════════════════════════════════════════════════╝

Example questions:
  1. How many portfolios do we have in total?
  ...

You: How many portfolios do we have in total?
Agent: We have 13 portfolios in total.

You: What are the sector exposures for the Tech Innovation Fund?
Agent: Sector Exposure Breakdown – Tech Innovation Fund
       Technology         72.50%
       Consumer Disc.     15.00%
       ...
```

### Single question (non-interactive)

```bash
python main.py --question "What is the total AUM for high risk portfolios?"
```

### Verbose mode (shows tool selection, generated SQL)

```bash
python main.py --verbose
```

---

## Running the Evaluator

```bash
# Evaluate all 10 ground-truth questions
python evaluator.py

# Only SQL questions
python evaluator.py --type text2sql

# Only exposure calculator questions
python evaluator.py --type exposure_calculator

# Specific question IDs
python evaluator.py --id 1 5 9

# Save results to JSON
python evaluator.py --output eval_results.json
```

Sample output:
```
Evaluating 10 question(s)…

[1/10] Q1 (text2sql, easy):
  Q: How many portfolios do we have in total?
  A: 13
  ✅ PASS: Correct count returned
  ⏱  2.1s

[9/10] Q9 (exposure_calculator, medium):
  Q: What are the sector exposures for the Tech Innovation Fund?
  A: Sector Exposure Breakdown – Tech Innovation Fund ...
  ✅ PASS: Sector percentages provided for correct portfolio
  ⏱  1.3s

════════════════════════════════════════════
EVALUATION SUMMARY
════════════════════════════════════════════
  Total questions :  10
  ✅ PASS          :   9
  ❌ FAIL          :   1
  ⚠️  ERROR         :   0
  Accuracy        : 90.0%
```

---

## Optional Streamlit UI

```bash
streamlit run streamlit_app.py
# Opens at http://localhost:8501
```

---

## Tool Selection Logic

The agent uses Gemini's function-calling to choose the right tool:

| Question type | Tool selected |
|---|---|
| Count / list portfolios | `sql_query_tool` |
| Filter / search by attribute | `sql_query_tool` |
| Aggregation (SUM, AVG) | `sql_query_tool` |
| Multi-table JOINs | `sql_query_tool` |
| Sector exposure / weight breakdown | `exposure_calculator_tool` |

---

## Design Decisions

1. **LangGraph ReAct loop** – The graph loops `llm_node → tools_node → llm_node` until Gemini stops requesting tools. This naturally handles multi-step reasoning without custom orchestration code.

2. **Schema injection** – The full table schema is injected into every SQL-generation prompt so Gemini always has accurate column names and types.

3. **Read-only SQL guard** – `DatabaseManager.execute_query()` rejects any statement that doesn't start with `SELECT`, `WITH`, or `EXPLAIN`, preventing accidental data modification.

4. **Exposure normalisation** – Sector weights are normalised over the equity-only sub-total (not the full portfolio weight including bonds), matching the README requirement.

5. **Semantic evaluation** – Instead of exact-match string comparison, the evaluator uses Gemini as a judge, which handles rephrased but correct answers gracefully.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(required)* | Google AI Studio API key |
| `DB_PATH` | `portfolio_database.db` | SQLite database path |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Gemini model to use |
| `LOG_LEVEL` | `INFO` | Python logging level |
