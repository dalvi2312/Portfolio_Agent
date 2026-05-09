"""
evaluator.py
────────────
Evaluates the Portfolio Agent against ground_truth_dataset.json.

For each question the evaluator:
  1. Runs it through the agent.
  2. Uses Ollama as a semantic judge (PASS / FAIL).
  3. Prints a per-question table and an accuracy summary.

Judge design
────────────
The judge prompt is kept strict and concrete to avoid vague responses.
The verdict parser looks for the FIRST occurrence of PASS or FAIL anywhere
in the response (not just at the start), which handles numbered/bulleted
judge outputs that previously caused false FAILs.

Usage:
    python evaluator.py                     # all questions
    python evaluator.py --id 1 5 9          # specific IDs
    python evaluator.py --type text2sql     # only SQL questions
    python evaluator.py --output results.json
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
Verdict = Literal["PASS", "FAIL", "ERROR"]


# ── Judge LLM ─────────────────────────────────────────────────────────────────

def _build_judge_llm():
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
    )


# ── Verdict parser ────────────────────────────────────────────────────────────

def _parse_verdict(text: str) -> tuple[Verdict, str]:
    """
    Extract PASS or FAIL from judge response.

    Searches the ENTIRE response for the first occurrence of PASS or FAIL
    (case-insensitive). This handles:
      - "PASS: reason"
      - "1. PASS: ..."
      - "The answer is a PASS because..."
      - Numbered/bulleted lists that start with a number not a verdict word
    """
    upper = text.upper()
    pass_idx = upper.find("PASS")
    fail_idx = upper.find("FAIL")

    if pass_idx == -1 and fail_idx == -1:
        return "FAIL", f"Judge gave unclear response: {text[:120]}"

    # Whichever appears first wins
    if pass_idx != -1 and (fail_idx == -1 or pass_idx < fail_idx):
        # Extract reason: everything after "PASS" on the same stretch of text
        reason_raw = text[pass_idx + 4 : pass_idx + 250].lstrip(":.- ").strip()
        # Keep only up to the next newline for brevity
        reason = reason_raw.split("\n")[0].strip() or "Correct"
        return "PASS", reason
    else:
        reason_raw = text[fail_idx + 4 : fail_idx + 250].lstrip(":.- ").strip()
        reason = reason_raw.split("\n")[0].strip() or "Incorrect"
        return "FAIL", reason


# ── Prompt builders ───────────────────────────────────────────────────────────

def _build_sql_judge_prompt(
    question: str, agent_answer: str, ground_truth: dict
) -> str:
    expected_type = ground_truth.get("expected_result_type", "any")
    explanation   = ground_truth.get("explanation", "")
    return f"""You are evaluating an AI assistant's answer to a portfolio database question.

QUESTION: {question}

EXPECTED RESULT TYPE: {expected_type}
EXPECTED BEHAVIOR: {explanation}

AGENT ANSWER:
{agent_answer}

EVALUATION CRITERIA – the agent PASSES if ALL of the following are true:
1. The answer contains actual data values (numbers, names, or a table) — not just SQL code or a statement about using a tool.
2. The data matches the expected result type: {expected_type}.
3. The answer appears to correctly address the question.

The agent FAILS if:
- It only shows SQL code without query results.
- It only says "I will use a tool" without returning data.
- The data is clearly wrong or unrelated.
- It returned an error message.

Start your response with PASS or FAIL, then give a one-sentence reason.
Example: PASS: The answer correctly lists 13 portfolio names.
Example: FAIL: The answer only shows SQL code without executing it."""


def _build_exposure_judge_prompt(
    question: str, agent_answer: str, ground_truth: dict
) -> str:
    portfolio_name = ground_truth.get("parameters", {}).get("portfolio_name", "")
    return f"""You are evaluating an AI assistant's answer to a sector exposure question.

QUESTION: {question}
EXPECTED PORTFOLIO: {portfolio_name}

AGENT ANSWER:
{agent_answer}

EVALUATION CRITERIA – the agent PASSES if ALL of the following are true:
1. The answer mentions sector name(s) with percentage value(s).
   Note: A single sector at 100% is a VALID answer for a single-sector portfolio.
2. The answer relates to the correct portfolio: {portfolio_name}.
3. The percentages appear to sum to approximately 100%.
4. Bond/fixed-income holdings are not included (equity only).

The agent FAILS if:
- No percentage values are present.
- The portfolio name is wrong or missing.
- It returned an error message.

Start your response with PASS or FAIL, then give a one-sentence reason.
Example: PASS: Lists Technology at 100% for a single-sector portfolio — correct.
Example: FAIL: No percentage values were provided."""


# ── Core evaluation loop ──────────────────────────────────────────────────────

def evaluate(
    questions: list[dict],
    filter_ids: list[int] | None = None,
    filter_type: str | None = None,
) -> list[dict]:
    from agent.agent import PortfolioAgent

    agent  = PortfolioAgent()
    judge  = _build_judge_llm()
    results: list[dict] = []

    if filter_ids:
        questions = [q for q in questions if q["id"] in filter_ids]
    if filter_type:
        questions = [q for q in questions if q["type"] == filter_type]

    total = len(questions)
    print(f"\nEvaluating {total} question(s)...\n")

    for idx, item in enumerate(questions, 1):
        q_id       = item["id"]
        q_text     = item["question"]
        q_type     = item["type"]
        difficulty = item["difficulty"]
        gt         = item["ground_truth"]

        print(f"[{idx}/{total}] Q{q_id} ({q_type}, {difficulty}):")
        print(f"  Q: {q_text}")

        # Run agent
        start_ts = time.time()
        try:
            agent_answer = agent.answer_question(q_text)
        except Exception as exc:
            agent_answer = f"AGENT ERROR: {exc}"
        elapsed = round(time.time() - start_ts, 2)

        display = (agent_answer[:200] + "...") if len(agent_answer) > 200 else agent_answer
        print(f"  A: {display}")

        # Judge
        if agent_answer.startswith(("AGENT ERROR", "Error:")):
            verdict, reason = "ERROR", agent_answer[:120]
        else:
            from tools.sql_tool import _content_to_str
            try:
                if q_type == "text2sql":
                    prompt = _build_sql_judge_prompt(q_text, agent_answer, gt)
                else:
                    prompt = _build_exposure_judge_prompt(q_text, agent_answer, gt)

                judge_response = judge.invoke(prompt)
                judge_text     = _content_to_str(judge_response.content)
                verdict, reason = _parse_verdict(judge_text)
            except Exception as exc:
                verdict, reason = "ERROR", f"Judge failed: {exc}"

        verdict_label = {"PASS": "[PASS]", "FAIL": "[FAIL]", "ERROR": "[ERROR]"}.get(verdict, verdict)
        print(f"  {verdict_label} {reason}")
        print(f"  Time: {elapsed}s\n")

        results.append({
            "id":           q_id,
            "type":         q_type,
            "difficulty":   difficulty,
            "question":     q_text,
            "agent_answer": agent_answer,
            "verdict":      verdict,
            "reason":       reason,
            "elapsed_s":    elapsed,
        })

    return results


# ── Summary printer ───────────────────────────────────────────────────────────

def _print_summary(results: list[dict]) -> None:
    total    = len(results)
    passed   = sum(1 for r in results if r["verdict"] == "PASS")
    failed   = sum(1 for r in results if r["verdict"] == "FAIL")
    errors   = sum(1 for r in results if r["verdict"] == "ERROR")
    accuracy = (passed / total * 100) if total else 0.0

    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Total questions : {total}")
    print(f"  PASS            : {passed}")
    print(f"  FAIL            : {failed}")
    print(f"  ERROR           : {errors}")
    print(f"  Accuracy        : {accuracy:.1f}%")
    print()

    for q_type in ("text2sql", "exposure_calculator"):
        subset = [r for r in results if r["type"] == q_type]
        if subset:
            sp = sum(1 for r in subset if r["verdict"] == "PASS")
            print(f"  {q_type:<28} {sp}/{len(subset)} passed")
    print()
    for diff in ("easy", "medium", "hard"):
        subset = [r for r in results if r["difficulty"] == diff]
        if subset:
            sp = sum(1 for r in subset if r["verdict"] == "PASS")
            print(f"  {diff.capitalize():<28} {sp}/{len(subset)} passed")

    print("=" * 60)
    print(f"\n  {'ID':<4} {'Type':<22} {'Diff':<8} {'Verdict':<8} {'Time':>6}")
    print("  " + "-" * 52)
    for r in results:
        print(
            f"  {r['id']:<4} {r['type']:<22} {r['difficulty']:<8} "
            f"{r['verdict']:<8} {r['elapsed_s']:>5.1f}s"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Portfolio Agent Evaluator")
    parser.add_argument("--ground-truth", default="ground_truth_dataset.json")
    parser.add_argument("--id", nargs="+", type=int, dest="filter_ids")
    parser.add_argument("--type", choices=["text2sql", "exposure_calculator"],
                        dest="filter_type")
    parser.add_argument("--output", type=str)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    db_path = os.getenv("DB_PATH", "portfolio_database.db")
    if not os.path.exists(db_path):
        print(f"[ERROR] Database not found at '{db_path}'. Run: python setup_database.py")
        sys.exit(1)

    if not os.path.exists(args.ground_truth):
        print(f"[ERROR] Ground truth not found: {args.ground_truth}")
        sys.exit(1)

    with open(args.ground_truth) as fh:
        dataset = json.load(fh)

    results = evaluate(
        dataset["questions"],
        filter_ids=args.filter_ids,
        filter_type=args.filter_type,
    )
    _print_summary(results)

    if args.output:
        payload = {
            "run_at":   datetime.utcnow().isoformat(),
            "total":    len(results),
            "passed":   sum(1 for r in results if r["verdict"] == "PASS"),
            "accuracy": round(
                sum(1 for r in results if r["verdict"] == "PASS") / len(results) * 100, 1
            ) if results else 0,
            "results":  results,
        }
        with open(args.output, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
