"""
main.py
───────
Command-line interface for the Portfolio Agent (Ollama / llama3:8b).

Usage:
    # Interactive REPL
    python main.py

    # Single question (non-interactive)
    python main.py --question "How many portfolios do we have?"

    # Verbose mode (shows tool selection, generated SQL, etc.)
    python main.py --verbose
"""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def _check_prerequisites() -> None:
    """Fail fast with a clear message if setup is incomplete."""
    db_path = os.getenv("DB_PATH", "portfolio_database.db")
    if not os.path.exists(db_path):
        print(
            f"\n[ERROR] Database not found at '{db_path}'.\n"
            "  Run:  python setup_database.py\n"
        )
        sys.exit(1)

    # Quick Ollama connectivity check
    import urllib.request
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        urllib.request.urlopen(f"{base_url}/api/tags", timeout=3)
    except Exception:
        print(
            f"\n[ERROR] Cannot reach Ollama at {base_url}.\n"
            "  Make sure Ollama is running:  ollama serve\n"
            "  And the model is pulled:      ollama pull llama3:8b\n"
        )
        sys.exit(1)


BANNER = """
+------------------------------------------------------+
|       Portfolio Intelligence Agent                   |
|  Powered by Ollama (llama3:8b) + LangGraph           |
|  Type a question or 'quit' to exit.                  |
+------------------------------------------------------+
"""

EXAMPLE_QUESTIONS = [
    "How many portfolios do we have in total?",
    "What are the names of all active portfolios?",
    "Which securities are in the Technology sector?",
    "What is the total AUM for high risk portfolios?",
    "What are the sector exposures for the Tech Innovation Fund?",
]


def run_repl(agent) -> None:
    print(BANNER)
    print("Example questions:")
    for i, q in enumerate(EXAMPLE_QUESTIONS, 1):
        print(f"  {i}. {q}")
    print()

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q", "bye"}:
            print("Goodbye!")
            break

        print("Agent: thinking...", end="\r")
        answer = agent.answer_question(question)
        print(" " * 30, end="\r")
        print(f"Agent: {answer}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Portfolio Intelligence Agent - CLI")
    parser.add_argument("--question", "-q", type=str,
                        help="Ask a single question and exit.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose/debug logging.")
    args = parser.parse_args()

    _configure_logging(args.verbose)
    _check_prerequisites()

    from agent.agent import PortfolioAgent
    agent = PortfolioAgent()

    if args.question:
        print(agent.answer_question(args.question))
    else:
        run_repl(agent)


if __name__ == "__main__":
    main()
