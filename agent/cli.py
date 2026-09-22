"""The terminal interface. Input, output, and the trace -- no agent logic."""

from __future__ import annotations

import argparse
import sys

from agent.agent import MAX_STEPS, Agent
from agent.llm import DEFAULT_MODEL, LLMUnavailable, OllamaLLM

DIM, BOLD, YELLOW, RESET = "\033[2m", "\033[1m", "\033[33m", "\033[0m"


def _print_trace(answer) -> None:
    for i, step in enumerate(answer.steps, 1):
        args = ", ".join(f"{k}={v!r}" for k, v in step.arguments.items())
        result = step.result if len(step.result) < 300 else step.result[:300] + "..."
        print(f"{DIM}  [{i}] {step.tool}({args})  {step.seconds:.2f}s{RESET}")
        for line in result.splitlines() or [""]:
            print(f"{DIM}      -> {line}{RESET}")
    print(
        f"{DIM}  {answer.model_calls} model call(s), "
        f"{len(answer.steps)} tool call(s), {answer.seconds:.1f}s total{RESET}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="agent", description="A small AI agent running on a local LLM."
    )
    parser.add_argument("prompt", nargs="*", help="Ask one question and exit.")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help="Ollama model.")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show every step of the loop."
    )
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = parser.parse_args()

    llm = OllamaLLM(model=args.model)
    try:
        llm.check()
    except LLMUnavailable as exc:
        print(f"{YELLOW}{exc}{RESET}", file=sys.stderr)
        return 1

    agent = Agent(llm=llm, max_steps=args.max_steps)

    def answer_once(question: str) -> None:
        answer = agent.ask(question)
        if answer.dropped_messages:
            print(
                f"{YELLOW}[dropped {answer.dropped_messages} old message(s) "
                f"to stay within the context limit]{RESET}"
            )
        if args.verbose:
            _print_trace(answer)
        print(answer.text or "(no answer)")

    # One-shot mode: `agent "what is 18% of 4250"`
    if args.prompt:
        answer_once(" ".join(args.prompt))
        return 0

    print(f"{BOLD}agent{RESET} on {args.model}. Ctrl-D or 'exit' to quit.")
    while True:
        try:
            question = input(f"\n{BOLD}you>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.lower() in ("exit", "quit"):
            return 0
        if not question:
            continue
        try:
            answer_once(question)
        except LLMUnavailable as exc:
            print(f"{YELLOW}{exc}{RESET}", file=sys.stderr)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
