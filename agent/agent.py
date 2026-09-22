"""The loop.

This is the file the project exists for. Everything else is here so that this
one can stay short enough to read in a sitting.

The difference between a chatbot and an agent is the `while` below. A chatbot
sends your message and prints the reply. An agent sends your message, checks
whether the reply is an answer or a request to act, and if it is a request, acts
and sends the result back -- as many times as needed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from agent import tools
from agent.llm import OllamaLLM
from agent.memory import Memory

MAX_STEPS = 6

SYSTEM_PROMPT = """You are a helpful assistant. You have tools available, but most questions do not need one.

Call a tool ONLY for these:
- arithmetic of any kind, however simple -> calculator
- the current time or today's date -> current_time
- what files are in a directory -> list_files
- storing something the user asks you to remember -> save_note
- looking up what was stored earlier -> read_notes

Answer everything else directly from your own knowledge, with NO tool call:
general knowledge, history, definitions, explanations, opinions, and ordinary
conversation. For example "Who wrote Dune?" and "What is the capital of France?"
take no tool at all.

Before calling a tool, check that its description matches what the user actually
asked. Saving and looking up are different requests: if the user asks what they
saved, read; if the user gives you something to remember, write. Never invent
arguments and never call a tool with empty arguments.

When a tool returns a result, read it and answer the question in plain language.
If a tool returns an error, do not call it again with the same arguments."""


@dataclass
class Step:
    """One tool call and its outcome. Purely for the trace."""

    tool: str
    arguments: dict[str, Any]
    result: str
    seconds: float


@dataclass
class Answer:
    """What one turn produced: the reply, plus how it got there."""

    text: str
    steps: list[Step] = field(default_factory=list)
    model_calls: int = 0
    seconds: float = 0.0
    hit_step_limit: bool = False
    dropped_messages: int = 0


class Agent:
    def __init__(
        self,
        llm: OllamaLLM | None = None,
        max_steps: int = MAX_STEPS,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.llm = llm or OllamaLLM()
        self.max_steps = max_steps
        self.memory = Memory(system_prompt)

    def ask(self, message: str) -> Answer:
        """Run the agent loop until the model answers, or we run out of steps."""
        started = time.monotonic()
        answer = Answer(text="")

        self.memory.add_user(message)
        answer.dropped_messages = self.memory.trim()

        for _ in range(self.max_steps):
            # 1. Ask the model what to do, given everything so far.
            reply = self.llm.chat(self.memory.messages, tools=tools.schemas())
            answer.model_calls += 1

            # 2. No tool call means the model is answering. The loop is over.
            if not reply["tool_calls"]:
                self.memory.add({"role": "assistant", "content": reply["content"]})
                answer.text = reply["content"].strip()
                break

            # 3. Record the request itself before running anything. The model
            #    needs to see its own tool call in the history, or the tool
            #    results that follow have nothing to attach to.
            self.memory.add(
                {
                    "role": "assistant",
                    "content": reply["content"],
                    "tool_calls": [
                        {"function": {"name": c["name"], "arguments": c["arguments"]}}
                        for c in reply["tool_calls"]
                    ],
                }
            )

            # 4. Run each requested tool and feed the result back as a message.
            #    tools.call never raises -- failures come back as strings the
            #    model can read and react to.
            for requested in reply["tool_calls"]:
                tool_started = time.monotonic()
                result = tools.call(requested["name"], requested["arguments"])
                self.memory.add_tool_result(requested["name"], result)
                answer.steps.append(
                    Step(
                        tool=requested["name"],
                        arguments=requested["arguments"],
                        result=result,
                        seconds=time.monotonic() - tool_started,
                    )
                )

            answer.dropped_messages += self.memory.trim()
            # 5. Round again: the model now sees the results it asked for.
        else:
            # Python runs this only if the for-loop was never broken out of,
            # which means the model asked for a tool on every single step.
            # Small models do this. Without the cap it would never stop.
            answer.hit_step_limit = True
            answer.text = (
                f"I stopped after {self.max_steps} steps without reaching an answer. "
                f"Tools I tried: {', '.join(s.tool for s in answer.steps)}."
            )

        answer.seconds = time.monotonic() - started
        return answer
