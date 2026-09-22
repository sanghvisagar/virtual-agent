"""Tests for the loop itself.

The model is replaced by a scripted fake. That is the whole trick: because
llm.py's only job is messages-in-reply-out, the loop can be tested exhaustively
in milliseconds without Ollama running at all.
"""

from __future__ import annotations

import pytest

from agent.agent import Agent
from agent.memory import MAX_MESSAGES, Memory


class FakeLLM:
    """Returns pre-scripted replies, and records what it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append([dict(m) for m in messages])
        if not self.replies:
            return {"role": "assistant", "content": "done", "tool_calls": []}
        return self.replies.pop(0)


def text(content):
    return {"role": "assistant", "content": content, "tool_calls": []}


def wants(name, **arguments):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"name": name, "arguments": arguments}],
    }


def test_a_plain_reply_ends_the_loop_immediately():
    agent = Agent(llm=FakeLLM([text("Frank Herbert wrote Dune.")]))
    answer = agent.ask("Who wrote Dune?")

    assert answer.text == "Frank Herbert wrote Dune."
    assert answer.steps == []
    assert answer.model_calls == 1


def test_a_tool_call_runs_the_tool_and_feeds_the_result_back():
    llm = FakeLLM([wants("calculator", expression="0.18 * 4250"), text("That is 765.")])
    agent = Agent(llm=llm)
    answer = agent.ask("What is 18% of 4250?")

    assert answer.text == "That is 765."
    assert [s.tool for s in answer.steps] == ["calculator"]
    assert answer.steps[0].result == "765"

    # The decisive assertion: the second model call saw the tool result.
    second_call = llm.calls[1]
    assert second_call[-1] == {"role": "tool", "name": "calculator", "content": "765"}


def test_two_tools_run_in_sequence_within_one_turn():
    llm = FakeLLM(
        [
            wants("calculator", expression="2 + 2"),
            wants("calculator", expression="4 * 10"),
            text("Forty."),
        ]
    )
    answer = Agent(llm=llm).ask("Add 2 and 2, then multiply by 10")

    assert [s.result for s in answer.steps] == ["4", "40"]
    assert answer.model_calls == 3


def test_the_step_limit_stops_a_model_that_never_answers():
    llm = FakeLLM([wants("calculator", expression="1 + 1")] * 20)
    answer = Agent(llm=llm, max_steps=3).ask("loop forever")

    assert answer.hit_step_limit
    assert answer.model_calls == 3
    assert "stopped after 3 steps" in answer.text


def test_an_unknown_tool_is_reported_to_the_model_and_the_loop_survives():
    llm = FakeLLM([wants("teleport", to="Mars"), text("I cannot do that.")])
    answer = Agent(llm=llm).ask("teleport me")

    assert answer.text == "I cannot do that."
    assert answer.steps[0].result.startswith("Error: unknown tool")


def test_a_failing_tool_is_reported_to_the_model_and_the_loop_survives():
    llm = FakeLLM(
        [wants("calculator", expression="1 / 0"), text("That is undefined.")]
    )
    answer = Agent(llm=llm).ask("what is 1/0")

    assert "division by zero" in answer.steps[0].result
    assert answer.text == "That is undefined."


def test_history_carries_across_turns():
    llm = FakeLLM([text("765."), text("Half of that is 382.5.")])
    agent = Agent(llm=llm)
    agent.ask("What is 18% of 4250?")
    agent.ask("And half of that?")

    roles = [m["role"] for m in llm.calls[1]]
    assert roles == ["system", "user", "assistant", "user"]


def test_trim_keeps_the_system_prompt_and_reports_what_it_dropped():
    memory = Memory("stay helpful")
    for i in range(MAX_MESSAGES + 10):
        memory.add_user(f"message {i}")

    dropped = memory.trim()

    assert dropped >= 10
    assert memory.messages[0] == {"role": "system", "content": "stay helpful"}
    assert len(memory) <= MAX_MESSAGES
    assert memory.messages[-1]["content"] == f"message {MAX_MESSAGES + 9}"


def test_trim_does_not_leave_an_orphaned_tool_result():
    memory = Memory("stay helpful")
    for i in range(MAX_MESSAGES):
        memory.add_user(f"message {i}")
    memory.add_tool_result("calculator", "765")
    memory.add_tool_result("calculator", "766")

    memory.trim()

    assert memory.messages[1]["role"] != "tool"
