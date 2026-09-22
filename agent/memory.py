"""Conversation memory.

There is less here than the name suggests, and that is the point. The model is
stateless: every call is independent, and it remembers nothing between them. The
only reason a conversation feels continuous is that we resend the entire history
every single time.

So "memory" is a list of dicts. Each has a `role`:

    system     the standing instructions, always first
    user       what the person typed
    assistant  what the model replied (may carry tool_calls instead of content)
    tool       the result of running a tool, fed back so the model can use it

The cost of this design is that context grows with every turn, and the model's
context window does not. Hence `trim`.
"""

from __future__ import annotations

from typing import Any

# llama3.2 has a 128k window, but Ollama defaults to 2048 tokens unless told
# otherwise, and long contexts are slow on CPU. This is a deliberate ceiling.
MAX_MESSAGES = 40


class Memory:
    """The message list, plus the rule for what to drop when it gets too long."""

    def __init__(self, system_prompt: str) -> None:
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]

    def add(self, message: dict[str, Any]) -> None:
        self.messages.append(message)

    def add_user(self, content: str) -> None:
        self.add({"role": "user", "content": content})

    def add_tool_result(self, name: str, content: str) -> None:
        self.add({"role": "tool", "name": name, "content": content})

    def trim(self) -> int:
        """Drop the oldest turns when the history grows past MAX_MESSAGES.

        The system prompt is never dropped -- without it the agent forgets how to
        behave mid-conversation. Returns how many messages were removed, so the
        caller can say so out loud. Silent truncation produces an agent that
        mysteriously forgets, which is far harder to debug than one that says
        "I dropped 4 messages".
        """
        if len(self.messages) <= MAX_MESSAGES:
            return 0

        system, rest = self.messages[0], self.messages[1:]
        keep = MAX_MESSAGES - 1
        dropped = len(rest) - keep

        # A `tool` message whose `assistant` tool-call request was just dropped
        # is an orphan and confuses the model, so skip past any leading ones.
        rest = rest[dropped:]
        while rest and rest[0]["role"] == "tool":
            rest.pop(0)
            dropped += 1

        self.messages = [system] + rest
        return dropped

    def __len__(self) -> int:
        return len(self.messages)
