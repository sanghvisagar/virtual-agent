"""The model client.

This file is the entire boundary between the agent and the language model. It
takes messages and tool schemas, and returns the model's reply. It does not know
what a tool does, what the loop is, or why it is being called.

That narrowness is the useful part: swapping llama3.2 for qwen2.5, or for a
hosted API, means editing this file and nothing else.
"""

from __future__ import annotations

from typing import Any

import ollama

DEFAULT_MODEL = "llama3.2:3b"


class LLMUnavailable(RuntimeError):
    """Ollama is not running, or the model is not pulled."""


class OllamaLLM:
    """A thin wrapper over one Ollama chat endpoint."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        keep_alive: str = "10m",
    ) -> None:
        self.model = model
        # Ollama unloads an idle model after 5 minutes, and reloading 2 GB from
        # disk on a machine without a GPU costs minutes, not seconds. Holding it
        # in RAM longer is the single biggest speed win available here.
        self.keep_alive = keep_alive
        # temperature=0 makes runs reproducible. When a tool call comes out
        # malformed you want to see it again, not chase a ghost.
        self.temperature = temperature
        self._client = ollama.Client()

    def check(self) -> None:
        """Fail early and usefully, before the first question is asked."""
        try:
            installed = [m.model for m in self._client.list().models]
        except Exception as exc:
            raise LLMUnavailable(
                "Cannot reach Ollama. Start it with:\n\n    ollama serve\n"
            ) from exc

        if not any(name == self.model or name.startswith(f"{self.model}:")
                   for name in installed):
            raise LLMUnavailable(
                f"Model {self.model!r} is not pulled. Get it with:\n\n"
                f"    ollama pull {self.model}\n\n"
                f"Installed: {', '.join(installed) or 'none'}"
            )

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Send the conversation, get one reply back.

        The reply is a dict with `content` (text) and possibly `tool_calls` (a
        list of requests to run something). Deciding what to do with those is
        agent.py's job, not this file's.
        """
        try:
            response = self._client.chat(
                model=self.model,
                messages=messages,
                tools=tools or None,
                options={"temperature": self.temperature},
                keep_alive=self.keep_alive,
            )
        except Exception as exc:
            raise LLMUnavailable(f"Ollama call failed: {exc}") from exc

        message = response["message"]
        return {
            "role": "assistant",
            "content": message.get("content", "") or "",
            "tool_calls": [
                {
                    "name": tc["function"]["name"],
                    "arguments": tc["function"].get("arguments") or {},
                }
                for tc in (message.get("tool_calls") or [])
            ],
        }
