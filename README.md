# virtual-agent

A small AI agent, written by hand, running a local open-source model. About 300
lines of Python and one dependency.

The point is not capability — a 3B model on a CPU is not impressive. The point is
that every step between your question and the answer is visible in code you can
read. No LangChain, no framework, nothing hidden.

## The whole idea

An agent is a language model plus a `while` loop.

A chatbot sends your message to the model and prints the reply. An agent sends
your message, checks whether the reply is an *answer* or a *request to act*, and
if it is a request, runs the tool and sends the result back — as many times as
needed.

```
you: What is 18% of 4250?

  model  -> "call calculator(expression='0.18 * 4250')"
  code   -> runs it, gets "765"
  code   -> sends "765" back as a message
  model  -> "18% of 4250 is 765."   (no tool call -> loop ends)
```

That loop is [`agent/agent.py`](agent/agent.py). Read that file first; the rest
exists so it can stay short.

## Setup

Needs [Ollama](https://ollama.com) and Python 3.10+.

```bash
ollama serve                 # in another terminal, if not already running
ollama pull llama3.2:3b

uv venv && uv pip install -e ".[dev]"
```

## Use

```bash
# One question
.venv/bin/python -m agent.cli "What is 18% of 4250?"

# Show every step of the loop — do this at least once
.venv/bin/python -m agent.cli -v "What time is it in Tokyo?"

# Interactive
.venv/bin/python -m agent.cli
```

`-v` prints each tool call, its arguments, its result and its timing. It is the
most useful thing here: when the agent does something strange, the trace shows
exactly which step went wrong.

## How it fits together

| File | What it owns |
| --- | --- |
| [`agent/agent.py`](agent/agent.py) | **The loop.** Dispatch, step cap, error recovery |
| [`agent/tools.py`](agent/tools.py) | `@tool` decorator, the registry, schema generation, the tools |
| [`agent/llm.py`](agent/llm.py) | Ollama client: messages in, reply out. Nothing else |
| [`agent/memory.py`](agent/memory.py) | The message list and what to drop when it overflows |
| [`agent/cli.py`](agent/cli.py) | Terminal input, output, the trace |

Four ideas are worth noticing as you read:

1. **The model knows what tools exist** because their JSON schemas go in every
   request. `@tool` builds those schemas from the function's type hints and
   docstring, so they cannot drift out of date.
2. **Tool results come back as messages.** The model has no memory between calls.
   A result exists only because we put it in the next request.
3. **Errors go to the model, not the user.** `tools.call()` never raises. An
   unknown tool, bad arguments or an exception all become `"Error: ..."` strings
   the model reads and reacts to. Crashing teaches it nothing.
4. **The loop is bounded.** `MAX_STEPS = 6`. Small models loop; without a cap
   that is a hang instead of an error message.

## Adding a tool

One decorated function. Nothing else to register.

```python
@tool
def word_count(text: str) -> str:
    """Count the words in some text.

    Args:
        text: The text to count.
    """
    return str(len(text.split()))
```

The type hints become the schema, the summary tells the model when to use it, and
the `Args:` lines describe each parameter. Tool descriptions are prompt
engineering — vague ones produce a model that calls the wrong tool.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q     # 27 tests, ~2 seconds, no Ollama needed
```

`tests/test_agent.py` swaps the model for a scripted fake. Because `llm.py` is
just messages-in-reply-out, the loop can be tested exhaustively in milliseconds.
Separating the loop from the model is the single most useful habit in agent work.

## Safety

An agent is code that decides what code to run, so the tools have edges:

- `calculator` walks the AST and allows arithmetic nodes only — never `eval()`.
  `__import__('os').system(...)` raises instead of running.
- `list_files` resolves paths and refuses anything outside the project directory.
- Notes write to one fixed filename, never one taken from model output.
- No tool reaches the network.

## What actually happens when you run it

Measured on an Intel i5-8257U with 8 GB RAM, CPU-only, llama3.2:3b.

| Prompt | Result |
| --- | --- |
| "What is 18% of 4250?" | ✅ `calculator('0.18 * 4250')` → 765 |
| "How many files in agent/, and what time in Tokyo?" | ✅ two tool calls in one turn, both correct |
| "Save a note that the demo is on Friday" | ✅ `save_note(...)` |
| "What notes did I save?" | ✅ `read_notes()`, read back correctly |
| "Who wrote the novel Dune?" | ⚠️ right answer, but it calls a tool first |

About 100 seconds per model call, so a typical task is 200–250 seconds. The first
call of a session adds ~2 minutes loading the model into RAM. That is the
hardware, not the code — to trade purity for speed, point `agent/llm.py` at a
hosted open-source endpoint (Groq, Together). One file, which is why the boundary
is drawn there.

### The one thing that does not work

llama3.2:3b **over-triggers**: given tools, it reaches for one even when the
question is pure general knowledge.

```
you> Who wrote the novel Dune?
  [1] save_note(text='Frank Herbert')
      -> Note saved.
The novel "Dune" was written by Frank Herbert...
```

The answer is correct; the tool call was pointless. This survived two rounds of
prompt work, including a system prompt that names *that exact question* as one
needing no tool, and rewriting the tool descriptions so `save_note` and
`read_notes` no longer share wording. (Those rewrites did fix a worse bug: asking
"what notes did I save?" used to call `save_note('')`.)

This is worth understanding rather than hiding, because it is the honest shape of
the problem:

- **It is a model limitation, not a code bug.** 3B models are weak at deciding
  *not* to act. Prompting reduces it; it does not remove it.
- **Tool descriptions are load-bearing.** They are the only thing standing
  between the model and the wrong action. Two tools with overlapping wording
  produce wrong choices, reliably.
- **This is why real agents gate write tools.** A spurious read is noise; a
  spurious write has consequences. `save_note` refuses empty text for this
  reason, and a production version would confirm writes rather than trust the
  model's judgement.

Try `-m qwen2.5:3b` (after `ollama pull qwen2.5:3b`) to see how much of an
agent's reliability is the model rather than the code.

## Where to go next

- Run it with `-v` and watch a two-step task. That is the whole lesson.
- Break something on purpose: set `MAX_STEPS = 1`, or make a tool always raise.
- Remove the "always use the calculator" line from `SYSTEM_PROMPT` and watch the
  model start doing arithmetic itself, badly. Prompt text *is* behaviour.
- Swap the model: `-m qwen2.5:3b`. Notice how much tool-calling reliability
  depends on the model rather than the code.
- Then read a framework's agent loop and see what it adds.
# virtual-agent
