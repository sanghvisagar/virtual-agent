"""The tool registry.

A "tool" is an ordinary Python function that the model is allowed to ask us to
run. The model never runs anything itself -- it emits a request, and the code in
agent.py dispatches it here.

For the model to ask for a tool, it has to know the tool exists and what
arguments it takes. That description is a JSON schema. Writing those by hand is
tedious and drifts out of date, so `@tool` generates the schema from the
function's own type hints and docstring.
"""

from __future__ import annotations

import ast
import inspect
import operator
import os
import zoneinfo
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, get_type_hints

# name -> {"fn": callable, "schema": {...}}
REGISTRY: dict[str, dict[str, Any]] = {}

# Python type -> JSON schema type. Anything not listed becomes a string, which
# is the honest default: the model emits text either way.
_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Split a docstring into a summary and per-argument descriptions.

    Recognises a Google-style ``Args:`` block. The summary is everything before
    it; each ``name: description`` line under it describes one parameter.
    """
    summary_lines: list[str] = []
    params: dict[str, str] = {}
    in_args = False

    for raw in (doc or "").splitlines():
        line = raw.strip()
        if line.lower() in ("args:", "arguments:", "parameters:"):
            in_args = True
            continue
        if in_args:
            if not line:
                continue
            if ":" in line:
                name, _, desc = line.partition(":")
                params[name.strip()] = desc.strip()
        else:
            summary_lines.append(line)

    return " ".join(summary_lines).strip(), params


def tool(fn: Callable) -> Callable:
    """Register a function as a tool the model may call.

    The function is returned unchanged, so it stays directly callable and
    directly testable. Registration is the only side effect.
    """
    hints = get_type_hints(fn)
    signature = inspect.signature(fn)
    description, param_docs = _parse_docstring(fn.__doc__ or "")

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in signature.parameters.items():
        annotation = hints.get(name, str)
        properties[name] = {
            "type": _JSON_TYPES.get(annotation, "string"),
            "description": param_docs.get(name, ""),
        }
        # No default means the model must supply it.
        if param.default is inspect.Parameter.empty:
            required.append(name)

    REGISTRY[fn.__name__] = {
        "fn": fn,
        "schema": {
            "type": "function",
            "function": {
                "name": fn.__name__,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        },
    }
    return fn


def schemas() -> list[dict[str, Any]]:
    """Every registered schema, in the shape Ollama's `tools` parameter wants."""
    return [entry["schema"] for entry in REGISTRY.values()]


def call(name: str, arguments: dict[str, Any]) -> str:
    """Run a tool by name and return its result as a string.

    Never raises. Every failure -- unknown tool, wrong arguments, an exception
    inside the tool -- comes back as an "Error: ..." string, because that string
    goes back to the model as a message. A model that is told what went wrong can
    try something else; a crashed process cannot.
    """
    entry = REGISTRY.get(name)
    if entry is None:
        available = ", ".join(sorted(REGISTRY)) or "none"
        return f"Error: unknown tool {name!r}. Available tools: {available}"

    try:
        return str(entry["fn"](**arguments))
    except TypeError as exc:
        return f"Error: wrong arguments for {name!r}: {exc}"
    except Exception as exc:  # noqa: BLE001 - the model is the error handler here
        return f"Error: {type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# The tools themselves
# ---------------------------------------------------------------------------

# Only these AST nodes are allowed through the calculator. eval() would also
# work and would also let the model run `__import__("os").system(...)`, which is
# exactly the kind of thing a model that has read the internet eventually emits.
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"only numbers are allowed, got {node.value!r}")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if type(node.op) in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise ValueError("division by zero")
        # Keep exponents small; 9**9**9 would hang the process.
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _ALLOWED_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_eval_node(node.operand))

    raise ValueError(f"unsupported expression element: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression. Use this for every calculation, however
    simple -- do not compute numbers yourself.

    Args:
        expression: Arithmetic only, e.g. "0.18 * 4250" or "(12 + 7) ** 2".
    """
    tree = ast.parse(expression, mode="eval")
    result = _eval_node(tree)
    # 765.0 reads worse than 765 in a tool result the model has to interpret.
    if isinstance(result, float) and result.is_integer():
        return str(int(result))
    return str(result)


@tool
def current_time(timezone: str = "UTC") -> str:
    """Get the current date and time. You have no clock, so use this tool when
    the user asks about the current time or today's date, and not otherwise.

    Args:
        timezone: An IANA timezone name, e.g. "Asia/Tokyo" or "Asia/Kolkata".
    """
    try:
        zone = zoneinfo.ZoneInfo(timezone)
    except Exception:
        raise ValueError(
            f"unknown timezone {timezone!r}; use an IANA name like 'Asia/Tokyo'"
        ) from None
    return datetime.now(zone).strftime("%Y-%m-%d %H:%M:%S %Z")


def _safe_path(directory: str) -> Path:
    """Resolve a model-supplied path, refusing anything outside the project."""
    root = Path.cwd().resolve()
    target = (root / directory).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"{directory!r} is outside the project directory")
    return target


@tool
def list_files(directory: str = ".") -> str:
    """List the files in a directory inside the project, with their sizes in bytes.

    Args:
        directory: Path relative to the project root. Defaults to the root.
    """
    target = _safe_path(directory)
    if not target.is_dir():
        raise ValueError(f"{directory!r} is not a directory")

    entries = sorted(p for p in target.iterdir() if not p.name.startswith("."))
    if not entries:
        return f"{directory} is empty"

    lines = [
        f"{p.name}/" if p.is_dir() else f"{p.name} ({p.stat().st_size} bytes)"
        for p in entries
    ]
    return "\n".join(lines)


# One fixed file. The name never comes from model output, so there is nothing
# for the model to redirect.
NOTES_FILE = Path("notes.txt")


@tool
def save_note(text: str) -> str:
    """Write a new note to storage. Use this ONLY when the user gives you
    something to remember. Never use it to look up existing notes, and never
    call it with empty text.

    Args:
        text: The exact note to store, taken from what the user said.
    """
    if not text.strip():
        raise ValueError("refusing to save an empty note")
    with NOTES_FILE.open("a", encoding="utf-8") as fh:
        fh.write(f"{datetime.now():%Y-%m-%d %H:%M} | {text.strip()}{os.linesep}")
    return "Note saved."


@tool
def read_notes() -> str:
    """Retrieve the notes already stored. Use this ONLY when the user asks what
    notes exist or what they saved earlier. It writes nothing."""
    if not NOTES_FILE.exists():
        return "No notes saved yet."
    content = NOTES_FILE.read_text(encoding="utf-8").strip()
    return content or "No notes saved yet."
