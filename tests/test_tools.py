"""Tests for the registry, the generated schemas, and the safety boundaries."""

from __future__ import annotations

import pytest

from agent import tools


def test_schema_is_generated_from_hints_and_docstring():
    schema = tools.REGISTRY["calculator"]["schema"]["function"]
    assert schema["name"] == "calculator"
    assert "calculation" in schema["description"]
    assert schema["parameters"]["properties"]["expression"]["type"] == "string"
    # A description the model can act on, taken from the Args: block.
    assert "Arithmetic only" in schema["parameters"]["properties"]["expression"]["description"]


def test_arguments_without_defaults_are_required():
    calculator = tools.REGISTRY["calculator"]["schema"]["function"]["parameters"]
    clock = tools.REGISTRY["current_time"]["schema"]["function"]["parameters"]
    assert calculator["required"] == ["expression"]
    assert clock["required"] == []  # timezone has a default


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("0.18 * 4250", "765"),
        ("2 + 3 * 4", "14"),
        ("(12 + 7) ** 2", "361"),
        ("10 / 4", "2.5"),
        ("-5 + 2", "-3"),
    ],
)
def test_calculator_arithmetic(expression, expected):
    assert tools.calculator(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo pwned')",
        "open('/etc/passwd').read()",
        "1 / 0",
        "9 ** 9 ** 9",
        "'a' * 3",
    ],
)
def test_calculator_refuses_anything_but_arithmetic(expression):
    with pytest.raises((ValueError, SyntaxError)):
        tools.calculator(expression)


def test_list_files_refuses_paths_outside_the_project():
    with pytest.raises(ValueError, match="outside the project"):
        tools.list_files("../../..")


def test_call_reports_an_unknown_tool_instead_of_raising():
    result = tools.call("teleport", {})
    assert result.startswith("Error: unknown tool 'teleport'")
    assert "calculator" in result  # tells the model what it could use instead


def test_call_reports_bad_arguments_instead_of_raising():
    assert "wrong arguments" in tools.call("calculator", {"wrong_name": "1+1"})


def test_call_turns_a_raising_tool_into_a_string():
    result = tools.call("current_time", {"timezone": "Mars/Olympus_Mons"})
    assert result.startswith("Error:")
    assert "Mars/Olympus_Mons" in result


def test_notes_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "NOTES_FILE", tmp_path / "notes.txt")
    assert tools.read_notes() == "No notes saved yet."
    tools.save_note("demo is on Friday")
    assert "demo is on Friday" in tools.read_notes()


def test_empty_note_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "NOTES_FILE", tmp_path / "notes.txt")
    assert tools.call("save_note", {"text": "   "}).startswith("Error:")
