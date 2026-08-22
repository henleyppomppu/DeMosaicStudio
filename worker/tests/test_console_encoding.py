"""Console output stays ASCII. AGENTS.md, CLAUDE.md section 4.

The development machine's console is cp949. Anything a script *prints* — including argparse's help,
which is the module docstring — raises UnicodeEncodeError there if it contains a section mark, an
em dash, or any other non-ASCII character.

This trap has been sprung three times in this repository: twice mid-measurement in
`eval_endtoend.py` (the data survived only because it had already been written to JSON) and once in
`run_job.py --help`. Each time the file itself was fine and the *output* was not, so no syntax check
or import test could catch it.

Comments and docstrings that are never printed may use anything. The rule applies to what reaches
stdout.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest


def _repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "fixtures").is_dir():
            return candidate
    raise AssertionError("no 'fixtures' directory above this test file")


REPO = _repository_root()
SCRIPTS = sorted((REPO / "scripts").glob("*.py"))


def _printed_strings(source: str) -> list[tuple[int, str]]:
    """Every string literal that reaches a print() call, with its line number.

    Walks the AST rather than scanning text: a section mark inside a comment or a non-printed
    docstring is harmless, and flagging those would make the rule annoying enough to be disabled.
    """
    tree = ast.parse(source)
    found: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        target = node.func
        name = getattr(target, "id", None) or getattr(target, "attr", None)
        if name not in {"print", "write"}:
            continue

        for argument in ast.walk(node):
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                found.append((argument.lineno, argument.value))

    return found


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_printed_strings_are_ascii(script: Path) -> None:
    source = script.read_text(encoding="utf-8")

    offenders = [
        (line, text) for line, text in _printed_strings(source) if not text.isascii()
    ]

    assert not offenders, "\n".join(
        f"{script.name}:{line} prints non-ASCII: {text[:70]!r}" for line, text in offenders
    )


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_the_module_docstring_is_ascii_when_argparse_prints_it(script: Path) -> None:
    """A docstring passed to ArgumentParser(description=...) becomes console output."""
    source = script.read_text(encoding="utf-8")

    if "description=__doc__" not in source:
        pytest.skip("docstring is not used as help text")

    docstring = ast.get_docstring(ast.parse(source)) or ""

    assert docstring.isascii(), (
        f"{script.name}'s docstring is argparse --help text and contains non-ASCII: "
        f"{sorted({c for c in docstring if not c.isascii()})}"
    )


def test_the_cli_help_survives_a_cp949_console() -> None:
    """The end-to-end version of the rule: run --help with stdout forced to cp949."""
    script = REPO / "scripts" / "run_job.py"
    if not script.exists():
        pytest.skip("run_job.py missing")

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        encoding="cp949",
        errors="strict",
        env={"PYTHONIOENCODING": "cp949", "SYSTEMROOT": r"C:\Windows", "PATH": ""},
        cwd=str(REPO),
    )

    assert result.returncode == 0, result.stderr[-800:]
    assert "usage:" in result.stdout
