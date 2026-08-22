"""Worker half of the error-code parity lock. prd.md §10, §13.4.

The host half lives in ``tests/DeMosaicStudio.Domain.Tests/ErrorCodeTests.cs`` and checks the same
fixture. Either implementation drifting turns one of the two red.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from demosaic_worker import errors


def _repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "fixtures").is_dir():
            return candidate
    raise AssertionError("no 'fixtures' directory above this test file")


def _fixture() -> list[dict[str, object]]:
    path = _repository_root() / "fixtures" / "parity" / "error_codes.json"
    return json.loads(path.read_text(encoding="utf-8"))["codes"]


def test_the_error_table_matches_the_parity_fixture_exactly() -> None:
    fixture = _fixture()

    fixture_codes = {entry["code"] for entry in fixture}
    table_codes = {code.code for code in errors.ALL}

    assert fixture_codes - table_codes == set(), "codes in the fixture but not in errors.py"
    assert table_codes - fixture_codes == set(), "codes in errors.py but not in the fixture"

    for entry in fixture:
        code = errors.get(str(entry["code"]))
        assert code.recoverable == entry["recoverable"], code.code
        assert code.severity.value == entry["severity"], code.code


def test_every_code_is_unique() -> None:
    codes = [c.code for c in errors.ALL]
    assert len(codes) == len(set(codes))


def test_warnings_are_never_recoverable() -> None:
    """prd.md §10.1: a warning is not a failure, so 'may the host retry it' does not apply."""
    for code in errors.ALL:
        if code.is_warning:
            assert not code.recoverable, code.code


def test_the_prefix_letter_matches_the_severity() -> None:
    for code in errors.ALL:
        expected = errors.Severity.WARNING if code.code.startswith("W") else errors.Severity.ERROR
        assert code.severity is expected, code.code


def test_an_unknown_code_raises_rather_than_returning_a_placeholder() -> None:
    assert errors.try_get("E0000") is None
    with pytest.raises(KeyError):
        errors.get("E0000")


def test_a_worker_error_carries_its_code_and_recoverability() -> None:
    error = errors.WorkerError(errors.E4002, "inference blew up", track_id=3)

    assert error.code is errors.E4002
    assert error.recoverable is True
    assert error.context == {"track_id": 3}
