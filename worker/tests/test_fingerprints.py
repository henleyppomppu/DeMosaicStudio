"""Worker half of the fingerprint parity lock. prd.md §9.3, §13.4."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from demosaic_worker import fingerprints
from demosaic_worker.fingerprints import Scope


def _repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "fixtures").is_dir():
            return candidate
    raise AssertionError("no 'fixtures' directory above this test file")


def _cases() -> list[dict[str, Any]]:
    path = _repository_root() / "fixtures" / "parity" / "fingerprints.json"
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", _cases(), ids=lambda c: str(c["name"]))
def test_the_canonical_form_and_digest_match_the_host(case: dict[str, Any]) -> None:
    """Byte-for-byte agreement with the C# implementation.

    Canonical text is asserted before the digest on purpose: when this fails, the text says *what*
    differs, while two mismatched hashes say only *that* something does.
    """
    for scope in Scope:
        expected = case["expected"][scope.value]

        assert fingerprints.canonicalize(case["settings"], scope) == expected["canonical"], scope.value
        assert fingerprints.compute(case["settings"], scope) == expected["digest"], scope.value


def test_a_missing_key_fails_loudly() -> None:
    """A silently absent key would change the digest and quietly invalidate every checkpoint."""
    settings = {"encode": {"codec": "H265", "constantQuality": 18}}

    with pytest.raises(KeyError):
        fingerprints.canonicalize(settings, Scope.ENCODE)


def test_an_unknown_fingerprint_counts_as_changed() -> None:
    assert fingerprints.changed(None, "sha256:abc") is True
    assert fingerprints.changed("sha256:abc", None) is True
    assert fingerprints.changed(None, None) is True
    assert fingerprints.changed("sha256:abc", "sha256:abc") is False


def test_invalidation_cascades_top_down() -> None:
    recorded = {"detection": "a", "restoration": "b", "encode": "c"}

    assert fingerprints.invalidated(recorded, recorded) == set()

    assert fingerprints.invalidated(recorded, {**recorded, "detection": "z"}) == {"analysis", "video"}
    assert fingerprints.invalidated(recorded, {**recorded, "restoration": "z"}) == {"video"}
    assert fingerprints.invalidated(recorded, {**recorded, "encode": "z"}) == {"video"}


def test_the_canonical_form_is_sorted() -> None:
    case = _cases()[0]

    for scope in Scope:
        lines = fingerprints.canonicalize(case["settings"], scope).split("\n")
        assert lines == sorted(lines)
