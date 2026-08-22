"""Scene cut detection. prd.md §5.12.

The cut and flash cases use **real frames from two different films' shots**, not synthetic patterns.
An earlier version of this file used sinusoidal test images and the "different scene" pair scored a
structure distance of 0.17 — the two synthetic scenes had nearly the same gradient statistics, so
the test was measuring the fixture rather than the detector. Real shots differ in ways a generated
pattern does not.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from demosaic_worker.scene.cuts import (
    classify_pair,
    detect_cuts,
    histogram_distance,
    same_scene_span,
    structure_distance,
)


def _repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "fixtures").is_dir():
            return candidate
    raise AssertionError("no 'fixtures' directory above this test file")


REPO = _repository_root()
MANIFEST = REPO / "training" / "datasets" / "clean-tos.manifest.json"
CORPUS = REPO / "training" / "datasets" / "clean"

corpus = pytest.mark.skipif(
    not MANIFEST.exists(), reason="corpus missing; run scripts/build_corpus.py"
)


def _frames(clip: str, count: int = 6) -> list[np.ndarray]:
    import av

    out: list[np.ndarray] = []
    with av.open(str(CORPUS / clip)) as container:
        for frame in container.decode(container.streams.video[0]):
            plane = frame.to_ndarray(format="gray").astype(np.float64)
            out.append(plane[::4, ::4])
            if len(out) >= count:
                break
    return out


@pytest.fixture(scope="module")
def shots() -> tuple[list[np.ndarray], list[np.ndarray]]:
    names = [c["name"] for c in json.loads(MANIFEST.read_text(encoding="utf-8"))["clips"]]
    return _frames(names[0]), _frames(names[12])


# --- distances --------------------------------------------------------------------------------


def test_histogram_distance_is_zero_for_identical_frames() -> None:
    frame = np.linspace(0, 255, 64 * 64).reshape(64, 64)

    assert histogram_distance(frame, frame) == pytest.approx(0.0, abs=1e-9)


def test_structure_distance_is_zero_for_identical_frames() -> None:
    frame = np.linspace(0, 255, 64 * 64).reshape(64, 64)

    assert structure_distance(frame, frame) == pytest.approx(0.0, abs=1e-9)


def test_distances_are_bounded() -> None:
    a = np.zeros((32, 32))
    b = np.full((32, 32), 255.0)

    assert 0.0 <= histogram_distance(a, b) <= 1.0
    assert 0.0 <= structure_distance(a, b) <= 1.0


# --- classification, on real frames -------------------------------------------------------------


@corpus
def test_consecutive_frames_of_one_shot_are_not_a_cut(shots) -> None:
    first, _ = shots

    for index in range(len(first) - 1):
        change = classify_pair(first[index], first[index + 1], index + 1)
        assert not change.is_cut, f"frames {index} and {index + 1} of one shot classified as a cut"


@corpus
def test_frames_from_different_shots_are_a_cut(shots) -> None:
    first, second = shots

    change = classify_pair(first[0], second[0], 1)

    assert change.is_cut, (
        f"histogram={change.histogram:.3f} structure={change.structure:.3f} — "
        "two unrelated shots should exceed both thresholds"
    )
    assert change.resets_temporal_context


@corpus
def test_a_flash_is_not_a_cut(shots) -> None:
    """prd.md §5.12 — resetting temporal context for a flash discards good neighbours for nothing.

    A flash scales luminance without moving edges. Normalising each frame's gradients by its own
    peak leaves the structure distance small even as the histogram distance goes large, which is
    exactly the asymmetry the classifier keys on.
    """
    first, _ = shots
    flashed = np.clip(first[1] * 2.6 + 60, 0, 255)

    change = classify_pair(first[0], flashed, 1)

    assert change.histogram > 0.3, "a flash should look large to a histogram"
    assert change.is_flash
    assert not change.is_cut
    assert not change.resets_temporal_context


# --- sequences --------------------------------------------------------------------------------


@corpus
def test_detect_cuts_reports_one_verdict_per_boundary(shots) -> None:
    first, second = shots
    frames = first[:3] + second[:3]

    changes = detect_cuts(frames)

    assert len(changes) == len(frames) - 1
    assert [c.index for c in changes] == [1, 2, 3, 4, 5]
    assert changes[2].is_cut, "the boundary between the two shots"
    assert not changes[0].is_cut
    assert not changes[4].is_cut


@corpus
def test_same_scene_span_stops_at_a_cut(shots) -> None:
    """This is what bounds the temporal window (§5.6): reach to the nearest cut and no further."""
    first, second = shots
    frames = first[:3] + second[:3]
    changes = detect_cuts(frames)

    start, end = same_scene_span(changes, target=1, radius=4, total_frames=len(frames))

    assert start == 0
    assert end == 2, "the window must not reach across the cut before frame 3"


@corpus
def test_same_scene_span_never_runs_past_the_sequence(shots) -> None:
    """Regression: the span used to return negative indices, which told the window policy that more
    frames were available than existed and mislabelled the reduction as a stream boundary."""
    first, _ = shots
    changes = detect_cuts(first)

    start, end = same_scene_span(changes, target=0, radius=5, total_frames=len(first))

    assert start == 0
    assert end <= len(first) - 1


def test_same_scene_span_is_bounded_by_the_radius() -> None:
    changes = detect_cuts([np.full((16, 16), 100.0 + i) for i in range(12)])

    start, end = same_scene_span(changes, target=6, radius=2, total_frames=12)

    assert (start, end) == (4, 8)
