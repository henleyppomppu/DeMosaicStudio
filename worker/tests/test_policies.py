"""Worker half of the policy parity lock, plus the behaviour the fixture cannot express.

prd.md §5.6, §5.6.1, §5.8, §5.8.1, §13.4. The host half is `PolicyParityTests.cs`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from demosaic_worker.analyze.motion import MotionBand
from demosaic_worker.analyze.profile import GridAnchor
from demosaic_worker.policies import (
    MULTI_FRAME_BANDS,
    WINDOW_BY_MOTION,
    ConfidenceGate,
    ConfidenceSmoother,
    QualityPreset,
    RestorationPath,
    RouteInputs,
    RouteReason,
    WindowDecision,
    WindowReductionReason,
    decide_window,
    route,
)


def _repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "fixtures").is_dir():
            return candidate
    raise AssertionError("no 'fixtures' directory above this test file")


FIXTURE = _repository_root() / "fixtures" / "parity" / "policies.json"


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


# --- parity ---------------------------------------------------------------------------------------


def test_the_window_fixture_reproduces_from_this_implementation() -> None:
    for case in _fixture()["windowCases"]:
        decision = decide_window(
            setting=case["setting"],
            preset=QualityPreset(case["preset"]),
            motion_pixels_per_frame=case["motion"],
            anchor=GridAnchor(case["anchor"]),
            same_scene_frames=case["sameSceneFrames"],
            stream_frames=case["streamFrames"],
            vram_max_window=case["vramMaxWindow"],
        )
        assert decision.effective == case["effective"], case
        assert decision.reason.value == case["reason"], case


def test_the_route_fixture_reproduces_from_this_implementation() -> None:
    for case in _fixture()["routeCases"]:
        decision = route(
            RouteInputs(
                has_region=case["hasRegion"],
                region_area=case["regionArea"],
                min_region_area=256,
                is_confirmed=case["isConfirmed"],
                user_disabled=case["userDisabled"],
                withheld_by_confidence_gate=case["withheldByConfidenceGate"],
                degradation_chain_exhausted=case["degradationChainExhausted"],
                anchor=GridAnchor(case["anchor"]),
                motion_pixels_per_frame=case["motion"],
                window=WindowDecision(
                    case["window"], 5, WindowReductionReason(case["windowReason"])
                ),
                valid_aligned_neighbours=case["validAlignedNeighbours"],
                mean_alignment_confidence=case["meanAlignmentConfidence"],
                align_conf_min=0.35,
                occlusion_invalidated_neighbours=case["occlusionInvalidatedNeighbours"],
            )
        )
        assert decision.path.value == case["path"], case
        assert decision.reason.value == case["reason"], case


def test_the_fixture_covers_every_routing_reason() -> None:
    """A fixture that missed a branch would pass while that branch drifted freely."""
    covered = {c["reason"] for c in _fixture()["routeCases"]}
    assert {r.value for r in RouteReason} - covered == set()


# --- the measured table (D-16) ---------------------------------------------------------------------


def test_the_window_table_matches_what_was_measured() -> None:
    """docs/phase2-alignment-report.md §3. Changing these means re-measuring, not re-guessing."""
    assert WINDOW_BY_MOTION[MotionBand.STATIC] == 1
    assert WINDOW_BY_MOTION[MotionBand.SLOW] == 3
    assert WINDOW_BY_MOTION[MotionBand.MEDIUM] == 3
    assert WINDOW_BY_MOTION[MotionBand.FAST] == 1


def test_only_slow_and_medium_permit_multi_frame() -> None:
    assert MULTI_FRAME_BANDS == frozenset({MotionBand.SLOW, MotionBand.MEDIUM})


@pytest.mark.parametrize("motion", [0.05, 20.0])
def test_motion_outside_the_operating_window_routes_to_single_frame(motion: float) -> None:
    decision = route(RouteInputs(motion_pixels_per_frame=motion))

    assert decision.path is RestorationPath.SINGLE_FRAME
    assert decision.reason is RouteReason.MOTION_OUTSIDE_OPERATING_WINDOW


def test_a_fixed_window_does_not_override_the_motion_gate() -> None:
    """The override replaces the motion *choice*, never the motion *gate* (§5.6.1)."""
    decision = decide_window(
        setting=9,
        preset=QualityPreset.QUALITY,
        motion_pixels_per_frame=20.0,
        anchor=GridAnchor.SCREEN,
        same_scene_frames=9,
        stream_frames=9,
    )

    assert decision.effective == 1
    assert decision.reason is WindowReductionReason.MOTION_BAND
    assert decision.was_reduced


# --- confidence gate (§5.8.1) -----------------------------------------------------------------------


def test_the_default_gate_withholds_nothing() -> None:
    gate = ConfidenceGate(0.0)

    assert gate.is_disabled
    assert not any(gate.should_withhold(1, 0.0) for _ in range(50))


def test_a_track_starts_gated() -> None:
    """Restoration is an intervention: it takes evidence to begin, not evidence to stop.

    This used to be the other way round, and the asymmetry was invisible until a gate set above
    every reachable confidence was measured still letting two frames per track through — one short
    of the hysteresis window, every time a track appeared.
    """
    gate = ConfidenceGate(0.40)

    assert gate.should_withhold(1, 0.1), "the very first frame of a track is withheld"
    assert gate.gated_track_count == 1


def test_sustained_low_confidence_stays_withheld() -> None:
    gate = ConfidenceGate(0.40)

    for _ in range(5):
        assert gate.should_withhold(1, 0.1)
    assert gate.gated_track_count == 1


def test_an_oscillating_signal_does_not_flip_the_gate() -> None:
    """The reason the decision is per track rather than per frame (§5.8.1 R-8.1c).

    0.41 clears the threshold but not the release margin, so a signal chattering across the
    threshold never opens the gate — and never chatters the picture either.
    """
    gate = ConfidenceGate(0.40)

    transitions = 0
    previous = True  # a track starts gated
    for frame in range(200):
        withheld = gate.should_withhold(1, 0.39 if frame % 2 == 0 else 0.41)
        if withheld != previous:
            transitions += 1
            previous = withheld

    assert transitions == 0


def test_a_confident_track_opens_the_gate_and_stays_open() -> None:
    """The cost of starting gated: a genuinely confident track waits out the hysteresis window."""
    gate = ConfidenceGate(0.40)

    assert gate.should_withhold(1, 0.9)
    assert gate.should_withhold(1, 0.9)
    assert not gate.should_withhold(1, 0.9), "three frames above the margin release it"

    for _ in range(20):
        assert not gate.should_withhold(1, 0.9)


def test_a_sustained_recovery_releases_the_gate_once() -> None:
    gate = ConfidenceGate(0.40)

    for _ in range(5):
        gate.should_withhold(1, 0.1)

    assert gate.should_withhold(1, 0.40), "one frame at the threshold is not three"
    assert gate.should_withhold(1, 0.80)
    assert not gate.should_withhold(1, 0.80), "three consecutive frames at or above it release it"

    # The margin now guards the closing side: a dip that stays within it does not re-close.
    assert not gate.should_withhold(1, 0.36)
    assert not gate.should_withhold(1, 0.36)
    assert not gate.should_withhold(1, 0.36)


def test_gate_state_is_per_track() -> None:
    gate = ConfidenceGate(0.40)

    for _ in range(5):
        gate.should_withhold(1, 0.05)
        gate.should_withhold(2, 0.95)

    assert gate.should_withhold(1, 0.05)
    assert not gate.should_withhold(2, 0.95)
    assert gate.gated_track_count == 1


def test_forgetting_a_track_drops_its_state() -> None:
    gate = ConfidenceGate(0.40)
    for _ in range(5):
        gate.should_withhold(1, 0.05)

    assert gate.gated_track_count == 1
    gate.forget(1)
    assert gate.gated_track_count == 0


def test_invalid_gate_parameters_are_rejected() -> None:
    with pytest.raises(ValueError):
        ConfidenceGate(0.5, hysteresis_frames=0)
    with pytest.raises(ValueError):
        ConfidenceGate(0.5, release_margin=-0.1)


# ------------------------------------------------------------------------------------------
# Confidence-gate parity. The gate was mirrored in both languages and locked by nothing, and it
# turned out to have a hole in both: a track started open, so a gate set above every reachable
# confidence still let `hysteresis - 1` frames through per track.
# ------------------------------------------------------------------------------------------


def test_the_confidence_gate_matches_the_parity_fixture() -> None:
    for case in _fixture()["confidenceGateCases"]:
        gate = ConfidenceGate(case["threshold"])
        got = [gate.should_withhold(1, c) for c in case["confidences"]]

        assert got == case["withheld"], case["description"]
        assert gate.gated_track_count == case["gatedTrackCount"], case["description"]


def test_the_confidence_gate_keeps_track_state_apart() -> None:
    case = _fixture()["confidenceGateInterleaved"]
    low, high = case["confidences"]

    gate = ConfidenceGate(case["threshold"])
    got = [[gate.should_withhold(1, low), gate.should_withhold(2, high)]
           for _ in case["withheld"]]

    assert got == case["withheld"], case["description"]
    assert gate.gated_track_count == case["gatedTrackCount"]


def test_the_confidence_smoother_matches_the_parity_fixture() -> None:
    for case in _fixture()["confidenceSmootherCases"]:
        smoother = ConfidenceSmoother(case["window"])
        got = [smoother.update(1, c) for c in case["confidences"]]

        assert got == pytest.approx(case["smoothed"], abs=1e-9), case["description"]


def test_the_confidence_smoother_keeps_track_state_apart() -> None:
    case = _fixture()["confidenceSmootherInterleaved"]
    first, second = case["confidences"]

    smoother = ConfidenceSmoother(case["window"])
    got = [[smoother.update(1, first), smoother.update(2, second)] for _ in case["smoothed"]]

    # pytest.approx does not compare nested lists; flatten both sides.
    assert [v for row in got for v in row] == pytest.approx(
        [v for row in case["smoothed"] for v in row], abs=1e-9
    ), case["description"]


def test_the_smoother_window_defaults_to_the_gate_hysteresis() -> None:
    """The time constant is the gate's own window, not a tuned number.

    If either moves independently the gate starts reasoning over a span the signal was not averaged
    over, which is the situation this pair was introduced to end.
    """
    assert ConfidenceSmoother().alpha == pytest.approx(1.0 / ConfidenceGate.HYSTERESIS_FRAMES)
