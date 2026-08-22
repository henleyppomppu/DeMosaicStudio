"""Tracking. prd.md §5.3."""

from __future__ import annotations

import numpy as np
import pytest

from demosaic_worker.detect.regions import Region
from demosaic_worker.errors import WorkerError
from demosaic_worker.track.tracker import (
    ALLOWED_TRANSITIONS,
    BoxKalman,
    TrackState,
    Tracker,
    can_transition,
    transition,
)


def _region(left: int, top: int, size: int = 40, confidence: float = 0.9) -> Region:
    mask = np.zeros((256, 256), dtype=bool)
    mask[top : top + size, left : left + size] = True
    return Region(mask, (left, top, left + size, top + size), size * size, confidence)


# --- state machine ------------------------------------------------------------------------------


def test_every_transition_in_the_table_is_allowed() -> None:
    for source, target in ALLOWED_TRANSITIONS:
        assert can_transition(source, target)
        assert transition(source, target) is target


def test_every_transition_absent_from_the_table_raises() -> None:
    """A tracker that quietly repairs its own state hides the bug that corrupted it."""
    for source in TrackState:
        for target in TrackState:
            if (source, target) in ALLOWED_TRANSITIONS:
                continue

            assert not can_transition(source, target)

            with pytest.raises(WorkerError) as caught:
                transition(source, target)
            assert caught.value.code.code == "E3201"


def test_nothing_leaves_the_terminated_state() -> None:
    for target in TrackState:
        assert not can_transition(TrackState.TERMINATED, target)
        assert not can_transition(TrackState.TERMINATED, target, end_of_stream=True)


def test_terminated_is_reachable_from_lost_or_at_end_of_stream_only() -> None:
    assert can_transition(TrackState.LOST, TrackState.TERMINATED)
    assert not can_transition(TrackState.ACTIVE, TrackState.TERMINATED)
    assert can_transition(TrackState.ACTIVE, TrackState.TERMINATED, end_of_stream=True)


# --- Kalman -------------------------------------------------------------------------------------


def test_the_filter_tracks_a_constant_velocity_box() -> None:
    kalman = BoxKalman((10, 10, 50, 50))

    for step in range(1, 12):
        kalman.predict()
        kalman.update((10 + 4 * step, 10, 50 + 4 * step, 50))

    left, _, right, _ = kalman.box
    assert left == pytest.approx(10 + 4 * 11, abs=3)
    assert right - left == pytest.approx(40, abs=3)

    vx, vy = kalman.velocity
    assert vx == pytest.approx(4.0, abs=1.0)
    assert abs(vy) < 1.0


def test_the_covariance_stays_symmetric() -> None:
    """An asymmetric P drifts, and on a two-hour job it drifts far enough to matter."""
    rng = np.random.default_rng(3)
    kalman = BoxKalman((0, 0, 20, 20))

    for _ in range(500):
        kalman.predict()
        jitter = rng.integers(-3, 4, size=4)
        kalman.update((int(jitter[0]), int(jitter[1]), 20 + int(jitter[2]), 20 + int(jitter[3])))

    assert np.allclose(kalman.P, kalman.P.T)
    assert np.all(np.isfinite(kalman.P))


# --- association --------------------------------------------------------------------------------


def test_a_steady_detection_is_confirmed_and_keeps_its_id() -> None:
    tracker = Tracker(min_confirm_frames=2)

    tracker.update([_region(10, 10)])
    assert tracker.tracks[0].state is TrackState.TENTATIVE
    assert not tracker.restorable, "an unconfirmed region must not be restored (§5.2.5b)"

    tracker.update([_region(12, 10)])
    assert tracker.tracks[0].state is TrackState.ACTIVE

    first_id = tracker.tracks[0].track_id
    tracker.update([_region(14, 10)])

    assert tracker.tracks[0].track_id == first_id
    assert tracker.restorable


def test_two_regions_get_two_ids() -> None:
    tracker = Tracker(min_confirm_frames=1)

    tracker.update([_region(10, 10), _region(150, 150)])
    tracker.update([_region(12, 10), _region(152, 150)])

    assert len({t.track_id for t in tracker.tracks}) == 2


def test_a_short_dropout_is_survived(monkeypatch=None) -> None:
    """prd.md §5.3.4 — predict through a brief detector miss rather than terminating."""
    tracker = Tracker(min_confirm_frames=1, max_missing_frames=3)

    tracker.update([_region(10, 10)])
    tracker.update([_region(12, 10)])
    assert tracker.tracks[0].state is TrackState.ACTIVE

    for _ in range(3):
        tracker.update([])

    assert tracker.tracks[0].state is TrackState.OCCLUDED
    assert tracker.tracks[0].is_restorable, "an occluded track stays restorable with a penalty"


def test_a_long_dropout_loses_the_track() -> None:
    tracker = Tracker(min_confirm_frames=1, max_missing_frames=3)

    tracker.update([_region(10, 10)])
    tracker.update([_region(12, 10)])

    for _ in range(5):
        tracker.update([])

    assert tracker.tracks[0].state is TrackState.LOST
    assert not tracker.tracks[0].is_restorable


def test_a_low_confidence_detection_can_rescue_a_track() -> None:
    """The second association stage — what ByteTrack is actually for."""
    tracker = Tracker(min_confirm_frames=1, high_confidence=0.5, low_confidence=0.1)

    tracker.update([_region(10, 10, confidence=0.9)])
    tracker.update([_region(12, 10, confidence=0.9)])

    tracker.update([_region(14, 10, confidence=0.25)])

    assert tracker.tracks[0].state is TrackState.ACTIVE
    assert tracker.tracks[0].misses == 0


def test_a_low_confidence_detection_does_not_start_a_track() -> None:
    """Otherwise every faint false positive would become a track and then a restoration."""
    tracker = Tracker(min_confirm_frames=1, high_confidence=0.5, low_confidence=0.1)

    tracker.update([_region(10, 10, confidence=0.2)])

    assert tracker.tracks == []


def test_end_of_stream_terminates_everything() -> None:
    tracker = Tracker(min_confirm_frames=1)
    tracker.update([_region(10, 10)])
    tracker.update([_region(12, 10)])

    tracker.update([], end_of_stream=True)

    assert tracker.tracks == []


def test_speed_reflects_the_motion_the_window_policy_will_see() -> None:
    tracker = Tracker(min_confirm_frames=1)

    for step in range(10):
        tracker.update([_region(10 + 5 * step, 10)])

    assert tracker.tracks[0].speed == pytest.approx(5.0, abs=1.5)
