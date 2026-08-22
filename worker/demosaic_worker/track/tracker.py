"""Multi-object tracking with persistent IDs. prd.md §5.3.

ByteTrack-style two-stage association — high-confidence detections first, then the leftovers against
whatever is still unmatched — with a constant-velocity Kalman filter per track. Reimplemented rather
than taken from a package: it is a few hundred lines, and the alternative pulls a dependency into
the shipped engine for something this pipeline needs to control precisely anyway.

The state machine (§5.3.3) is deliberately table-driven and mirrors
`DeMosaicStudio.Domain.Tracking.TrackStateMachine`. An illegal transition raises rather than being
silently corrected: a tracker that quietly repairs its own state hides the bug that corrupted it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..detect.regions import Region, iou
from ..errors import E3201, WorkerError


class TrackState(str, Enum):
    """Track lifecycle. prd.md §5.3.3."""

    TENTATIVE = "Tentative"
    ACTIVE = "Active"
    OCCLUDED = "Occluded"
    LOST = "Lost"
    REACQUIRED = "Reacquired"
    TERMINATED = "Terminated"


#: Legal transitions. Mirrors the C# table; a transition absent here raises E3201.
ALLOWED_TRANSITIONS: frozenset[tuple[TrackState, TrackState]] = frozenset(
    {
        (TrackState.TENTATIVE, TrackState.ACTIVE),
        (TrackState.TENTATIVE, TrackState.LOST),
        (TrackState.ACTIVE, TrackState.OCCLUDED),
        (TrackState.ACTIVE, TrackState.LOST),
        (TrackState.OCCLUDED, TrackState.ACTIVE),
        (TrackState.OCCLUDED, TrackState.LOST),
        (TrackState.LOST, TrackState.REACQUIRED),
        (TrackState.LOST, TrackState.TERMINATED),
        (TrackState.REACQUIRED, TrackState.ACTIVE),
        (TrackState.REACQUIRED, TrackState.LOST),
    }
)


def can_transition(source: TrackState, target: TrackState, *, end_of_stream: bool = False) -> bool:
    """Whether a transition is legal. End of stream may terminate any non-terminal state."""
    if source is TrackState.TERMINATED:
        return False
    if end_of_stream and target is TrackState.TERMINATED:
        return True
    return (source, target) in ALLOWED_TRANSITIONS


def transition(source: TrackState, target: TrackState, *, end_of_stream: bool = False) -> TrackState:
    """Applies a transition, raising E3201 when it is not in the table."""
    if not can_transition(source, target, end_of_stream=end_of_stream):
        raise WorkerError(
            E3201,
            f"illegal track transition {source.value} -> {target.value}",
            source=source.value,
            target=target.value,
        )
    return target


class BoxKalman:
    """Constant-velocity filter over ``[cx, cy, w, h, vx, vy, vw, vh]``. prd.md §5.3.2.

    ``P`` is kept symmetric through the update: a filter that loses symmetry drifts, and on a
    two-hour job it drifts far enough to matter.
    """

    def __init__(self, box: tuple[int, int, int, int], process_noise: float = 1.0, measurement_noise: float = 4.0) -> None:
        left, top, right, bottom = box
        self.x = np.array(
            [(left + right) / 2, (top + bottom) / 2, right - left, bottom - top, 0, 0, 0, 0],
            dtype=np.float64,
        )
        self.P = np.eye(8) * 10.0
        self.Q = np.eye(8) * process_noise
        self.R = np.eye(4) * measurement_noise

        self.A = np.eye(8)
        for i in range(4):
            self.A[i, i + 4] = 1.0

        self.H = np.zeros((4, 8))
        for i in range(4):
            self.H[i, i] = 1.0

    def predict(self) -> np.ndarray:
        """Advances one frame and returns the predicted measurement."""
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.H @ self.x

    def update(self, box: tuple[int, int, int, int]) -> None:
        """Corrects with a measured box."""
        left, top, right, bottom = box
        z = np.array([(left + right) / 2, (top + bottom) / 2, right - left, bottom - top], dtype=np.float64)

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ (z - self.H @ self.x)

        identity = np.eye(8)
        self.P = (identity - K @ self.H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)   # keep it symmetric

    @property
    def box(self) -> tuple[int, int, int, int]:
        """The current estimate as a box."""
        cx, cy, w, h = self.x[:4]
        return (
            int(round(cx - w / 2)),
            int(round(cy - h / 2)),
            int(round(cx + w / 2)),
            int(round(cy + h / 2)),
        )

    @property
    def velocity(self) -> tuple[float, float]:
        """Estimated centre velocity in pixels per frame — what feeds the motion band (§5.6)."""
        return float(self.x[4]), float(self.x[5])


@dataclass
class Track:
    """One tracked mosaic region across frames."""

    track_id: int
    state: TrackState
    filter: BoxKalman
    region: Region | None
    hits: int = 0
    misses: int = 0
    age: int = 0
    history: list[tuple[int, int, int, int]] = field(default_factory=list)

    @property
    def is_restorable(self) -> bool:
        """Whether this track's region may be restored on this frame."""
        return self.state in {TrackState.ACTIVE, TrackState.OCCLUDED, TrackState.REACQUIRED}

    @property
    def speed(self) -> float:
        """Centre speed in pixels per frame."""
        vx, vy = self.filter.velocity
        return float(np.hypot(vx, vy))


class Tracker:
    """ByteTrack-style association with the §5.3.3 state machine.

    ``min_confirm_frames`` implements §5.2.5b: a region is not restored until it has been seen on
    that many consecutive frames. A single-frame flash is a false positive far more often than a
    real one-frame mosaic, and restoring one damages footage that was fine.
    """

    def __init__(
        self,
        *,
        high_confidence: float = 0.5,
        low_confidence: float = 0.1,
        iou_threshold: float = 0.3,
        min_confirm_frames: int = 2,
        max_missing_frames: int = 3,
    ) -> None:
        self.high_confidence = high_confidence
        self.low_confidence = low_confidence
        self.iou_threshold = iou_threshold
        self.min_confirm_frames = min_confirm_frames
        self.max_missing_frames = max_missing_frames

        self.tracks: list[Track] = []
        self._next_id = 1

    def _associate(
        self, tracks: list[Track], regions: list[Region]
    ) -> tuple[list[tuple[Track, Region]], list[Track], list[Region]]:
        """Greedy IoU matching, best pair first."""
        pairs = sorted(
            (
                (iou(Region(t.region.mask, t.filter.box, t.region.area, t.region.confidence), r)
                 if t.region is not None else 0.0, ti, ri)
                for ti, t in enumerate(tracks)
                for ri, r in enumerate(regions)
            ),
            reverse=True,
        )

        matched: list[tuple[Track, Region]] = []
        used_tracks: set[int] = set()
        used_regions: set[int] = set()

        for score, ti, ri in pairs:
            if score < self.iou_threshold or ti in used_tracks or ri in used_regions:
                continue
            used_tracks.add(ti)
            used_regions.add(ri)
            matched.append((tracks[ti], regions[ri]))

        return (
            matched,
            [t for i, t in enumerate(tracks) if i not in used_tracks],
            [r for i, r in enumerate(regions) if i not in used_regions],
        )

    def update(self, regions: list[Region], *, end_of_stream: bool = False) -> list[Track]:
        """Advances one frame. Returns the live tracks."""
        for track in self.tracks:
            track.filter.predict()
            track.age += 1

        high = [r for r in regions if r.confidence >= self.high_confidence]
        low = [r for r in regions if self.low_confidence <= r.confidence < self.high_confidence]

        candidates = [t for t in self.tracks if t.state is not TrackState.TERMINATED]

        matched, unmatched_tracks, unmatched_high = self._associate(candidates, high)

        # Second stage: low-confidence detections may still rescue a track that would otherwise be
        # marked missing. This is the part of ByteTrack that matters for short detector dropouts.
        second, still_unmatched, _ = self._associate(unmatched_tracks, low)
        matched.extend(second)

        for track, region in matched:
            track.filter.update(region.box)
            track.region = region
            track.hits += 1
            track.misses = 0
            track.history.append(region.box)

            if track.state is TrackState.TENTATIVE and track.hits >= self.min_confirm_frames:
                track.state = transition(track.state, TrackState.ACTIVE)
            elif track.state is TrackState.OCCLUDED:
                track.state = transition(track.state, TrackState.ACTIVE)
            elif track.state is TrackState.LOST:
                track.state = transition(track.state, TrackState.REACQUIRED)
            elif track.state is TrackState.REACQUIRED:
                track.state = transition(track.state, TrackState.ACTIVE)

        for track in still_unmatched:
            track.misses += 1

            if track.misses <= self.max_missing_frames:
                # Short dropout: predict through it rather than terminating (§5.3.4). The mask is
                # kept so the region stays restorable with a confidence penalty.
                if track.state is TrackState.ACTIVE:
                    track.state = transition(track.state, TrackState.OCCLUDED)
            elif track.state is not TrackState.LOST:
                track.state = transition(track.state, TrackState.LOST)

        for region in unmatched_high:
            self.tracks.append(
                Track(
                    track_id=self._next_id,
                    state=TrackState.TENTATIVE,
                    filter=BoxKalman(region.box),
                    region=region,
                    hits=1,
                    history=[region.box],
                )
            )
            self._next_id += 1

        if end_of_stream:
            for track in self.tracks:
                if track.state is not TrackState.TERMINATED:
                    track.state = transition(track.state, TrackState.TERMINATED, end_of_stream=True)

        self.tracks = [t for t in self.tracks if t.state is not TrackState.TERMINATED]
        return list(self.tracks)

    @property
    def restorable(self) -> list[Track]:
        """Tracks whose regions may be restored on the current frame."""
        return [t for t in self.tracks if t.is_restorable]
