"""Worker-side mirrors of the Domain policies. prd.md §13.4.

These are the *same rules* as `DeMosaicStudio.Domain.Policies`, implemented a second time because
both processes need them: the host to show the user what will happen, the worker to actually do it.
That duplication is deliberate and is the reason §13.4 exists — `fixtures/parity/policies.json`
locks the two together, so neither can drift without a red test on the other side.

The numbers here are not guesses. `docs/phase2-alignment-report.md` measured them, and D-16 records
why the earlier table (low motion → K of 7-9) was replaced.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .analyze.motion import MotionBand, classify
from .analyze.profile import GridAnchor


class QualityPreset(str, Enum):
    """User-facing preset. prd.md §15."""

    FAST = "Fast"
    BALANCED = "Balanced"
    QUALITY = "Quality"


class WindowReductionReason(str, Enum):
    """Why the effective window is smaller than requested. Warning W4103."""

    NONE = "None"
    SCENE_BOUNDARY = "SceneBoundary"
    STREAM_BOUNDARY = "StreamBoundary"
    OBJECT_ANCHORED_GRID = "ObjectAnchoredGrid"
    VRAM_PRESSURE = "VramPressure"
    MOTION_BAND = "MotionBand"


class RestorationPath(str, Enum):
    """The three paths. prd.md §5.8."""

    MULTI_FRAME = "MultiFrame"
    SINGLE_FRAME = "SingleFrame"
    PASS_THROUGH = "PassThrough"


class RouteReason(str, Enum):
    """Closed enum of routing reasons. A router that cannot explain itself cannot be debugged."""

    SUFFICIENT_TEMPORAL_EVIDENCE = "SufficientTemporalEvidence"
    SCENE_CUT_TRUNCATED_WINDOW = "SceneCutTruncatedWindow"
    POOR_ALIGNMENT = "PoorAlignment"
    SINGLE_VALID_FRAME = "SingleValidFrame"
    OCCLUSION_INVALIDATED_NEIGHBOURS = "OcclusionInvalidatedNeighbours"
    OBJECT_ANCHORED_GRID = "ObjectAnchoredGrid"
    MOTION_OUTSIDE_OPERATING_WINDOW = "MotionOutsideOperatingWindow"
    VRAM_FORCED_SINGLE_FRAME = "VramForcedSingleFrame"
    NO_REGION = "NoRegion"
    REGION_TOO_SMALL = "RegionTooSmall"
    REGION_UNCONFIRMED = "RegionUnconfirmed"
    USER_DISABLED = "UserDisabled"
    LOW_CONFIDENCE_GATE = "LowConfidenceGate"
    DEGRADATION_CHAIN_EXHAUSTED = "DegradationChainExhausted"


SINGLE_FRAME = 1

#: Preset ceiling on the window. D-13.
PRESET_MAX_WINDOW = {
    QualityPreset.FAST: 3,
    QualityPreset.BALANCED: 7,
    QualityPreset.QUALITY: 9,
}

#: Window by motion band. **Measured**, not assumed — D-16, docs/phase2-alignment-report.md §3.
#:
#: static and fast are 1 (multi-frame off) because measurement put them below single-frame even with
#: perfect alignment: static has no phase diversity to exploit, fast has no content correspondence
#: left to align.
WINDOW_BY_MOTION = {
    MotionBand.STATIC: 1,
    MotionBand.SLOW: 3,
    MotionBand.MEDIUM: 3,
    MotionBand.FAST: 1,
}

#: Bands where multi-frame is permitted at all. Everything else routes to single-frame (§5.8).
MULTI_FRAME_BANDS = frozenset({MotionBand.SLOW, MotionBand.MEDIUM})

#: Medium motion only qualifies when alignment is good; slow qualifies regardless.
MEDIUM_BAND_ALIGNMENT_MIN = 0.60

MIN_NEIGHBOURS_FOR_MULTI_FRAME = 2


@dataclass(frozen=True, slots=True)
class WindowDecision:
    """The window actually used, and why it differs from the request."""

    effective: int
    requested: int
    reason: WindowReductionReason

    @property
    def was_reduced(self) -> bool:
        """True when a safety rule reduced the window and W4103 should be emitted."""
        return self.effective < self.requested


def _odd_at_most(value: int, cap: int = 9) -> int:
    if value < SINGLE_FRAME:
        return SINGLE_FRAME
    clamped = min(value, cap)
    return clamped - 1 if clamped % 2 == 0 else clamped


def requested_window(
    setting: int | None,
    preset: QualityPreset,
    motion_pixels_per_frame: float,
) -> int:
    """The window asked for, before any safety reduction. prd.md §5.6, §5.6.1, D-16."""
    if setting is not None:
        return setting

    band = classify(motion_pixels_per_frame)
    return min(WINDOW_BY_MOTION[band], PRESET_MAX_WINDOW[preset])


def decide_window(
    *,
    setting: int | None,
    preset: QualityPreset,
    motion_pixels_per_frame: float,
    anchor: GridAnchor,
    same_scene_frames: int,
    stream_frames: int,
    vram_max_window: int = 9,
) -> WindowDecision:
    """Decides the window for one frame.

    A fixed ``setting`` replaces the **motion-based** choice only. Every safety reduction still
    applies on top of it — a user-forced window that ignored them would be a corruption and OOM
    lever rather than a quality control (§5.6.1).
    """
    requested = requested_window(setting, preset, motion_pixels_per_frame)
    band = classify(motion_pixels_per_frame)

    constraints = [
        (SINGLE_FRAME if anchor is GridAnchor.OBJECT else requested, WindowReductionReason.OBJECT_ANCHORED_GRID),
        (SINGLE_FRAME if band not in MULTI_FRAME_BANDS else requested, WindowReductionReason.MOTION_BAND),
        (_odd_at_most(same_scene_frames), WindowReductionReason.SCENE_BOUNDARY),
        (_odd_at_most(stream_frames), WindowReductionReason.STREAM_BOUNDARY),
        (_odd_at_most(vram_max_window), WindowReductionReason.VRAM_PRESSURE),
    ]

    effective = requested
    reason = WindowReductionReason.NONE

    for ceiling, candidate in constraints:
        if ceiling < effective:
            effective = ceiling
            reason = candidate

    return WindowDecision(max(effective, SINGLE_FRAME), requested, reason)


@dataclass(frozen=True, slots=True)
class RouteInputs:
    """Everything the router needs, as data. prd.md §5.8."""

    has_region: bool = True
    region_area: int = 4096
    min_region_area: int = 256
    is_confirmed: bool = True
    user_disabled: bool = False
    withheld_by_confidence_gate: bool = False
    degradation_chain_exhausted: bool = False
    anchor: GridAnchor = GridAnchor.SCREEN
    motion_pixels_per_frame: float = 0.5
    window: WindowDecision = WindowDecision(3, 3, WindowReductionReason.NONE)
    valid_aligned_neighbours: int = 2
    mean_alignment_confidence: float = 0.9
    align_conf_min: float = 0.35
    occlusion_invalidated_neighbours: bool = False


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """Which path runs, and why."""

    path: RestorationPath
    reason: RouteReason


def route(inputs: RouteInputs) -> RouteDecision:
    """Selects the restoration path. prd.md §5.8, with the v3.3 motion gate."""
    if not inputs.has_region:
        return RouteDecision(RestorationPath.PASS_THROUGH, RouteReason.NO_REGION)
    if inputs.user_disabled:
        return RouteDecision(RestorationPath.PASS_THROUGH, RouteReason.USER_DISABLED)
    if inputs.region_area < inputs.min_region_area:
        return RouteDecision(RestorationPath.PASS_THROUGH, RouteReason.REGION_TOO_SMALL)
    if not inputs.is_confirmed:
        return RouteDecision(RestorationPath.PASS_THROUGH, RouteReason.REGION_UNCONFIRMED)
    if inputs.degradation_chain_exhausted:
        return RouteDecision(RestorationPath.PASS_THROUGH, RouteReason.DEGRADATION_CHAIN_EXHAUSTED)
    if inputs.withheld_by_confidence_gate:
        return RouteDecision(RestorationPath.PASS_THROUGH, RouteReason.LOW_CONFIDENCE_GATE)

    if inputs.anchor is GridAnchor.OBJECT:
        return RouteDecision(RestorationPath.SINGLE_FRAME, RouteReason.OBJECT_ANCHORED_GRID)

    # v3.3 motion gate. Measured: outside this window a classical multi-frame solver scores below
    # single-frame, so attempting it is not merely wasteful but harmful (D-16).
    band = classify(inputs.motion_pixels_per_frame)
    if band not in MULTI_FRAME_BANDS:
        return RouteDecision(RestorationPath.SINGLE_FRAME, RouteReason.MOTION_OUTSIDE_OPERATING_WINDOW)
    if band is MotionBand.MEDIUM and inputs.mean_alignment_confidence < MEDIUM_BAND_ALIGNMENT_MIN:
        return RouteDecision(RestorationPath.SINGLE_FRAME, RouteReason.MOTION_OUTSIDE_OPERATING_WINDOW)

    if inputs.occlusion_invalidated_neighbours:
        return RouteDecision(RestorationPath.SINGLE_FRAME, RouteReason.OCCLUSION_INVALIDATED_NEIGHBOURS)

    if inputs.window.effective <= SINGLE_FRAME:
        reason = {
            WindowReductionReason.SCENE_BOUNDARY: RouteReason.SCENE_CUT_TRUNCATED_WINDOW,
            WindowReductionReason.VRAM_PRESSURE: RouteReason.VRAM_FORCED_SINGLE_FRAME,
            WindowReductionReason.OBJECT_ANCHORED_GRID: RouteReason.OBJECT_ANCHORED_GRID,
            WindowReductionReason.MOTION_BAND: RouteReason.MOTION_OUTSIDE_OPERATING_WINDOW,
        }.get(inputs.window.reason, RouteReason.SINGLE_VALID_FRAME)
        return RouteDecision(RestorationPath.SINGLE_FRAME, reason)

    if inputs.valid_aligned_neighbours < MIN_NEIGHBOURS_FOR_MULTI_FRAME:
        reason = (
            RouteReason.SCENE_CUT_TRUNCATED_WINDOW
            if inputs.window.reason is WindowReductionReason.SCENE_BOUNDARY
            else RouteReason.SINGLE_VALID_FRAME
        )
        return RouteDecision(RestorationPath.SINGLE_FRAME, reason)

    if inputs.mean_alignment_confidence < inputs.align_conf_min:
        return RouteDecision(RestorationPath.SINGLE_FRAME, RouteReason.POOR_ALIGNMENT)

    return RouteDecision(RestorationPath.MULTI_FRAME, RouteReason.SUFFICIENT_TEMPORAL_EVIDENCE)


class ConfidenceGate:
    """The ``minRestorationConfidence`` gate, per track with hysteresis. prd.md §5.8.1.

    Mirror of `DeMosaicStudio.Domain.Policies.ConfidenceGate`. The hysteresis is the point: a raw
    per-frame threshold flips a marginal region between restored and original on alternate frames,
    which looks worse than either choice held consistently.
    """

    HYSTERESIS_FRAMES = 3
    RELEASE_MARGIN = 0.05

    def __init__(
        self,
        threshold: float,
        hysteresis_frames: int = HYSTERESIS_FRAMES,
        release_margin: float = RELEASE_MARGIN,
    ) -> None:
        if hysteresis_frames < 1:
            raise ValueError("hysteresis_frames must be >= 1")
        if release_margin < 0:
            raise ValueError("release_margin must be >= 0")

        self._threshold = threshold
        self._hysteresis = hysteresis_frames
        self._margin = release_margin
        self._state: dict[int, list] = {}

    @property
    def is_disabled(self) -> bool:
        """Zero or below means the gate never withholds anything — the default (R-8.1b)."""
        return self._threshold <= 0.0

    @property
    def gated_track_count(self) -> int:
        """Regions currently withheld, for the job summary (R-8.1d)."""
        return sum(1 for gated, _, _ in self._state.values() if gated)

    def should_withhold(self, track_id: int, smoothed_confidence: float) -> bool:
        """Feeds one frame's confidence and returns whether to keep the original pixels."""
        if self.is_disabled:
            return False

        gated, below, above = self._state.get(track_id, (False, 0, 0))

        if gated:
            if smoothed_confidence > self._threshold + self._margin:
                above += 1
                if above >= self._hysteresis:
                    gated, below, above = False, 0, 0
            else:
                above = 0
        else:
            if smoothed_confidence < self._threshold:
                below += 1
                if below >= self._hysteresis:
                    gated, below, above = True, 0, 0
            else:
                below = 0

        self._state[track_id] = [gated, below, above]
        return gated

    def forget(self, track_id: int) -> None:
        """Drops a terminated track's state so a long job does not accumulate it."""
        self._state.pop(track_id, None)
