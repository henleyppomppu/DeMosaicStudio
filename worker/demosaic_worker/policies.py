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

#: Window by motion band. **Re-measured against the accumulator** — D-31, superseding D-16.
#:
#: The size is inert: the accumulator ignores K entirely, and window 1, 3 and 9 produce byte-for-byte
#: the same output (+2.83 dB on the same clip). Only ``> SINGLE_FRAME`` is load-bearing, and it is
#: kept as a number so the routing reasons stay comparable across versions.
#:
#: D-16 switched static and fast **off**, measured with the batch solver. Against the accumulator the
#: fast band is the *best* one, because fast motion is what exposes the region soonest:
#:
#: =====  ========  ======
#: pan    band      gain
#: =====  ========  ======
#: 1      slow      +2.83
#: 2      medium    +2.43
#: 4      medium    +2.87
#: 8      medium    +2.63
#: 16     **fast**  **+5.03**
#: 24     **fast**  **+6.45**
#: =====  ========  ======
WINDOW_BY_MOTION = {
    MotionBand.STATIC: 3,
    MotionBand.SLOW: 3,
    MotionBand.MEDIUM: 3,
    MotionBand.FAST: 3,
}

#: Bands where multi-frame is permitted at all. All of them, now that the solver chains one-frame
#: baselines rather than reaching across a window: the rule that excluded static and fast described
#: the batch solver's failure modes and not this one's (D-31).
MULTI_FRAME_BANDS = frozenset(MotionBand)

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
        """Feeds one frame's confidence and returns whether to keep the original pixels.

        **A track starts gated.** Restoration is an intervention: it should take evidence to begin,
        not evidence to stop. Starting open meant every new track was restored for
        ``hysteresis_frames - 1`` frames no matter how low its confidence - a gate set above every
        reachable confidence still let two frames per track through, which is how this was found.
        """
        if self.is_disabled:
            return False

        gated, below, above = self._state.get(track_id, (True, 0, 0))

        # The margin sits on the *closing* side. A user who sets minRestorationConfidence to X
        # means "restore where confidence is at least X"; putting the margin on the opening side
        # made X itself unreachable, and made every X within a margin of the confidence ceiling
        # silently mean "never restore" - the ceiling is 0.25 + 0.35 + 0.4 * blockPenalty, so for a
        # 10 px mosaic it is 0.90 and no threshold above 0.85 could ever open the gate.
        if gated:
            if smoothed_confidence >= self._threshold:
                above += 1
                if above >= self._hysteresis:
                    gated, below, above = False, 0, 0
            else:
                above = 0
        else:
            if smoothed_confidence < self._threshold - self._margin:
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


class ConfidenceSmoother:
    """Smooths a track's restoration confidence before the gate sees it. prd.md §5.8.1.

    :class:`ConfidenceGate` takes a parameter called ``smoothed_confidence`` and the pipeline was
    handing it the raw per-frame value. The mismatch was not cosmetic. Confidence varies frame to
    frame; the gate's hysteresis is per track and sticky in both directions; so a long track would
    open on a run of good frames and then coast through the bad ones. Measured on one clip, the
    per-frame signal could take the output from -0.82 dB to +0.075 dB, and the gate fed raw
    confidence could reach only 0.0 - by withholding everything.

    The time constant is the gate's own hysteresis window rather than a tuned number: the gate
    reasons over ``hysteresis_frames``, so the signal it reasons about is averaged over the same
    span.

    Mirror of ``DeMosaicStudio.Domain.Policies.ConfidenceSmoother`` (§13.4).
    """

    def __init__(self, window: int = ConfidenceGate.HYSTERESIS_FRAMES) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")

        self._alpha = 1.0 / window
        self._state: dict[int, float] = {}

    @property
    def alpha(self) -> float:
        """The exponential weight given to the newest frame."""
        return self._alpha

    def update(self, track_id: int, confidence: float) -> float:
        """Feeds one frame's confidence and returns the smoothed value for this track.

        The first frame of a track passes through unchanged: there is nothing to average it with,
        and seeding from zero would withhold every track's opening frames for a reason that has
        nothing to do with the evidence.
        """
        previous = self._state.get(track_id)
        smoothed = (
            confidence if previous is None
            else self._alpha * confidence + (1.0 - self._alpha) * previous
        )
        self._state[track_id] = smoothed
        return smoothed

    def forget(self, track_id: int) -> None:
        """Drops a terminated track's state."""
        self._state.pop(track_id, None)
