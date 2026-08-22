"""Scene cut detection. prd.md §5.12.

A cut invalidates temporal context: frames on the far side of one describe a different scene, and
fusing across it produces the worst kind of artifact — one that looks like a restoration failure but
is really a scheduling failure.

**A camera flash is not a cut.** A global luminance spike with unchanged structure is a flash, and
resetting temporal context for it throws away good neighbours for no reason.

Two signals are needed because neither alone can tell a flash from a cut:

* a **histogram** distance separates a continuation from everything else, but a flash looks exactly
  as large as a cut to it;
* a **structure** distance — ``1 - |normalised cross-correlation|`` — is invariant to the affine
  luminance change a flash applies, so it stays near zero for a flash and near one for a cut.

Both thresholds were calibrated on the corpus by `scripts/calibrate_scene_cuts.py` rather than
chosen. The first structure measure tried here compared peak-normalised gradient maps and did not
separate the populations at all (within-shot p95 0.058 against across-shot p05 0.042); the
calibration is what caught it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Histogram distance above which a frame pair is a cut candidate.
#:
#: **Calibrated, not chosen** — `scripts/calibrate_scene_cuts.py` on the corpus:
#: continuations reach 0.047 at p95, genuine cuts start at 0.135 at p05. A flash also scores high
#: here (median 0.708), which is exactly why a second signal is needed to tell the two apart.
DEFAULT_HISTOGRAM_THRESHOLD = 0.09

#: Structural change required to confirm a candidate, as ``1 - |NCC|``.
#:
#: **Calibrated against the flash population, which is the one this signal can be confused with:**
#: flashes reach 0.337 at p95, genuine cuts start at 0.693 at p05. Continuations sit far below both
#: (median 0.020).
DEFAULT_STRUCTURE_THRESHOLD = 0.515

#: Bins for the luma histogram. Coarse on purpose: fine bins make grain look like a scene change.
HISTOGRAM_BINS = 32


def _histogram(frame: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(frame, bins=HISTOGRAM_BINS, range=(0, 256))
    total = counts.sum()
    return counts / total if total else counts.astype(np.float64)


def histogram_distance(previous: np.ndarray, current: np.ndarray) -> float:
    """Total-variation distance between two frames' luma histograms, in ``[0, 1]``."""
    return float(0.5 * np.abs(_histogram(previous) - _histogram(current)).sum())


def structure_distance(previous: np.ndarray, current: np.ndarray) -> float:
    """``1 - |normalised cross-correlation|``, in ``[0, 1]``. Zero means identical structure.

    **This measure was chosen by measurement, not by intuition.** An earlier version compared
    peak-normalised gradient maps, and `scripts/calibrate_scene_cuts.py` showed it did not separate
    the populations at all: within-shot p95 0.058 against across-shot p05 0.042, fully overlapping.

    Normalised cross-correlation is invariant to an affine luminance change — exactly the transform
    a camera flash applies — so a flash scores near zero while a genuine cut, which changes what is
    actually in the frame, scores near one. That is the asymmetry the classifier needs and the
    gradient measure did not provide.
    """
    a = previous.astype(np.float64).ravel()
    b = current.astype(np.float64).ravel()

    a = a - a.mean()
    b = b - b.mean()

    denominator = np.sqrt(float(a @ a) * float(b @ b))
    if denominator < 1e-9:
        # A flat frame has no structure to compare; treat it as unchanged rather than as a cut.
        return 0.0

    return float(1.0 - abs(float(a @ b) / denominator))


@dataclass(frozen=True, slots=True)
class SceneChange:
    """The verdict for one frame pair."""

    index: int
    histogram: float
    structure: float
    is_cut: bool
    is_flash: bool

    @property
    def resets_temporal_context(self) -> bool:
        """Only a cut resets the window (§5.12). A flash deliberately does not."""
        return self.is_cut


def classify_pair(
    previous: np.ndarray,
    current: np.ndarray,
    index: int,
    *,
    histogram_threshold: float = DEFAULT_HISTOGRAM_THRESHOLD,
    structure_threshold: float = DEFAULT_STRUCTURE_THRESHOLD,
) -> SceneChange:
    """Classifies one adjacent frame pair as a cut, a flash, or neither."""
    histogram = histogram_distance(previous, current)
    structure = structure_distance(previous, current)

    candidate = histogram >= histogram_threshold
    structural = structure >= structure_threshold

    return SceneChange(
        index=index,
        histogram=histogram,
        structure=structure,
        is_cut=candidate and structural,
        is_flash=candidate and not structural,
    )


def detect_cuts(frames: list[np.ndarray], **kwargs: float) -> list[SceneChange]:
    """Classifies every adjacent pair in a sequence.

    ``result[i]`` describes the boundary *before* ``frames[i + 1]``.
    """
    return [
        classify_pair(frames[i], frames[i + 1], i + 1, **kwargs)  # type: ignore[arg-type]
        for i in range(len(frames) - 1)
    ]


def same_scene_span(
    changes: list[SceneChange],
    target: int,
    radius: int,
    total_frames: int | None = None,
) -> tuple[int, int]:
    """The inclusive frame range around ``target`` that shares its scene, bounded by ``radius``.

    This is what feeds ``same_scene_frames`` in the window policy: the window may reach as far as the
    nearest cut and no further.

    ``total_frames`` bounds the result to the sequence. Without it the span can run past frame 0 —
    the window policy would then be told more frames are available than exist, and the reduction
    would be attributed to the stream boundary instead of the scene, which is the wrong warning.
    """
    cuts = {c.index for c in changes if c.resets_temporal_context}

    last = (total_frames - 1) if total_frames is not None else (len(changes))
    target = max(0, min(target, last))

    start = target
    while start > max(0, target - radius) and start not in cuts:
        start -= 1

    end = target
    while end < min(last, target + radius) and (end + 1) not in cuts:
        end += 1

    return start, end
