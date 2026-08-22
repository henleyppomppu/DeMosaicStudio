"""Classical multi-frame reconstruction by iterative back-projection. prd.md §16 Phase 0, §1.4.

Why this and not a learned model
--------------------------------
The Phase 0 gate asks one question: **is the information there?** A learned restorer cannot answer
it, because a poor result then means either "no information survived" or "this model is bad", and
the two are indistinguishable. Iterative back-projection has no training, no weights and no
randomness — it just inverts the forward operator as far as the data allows.

It also makes the comparison honest in a way two different algorithms never could: the single-frame
baseline and the multi-frame treatment are **the same code with a different K**. Nothing else
varies, so a difference is attributable to temporal evidence and to nothing else.

The forward model
-----------------
For a screen-anchored grid, frame ``t`` observes::

    y_t = Blur_grid( Shift_{d_t}( x ) )

where ``x`` is the clean target frame, ``d_t`` the global motion from the target to frame ``t``,
and ``Blur_grid`` the block averaging with a fixed phase in frame coordinates. Because the grid is
fixed and the content moves, each frame samples ``x`` at a *different* alignment — that is the
phase diversity of §1.4.1, and it is the only reason more frames can help at all.

Back-projection then iterates::

    x <- x + alpha * mean_t A_t^T ( y_t - A_t x )

with ``A_t^T`` implemented as "replicate each block value over its block, then warp back".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..analyze.profile import MosaicProfile
from ..metrics import shift_bilinear


def grid_edges(length: int, block: int, phase: int) -> np.ndarray:
    """Block boundaries along one axis for a grid starting at ``-phase``.

    A non-zero phase means the first block is partial: with ``block=8`` and ``phase=3`` the grid
    started three pixels before the image, so the first boundary inside the image is at 5.
    """
    if block < 1:
        raise ValueError("block size must be >= 1")

    first = (block - (phase % block)) % block
    interior = [e for e in range(first, length, block) if e > 0]

    return np.array([0, *interior], dtype=np.int64)


def block_average(image: np.ndarray, spec: MosaicProfile, phase: tuple[int, int]) -> np.ndarray:
    """Applies the forward operator: block means on the grid, kept at full resolution.

    Vectorised with ``np.add.reduceat``, which handles the partial blocks at the edges exactly —
    the naive reshape trick cannot, and silently getting the edge blocks wrong would bias every
    measurement taken near an ROI boundary. ``test_ibp.py`` pins this against a literal loop.
    """
    height, width = image.shape
    y_edges = grid_edges(height, spec.block_height, phase[1])
    x_edges = grid_edges(width, spec.block_width, phase[0])

    data = image.astype(np.float64)
    sums = np.add.reduceat(np.add.reduceat(data, y_edges, axis=0), x_edges, axis=1)

    y_sizes = np.diff(np.append(y_edges, height))
    x_sizes = np.diff(np.append(x_edges, width))
    counts = np.outer(y_sizes, x_sizes)

    means = sums / counts

    return np.repeat(np.repeat(means, y_sizes, axis=0), x_sizes, axis=1)


@dataclass(frozen=True, slots=True)
class Observation:
    """One frame's evidence about the target frame."""

    #: The frame as observed, at full resolution.
    observed: np.ndarray

    #: Global motion from the *target* frame to this one, in pixels.
    dx: float
    dy: float

    #: Where *this* frame is mosaicked, in its own coordinates. ``None`` means "everywhere", which
    #: is the model the pipeline used before masks existed. See :func:`forward_and_adjoint`.
    mask: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class IbpResult:
    """A reconstruction and how it got there."""

    image: np.ndarray
    iterations: int
    residuals: list[float]

    @property
    def converged(self) -> bool:
        """True when the last iteration reduced the residual by less than a per-mille."""
        if len(self.residuals) < 2:
            return False
        return abs(self.residuals[-2] - self.residuals[-1]) < 1e-3 * max(self.residuals[-1], 1e-9)


#: Neighbours needed before modelling the mask pays for itself. **Measured**, not assumed.
#:
#: A neighbour ``d`` pixels away exposes a crescent of roughly ``2 * d * ry`` out of an ellipse of
#: ``pi * rx * ry`` - about ``2 * d / (pi * rx)`` of what the target lost. For a 300 px mosaic and
#: 4 px per frame that is 1.7% per neighbour, and the measurement matches it to a tenth of a
#: percent. So the evidence accumulates with the window, and the model that exploits it only starts
#: winning once enough of the region has been seen:
#:
#: ===========  ========  =========  ==========
#: neighbours   coverage  no mask    mask-aware
#: ===========  ========  =========  ==========
#: 2            3.6%      23.22      23.24
#: 8            14.6%     22.69      22.62
#: 16           28.1%     22.16      24.68
#: 24           49.4%     21.65      26.04
#: ===========  ========  =========  ==========
#:
#: Note the directions: more evidence makes the all-masked model *worse* and the mask-aware model
#: *better*. That is the signature of a forward model that is finally describing the data.
MASK_MODEL_MIN_NEIGHBOURS = 16


def forward_and_adjoint(
    warped: np.ndarray,
    observed: np.ndarray,
    spec: MosaicProfile,
    phase: tuple[int, int],
    mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Applies the forward operator and back-projects the residual. Returns ``(residual, spread)``.

    **The mosaic covers part of a frame, not all of it**, and that distinction is the whole reason
    more than one frame can help. Where a frame is mosaicked it observes block averages of the
    scene; where it is not, it observes the scene *directly*, at full resolution. A neighbour whose
    mask does not cover a piece of content the target has lost is not weak evidence about it - it is
    the answer.

    Modelling every pixel as block-averaged threw that away. It also made the operator wrong at the
    ROI's corners, which are picture that was never degraded: the solver was told they were block
    averages and dutifully pushed them towards being block averages.

    So::

        A(x) = block_average(x)  where the frame is mosaicked
             = x                 where it is not

    and the adjoint follows: spread the residual over its block where masked, apply it directly
    where not. The block mean is taken over the masked pixels only, so a block straddling the mask
    boundary is not diluted by pixels that were never averaged.
    """
    simulated = block_average(warped, spec, phase)
    if mask is not None:
        simulated = np.where(mask, simulated, warped)

    residual = observed.astype(np.float64) - simulated

    if mask is None:
        return residual, block_average(residual, spec, phase)

    weight = mask.astype(np.float64)
    covered = block_average(weight, spec, phase)
    spread_masked = block_average(residual * weight, spec, phase) / np.maximum(covered, 1e-6)

    return residual, np.where(mask, spread_masked, residual)


def reconstruct(
    observations: list[Observation],
    spec: MosaicProfile,
    phase: tuple[int, int],
    *,
    iterations: int = 40,
    step: float = 1.0,
    smoothing: float = 0.0,
) -> IbpResult:
    """Reconstructs the target frame from one or more block-averaged observations.

    ``observations[0]`` must be the target frame itself (``dx = dy = 0``). Passing only that one
    observation is the **single-frame baseline**; passing more is the multi-frame treatment. That
    is the whole experiment.
    """
    if not observations:
        raise ValueError("need at least one observation")
    if observations[0].dx != 0.0 or observations[0].dy != 0.0:
        raise ValueError("observations[0] must be the target frame, with zero motion")

    # Starting from the target's own block averages rather than from zero: the block means are
    # already the correct low-frequency content, so the iteration only has to add detail.
    estimate = observations[0].observed.astype(np.float64).copy()
    best_estimate = estimate.copy()
    best_residual = float("inf")
    worse = 0

    residuals: list[float] = []

    for _ in range(iterations):
        correction = np.zeros_like(estimate)
        total_residual = 0.0

        for observation in observations:
            # Forward: move the estimate into this frame's coordinates, then average on the grid.
            warped = (
                estimate
                if observation.dx == 0.0 and observation.dy == 0.0
                else shift_bilinear(estimate, observation.dx, observation.dy)
            )
            residual, spread = forward_and_adjoint(
                warped, observation.observed, spec, phase, observation.mask
            )
            total_residual += float(np.mean(np.abs(residual)))
            back = (
                spread
                if observation.dx == 0.0 and observation.dy == 0.0
                else shift_bilinear(spread, -observation.dx, -observation.dy)
            )
            correction += back

        estimate += step * correction / len(observations)

        if smoothing > 0.0:
            estimate = _smooth(estimate, smoothing)

        residual_now = total_residual / len(observations)
        residuals.append(residual_now)

        # **Keep the best iterate, and stop when it stops being the latest one.**
        #
        # Back-projection with a *dense* flow is not a descent on a consistent objective: the
        # to-target and to-neighbour fields are estimated separately, so the forward warp and the
        # back warp are not exact inverses and the iteration can walk away from the answer. It
        # does. Measured on synthetic content with the real aligner: +0.58 dB at 5 iterations,
        # -0.18 at 20, -2.79 at 40. The pipeline was running 20.
        #
        # The old stopping rule watched the *change* in the residual, so it never noticed growth.
        # The residual is the only thing available at runtime, so the honest thing is to return
        # the iterate that fit the data best rather than the last one computed.
        if residual_now < best_residual - 1e-9:
            best_residual = residual_now
            best_estimate = estimate.copy()
            worse = 0
        else:
            worse += 1
            if worse >= 2:
                break

        if len(residuals) >= 2 and abs(residuals[-2] - residuals[-1]) < 1e-6:
            break

    return IbpResult(np.clip(best_estimate, 0.0, 255.0), len(residuals), residuals)


def _smooth(image: np.ndarray, weight: float) -> np.ndarray:
    """A mild isotropic prior, applied between iterations to keep the solve stable."""
    padded = np.pad(image, 1, mode="reflect")
    neighbours = (
        padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
    ) / 4.0

    return (1.0 - weight) * image + weight * neighbours


def upsample_baseline(observed: np.ndarray) -> np.ndarray:
    """The do-nothing reference: the block averages themselves.

    prd.md §12.3 requires pass-through as a baseline in every restoration report, because a
    restoration that does not beat "leave it alone" is not a restoration.
    """
    return observed.astype(np.float64).copy()


# --- dense-flow reconstruction (prd.md §5.7) ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FlowObservation:
    """One neighbour aligned by a dense flow field rather than a single translation.

    The global-shift :class:`Observation` above is what the Phase 0 gate measured at −0.86 dB. This
    is the replacement the gate's verdict called for.
    """

    #: Block-averaged neighbour frame, at full resolution.
    observed: np.ndarray

    #: Flow from the target frame to this neighbour, ``(H, W, 2)``. Used to bring residuals back.
    to_neighbour: np.ndarray

    #: Flow from this neighbour to the target. Used to simulate the observation.
    to_target: np.ndarray

    #: Per-pixel confidence in target coordinates, ``[0, 1]``.
    confidence: np.ndarray

    #: Where *this* frame is mosaicked, in its own coordinates. ``None`` means "everywhere".
    mask: np.ndarray | None = None

    @staticmethod
    def target(observed: np.ndarray, mask: np.ndarray | None = None) -> "FlowObservation":
        """The target frame itself: identity flow, full confidence."""
        zeros = np.zeros((*observed.shape, 2), dtype=np.float32)
        return FlowObservation(
            observed,
            zeros,
            zeros.copy(),
            np.ones(observed.shape, dtype=np.float32),
            mask,
        )

    @property
    def is_target(self) -> bool:
        """True for the identity observation."""
        return not self.to_target.any()


def reconstruct_flow(
    observations: list[FlowObservation],
    spec: MosaicProfile,
    phase: tuple[int, int],
    *,
    iterations: int = 40,
    step: float = 1.0,
    smoothing: float = 0.0,
) -> IbpResult:
    """Iterative back-projection with dense alignment and per-pixel weighting.

    Two differences from :func:`reconstruct`, both of which the Phase 0 gate asked for:

    * warping is by a **dense flow field**, so parallax and independent object motion are described
      rather than approximated by one translation;
    * each neighbour's contribution is weighted **per pixel** by its flow confidence, so a frame can
      contribute where it is trustworthy and be ignored where it is not. Normalising by the summed
      weight rather than the observation count means a pixel that only the target can see still gets
      a full-strength correction instead of a diluted one.
    """
    if not observations:
        raise ValueError("need at least one observation")
    if not observations[0].is_target:
        raise ValueError("observations[0] must be the target, with identity flow")

    from .flow import warp_by_flow  # local import: keeps ibp usable without torchvision

    estimate = observations[0].observed.astype(np.float64).copy()
    best_estimate = estimate.copy()
    best_residual = float("inf")
    worse = 0

    residuals: list[float] = []

    weights = [o.confidence.astype(np.float64) for o in observations]
    total_weight = np.maximum(sum(weights), 1e-6)

    for _ in range(iterations):
        correction = np.zeros_like(estimate)
        total_residual = 0.0

        for observation, weight in zip(observations, weights, strict=True):
            warped = (
                estimate
                if observation.is_target
                else warp_by_flow(estimate, observation.to_target)
            )
            residual, spread = forward_and_adjoint(
                warped, observation.observed, spec, phase, observation.mask
            )
            total_residual += float(np.mean(np.abs(residual)))
            back = (
                spread
                if observation.is_target
                else warp_by_flow(spread, observation.to_neighbour)
            )

            correction += weight * back

        estimate += step * correction / total_weight

        if smoothing > 0.0:
            estimate = _smooth(estimate, smoothing)

        residual_now = total_residual / len(observations)
        residuals.append(residual_now)

        # **Keep the best iterate, and stop when it stops being the latest one.**
        #
        # Back-projection with a *dense* flow is not a descent on a consistent objective: the
        # to-target and to-neighbour fields are estimated separately, so the forward warp and the
        # back warp are not exact inverses and the iteration can walk away from the answer. It
        # does. Measured on synthetic content with the real aligner: +0.58 dB at 5 iterations,
        # -0.18 at 20, -2.79 at 40. The pipeline was running 20.
        #
        # The old stopping rule watched the *change* in the residual, so it never noticed growth.
        # The residual is the only thing available at runtime, so the honest thing is to return
        # the iterate that fit the data best rather than the last one computed.
        if residual_now < best_residual - 1e-9:
            best_residual = residual_now
            best_estimate = estimate.copy()
            worse = 0
        else:
            worse += 1
            if worse >= 2:
                break

        if len(residuals) >= 2 and abs(residuals[-2] - residuals[-1]) < 1e-6:
            break

    return IbpResult(np.clip(best_estimate, 0.0, 255.0), len(residuals), residuals)
