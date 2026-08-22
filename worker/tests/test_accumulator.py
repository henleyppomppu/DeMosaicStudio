"""Evidence carried forward. prd.md section 5.6, D-28.

The accumulator replaces a batch of K alignments per frame with one, and its correctness rests
almost entirely on when it *stops* carrying: a scene cut, a dropped frame or a region that jumped
means the accumulated pixels describe different content, and compositing them would put one shot's
picture into another. Those are the cases here.
"""

from __future__ import annotations

import numpy as np
import pytest

from demosaic_worker.analyze.profile import GridAnchor, MosaicProfile
from demosaic_worker.restore.accumulator import (
    MIN_ROI_OVERLAP,
    EvidenceAccumulator,
    _overlap,
    _reembed,
)
from demosaic_worker.restore.ibp import block_average
from demosaic_worker.metrics import psnr, shift_bilinear

SPEC = MosaicProfile(block_width=8, block_height=8, anchor=GridAnchor.SCREEN)


def _scene(height: int = 128, width: int = 192, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Kept inside [0, 255]: the accumulator clips, so a scene outside the range would fail a
    # comparison for reasons that have nothing to do with the accumulator.
    scene = rng.uniform(0, 195, size=(height, width)) + np.linspace(0, 60, width)[None, :]
    return np.clip(scene, 0.0, 255.0)


def _identity_flow(shape: tuple[int, int]) -> np.ndarray:
    return np.zeros((*shape, 2), dtype=np.float32)


def _shift_flow(shape: tuple[int, int], dx: float) -> np.ndarray:
    """A uniform flow field. ``warp_by_flow`` reads ``out[p] = image[p + flow[p]]``, so a positive
    ``dx`` samples further right - it undoes content that moved left."""
    flow = np.zeros((*shape, 2), dtype=np.float32)
    flow[..., 0] = dx
    return flow


# --------------------------------------------------------------------------------------------
# Restarting: the cases where carrying evidence forward would be wrong
# --------------------------------------------------------------------------------------------


def _fold(accumulator, track, index, observation, previous, bounds, mask, flow, same_scene=True):
    return accumulator.update(
        track,
        frame_index=index,
        observation=observation,
        previous_observation=previous,
        bounds=bounds,
        mask=mask,
        spec=SPEC,
        phase=(0, 0),
        flow_to_previous=flow,
        same_scene=same_scene,
    )


def test_the_first_frame_of_a_track_returns_its_observation_unchanged() -> None:
    """One frame of evidence can produce exactly one frame's worth of answer."""
    scene = _scene()
    accumulator = EvidenceAccumulator()

    got = _fold(accumulator, 1, 0, scene, None, (0, 0, 192, 128), None, None)

    assert np.array_equal(got, scene)
    assert accumulator.depth(1) == 1


@pytest.mark.parametrize(
    "reason,kwargs",
    [
        ("a scene cut", {"same_scene": False}),
        ("no usable flow", {"flow": None}),
        ("no previous observation", {"previous": None}),
    ],
)
def test_the_chain_restarts_when_the_evidence_would_be_about_other_content(
    reason: str, kwargs: dict
) -> None:
    scene = _scene()
    bounds = (0, 0, 192, 128)
    accumulator = EvidenceAccumulator()

    _fold(accumulator, 1, 0, scene, None, bounds, None, None)
    assert accumulator.depth(1) == 1

    call = {"previous": scene, "flow": _identity_flow(scene.shape), "same_scene": True}
    call.update(kwargs)
    got = _fold(accumulator, 1, 1, scene, call["previous"], bounds, None, call["flow"],
                same_scene=call["same_scene"])

    assert accumulator.depth(1) == 1, f"{reason} should have restarted the chain"
    assert np.array_equal(got, scene)


def test_a_skipped_frame_restarts_the_chain() -> None:
    """Frame indices have to be consecutive: the flow describes one frame of motion, not three."""
    scene = _scene()
    bounds = (0, 0, 192, 128)
    accumulator = EvidenceAccumulator()

    _fold(accumulator, 1, 0, scene, None, bounds, None, None)
    _fold(accumulator, 1, 3, scene, scene, bounds, None, _identity_flow(scene.shape))

    assert accumulator.depth(1) == 1


def test_a_region_that_jumps_restarts_the_chain() -> None:
    scene = _scene()
    accumulator = EvidenceAccumulator()

    _fold(accumulator, 1, 0, scene[:64, :64], None, (0, 0, 64, 64), None, None)
    # A box sharing almost nothing with the first.
    far = (120, 60, 184, 124)
    assert _overlap((0, 0, 64, 64), far) < MIN_ROI_OVERLAP

    _fold(accumulator, 1, 1, scene[60:124, 120:184], scene[60:124, 120:184], far, None,
          _identity_flow((64, 64)))

    assert accumulator.depth(1) == 1


def test_depth_grows_while_the_chain_holds() -> None:
    scene = _scene()
    bounds = (0, 0, 192, 128)
    accumulator = EvidenceAccumulator()

    for index in range(5):
        _fold(accumulator, 1, index, scene, scene if index else None, bounds, None,
              _identity_flow(scene.shape) if index else None)

    assert accumulator.depth(1) == 5


def test_tracks_do_not_share_evidence() -> None:
    scene = _scene()
    bounds = (0, 0, 192, 128)
    accumulator = EvidenceAccumulator()

    _fold(accumulator, 1, 0, scene, None, bounds, None, None)
    _fold(accumulator, 1, 1, scene, scene, bounds, None, _identity_flow(scene.shape))
    _fold(accumulator, 2, 1, scene, None, bounds, None, None)

    assert accumulator.depth(1) == 2
    assert accumulator.depth(2) == 1

    accumulator.forget(1)
    assert accumulator.depth(1) == 0
    assert accumulator.depth(2) == 1


# --------------------------------------------------------------------------------------------
# Re-embedding: the ROI follows the region, and the estimate has to follow the ROI
# --------------------------------------------------------------------------------------------


def test_reembedding_carries_the_overlap_and_fills_the_rest_from_the_frame() -> None:
    """Filling with zeros would inject a black border the next fold would treat as evidence."""
    # The fill is the previous frame seen through the *new* ROI, so it already has its shape.
    fill = np.full((40, 40), 50.0)
    estimate = np.full((40, 40), 200.0)

    carried = _reembed(estimate, (10, 10, 50, 50), (30, 30, 70, 70), fill)

    assert carried.shape == (40, 40)
    assert np.all(carried[:20, :20] == 200.0), "the overlap keeps the accumulated estimate"
    assert np.all(carried[20:, 20:] == 50.0), "the rest comes from the previous frame"


def test_reembedding_with_no_overlap_takes_the_frame_entirely() -> None:
    fill = np.full((20, 20), 50.0)
    estimate = np.full((20, 20), 200.0)

    carried = _reembed(estimate, (0, 0, 20, 20), (60, 60, 80, 80), fill)

    assert np.all(carried == 50.0)


# --------------------------------------------------------------------------------------------
# The mechanism itself
# --------------------------------------------------------------------------------------------


def test_evidence_accumulates_into_something_better_than_the_observation() -> None:
    """The point of the whole design: content that passed through the mosaic is recovered.

    A screen-fixed mosaic over content panning past it. Each frame the accumulator carries what it
    has, warps it by one frame of motion and folds in what this frame observed. After enough frames
    the region has been seen, a piece at a time, from outside the mask.
    """
    scene = _scene(height=96, width=320, seed=11)
    height, width = 96, 160
    bounds = (0, 0, width, height)

    mask = np.zeros((height, width), dtype=bool)
    mask[24:72, 48:112] = True

    def observed(index: int) -> np.ndarray:
        window = scene[:, index * 4 : index * 4 + width]
        return np.where(mask, block_average(window, SPEC, (0, 0)), window)

    accumulator = EvidenceAccumulator()
    frames = 24
    previous = None
    for index in range(frames):
        current = observed(index)
        flow = None if previous is None else _shift_flow(current.shape, 4.0)
        estimate = _fold(accumulator, 1, index, current, previous, bounds, mask, flow)
        previous = current

    truth = scene[:, (frames - 1) * 4 : (frames - 1) * 4 + width]
    before = psnr(truth[mask], observed(frames - 1)[mask])
    after = psnr(truth[mask], estimate[mask])

    assert accumulator.depth(1) == frames
    assert after > before + 2.0, (
        f"the accumulator recovered nothing behind the mosaic: {before:.2f} -> {after:.2f} dB"
    )


def test_an_unmasked_pixel_is_taken_as_observed() -> None:
    """Outside the mask a frame observes the scene directly, so the estimate must agree with it.

    This is the property that makes the accumulator work at all: it is how clean picture enters and
    is then carried under the mask as the content moves.
    """
    scene = _scene(height=64, width=64, seed=13)
    bounds = (0, 0, 64, 64)
    mask = np.zeros((64, 64), dtype=bool)
    mask[16:48, 16:48] = True

    accumulator = EvidenceAccumulator()
    _fold(accumulator, 1, 0, np.zeros_like(scene), None, bounds, mask, None)
    got = _fold(accumulator, 1, 1, scene, np.zeros_like(scene), bounds, mask,
                _identity_flow(scene.shape))

    assert np.allclose(got[~mask], scene[~mask]), "unmasked pixels must equal the observation"
    assert not np.allclose(got[mask], scene[mask]), "masked pixels are inferred, not copied"


def test_the_shift_helper_matches_the_flow_convention() -> None:
    """Guards the direction of the flow the accumulator is given, which is easy to get backwards."""
    scene = _scene(height=32, width=64, seed=17)
    moved = shift_bilinear(scene, 4.0, 0.0)

    from demosaic_worker.restore.flow import warp_by_flow

    # shift_bilinear(image, dx) reads out[p] = image[p - dx], so content moved right by 4;
    # sampling 4 further right puts it back.
    recovered = warp_by_flow(moved, _shift_flow(scene.shape, 4.0))

    assert psnr(scene[:, 8:-8], recovered[:, 8:-8]) > 40.0
