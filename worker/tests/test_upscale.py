"""Decimate-then-upscale restoration, and the temporal blend over it. D-43.

Two rules under test that a working demo would not show:

* **Decimation must agree with the forward operator.** The mosaic is block means on a grid with a
  phase; if the small image is cut on a different grid, the network is handed a picture of the
  wrong thing and every restoration is shifted by up to a block.
* **The blend must reset when blending would be wrong** - a cut, a gap, a jump. An EMA that does
  not reset produces a ghost of the previous shot on the first frame of the next one.

The network's weights are not in the repository. The tests that need a network build one with
random weights: shape, batching and dtype are properties of the architecture, not the training.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from demosaic_worker.analyze.profile import GridAnchor, MosaicProfile
from demosaic_worker.errors import WorkerError
from demosaic_worker.restore.ibp import block_average
from demosaic_worker.restore.temporal import DEFAULT_ALPHA, TemporalSmoother
from demosaic_worker.restore.upscale import (
    NETWORK_SCALE,
    SRVGGNetCompact,
    Upscaler,
    bicubic_restore,
    decimate,
    resize,
)

torch = pytest.importorskip("torch")

SPEC = MosaicProfile(block_width=8, block_height=8, anchor=GridAnchor.SCREEN)


def _scene(height: int = 64, width: int = 96, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    return np.clip(
        128 + 60 * np.sin(x / 7.0) * np.cos(y / 5.0) + rng.normal(0, 8, (height, width)), 0, 255
    )


# --------------------------------------------------------------------------------------------
# decimate
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("phase", [(0, 0), (3, 5), (7, 1)])
def test_decimation_is_the_forward_operator_at_its_true_resolution(phase: tuple[int, int]) -> None:
    """Repeating the small image back up on the same grid must give block_average exactly."""
    image = _scene()
    small = decimate(image, SPEC, phase)
    averaged = block_average(image, SPEC, phase)

    # Rebuild the full-size mosaic from the small image using the same edges block_average uses.
    from demosaic_worker.restore.ibp import grid_edges

    y_edges = grid_edges(image.shape[0], SPEC.block_height, phase[1])
    x_edges = grid_edges(image.shape[1], SPEC.block_width, phase[0])
    y_sizes = np.diff(np.append(y_edges, image.shape[0]))
    x_sizes = np.diff(np.append(x_edges, image.shape[1]))
    rebuilt = np.repeat(np.repeat(small, y_sizes, axis=0), x_sizes, axis=1)

    assert small.shape == (len(y_edges), len(x_edges))
    np.testing.assert_allclose(rebuilt, averaged)


def test_a_mosaicked_input_decimates_losslessly() -> None:
    """A crop that already is a mosaic loses nothing: one number per block, and it is that number."""
    mosaic = block_average(_scene(), SPEC, (0, 0))
    small = decimate(mosaic, SPEC, (0, 0))

    np.testing.assert_allclose(small, mosaic[::8, ::8])


# --------------------------------------------------------------------------------------------
# bicubic floor
# --------------------------------------------------------------------------------------------


def test_the_bicubic_floor_removes_the_grid() -> None:
    """The point of the floor: no block edges. Measured as horizontal step energy at block seams."""
    mosaic = block_average(_scene(), SPEC, (0, 0))
    restored = bicubic_restore(mosaic, SPEC, (0, 0))

    def seam_energy(image: np.ndarray) -> float:
        steps = np.abs(np.diff(image, axis=1))
        at_seams = steps[:, 7::8].mean()      # columns 7|8, 15|16, ... are block edges
        elsewhere = np.delete(steps, np.s_[7::8], axis=1).mean()
        return at_seams / max(elsewhere, 1e-9)

    assert seam_energy(mosaic) > 5.0, "the input should have strong seams"
    assert seam_energy(restored) < 1.5, "the floor should have none"
    assert restored.shape == mosaic.shape
    assert 0.0 <= restored.min() and restored.max() <= 255.0


def test_resize_round_trips_a_constant() -> None:
    flat = np.full((16, 24), 77.0)
    assert np.allclose(resize(flat, (64, 96)), 77.0, atol=0.01)


# --------------------------------------------------------------------------------------------
# the network
# --------------------------------------------------------------------------------------------


def _write_restorer(directory: Path, *, num_feat: int = 8, num_conv: int = 2) -> Path:
    """A tiny random network in the store's own format. Weights are not the subject here."""
    import hashlib

    directory.mkdir(parents=True, exist_ok=True)
    model = SRVGGNetCompact(num_feat=num_feat, num_conv=num_conv)
    weights = directory / "model.pt"
    torch.save(
        {"state_dict": model.state_dict(), "num_feat": num_feat, "num_conv": num_conv}, weights
    )
    (directory / "metadata.json").write_text(
        json.dumps({
            "id": "sr-test", "version": "0",
            "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        }),
        encoding="utf-8",
    )
    return directory


def test_the_architecture_upscales_by_the_network_scale() -> None:
    net = SRVGGNetCompact(num_feat=8, num_conv=2)
    out = net(torch.zeros(1, 3, 6, 9))
    assert out.shape == (1, 3, 6 * NETWORK_SCALE, 9 * NETWORK_SCALE)


def test_the_real_checkpoint_layout_loads() -> None:
    """realesr-general-x4v3 is 64 features, 32 convolutions.

    body = first conv + PReLU (2), 32 x (conv + PReLU) (64), last conv (1): indices 0..66. The
    released checkpoint's last key is ``body.66.bias``, and a mismatch here is a load failure.
    """
    net = SRVGGNetCompact()
    keys = list(net.state_dict())
    assert keys[0] == "body.0.weight"
    assert keys[-1] == "body.66.bias"
    assert sum(p.numel() for p in net.parameters()) == pytest.approx(1_211_000, rel=0.02)


def test_restore_many_returns_each_crop_at_its_own_size(tmp_path: Path) -> None:
    """Regions differ in size; they go through together and come back separately."""
    upscaler = Upscaler(_write_restorer(tmp_path / "sr"), device="cpu")
    crops = [_scene(64, 96), _scene(40, 40, seed=2), _scene(56, 72, seed=3)]
    specs = [SPEC, SPEC, MosaicProfile(block_width=4, block_height=4)]
    phases = [(0, 0), (3, 3), (1, 0)]

    outputs = upscaler.restore_many(crops, specs, phases)

    assert [o.shape for o in outputs] == [c.shape for c in crops]
    assert all(0.0 <= o.min() and o.max() <= 255.0 for o in outputs)
    assert all(np.isfinite(o).all() for o in outputs)


def test_restore_many_with_nothing_to_do_returns_nothing(tmp_path: Path) -> None:
    assert Upscaler(_write_restorer(tmp_path / "sr"), device="cpu").restore_many([], [], []) == []


def test_a_missing_restorer_is_e4001_not_e3001(tmp_path: Path) -> None:
    """The detector's loader is reused; the code it raises must be the restorer's."""
    with pytest.raises(WorkerError) as failure:
        Upscaler(tmp_path / "absent", device="cpu")
    assert failure.value.code.code == "E4001"


def test_wrong_weights_are_e4001(tmp_path: Path) -> None:
    directory = _write_restorer(tmp_path / "sr", num_feat=8, num_conv=2)
    # Rewrite the metadata claim so the architecture does not match the weights.
    checkpoint = torch.load(directory / "model.pt", weights_only=True)
    checkpoint["num_conv"] = 5
    torch.save(checkpoint, directory / "model.pt")
    import hashlib

    (directory / "metadata.json").write_text(json.dumps({
        "id": "sr-test", "version": "0",
        "sha256": hashlib.sha256((directory / "model.pt").read_bytes()).hexdigest(),
    }))

    with pytest.raises(WorkerError) as failure:
        Upscaler(directory, device="cpu")
    assert failure.value.code.code == "E4001"


# --------------------------------------------------------------------------------------------
# temporal blend
# --------------------------------------------------------------------------------------------


def test_the_first_frame_of_a_track_passes_through_unchanged() -> None:
    smoother = TemporalSmoother()
    out = smoother.smooth(1, np.full((8, 8), 100.0), observation=np.zeros_like(np.full((8, 8), 100.0)), bounds=(0, 0, 8, 8), frame_index=0)
    np.testing.assert_allclose(out, 100.0)


def test_the_blend_is_seven_to_three_by_default() -> None:
    smoother = TemporalSmoother()
    assert smoother.alpha == DEFAULT_ALPHA == 0.3

    smoother.smooth(1, np.full((8, 8), 100.0), observation=np.zeros_like(np.full((8, 8), 100.0)), bounds=(0, 0, 8, 8), frame_index=0)
    out = smoother.smooth(1, np.full((8, 8), 200.0), observation=np.zeros_like(np.full((8, 8), 200.0)), bounds=(0, 0, 8, 8), frame_index=1)

    np.testing.assert_allclose(out, 0.7 * 100.0 + 0.3 * 200.0)


def test_a_scene_cut_resets_the_blend() -> None:
    """Otherwise the first frame after a cut carries a ghost of the last frame before it."""
    smoother = TemporalSmoother()
    smoother.smooth(1, np.full((8, 8), 100.0), observation=np.zeros_like(np.full((8, 8), 100.0)), bounds=(0, 0, 8, 8), frame_index=0)
    out = smoother.smooth(
        1, np.full((8, 8), 200.0), observation=np.zeros((8, 8)), bounds=(0, 0, 8, 8),
        frame_index=1, same_scene=False,
    )
    np.testing.assert_allclose(out, 200.0)


def test_a_skipped_frame_resets_the_blend() -> None:
    smoother = TemporalSmoother()
    smoother.smooth(1, np.full((8, 8), 100.0), observation=np.zeros_like(np.full((8, 8), 100.0)), bounds=(0, 0, 8, 8), frame_index=0)
    out = smoother.smooth(1, np.full((8, 8), 200.0), observation=np.zeros_like(np.full((8, 8), 200.0)), bounds=(0, 0, 8, 8), frame_index=5)
    np.testing.assert_allclose(out, 200.0)


def test_a_region_that_jumped_resets_and_one_that_moved_does_not() -> None:
    smoother = TemporalSmoother()
    smoother.smooth(1, np.full((8, 8), 100.0), observation=np.zeros_like(np.full((8, 8), 100.0)), bounds=(0, 0, 8, 8), frame_index=0)

    moved = smoother.smooth(1, np.full((8, 8), 200.0), observation=np.zeros_like(np.full((8, 8), 200.0)), bounds=(2, 0, 10, 8), frame_index=1)
    # The overlapping six columns were blended; the two new columns had nothing to blend with.
    np.testing.assert_allclose(moved[:, :6], 0.7 * 100.0 + 0.3 * 200.0)
    np.testing.assert_allclose(moved[:, 6:], 200.0)

    jumped = smoother.smooth(1, np.full((8, 8), 50.0), observation=np.zeros_like(np.full((8, 8), 50.0)), bounds=(100, 100, 108, 108), frame_index=2)
    np.testing.assert_allclose(jumped, 50.0)


def test_tracks_are_independent() -> None:
    smoother = TemporalSmoother()
    smoother.smooth(1, np.full((4, 4), 0.0), observation=np.zeros_like(np.full((4, 4), 0.0)), bounds=(0, 0, 4, 4), frame_index=0)
    other = smoother.smooth(2, np.full((4, 4), 255.0), observation=np.zeros_like(np.full((4, 4), 255.0)), bounds=(0, 0, 4, 4), frame_index=0)
    np.testing.assert_allclose(other, 255.0)
    assert smoother.has(1) and smoother.has(2)

    smoother.reset(1)
    assert not smoother.has(1) and smoother.has(2)


def test_the_blend_applies_only_where_the_observation_did_not_move() -> None:
    """The measurement behind this: an unconditional 7:3 blend scored -4.52 dB on a panning clip
    against +1.00 dB with no blend. The observation says which pixels moved."""
    smoother = TemporalSmoother()
    still = np.full((8, 8), 50.0)
    smoother.smooth(1, np.full((8, 8), 100.0), observation=still, bounds=(0, 0, 8, 8), frame_index=0)

    # Left half of the picture moved under the mosaic; right half did not.
    moved = still.copy()
    moved[:, :4] += 40.0
    out = smoother.smooth(1, np.full((8, 8), 200.0), observation=moved, bounds=(0, 0, 8, 8), frame_index=1)

    np.testing.assert_allclose(out[:, :4], 200.0)                    # moved: new frame only
    np.testing.assert_allclose(out[:, 4:], 0.7 * 100.0 + 0.3 * 200.0)  # still: 7:3


def test_codec_level_noise_in_the_observation_still_blends() -> None:
    smoother = TemporalSmoother()
    base = np.full((6, 6), 120.0)
    smoother.smooth(1, np.full((6, 6), 0.0), observation=base, bounds=(0, 0, 6, 6), frame_index=0)
    # Half a level: what a block mean drifts by under a codec. Inside the 1.0 tolerance.
    out = smoother.smooth(1, np.full((6, 6), 100.0), observation=base + 0.5, bounds=(0, 0, 6, 6), frame_index=1)
    np.testing.assert_allclose(out, 30.0)
