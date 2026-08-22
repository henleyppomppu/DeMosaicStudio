"""Metric sanity. prd.md §12.3, §12.4."""

from __future__ import annotations

import numpy as np
import pytest

from metrics import psnr, shift_bilinear, ssim, warping_error


def _image(seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(64, 64)).astype(np.float64)


def test_identical_images_score_perfectly() -> None:
    a = _image()
    assert psnr(a, a.copy()) == float("inf")
    assert ssim(a, a.copy()) == pytest.approx(1.0, abs=1e-9)


def test_psnr_falls_as_noise_grows() -> None:
    a = _image()
    rng = np.random.default_rng(11)

    quiet = psnr(a, np.clip(a + rng.normal(0, 2, a.shape), 0, 255))
    loud = psnr(a, np.clip(a + rng.normal(0, 20, a.shape), 0, 255))

    assert quiet > loud


def test_ssim_falls_as_structure_is_destroyed() -> None:
    a = _image()
    blurred = np.repeat(np.repeat(a[::8, ::8], 8, axis=0), 8, axis=1)

    assert ssim(a, blurred) < ssim(a, np.clip(a + 1.0, 0, 255))


def test_mismatched_shapes_are_rejected() -> None:
    with pytest.raises(ValueError):
        psnr(_image(), np.zeros((8, 8)))
    with pytest.raises(ValueError):
        ssim(_image(), np.zeros((8, 8)))


def test_a_sub_pixel_shift_interpolates() -> None:
    image = np.zeros((8, 8))
    image[4, 4] = 100.0

    half = shift_bilinear(image, 0.5, 0.0)

    assert half[4, 4] == pytest.approx(50.0)
    assert half[4, 5] == pytest.approx(50.0)


def test_an_integer_shift_moves_the_content() -> None:
    image = np.zeros((8, 8))
    image[4, 4] = 100.0

    moved = shift_bilinear(image, 2.0, 0.0)

    assert moved[4, 6] == pytest.approx(100.0)


def test_warping_error_is_zero_for_a_perfectly_tracked_pan() -> None:
    base = _image()
    frames = [base, shift_bilinear(base, 3.0, 0.0)]

    # Warping frame 0 forward by the known shift should land on frame 1.
    error = warping_error(frames, [(3.0, 0.0)])

    assert error < 1.0


def test_warping_error_grows_when_frames_are_inconsistent() -> None:
    base = _image()
    rng = np.random.default_rng(5)
    flickering = [base, np.clip(base + rng.normal(0, 40, base.shape), 0, 255)]

    steady = warping_error([base, base.copy()], [(0.0, 0.0)])
    noisy = warping_error(flickering, [(0.0, 0.0)])

    assert noisy > steady + 10
