"""Image quality metrics. prd.md §12.3.

Implemented in numpy rather than pulled from a library so the Phase 0 gate has no dependency on
torch. LPIPS is deliberately absent: it needs a pretrained network, and §1.4.3's gate is decided on
PSNR, LPIPS *and* warping error together — so when LPIPS arrives with Phase 2, the gate is re-run
rather than retro-fitted. Until then the gate reports what it measured and says what it did not.
"""

from __future__ import annotations

import numpy as np

#: Peak signal value for 8-bit images.
MAX_VALUE = 255.0


def psnr(reference: np.ndarray, test: np.ndarray, *, data_range: float = MAX_VALUE) -> float:
    """Peak signal-to-noise ratio in dB.

    Returns ``inf`` for identical inputs, which callers must handle: averaging a set of PSNRs that
    contains one identical pair otherwise silently yields ``inf`` for the whole set.
    """
    if reference.shape != test.shape:
        raise ValueError(f"shape mismatch: {reference.shape} vs {test.shape}")

    mse = float(np.mean((reference.astype(np.float64) - test.astype(np.float64)) ** 2))
    if mse == 0.0:
        return float("inf")

    return float(10.0 * np.log10((data_range**2) / mse))


def _gaussian_kernel(sigma: float = 1.5, radius: int = 5) -> np.ndarray:
    coords = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(coords**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def _separable_blur(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    radius = len(kernel) // 2
    padded = np.pad(image, radius, mode="reflect")

    rows = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), axis=1, arr=padded)
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), axis=0, arr=rows)


def ssim(
    reference: np.ndarray,
    test: np.ndarray,
    *,
    data_range: float = MAX_VALUE,
    sigma: float = 1.5,
) -> float:
    """Structural similarity, Gaussian-windowed, as in Wang et al. 2004.

    Operates on a single channel. For colour, call it per channel and average, or pass luma.
    """
    if reference.shape != test.shape:
        raise ValueError(f"shape mismatch: {reference.shape} vs {test.shape}")
    if reference.ndim != 2:
        raise ValueError("ssim expects a 2-D single-channel image")

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    a = reference.astype(np.float64)
    b = test.astype(np.float64)
    kernel = _gaussian_kernel(sigma)

    mu_a = _separable_blur(a, kernel)
    mu_b = _separable_blur(b, kernel)

    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b

    sigma_a2 = _separable_blur(a * a, kernel) - mu_a2
    sigma_b2 = _separable_blur(b * b, kernel) - mu_b2
    sigma_ab = _separable_blur(a * b, kernel) - mu_ab

    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)

    return float(np.mean(numerator / denominator))


def warping_error(
    frames: list[np.ndarray],
    shifts: list[tuple[float, float]],
) -> float:
    """Temporal consistency proxy. prd.md §12.4.

    Warps each frame onto its successor using the known global shift and measures the mean absolute
    residual. A restoration that flickers scores worse than one that does not, even when both have
    the same per-frame PSNR — which is the entire reason §5.10 treats flicker as a first-class
    defect rather than polish.
    """
    if len(frames) < 2:
        return 0.0
    if len(shifts) < len(frames) - 1:
        raise ValueError("need one shift per consecutive frame pair")

    residuals = []
    for index in range(len(frames) - 1):
        dx, dy = shifts[index]
        warped = shift_bilinear(frames[index], dx, dy)
        residuals.append(float(np.mean(np.abs(warped - frames[index + 1]))))

    return float(np.mean(residuals))


def shift_bilinear(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Shifts an image by a sub-pixel amount with bilinear interpolation and edge replication."""
    height, width = image.shape[:2]

    ys, xs = np.mgrid[0:height, 0:width]
    src_x = np.clip(xs - dx, 0, width - 1)
    src_y = np.clip(ys - dy, 0, height - 1)

    x0 = np.floor(src_x).astype(np.int64)
    y0 = np.floor(src_y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)

    wx = src_x - x0
    wy = src_y - y0

    top = image[y0, x0] * (1 - wx) + image[y0, x1] * wx
    bottom = image[y1, x0] * (1 - wx) + image[y1, x1] * wx

    return top * (1 - wy) + bottom * wy
