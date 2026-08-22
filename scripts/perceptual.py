"""Perceptual distance, for evaluation only. prd.md section 1.4.3, section 12.3.

section 1.4.3 asks for PSNR, LPIPS and a warping error together, and until now only PSNR existed. That
mattered more than it sounds: PSNR is a weak proxy for a restoration, and a restorer that invents
plausible detail should be expected to score *worse* on it than the blocky input while looking
better. Every quality decision this project has made rested on the one metric that cannot see that.

**This is not part of the shipped worker.** LPIPS pulls a 233 MB AlexNet backbone, and the worker
has no business carrying a perceptual model to restore a video. It lives in `scripts/` so that
evaluation can use it and the product does not.

The model is loaded once, lazily, because loading it costs a second and most runs of most scripts do
not need it.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

#: Zhang et al. 2018. AlexNet trunk, which is the paper's default and the cheapest of the three.
TRUNK = "alex"


@lru_cache(maxsize=1)
def _model():
    """Loads LPIPS once. Imports inside so a script that never scores never pays for torch."""
    import warnings

    import lpips

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return lpips.LPIPS(net=TRUNK).eval()


def is_available() -> bool:
    """True when LPIPS can be scored here. Its weights are part of the machine, not the repository."""
    try:
        import lpips  # noqa: F401
    except ImportError:
        return False

    return True


def _to_tensor(image: np.ndarray):
    """RGB uint8 or float in [0, 255] to the [-1, 1] NCHW tensor LPIPS wants."""
    import torch

    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)

    scaled = image.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(scaled).permute(2, 0, 1)[None]


def distance(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Perceptual distance between two RGB images. **Lower is better**, unlike every other number here.

    Raises ``ImportError`` when LPIPS is not installed, rather than returning a sentinel that would
    be averaged into a table and read as a measurement.
    """
    import torch

    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} against {candidate.shape}")

    with torch.no_grad():
        return float(_model()(_to_tensor(reference), _to_tensor(candidate)).item())


def distance_over(references: list[np.ndarray], candidates: list[np.ndarray],
                  stride: int = 1) -> float:
    """Mean distance over a sequence, sampling every ``stride`` frames.

    Sampling is offered because LPIPS is far slower than PSNR and a 96-frame clip does not need
    every frame to say which of two outputs is closer.
    """
    if stride < 1:
        raise ValueError(f"stride must be at least 1, got {stride}")

    count = min(len(references), len(candidates))
    scores = [distance(references[i], candidates[i]) for i in range(0, count, stride)]

    return float(np.mean(scores)) if scores else 0.0
