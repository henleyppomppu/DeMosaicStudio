"""Single-frame restoration: decimate the mosaic to its true resolution, then upscale. D-43.

A mosaic of block size B **is** a B-times downsample followed by nearest-neighbour upsampling. Every
block carries one number. So the honest representation of a mosaicked crop is the small image of
block means — and from there the problem is ordinary super-resolution, which is a problem other
people have trained on millions of photographs.

That reframing is what makes third-party weights usable here. D-04 rejected them because they were
trained to invert bicubic downsampling, not pixelation, and would produce confidently wrong texture
when handed a mosaic. Handed the *decimated* image instead, they are given exactly the input they
were trained on. D-43 records the revision.

Two backends, one interface:

* :func:`bicubic_restore` — no model at all. Decimate, interpolate back. About a millisecond, and
  it removes the grid; what it leaves is blur. This is the floor, and it is what runs when no
  weights are installed.
* :class:`Upscaler` — a compact SR network (D-43 names which) over the same decimated input, all
  of a frame's regions in one batch. Invents plausible detail. **Every restored pixel is a guess**:
  no neighbouring frame is consulted, so nothing here is evidence in the §7.4 sense, and the
  confidence it reports says so.

Everything works on luma. The RGB round trip cost 45 dB on its own when it was measured
(docs/phase3-endtoend-report.md), and the network does not need colour to invent texture: a
grey image replicated to three channels is a valid photograph.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..analyze.profile import MosaicProfile
from ..errors import E4001, E4002, WorkerError
from .ibp import grid_edges

#: The model directory under ``models/restorer/``. One id, not a setting: the preset chooses
#: whether a network runs at all, and which one is a decision recorded in D-43, not a knob.
RESTORER_ID = "realesr-general-x4v3"

#: The upscaling factor the network was trained for. Outputs are resized to the crop afterwards,
#: so a block size other than this is handled by interpolation on top, not by a different model.
NETWORK_SCALE = 4


def decimate(image: np.ndarray, spec: MosaicProfile, phase: tuple[int, int]) -> np.ndarray:
    """One value per block: the mosaic at its true resolution.

    Uses the same grid edges as the forward operator in :mod:`ibp`, so a partial block at the edge
    of the crop is averaged over the pixels it actually covers rather than assumed full-size.
    """
    height, width = image.shape
    y_edges = grid_edges(height, spec.block_height, phase[1])
    x_edges = grid_edges(width, spec.block_width, phase[0])

    data = image.astype(np.float64)
    sums = np.add.reduceat(np.add.reduceat(data, y_edges, axis=0), x_edges, axis=1)
    counts = np.outer(np.diff(np.append(y_edges, height)), np.diff(np.append(x_edges, width)))

    return sums / counts


def resize(image: np.ndarray, size: tuple[int, int], *, device: Any = None) -> np.ndarray:
    """Bicubic resize to ``(height, width)``. Torch rather than PIL so it can stay on the GPU."""
    import torch
    import torch.nn.functional as functional

    height, width = size
    tensor = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))[None, None]
    if device is not None:
        tensor = tensor.to(device)

    out = functional.interpolate(
        tensor, size=(height, width), mode="bicubic", align_corners=False, antialias=True
    )
    return out[0, 0].clamp_(0.0, 255.0).cpu().numpy().astype(np.float64)


def bicubic_restore(crop: np.ndarray, spec: MosaicProfile, phase: tuple[int, int]) -> np.ndarray:
    """The floor: decimate and interpolate back. Removes the grid, leaves blur, costs nothing."""
    return resize(decimate(crop, spec, phase), crop.shape)


class SRVGGNetCompact:
    """The compact Real-ESRGAN generator (``realesr-general-x4v3``), written in-house.

    A stack of 3x3 convolutions with PReLU, a pixel-shuffle upsampler, and a nearest-neighbour
    skip from the input. Reimplemented rather than imported so the shipped worker carries no
    dependency on the ``realesrgan`` package and its transitive pins; the architecture is thirty
    lines and the weights are the only thing that matters.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        import torch
        from torch import nn

        class _Net(nn.Module):
            def __init__(
                self,
                num_in_ch: int = 3,
                num_out_ch: int = 3,
                num_feat: int = 64,
                num_conv: int = 32,
                upscale: int = NETWORK_SCALE,
            ) -> None:
                super().__init__()
                self.upscale = upscale
                layers: list[nn.Module] = [nn.Conv2d(num_in_ch, num_feat, 3, 1, 1), nn.PReLU(num_feat)]
                for _ in range(num_conv):
                    layers += [nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.PReLU(num_feat)]
                layers.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
                self.body = nn.Sequential(*layers)
                self.upsampler = nn.PixelShuffle(upscale)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = self.upsampler(self.body(x))
                base = torch.nn.functional.interpolate(x, scale_factor=self.upscale, mode="nearest")
                return out + base

        return _Net(*args, **kwargs)


@dataclass(frozen=True, slots=True)
class UpscalerInfo:
    """What was loaded, for `job.json`."""

    model_id: str
    version: str
    sha256: str

    def as_dict(self) -> dict[str, str]:
        """The form recorded in the checkpoint."""
        return {"id": self.model_id, "version": self.version, "sha256": self.sha256}


class Upscaler:
    """Batched single-frame super-resolution over decimated mosaic crops.

    Constructed once per job. Loading failures are E4001 before any frame is decoded (§14.3),
    the same rule the detector follows.
    """

    def __init__(self, model_directory: Path, *, device: str = "auto") -> None:
        import torch

        from ..detect.segmenter import load_model_info

        info = load_model_info(model_directory, error_code=E4001)
        self.info = UpscalerInfo(info.model_id, info.version, info.sha256)

        resolved = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.device = torch.device(resolved)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        try:
            checkpoint = torch.load(info.path, map_location=self.device, weights_only=True)
        except Exception as exc:  # noqa: BLE001 - any load failure is E4001
            raise WorkerError(E4001, f"cannot load restorer weights: {exc}", path=info.path) from exc

        # Real-ESRGAN releases store the weights under "params_ema"/"params"; our own store uses
        # "state_dict". Accept both so a converted file and an as-downloaded one both load.
        state = next(
            (checkpoint[key] for key in ("state_dict", "params_ema", "params") if key in checkpoint),
            checkpoint,
        )

        self.model = SRVGGNetCompact(
            num_feat=int(checkpoint.get("num_feat", 64)),
            num_conv=int(checkpoint.get("num_conv", 32)),
        ).to(self.device)
        try:
            self.model.load_state_dict(state)
        except Exception as exc:  # noqa: BLE001
            raise WorkerError(E4001, f"restorer weights and architecture disagree: {exc}") from exc

        self.model.eval().to(self.dtype)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self._torch = torch

    def restore_many(
        self,
        crops: list[np.ndarray],
        specs: list[MosaicProfile],
        phases: list[tuple[int, int]],
    ) -> list[np.ndarray]:
        """Restores every crop, in one forward pass, each returned at its own crop size.

        The decimated inputs differ in size from region to region. They are reflect-padded to the
        largest in the batch, run together, and each output is cut back to its own size before the
        final resize — padding to a common size is what lets N regions cost one launch rather
        than N.
        """
        if not crops:
            return []

        torch = self._torch
        functional = torch.nn.functional

        smalls = [decimate(crop, spec, phase) for crop, spec, phase in zip(crops, specs, phases)]
        max_h = max(s.shape[0] for s in smalls)
        max_w = max(s.shape[1] for s in smalls)

        batch = np.zeros((len(smalls), 3, max_h, max_w), dtype=np.float32)
        for i, small in enumerate(smalls):
            h, w = small.shape
            # Edge padding: reflect needs the image to be wider than the pad, and a decimated crop
            # can be a handful of pixels. The padded area is cut away before the resize anyway.
            padded = np.pad(small, ((0, max_h - h), (0, max_w - w)), mode="edge")
            batch[i] = padded[None].repeat(3, axis=0) / 255.0

        tensor = torch.from_numpy(batch).to(self.device, self.dtype)
        try:
            with torch.no_grad():
                out = self.model(tensor).float()
        except Exception as exc:  # noqa: BLE001 - one frame failing is not the job failing
            raise WorkerError(E4002, f"restorer inference failed: {exc}") from exc

        results: list[np.ndarray] = []
        for i, (small, crop) in enumerate(zip(smalls, crops)):
            h, w = small.shape
            region = out[i : i + 1, :, : h * NETWORK_SCALE, : w * NETWORK_SCALE].mean(dim=1, keepdim=True)
            region = functional.interpolate(
                region, size=crop.shape, mode="bicubic", align_corners=False, antialias=True
            )
            results.append((region[0, 0] * 255.0).clamp_(0.0, 255.0).cpu().numpy().astype(np.float64))

        return results
