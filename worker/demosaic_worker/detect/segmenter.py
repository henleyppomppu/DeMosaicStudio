"""Mosaic segmentation at inference time. prd.md §5.2.2, §14.

Loads a versioned model from the store and produces a probability map per frame.

Two things here are requirements rather than conveniences.

**The model is identified by hash, not by path.** `job.json` records the hash that produced each
artifact (§14.1 R-14.1a), and a resume with a different model must discard the work rather than mix
outputs from two models in one file. A loader that accepted "whatever is at this path" would make
that impossible to enforce.

**Loading failures are numbered and immediate.** E3001 before any decoding starts (§14.3): a job
that fails after minutes of decode because a weights file was missing is the exact failure the
model check exists to prevent.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..errors import E3001, E3003, WorkerError

#: Inference resolution: fixed short side, aspect preserved (§5.2.2).
INFERENCE_SHORT_SIDE = 512

#: Tile size and overlap for inputs larger than the model comfortably takes.
TILE = 512
TILE_OVERLAP = 64

#: The network downsamples by 2 per level; inputs must be a multiple of this.
SIZE_MULTIPLE = 16


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """What was loaded, for the record in `job.json` (§9.2, §14.1)."""

    model_id: str
    version: str
    sha256: str
    path: str
    trained_on: str
    notes: str

    def as_dict(self) -> dict[str, str]:
        """The form recorded in the checkpoint."""
        return {
            "id": self.model_id,
            "version": self.version,
            "sha256": self.sha256,
            "trainedOn": self.trained_on,
        }


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_model_info(directory: Path) -> ModelInfo:
    """Reads a model directory's metadata and verifies the weights hash.

    A mismatch is E3001, not a warning: silently running a model whose weights are not the ones the
    metadata describes makes every downstream number untraceable.
    """
    metadata_path = directory / "metadata.json"
    weights_path = directory / "model.pt"

    if not metadata_path.exists() or not weights_path.exists():
        raise WorkerError(E3001, f"incomplete model directory: {directory.name}", path=str(directory))

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    actual = _digest(weights_path)
    expected = metadata.get("sha256")

    if expected and expected != actual:
        raise WorkerError(
            E3001,
            f"model weights do not match metadata for {metadata.get('id')}",
            expected=expected,
            actual=actual,
        )

    return ModelInfo(
        model_id=metadata["id"],
        version=metadata["version"],
        sha256=actual,
        path=str(weights_path),
        trained_on=metadata.get("trainedOn", "unknown"),
        notes=metadata.get("notes", ""),
    )


class Segmenter:
    """Runs the mosaic segmentation network over frames.

    Constructed once per job. Torch is imported lazily so that the media and policy layers stay
    usable — and testable — on a machine without it.
    """

    def __init__(self, model_directory: Path, *, device: str = "auto", threshold: float = 0.5) -> None:
        import torch

        from .unet import MosaicUNet

        self.info = load_model_info(model_directory)
        self.threshold = threshold

        resolved = (
            ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        )
        self.device = torch.device(resolved)

        try:
            checkpoint = torch.load(self.info.path, map_location=self.device, weights_only=True)
        except Exception as exc:  # noqa: BLE001 - any load failure is E3001
            raise WorkerError(E3001, f"cannot load model weights: {exc}", path=self.info.path) from exc

        width = int(checkpoint.get("width", 32))
        self.model = MosaicUNet(width=width).to(self.device)

        try:
            self.model.load_state_dict(checkpoint["state_dict"])
        except Exception as exc:  # noqa: BLE001
            raise WorkerError(E3003, f"model and architecture disagree: {exc}") from exc

        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self._torch = torch

    def probability(self, luma: np.ndarray) -> np.ndarray:
        """Returns a per-pixel mosaic probability map at the input resolution.

        Large frames are tiled with overlap and averaged in the seams: a hard tile boundary would
        put a straight edge into the mask, and §5.11 blends on the mask, so that edge would end up
        in the picture.
        """
        if luma.ndim != 2:
            raise ValueError(f"expected a 2-D luma frame, got {luma.shape}")

        height, width = luma.shape
        if height <= TILE and width <= TILE:
            return self._infer(luma)

        accumulator = np.zeros((height, width), dtype=np.float64)
        weights = np.zeros((height, width), dtype=np.float64)
        step = TILE - TILE_OVERLAP

        for top in range(0, max(1, height - TILE_OVERLAP), step):
            for left in range(0, max(1, width - TILE_OVERLAP), step):
                bottom = min(top + TILE, height)
                right = min(left + TILE, width)
                tile_top = max(0, bottom - TILE)
                tile_left = max(0, right - TILE)

                tile = luma[tile_top:bottom, tile_left:right]
                accumulator[tile_top:bottom, tile_left:right] += self._infer(tile)
                weights[tile_top:bottom, tile_left:right] += 1.0

        return accumulator / np.maximum(weights, 1.0)

    def _infer(self, luma: np.ndarray) -> np.ndarray:
        torch = self._torch

        height, width = luma.shape
        pad_h = (-height) % SIZE_MULTIPLE
        pad_w = (-width) % SIZE_MULTIPLE

        padded = np.pad(luma, ((0, pad_h), (0, pad_w)), mode="reflect") if (pad_h or pad_w) else luma

        tensor = torch.from_numpy(np.ascontiguousarray(padded, dtype=np.float32) / 255.0)
        tensor = tensor[None, None].to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probability = torch.sigmoid(logits)[0, 0].cpu().numpy()

        return probability[:height, :width].astype(np.float64)
