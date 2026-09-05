"""A diffusion refiner over the single-frame restoration. D-44.

Runs a user-chosen image-to-image diffusion pipeline over each restored region at low strength,
with the region's own restoration (bicubic or the SR network) as the init image. Measured on the
quality fixture at strength 0.2: LPIPS 0.161 → 0.092 against bicubic alone, PSNR −1.7 dB (the price
an inventing restorer pays), flicker under what real motion costs. At 0.3 it begins to grow shapes
that are not there; at 0.5 it invents.

**Everything here is a guess dressed as a picture.** §1.3 and C-3 apply with full force: the
output is synthetic and is labelled so. C-4 is satisfied by construction — there is no prompt,
no reference image, no way to steer the model toward anyone. The models themselves are not part
of the product: the user puts them in ``models/diffusion``, ``models/lora`` and
``models/embeddings`` and picks by name. An embedding trained to reproduce a person is out of
scope; the tool cannot tell and the person choosing can.

**Luma in, luma out.** The pipeline works on luma and leaves chroma as decoded (D-24: the RGB
round trip cost 45 dB). The diffusion model wants colour, so the region is rebuilt as RGB from
the restored luma and the frame's own chroma, refined, and only the *change in luma* is taken
back — ``luma(refined) − luma(input)`` under one matrix, so any matrix bias cancels rather than
shifting the region's brightness every frame.

``diffusers`` is an optional dependency of the shipped engine. Absent, an enabled refiner emits
W6101 once and the job proceeds unrefined; a refiner nobody enabled costs nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import E4001, W6101, WorkerError

#: SD1.5-class models were trained at this size and do not work far below it. Regions are
#: upscaled by an integer factor to about this on the long side, refined, and brought back down.
WORK_SIZE = 512

#: Classifier-free guidance. LCM models want 1–2; the probe found 1.5 fine and the prompt is
#: empty anyway, so there is little to guide toward.
GUIDANCE = 1.5


@dataclass(frozen=True, slots=True)
class RefineSettings:
    """What the user chose. Mirrors ``restoration.refine`` in the protocol (1.3)."""

    enabled: bool = False
    strength: float = 0.2
    model: str = ""
    lora: str | None = None
    #: Embeddings whose token goes into the positive prompt. The prompt is *only* these tokens.
    embeddings: tuple[str, ...] = ()
    #: Embeddings whose token goes into the negative prompt - quality suppressors like
    #: EasyNegative. Same rule: tokens only, never free text.
    negative_embeddings: tuple[str, ...] = ()
    steps: int = 8
    seed: int = 7
    #: Where the folders live; empty means the default beside the worker. Not fingerprinted:
    #: the folder does not change the output, the names chosen from it do.
    store_root: str = ""

    @classmethod
    def from_settings(cls, settings: dict[str, Any]) -> "RefineSettings":
        """Reads the ``restoration.refine`` object, tolerating its absence."""
        refine = (settings.get("restoration") or {}).get("refine") or {}
        lora = refine.get("lora") or None
        embeddings = refine.get("embeddings") or []
        negatives = refine.get("negativeEmbeddings") or []
        return cls(
            enabled=bool(refine.get("enabled", False)),
            strength=float(np.clip(float(refine.get("strength", 0.2)), 0.0, 1.0)),
            model=str(refine.get("model") or ""),
            lora=str(lora) if lora else None,
            embeddings=tuple(str(e) for e in embeddings if e),
            negative_embeddings=tuple(str(e) for e in negatives if e),
            steps=max(1, int(refine.get("steps", 8))),
            seed=int(refine.get("seed", 7)),
            store_root=str(refine.get("storeRoot") or "").strip(),
        )


def _luma_of_rgb(rgb: np.ndarray) -> np.ndarray:
    """BT.601 luma of an RGB float array. Only ever used as a *difference*, so the exact
    matrix does not matter as long as both sides use the same one."""
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def rgb_from_yuv(luma: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Studio-range BT.601 YUV (u, v at the luma's resolution) to RGB in 0..255, float."""
    y = (luma.astype(np.float64) - 16.0) * (255.0 / 219.0)
    cb = u.astype(np.float64) - 128.0
    cr = v.astype(np.float64) - 128.0
    r = y + 1.596 * cr
    g = y - 0.392 * cb - 0.813 * cr
    b = y + 2.017 * cb
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 255.0)


class DiffusionRefiner:
    """Loads a pipeline once per job and refines regions on request.

    Construction is cheap; the pipeline loads on first use so a job that gates every region
    never pays for it. A missing model directory is E4001 at first use, like the detector.
    """

    def __init__(self, store: Path, settings: RefineSettings) -> None:
        self.store = store
        self.settings = settings
        self._pipe: Any = None
        self._torch: Any = None
        self._unavailable: str | None = None

    @property
    def available(self) -> bool:
        """False once loading failed for a reason that will not change during the job."""
        return self._unavailable is None

    def _load(self) -> Any:
        if self._pipe is not None:
            return self._pipe
        try:
            import torch
            from diffusers import AutoPipelineForImage2Image, LCMScheduler
        except ImportError as exc:
            self._unavailable = f"diffusers is not installed ({exc.name})"
            raise WorkerError(E4001, self._unavailable) from exc

        model_dir = self.store / "diffusion" / self.settings.model
        if not self.settings.model or not model_dir.exists():
            self._unavailable = f"no diffusion model named {self.settings.model!r} in {self.store / 'diffusion'}"
            raise WorkerError(E4001, self._unavailable)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        try:
            single = next(model_dir.glob("*.safetensors"), None) if not (model_dir / "model_index.json").exists() else None
            if single is not None:
                from diffusers import StableDiffusionImg2ImgPipeline

                pipe = StableDiffusionImg2ImgPipeline.from_single_file(
                    str(single), torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
                )
            else:
                kwargs: dict[str, Any] = {"torch_dtype": dtype, "safety_checker": None, "requires_safety_checker": False}
                # fp16 variants are what fetch_diffusion.py keeps; a hand-copied repository may have full ones.
                if any(model_dir.rglob("*.fp16.safetensors")):
                    kwargs["variant"] = "fp16"
                pipe = AutoPipelineForImage2Image.from_pretrained(str(model_dir), **kwargs)
            pipe = pipe.to(device)

            if self.settings.lora:
                lora = self._find(self.store / "lora", self.settings.lora)
                pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
                pipe.load_lora_weights(str(lora.parent), weight_name=lora.name)
                pipe.fuse_lora()
            for name in (*self.settings.embeddings, *self.settings.negative_embeddings):
                embedding = self._find(self.store / "embeddings", name)
                pipe.load_textual_inversion(str(embedding), token=embedding.stem)
        except WorkerError:
            raise
        except Exception as exc:  # noqa: BLE001 - any load failure is E4001
            self._unavailable = f"cannot load diffusion pipeline {self.settings.model!r}: {exc}"
            raise WorkerError(E4001, self._unavailable) from exc

        pipe.set_progress_bar_config(disable=True)
        self._pipe, self._torch = pipe, torch
        return pipe

    @staticmethod
    def _find(directory: Path, name: str) -> Path:
        """A file by name, with or without its extension."""
        for candidate in (directory / name, *directory.glob(f"{name}.*")):
            if candidate.is_file():
                return candidate
        raise WorkerError(E4001, f"no file named {name!r} in {directory}")

    def refine_luma(self, luma: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Refines one region. Returns luma of the same shape; raises E4001 if the pipeline cannot load.

        ``u`` and ``v`` are the region's chroma at the luma's resolution (the caller upsamples the
        4:2:0 planes). Only the luma change is returned into the pipeline; chroma stays decoded.
        """
        pipe = self._load()
        torch = self._torch
        from PIL import Image

        rgb = rgb_from_yuv(luma, u, v)
        h, w = luma.shape
        factor = max(1, round(WORK_SIZE / max(h, w)))
        big = Image.fromarray(rgb.astype(np.uint8)).resize((w * factor, h * factor), Image.BICUBIC)
        pad_w, pad_h = (-big.width) % 8, (-big.height) % 8
        if pad_w or pad_h:
            padded = Image.new("RGB", (big.width + pad_w, big.height + pad_h))
            padded.paste(big, (0, 0))
            big = padded

        # The prompts are the chosen embeddings' tokens and nothing else. There is no text field
        # anywhere in the product, so nothing a person types can reach here (section 2.3 C-4);
        # what *can* reach here is the name of a file the user placed in the embeddings folder.
        generator = torch.Generator(pipe.device).manual_seed(self.settings.seed)
        out = pipe(
            prompt=" ".join(self.settings.embeddings),
            negative_prompt=" ".join(self.settings.negative_embeddings) or None,
            image=big,
            strength=self.settings.strength,
            num_inference_steps=self.settings.steps,
            guidance_scale=GUIDANCE,
            generator=generator,
        ).images[0]
        out = out.crop((0, 0, big.width - pad_w, big.height - pad_h)).resize((w, h), Image.LANCZOS)
        refined = np.asarray(out, dtype=np.float64)

        delta = _luma_of_rgb(refined) - _luma_of_rgb(rgb)
        return np.clip(luma.astype(np.float64) + delta, 0.0, 255.0)


def chroma_for(planes: np.ndarray, plane_height: int, bounds: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """The region's U and V from a yuv420p plane stack, upsampled to luma resolution.

    PyAV's ``to_ndarray(format="yuv420p")`` stacks Y (H rows) then U and V (H/4 rows each, W wide,
    holding a (H/2, W/2) plane row-major). This unpacks that layout and nearest-upsamples.
    """
    left, top, right, bottom = bounds
    height, width = plane_height, planes.shape[1]
    ch, cw = height // 2, width // 2
    chroma = planes[plane_height:].ravel()
    u = chroma[: ch * cw].reshape(ch, cw)
    v = chroma[ch * cw : 2 * ch * cw].reshape(ch, cw)

    def up(plane: np.ndarray) -> np.ndarray:
        full = np.repeat(np.repeat(plane, 2, axis=0), 2, axis=1)[:height, :width]
        # The luma crop is reflect-padded past the frame edge to an alignment multiple
        # (`Roi.crop_bounds`); the chroma must be padded the same way or the two disagree in
        # shape at the edge of the picture - measured as "operands could not be broadcast
        # together with shapes (160,176) (160,167)" on a region touching the right edge.
        inside = full[max(top, 0):min(bottom, height), max(left, 0):min(right, width)]
        pads = ((max(-top, 0), max(bottom - height, 0)), (max(-left, 0), max(right - width, 0)))
        if any(a or b for a, b in pads):
            mode = "reflect" if all(s > max(a, b) for s, (a, b) in zip(inside.shape, pads)) else "edge"
            inside = np.pad(inside, pads, mode=mode)
        return inside

    return up(u), up(v)


@dataclass
class RefineStats:
    """What happened, for the summary and the log."""

    regions: int = 0
    seconds: float = 0.0
    warned: bool = False
