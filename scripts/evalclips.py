"""The clips everything is measured on, defined once. prd.md section 1.4.1, D-27.

Every evaluation script used to name its own clip and rebuild its own ground-truth mask inline. That
is how the repository came to measure everything on the ladder input for weeks without noticing what
was in it: the mosaic there drifts 3 px per frame over content that moves 0.1 to 0.6, so 1.6% of
what the target loses is ever seen clean by its neighbour. That is the object-anchored regime
section 1.4.1 predicts to be unrecoverable.

So the clips live here, with what they are for written next to them, and the default is the one that
can actually demonstrate restoration.

**The masks are ground truth**, available only because the degradation is synthetic. Nothing here
transfers to real footage; it exists so that "did the restoration help" has an answer.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

REPO = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO / "artifacts"
CORPUS = REPO / "training" / "datasets" / "clean"
PYTHON = REPO / ".venv" / "Scripts" / "python.exe"


@dataclass(frozen=True, slots=True)
class Clip:
    """One evaluation clip: a clean reference, a mosaicked input, and where the mosaic is."""

    name: str
    what: str
    clean: Path
    degraded: Path

    #: Ground truth: the mosaicked region on a given frame, for a frame of the given shape.
    mask: Callable[[int, tuple[int, int]], np.ndarray]

    #: How much of the region a neighbour one frame back saw unmosaicked, as measured.
    coverage_per_frame: float

    #: What builds it, if it is missing: either a command to run or a callable.
    builder: list[str] | Callable[[], None] | None = None

    def ensure(self) -> "Clip":
        """Builds the clip if it is not on disk. Artifacts are gitignored, so this is normal."""
        if self.clean.exists() and self.degraded.exists():
            return self

        if self.builder is None:
            raise SystemExit(
                f"{self.name} is missing and has no builder: {self.clean}, {self.degraded}"
            )

        print(f"building {self.name}...", file=sys.stderr, flush=True)
        if callable(self.builder):
            self.builder()
        else:
            subprocess.run([str(PYTHON), *self.builder], check=True, cwd=str(REPO))
        return self


def _ellipse(shape: tuple[int, int], centre: tuple[int, int], radii: tuple[int, int]) -> np.ndarray:
    height, width = shape
    ys, xs = np.mgrid[0:height, 0:width]
    return (((ys - centre[0]) / radii[0]) ** 2 + ((xs - centre[1]) / radii[1]) ** 2) <= 1.0


def _screen_mask(_index: int, shape: tuple[int, int]) -> np.ndarray:
    """A fixed ellipse in screen coordinates. It never moves; the content moves past it."""
    height, width = shape
    return _ellipse(shape, (height // 2, width // 2), (110, 150))


def _ladder_mask(index: int, shape: tuple[int, int]) -> np.ndarray:
    """The drifting ellipse `eval_endtoend.py` paints: 3 px per frame in x."""
    height, width = shape
    return _ellipse(shape, (height // 2, width // 3 + index * 3), (110, 150))


def _build_ladder() -> None:
    """Paints the drifting ellipse onto a corpus clip, the way the first ladder did.

    Kept because every number in this repository before 2026-08-23 was measured on it, and a
    comparison against those is only meaningful against the same input.
    """
    import av
    sys.path.insert(0, str(REPO / "training"))
    sys.path.insert(0, str(REPO / "worker"))
    from degradation.mosaic import pixelate
    from demosaic_worker.analyze.profile import GridAnchor, MosaicProfile

    spec = MosaicProfile(block_width=10, block_height=10, anchor=GridAnchor.SCREEN)
    source = CORPUS / "tos_002.mp4"
    destination = ARTIFACTS / "ladder_input_tos_002.mp4"
    destination.parent.mkdir(parents=True, exist_ok=True)

    with av.open(str(source)) as container, av.open(str(destination), mode="w") as out:
        stream = container.streams.video[0]
        encoder = out.add_stream("libx264", rate=stream.average_rate)
        encoder.width = stream.codec_context.width
        encoder.height = stream.codec_context.height
        encoder.pix_fmt = "yuv420p"
        encoder.time_base = stream.time_base
        encoder.options = {"crf": "18", "preset": "medium"}

        for index, frame in enumerate(container.decode(stream)):
            rgb = frame.to_ndarray(format="rgb24")
            region = _ladder_mask(index, rgb.shape[:2])
            blocks = np.stack([pixelate(rgb[:, :, c], spec) for c in range(3)], axis=2)
            rgb = np.where(region[:, :, None], blocks, rgb).astype(np.uint8)

            picture = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            picture.pts = frame.pts
            picture.time_base = frame.time_base
            for packet in encoder.encode(picture):
                out.mux(packet)
        for packet in encoder.encode():
            out.mux(packet)


SCREEN = Clip(
    name="screen",
    what="content pans 4 px/frame past a mosaic that does not move - the case multi-frame is for",
    clean=ARTIFACTS / "screen_clean.mp4",
    degraded=ARTIFACTS / "screen_input.mp4",
    mask=_screen_mask,
    coverage_per_frame=0.018,
    builder=["scripts/make_screen_clip.py"],
)

LADDER = Clip(
    name="ladder",
    what="the mosaic tracks near-static content - the regime section 1.4.1 calls unrecoverable",
    clean=CORPUS / "tos_002.mp4",
    degraded=ARTIFACTS / "ladder_input_tos_002.mp4",
    mask=_ladder_mask,
    coverage_per_frame=0.016,
    builder=_build_ladder,
)

#: Keyed by the name `--clip` takes. The screen-anchored one is the default: it is the only one
#: where restoration has anything to work with, and the numbers that matter are measured on it.
CLIPS = {clip.name: clip for clip in (SCREEN, LADDER)}
DEFAULT = SCREEN.name


def resolve(name: str) -> Clip:
    """Looks a clip up by name and builds it if it is missing."""
    if name not in CLIPS:
        raise SystemExit(f"unknown clip {name!r}; known: {', '.join(sorted(CLIPS))}")
    return CLIPS[name].ensure()


def add_argument(parser) -> None:
    """Adds `--clip` with the shared choices, so every script spells it the same way."""
    parser.add_argument(
        "--clip",
        default=DEFAULT,
        choices=sorted(CLIPS),
        help="; ".join(f"{clip.name}: {clip.what}" for clip in CLIPS.values()),
    )
