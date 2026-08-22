"""Builds a screen-anchored evaluation clip. prd.md section 1.4.1, D-27.

Every end-to-end number in this repository was measured on the ladder input, and the ladder input
puts the mosaic in the regime the design says is hopeless. Measured with the exact ellipse and
motion taken from aligning the clean frames: the mask moves 3 px per frame over content that moves
0.1 to 0.6 px per frame, so **1.6%** of what the target lost was ever seen clean by its neighbour.
The mosaic slides over near-static content and the same picture is covered in every frame.

This one pans the content past a mosaic that does not move, which is what a screen-anchored mosaic
on moving footage actually looks like - and it is the only case where multi-frame restoration has
anything to work with. A neighbour d pixels away exposes about 2*d / (pi * rx) of the region, so the
evidence accumulates with the window rather than arriving all at once.

It is synthetic in a second way and the limitation is worth keeping in view: the pan is a crop of a
real frame, so the motion is a pure translation with no parallax, occlusion or object motion. It
exercises the mechanism; it does not represent real footage.

Usage:

    .venv/Scripts/python.exe scripts/make_screen_clip.py
    .venv/Scripts/python.exe scripts/make_screen_clip.py --pan 8 --block 14
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import av
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "worker"))
sys.path.insert(0, str(REPO / "training"))

from degradation.mosaic import pixelate  # noqa: E402
from demosaic_worker.analyze.profile import GridAnchor, MosaicProfile  # noqa: E402

CORPUS = REPO / "training" / "datasets" / "clean"
ARTIFACTS = REPO / "artifacts"


def elliptical_mask(shape: tuple[int, int], ry: int, rx: int) -> np.ndarray:
    """A fixed ellipse in **screen** coordinates. It never moves; the content moves past it."""
    height, width = shape
    ys, xs = np.mgrid[0:height, 0:width]
    return (((ys - height // 2) / ry) ** 2 + ((xs - width // 2) / rx) ** 2) <= 1.0


def coverage_per_neighbour(pan: int, rx: int) -> float:
    """The fraction of the region a neighbour one frame away saw unmosaicked.

    A shift of ``d`` exposes a crescent of roughly ``2 * d * ry`` out of ``pi * rx * ry``. The ``ry``
    cancels, which is why the height of the mosaic does not matter and its width does.
    """
    return 2.0 * pan / (np.pi * rx)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--clip", default="tos_002.mp4")
    parser.add_argument("--pan", type=int, default=4, help="content motion in px per frame")
    parser.add_argument("--block", type=int, default=10)
    parser.add_argument("--ry", type=int, default=110)
    parser.add_argument("--rx", type=int, default=150)
    parser.add_argument("--margin", type=int, default=120, help="crop margin, sets the pan budget")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--prefix", default="screen")
    args = parser.parse_args(argv)

    source = CORPUS / args.clip
    if not source.exists():
        print(f"missing: {source}", file=sys.stderr)
        return 2

    spec = MosaicProfile(block_width=args.block, block_height=args.block, anchor=GridAnchor.SCREEN)

    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
        rate, time_base = stream.average_rate, stream.time_base

    if not frames:
        print(f"{args.clip} decoded to nothing", file=sys.stderr)
        return 1

    def pan_to(frame: np.ndarray, offset: int) -> np.ndarray:
        _, width, _ = frame.shape
        start = args.margin + offset - args.margin
        return frame[:, start:start + (width - 2 * args.margin)]

    height, width, _ = pan_to(frames[0], 0).shape
    mask = elliptical_mask((height, width), args.ry, args.rx)
    budget = 2 * args.margin - 4

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    outputs = {
        False: ARTIFACTS / f"{args.prefix}_clean.mp4",
        True: ARTIFACTS / f"{args.prefix}_input.mp4",
    }

    for degrade, destination in outputs.items():
        with av.open(str(destination), mode="w") as out:
            encoder = out.add_stream("libx264", rate=rate)
            encoder.width, encoder.height = width, height
            encoder.pix_fmt = "yuv420p"
            encoder.time_base = time_base
            encoder.options = {"crf": str(args.crf), "preset": "medium"}

            for index, frame in enumerate(frames):
                picture = pan_to(frame, min(index * args.pan, budget))
                if degrade:
                    blocks = np.stack(
                        [pixelate(picture[:, :, c], spec) for c in range(3)], axis=2
                    )
                    picture = np.where(mask[:, :, None], blocks, picture).astype(np.uint8)

                encoded_frame = av.VideoFrame.from_ndarray(
                    np.ascontiguousarray(picture), format="rgb24"
                )
                encoded_frame.pts = index
                encoded_frame.time_base = time_base
                for packet in encoder.encode(encoded_frame):
                    out.mux(packet)

            for packet in encoder.encode():
                out.mux(packet)

    per_neighbour = coverage_per_neighbour(args.pan, args.rx)
    print(f"wrote {outputs[False].name} and {outputs[True].name}")
    print(f"  {width}x{height}, {len(frames)} frames, {args.block} px blocks")
    print(f"  content pans {args.pan} px/frame; the mosaic does not move")
    print(f"  a neighbour sees about {per_neighbour:.1%} of the region unmosaicked, so a window of")
    print(f"  K needs about {int(np.ceil(0.28 / per_neighbour)) + 1} neighbours to reach the 28%")
    print("  coverage where the mask model starts winning (D-26)")

    if min(len(frames) * args.pan, budget) < budget:
        print(f"  note: the pan runs out of crop margin after {budget // args.pan} frames")

    return 0


if __name__ == "__main__":
    sys.exit(main())
