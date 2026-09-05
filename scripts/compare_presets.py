"""Side by side: the same mosaicked frame through every preset, with the numbers beside it. D-43.

Builds the synthetic clip `test_endtoend_quality.py` uses - a screen-fixed mosaic over a panning
crop of Tears of Steel - runs it through Fast, Balanced and Quality, and writes one PNG with the
clean frame, the input and the three outputs in a row, cropped to the region so the block edges
are visible at 1:1. Prints PSNR and LPIPS per preset against the clean frame.

This exists because the question "is the invented detail worth having?" cannot be answered by
PSNR - an inventing restorer is *expected* to score worse on it than the blocky input while
looking better - and because nobody had looked at any output (D-33, D-36). Look.

Usage::

    .venv/Scripts/python.exe scripts/compare_presets.py [--frame 20] [--out artifacts/compare.png]

Prints ASCII only: the console this runs in is cp949.
"""

from __future__ import annotations

import argparse
import io
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "worker"))
sys.path.insert(0, str(REPO / "worker" / "tests"))
sys.path.insert(0, str(REPO / "scripts"))

import av  # noqa: E402
import test_endtoend_quality as T  # noqa: E402
from demosaic_worker.jobs import JobContext, JobRunner  # noqa: E402
from demosaic_worker.messages import Emitter  # noqa: E402
from demosaic_worker.metrics import psnr  # noqa: E402
from demosaic_worker.restore.ibp import block_average  # noqa: E402

PRESETS = ("Fast", "Balanced", "Quality")


def build_clip(directory: Path) -> tuple[Path, Path, np.ndarray]:
    mask = T._region()
    with av.open(str(T.CORPUS)) as container:
        source = [f.to_ndarray(format="rgb24")
                  for _, f in zip(range(T.FRAMES), container.decode(container.streams.video[0]))]
    clean, degraded = [], []
    for index, frame in enumerate(source):
        offset = min(index * T.PAN, 120)
        picture = frame[200:200 + T.HEIGHT, 300 + offset:300 + offset + T.WIDTH].copy()
        clean.append(picture)
        blocks = np.stack([block_average(picture[:, :, c].astype(np.float64), T.SPEC, (0, 0))
                           for c in range(3)], axis=2)
        degraded.append(np.where(mask[:, :, None], blocks, picture).astype(np.uint8))
    cp, ip = directory / "clean.mp4", directory / "input.mp4"
    T._write(cp, clean)
    T._write(ip, degraded)
    return cp, ip, mask


def run(preset: str, source: Path, directory: Path) -> Path:
    out = directory / f"out-{preset}.mp4"
    context = JobContext(job_id=preset, source_path=str(source), output_path=str(out), settings={
        "detection": {"confidence": 0.45, "maskThreshold": 0.5, "minRegionArea": 512},
        "restoration": {"preset": preset, "temporalWindow": "auto"},
        "encode": {"codec": "H264", "constantQuality": 14, "preset": "veryfast"},
        "modelVersion": "0.2.0",
    })
    JobRunner().run(context, Emitter(stream=io.StringIO()))
    return out


def frames_rgb(path: Path) -> list[np.ndarray]:
    with av.open(str(path)) as container:
        return [f.to_ndarray(format="rgb24") for f in container.decode(container.streams.video[0])]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="side-by-side of every preset on one frame")
    parser.add_argument("--frame", type=int, default=20)
    parser.add_argument("--out", type=Path, default=REPO / "artifacts" / "compare-presets.png")
    args = parser.parse_args(argv)

    from PIL import Image, ImageDraw

    directory = Path(tempfile.mkdtemp(prefix="compare-"))
    clean_path, input_path, mask = build_clip(directory)
    outputs = {preset: run(preset, input_path, directory) for preset in PRESETS}

    clean = frames_rgb(clean_path)
    degraded = frames_rgb(input_path)
    results = {p: frames_rgb(path) for p, path in outputs.items()}

    try:
        from perceptual import distance, is_available
        lpips_ok = is_available()
    except ImportError:
        lpips_ok = False

    ys, xs = np.nonzero(mask)
    top, bottom, left, right = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    pad = 16
    top, left = max(0, top - pad), max(0, left - pad)
    bottom, right = min(mask.shape[0], bottom + pad), min(mask.shape[1], right + pad)

    i = min(args.frame, len(clean) - 1, min(len(v) for v in results.values()) - 1)
    panels = [("clean", clean[i]), ("input (mosaic)", degraded[i])]
    panels += [(p, results[p][i]) for p in PRESETS]

    print("frame %d, region %dx%d" % (i, right - left, bottom - top))
    print("%-16s %9s %9s" % ("", "PSNR dB", "LPIPS"))
    grey = lambda a: a.astype(np.float64).mean(axis=2)  # noqa: E731
    for label, image in panels[1:]:
        p = psnr(grey(clean[i])[mask], grey(image)[mask])
        l = distance(clean[i][top:bottom, left:right], image[top:bottom, left:right]) if lpips_ok else float("nan")
        print("%-16s %9.2f %9.4f" % (label, p, l))
    if not lpips_ok:
        print("(LPIPS unavailable: pip install lpips)")

    scale = 2
    tiles = []
    for label, image in panels:
        crop = Image.fromarray(image[top:bottom, left:right]).resize(
            ((right - left) * scale, (bottom - top) * scale), Image.NEAREST)
        canvas = Image.new("RGB", (crop.width, crop.height + 22), "black")
        canvas.paste(crop, (0, 22))
        ImageDraw.Draw(canvas).text((4, 4), label, fill="white")
        tiles.append(canvas)
    sheet = Image.new("RGB", (sum(t.width for t in tiles) + 6 * (len(tiles) - 1), tiles[0].height), "black")
    x = 0
    for t in tiles:
        sheet.paste(t, (x, 0))
        x += t.width + 6
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
