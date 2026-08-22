"""Cuts a source film into stratified clean clips. prd.md §11.2, §11.5, §11.6.

The clips are the *ground truth*: the degradation generator applies a mosaic to them, and the gate
compares a restoration against the untouched original. So two things matter here and nothing else
does.

**Stratification.** Each clip's global motion is measured, not guessed, and recorded in the
manifest. The feasibility question (§1.4.1) is entirely about whether the subject moves across a
screen-anchored grid, so a corpus that is accidentally all-static would answer it wrongly and
confidently.

**Split hygiene** (§11.6). Clips are cut at least ``--min-gap`` apart and each is tagged with its
source, so a later train/val/test split can keep whole regions of the film on one side. Frames from
the same shot on both sides of a split inflate every metric.

Usage::

    .venv/Scripts/python.exe scripts/build_corpus.py SOURCE.mov --clips 20 --seconds 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

from degradation.motion import MotionSummary, summarize  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
FFMPEG = REPO / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
CORPUS = REPO / "training" / "datasets" / "clean"


@dataclass(frozen=True, slots=True)
class ClipRecord:
    """One clip's entry in the manifest."""

    name: str
    source: str
    start_seconds: float
    duration_seconds: float
    width: int
    height: int
    frames: int
    motion_band: str
    motion_median_px: float
    motion_mean_px: float
    motion_max_px: float
    sha256: str
    size_bytes: int


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _extract(source: Path, start: float, seconds: float, destination: Path, crf: int) -> None:
    """Cuts one clip.

    Re-encoded rather than stream-copied because a stream copy can only cut on keyframes, which
    would make the requested start times approximate and the clip lengths uneven. CRF is kept low
    enough that the clip is visually indistinguishable from the source it came from.
    """
    result = subprocess.run(
        [
            str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{seconds:.3f}",
            "-an", "-sn",
            "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-x264-params", "keyint=48:min-keyint=48",
            str(destination),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg failed on {destination.name}:\n{result.stderr}")


def _measure(path: Path, sample: int = 48, scale: int = 4) -> tuple[MotionSummary, int, int, int]:
    """Measures motion and geometry from a clip.

    Luma is downscaled before phase correlation: it costs a fraction of the FFT time and the
    resulting magnitudes are rescaled back, which is accurate enough to place a clip in a band.
    """
    luma: list[np.ndarray] = []
    width = height = 0
    total = 0

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        width, height = stream.codec_context.width, stream.codec_context.height

        for frame in container.decode(stream):
            total += 1
            if len(luma) < sample:
                plane = frame.to_ndarray(format="gray")
                luma.append(plane[::scale, ::scale].astype(np.float64))

    summary = summarize(luma)
    rescaled = MotionSummary(
        mean_pixels_per_frame=summary.mean_pixels_per_frame * scale,
        median_pixels_per_frame=summary.median_pixels_per_frame * scale,
        max_pixels_per_frame=summary.max_pixels_per_frame * scale,
        frames=summary.frames,
    )

    return rescaled, width, height, total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--clips", type=int, default=20)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--skip-head", type=float, default=45.0, help="seconds of titles to skip")
    parser.add_argument("--skip-tail", type=float, default=120.0, help="seconds of credits to skip")
    parser.add_argument("--crf", type=int, default=12)
    parser.add_argument("--prefix", default="tos")
    parser.add_argument("--attribution", default="")
    parser.add_argument("--license", default="")
    args = parser.parse_args(argv)

    if not FFMPEG.exists():
        raise SystemExit(f"FFmpeg not found at {FFMPEG}. Run scripts/setup-worker.ps1 first.")
    if not args.source.exists():
        raise SystemExit(f"source not found: {args.source}")

    with av.open(str(args.source)) as container:
        duration = float(container.duration or 0) / 1_000_000

    usable_start = args.skip_head
    usable_end = duration - args.skip_tail
    if usable_end - usable_start < args.clips * args.seconds:
        raise SystemExit("source is too short for the requested number of clips")

    # Even spacing rather than random sampling: it covers the whole film deterministically, and
    # the gaps keep clips from neighbouring shots landing on both sides of a later split (§11.6).
    step = (usable_end - usable_start) / args.clips

    CORPUS.mkdir(parents=True, exist_ok=True)
    records: list[ClipRecord] = []

    for index in range(args.clips):
        start = usable_start + index * step
        name = f"{args.prefix}_{index:03d}.mp4"
        destination = CORPUS / name

        print(f"[{index + 1}/{args.clips}] {name}  t={start:8.2f}s", flush=True)
        _extract(args.source, start, args.seconds, destination, args.crf)

        motion, width, height, frames = _measure(destination)

        records.append(
            ClipRecord(
                name=name,
                source=args.source.name,
                start_seconds=round(start, 3),
                duration_seconds=args.seconds,
                width=width,
                height=height,
                frames=frames,
                motion_band=motion.band.value,
                motion_median_px=round(motion.median_pixels_per_frame, 3),
                motion_mean_px=round(motion.mean_pixels_per_frame, 3),
                motion_max_px=round(motion.max_pixels_per_frame, 3),
                sha256=_digest(destination),
                size_bytes=destination.stat().st_size,
            )
        )

    manifest = {
        "version": 1,
        "source": {
            "file": args.source.name,
            "attribution": args.attribution,
            "license": args.license,
            "duration_seconds": round(duration, 3),
        },
        "parameters": {
            "clips": args.clips,
            "seconds": args.seconds,
            "crf": args.crf,
            "skip_head": args.skip_head,
            "skip_tail": args.skip_tail,
        },
        "clips": [asdict(r) for r in records],
    }

    manifest_path = REPO / "training" / "datasets" / f"clean-{args.prefix}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    bands: dict[str, int] = {}
    for record in records:
        bands[record.motion_band] = bands.get(record.motion_band, 0) + 1

    print()
    print(f"manifest: {manifest_path.relative_to(REPO)}")
    print(f"clips:    {len(records)}  ({sum(r.size_bytes for r in records) / 1e6:.1f} MB)")
    print("motion:   " + ", ".join(f"{k}={v}" for k, v in sorted(bands.items())))

    return 0


if __name__ == "__main__":
    sys.exit(main())
