"""End-to-end quality, as a ladder. prd.md section 5.1.8, section 12.

The first end-to-end run produced output worse than its input and named three causes: an over-firing
detector, a non-transparent encode, and multi-frame outside its operating window. This runs the
pipeline over a fixed input with **one variable changed per rung**, so each cause gets a number
rather than a share of the blame.

The score is taken twice, because a restoration job can fail in two independent ways:

* **inside the mosaicked region** - did the restoration help?
* **outside it** - did the pipeline damage picture it was supposed to leave alone? section 5.1.8 exists
  because a full re-encode degrades 100% of the frame to fix a few percent of it.

Usage::

    .venv/Scripts/python.exe scripts/eval_endtoend.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import av
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "worker"))
sys.path.insert(0, str(REPO / "training"))

from degradation.mosaic import pixelate  # noqa: E402
from demosaic_worker.analyze.profile import GridAnchor, MosaicProfile  # noqa: E402
from demosaic_worker.metrics import psnr, ssim  # noqa: E402

PYTHON = REPO / ".venv" / "Scripts" / "python.exe"
CORPUS = REPO / "training" / "datasets" / "clean"
ARTIFACTS = REPO / "artifacts"


@dataclass(frozen=True, slots=True)
class Rung:
    """One configuration of the ladder."""

    name: str
    changed: str
    model_version: str
    mask_threshold: float
    crf: int


@dataclass(frozen=True, slots=True)
class Score:
    """What one rung produced."""

    name: str
    changed: str
    inside_psnr: float
    inside_ssim: float
    outside_psnr: float
    frames_improved: float
    regions_detected: int
    frames_restored: int
    seconds: float


def build_input(clip: str, destination: Path, block: int = 10) -> None:
    """Applies a screen-anchored mosaic to a drifting ellipse, so ground truth exists."""
    spec = MosaicProfile(block_width=block, block_height=block, anchor=GridAnchor.SCREEN)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with av.open(str(CORPUS / clip)) as source, av.open(str(destination), mode="w") as out:
        stream = source.streams.video[0]
        encoder = out.add_stream("libx264", rate=stream.average_rate)
        encoder.width = stream.codec_context.width
        encoder.height = stream.codec_context.height
        encoder.pix_fmt = "yuv420p"
        encoder.time_base = stream.time_base
        encoder.options = {"crf": "18", "preset": "medium"}

        for index, frame in enumerate(source.decode(stream)):
            rgb = frame.to_ndarray(format="rgb24")
            height, width, _ = rgb.shape

            cy, cx = height // 2, width // 3 + index * 3
            ry, rx = 110, 150
            ys, xs = np.mgrid[0:height, 0:width]
            region = (((ys - cy) / ry) ** 2 + ((xs - cx) / rx) ** 2) <= 1.0

            degraded = np.stack([pixelate(rgb[:, :, c], spec) for c in range(3)], axis=2)
            rgb = np.where(region[:, :, None], degraded, rgb).astype(np.uint8)

            replacement = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            replacement.pts = frame.pts
            replacement.time_base = frame.time_base
            for packet in encoder.encode(replacement):
                out.mux(packet)

        for packet in encoder.encode():
            out.mux(packet)


def run_worker(source: Path, output: Path, rung: Rung) -> dict:
    """Drives the worker over the protocol and returns its result summary."""
    output.unlink(missing_ok=True)

    requests = [
        {"v": "1.0", "type": "hello", "id": "1"},
        {
            "v": "1.0", "type": "process", "id": "2", "jobId": f"ladder-{rung.name}",
            "sourcePath": str(source), "outputPath": str(output),
            "settings": {
                "detection": {"confidence": 0.45, "maskThreshold": rung.mask_threshold,
                              "minRegionArea": 1024},
                "restoration": {"preset": "Balanced", "temporalWindow": "auto"},
                "encode": {"profile": "QualityX265", "codec": "H265",
                           "constantQuality": rung.crf, "preset": "fast"},
                "modelVersion": rung.model_version,
            },
        },
        {"v": "1.0", "type": "shutdown", "id": "3"},
    ]

    proc = subprocess.run(
        [str(PYTHON), "-m", "demosaic_worker.main_loop"],
        input="\n".join(json.dumps(r) for r in requests) + "\n",
        capture_output=True, text=True, timeout=3600, cwd=str(REPO),
        env={"PYTHONPATH": str(REPO / "worker"), "SYSTEMROOT": r"C:\Windows",
             "PATH": r"C:\Windows\System32"},
    )

    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        message = json.loads(line)
        if message["type"] == "result":
            return message["summary"]
        if message["type"] == "error":
            raise SystemExit(f"worker error {message['code']}: {message['message']}")

    raise SystemExit(f"no result from the worker (exit {proc.returncode})")


def _luma(path: Path, limit: int = 200) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    with av.open(str(path)) as container:
        for frame in container.decode(container.streams.video[0]):
            out.append(frame.to_ndarray(format="gray").astype(np.float64))
            if len(out) >= limit:
                break
    return out


def score(clean_path: Path, input_path: Path, output_path: Path) -> tuple[float, float, float, float]:
    """Scores inside and outside the mosaicked region separately."""
    clean = _luma(clean_path)
    degraded = _luma(input_path)
    restored = _luma(output_path)

    count = min(len(clean), len(degraded), len(restored))
    inside: list[tuple[float, float, float]] = []
    outside: list[float] = []

    for index in range(count):
        difference = np.abs(clean[index] - degraded[index])
        region = difference > 6
        untouched = difference <= 1

        if region.sum() > 5000:
            ys, xs = np.nonzero(region)
            box = (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
            crop = lambda a: a[index][box[0]:box[1], box[2]:box[3]]  # noqa: E731
            inside.append(
                (psnr(crop(clean), crop(degraded)), psnr(crop(clean), crop(restored)),
                 ssim(crop(clean), crop(restored)))
            )

        if untouched.sum() > 10000:
            outside.append(psnr(clean[index][untouched], restored[index][untouched]))

    array = np.asarray(inside)
    improved = float((array[:, 1] > array[:, 0]).mean()) if len(array) else 0.0

    return (
        float(array[:, 1].mean()) if len(array) else 0.0,
        float(array[:, 2].mean()) if len(array) else 0.0,
        float(np.mean(outside)) if outside else 0.0,
        improved,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", default="tos_002.mp4")
    parser.add_argument("--out", type=Path, default=REPO / "docs" / "endtoend-ladder.json")
    args = parser.parse_args(argv)

    ladder = [
        Rung("baseline", "detector v0.1.0, mask 0.5, CRF 20", "0.1.0", 0.5, 20),
        Rung("model", "+ detector v0.2.0", "0.2.0", 0.5, 20),
        Rung("threshold", "+ mask threshold 0.9 (calibrated)", "0.2.0", 0.9, 20),
        Rung("encode", "+ CRF 12 (measured transparent)", "0.2.0", 0.9, 12),
    ]

    source = ARTIFACTS / f"ladder_input_{args.clip}"
    print(f"building input from {args.clip}", flush=True)
    build_input(args.clip, source)

    scores: list[Score] = []
    for rung in ladder:
        output = ARTIFACTS / f"ladder_{rung.name}.mp4"
        print(f"\n[{rung.name}] {rung.changed}", flush=True)

        started = time.time()
        summary = run_worker(source, output, rung)
        elapsed = time.time() - started

        inside_psnr, inside_ssim, outside_psnr, improved = score(
            CORPUS / args.clip, source, output
        )
        scores.append(
            Score(rung.name, rung.changed, round(inside_psnr, 3), round(inside_ssim, 4),
                  round(outside_psnr, 3), round(improved, 3),
                  summary["regionsDetected"], summary["framesRestored"], round(elapsed, 1))
        )
        print(f"  inside {inside_psnr:.2f} dB | outside {outside_psnr:.2f} dB | "
              f"{summary['regionsDetected']} regions | {elapsed:.0f}s", flush=True)

    # The input's own scores are the bar: the pipeline has to beat what it was given.
    clean = _luma(CORPUS / args.clip)
    degraded = _luma(source)
    reference_inside = []
    reference_outside = []
    for index in range(min(len(clean), len(degraded))):
        difference = np.abs(clean[index] - degraded[index])
        region, untouched = difference > 6, difference <= 1
        if region.sum() > 5000:
            ys, xs = np.nonzero(region)
            box = (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
            reference_inside.append(
                psnr(clean[index][box[0]:box[1], box[2]:box[3]],
                     degraded[index][box[0]:box[1], box[2]:box[3]])
            )
        if untouched.sum() > 10000:
            reference_outside.append(psnr(clean[index][untouched], degraded[index][untouched]))

    baseline_inside = float(np.mean(reference_inside))
    baseline_outside = float(np.mean(reference_outside))

    report = {
        "clip": args.clip,
        "input": {"inside_psnr": round(baseline_inside, 3), "outside_psnr": round(baseline_outside, 3)},
        "rungs": [asdict(s) for s in scores],
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print(f"{'rung':12} {'changed':38} {'inside':>8} {'vs in':>7} {'outside':>8} {'vs in':>7} {'improved':>9}")
    # ASCII only: this runs on a cp949 console, where an em dash raises UnicodeEncodeError
    # (AGENTS.md, CLAUDE.md §4). It did, twice, after the measurements had already been written.
    print(f"{'(input)':12} {'the mosaicked video itself':38} {baseline_inside:8.2f} "
          f"{'-':>7} {baseline_outside:8.2f} {'-':>7} {'-':>9}")
    for s in scores:
        print(f"{s.name:12} {s.changed[:38]:38} {s.inside_psnr:8.2f} "
              f"{s.inside_psnr - baseline_inside:+7.2f} {s.outside_psnr:8.2f} "
              f"{s.outside_psnr - baseline_outside:+7.2f} {s.frames_improved:9.0%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
