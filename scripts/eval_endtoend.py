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

from demosaic_worker.metrics import psnr, ssim  # noqa: E402

import evalclips  # noqa: E402
import perceptual  # noqa: E402

PYTHON = REPO / ".venv" / "Scripts" / "python.exe"
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
    inside_lpips: float
    outside_psnr: float
    frames_improved: float
    regions_detected: int
    frames_restored: int
    seconds: float


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


def _rgb(path: Path, limit: int = 200) -> list[np.ndarray]:
    """Colour frames, for LPIPS. It is trained on RGB and greyscale would be a different question."""
    out: list[np.ndarray] = []
    with av.open(str(path)) as container:
        for frame in container.decode(container.streams.video[0]):
            out.append(frame.to_ndarray(format="rgb24"))
            if len(out) >= limit:
                break
    return out


def _luma(path: Path, limit: int = 200) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    with av.open(str(path)) as container:
        for frame in container.decode(container.streams.video[0]):
            out.append(frame.to_ndarray(format="gray").astype(np.float64))
            if len(out) >= limit:
                break
    return out


def perceptual_distance(clean_path: Path, output_path: Path, mask_of, stride: int = 6) -> float:
    """LPIPS over the region's bounding box. **Lower is better** (section 1.4.3).

    The box rather than the mask: LPIPS reads a whole image, and handing it a masked-out frame would
    score the black surround as much as the restoration.
    """
    if not perceptual.is_available():
        return float("nan")

    clean = _rgb(clean_path)
    restored = _rgb(output_path)
    count = min(len(clean), len(restored))
    scores: list[float] = []

    for index in range(0, count, stride):
        region = mask_of(index, clean[index].shape[:2])
        ys, xs = np.nonzero(region)
        if len(ys) < 100:
            continue
        box = (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
        crop = lambda a: a[box[0]:box[1], box[2]:box[3]]  # noqa: E731
        scores.append(perceptual.distance(crop(clean[index]), crop(restored[index])))

    return float(np.mean(scores)) if scores else float("nan")


def score(clean_path: Path, input_path: Path, output_path: Path,
          mask_of) -> tuple[float, float, float, float]:
    """Scores inside and outside the mosaicked region separately."""
    clean = _luma(clean_path)
    degraded = _luma(input_path)
    restored = _luma(output_path)

    count = min(len(clean), len(degraded), len(restored))
    inside: list[tuple[float, float, float]] = []
    outside: list[float] = []

    for index in range(count):
        # The region comes from the clip, not from thresholding a difference: a threshold marks
        # only the pixels the block average moved far, which is a holey mask rather than the
        # region (D-27).
        region = mask_of(index, clean[index].shape)
        untouched = np.abs(clean[index] - degraded[index]) <= 1

        if region.sum() > 5000:
            inside.append(
                (psnr(clean[index][region], degraded[index][region]),
                 psnr(clean[index][region], restored[index][region]),
                 ssim(clean[index], restored[index]))
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
    evalclips.add_argument(parser)
    parser.add_argument("--out", type=Path, default=REPO / "docs" / "endtoend-ladder.json")
    args = parser.parse_args(argv)

    ladder = [
        Rung("baseline", "detector v0.1.0, mask 0.5, CRF 20", "0.1.0", 0.5, 20),
        Rung("model", "+ detector v0.2.0", "0.2.0", 0.5, 20),
        Rung("threshold", "+ mask threshold 0.9 (calibrated)", "0.2.0", 0.9, 20),
        Rung("encode", "+ CRF 12 (measured transparent)", "0.2.0", 0.9, 12),
    ]

    clip = evalclips.resolve(args.clip)
    source = clip.degraded
    print(f"clip: {clip.name} - {clip.what}", flush=True)

    scores: list[Score] = []
    for rung in ladder:
        output = ARTIFACTS / f"ladder_{rung.name}.mp4"
        print(f"\n[{rung.name}] {rung.changed}", flush=True)

        started = time.time()
        summary = run_worker(source, output, rung)
        elapsed = time.time() - started

        inside_psnr, inside_ssim, outside_psnr, improved = score(
            clip.clean, source, output, clip.mask
        )
        inside_lpips = perceptual_distance(clip.clean, output, clip.mask)
        scores.append(
            Score(rung.name, rung.changed, round(inside_psnr, 3), round(inside_ssim, 4),
                  round(inside_lpips, 4), round(outside_psnr, 3), round(improved, 3),
                  summary["regionsDetected"], summary["framesRestored"], round(elapsed, 1))
        )
        print(f"  inside {inside_psnr:.2f} dB | LPIPS {inside_lpips:.4f} | "
              f"outside {outside_psnr:.2f} dB | {summary['regionsDetected']} regions | "
              f"{elapsed:.0f}s", flush=True)

    # The input's own scores are the bar: the pipeline has to beat what it was given.
    clean = _luma(clip.clean)
    degraded = _luma(source)
    reference_inside = []
    reference_outside = []
    for index in range(min(len(clean), len(degraded))):
        region = clip.mask(index, clean[index].shape)
        untouched = np.abs(clean[index] - degraded[index]) <= 1
        if region.sum() > 5000:
            reference_inside.append(psnr(clean[index][region], degraded[index][region]))
        if untouched.sum() > 10000:
            reference_outside.append(psnr(clean[index][untouched], degraded[index][untouched]))

    baseline_inside = float(np.mean(reference_inside))
    baseline_outside = float(np.mean(reference_outside))
    baseline_lpips = perceptual_distance(clip.clean, source, clip.mask)

    report = {
        "clip": clip.name,
        "input": {"inside_psnr": round(baseline_inside, 3),
                  "inside_lpips": round(baseline_lpips, 4),
                  "outside_psnr": round(baseline_outside, 3)},
        "rungs": [asdict(s) for s in scores],
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print(f"{'rung':12} {'changed':34} {'inside':>8} {'vs in':>7} {'LPIPS':>8} "
          f"{'outside':>8} {'vs in':>7} {'improved':>9}")
    # ASCII only: this runs on a cp949 console, where an em dash raises UnicodeEncodeError
    # (AGENTS.md, CLAUDE.md §4). It did, twice, after the measurements had already been written.
    print(f"{'(input)':12} {'the mosaicked video itself':34} {baseline_inside:8.2f} "
          f"{'-':>7} {baseline_lpips:8.4f} {baseline_outside:8.2f} {'-':>7} {'-':>9}")
    for s in scores:
        print(f"{s.name:12} {s.changed[:34]:34} {s.inside_psnr:8.2f} "
              f"{s.inside_psnr - baseline_inside:+7.2f} {s.inside_lpips:8.4f} "
              f"{s.outside_psnr:8.2f} {s.outside_psnr - baseline_outside:+7.2f} "
              f"{s.frames_improved:9.0%}")

    print()
    print("LPIPS is a perceptual distance: **lower is better**, unlike every other column.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
