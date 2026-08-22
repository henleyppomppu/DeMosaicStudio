"""Picks the detector's operating point from data. prd.md §5.2.3, §5.2.5, §12.1.

`docs/phase1-detector-report.md` §5 recorded that the detection threshold was fixed at 0.5 with no
sweep, and that §5.2.3's default of 0.45 had never been calibrated against anything. The first
end-to-end run then found 843 regions in a clip containing one, so the operating point is now the
top blocker rather than a loose end.

This sweeps the threshold on **video with a known ground-truth mask** — a clean corpus clip with a
synthetic mosaic applied to a known region — and reports, per threshold:

* pixel precision and recall, which say whether the mask is right;
* **regions per frame**, which says whether the *count* is right. A detector can score well on
  pixels while shattering one region into nine, and nine regions means nine restorations, nine
  dilations and nine chances to damage clean picture.
* **false-positive area on frames with no mosaic at all**, which is what §5.2.5a actually asks about.

Usage::

    .venv/Scripts/python.exe scripts/calibrate_detector.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import av
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "worker"))
sys.path.insert(0, str(REPO / "training"))

from degradation.mosaic import pixelate  # noqa: E402
from demosaic_worker.analyze.profile import GridAnchor, MosaicProfile  # noqa: E402
from demosaic_worker.detect.regions import extract_regions  # noqa: E402
from demosaic_worker.detect.segmenter import Segmenter  # noqa: E402

CORPUS = REPO / "training" / "datasets" / "clean"
MANIFEST = REPO / "training" / "datasets" / "clean-tos.manifest.json"
MODELS = REPO / "models" / "detector"


@dataclass(frozen=True, slots=True)
class Point:
    """One threshold's behaviour."""

    threshold: float
    precision: float
    recall: float
    iou: float
    regions_per_frame: float
    region_recall: float
    clean_fp_area: float
    clean_frames_firing: float


def _frames(path: Path, count: int, stride: int = 2) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(container.streams.video[0])):
            if index % stride:
                continue
            out.append(frame.to_ndarray(format="gray").astype(np.float64))
            if len(out) >= count:
                break
    return out


def _mosaic(frame: np.ndarray, index: int, block: int) -> tuple[np.ndarray, np.ndarray]:
    """Applies a drifting elliptical mosaic and returns ``(degraded, truth_mask)``."""
    height, width = frame.shape
    spec = MosaicProfile(block_width=block, block_height=block, anchor=GridAnchor.SCREEN)

    cy, cx = height // 2, width // 3 + index * 3
    ry, rx = height // 7, width // 13
    ys, xs = np.mgrid[0:height, 0:width]
    region = (((ys - cy) / ry) ** 2 + ((xs - cx) / rx) ** 2) <= 1.0

    return np.where(region, pixelate(frame, spec), frame), region


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="det-unet-0.1.0")
    parser.add_argument("--clips", type=int, default=4)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--blocks", type=int, nargs="+", default=[5, 10, 18])
    parser.add_argument(
        "--thresholds", type=float, nargs="+",
        default=[0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
    )
    parser.add_argument("--min-area", type=int, default=1024)
    parser.add_argument("--out", type=Path, default=REPO / "docs" / "detector-calibration.json")
    args = parser.parse_args(argv)

    names = [c["name"] for c in json.loads(MANIFEST.read_text(encoding="utf-8"))["clips"]]
    names = names[: args.clips]

    segmenter = Segmenter(MODELS / args.model)
    print(f"model {segmenter.info.model_id} {segmenter.info.version} on {segmenter.device}", flush=True)

    # Probability maps are computed once and thresholded many times: the sweep costs one inference
    # pass, not one per threshold.
    positives: list[tuple[np.ndarray, np.ndarray]] = []
    negatives: list[np.ndarray] = []

    for name in names:
        clean = _frames(CORPUS / name, args.frames)
        for index, frame in enumerate(clean):
            block = args.blocks[index % len(args.blocks)]
            degraded, truth = _mosaic(frame, index, block)
            positives.append((segmenter.probability(degraded), truth))
            negatives.append(segmenter.probability(frame))
        print(f"  {name}: {len(clean)} frames", flush=True)

    points: list[Point] = []
    for threshold in args.thresholds:
        tp = fp = fn = 0
        regions_total = 0
        region_hits = 0

        for probability, truth in positives:
            predicted = probability >= threshold
            tp += int((predicted & truth).sum())
            fp += int((predicted & ~truth).sum())
            fn += int((~predicted & truth).sum())

            regions, _ = extract_regions(
                probability, threshold=threshold, min_area=args.min_area, max_regions=64
            )
            regions_total += len(regions)
            if any((r.mask & truth).sum() > 0.25 * truth.sum() for r in regions):
                region_hits += 1

        clean_areas = []
        for probability in negatives:
            clean_areas.append(float((probability >= threshold).mean()))

        clean = np.asarray(clean_areas)

        points.append(
            Point(
                threshold=threshold,
                precision=round(tp / max(tp + fp, 1), 4),
                recall=round(tp / max(tp + fn, 1), 4),
                iou=round(tp / max(tp + fp + fn, 1), 4),
                regions_per_frame=round(regions_total / max(len(positives), 1), 2),
                region_recall=round(region_hits / max(len(positives), 1), 4),
                clean_fp_area=round(float(clean.mean()), 5),
                clean_frames_firing=round(float((clean > 0.005).mean()), 4),
            )
        )

    report = {
        "model": {"id": segmenter.info.model_id, "version": segmenter.info.version},
        "positives": len(positives),
        "negatives": len(negatives),
        "blocks": args.blocks,
        "minArea": args.min_area,
        "points": [asdict(p) for p in points],
        "requirement": {
            "source": "prd.md §5.2.5a",
            "cleanFramesFiringMax": 0.005,
            "note": "at most 0.5% of negative frames may produce any region",
        },
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print(f"{'thr':>6} {'prec':>7} {'recall':>7} {'IoU':>7} {'reg/frm':>8} {'regRec':>7} "
          f"{'cleanFP':>8} {'cleanFire':>10}")
    for p in points:
        print(f"{p.threshold:6.2f} {p.precision:7.3f} {p.recall:7.3f} {p.iou:7.3f} "
              f"{p.regions_per_frame:8.2f} {p.region_recall:7.3f} {p.clean_fp_area:8.4f} "
              f"{p.clean_frames_firing:10.1%}")

    usable = [p for p in points if p.clean_frames_firing <= 0.005]
    print()
    if usable:
        best = max(usable, key=lambda p: p.iou)
        print(f"meets §5.2.5a at threshold {best.threshold} (IoU {best.iou:.3f}, "
              f"{best.regions_per_frame:.1f} regions/frame)")
    else:
        tightest = min(points, key=lambda p: p.clean_frames_firing)
        print(
            f"NO threshold meets §5.2.5a. Best is {tightest.threshold} with "
            f"{tightest.clean_frames_firing:.1%} of clean frames firing against a 0.5% bar."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
